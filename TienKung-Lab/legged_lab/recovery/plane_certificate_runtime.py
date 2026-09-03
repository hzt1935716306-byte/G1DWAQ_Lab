"""Plane-generalized runtime adapter around the unchanged 1--5 step LP."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .certificate import (
    CertificateResult,
    CertificateState,
    CertificateStatus,
    HalfspaceRegion2D,
    RecoverabilityConfig,
    certify_recoverability,
)
from .certificate_process_pool import CertificateProcessPool
from .g1_certificate_runtime import CalibratedG1CertificateEvaluator, PendingCertificateBatch
from .plane_adapter import adapt_flat_capability
from .plane_nominal_params import PlaneNominalGait, PlaneNominalParameterTable


@dataclass(frozen=True)
class PlaneCertificateQuery:
    command: np.ndarray
    b: np.ndarray
    q: np.ndarray
    support_side: str
    phase: float
    alpha: float
    adapter_valid: bool
    invalid_reason: str = ""


@dataclass(frozen=True)
class PendingPlaneCertificateBatch(PendingCertificateBatch):
    queries: tuple[PlaneCertificateQuery, ...]


def plane_periodic_state(
    vx_cmd: float,
    vy_cmd: float,
    step_period: float,
    omega: float,
    step_width: float,
) -> dict[str, tuple[float, float]]:
    """Periodic state with heading-horizontal commands (no cos(alpha) factor)."""

    gamma = math.exp(float(omega) * float(step_period))
    landing_left = np.asarray(
        (float(vx_cmd) * step_period, float(vy_cmd) * step_period - step_width),
        dtype=np.float64,
    )
    landing_right = np.asarray(
        (float(vx_cmd) * step_period, float(vy_cmd) * step_period + step_width),
        dtype=np.float64,
    )
    denominator = gamma * gamma - 1.0
    b_left = (gamma * landing_left + landing_right) / denominator
    b_right = (landing_left + gamma * landing_right) / denominator
    return {
        "landing_left": tuple(landing_left),
        "landing_right": tuple(landing_right),
        "b_left": tuple(b_left),
        "b_right": tuple(b_right),
        "q_left": tuple(-landing_right),
        "q_right": tuple(-landing_left),
    }


class PlaneCalibratedG1CertificateEvaluator(CalibratedG1CertificateEvaluator):
    """Adapt flat C/L/v_max and query policy-dependent plane nominal gait."""

    def __init__(
        self,
        flat_parameters_path: str | Path,
        nominal_parameters_path: str | Path,
        workers: int = 1,
        executor_type: str = "thread",
        failure_window_size: int = 4096,
        failure_rate_threshold: float = 0.01,
        z_sole: float = -0.045,
    ) -> None:
        # Initialize all legacy diagnostics without starting the legacy pool.
        super().__init__(
            flat_parameters_path,
            workers=1,
            executor_type="sequential",
            failure_window_size=failure_window_size,
            failure_rate_threshold=failure_rate_threshold,
        )
        self.nominal_parameters_path = Path(nominal_parameters_path).expanduser().resolve()
        self.nominal_table = PlaneNominalParameterTable.from_yaml(self.nominal_parameters_path)
        self.z_sole = float(z_sole)
        self.workers = max(1, int(workers))
        self.executor_type = str(executor_type)
        if self.executor_type == "subprocess":
            self._executor = CertificateProcessPool(
                self.parameters_path,
                self.workers,
                worker_mode="plane",
                nominal_parameters_path=self.nominal_parameters_path,
            )
        elif self.executor_type == "thread" and self.workers > 1:
            self._executor = ThreadPoolExecutor(max_workers=self.workers)
        elif self.executor_type == "sequential" or (
            self.executor_type == "thread" and self.workers == 1
        ):
            self._executor = None
        else:
            raise ValueError("certificate executor_type must be sequential, thread, or subprocess")

    def lookup_nominal(self, command: Sequence[float], alpha: float):
        return self.nominal_table.lookup_command(alpha, command)

    def _plane_config(
        self,
        command: Sequence[float],
        alpha: float,
        nominal: PlaneNominalGait | None = None,
    ) -> RecoverabilityConfig:
        if nominal is None:
            lookup = self.lookup_nominal(command, alpha)
            if not lookup.valid or lookup.value is None:
                raise ValueError(lookup.reason)
            nominal = lookup.value
        capability = adapt_flat_capability(self.parameters, alpha, self.z_sole)
        if not capability.nominal_cop_valid:
            raise ValueError("nominal CoP [0,0] lies outside projected C")
        theory = plane_periodic_state(
            float(command[0]),
            float(command[1]),
            nominal.step_period,
            nominal.omega,
            nominal.step_width,
        )
        return RecoverabilityConfig(
            gravity=9.81,
            h_eff=nominal.h_eff,
            step_period=nominal.step_period,
            max_steps=5,
            cop_left=HalfspaceRegion2D.box(capability.cop_left.x, capability.cop_left.y),
            cop_right=HalfspaceRegion2D.box(capability.cop_right.x, capability.cop_right.y),
            landing_left=HalfspaceRegion2D.box(
                capability.landing_left.x, capability.landing_left.y
            ),
            landing_right=HalfspaceRegion2D.box(
                capability.landing_right.x, capability.landing_right.y
            ),
            swing_velocity_limits=capability.swing_velocity_limits,
            nominal_cop_left=(0.0, 0.0),
            nominal_cop_right=(0.0, 0.0),
            nominal_step_left=theory["landing_left"],
            nominal_step_right=theory["landing_right"],
            nominal_b_left=theory["b_left"],
            nominal_b_right=theory["b_right"],
            nominal_q_left=theory["q_left"],
            nominal_q_right=theory["q_right"],
            epsilon_b=nominal.epsilon_b,
            epsilon_q=nominal.epsilon_q,
        )

    @staticmethod
    def _invalid_result(reason: str) -> CertificateResult:
        return CertificateResult(
            CertificateStatus.INVALID_INPUT,
            6,
            -3.0,
            None,
            (),
            reason,
        )

    def _solve(self, query: PlaneCertificateQuery) -> CertificateResult:
        if not query.adapter_valid:
            return self._invalid_result(query.invalid_reason)
        try:
            config = self._plane_config(query.command, query.alpha)
            return certify_recoverability(
                CertificateState(
                    b=query.b,
                    q=query.q,
                    support_side=query.support_side,
                    phase=query.phase,
                    step_period=config.step_period,
                    omega=config.omega,
                ),
                config,
            )
        except Exception as exc:
            return CertificateResult(
                CertificateStatus.SOLVER_FAILURE,
                None,
                None,
                None,
                (),
                f"Unexpected plane certificate exception: {type(exc).__name__}: {exc}",
                diagnostic={"kind": "unexpected_plane_certificate_exception"},
            )

    def submit(self, state, env_ids: torch.Tensor) -> PendingPlaneCertificateBatch:
        if env_ids.numel() == 0:
            return PendingPlaneCertificateBatch((), (), (), env_ids.device)
        ids = tuple(int(value) for value in env_ids.detach().cpu().tolist())
        commands = state.command_velocity[env_ids].detach().cpu().numpy()
        alphas = state.signed_slope[env_ids].detach().cpu().numpy()
        geometry_valid = state.terrain_plane_valid[env_ids].detach().cpu().numpy()
        com_positions = state.com_position[env_ids, :2].detach().cpu().numpy()
        com_velocities = state.com_velocity[env_ids, :2].detach().cpu().numpy()
        left_positions = state.left_foot_position[env_ids, :2].detach().cpu().numpy()
        right_positions = state.right_foot_position[env_ids, :2].detach().cpu().numpy()
        q_values = state.q[env_ids].detach().cpu().numpy()
        support_left = state.support_is_left[env_ids].detach().cpu().numpy()

        queries = []
        jobs = []
        for command, alpha, plane_valid, com, velocity, left, right, q, is_left in zip(
            commands,
            alphas,
            geometry_valid,
            com_positions,
            com_velocities,
            left_positions,
            right_positions,
            q_values,
            support_left,
        ):
            lookup = self.lookup_nominal(command, float(alpha)) if plane_valid else None
            valid = bool(plane_valid and lookup is not None and lookup.valid and lookup.value is not None)
            reason = "" if valid else (
                lookup.reason if lookup is not None else "invalid terrain plane"
            )
            if valid:
                try:
                    capability = adapt_flat_capability(
                        self.parameters, float(alpha), self.z_sole
                    )
                except ValueError as exc:
                    valid = False
                    reason = str(exc)
                else:
                    if not capability.nominal_cop_valid:
                        valid = False
                        reason = "nominal CoP [0,0] lies outside projected C"
            if valid:
                nominal = lookup.value
                assert nominal is not None
                support = left if is_left else right
                # b and q remain measured heading-horizontal quantities.  In
                # particular, neither is multiplied by P_alpha.
                b = com + velocity / nominal.omega - support
            else:
                b = np.asarray(state.b[ids[len(queries)]].detach().cpu(), dtype=np.float64)
            query = PlaneCertificateQuery(
                command=np.asarray(command, dtype=np.float64),
                b=np.asarray(b, dtype=np.float64),
                q=np.asarray(q, dtype=np.float64),
                support_side="left" if is_left else "right",
                phase=0.0,
                alpha=float(alpha),
                adapter_valid=valid,
                invalid_reason=reason,
            )
            queries.append(query)
            if not valid:
                jobs.append(self._invalid_result(reason))
            elif self._executor is None:
                jobs.append(self._solve(query))
            elif self.executor_type == "subprocess":
                jobs.append(self._executor.submit(query))
            else:
                jobs.append(self._executor.submit(self._solve, query))
        return PendingPlaneCertificateBatch(ids, tuple(queries), tuple(jobs), env_ids.device)

    def _failure_record(self, env_id, query, result):
        config = self._plane_config(query.command, query.alpha)
        return {
            "schema_version": 1,
            "env_id": int(env_id),
            "parameters_path": str(self.parameters_path),
            "nominal_parameters_path": str(self.nominal_parameters_path),
            "input": {
                "b": query.b.tolist(),
                "q": query.q.tolist(),
                "phase": query.phase,
                "T": config.step_period,
                "support": query.support_side,
                "command": query.command.tolist(),
                "omega": config.omega,
                "alpha": query.alpha,
            },
            "result": {
                "status": result.status.value,
                "N": result.n_min,
                "margin": result.margin,
                "message": result.message,
            },
            "failure": result.diagnostic,
        }

    def _resolve_with_validity(
        self,
        pending: PendingPlaneCertificateBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not pending.env_ids:
            return (
                torch.empty(0, dtype=torch.long, device=pending.device),
                torch.empty(0, dtype=torch.float32, device=pending.device),
                torch.empty(0, dtype=torch.bool, device=pending.device),
            )
        raw_results: Sequence[CertificateResult] = [
            job.result() if isinstance(job, Future) else job for job in pending.jobs
        ]
        results = []
        valid_results = []
        for env_id, query, result in zip(pending.env_ids, pending.queries, raw_results):
            if not query.adapter_valid:
                results.append(self._invalid_result(query.invalid_reason))
                valid_results.append(False)
                continue
            normal = (
                result.status in (CertificateStatus.FINITE, CertificateStatus.OVER_HORIZON)
                and result.n_min is not None
                and result.margin is not None
                and not result.margin_fallback
                and not result.solver_fallback
            )
            if not normal:
                self._save_failure_record(self._failure_record(env_id, query, result))
                result = CertificateResult(
                    CertificateStatus.OVER_HORIZON,
                    6,
                    -3.0,
                    result.witness,
                    result.feasible_horizons,
                    f"{result.message} Conservative runtime fallback.",
                    margin_saturated=True,
                    solver_fallback=True,
                    solver_retried=result.solver_retried,
                    diagnostic=result.diagnostic,
                )
            self._register_result(result)
            results.append(result)
            valid_results.append(normal)
        self._check_failure_rate()
        return (
            torch.tensor([result.n_min for result in results], dtype=torch.long, device=pending.device),
            torch.tensor([result.margin for result in results], dtype=torch.float32, device=pending.device),
            torch.tensor(valid_results, dtype=torch.bool, device=pending.device),
        )


__all__ = [
    "PendingPlaneCertificateBatch",
    "PlaneCalibratedG1CertificateEvaluator",
    "PlaneCertificateQuery",
    "plane_periodic_state",
]
