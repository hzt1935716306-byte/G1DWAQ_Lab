"""Runtime adapter from calibrated G1 privileged state to certificate.py."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml

from .certificate import (
    CertificateResult,
    CertificateState,
    CertificateStatus,
    HalfspaceRegion2D,
    RecoverabilityConfig,
    certify_recoverability,
)
from .state_extractor import G1PrivilegedRecoveryState, theoretical_periodic_state


def _bounds(region: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    return tuple(region["x"]), tuple(region["y"])


class CalibratedG1CertificateEvaluator:
    """Evaluate the unchanged calibrated certificate for a small event batch."""

    def __init__(
        self,
        parameters_path: str | Path,
        workers: int = 1,
        failure_window_size: int = 4096,
        failure_rate_threshold: float = 0.01,
    ) -> None:
        self.parameters_path = Path(parameters_path).expanduser().resolve()
        with self.parameters_path.open("r", encoding="utf-8") as stream:
            self.parameters = yaml.safe_load(stream)
        self.workers = max(1, int(workers))
        self.failure_window_size = int(failure_window_size)
        self.failure_rate_threshold = float(failure_rate_threshold)
        if self.failure_window_size <= 0:
            raise ValueError("certificate failure_window_size must be positive")
        if not 0.0 < self.failure_rate_threshold <= 1.0:
            raise ValueError("certificate failure_rate_threshold must lie in (0, 1]")
        self._executor = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None
        self._failure_window: deque[int] = deque(maxlen=self.failure_window_size)
        self._evaluation_count = 0
        self._failure_count = 0
        self._margin_fallback_count = 0
        self._solver_fallback_count = 0
        self._retry_result_count = 0
        self._failure_records: list[dict[str, object]] = []
        self._diagnostics_path: Path | None = None

    def configure_diagnostics(self, log_dir: str | Path) -> None:
        path = Path(log_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        self._diagnostics_path = path / "certificate_solver_fallbacks.jsonl"

    @property
    def statistics(self) -> dict[str, float | int]:
        window_failures = sum(self._failure_window)
        window_count = len(self._failure_window)
        return {
            "evaluations": self._evaluation_count,
            "fallbacks": self._failure_count,
            "margin_fallbacks": self._margin_fallback_count,
            "solver_fallbacks": self._solver_fallback_count,
            "retried_results": self._retry_result_count,
            "window_count": window_count,
            "window_failures": window_failures,
            "window_failure_rate": window_failures / max(window_count, 1),
        }

    @property
    def failure_records(self) -> tuple[dict[str, object], ...]:
        return tuple(self._failure_records)

    def _period_at(self, speed: float) -> float:
        model = self.parameters["step_period"]
        if model["type"] == "linear":
            return max(0.10, float(model["intercept"] + model["slope"] * speed))
        return float(model["value"])

    def _config(self, speed: float) -> RecoverabilityConfig:
        parameters = self.parameters
        period = self._period_at(speed)
        omega = float(parameters["omega"])
        theory = theoretical_periodic_state(speed, 0.0, period, omega, float(parameters["w"]))
        c_left_x, c_left_y = _bounds(parameters["C_left"])
        c_right_x, c_right_y = _bounds(parameters["C_right"])
        l_left_x, l_left_y = _bounds(parameters["L_left"])
        l_right_x, l_right_y = _bounds(parameters["L_right"])
        return RecoverabilityConfig(
            gravity=9.81,
            h_eff=float(parameters["h_eff"]),
            step_period=period,
            max_steps=5,
            cop_left=HalfspaceRegion2D.box(c_left_x, c_left_y),
            cop_right=HalfspaceRegion2D.box(c_right_x, c_right_y),
            landing_left=HalfspaceRegion2D.box(l_left_x, l_left_y),
            landing_right=HalfspaceRegion2D.box(l_right_x, l_right_y),
            swing_velocity_limits=(
                float(parameters["v_max"]["x"]),
                float(parameters["v_max"]["y"]),
            ),
            nominal_cop_left=(0.0, 0.0),
            nominal_cop_right=(0.0, 0.0),
            nominal_step_left=theory["landing_left"],
            nominal_step_right=theory["landing_right"],
            nominal_b_left=theory["b_left"],
            nominal_b_right=theory["b_right"],
            nominal_q_left=theory["q_left"],
            nominal_q_right=theory["q_right"],
            epsilon_b=(float(parameters["epsilon_b"]["x"]), float(parameters["epsilon_b"]["y"])),
            epsilon_q=(float(parameters["epsilon_q"]["x"]), float(parameters["epsilon_q"]["y"])),
        )

    def _solve(
        self,
        query: tuple[np.ndarray, np.ndarray, np.ndarray, str, float],
    ) -> CertificateResult:
        command, b, q, support_side, phase = query
        try:
            config = self._config(float(command[0]))
            return certify_recoverability(
                CertificateState(
                    b=b,
                    q=q,
                    support_side=support_side,
                    phase=phase,
                    step_period=config.step_period,
                    omega=config.omega,
                ),
                config,
            )
        except Exception as exc:  # keep one malformed environment from killing a vector batch
            return CertificateResult(
                CertificateStatus.SOLVER_FAILURE,
                None,
                None,
                None,
                (),
                f"Unexpected certificate exception: {type(exc).__name__}: {exc}",
                diagnostic={
                    "kind": "unexpected_certificate_exception",
                    "attempted_horizon": None,
                    "solver": {
                        "initial": {
                            "success": False,
                            "status": -1,
                            "message": f"{type(exc).__name__}: {exc}",
                            "nit": 0,
                        }
                    },
                    "witness": None,
                    "witness_residual": None,
                    "margin_zero_residual": None,
                },
            )

    def _conservative_runtime_fallback(
        self,
        result: CertificateResult,
        speed: float,
    ) -> CertificateResult:
        config = self._config(speed)
        return CertificateResult(
            CertificateStatus.OVER_HORIZON,
            6,
            -config.eta_max,
            result.witness,
            result.feasible_horizons,
            f"{result.message} Conservative runtime fallback.",
            margin_saturated=True,
            solver_fallback=True,
            solver_retried=result.solver_retried,
            diagnostic=result.diagnostic,
        )

    def _failure_record(
        self,
        env_id: int,
        query: tuple[np.ndarray, np.ndarray, np.ndarray, str, float],
        result: CertificateResult,
    ) -> dict[str, object]:
        command, b, q, support_side, phase = query
        config = self._config(float(command[0]))
        return {
            "schema_version": 1,
            "env_id": int(env_id),
            "parameters_path": str(self.parameters_path),
            "input": {
                "b": b.tolist(),
                "q": q.tolist(),
                "phase": float(phase),
                "T": float(config.step_period),
                "support": support_side,
                "command": command.tolist(),
                "omega": float(config.omega),
            },
            "result": {
                "status": result.status.value,
                "N": result.n_min,
                "margin": result.margin,
                "margin_fallback": result.margin_fallback,
                "solver_fallback": result.solver_fallback,
                "solver_retried": result.solver_retried,
                "message": result.message,
            },
            "failure": result.diagnostic,
        }

    def _save_failure_record(self, record: dict[str, object]) -> None:
        self._failure_records.append(record)
        if self._diagnostics_path is not None:
            with self._diagnostics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _register_result(self, result: CertificateResult) -> None:
        fallback = result.margin_fallback or result.solver_fallback
        self._evaluation_count += 1
        self._failure_window.append(int(fallback))
        if result.solver_retried:
            self._retry_result_count += 1
        if fallback:
            self._failure_count += 1
            self._margin_fallback_count += int(result.margin_fallback)
            self._solver_fallback_count += int(result.solver_fallback)

    def _check_failure_rate(self) -> None:
        if len(self._failure_window) < self.failure_window_size:
            return
        failure_rate = sum(self._failure_window) / self.failure_window_size
        if failure_rate > self.failure_rate_threshold:
            raise RuntimeError(
                "G1 certificate fallback rate exceeded the configured threshold: "
                f"{failure_rate:.6f} > {self.failure_rate_threshold:.6f} over "
                f"{self.failure_window_size} evaluations; diagnostics={self._diagnostics_path}"
            )

    def evaluate(
        self,
        state: G1PrivilegedRecoveryState,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return N_min and margin tensors on the environment device."""

        if env_ids.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long, device=env_ids.device),
                torch.empty(0, dtype=torch.float32, device=env_ids.device),
            )
        ids = env_ids.detach().cpu().tolist()
        b_values = state.b[env_ids].detach().cpu().numpy()
        q_values = state.q[env_ids].detach().cpu().numpy()
        commands = state.command_velocity[env_ids].detach().cpu().numpy()
        phases = state.phase[env_ids].detach().cpu().tolist()
        support_left = state.support_is_left[env_ids].detach().cpu().tolist()
        queries = [
            (
                np.asarray(command, dtype=np.float64),
                np.asarray(b, dtype=np.float64),
                np.asarray(q, dtype=np.float64),
                "left" if is_left else "right",
                float(phase),
            )
            for command, b, q, is_left, phase in zip(
                commands,
                b_values,
                q_values,
                support_left,
                phases,
            )
        ]
        if self._executor is None:
            results: Sequence[CertificateResult] = [self._solve(query) for query in queries]
        else:
            results = list(self._executor.map(self._solve, queries))

        resolved_results: list[CertificateResult] = []
        for env_id, query, result in zip(ids, queries, results):
            if result.status == CertificateStatus.CONSTRAINT_BUILDER_MISMATCH:
                record = self._failure_record(env_id, query, result)
                self._save_failure_record(record)
                raise RuntimeError(
                    "G1 certificate feasibility/margin constraint builder mismatch; "
                    f"env_id={env_id}, diagnostics={self._diagnostics_path}, record={record}"
                )
            if (
                result.status not in (CertificateStatus.FINITE, CertificateStatus.OVER_HORIZON)
                or result.n_min is None
                or result.margin is None
            ):
                result = self._conservative_runtime_fallback(result, float(query[0][0]))
            if result.margin_fallback or result.solver_fallback:
                record = self._failure_record(env_id, query, result)
                self._save_failure_record(record)
                if self._failure_count == 0:
                    print(
                        "[G1Certificate] isolated solver fallback; training continues. "
                        f"env_id={env_id}, diagnostics={self._diagnostics_path}",
                        flush=True,
                    )
            self._register_result(result)
            resolved_results.append(result)

        self._check_failure_rate()
        n_min = torch.tensor(
            [result.n_min for result in resolved_results],
            dtype=torch.long,
            device=env_ids.device,
        )
        margin = torch.tensor(
            [result.margin for result in resolved_results],
            dtype=torch.float32,
            device=env_ids.device,
        )
        return n_min, margin

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
