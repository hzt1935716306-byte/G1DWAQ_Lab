"""Plane-generalized runtime adapter around the unchanged 1--5 step LP."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import time
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
from .certificate_ipc import CertificateBatchResponse
from .certificate_query_corpus import (
    PlaneCertificateQueryRecorder,
    RecordedPlaneCertificateQuery,
)
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


def mirror_plane_certificate_query(query: PlaneCertificateQuery) -> PlaneCertificateQuery:
    """Mirror one plane query laterally while leaving its signed slope unchanged."""

    command = np.asarray(query.command, dtype=np.float64).copy()
    command[1] *= -1.0
    if command.size >= 3:
        command[2] *= -1.0
    b = np.asarray(query.b, dtype=np.float64).copy()
    q = np.asarray(query.q, dtype=np.float64).copy()
    b[1] *= -1.0
    q[1] *= -1.0
    return PlaneCertificateQuery(
        command=command,
        b=b,
        q=q,
        support_side="right" if query.support_side == "left" else "left",
        phase=0.0,
        alpha=query.alpha,
        adapter_valid=query.adapter_valid,
        invalid_reason=query.invalid_reason,
    )


@dataclass(frozen=True)
class PendingPlaneCertificateBatch(PendingCertificateBatch):
    queries: tuple[PlaneCertificateQuery, ...]
    batch_job: Future | None = None


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
        use_state_b: bool = False,
        profile_enabled: bool = False,
        query_record_path: str | Path | None = None,
        ipc_batch_enabled: bool = False,
        ipc_chunk_size: int = 1,
        dynamic_dispatch: bool = False,
        exact_alpha_cache: bool = False,
        exact_alpha_cache_max_entries: int = 8192,
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
        self.use_state_b = bool(use_state_b)
        self.profile_enabled = bool(profile_enabled)
        self.ipc_batch_enabled = bool(ipc_batch_enabled)
        self.ipc_chunk_size = int(ipc_chunk_size)
        self.dynamic_dispatch = bool(dynamic_dispatch)
        self.exact_alpha_cache = bool(exact_alpha_cache)
        self.exact_alpha_cache_max_entries = int(exact_alpha_cache_max_entries)
        if self.exact_alpha_cache_max_entries <= 0:
            raise ValueError("exact-alpha cache maximum entries must be positive")
        if self.ipc_chunk_size <= 0:
            raise ValueError("certificate IPC chunk size must be positive")
        self._capability_cache = OrderedDict()
        self._capability_cache_hits = 0
        self._capability_cache_misses = 0
        self._profile_totals: dict[str, float] = {}
        self._profile_counts: dict[str, int] = {}
        self._profile_samples: dict[str, list[float]] = {}
        self._query_record_index = 0
        self._query_recorder = (
            PlaneCertificateQueryRecorder(query_record_path) if query_record_path else None
        )
        self._solve_depth_reached_counts = np.zeros(5, dtype=np.int64)
        self.workers = max(1, int(workers))
        self.executor_type = str(executor_type)
        if self.executor_type == "subprocess":
            self._executor = CertificateProcessPool(
                self.parameters_path,
                self.workers,
                worker_mode="plane",
                nominal_parameters_path=self.nominal_parameters_path,
                z_sole=self.z_sole,
                exact_alpha_cache=self.exact_alpha_cache,
                exact_alpha_cache_max_entries=self.exact_alpha_cache_max_entries,
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

    def _record_profile(self, name: str, milliseconds: float, count: int = 1) -> None:
        if not self.profile_enabled:
            return
        self._profile_totals[name] = self._profile_totals.get(name, 0.0) + float(milliseconds)
        self._profile_counts[name] = self._profile_counts.get(name, 0) + int(count)
        self._profile_samples.setdefault(name, []).append(float(milliseconds))

    def _capability_at(self, alpha: float):
        key = float(alpha)
        if self.exact_alpha_cache and key in self._capability_cache:
            self._capability_cache_hits += 1
            self._capability_cache.move_to_end(key)
            return self._capability_cache[key]
        capability = adapt_flat_capability(self.parameters, key, self.z_sole)
        self._capability_cache_misses += 1
        if self.exact_alpha_cache:
            self._capability_cache[key] = capability
            self._capability_cache.move_to_end(key)
            if len(self._capability_cache) > self.exact_alpha_cache_max_entries:
                self._capability_cache.popitem(last=False)
        return capability

    @property
    def capability_cache_statistics(self) -> dict[str, int | float]:
        requests = self._capability_cache_hits + self._capability_cache_misses
        return {
            "size": len(self._capability_cache),
            "hits": self._capability_cache_hits,
            "misses": self._capability_cache_misses,
            "hit_rate": self._capability_cache_hits / max(requests, 1),
        }

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
        capability = self._capability_at(alpha)
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
                diagnostic={
                    "kind": "unexpected_plane_certificate_exception",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )

    def submit(self, state, env_ids: torch.Tensor) -> PendingPlaneCertificateBatch:
        if env_ids.numel() == 0:
            return PendingPlaneCertificateBatch((), (), (), env_ids.device)
        d2h_started = time.perf_counter_ns()
        ids = tuple(int(value) for value in env_ids.detach().cpu().tolist())
        if self.ipc_batch_enabled:
            gather_started = time.perf_counter_ns()
            gathered_values = torch.cat(
                (
                    state.command_velocity[env_ids],
                    state.signed_slope[env_ids, None],
                    state.com_position[env_ids, :2],
                    state.com_velocity[env_ids, :2],
                    state.left_foot_position[env_ids, :2],
                    state.right_foot_position[env_ids, :2],
                    state.q[env_ids],
                    state.b[env_ids],
                ),
                dim=1,
            ).detach()
            gathered_flags = torch.stack(
                (
                    state.terrain_plane_valid[env_ids],
                    state.support_is_left[env_ids],
                ),
                dim=1,
            ).detach()
            if self.profile_enabled and gathered_values.device.type == "cuda":
                torch.cuda.synchronize(gathered_values.device)
            self._record_profile("tensor_gather_ms", (time.perf_counter_ns() - gather_started) / 1.0e6)
            transfer_started = time.perf_counter_ns()
            values = gathered_values.cpu().numpy()
            flags = gathered_flags.cpu().numpy()
            self._record_profile("d2h_ms", (time.perf_counter_ns() - transfer_started) / 1.0e6)
            commands = values[:, 0:3]
            alphas = values[:, 3]
            com_positions = values[:, 4:6]
            com_velocities = values[:, 6:8]
            left_positions = values[:, 8:10]
            right_positions = values[:, 10:12]
            q_values = values[:, 12:14]
            state_b_values = values[:, 14:16]
            geometry_valid = flags[:, 0]
            support_left = flags[:, 1]
        else:
            commands = state.command_velocity[env_ids].detach().cpu().numpy()
            alphas = state.signed_slope[env_ids].detach().cpu().numpy()
            geometry_valid = state.terrain_plane_valid[env_ids].detach().cpu().numpy()
            com_positions = state.com_position[env_ids, :2].detach().cpu().numpy()
            com_velocities = state.com_velocity[env_ids, :2].detach().cpu().numpy()
            left_positions = state.left_foot_position[env_ids, :2].detach().cpu().numpy()
            right_positions = state.right_foot_position[env_ids, :2].detach().cpu().numpy()
            q_values = state.q[env_ids].detach().cpu().numpy()
            state_b_values = state.b[env_ids].detach().cpu().numpy()
            support_left = state.support_is_left[env_ids].detach().cpu().numpy()
            self._record_profile("d2h_ms", (time.perf_counter_ns() - d2h_started) / 1.0e6)

        queries = []
        nominal_ms = 0.0
        capability_ms = 0.0
        query_build_ms = 0.0
        for command, alpha, plane_valid, com, velocity, left, right, q, state_b, is_left in zip(
            commands,
            alphas,
            geometry_valid,
            com_positions,
            com_velocities,
            left_positions,
            right_positions,
            q_values,
            state_b_values,
            support_left,
        ):
            nominal_started = time.perf_counter_ns()
            lookup = self.lookup_nominal(command, float(alpha)) if plane_valid else None
            nominal_ms += (time.perf_counter_ns() - nominal_started) / 1.0e6
            valid = bool(plane_valid and lookup is not None and lookup.valid and lookup.value is not None)
            reason = "" if valid else (
                lookup.reason if lookup is not None else "invalid terrain plane"
            )
            if valid:
                try:
                    capability_started = time.perf_counter_ns()
                    capability = self._capability_at(float(alpha))
                    capability_ms += (time.perf_counter_ns() - capability_started) / 1.0e6
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
                b = (
                    np.asarray(state_b, dtype=np.float64)
                    if self.use_state_b
                    else com + velocity / nominal.omega - support
                )
            else:
                b = np.asarray(state_b, dtype=np.float64)
            query_started = time.perf_counter_ns()
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
            query_build_ms += (time.perf_counter_ns() - query_started) / 1.0e6
            queries.append(query)
        self._record_profile("nominal_lookup_ms", nominal_ms)
        self._record_profile("parent_capability_ms", capability_ms)
        self._record_profile("query_build_ms", query_build_ms)
        return self.submit_queries(tuple(queries), env_ids.device, env_ids=ids)

    def submit_queries(
        self,
        queries: Sequence[PlaneCertificateQuery],
        device: torch.device,
        *,
        env_ids: Sequence[int] | None = None,
    ) -> PendingPlaneCertificateBatch:
        """Submit already-formed queries for diagnostics without fabricating state."""

        ids = tuple(range(len(queries))) if env_ids is None else tuple(env_ids)
        if len(ids) != len(queries):
            raise ValueError("env_ids and plane queries must have the same length")
        if self._query_recorder is not None:
            records = []
            for env_id, query in zip(ids, queries):
                records.append(
                    RecordedPlaneCertificateQuery(self._query_record_index, int(env_id), query)
                )
                self._query_record_index += 1
            self._query_recorder.append(records)

        if (
            self.ipc_batch_enabled
            and self.executor_type == "subprocess"
            and self._executor is not None
        ):
            jobs = []
            indexed_valid = []
            for index, query in enumerate(queries):
                if query.adapter_valid:
                    jobs.append(None)
                    indexed_valid.append((index, query))
                else:
                    jobs.append(self._invalid_result(query.invalid_reason))
            batch_job = self._executor.submit_batches(
                tuple(indexed_valid),
                chunk_size=self.ipc_chunk_size,
                profile_enabled=self.profile_enabled,
                dynamic_dispatch=self.dynamic_dispatch,
            )
            return PendingPlaneCertificateBatch(ids, tuple(queries), tuple(jobs), device, batch_job)
        jobs = []
        for query in queries:
            valid = query.adapter_valid
            if not valid:
                jobs.append(self._invalid_result(query.invalid_reason))
            elif self._executor is None:
                jobs.append(self._solve(query))
            elif self.executor_type == "subprocess":
                jobs.append(self._executor.submit(query))
            else:
                jobs.append(self._executor.submit(self._solve, query))
        return PendingPlaneCertificateBatch(ids, tuple(queries), tuple(jobs), device)

    def _raw_results(self, pending: PendingPlaneCertificateBatch) -> list[CertificateResult]:
        if pending.batch_job is not None:
            try:
                response, chunk_responses = pending.batch_job.result()
                assert isinstance(response, CertificateBatchResponse)
                indexed = dict(response.items)
                raw_results = []
                for index, job in enumerate(pending.jobs):
                    raw_results.append(job if isinstance(job, CertificateResult) else indexed[index])
                for chunk_response in chunk_responses:
                    worker = chunk_response.worker_profile
                    transport = chunk_response.transport_profile
                    if worker is not None:
                        self._record_profile("worker_wait_ms", worker.worker_wait_ms)
                        self._record_profile("worker_solve_ms", worker.worker_solve_ms)
                        self._record_profile("result_serialize_ms", worker.result_serialize_ms)
                        self._record_profile("worker_cache_size", worker.capability_cache_size)
                        self._record_profile("worker_cache_hits", worker.capability_cache_hits)
                        self._record_profile("worker_cache_misses", worker.capability_cache_misses)
                    if transport is not None:
                        self._record_profile("serialize_ms", transport.serialize_ms)
                        self._record_profile("ipc_dispatch_ms", transport.ipc_dispatch_ms)
                        self._record_profile("ipc_receive_ms", transport.ipc_receive_ms)
                        self._record_profile("result_rebuild_ms", transport.result_rebuild_ms)
                        self._record_profile("request_bytes", transport.request_bytes)
                        self._record_profile("response_bytes", transport.response_bytes)
                return raw_results
            except Exception as exc:
                fallback = self._transport_failure_result(exc)
                return [job if isinstance(job, CertificateResult) else fallback for job in pending.jobs]
        raw_results = []
        for job in pending.jobs:
            if not isinstance(job, Future):
                raw_results.append(job)
                continue
            try:
                raw_results.append(job.result())
            except Exception as exc:
                raw_results.append(self._transport_failure_result(exc))
        return raw_results

    def resolve_raw_results(self, pending: PendingPlaneCertificateBatch) -> tuple[CertificateResult, ...]:
        """Resolve in original query order without applying runtime fallback transforms."""

        return tuple(self._raw_results(pending))

    def _failure_record(self, env_id, query, result):
        period = None
        omega = None
        configuration_lookup_error = None
        try:
            config = self._plane_config(query.command, query.alpha)
            period = float(config.step_period)
            omega = float(config.omega)
        except Exception as exc:
            configuration_lookup_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        record = {
            "schema_version": 1,
            "env_id": int(env_id),
            "parameters_path": str(self.parameters_path),
            "nominal_parameters_path": str(self.nominal_parameters_path),
            "input": {
                "b": query.b.tolist(),
                "q": query.q.tolist(),
                "phase": query.phase,
                "T": period,
                "support": query.support_side,
                "command": query.command.tolist(),
                "omega": omega,
                "alpha": query.alpha,
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
        if configuration_lookup_error is not None:
            record["configuration_lookup_error"] = configuration_lookup_error
        return record

    @staticmethod
    def _transport_failure_result(exc: Exception) -> CertificateResult:
        return CertificateResult(
            CertificateStatus.SOLVER_FAILURE,
            None,
            None,
            None,
            (),
            f"Plane certificate worker transport failure: {type(exc).__name__}: {exc}",
            diagnostic={
                "kind": "worker_transport_failure",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )

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
        raw_results = self._raw_results(pending)
        results = []
        valid_results = []
        for env_id, query, result in zip(pending.env_ids, pending.queries, raw_results):
            if result.status == CertificateStatus.CONSTRAINT_BUILDER_MISMATCH:
                record = self._failure_record(env_id, query, result)
                self._save_failure_record(record)
                raise RuntimeError(
                    "G1 plane certificate feasibility/margin constraint builder mismatch; "
                    f"env_id={env_id}, diagnostics={self._diagnostics_path}, record={record}"
                )
            if not query.adapter_valid:
                results.append(self._invalid_result(query.invalid_reason))
                valid_results.append(False)
                continue
            attempted_depth = min(max(len(result.feasible_horizons) - 1, 0), 5)
            if attempted_depth:
                self._solve_depth_reached_counts[:attempted_depth] += 1
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
        h2d_started = time.perf_counter_ns()
        n_tensor = torch.tensor(
            [result.n_min for result in results], dtype=torch.long, device=pending.device
        )
        margin_tensor = torch.tensor(
            [result.margin for result in results], dtype=torch.float32, device=pending.device
        )
        valid_tensor = torch.tensor(valid_results, dtype=torch.bool, device=pending.device)
        self._record_profile("h2d_ms", (time.perf_counter_ns() - h2d_started) / 1.0e6)
        return n_tensor, margin_tensor, valid_tensor

    @property
    def profile_statistics(self) -> dict[str, object]:
        stages = {}
        for name, total in self._profile_totals.items():
            values = np.asarray(self._profile_samples.get(name, ()), dtype=np.float64)
            stages[name] = {
                "total": total,
                "count": self._profile_counts.get(name, 0),
                "mean": total / max(self._profile_counts.get(name, 0), 1),
                "p50": float(np.quantile(values, 0.50)) if values.size else 0.0,
                "p95": float(np.quantile(values, 0.95)) if values.size else 0.0,
                "p99": float(np.quantile(values, 0.99)) if values.size else 0.0,
            }
        worker_hits = self._profile_totals.get("worker_cache_hits", 0.0)
        worker_misses = self._profile_totals.get("worker_cache_misses", 0.0)
        return {
            "enabled": self.profile_enabled,
            "stages": stages,
            "parent_capability_cache": self.capability_cache_statistics,
            "worker_capability_cache": {
                "max_size": max(self._profile_samples.get("worker_cache_size", [0.0])),
                "hits": int(worker_hits),
                "misses": int(worker_misses),
                "hit_rate": worker_hits / max(worker_hits + worker_misses, 1.0),
            },
            "query_record_count": self._query_recorder.count if self._query_recorder else 0,
            "query_record_valid_count": self._query_recorder.valid_count if self._query_recorder else 0,
        }

    @property
    def statistics(self) -> dict[str, float | int]:
        statistics = dict(super().statistics)
        for horizon, count in enumerate(self._solve_depth_reached_counts, start=1):
            statistics[f"reached_F{horizon}"] = int(count)
        return statistics

    def close(self) -> None:
        if self._query_recorder is not None:
            self._query_recorder.close()
        super().close()


__all__ = [
    "PendingPlaneCertificateBatch",
    "PlaneCalibratedG1CertificateEvaluator",
    "PlaneCertificateQuery",
    "mirror_plane_certificate_query",
    "plane_periodic_state",
]
