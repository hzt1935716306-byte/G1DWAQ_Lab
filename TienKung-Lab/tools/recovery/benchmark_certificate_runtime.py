#!/usr/bin/env python3
"""Offline benchmark for exact calibrated G1 certificate queries.

This script deliberately does not import or launch Isaac Lab.  It replays real
touchdown queries saved by the existing recovery diagnostics and reports both
whole-query concurrency performance and per-LP solver timing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import threading
import time
from typing import Any

import numpy as np
import scipy
import torch
import yaml

import legged_lab.recovery.certificate as certificate
from legged_lab.recovery.g1_certificate_runtime import (
    CalibratedG1CertificateEvaluator,
    CertificateQuery,
)


DEFAULT_INPUT = Path("tools/recovery/generated/g1_q_memory_diagnostic_30_report_raw.yaml")
DEFAULT_PARAMETERS = Path("tools/recovery/generated/g1_recovery_params.yaml")

_WORKER_EVALUATOR: CalibratedG1CertificateEvaluator | None = None


def _load_queries(path: Path, count: int) -> list[CertificateQuery]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    trials = data.get("trials", data if isinstance(data, list) else [])
    queries: list[CertificateQuery] = []
    for trial in trials:
        for touchdown in trial.get("certificate_trace", []):
            raw = touchdown.get("_query")
            if raw is None:
                continue
            queries.append(
                (
                    np.asarray((raw["command_vx"], 0.0, 0.0), dtype=np.float64),
                    np.asarray(raw["b"], dtype=np.float64),
                    np.asarray(raw["q"], dtype=np.float64),
                    str(raw["support_side"]),
                    float(raw["phase"]),
                )
            )
            if len(queries) >= count:
                return queries
    if len(queries) < count:
        raise RuntimeError(f"only found {len(queries)} touchdown queries in {path}; need {count}")
    return queries


def _result_tuple(result) -> tuple[str, int | None, float | None, bool]:
    valid = (
        result.status in (certificate.CertificateStatus.FINITE, certificate.CertificateStatus.OVER_HORIZON)
        and result.n_min is not None
        and result.margin is not None
        and not result.margin_fallback
        and not result.solver_fallback
    )
    return result.status.value, result.n_min, result.margin, valid


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _versions() -> dict[str, Any]:
    highs_version = None
    try:
        from scipy.optimize import _highspy  # type: ignore[attr-defined]

        core = _highspy._core
        highs_version = ".".join(
            str(getattr(core, name))
            for name in ("HIGHS_VERSION_MAJOR", "HIGHS_VERSION_MINOR", "HIGHS_VERSION_PATCH")
        )
    except ImportError:
        pass
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "highs": highs_version,
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def _equivalence(
    reference: list[tuple[str, int | None, float | None, bool]],
    candidate: list[tuple[str, int | None, float | None, bool]],
) -> dict[str, Any]:
    status_match = all(a[0] == b[0] for a, b in zip(reference, candidate))
    n_match = all(a[1] == b[1] for a, b in zip(reference, candidate))
    valid_match = all(a[3] == b[3] for a, b in zip(reference, candidate))
    differences = [
        abs(float(a[2]) - float(b[2]))
        for a, b in zip(reference, candidate)
        if a[2] is not None and b[2] is not None
    ]
    max_margin_difference = max(differences, default=0.0)
    return {
        "status_exact": status_match,
        "N_exact": n_match,
        "valid_exact": valid_match,
        "max_abs_margin_difference": max_margin_difference,
        "within_1e-6": status_match and n_match and valid_match and max_margin_difference <= 1.0e-6,
    }


def _worker_init(parameters: str) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = CalibratedG1CertificateEvaluator(parameters, workers=1)


def _timed_worker(index_query: tuple[int, CertificateQuery]) -> tuple[int, tuple, dict[str, Any]]:
    index, query = index_query
    assert _WORKER_EVALUATOR is not None
    started = time.perf_counter()
    result = _WORKER_EVALUATOR._solve(query)
    ended = time.perf_counter()
    return index, _result_tuple(result), {
        "start": started,
        "end": ended,
        "solve_ms": 1000.0 * (ended - started),
        "pid": os.getpid(),
        "thread": threading.get_ident(),
    }


def _run_concurrency(
    mode: str,
    workers: int,
    queries: list[CertificateQuery],
    parameters: Path,
    reference: list[tuple[str, int | None, float | None, bool]] | None,
) -> tuple[dict[str, Any], list[tuple[str, int | None, float | None, bool]]]:
    indexed = list(enumerate(queries))
    submitted: dict[int, float] = {}
    timelines: list[dict[str, Any] | None] = [None] * len(queries)
    results: list[tuple[str, int | None, float | None, bool] | None] = [None] * len(queries)
    started = time.perf_counter()
    if mode == "sequential":
        _worker_init(str(parameters))
        for item in indexed:
            submitted[item[0]] = time.perf_counter()
            index, result, timeline = _timed_worker(item)
            timeline["submit"] = submitted[index]
            timeline["resolve"] = time.perf_counter()
            timelines[index] = timeline
            results[index] = result
    else:
        executor_type = ThreadPoolExecutor if mode == "thread" else ProcessPoolExecutor
        kwargs: dict[str, Any] = {"max_workers": workers}
        if mode == "process":
            kwargs.update(initializer=_worker_init, initargs=(str(parameters),))
        else:
            _worker_init(str(parameters))
        with executor_type(**kwargs) as executor:
            futures = {}
            for item in indexed:
                submitted[item[0]] = time.perf_counter()
                future = executor.submit(_timed_worker, item)
                futures[future] = item[0]
            for future in as_completed(futures):
                index, result, timeline = future.result()
                timeline["submit"] = submitted[index]
                timeline["resolve"] = time.perf_counter()
                timelines[index] = timeline
                results[index] = result
    ended = time.perf_counter()
    ordered_results = [item for item in results if item is not None]
    ordered_timelines = [item for item in timelines if item is not None]
    if len(ordered_results) != len(queries):
        raise RuntimeError("concurrency benchmark lost query results")
    report = {
        "mode": mode,
        "workers": workers,
        "query_count": len(queries),
        "wall_seconds": ended - started,
        "certificates_per_second": len(queries) / (ended - started),
        "full_query_ms": _summary([item["solve_ms"] for item in ordered_timelines]),
        "submit_to_start_ms": _summary(
            [1000.0 * (item["start"] - item["submit"]) for item in ordered_timelines]
        ),
        "end_to_resolve_ms": _summary(
            [1000.0 * (item["resolve"] - item["end"]) for item in ordered_timelines]
        ),
        "worker_processes": len({item["pid"] for item in ordered_timelines}),
        "worker_threads": len({(item["pid"], item["thread"]) for item in ordered_timelines}),
    }
    if reference is not None:
        report["equivalence"] = _equivalence(reference, ordered_results)
    return report, ordered_results


def _run_profile(
    queries: list[CertificateQuery],
    parameters: Path,
) -> tuple[dict[str, Any], list[tuple[str, int | None, float | None, bool]]]:
    evaluator = CalibratedG1CertificateEvaluator(parameters, workers=1)
    local = threading.local()
    original_build = certificate._build_problem
    original_feasibility = certificate._solve_feasibility
    original_margin = certificate._solve_with_margin_bound

    def timed_build(state, horizon, config):
        started = time.perf_counter()
        problem = original_build(state, horizon, config)
        local.events.append(
            {
                "stage": "build",
                "horizon": horizon,
                "elapsed_ms": 1000.0 * (time.perf_counter() - started),
                "variables": problem.layout.size,
                "equalities": problem.a_eq.shape[0],
                "inequalities": problem.a_ub.shape[0],
            }
        )
        return problem

    def timed_feasibility(problem, *, retry=False):
        started = time.perf_counter()
        result = original_feasibility(problem, retry=retry)
        local.events.append(
            {
                "stage": "feasibility",
                "mode": "retry" if retry else "initial",
                "horizon": problem.layout.horizon,
                "elapsed_ms": 1000.0 * (time.perf_counter() - started),
                "variables": problem.layout.size,
                "equalities": problem.a_eq.shape[0],
                "inequalities": problem.a_ub.shape[0],
                "status": int(result.status),
                "success": bool(result.success),
                "nit": int(getattr(result, "nit", 0) or 0),
            }
        )
        return result

    def timed_margin(problem, mode, bound, *, retry=False):
        started = time.perf_counter()
        result = original_margin(problem, mode, bound, retry=retry)
        local.events.append(
            {
                "stage": mode,
                "mode": "retry" if retry else "initial",
                "horizon": problem.layout.horizon,
                "elapsed_ms": 1000.0 * (time.perf_counter() - started),
                "variables": problem.layout.size + 1,
                "equalities": problem.a_eq.shape[0],
                "inequalities": problem.a_ub.shape[0],
                "status": int(result.status),
                "success": bool(result.success),
                "nit": int(getattr(result, "nit", 0) or 0),
            }
        )
        return result

    certificate._build_problem = timed_build
    certificate._solve_feasibility = timed_feasibility
    certificate._solve_with_margin_bound = timed_margin
    records = []
    results = []
    try:
        for index, query in enumerate(queries):
            config_started = time.perf_counter()
            config = evaluator._config(float(query[0][0]))
            config_ms = 1000.0 * (time.perf_counter() - config_started)
            state = certificate.CertificateState(
                b=query[1],
                q=query[2],
                support_side=query[3],
                phase=query[4],
                step_period=config.step_period,
                omega=config.omega,
            )
            local.events = []
            total_started = time.perf_counter()
            result = certificate.certify_recoverability(state, config)
            total_ms = 1000.0 * (time.perf_counter() - total_started) + config_ms
            results.append(_result_tuple(result))
            records.append(
                {
                    "index": index,
                    "config_ms": config_ms,
                    "total_ms": total_ms,
                    "result": _result_tuple(result),
                    "solver_retried": result.solver_retried,
                    "margin_fallback": result.margin_fallback,
                    "solver_fallback": result.solver_fallback,
                    "events": local.events,
                }
            )
    finally:
        certificate._build_problem = original_build
        certificate._solve_feasibility = original_feasibility
        certificate._solve_with_margin_bound = original_margin

    stage_values: dict[str, list[float]] = {"build": [], "inset": [], "relaxed": []}
    feasibility_by_horizon: dict[int, list[float]] = {horizon: [] for horizon in range(1, 6)}
    solver_events = []
    for record in records:
        for event in record["events"]:
            if event["stage"] == "feasibility":
                feasibility_by_horizon[event["horizon"]].append(event["elapsed_ms"])
            else:
                stage_values[event["stage"]].append(event["elapsed_ms"])
            if event["stage"] != "build":
                solver_events.append(event)
    slow_events = sorted(solver_events, key=lambda item: item["elapsed_ms"], reverse=True)[:20]
    report = {
        "query_count": len(queries),
        "config_ms": _summary([record["config_ms"] for record in records]),
        "build_ms": _summary(stage_values["build"]),
        "feasibility_ms_by_horizon": {
            f"F{horizon}": _summary(values)
            for horizon, values in feasibility_by_horizon.items()
        },
        "inset_ms": _summary(stage_values["inset"]),
        "relaxed_ms": _summary(stage_values["relaxed"]),
        "full_query_ms": _summary([record["total_ms"] for record in records]),
        "certificates_per_second": 1000.0
        / np.mean([record["total_ms"] for record in records]),
        "retry_fraction": float(np.mean([record["solver_retried"] for record in records])),
        "margin_fallback_fraction": float(
            np.mean([record["margin_fallback"] for record in records])
        ),
        "solver_fallback_fraction": float(
            np.mean([record["solver_fallback"] for record in records])
        ),
        "slowest_solver_events": slow_events,
    }
    return report, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("profile", "sequential", "thread", "process"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queries = _load_queries(args.input, args.samples)
    reference = None
    if args.reference is not None:
        reference_data = json.loads(args.reference.read_text(encoding="utf-8"))
        reference = [tuple(item) for item in reference_data["results"]]
    if args.mode == "profile":
        benchmark, results = _run_profile(queries, args.parameters)
    else:
        benchmark, results = _run_concurrency(
            args.mode, args.workers, queries, args.parameters, reference
        )
    report = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "parameters": str(args.parameters.resolve()),
        "versions": _versions(),
        "benchmark": benchmark,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"versions": report["versions"], "benchmark": benchmark}, indent=2))


if __name__ == "__main__":
    main()
