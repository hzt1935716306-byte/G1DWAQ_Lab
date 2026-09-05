#!/usr/bin/env python3
"""Replay exact production Plane queries through legacy and optimized runtimes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import time
from typing import Any

import numpy as np
import scipy
import torch

from legged_lab.recovery.certificate import CertificateResult, CertificateStatus
from legged_lab.recovery.certificate_query_corpus import load_plane_query_corpus
from legged_lab.recovery.plane_certificate_runtime import PlaneCalibratedG1CertificateEvaluator


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_FLAT = PROJECT / "tools/recovery/generated/g1_recovery_params.yaml"
DEFAULT_NOMINAL = PROJECT / "tools/recovery/generated/g1_plane_nominal_params_g1_slope_sys_d_candidate.yaml"


def _quantiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(data, 0.50)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(np.max(data)),
    }


def _ordered_int(value: float) -> int:
    raw = int(np.asarray(value, dtype=np.float64).view(np.uint64))
    return (~raw & ((1 << 64) - 1)) if raw >> 63 else raw | (1 << 63)


def _ulp_distance(left: float, right: float) -> int:
    return abs(_ordered_int(left) - _ordered_int(right))


def _normal(result: CertificateResult) -> bool:
    return (
        result.status in (CertificateStatus.FINITE, CertificateStatus.OVER_HORIZON)
        and result.n_min is not None
        and result.margin is not None
        and not result.margin_fallback
        and not result.solver_fallback
    )


def _equivalence(
    reference: list[CertificateResult], candidate: list[CertificateResult]
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("result lengths differ")
    margin_abs = []
    margin_rel = []
    margin_ulp = []
    for old, new in zip(reference, candidate):
        if old.margin is None or new.margin is None:
            continue
        difference = abs(float(old.margin) - float(new.margin))
        margin_abs.append(difference)
        margin_rel.append(difference / max(abs(float(old.margin)), abs(float(new.margin)), np.finfo(float).tiny))
        margin_ulp.append(_ulp_distance(float(old.margin), float(new.margin)))
    return {
        "query_count": len(reference),
        "valid_mismatch": sum(_normal(a) != _normal(b) for a, b in zip(reference, candidate)),
        "N_min_mismatch": sum(a.n_min != b.n_min for a, b in zip(reference, candidate)),
        "status_mismatch": sum(a.status != b.status for a, b in zip(reference, candidate)),
        "failure_semantics_mismatch": sum(
            (a.margin_fallback, a.solver_fallback, a.solver_retried, a.diagnostic)
            != (b.margin_fallback, b.solver_fallback, b.solver_retried, b.diagnostic)
            for a, b in zip(reference, candidate)
        ),
        "margin_bitwise_mismatch": sum(value != 0 for value in margin_ulp),
        "margin_max_abs": max(margin_abs, default=0.0),
        "margin_max_relative": max(margin_rel, default=0.0),
        "margin_max_ulp": max(margin_ulp, default=0),
        "margin_p99_ulp": float(np.quantile(margin_ulp, 0.99)) if margin_ulp else 0.0,
    }


def _run(
    queries,
    *,
    workers: int,
    batch: bool,
    chunk_size: int,
    cache: bool,
    block_size: int,
    profile: bool,
    flat_parameters: Path,
    nominal_parameters: Path,
) -> tuple[dict[str, Any], list[CertificateResult]]:
    evaluator = PlaneCalibratedG1CertificateEvaluator(
        flat_parameters,
        nominal_parameters,
        workers=workers,
        executor_type="subprocess",
        ipc_batch_enabled=batch,
        ipc_chunk_size=chunk_size,
        dynamic_dispatch=batch,
        exact_alpha_cache=cache,
        profile_enabled=profile,
    )
    results: list[CertificateResult] = []
    batch_ms = []
    started = time.perf_counter()
    try:
        for offset in range(0, len(queries), block_size):
            block = queries[offset : offset + block_size]
            batch_started = time.perf_counter()
            pending = evaluator.submit_queries(block, torch.device("cpu"))
            results.extend(evaluator.resolve_raw_results(pending))
            batch_ms.append((time.perf_counter() - batch_started) * 1000.0)
    finally:
        evaluator.close()
    wall = time.perf_counter() - started
    statuses = Counter(result.status.value for result in results)
    n_values = Counter(str(result.n_min) for result in results)
    depth = Counter()
    for result in results:
        reached = min(max(len(result.feasible_horizons) - 1, 0), 5)
        for horizon in range(1, reached + 1):
            depth[f"F{horizon}"] += 1
    return {
        "workers": workers,
        "ipc_batch": batch,
        "chunk_size": chunk_size,
        "dynamic_dispatch": batch,
        "exact_alpha_cache": cache,
        "queries": len(queries),
        "wall_seconds": wall,
        "queries_per_second": len(queries) / wall,
        "batch_latency_ms": _quantiles(batch_ms),
        "status_distribution": dict(statuses),
        "N_distribution": dict(n_values),
        "solve_depth": dict(depth),
        "profile": evaluator.profile_statistics,
    }, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--flat_parameters", type=Path, default=DEFAULT_FLAT)
    parser.add_argument("--nominal_parameters", type=Path, default=DEFAULT_NOMINAL)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--worker_counts", type=int, nargs="+", default=(8, 12, 16))
    parser.add_argument("--chunk_sizes", type=int, nargs="+", default=(1, 4, 8, 16, 32))
    parser.add_argument("--best_workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    records = load_plane_query_corpus(args.corpus, valid_only=True, limit=args.count)
    if len(records) < args.count:
        raise RuntimeError(f"only {len(records)} valid queries in corpus; need {args.count}")
    queries = tuple(record.query for record in records)
    payload_sizes = [len(pickle.dumps(query, protocol=pickle.HIGHEST_PROTOCOL)) for query in queries[:10000]]
    report: dict[str, Any] = {
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "flat_parameters": str(args.flat_parameters.resolve()),
        "flat_parameters_sha256": hashlib.sha256(args.flat_parameters.read_bytes()).hexdigest(),
        "nominal_parameters": str(args.nominal_parameters.resolve()),
        "nominal_parameters_sha256": hashlib.sha256(args.nominal_parameters.read_bytes()).hexdigest(),
        "query_count": len(queries),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cpu": platform.processor(),
            "logical_cpus": __import__("os").cpu_count(),
        },
        "pickle_payload_bytes_per_query": _quantiles([float(value) for value in payload_sizes]),
    }

    legacy_runs = []
    reference_results = None
    for workers in args.worker_counts:
        summary, results = _run(
            queries,
            workers=workers,
            batch=False,
            chunk_size=1,
            cache=False,
            block_size=args.block_size,
            profile=False,
            flat_parameters=args.flat_parameters,
            nominal_parameters=args.nominal_parameters,
        )
        if reference_results is None:
            reference_results = results
        else:
            summary["equivalence_to_legacy_reference"] = _equivalence(reference_results, results)
        legacy_runs.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    assert reference_results is not None

    repeat_summary, repeat_results = _run(
        queries,
        workers=args.worker_counts[0],
        batch=False,
        chunk_size=1,
        cache=False,
        block_size=args.block_size,
        profile=False,
        flat_parameters=args.flat_parameters,
        nominal_parameters=args.nominal_parameters,
    )
    repeat_summary["old_vs_old"] = _equivalence(reference_results, repeat_results)
    report["legacy_worker_sweep"] = legacy_runs
    report["legacy_repeat"] = repeat_summary

    chunk_runs = []
    for chunk_size in args.chunk_sizes:
        summary, results = _run(
            queries,
            workers=args.best_workers,
            batch=True,
            chunk_size=chunk_size,
            cache=True,
            block_size=args.block_size,
            profile=args.profile,
            flat_parameters=args.flat_parameters,
            nominal_parameters=args.nominal_parameters,
        )
        summary["equivalence_to_legacy_reference"] = _equivalence(reference_results, results)
        chunk_runs.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    report["optimized_chunk_sweep"] = chunk_runs

    no_cache, no_cache_results = _run(
        queries,
        workers=args.best_workers,
        batch=True,
        chunk_size=min(args.chunk_sizes, key=lambda value: abs(value - 8)),
        cache=False,
        block_size=args.block_size,
        profile=args.profile,
        flat_parameters=args.flat_parameters,
        nominal_parameters=args.nominal_parameters,
    )
    no_cache["equivalence_to_legacy_reference"] = _equivalence(reference_results, no_cache_results)
    report["optimized_no_cache"] = no_cache
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
