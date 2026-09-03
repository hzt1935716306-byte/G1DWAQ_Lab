#!/usr/bin/env python3
"""Offline left/right certificate diagnostic on saved real G1 touchdown states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from legged_lab.recovery.g1_certificate_runtime import CalibratedG1CertificateEvaluator


def _load_queries(path: Path, count: int, include_nonzero_phase: bool) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries = []
    for trial in data["trials"]:
        for touchdown in trial.get("certificate_trace", []):
            raw = touchdown.get("_query")
            if raw is None:
                continue
            phase = float(raw["phase"])
            if not include_nonzero_phase and abs(phase) > 1.0e-12:
                continue
            queries.append(
                {
                    "command": np.asarray((raw["command_vx"], 0.0, 0.0), dtype=np.float64),
                    "b": np.asarray(raw["b"], dtype=np.float64),
                    "q": np.asarray(raw["q"], dtype=np.float64),
                    "support": str(raw["support_side"]),
                    "phase": phase,
                }
            )
            if len(queries) >= count:
                return queries
    raise RuntimeError(f"found only {len(queries)} queries in {path}; requested {count}")


def _solve(evaluator, item: dict):
    query = (item["command"], item["b"], item["q"], item["support"], item["phase"])
    result = evaluator._solve(query)
    valid = (
        result.n_min is not None
        and result.margin is not None
        and not result.margin_fallback
        and not result.solver_fallback
    )
    return result, valid


def _mirrored(item: dict) -> dict:
    command = item["command"].copy()
    b = item["b"].copy()
    q = item["q"].copy()
    command[1:] *= -1.0
    b[1] *= -1.0
    q[1] *= -1.0
    return {
        "command": command,
        "b": b,
        "q": q,
        "support": "right" if item["support"] == "left" else "left",
        "phase": item["phase"],
    }


def _stats(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["valid_pair"]]
    differences = np.asarray([row["margin_abs_difference"] for row in valid])
    mismatches = np.asarray([row["N_mismatch"] for row in valid], dtype=np.float64)
    return {
        "sample_count": len(rows),
        "valid_pair_count": len(valid),
        "N_mismatch_count": int(np.sum(mismatches)),
        "N_mismatch_rate": float(np.mean(mismatches)) if len(valid) else None,
        "margin_abs_difference_median": float(np.quantile(differences, 0.50)) if len(valid) else None,
        "margin_abs_difference_p90": float(np.quantile(differences, 0.90)) if len(valid) else None,
        "margin_abs_difference_p95": float(np.quantile(differences, 0.95)) if len(valid) else None,
        "margin_abs_difference_p99": float(np.quantile(differences, 0.99)) if len(valid) else None,
        "margin_abs_difference_max": float(np.max(differences)) if len(valid) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tools/recovery/generated/g1_q_memory_diagnostic_30_report_raw.yaml"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("tools/recovery/generated/g1_recovery_params.yaml"),
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--include_nonzero_phase",
        action="store_true",
        help="Also include post-push queries that are not touchdown phase zero.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/recovery/generated/certificate_mirror_real_touchdown_100.json"),
    )
    args = parser.parse_args()

    evaluator = CalibratedG1CertificateEvaluator(
        args.parameters,
        workers=1,
        executor_type="sequential",
    )
    rows = []
    for index, item in enumerate(
        _load_queries(args.input, args.samples, args.include_nonzero_phase)
    ):
        original, original_valid = _solve(evaluator, item)
        mirror, mirror_valid = _solve(evaluator, _mirrored(item))
        valid_pair = original_valid and mirror_valid
        rows.append(
            {
                "index": index,
                "support": item["support"],
                "phase": item["phase"],
                "command_vx": float(item["command"][0]),
                "N_original": original.n_min,
                "N_mirror": mirror.n_min,
                "margin_original": original.margin,
                "margin_mirror": mirror.margin,
                "valid_pair": valid_pair,
                "N_mismatch": bool(valid_pair and original.n_min != mirror.n_min),
                "margin_abs_difference": (
                    abs(float(original.margin) - float(mirror.margin)) if valid_pair else None
                ),
            }
        )

    phase_zero = [row for row in rows if abs(row["phase"]) <= 1.0e-12]
    phase_nonzero = [row for row in rows if abs(row["phase"]) > 1.0e-12]
    report = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "parameters": str(args.parameters.resolve()),
        "overall": _stats(rows),
        "by_support": {
            side: _stats([row for row in rows if row["support"] == side])
            for side in ("left", "right")
        },
        "by_phase": {
            "touchdown_phase_zero": _stats(phase_zero),
            "post_push_nonzero_phase": _stats(phase_nonzero),
        },
        "by_original_N": {
            str(n_min): _stats([row for row in rows if row["N_original"] == n_min])
            for n_min in sorted({row["N_original"] for row in rows})
        },
        "mismatch_rows": [row for row in rows if row["N_mismatch"]],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
