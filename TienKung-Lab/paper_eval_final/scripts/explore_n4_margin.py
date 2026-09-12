"""Deterministic certificate-only exploration for the sparse Nmin=4/high-margin cell.

This diagnostic never reads recovery, survival, fall, or touchdown outcomes.  It
perturbs only the offline certificate query state recorded at TD0 and therefore
cannot be used as an analysis sample; it is solely a pilot coverage-design aid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from legged_lab.recovery.plane_certificate_runtime import (
    PlaneCalibratedG1CertificateEvaluator,
    PlaneCertificateQuery,
)
from paper_eval_final.src.common import LAB, ROOT, write_json


def _queries(record_root: Path) -> list[PlaneCertificateQuery]:
    rows = [json.loads(path.read_text()) for path in record_root.glob("*.json")]
    rows.sort(key=lambda row: (int(row.get("sequence", 10**18)), row["trial_id"]))
    output = []
    for row in rows:
        calculation = row.get("certificate_calculation") or {}
        inputs = calculation.get("input") or {}
        if not row.get("certificate_valid") or row.get("Nmin") != 4:
            continue
        output.append(PlaneCertificateQuery(
            command=np.asarray(inputs["command"], dtype=np.float64),
            b=np.asarray(inputs["b"], dtype=np.float64),
            q=np.asarray(inputs["q"], dtype=np.float64),
            support_side=str(inputs["support_side"]),
            phase=float(inputs["phase"]),
            alpha=float(inputs["slope_rad"]),
            adapter_valid=True,
        ))
    if not output:
        raise ValueError("no certificate-valid Nmin=4 queries found")
    return output


def explore(record_root: Path, *, samples: int, seed: int, q2: float) -> dict:
    source = _queries(record_root)
    rng = np.random.default_rng(seed)
    evaluator = PlaneCalibratedG1CertificateEvaluator(
        LAB / "tools/recovery/generated/g1_recovery_params.yaml",
        LAB / "tools/recovery/generated/g1_plane_nominal_params_g1_slope_sys_d_candidate.yaml",
        workers=1,
        executor_type="sequential",
        use_state_b=True,
    )
    scales = (0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.25, 0.40)
    counts = {str(scale): {"evaluated": 0, "N4": 0, "N4_HIGH": 0} for scale in scales}
    best: list[dict] = []
    cross_grid: list[dict] = []
    try:
        for index in range(samples):
            base = source[index % len(source)]
            scale = scales[(index // len(source)) % len(scales)]
            query = PlaneCertificateQuery(
                command=base.command.copy(),
                b=base.b + rng.normal(0.0, scale, size=2),
                q=base.q + rng.normal(0.0, scale, size=2),
                support_side=base.support_side,
                phase=0.0,
                alpha=base.alpha,
                adapter_valid=True,
            )
            result = evaluator._solve(query)
            bucket = counts[str(scale)]
            bucket["evaluated"] += 1
            if result.n_min == 4 and result.margin is not None:
                bucket["N4"] += 1
                if float(result.margin) >= q2:
                    bucket["N4_HIGH"] += 1
                best.append({
                    "margin_raw": float(result.margin),
                    "Nmin": 4,
                    "perturbation_scale": scale,
                    "command": query.command.tolist(),
                    "slope_rad": query.alpha,
                    "support_side": query.support_side,
                    "b": query.b.tolist(),
                    "q": query.q.tolist(),
                })
        # Re-evaluate every observed N4 state on all calibrated slope/speed
        # nodes.  This diagnoses missing condition coverage without assuming
        # that a synthetic query is itself a physical trial.
        for base in source:
            for slope_deg in (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0):
                for speed in (0.2, 0.4, 0.6, 0.8, 1.0):
                    command = base.command.copy()
                    command[0] = speed
                    query = PlaneCertificateQuery(
                        command=command,
                        b=base.b.copy(),
                        q=base.q.copy(),
                        support_side=base.support_side,
                        phase=0.0,
                        alpha=np.deg2rad(slope_deg),
                        adapter_valid=True,
                    )
                    result = evaluator._solve(query)
                    if result.n_min == 4 and result.margin is not None:
                        cross_grid.append({
                            "margin_raw": float(result.margin), "Nmin": 4,
                            "speed": speed, "slope_deg": slope_deg,
                            "support_side": query.support_side,
                            "b": query.b.tolist(), "q": query.q.tolist(),
                        })
    finally:
        evaluator.close()
    best.sort(key=lambda item: item["margin_raw"], reverse=True)
    cross_grid.sort(key=lambda item: item["margin_raw"], reverse=True)
    return {
        "purpose": "pilot coverage design only; not an experimental analysis dataset",
        "selection_fields_excluded": [
            "task_outcome", "recovered_sustained", "recovery_time", "nTD0",
            "Krec", "Krec_physical", "fall", "survival",
        ],
        "source_N4_query_count": len(source),
        "samples": samples,
        "seed": seed,
        "shared_q2": q2,
        "counts_by_query_perturbation_scale": counts,
        "N4_count": len(best),
        "N4_HIGH_count": sum(item["margin_raw"] >= q2 for item in best),
        "maximum_N4_margin": best[0]["margin_raw"] if best else None,
        "top_N4_queries": best[:100],
        "cross_grid_N4_count": len(cross_grid),
        "cross_grid_N4_HIGH_count": sum(item["margin_raw"] >= q2 for item in cross_grid),
        "cross_grid_maximum_N4_margin": cross_grid[0]["margin_raw"] if cross_grid else None,
        "cross_grid_top_N4_queries": cross_grid[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record_root", type=Path)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=234_20260912)
    parser.add_argument("--q2", type=float, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/pilot/experiment_1_n4_margin_query_exploration.json",
    )
    args = parser.parse_args()
    result = explore(args.record_root, samples=args.samples, seed=args.seed, q2=args.q2)
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "source_N4_query_count", "samples", "N4_count", "N4_HIGH_count", "maximum_N4_margin",
        "cross_grid_N4_count", "cross_grid_N4_HIGH_count", "cross_grid_maximum_N4_margin",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
