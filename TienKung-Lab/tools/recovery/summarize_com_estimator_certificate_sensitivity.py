#!/usr/bin/env python3
"""Aggregate paired GT/standalone-estimator Plane certificate diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from legged_lab.recovery.dwaq_estimator_diagnostic import (
    certificate_agreement,
    dcm_velocity_error_statistics,
    terminal_ordering,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
GENERATED = PROJECT_DIR / "tools/recovery/generated"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--inputs",
    type=Path,
    nargs="+",
    default=[
        GENERATED / "g1_com_velocity_estimator_certificate_sensitivity_slope_minus10.yaml",
        GENERATED / "g1_com_velocity_estimator_certificate_sensitivity_slope_plus0.yaml",
        GENERATED / "g1_com_velocity_estimator_certificate_sensitivity_slope_plus10.yaml",
    ],
)
parser.add_argument(
    "--output",
    type=Path,
    default=GENERATED / "g1_com_velocity_estimator_certificate_sensitivity.yaml",
)
args = parser.parse_args()


def _load(path: Path) -> dict:
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _samples(report: dict) -> list[dict]:
    rows = []
    for trial in report["gate2_disturbed"]["trials"]:
        for sample in trial["certificate_trace"]:
            velocity = sample.get("velocity_diagnostic")
            if velocity is None:
                continue
            rows.append(
                {
                    **sample,
                    **velocity,
                    "N_actual_terminal": trial["N_actual_terminal"],
                    "slope_degrees": float(report["gate2_disturbed"]["conditions"]["slope_degrees"]),
                    "push_magnitude": trial["push_magnitude"],
                    "push_direction": trial["push_direction"],
                    "target_phase": trial["target_phase"],
                }
            )
    return rows


def _dcm(rows: list[dict]) -> dict:
    if not rows:
        return {"sample_count": 0}
    return dcm_velocity_error_statistics(
        [row["direct_com_est_heading"][:2] for row in rows],
        [row["com_GT_heading"][:2] for row in rows],
        [row["omega"] for row in rows],
    )


def _degradation(gt: dict, estimate: dict, metric: str) -> float | None:
    gt_value = gt.get(metric)
    estimate_value = estimate.get(metric)
    if gt_value is None or estimate_value is None:
        return None
    return float(estimate_value - gt_value)


def main() -> None:
    paths = [path.expanduser().resolve() for path in args.inputs]
    reports = [_load(path) for path in paths]
    for report in reports:
        if report.get("validation_mode") != "plane_com_velocity_estimator_certificate_sensitivity":
            raise ValueError("input is not a standalone CoM estimator sensitivity report")
    all_samples = [row for report in reports for row in _samples(report)]
    touchdown_groups = {
        "TD0": [row for row in all_samples if row["touchdown"] == 0],
        "TD1": [row for row in all_samples if row["touchdown"] == 1],
        "TD2": [row for row in all_samples if row["touchdown"] == 2],
        "TD3_plus": [row for row in all_samples if 3 <= row["touchdown"] <= 5],
        "TD0_to_TD5": all_samples,
    }
    agreement = {
        name: certificate_agreement(rows, "EST")
        for name, rows in touchdown_groups.items()
    }
    ordering_by_slope = {}
    for report, rows in zip(reports, [_samples(report) for report in reports]):
        slope = float(report["gate2_disturbed"]["conditions"]["slope_degrees"])
        paired = [
            row
            for row in rows
            if row.get("certificate_valid_GT", False)
            and row.get("certificate_valid_EST", False)
        ]
        gt = terminal_ordering(paired, "GT")
        estimate = terminal_ordering(paired, "EST")
        ordering_by_slope[f"{slope:+g}"] = {
            "GT": gt,
            "EST": estimate,
            "EST_minus_GT": {
                "N_vs_terminal_spearman": _degradation(
                    gt, estimate, "N_vs_terminal_spearman"
                ),
                "margin_vs_terminal_spearman": _degradation(
                    gt, estimate, "margin_vs_terminal_spearman"
                ),
            },
        }
    trial_counts = {
        f"{float(report['gate2_disturbed']['conditions']['slope_degrees']):+g}": int(
            report["gate2_disturbed"]["trial_count"]
        )
        for report in reports
    }
    output = {
        "schema_version": 1,
        "diagnostic": "standalone_5x96_CoM_velocity_estimator_to_plane_certificate_sensitivity",
        "source_reports": [str(path) for path in paths],
        "frozen_policy": reports[0]["checkpoint"],
        "estimator": reports[0]["estimator_diagnostic"],
        "unchanged_inputs": [
            "whole-body CoM position",
            "feet positions and q",
            "support side",
            "terrain/slope",
            "command",
            "T/h/omega/nominal parameters",
            "C/L/vmax and certificate implementation",
        ],
        "only_replaced_input": "whole-body CoM velocity XY in heading frame",
        "conditions": {
            "slopes_degrees": [-10.0, 0.0, 10.0],
            "command": [0.4, 0.0, 0.0],
            "push_directions": ["+x", "-x", "+y", "-y"],
            "push_magnitudes_m_per_s": [0.25, 0.50, 0.75, 1.00],
            "target_phases": [0.25, 0.75],
            "repeats": 2,
            "trial_count_by_slope": trial_counts,
            "total_trial_count": int(sum(trial_counts.values())),
        },
        "legal_touchdown_sample_counts": {
            name: len(rows) for name, rows in touchdown_groups.items()
        },
        "certificate_agreement": agreement,
        "DCM_velocity_induced_error_by_touchdown_cm": {
            name: _dcm(touchdown_groups[name]) for name in ("TD0", "TD1", "TD2")
        },
        # Retain the original field for compatibility with the V1 report reader.
        "TD0_DCM_velocity_induced_error_cm": _dcm(touchdown_groups["TD0"]),
        "Gate_B_terminal_ordering_by_slope": ordering_by_slope,
        "applicability": {
            f"{float(report['gate2_disturbed']['conditions']['slope_degrees']):+g}": report[
                "gate2_disturbed"
            ]["applicability"]
            for report in reports
        },
        "notes": {
            "actual_difficulty_label": "N_actual_terminal; practical detector is not used for estimator classification",
            "N_encoding": "0..5 finite recovery steps; 6 means >5/fall",
            "epsilon_b_modified": False,
            "models_retrained": False,
        },
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False, allow_unicode=True)
    print(yaml.safe_dump(output, sort_keys=False, allow_unicode=True))
    print(f"[INFO] Saved aggregate sensitivity report to {output_path}")


if __name__ == "__main__":
    main()
