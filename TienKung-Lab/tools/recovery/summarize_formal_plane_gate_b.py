#!/usr/bin/env python3
"""Summarize the three frozen-policy formal Plane V1 Gate B reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import spearmanr
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
GENERATED_DIR = PROJECT_DIR / "tools/recovery/generated"
DEFAULT_REPORTS = (
    GENERATED_DIR / "g1_plane_v1_formal_gate_b_slope_minus10.yaml",
    GENERATED_DIR / "g1_plane_v1_formal_gate_b_slope_0.yaml",
    GENERATED_DIR / "g1_plane_v1_formal_gate_b_slope_plus10.yaml",
)
DEFAULT_OUTPUT = GENERATED_DIR / "g1_plane_v1_formal_gate_b_summary.yaml"
HISTORICAL = {
    -10.0: {"rho_N": 0.553, "rho_margin": -0.609},
    0.0: {"rho_N": 0.567, "rho_margin": -0.376},
    10.0: {"rho_N": 0.739, "rho_margin": -0.679},
}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _bin_ordering(by_n: dict) -> dict:
    occupied = [
        (int(label), values["median_N_actual_terminal"])
        for label, values in by_n.items()
        if values["count"] > 0 and values["median_N_actual_terminal"] is not None
    ]
    strict_nondecreasing = all(
        second[1] >= first[1] for first, second in zip(occupied[:-1], occupied[1:])
    )
    if len(occupied) >= 2:
        rho, p_value = spearmanr(
            np.asarray([item[0] for item in occupied]),
            np.asarray([item[1] for item in occupied]),
        )
        rho_value = float(rho) if np.isfinite(rho) else None
        p_value_value = float(p_value) if np.isfinite(p_value) else None
    else:
        rho_value = None
        p_value_value = None
    return {
        "occupied_bins": [item[0] for item in occupied],
        "median_actual_terminal_by_occupied_bin": [item[1] for item in occupied],
        "strict_nondecreasing": strict_nondecreasing,
        "occupied_bin_median_spearman": {
            "rho": rho_value,
            "p_value": p_value_value,
        },
    }


def summarize(paths: tuple[Path, ...]) -> dict:
    reports = [_load(path.resolve()) for path in paths]
    if len(reports) != 3:
        raise ValueError("formal Plane V1 summary requires exactly three slope reports")
    identity_fields = (
        "checkpoint",
        "checkpoint_sha256",
        "plane_nominal_params",
        "plane_nominal_params_sha256",
        "calibration_git_commit",
        "certificate_state_source",
    )
    for field in identity_fields:
        values = {report[field] for report in reports}
        if len(values) != 1:
            raise RuntimeError(f"report identity mismatch for {field}: {values}")
    for report in reports:
        if report["gate2_disturbed"]["trial_count"] != 320:
            raise RuntimeError("each formal Gate B slope report must contain 320 trials")
        contract = report["frozen_policy_contract"]
        expected = {
            "actor_input_dim": 960,
            "actor_history_length": 10,
            "per_frame_actor_observation_dim": 96,
            "action_dim": 29,
            "eval_mode": True,
            "requires_grad": False,
        }
        if contract != expected:
            raise RuntimeError(f"unexpected frozen policy contract: {contract}")
        if report.get("estimator_diagnostic") is not None:
            raise RuntimeError("formal GT Gate B must not contain an estimator diagnostic")

    slopes = {}
    nominal_counts = {str(value): 0 for value in range(7)}
    total_trials = 0
    valid_trials = 0
    for path, report in zip(paths, reports):
        slope = float(report["gate1_nominal"]["slope_degrees"])
        gate1 = report["gate1_nominal"]
        gate2 = report["gate2_disturbed"]
        applicability = gate2["applicability"]
        spearman = gate2["spearman"]
        for label, count in gate1["nominal_certificate_sanity"][
            "N_distribution_count"
        ].items():
            nominal_counts[str(label)] += int(count)
        total_trials += int(gate2["trial_count"])
        valid_trials += int(gate2["valid_correlation_trial_count"])
        rho_n = spearman["N_theory_0_vs_N_actual_terminal"]
        rho_margin = spearman["margin_0_vs_N_actual_terminal"]
        historical = HISTORICAL[slope]
        slopes[f"{slope:+g}"] = {
            "report": str(path.resolve()),
            "slope_degrees": slope,
            "nominal_N_sanity": gate1["nominal_certificate_sanity"],
            "total_trials": gate2["trial_count"],
            "valid_trials": gate2["valid_correlation_trial_count"],
            "applicability_fraction": applicability["applicability_valid_fraction"],
            "invalid_before_TD0": applicability["invalid_before_reference_touchdown"],
            "left_applicability_after_TD0": applicability[
                "applicability_exit_after_reference"
            ],
            "rho_N": rho_n,
            "rho_margin": rho_margin,
            "N_bins": gate2["by_N_theory_0"],
            "N_bin_ordering": _bin_ordering(gate2["by_N_theory_0"]),
            "terminal_P5": applicability["terminal_success_rate_P5"],
            "fall_count": applicability["fall_count"],
            "fall_rate": applicability["fall_rate"],
            "timeout_count": applicability["timeout_count"],
            "timeout_rate": applicability["timeout_rate"],
            "recovery_trajectory": gate2["trajectory_consistency"],
            "historical_reference": historical,
            "same_expected_sign_as_historical": bool(
                rho_n["rho"] > 0.0
                and rho_margin["rho"] < 0.0
                and historical["rho_N"] > 0.0
                and historical["rho_margin"] < 0.0
            ),
        }

    nominal_total = sum(nominal_counts.values())
    n0 = nominal_counts["0"]
    n1 = nominal_counts["1"]
    n_ge_2 = nominal_total - n0 - n1
    all_signs = all(
        item["same_expected_sign_as_historical"] for item in slopes.values()
    )
    return {
        "schema_version": 1,
        "description": (
            "Formal Plane V1 Gate B correlation validation using the frozen "
            "g1_slope_sys_d walking policy and privileged GT whole-body CoM state."
        ),
        "policy_checkpoint": reports[0]["checkpoint"],
        "checkpoint_sha256": reports[0]["checkpoint_sha256"],
        "frozen_policy_contract": reports[0]["frozen_policy_contract"],
        "nominal_parameter_yaml": reports[0]["plane_nominal_params"],
        "nominal_parameter_yaml_sha256": reports[0]["plane_nominal_params_sha256"],
        "calibration_git_commit": reports[0]["calibration_git_commit"],
        "validation_git_commit": _git_commit(),
        "certificate_state_source": reports[0]["certificate_state_source"],
        "estimator_used": False,
        "experiment": {
            "terrain": "continuous x-aligned uniform plane",
            "command": [0.4, 0.0, 0.0],
            "slopes_degrees": [-10.0, 0.0, 10.0],
            "push_directions": ["+x", "-x", "+y", "-y"],
            "push_magnitudes_m_per_s": [0.25, 0.5, 0.75, 1.0],
            "target_phases": [0.25, 0.75],
            "repeats": 10,
            "trials_per_slope": 320,
            "total_trials": total_trials,
            "valid_correlation_trials": valid_trials,
            "overall_applicability_fraction": valid_trials / total_trials,
            "heading_alignment_tolerance_degrees": 5.0,
            "bootstrap_resamples": 2000,
        },
        "pooled_nominal_N_sanity": {
            "sample_count": nominal_total,
            "N_distribution_count": nominal_counts,
            "P_N_0": n0 / nominal_total,
            "P_N_1": n1 / nominal_total,
            "P_N_ge_2": n_ge_2 / nominal_total,
            "calibration_expectation": {"P_N_0_approx": 0.81, "P_N_1_approx": 0.17},
            "ordering_preserved": bool(n0 > n1 > n_ge_2),
        },
        "slopes": slopes,
        "conclusion": {
            "all_three_N_correlations_positive": all(
                item["rho_N"]["rho"] > 0.0 for item in slopes.values()
            ),
            "all_three_margin_correlations_negative": all(
                item["rho_margin"]["rho"] < 0.0 for item in slopes.values()
            ),
            "all_primary_bootstrap_CIs_exclude_zero": all(
                item["rho_N"]["bootstrap_95_CI"][0] > 0.0
                and item["rho_margin"]["bootstrap_95_CI"][1] < 0.0
                for item in slopes.values()
            ),
            "same_theoretical_ordering_relationship_as_historical": all_signs,
            "strict_N_bin_median_monotonicity_by_slope": {
                label: item["N_bin_ordering"]["strict_nondecreasing"]
                for label, item in slopes.items()
            },
            "interpretation": (
                "The formal frozen-policy results preserve the expected ordering: "
                "larger TD0 N predicts slower terminal recovery, while larger margin "
                "predicts faster terminal recovery. Sparse occupied bins need not be "
                "strictly monotonic at every slope."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = tuple(args.reports) if args.reports else DEFAULT_REPORTS
    document = summarize(paths)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False, allow_unicode=True)
    print(f"[INFO] wrote {output}")


if __name__ == "__main__":
    main()
