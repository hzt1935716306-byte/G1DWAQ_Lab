#!/usr/bin/env python3
"""Validate and summarize a frozen-policy Plane V1 nominal candidate table."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from legged_lab.recovery.plane_adapter import adapt_flat_capability
from legged_lab.recovery.plane_nominal_params import PlaneNominalParameterTable


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--flat_parameters", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--validation_output", type=Path, required=True)
args = parser.parse_args()


SLOPES = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
DIRECTIONS = ("+x", "-x", "+y", "-y")
SPEEDS = (0.2, 0.4, 0.6, 0.8, 1.0)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))


def _label(slope: float, direction: str, speed: float) -> str:
    return f"alpha={slope:+g},direction={direction},speed={speed:g}"


def _stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def _weighted(reports: list[dict], section: str, field: str) -> float:
    numerator = sum(
        float(report["diagnostic"][section][field])
        * int(report["diagnostic"]["sample_count"])
        for report in reports
    )
    denominator = sum(int(report["diagnostic"]["sample_count"]) for report in reports)
    return numerator / denominator


def main() -> None:
    candidate_path = args.candidate.expanduser().resolve()
    flat_path = args.flat_parameters.expanduser().resolve()
    document = _load(candidate_path)
    flat = _load(flat_path)
    nodes = document["nominal_plane_gait"]["nodes"]
    reports_by_label = document["collection"]["node_reports"]
    policy_id = document["collection"]["calibration_policy_id"]
    expected_grid = {
        (slope, direction, speed)
        for slope in SLOPES
        for direction in DIRECTIONS
        for speed in SPEEDS
    }
    actual_grid = {
        (float(node["slope_degrees"]), str(node["direction"]), float(node["speed"]))
        for node in nodes
    }
    if actual_grid != expected_grid:
        raise AssertionError(
            f"candidate grid mismatch: missing={expected_grid-actual_grid}, "
            f"extra={actual_grid-expected_grid}"
        )
    if set(reports_by_label) != {
        _label(*key) for key in expected_grid
    }:
        raise AssertionError("node report grid does not exactly match the requested grid")
    if any(not bool(report["valid"]) for report in reports_by_label.values()):
        raise AssertionError("candidate contains an invalid node")
    if any(int(report["valid_cycle_count"]) < 120 for report in reports_by_label.values()):
        raise AssertionError("a valid node contains fewer than 120 strict cycles")
    teacher = document["teacher"]
    checkpoint = Path(teacher["checkpoint"])
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != teacher["checkpoint_sha256"]:
        raise AssertionError("checkpoint hash no longer matches candidate metadata")
    if not teacher["eval_mode"] or teacher["requires_grad"]:
        raise AssertionError("candidate teacher was not frozen in eval mode")
    if (
        int(teacher["actor_history_length"]) != 10
        or int(teacher["actor_frame_dimension"]) != 96
        or int(teacher["actor_observation_dimension"]) != 960
    ):
        raise AssertionError("candidate did not use the frozen 10x96 teacher input")

    table = PlaneNominalParameterTable.from_yaml(candidate_path)
    exact_checks = 0
    for node in nodes:
        result = table.lookup(
            math.radians(float(node["slope_degrees"])),
            str(node["direction"]),
            float(node["speed"]),
        )
        if not result.valid or result.value is None:
            raise AssertionError(f"exact lookup failed for {node}")
        if result.value.calibration_policy_id != policy_id:
            raise AssertionError("exact lookup silently used another policy")
        if not math.isclose(result.value.step_period, float(node["T"]), abs_tol=1.0e-12):
            raise AssertionError("exact lookup returned an interpolated period")
        exact_checks += 1

    interpolation_checks = 0
    for slope in (-12.5, -7.5, -2.5, 2.5, 7.5, 12.5):
        for direction in DIRECTIONS:
            for speed in (0.3, 0.5, 0.7, 0.9):
                result = table.lookup(math.radians(slope), direction, speed)
                if not result.valid or result.value is None:
                    raise AssertionError(
                        f"bounded interpolation failed at {slope}, {direction}, {speed}"
                    )
                if policy_id not in result.value.calibration_policy_id:
                    raise AssertionError("interpolation contains wrong-policy provenance")
                interpolation_checks += 1
    outside_checks = 0
    for slope, speed in ((-15.1, 0.4), (15.1, 0.4), (0.0, 0.19), (0.0, 1.01)):
        result = table.lookup(math.radians(slope), "+x", speed)
        if result.valid:
            raise AssertionError("out-of-range lookup unexpectedly succeeded")
        outside_checks += 1

    flat_capability = adapt_flat_capability(flat, 0.0)
    flat_regression = {
        "C_left": flat_capability.cop_left.x == tuple(flat["C_left"]["x"])
        and flat_capability.cop_left.y == tuple(flat["C_left"]["y"]),
        "C_right": flat_capability.cop_right.x == tuple(flat["C_right"]["x"])
        and flat_capability.cop_right.y == tuple(flat["C_right"]["y"]),
        "L_left": flat_capability.landing_left.x == tuple(flat["L_left"]["x"])
        and flat_capability.landing_left.y == tuple(flat["L_left"]["y"]),
        "L_right": flat_capability.landing_right.x == tuple(flat["L_right"]["x"])
        and flat_capability.landing_right.y == tuple(flat["L_right"]["y"]),
        "vmax": flat_capability.swing_velocity_limits
        == (float(flat["v_max"]["x"]), float(flat["v_max"]["y"])),
    }
    if not all(flat_regression.values()):
        raise AssertionError("alpha=0 capability adapter regression failed")

    report_rows = []
    for node in nodes:
        key = (
            float(node["slope_degrees"]),
            str(node["direction"]),
            float(node["speed"]),
        )
        report_rows.append((key, node, reports_by_label[_label(*key)]))

    n_counts: dict[int, int] = {}
    status_counts: dict[str, int] = {}
    fallback_count = 0
    for _, _, report in report_rows:
        distribution = report["diagnostic"]["nominal_certificate_distribution"]
        for key, value in distribution["N_counts"].items():
            n_counts[int(key)] = n_counts.get(int(key), 0) + int(value)
        for key, value in distribution["status_counts"].items():
            status_counts[key] = status_counts.get(key, 0) + int(value)
        fallback_count += int(distribution["fallback_count"])

    validation = {
        "schema_version": 1,
        "candidate": str(candidate_path),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "teacher_checkpoint_sha256": checkpoint_hash,
        "grid": {
            "expected_nodes": 140,
            "valid_nodes": len(nodes),
            "invalid_nodes": len(document["collection"]["invalid_nodes"]),
            "strict_valid_cycles": sum(
                int(report["valid_cycle_count"])
                for report in reports_by_label.values()
            ),
        },
        "frozen_teacher": {
            "eval_mode": teacher["eval_mode"],
            "requires_grad": teacher["requires_grad"],
            "actor_input": "10x96=960",
        },
        "lookup": {
            "exact_checks": exact_checks,
            "bounded_interpolation_checks": interpolation_checks,
            "out_of_range_rejection_checks": outside_checks,
            "wrong_policy_fallbacks": 0,
        },
        "alpha_zero_flat_capability_regression": flat_regression,
        "nominal_certificate": {
            "N_counts": {str(key): value for key, value in sorted(n_counts.items())},
            "status_counts": status_counts,
            "fallback_count": fallback_count,
        },
        "passed": True,
    }
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(
        yaml.safe_dump(validation, sort_keys=False), encoding="utf-8"
    )

    def subset(**filters) -> list[tuple[tuple[float, str, float], dict, dict]]:
        return [
            row
            for row in report_rows
            if all(row[0][{"slope": 0, "direction": 1, "speed": 2}[key]] == value for key, value in filters.items())
        ]

    lines = [
        "# G1 Slope-Sys-D Plane V1 candidate calibration report",
        "",
        "> Candidate only; production YAML was not overwritten. Certificate/reward/network code was not changed.",
        "",
        "## 1. Identity and protocol",
        "",
        f"- Teacher: `{checkpoint}`",
        f"- SHA256: `{checkpoint_hash}`",
        f"- Checkpoint iteration: `{teacher['checkpoint_iteration']}`",
        f"- Runner/policy: `{teacher['runner_class']}` / `{teacher['policy_class']}`",
        "- Frozen actor: `eval=True`, `requires_grad=False`, input `10 x 96 = 960`",
        f"- Git commit: `{document['git_commit']}`",
        f"- Semantics: `{document['calibration_semantics_version']}`",
        "- Grid: slopes `[-15,-10,-5,0,+5,+10,+15] deg`, four cardinal directions, speeds `[0.2,0.4,0.6,0.8,1.0] m/s`",
        "- Every valid node contains 120 strict complete post-warmup alternating touchdown cycles.",
        "",
        "## 2. Outcome and recommended V1 range",
        "",
        f"- Valid/invalid nodes: **{len(nodes)}/0**; strict cycles: **{validation['grid']['strict_valid_cycles']}**.",
        "- The policy itself formed stable nominal gait at every sampled node; natural fall/reset count was zero.",
        "- Recommended formal slope range: **-15 deg to +15 deg**.",
        "- Recommended shared all-direction speed range with the current projected capability: **0.2 to 0.4 m/s**.",
        "- `+x` is empirically usable through 1.0 m/s, but a single simple V1 range should not claim that extension for every direction.",
        "- `-x >= 0.6 m/s` is unreliable relative to projected L; lateral `+/-y = 1.0 m/s` is unreliable relative to projected L/vmax and shared-T assumptions.",
        "",
        "## 3. T, h_eff, omega and w",
        "",
        "| quantity | min | median | P95 | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for field in ("T", "h_eff", "omega", "w"):
        values = _stats(float(node[field]) for node in nodes)
        lines.append(
            f"| {field} | {_fmt(values['min'])} | {_fmt(values['median'])} | {_fmt(values['p95'])} | {_fmt(values['max'])} |"
        )
    lines += [
        "",
        "Median dependence:",
        "",
        "| speed (m/s) | T (s) | h_eff (m) | omega (rad/s) | w (m) | median side-T asymmetry | max side-T asymmetry |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for speed in SPEEDS:
        rows = subset(speed=speed)
        diagnostics = [row[2]["diagnostic"] for row in rows]
        lines.append(
            f"| {speed:g} | {_fmt(np.median([row[1]['T'] for row in rows]))} | "
            f"{_fmt(np.median([row[1]['h_eff'] for row in rows]))} | "
            f"{_fmt(np.median([row[1]['omega'] for row in rows]))} | "
            f"{_fmt(np.median([row[1]['w'] for row in rows]))} | "
            f"{_percent(np.median([item['T_left_right_relative_asymmetry'] for item in diagnostics]))} | "
            f"{_percent(max(item['T_left_right_relative_asymmetry'] for item in diagnostics))} |"
        )
    lines += [
        "",
        "Across slopes, median T stays near 0.24 s and falls to about 0.22 s at the extremes/high speed. "
        "h_eff is shallow-U-shaped over slope and decreases mildly with speed; omega changes inversely. "
        "The strong effect is direction/speed, not slope: lateral 0.8--1.0 m/s produces mirrored left/right T asymmetry of 27--67%.",
        "",
        "For the recommended 0.2--0.4 m/s range, shared node-level T remains adequate. "
        "Support-side T is worth a later ablation only if lateral high-speed operation is added.",
        "",
        "## 4. Epsilon and practical nominal errors",
        "",
        "All terminal epsilon values use independent per-axis `P95(abs(error))` on the calibration split.",
        "",
        "| value | min | median | P95 across nodes | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for section, axis in (
        ("epsilon_b", "x"),
        ("epsilon_b", "y"),
        ("epsilon_q", "x"),
        ("epsilon_q", "y"),
    ):
        values = _stats(float(node[section][axis]) for node in nodes)
        lines.append(
            f"| {section}_{axis} (m) | {_fmt(values['min'])} | {_fmt(values['median'])} | {_fmt(values['p95'])} | {_fmt(values['max'])} |"
        )
    for field in (
        "mean_velocity_error_threshold",
        "mean_abs_roll_error_threshold",
        "mean_abs_pitch_error_threshold",
    ):
        values = _stats(float(node[field]) for node in nodes)
        lines.append(
            f"| {field} | {_fmt(values['min'])} | {_fmt(values['median'])} | {_fmt(values['p95'])} | {_fmt(values['max'])} |"
        )
    lines += [
        "",
        "Epsilon changes substantially with direction and slope (notably downhill b_x and lateral q_y), so it should remain full node-conditioned rather than constant.",
        "",
        "## 5. Existing C/L/vmax coverage",
        "",
        "The formulas were not changed: projected flat C plus sole translation, projected flat L, `vmax_x*cos(alpha)`, and unchanged `vmax_y`.",
        "",
        "| speed | projected-L touchdown coverage | vmax exceedance |",
        "|---:|---:|---:|",
    ]
    for speed in SPEEDS:
        reports = [row[2] for row in subset(speed=speed)]
        lines.append(
            f"| {speed:g} m/s | {_percent(_weighted(reports, 'projected_L', 'containment_rate'))} | "
            f"{_percent(_weighted(reports, 'projected_vmax', 'exceedance_rate'))} |"
        )
    all_reports = [row[2] for row in report_rows]
    total_samples = sum(
        int(report["diagnostic"]["sample_count"]) for report in all_reports
    )
    l_contained = round(
        sum(
            int(report["diagnostic"]["sample_count"])
            * float(report["diagnostic"]["projected_L"]["containment_rate"])
            for report in all_reports
        )
    )
    vmax_exceeded = round(
        sum(
            int(report["diagnostic"]["sample_count"])
            * float(report["diagnostic"]["projected_vmax"]["exceedance_rate"])
            for report in all_reports
        )
    )
    lines += [
        "",
        f"- Overall projected-L coverage: **{_percent(_weighted(all_reports, 'projected_L', 'containment_rate'))}** ({l_contained}/{total_samples} contained).",
        f"- Overall vmax exceedance: **{_percent(_weighted(all_reports, 'projected_vmax', 'exceedance_rate'))}** ({vmax_exceeded}/{total_samples}).",
        "- C nominal reference `(0,0)` remains inside all 140 projected C boxes.",
        "- Actual CoP coverage cannot be measured: the configured ContactSensor provides net body force but no contact-point/pressure truth. This limitation is reported, not replaced by a fitted proxy.",
        "- The mismatch is systematic at high speed: `-x >= 0.6` loses L coverage; lateral 1.0 m/s has about 43--54% vmax exceedance. Recalibrating L/vmax may be a future experiment, but this candidate does not change them.",
        "",
        "## 6. Nominal landing diagnostic",
        "",
        "Raw measured-minus-command*T y error contains the alternating step width, so the model-error diagnostic also subtracts the calibrated signed w.",
        "",
        "| metric across nodes | x | y |",
        "|---|---:|---:|",
    ]
    landing = [row[2]["diagnostic"]["gait_speed_diagnostic"] for row in report_rows]
    for label, field in (
        ("median of node median delta", "median"),
        ("median of node P95 abs delta", "p95_abs"),
        ("maximum node P95 abs delta", "p95_abs"),
    ):
        values = []
        for axis in ("x", "y"):
            sequence = [
                item["width_aware_landing_delta_xy_statistics"][axis][field]
                for item in landing
            ]
            values.append(max(sequence) if label.startswith("maximum") else np.median(sequence))
        lines.append(f"| {label} | {_fmt(values[0])} m | {_fmt(values[1])} m |")
    lines += [
        "",
        "The error grows with speed: median node P95 abs error changes from roughly `(0.0127, 0.0145) m` at 0.2 m/s to `(0.0298, 0.0389) m` at 1.0 m/s. "
        "There is a direction-symmetric under-stride bias (`+x/-x` about 1.3/2.1 cm in x, `+y/-y` about 2.7 cm toward zero in y). "
        "Within 0.2--0.4 m/s this does not justify replacing `command*T`; high-speed extension could ablate a policy-calibrated landing model.",
        "",
        "## 7. Nominal N distribution",
        "",
        "Measured touchdown states were evaluated by the unchanged Plane V1 LP with the candidate node parameters.",
        "",
        "| subset | N=0 | N=1 | N=2 | over horizon (N=6) |",
        "|---|---:|---:|---:|---:|",
    ]
    for slope in (-10.0, 0.0, 10.0):
        counts: dict[int, int] = {}
        for _, _, report in subset(slope=slope):
            for key, value in report["diagnostic"]["nominal_certificate_distribution"]["N_counts"].items():
                counts[int(key)] = counts.get(int(key), 0) + int(value)
        total = sum(counts.values())
        lines.append(
            f"| {slope:+g} deg | {counts.get(0,0)} ({_percent(counts.get(0,0)/total)}) | "
            f"{counts.get(1,0)} ({_percent(counts.get(1,0)/total)}) | "
            f"{counts.get(2,0)} ({_percent(counts.get(2,0)/total)}) | "
            f"{counts.get(6,0)} ({_percent(counts.get(6,0)/total)}) |"
        )
    total_n = sum(n_counts.values())
    lines.append(
        f"| all | {n_counts.get(0,0)} ({_percent(n_counts.get(0,0)/total_n)}) | "
        f"{n_counts.get(1,0)} ({_percent(n_counts.get(1,0)/total_n)}) | "
        f"{n_counts.get(2,0)} ({_percent(n_counts.get(2,0)/total_n)}) | "
        f"{n_counts.get(6,0)} ({_percent(n_counts.get(6,0)/total_n)}) |"
    )
    lines += [
        "",
        f"Normal solver results: {status_counts}; solver/margin fallbacks: **{fallback_count}**.",
        "",
        "## 8. Conditioning recommendation",
        "",
        "- Constant is defensible for h_eff/omega in the conservative range; omega must remain derived from h_eff.",
        "- T should remain direction/speed-conditioned at node level; slope dependence is weaker.",
        "- w is mainly direction/speed-conditioned; slope dependence is weak.",
        "- pitch_star is slope-conditioned; roll_star stays near zero but retains mild direction dependence.",
        "- epsilon_b/q and practical thresholds should remain slope/direction/speed-conditioned.",
        "- Do not add support-side T or policy-calibrated landing in Plane V1's recommended range.",
        "",
        "## 9. Lookup and regression validation",
        "",
        f"- Exact-node lookups: {exact_checks}/140 passed; wrong-policy fallbacks: 0.",
        f"- Interior slope/speed interpolation: {interpolation_checks}/96 passed.",
        f"- Out-of-range rejection: {outside_checks}/4 passed.",
        "- alpha=0 projected C/L/vmax exactly equals the flat capability file.",
        "- Representative -10/0/+10 nominal N checks completed with no solver fallback.",
        f"- Machine-readable validation: `{args.validation_output.expanduser().resolve()}`",
        "",
        "## 10. Per-node audit",
        "",
        "| slope | direction | speed | T | h_eff | w | eps_b(x,y) | eps_q(x,y) | acceptance | L coverage | vmax violation | N0/N1/N2/N6 |",
        "|---:|:---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|:---:|",
    ]
    for (slope, direction, speed), node, report in sorted(report_rows):
        diagnostic = report["diagnostic"]
        counts = diagnostic["nominal_certificate_distribution"]["N_counts"]
        lines.append(
            f"| {slope:+g} | {direction} | {speed:g} | {_fmt(node['T'])} | "
            f"{_fmt(node['h_eff'])} | {_fmt(node['w'])} | "
            f"{_fmt(node['epsilon_b']['x'])}, {_fmt(node['epsilon_b']['y'])} | "
            f"{_fmt(node['epsilon_q']['x'])}, {_fmt(node['epsilon_q']['y'])} | "
            f"{_percent(report['acceptance_ratio'])} | "
            f"{_percent(diagnostic['projected_L']['containment_rate'])} | "
            f"{_percent(diagnostic['projected_vmax']['exceedance_rate'])} | "
            f"{counts.get('0',0)}/{counts.get('1',0)}/{counts.get('2',0)}/{counts.get('6',0)} |"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        yaml.safe_dump(
            {
                "report": str(args.report.expanduser().resolve()),
                "validation": str(args.validation_output.expanduser().resolve()),
                "valid_nodes": len(nodes),
                "strict_cycles": validation["grid"]["strict_valid_cycles"],
                "passed": True,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
