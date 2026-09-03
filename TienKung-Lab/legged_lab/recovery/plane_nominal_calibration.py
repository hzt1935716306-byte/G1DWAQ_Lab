"""Offline nominal-plane calibration and projection diagnostics.

This module estimates only policy-dependent nominal gait quantities.  It never
fits or changes C, L, or v_max; slope touchdown data only diagnoses the fixed
flat-capability projection assumption.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .plane_adapter import Box2D, inverse_project_horizontal
from .plane_certificate_runtime import plane_periodic_state


def _statistics(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "std": float(np.std(array)),
    }


def _overflow(vector: np.ndarray, box: Box2D) -> np.ndarray:
    lower = np.asarray((box.x[0], box.y[0]))
    upper = np.asarray((box.x[1], box.y[1]))
    return np.maximum(lower - vector, 0.0) + np.maximum(vector - upper, 0.0)


def calibrate_nominal_node(
    samples: Sequence[Mapping[str, object]],
    flat_parameters: Mapping[str, object],
    *,
    slope_degrees: float,
    direction: str,
    speed: float,
    calibration_policy_id: str,
    calibration_fraction: float = 0.8,
    epsilon_floor: float = 1.0e-6,
) -> tuple[dict[str, object], dict[str, object]]:
    """Calibrate one node and return its independent diagnostic report."""

    if len(samples) < 10:
        raise ValueError("at least ten valid touchdown samples are required")
    if not 0.5 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie in [0.5, 1)")
    split = max(1, min(len(samples) - 1, int(len(samples) * calibration_fraction)))
    calibration = samples[:split]
    holdout = samples[split:]

    period = float(np.median([float(row["T"]) for row in calibration]))
    h_eff = float(np.median([float(row["h_geom"]) for row in calibration]))
    roll_star = float(np.median([float(row["roll"]) for row in calibration]))
    pitch_star = float(np.median([float(row["pitch"]) for row in calibration]))
    if period <= 0.0 or h_eff < 0.20:
        raise ValueError("calibrated period must be positive and h_eff must be at least 0.20 m")

    left_width = [
        -(float(row["l_H"][1]) - float(row["command"][1]) * float(row["T"]))
        for row in calibration
        if row["transition_support"] == "left"
    ]
    right_width = [
        float(row["l_H"][1]) - float(row["command"][1]) * float(row["T"])
        for row in calibration
        if row["transition_support"] == "right"
    ]
    if not left_width or not right_width:
        raise ValueError("both left- and right-support transitions are required")
    w_left = float(np.median(left_width))
    w_right = float(np.median(right_width))
    step_width = 0.5 * (w_left + w_right)
    omega = math.sqrt(9.81 / h_eff)

    def errors(rows):
        values = []
        for row in rows:
            command = np.asarray(row["command"], dtype=np.float64)
            theory = plane_periodic_state(command[0], command[1], period, omega, step_width)
            support = str(row["support_side"])
            support_position = np.asarray(row["support_position_H"], dtype=np.float64)
            com = np.asarray(row["com_position_H"], dtype=np.float64)
            velocity = np.asarray(row["com_velocity_H"], dtype=np.float64)
            b_measured = com + velocity / omega - support_position
            q_measured = np.asarray(row["q_H"], dtype=np.float64)
            values.append(
                np.concatenate(
                    (
                        b_measured - np.asarray(theory[f"b_{support}"]),
                        q_measured - np.asarray(theory[f"q_{support}"]),
                    )
                )
            )
        return np.asarray(values)

    calibration_errors = errors(calibration)
    holdout_errors = errors(holdout)
    scales = np.median(np.abs(calibration_errors), axis=0) + epsilon_floor
    normalized_max = np.max(np.abs(calibration_errors) / scales, axis=1)
    kappa = float(np.quantile(normalized_max, 0.95))
    epsilon = np.maximum(kappa * scales, epsilon_floor)
    holdout_covered = np.all(np.abs(holdout_errors) <= epsilon, axis=1)

    velocity_errors = np.asarray([float(row["velocity_error"]) for row in calibration])
    roll_errors = np.abs(np.asarray([float(row["roll"]) for row in calibration]) - roll_star)
    pitch_errors = np.abs(np.asarray([float(row["pitch"]) for row in calibration]) - pitch_star)

    node = {
        "slope_degrees": float(slope_degrees),
        "direction": direction,
        "speed": float(speed),
        "T": period,
        "h_eff": h_eff,
        "w": step_width,
        "epsilon_b": {"x": float(epsilon[0]), "y": float(epsilon[1])},
        "epsilon_q": {"x": float(epsilon[2]), "y": float(epsilon[3])},
        "roll_star": roll_star,
        "pitch_star": pitch_star,
        "mean_velocity_error_threshold": max(epsilon_floor, float(np.quantile(velocity_errors, 0.95))),
        "mean_abs_roll_error_threshold": max(epsilon_floor, float(np.quantile(roll_errors, 0.95))),
        "mean_abs_pitch_error_threshold": max(epsilon_floor, float(np.quantile(pitch_errors, 0.95))),
        "sample_count": len(samples),
        "calibration_policy_id": calibration_policy_id,
    }

    alpha = math.radians(float(slope_degrees))
    flat_landing = {
        "left": Box2D(tuple(flat_parameters["L_left"]["x"]), tuple(flat_parameters["L_left"]["y"])),
        "right": Box2D(tuple(flat_parameters["L_right"]["x"]), tuple(flat_parameters["L_right"]["y"])),
    }
    flat_vmax = np.asarray(
        (flat_parameters["v_max"]["x"], flat_parameters["v_max"]["y"]),
        dtype=np.float64,
    )
    containment = []
    overflow_magnitudes = []
    vmax_ratios = []
    diagnostic_rows = []
    for row in samples:
        transition_support = str(row["transition_support"])
        l_s = inverse_project_horizontal(row["l_H"], alpha)
        overflow = _overflow(l_s, flat_landing[transition_support])
        contained = bool(np.all(overflow <= 0.0))
        d_s = inverse_project_horizontal(
            np.asarray(row["l_H"]) - np.asarray(row["q_start_H"]), alpha
        )
        required_velocity = np.abs(d_s) / float(row["T"])
        ratio = float(np.max(required_velocity / flat_vmax))
        containment.append(contained)
        overflow_magnitudes.append(float(np.linalg.norm(overflow)))
        vmax_ratios.append(ratio)
        diagnostic_rows.append(
            {
                "transition_support": transition_support,
                "contained_in_L_flat": contained,
                "L_overflow_m": float(np.linalg.norm(overflow)),
                "vmax_ratio": ratio,
            }
        )

    overflow_array = np.asarray(overflow_magnitudes)
    ratio_array = np.asarray(vmax_ratios)
    report = {
        "sample_count": len(samples),
        "calibration_count": len(calibration),
        "holdout_count": len(holdout),
        "T": _statistics([float(row["T"]) for row in samples]),
        "w_left": w_left,
        "w_right": w_right,
        "w_abs_side_difference": abs(w_left - w_right),
        "holdout_joint_coverage": float(np.mean(holdout_covered)),
        "projected_L": {
            "containment_rate": float(np.mean(containment)),
            "overflow_p90_m": float(np.quantile(overflow_array, 0.90)),
            "overflow_p95_m": float(np.quantile(overflow_array, 0.95)),
            "overflow_p99_m": float(np.quantile(overflow_array, 0.99)),
            "overflow_max_m": float(np.max(overflow_array)),
        },
        "projected_vmax": {
            "exceedance_rate": float(np.mean(ratio_array > 1.0)),
            "ratio_p95": float(np.quantile(ratio_array, 0.95)),
            "ratio_p99": float(np.quantile(ratio_array, 0.99)),
            "ratio_max": float(np.max(ratio_array)),
        },
        "by_transition_support": {
            side: {
                "count": sum(row["transition_support"] == side for row in diagnostic_rows),
                "L_containment_rate": float(
                    np.mean(
                        [row["contained_in_L_flat"] for row in diagnostic_rows if row["transition_support"] == side]
                    )
                ),
                "vmax_exceedance_rate": float(
                    np.mean(
                        [row["vmax_ratio"] > 1.0 for row in diagnostic_rows if row["transition_support"] == side]
                    )
                ),
            }
            for side in ("left", "right")
        },
    }
    return node, report


__all__ = ["calibrate_nominal_node"]
