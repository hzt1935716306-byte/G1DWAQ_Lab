"""Offline nominal-plane calibration and projection diagnostics.

This module estimates only policy-dependent nominal gait quantities.  It never
fits or changes C, L, or v_max; slope touchdown data only diagnoses the fixed
flat-capability projection assumption.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .certificate import (
    CertificateState,
    HalfspaceRegion2D,
    RecoverabilityConfig,
    certify_recoverability,
)
from .plane_adapter import Box2D, adapt_flat_capability, inverse_project_horizontal
from .plane_certificate_runtime import plane_periodic_state
from .plane_nominal_params import PRACTICAL_METRIC_INTERVAL_MEAN_V1
from .practical_metrics import practical_interval_means_from_sums


def mark_collection_done(
    env_ids: Sequence[int],
    collection_done: list[bool],
    previous_touchdown: list[object | None],
    *interval_buffers: list[list[float]],
) -> None:
    """Stop completed collector environments and release their interval frames."""

    for env_id in env_ids:
        collection_done[env_id] = True
        previous_touchdown[env_id] = None
        for buffer in interval_buffers:
            buffer[env_id] = []


def _statistics(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "std": float(np.std(array)),
    }


def _axis_error_statistics(vectors: Sequence[Sequence[float]]) -> dict[str, dict[str, float]]:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise ValueError("two-axis diagnostics require at least one finite 2-vector")
    return {
        axis: {
            "median": float(np.median(array[:, index])),
            "mean": float(np.mean(array[:, index])),
            "p95_abs": float(np.quantile(np.abs(array[:, index]), 0.95)),
            "maximum_abs": float(np.max(np.abs(array[:, index]))),
        }
        for index, axis in enumerate(("x", "y"))
    }


def _overflow(vector: np.ndarray, box: Box2D) -> np.ndarray:
    lower = np.asarray((box.x[0], box.y[0]))
    upper = np.asarray((box.x[1], box.y[1]))
    return np.maximum(lower - vector, 0.0) + np.maximum(vector - upper, 0.0)


def _nominal_certificate_distribution(
    samples: Sequence[Mapping[str, object]],
    capability,
    *,
    period: float,
    h_eff: float,
    omega: float,
    step_width: float,
    epsilon: np.ndarray,
) -> dict[str, object]:
    """Evaluate measured nominal touchdown states with the unchanged Plane V1 LP."""

    command = np.asarray(samples[0]["command"], dtype=np.float64)
    theory = plane_periodic_state(
        command[0], command[1], period, omega, step_width
    )
    config = RecoverabilityConfig(
        gravity=9.81,
        h_eff=h_eff,
        step_period=period,
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
        epsilon_b=(float(epsilon[0]), float(epsilon[1])),
        epsilon_q=(float(epsilon[2]), float(epsilon[3])),
    )
    counts = {value: 0 for value in range(config.max_steps + 2)}
    status_counts: dict[str, int] = {}
    fallback_count = 0
    for row in samples:
        support = str(row["support_side"])
        support_position = np.asarray(row["support_position_H"], dtype=np.float64)
        com = np.asarray(row["com_position_H"], dtype=np.float64)
        velocity = np.asarray(row["com_velocity_H"], dtype=np.float64)
        state = CertificateState(
            b=com + velocity / omega - support_position,
            q=np.asarray(row["q_H"], dtype=np.float64),
            support_side=support,
            phase=0.0,
            step_period=period,
            omega=omega,
        )
        result = certify_recoverability(state, config)
        n_value = config.max_steps + 1 if result.n_min is None else int(result.n_min)
        counts[n_value] = counts.get(n_value, 0) + 1
        status = result.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
        fallback_count += int(result.solver_fallback or result.margin_fallback)
    return {
        "sample_count": len(samples),
        "N_counts": {str(key): value for key, value in sorted(counts.items())},
        "N0_fraction": counts.get(0, 0) / len(samples),
        "N0_or_N1_fraction": (
            counts.get(0, 0) + counts.get(1, 0)
        )
        / len(samples),
        "status_counts": status_counts,
        "fallback_count": fallback_count,
    }


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
    terminal_epsilon_semantics: str = "joint_normalized_max_p95",
) -> tuple[dict[str, object], dict[str, object]]:
    """Calibrate one node and return its independent diagnostic report."""

    if len(samples) < 10:
        raise ValueError("at least ten valid touchdown samples are required")
    if not 0.5 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie in [0.5, 1)")
    if terminal_epsilon_semantics not in (
        "joint_normalized_max_p95",
        "per_axis_absolute_p95",
    ):
        raise ValueError("unsupported terminal_epsilon_semantics")
    split = max(1, min(len(samples) - 1, int(len(samples) * calibration_fraction)))
    calibration = samples[:split]
    holdout = samples[split:]

    period = float(np.median([float(row["T"]) for row in calibration]))
    h_eff = float(np.median([float(row["h_geom"]) for row in calibration]))
    calibration_roll_frames = np.concatenate(
        [np.asarray(row["interval_roll"], dtype=np.float64) for row in calibration]
    )
    calibration_pitch_frames = np.concatenate(
        [np.asarray(row["interval_pitch"], dtype=np.float64) for row in calibration]
    )
    if calibration_roll_frames.size == 0 or calibration_pitch_frames.size == 0:
        raise ValueError("complete touchdown intervals must contain policy-step frames")
    roll_star = float(np.median(calibration_roll_frames))
    pitch_star = float(np.median(calibration_pitch_frames))
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
    if terminal_epsilon_semantics == "per_axis_absolute_p95":
        epsilon = np.maximum(
            np.quantile(np.abs(calibration_errors), 0.95, axis=0), epsilon_floor
        )
    else:
        scales = np.median(np.abs(calibration_errors), axis=0) + epsilon_floor
        normalized_max = np.max(np.abs(calibration_errors) / scales, axis=1)
        kappa = float(np.quantile(normalized_max, 0.95))
        epsilon = np.maximum(kappa * scales, epsilon_floor)
    holdout_covered = np.all(np.abs(holdout_errors) <= epsilon, axis=1)

    interval_metrics = []
    for row in calibration:
        velocity_frames = np.asarray(row["interval_velocity_error"], dtype=np.float64)
        roll_frames = np.asarray(row["interval_roll"], dtype=np.float64)
        pitch_frames = np.asarray(row["interval_pitch"], dtype=np.float64)
        if not (
            velocity_frames.ndim == roll_frames.ndim == pitch_frames.ndim == 1
            and velocity_frames.size == roll_frames.size == pitch_frames.size
            and velocity_frames.size > 0
        ):
            raise ValueError("practical metrics require equally sized non-empty interval frames")
        attitude_error_sum = np.asarray(
            (
                np.abs(roll_frames - roll_star).sum(),
                np.abs(pitch_frames - pitch_star).sum(),
            ),
            dtype=np.float64,
        )
        mean_velocity_error, mean_attitude_error = practical_interval_means_from_sums(
            velocity_frames.sum(),
            attitude_error_sum,
            velocity_frames.size,
        )
        interval_metrics.append(
            (mean_velocity_error, mean_attitude_error[0], mean_attitude_error[1])
        )
    interval_metrics = np.asarray(interval_metrics, dtype=np.float64)
    velocity_errors = interval_metrics[:, 0]
    roll_errors = interval_metrics[:, 1]
    pitch_errors = interval_metrics[:, 2]

    node = {
        "slope_degrees": float(slope_degrees),
        "direction": direction,
        "speed": float(speed),
        "T": period,
        "h_eff": h_eff,
        "omega": omega,
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
        "practical_metric_version": PRACTICAL_METRIC_INTERVAL_MEAN_V1,
        "terminal_epsilon_semantics": terminal_epsilon_semantics,
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
    actual_velocity_frames = np.concatenate(
        [
            np.column_stack(
                (
                    np.asarray(row["interval_actual_vx"], dtype=np.float64),
                    np.asarray(
                        row.get(
                            "interval_actual_vy",
                            [float(row["com_velocity_H"][1])]
                            * len(row["interval_actual_vx"]),
                        ),
                        dtype=np.float64,
                    ),
                )
            )
            for row in samples
        ],
        axis=0,
    )
    measured_landing = np.asarray([row["l_H"] for row in samples], dtype=np.float64)
    command_landing = np.asarray(
        [
            np.asarray(row["command"], dtype=np.float64) * float(row["T"])
            for row in samples
        ],
        dtype=np.float64,
    )
    landing_delta = measured_landing - command_landing
    width_aware_landing = command_landing.copy()
    width_aware_landing[:, 1] += np.asarray(
        [
            step_width if row["transition_support"] == "right" else -step_width
            for row in samples
        ],
        dtype=np.float64,
    )
    width_aware_landing_delta = measured_landing - width_aware_landing
    side_periods = {
        side: np.asarray(
            [float(row["T"]) for row in samples if row["support_side"] == side],
            dtype=np.float64,
        )
        for side in ("left", "right")
    }
    median_t_left = float(np.median(side_periods["left"]))
    median_t_right = float(np.median(side_periods["right"]))
    t_side_difference = abs(median_t_left - median_t_right)
    projected_capability = adapt_flat_capability(flat_parameters, alpha)
    report = {
        "sample_count": len(samples),
        "calibration_count": len(calibration),
        "holdout_count": len(holdout),
        "T": _statistics([float(row["T"]) for row in samples]),
        "h_eff": _statistics([float(row["h_geom"]) for row in samples]),
        "omega": omega,
        "T_left": _statistics(side_periods["left"]),
        "T_right": _statistics(side_periods["right"]),
        "T_left_right_median_abs_difference_s": t_side_difference,
        "T_left_right_relative_asymmetry": t_side_difference / period,
        "w_left": w_left,
        "w_right": w_right,
        "w_abs_side_difference": abs(w_left - w_right),
        "holdout_joint_coverage": float(np.mean(holdout_covered)),
        "holdout_per_axis_coverage": {
            name: float(np.mean(np.abs(holdout_errors[:, index]) <= epsilon[index]))
            for index, name in enumerate(("b_x", "b_y", "q_x", "q_y"))
        },
        "terminal_epsilon_semantics": terminal_epsilon_semantics,
        "epsilon_b": {"x": float(epsilon[0]), "y": float(epsilon[1])},
        "epsilon_q": {"x": float(epsilon[2]), "y": float(epsilon[3])},
        "gait_speed_diagnostic": {
            "actual_mean_vx_m_per_s": float(
                np.mean(actual_velocity_frames[:, 0])
            ),
            "actual_mean_vy_m_per_s": float(np.mean(actual_velocity_frames[:, 1])),
            "actual_velocity_xy_statistics": {
                axis: _statistics(actual_velocity_frames[:, index])
                for index, axis in enumerate(("x", "y"))
            },
            "median_T_left_s": float(
                np.median(
                    [float(row["T"]) for row in samples if row["support_side"] == "left"]
                )
            ),
            "median_T_right_s": float(
                np.median(
                    [float(row["T"]) for row in samples if row["support_side"] == "right"]
                )
            ),
            "measured_median_landing_left_xy_m": np.median(
                np.asarray(
                    [row["l_H"] for row in samples if row["support_side"] == "left"],
                    dtype=np.float64,
                ),
                axis=0,
            ).tolist(),
            "measured_median_landing_right_xy_m": np.median(
                np.asarray(
                    [row["l_H"] for row in samples if row["support_side"] == "right"],
                    dtype=np.float64,
                ),
                axis=0,
            ).tolist(),
            "median_command_vx_times_T_left_m": float(
                np.median(
                    [
                        float(row["command"][0]) * float(row["T"])
                        for row in samples
                        if row["support_side"] == "left"
                    ]
                )
            ),
            "median_command_vx_times_T_right_m": float(
                np.median(
                    [
                        float(row["command"][0]) * float(row["T"])
                        for row in samples
                        if row["support_side"] == "right"
                    ]
                )
            ),
            "median_delta_l_x_left_m": float(
                np.median(
                    [
                        float(row["l_H"][0])
                        - float(row["command"][0]) * float(row["T"])
                        for row in samples
                        if row["support_side"] == "left"
                    ]
                )
            ),
            "median_delta_l_x_right_m": float(
                np.median(
                    [
                        float(row["l_H"][0])
                        - float(row["command"][0]) * float(row["T"])
                        for row in samples
                        if row["support_side"] == "right"
                    ]
                )
            ),
            "command_times_T_landing_left_xy_m": np.median(
                command_landing[
                    np.asarray(
                        [row["support_side"] == "left" for row in samples],
                        dtype=np.bool_,
                    )
                ],
                axis=0,
            ).tolist(),
            "command_times_T_landing_right_xy_m": np.median(
                command_landing[
                    np.asarray(
                        [row["support_side"] == "right" for row in samples],
                        dtype=np.bool_,
                    )
                ],
                axis=0,
            ).tolist(),
            "landing_delta_xy_statistics": _axis_error_statistics(landing_delta),
            "landing_delta_semantics": (
                "measured relative touchdown minus command_velocity*T; the raw y "
                "delta therefore retains the alternating step width"
            ),
            "width_aware_landing_delta_xy_statistics": _axis_error_statistics(
                width_aware_landing_delta
            ),
            "width_aware_landing_delta_semantics": (
                "measured relative touchdown minus (command_velocity*T plus the "
                "node-level alternating lateral step width)"
            ),
            "landing_delta_left_xy_statistics": _axis_error_statistics(
                landing_delta[
                    np.asarray(
                        [row["support_side"] == "left" for row in samples],
                        dtype=np.bool_,
                    )
                ]
            ),
            "landing_delta_right_xy_statistics": _axis_error_statistics(
                landing_delta[
                    np.asarray(
                        [row["support_side"] == "right" for row in samples],
                        dtype=np.bool_,
                    )
                ]
            ),
        },
        "practical_nominal_error_statistics": {
            "interval_mean_velocity_error": _statistics(velocity_errors),
            "interval_mean_abs_roll_error": _statistics(roll_errors),
            "interval_mean_abs_pitch_error": _statistics(pitch_errors),
        },
        "projected_C": {
            "nominal_reference_checked": True,
            "nominal_reference_contained": projected_capability.nominal_cop_valid,
            "actual_cop_samples_available": False,
            "actual_cop_violation_rate": None,
            "reason": (
                "The configured Isaac Lab ContactSensor exposes net body force but "
                "not contact-point/pressure truth; actual CoP cannot be inferred "
                "without changing the measurement setup."
            ),
        },
        "nominal_certificate_distribution": _nominal_certificate_distribution(
            samples,
            projected_capability,
            period=period,
            h_eff=h_eff,
            omega=omega,
            step_width=step_width,
            epsilon=epsilon,
        ),
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


__all__ = ["calibrate_nominal_node", "mark_collection_done"]
