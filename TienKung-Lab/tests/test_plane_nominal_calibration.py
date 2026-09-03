"""Synthetic checks for nominal-plane calibration and fixed-capability diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import yaml

from legged_lab.recovery.plane_certificate_runtime import plane_periodic_state
from legged_lab.recovery.plane_nominal_calibration import (
    calibrate_nominal_node,
    mark_collection_done,
)
from legged_lab.recovery.plane_nominal_params import PRACTICAL_METRIC_INTERVAL_MEAN_V1
from legged_lab.recovery.practical_metrics import practical_interval_means_from_sums


def _samples(count: int = 20):
    period = 0.25
    height = 0.70
    width = 0.22
    command = np.asarray((0.4, 0.0))
    omega = math.sqrt(9.81 / height)
    theory = plane_periodic_state(*command, period, omega, width)
    rows = []
    for index in range(count):
        support = "left" if index % 2 == 0 else "right"
        transition = "right" if support == "left" else "left"
        support_position = np.zeros(2)
        velocity = command.copy()
        b = np.asarray(theory[f"b_{support}"])
        q = np.asarray(theory[f"q_{support}"])
        com = b - velocity / omega
        rows.append(
            {
                "T": period,
                "h_geom": height,
                "command": command.tolist(),
                "support_side": support,
                "transition_support": transition,
                "com_position_H": com.tolist(),
                "com_velocity_H": velocity.tolist(),
                "support_position_H": support_position.tolist(),
                "q_H": q.tolist(),
                "q_start_H": np.asarray(theory[f"q_{transition}"]).tolist(),
                "l_H": (-q).tolist(),
                "interval_roll": [0.01, 0.01],
                "interval_pitch": [-0.02, -0.02],
                "interval_velocity_error": [0.03, 0.03],
            }
        )
    return rows


def test_calibration_uses_joint_holdout_and_only_diagnoses_capability() -> None:
    with open("tools/recovery/generated/g1_recovery_params.yaml", encoding="utf-8") as stream:
        flat = yaml.safe_load(stream)
    before = yaml.safe_dump(flat, sort_keys=True)
    node, report = calibrate_nominal_node(
        _samples(),
        flat,
        slope_degrees=-10.0,
        direction="+x",
        speed=0.4,
        calibration_policy_id="synthetic-policy",
    )

    assert node["T"] == pytest.approx(0.25)
    assert node["h_eff"] == pytest.approx(0.70)
    assert node["w"] == pytest.approx(0.22)
    assert node["roll_star"] == pytest.approx(0.01)
    assert node["pitch_star"] == pytest.approx(-0.02)
    assert all(value > 0.0 for value in node["epsilon_b"].values())
    assert all(value > 0.0 for value in node["epsilon_q"].values())
    assert report["calibration_count"] == 16
    assert report["holdout_count"] == 4
    assert report["holdout_joint_coverage"] == pytest.approx(1.0)
    assert "projected_L" in report and "projected_vmax" in report
    assert yaml.safe_dump(flat, sort_keys=True) == before


def test_practical_thresholds_use_complete_interval_means() -> None:
    with open("tools/recovery/generated/g1_recovery_params.yaml", encoding="utf-8") as stream:
        flat = yaml.safe_load(stream)
    rows = _samples()
    for row in rows:
        # These complete-cycle values deliberately differ from any plausible
        # instantaneous touchdown value.
        row["interval_velocity_error"] = [0.0, 0.2]
        row["interval_roll"] = [0.0, 0.2]
        row["interval_pitch"] = [-0.1, 0.1]
        row["velocity_error"] = 9.0
        row["roll"] = 9.0
        row["pitch"] = 9.0

    node, _ = calibrate_nominal_node(
        rows,
        flat,
        slope_degrees=-10.0,
        direction="+x",
        speed=0.4,
        calibration_policy_id="synthetic-policy",
    )
    runtime_velocity, runtime_attitude = practical_interval_means_from_sums(
        torch.tensor(0.2),
        torch.tensor((0.2, 0.2)),
        torch.tensor(2),
    )

    assert node["roll_star"] == pytest.approx(0.1)
    assert node["pitch_star"] == pytest.approx(0.0)
    assert node["mean_velocity_error_threshold"] == pytest.approx(0.1)
    assert node["mean_abs_roll_error_threshold"] == pytest.approx(0.1)
    assert node["mean_abs_pitch_error_threshold"] == pytest.approx(0.1)
    assert node["practical_metric_version"] == PRACTICAL_METRIC_INTERVAL_MEAN_V1
    assert runtime_velocity.item() == pytest.approx(node["mean_velocity_error_threshold"])
    assert runtime_attitude.tolist() == pytest.approx(
        [
            node["mean_abs_roll_error_threshold"],
            node["mean_abs_pitch_error_threshold"],
        ]
    )


def test_completed_node_stops_collection_and_clears_interval_buffers() -> None:
    collection_done = [False, False, False]
    previous_touchdown = [object(), object(), object()]
    velocity = [[1.0], [2.0], [3.0]]
    roll = [[4.0], [5.0], [6.0]]
    pitch = [[7.0], [8.0], [9.0]]

    mark_collection_done(
        (0, 2),
        collection_done,
        previous_touchdown,
        velocity,
        roll,
        pitch,
    )

    assert collection_done == [True, False, True]
    assert previous_touchdown[0] is None and previous_touchdown[2] is None
    assert velocity == [[], [2.0], []]
    assert roll == [[], [5.0], []]
    assert pitch == [[], [8.0], []]
