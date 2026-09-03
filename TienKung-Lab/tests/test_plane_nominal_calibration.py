"""Synthetic checks for nominal-plane calibration and fixed-capability diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from legged_lab.recovery.plane_certificate_runtime import plane_periodic_state
from legged_lab.recovery.plane_nominal_calibration import calibrate_nominal_node


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
                "roll": 0.01,
                "pitch": -0.02,
                "velocity_error": 0.03,
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
