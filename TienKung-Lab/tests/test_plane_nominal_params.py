"""Lookup and periodic-state tests for plane nominal gait parameters."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from legged_lab.recovery.plane_certificate_runtime import plane_periodic_state
from legged_lab.recovery.plane_nominal_params import (
    PlaneNominalParameterTable,
    command_direction,
)


@pytest.fixture(scope="module")
def table():
    return PlaneNominalParameterTable.from_yaml(
        Path("tools/recovery/generated/g1_plane_nominal_params.yaml")
    )


def test_heading_horizontal_nominal_lx_has_no_slope_projection() -> None:
    state = plane_periodic_state(0.6, 0.0, 0.25, 3.7, 0.22)
    assert state["landing_left"][0] == pytest.approx(0.15)
    assert state["landing_right"][0] == pytest.approx(0.15)


def test_motion_direction_never_changes_terrain_slope_sign() -> None:
    alpha = math.radians(10.0)
    direction, speed, _ = command_direction(-0.4, 0.0)
    assert direction == "-x" and speed == pytest.approx(0.4)
    assert alpha > 0.0
    assert command_direction(0.0, 0.4)[0] == "+y"
    assert command_direction(0.0, -0.4)[0] == "-y"
    assert alpha > 0.0


def test_missing_direction_and_slope_are_invalid_without_flat_fallback(table) -> None:
    anchor_only = PlaneNominalParameterTable(
        node for node in table.nodes if node.direction == "+x" and node.alpha == 0.0
    )
    assert not anchor_only.lookup_command(0.0, (-0.4, 0.0, 0.0)).valid
    assert not anchor_only.lookup_command(math.radians(-5.0), (0.4, 0.0, 0.0)).valid
    assert not anchor_only.lookup_command(0.0, (0.4, 0.2, 0.0)).valid


def test_flat_plus_x_interpolation_matches_legacy_linear_period(table) -> None:
    result = table.lookup_command(0.0, (0.6, 0.0, 0.0))
    assert result.valid and result.value is not None
    expected = 0.24399962425231939 + 0.02000069618225101 * 0.6
    assert result.value.step_period == pytest.approx(expected, abs=1.0e-15)
    assert result.value.h_eff == pytest.approx(0.6884990671277046)
    assert result.value.step_width == pytest.approx(0.22294799983501434)
