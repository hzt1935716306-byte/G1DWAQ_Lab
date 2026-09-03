"""Pure geometry regression tests for the plane capability adapter."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from legged_lab.recovery.plane_adapter import (
    adapt_flat_capability,
    cop_translation,
    projection_matrix,
    signed_slope_from_heading_normal,
    vertical_height_above_plane,
)


@pytest.fixture(scope="module")
def flat_parameters():
    path = Path("tools/recovery/generated/g1_recovery_params.yaml")
    return yaml.safe_load(path.read_text())


def test_flat_projection_is_exact_identity(flat_parameters) -> None:
    capability = adapt_flat_capability(flat_parameters, 0.0)
    np.testing.assert_array_equal(projection_matrix(0.0), np.eye(2))
    np.testing.assert_array_equal(cop_translation(0.0), np.zeros(2))
    assert capability.cop_left.x == tuple(flat_parameters["C_left"]["x"])
    assert capability.cop_left.y == tuple(flat_parameters["C_left"]["y"])
    assert capability.landing_left.x == tuple(flat_parameters["L_left"]["x"])
    assert capability.landing_right.y == tuple(flat_parameters["L_right"]["y"])
    assert capability.swing_velocity_limits == (
        flat_parameters["v_max"]["x"],
        flat_parameters["v_max"]["y"],
    )


def test_cop_translation_has_signed_slope_direction(flat_parameters) -> None:
    positive = adapt_flat_capability(flat_parameters, math.radians(10.0))
    negative = adapt_flat_capability(flat_parameters, math.radians(-10.0))
    assert positive.cop_left.x[0] > flat_parameters["C_left"]["x"][0] * math.cos(math.radians(10.0))
    assert negative.cop_left.x[0] < flat_parameters["C_left"]["x"][0] * math.cos(math.radians(10.0))


def test_l_and_vmax_are_even_in_signed_slope(flat_parameters) -> None:
    positive = adapt_flat_capability(flat_parameters, math.radians(12.0))
    negative = adapt_flat_capability(flat_parameters, math.radians(-12.0))
    assert positive.landing_left == negative.landing_left
    assert positive.landing_right == negative.landing_right
    assert positive.swing_velocity_limits == negative.swing_velocity_limits


def _normal(slope_degrees: float, direction_offset_degrees: float = 0.0):
    slope = math.radians(slope_degrees)
    offset = math.radians(direction_offset_degrees)
    return (
        -math.sin(slope) * math.cos(offset),
        -math.sin(slope) * math.sin(offset),
        math.cos(slope),
    )


@pytest.mark.parametrize(
    ("slope_degrees", "offset_degrees", "valid"),
    (
        (10.0, 0.0, True),
        (10.0, 3.0, True),
        (10.0, 10.0, False),
        (1.0, 30.0, False),
        (0.1, 80.0, True),
    ),
)
def test_slope_alignment_uses_angular_direction_error(
    slope_degrees: float,
    offset_degrees: float,
    valid: bool,
) -> None:
    result = signed_slope_from_heading_normal(
        _normal(slope_degrees, offset_degrees)
    )
    assert result.valid is valid


def test_signed_slope_keeps_normal_based_sign_definition() -> None:
    uphill = signed_slope_from_heading_normal(_normal(10.0))
    downhill = signed_slope_from_heading_normal(_normal(-10.0))
    assert uphill.valid and downhill.valid
    assert uphill.alpha == pytest.approx(math.radians(10.0))
    assert downhill.alpha == pytest.approx(math.radians(-10.0))


def test_vertical_height_is_not_normal_distance() -> None:
    alpha = math.radians(10.0)
    normal = np.asarray((-math.sin(alpha), 0.0, math.cos(alpha)))
    point = np.asarray((2.0, 0.0, math.tan(alpha) * 2.0 + 0.7))
    assert vertical_height_above_plane(normal, (0.0, 0.0, 0.0), point) == pytest.approx(0.7)
