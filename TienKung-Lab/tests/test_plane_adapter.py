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


def test_signed_slope_and_cross_slope_rejection() -> None:
    alpha = math.radians(10.0)
    normal = (-math.sin(alpha), 0.0, math.cos(alpha))
    result = signed_slope_from_heading_normal(normal, 0.05)
    assert result.valid
    assert result.alpha == pytest.approx(alpha)

    invalid = signed_slope_from_heading_normal((normal[0], 0.1, normal[2]), 0.05)
    assert not invalid.valid
    assert "cross-slope" in invalid.reason


def test_vertical_height_is_not_normal_distance() -> None:
    alpha = math.radians(10.0)
    normal = np.asarray((-math.sin(alpha), 0.0, math.cos(alpha)))
    point = np.asarray((2.0, 0.0, math.tan(alpha) * 2.0 + 0.7))
    assert vertical_height_above_plane(normal, (0.0, 0.0, 0.0), point) == pytest.approx(0.7)
