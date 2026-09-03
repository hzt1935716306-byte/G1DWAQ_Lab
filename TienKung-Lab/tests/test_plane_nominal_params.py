"""Lookup and periodic-state tests for plane nominal gait parameters."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest
import yaml

from legged_lab.recovery.plane_certificate_runtime import plane_periodic_state
from legged_lab.recovery.plane_nominal_params import (
    PRACTICAL_METRIC_INTERVAL_MEAN_V1,
    PRACTICAL_METRIC_LEGACY_V0,
    PlaneNominalParameterTable,
    command_direction,
    nominal_node_key,
    upsert_nominal_nodes,
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


def test_collector_upsert_preserves_uncollected_nodes_and_deduplicates() -> None:
    negative = {
        "slope_degrees": -10.0,
        "direction": "+x",
        "speed": 0.4,
        "marker": "preserved",
    }
    old_positive = {
        "slope_degrees": 10.0,
        "direction": "+x",
        "speed": 0.4,
        "marker": "old",
    }
    new_positive = dict(old_positive, marker="new")
    merged = upsert_nominal_nodes(
        [negative, old_positive],
        [new_positive, new_positive],
    )

    assert [nominal_node_key(node) for node in merged] == [
        (-10.0, "+x", 0.4),
        (10.0, "+x", 0.4),
    ]
    assert merged[0]["marker"] == "preserved"
    assert merged[1]["marker"] == "new"


def test_interpolation_does_not_cross_known_failed_slope(table) -> None:
    base = next(
        node
        for node in table.nodes
        if node.direction == "+x" and node.alpha == 0.0 and node.speed == 0.2
    )
    alpha_15 = math.radians(-15.0)
    alpha_10 = math.radians(-10.0)
    alpha_5 = math.radians(-5.0)
    outer_nodes = (
        replace(base, alpha=alpha_15),
        replace(base, alpha=alpha_5),
    )
    hole = PlaneNominalParameterTable(
        outer_nodes,
        known_missing_slope_directions=((alpha_10, "+x"),),
    )

    for slope in (-12.0, -7.0):
        result = hole.lookup(math.radians(slope), "+x", base.speed)
        assert not result.valid
        assert result.reason == "interpolation would cross an explicitly failed calibration slope"

    filled = PlaneNominalParameterTable(
        (*outer_nodes, replace(base, alpha=alpha_10)),
        known_missing_slope_directions=((alpha_10, "+x"),),
    )
    assert filled.lookup(math.radians(-12.0), "+x", base.speed).valid
    assert filled.lookup(math.radians(-7.0), "+x", base.speed).valid


def test_practical_metric_semantics_are_explicit_and_never_mixed(table) -> None:
    path = Path("tools/recovery/generated/g1_plane_nominal_params.yaml")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    yaml_nodes = document["nominal_plane_gait"]["nodes"]
    assert yaml_nodes
    assert all(
        node["practical_metric_version"] == PRACTICAL_METRIC_LEGACY_V0
        for node in yaml_nodes
    )
    assert all(
        node.practical_metric_version == PRACTICAL_METRIC_LEGACY_V0
        for node in table.nodes
    )

    base = next(
        node
        for node in table.nodes
        if node.direction == "+x" and node.alpha == 0.0 and node.speed == 0.2
    )
    cycle_mean = replace(
        base,
        alpha=math.radians(5.0),
        practical_metric_version=PRACTICAL_METRIC_INTERVAL_MEAN_V1,
    )
    mixed = PlaneNominalParameterTable((base, cycle_mean))
    result = mixed.lookup(math.radians(2.5), "+x", base.speed)
    assert not result.valid
    assert result.reason == (
        "cannot interpolate nominal nodes with different practical metric semantics"
    )
