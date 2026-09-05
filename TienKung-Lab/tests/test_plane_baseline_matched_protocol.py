"""Gates A--D for the baseline-matched Plane V1 protocol."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from legged_lab.recovery.baseline_matched_protocol import (
    CARDINAL_MIN_SPEED,
    configure_matched_command_and_reset,
    lookup_matched_slope,
    matched_command_standing_mask,
    sample_baseline_matched_cardinal_commands,
    terrain_curriculum_decisions,
)
from legged_lab.recovery.plane_nominal_params import command_direction
from legged_lab.recovery.stage2_reward import plane_v1_touchdown_reward
from legged_lab.recovery.plane_terrain_math import (
    MATCHED_COLS,
    MATCHED_ROWS,
    column_types,
    slope_coefficient,
    slope_table,
    upward_normal,
)


@pytest.mark.parametrize("difficulty", (0.0, 0.25, 0.5, 0.75, 1.0))
def test_difficulty_to_slope_matches_isaac_pyramid_formula(difficulty: float) -> None:
    expected = difficulty * math.tan(math.radians(15.0))
    assert slope_coefficient(difficulty) == pytest.approx(expected)
    assert slope_coefficient(difficulty, inverted=True) == pytest.approx(-expected)


def test_all_200_mesh_normals_match_row_column_slope_metadata() -> None:
    table = slope_table(42)
    assert table.shape == (MATCHED_ROWS, MATCHED_COLS)
    assert column_types().count("flat") == 8
    assert column_types().count("uphill") == 6
    assert column_types().count("downhill") == 6
    for alpha in table.ravel():
        expected = torch.tensor(
            (-math.sin(alpha), 0.0, math.cos(alpha)), dtype=torch.float64
        )
        assert torch.max(torch.abs(torch.from_numpy(upward_normal(alpha)) - expected)) < 1.0e-12

    levels = torch.arange(MATCHED_ROWS).repeat_interleave(MATCHED_COLS)
    types = torch.arange(MATCHED_COLS).repeat(MATCHED_ROWS)
    selected, valid = lookup_matched_slope(torch.from_numpy(table), levels, types)
    assert torch.all(valid)
    assert torch.equal(selected.reshape(MATCHED_ROWS, MATCHED_COLS), torch.from_numpy(table))


def test_curriculum_uses_four_metre_reference_and_baseline_down_rule() -> None:
    distance = torch.tensor((4.01, 4.0, 0.5, 5.0))
    speed = torch.tensor((0.1, 0.1, 1.0, 1.0))
    up, down = terrain_curriculum_decisions(distance, speed, 20.0)
    assert up.tolist() == [True, False, False, True]
    assert down.tolist() == [False, False, True, False]


def test_command_distribution_gate_with_100000_samples() -> None:
    generator = torch.Generator().manual_seed(42)
    command, standing, directions = sample_baseline_matched_cardinal_commands(
        100_000, generator=generator
    )
    assert float(standing.float().mean()) == pytest.approx(0.2, abs=0.005)
    moving = ~standing
    for direction in range(4):
        fraction = float((directions[moving] == direction).float().mean())
        assert fraction == pytest.approx(0.25, abs=0.008)
    assert not torch.any((command[:, 0] != 0.0) & (command[:, 1] != 0.0))
    assert torch.all(command[:, 2] == 0.0)
    assert float(command[:, 0].max()) <= 1.0
    assert float(command[:, 0].min()) >= -0.6
    assert float(command[:, 1].abs().max()) <= 0.5
    moving_speed = torch.linalg.vector_norm(command[moving, :2], dim=1)
    assert float(moving_speed.min()) >= CARDINAL_MIN_SPEED


def test_shared_command_and_reset_configuration_contract() -> None:
    cfg = SimpleNamespace(
        commands=SimpleNamespace(
            resampling_time_range=(0.0, 0.0),
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            heading_command=True,
            ranges=SimpleNamespace(),
        ),
        domain_rand=SimpleNamespace(
            events=SimpleNamespace(
                reset_base=SimpleNamespace(
                    params={
                        "pose_range": {"yaw": (-3.14, 3.14)},
                        "velocity_range": {"yaw": (-0.5, 0.5)},
                    }
                )
            )
        ),
    )
    configure_matched_command_and_reset(cfg)
    assert cfg.commands.resampling_time_range == (10.0, 10.0)
    assert cfg.commands.rel_standing_envs == 0.2
    assert not cfg.commands.heading_command
    assert cfg.commands.rel_heading_envs == 0.0
    assert cfg.commands.ranges.lin_vel_x == (-0.6, 1.0)
    assert cfg.commands.ranges.lin_vel_y == (-0.5, 0.5)
    assert cfg.commands.ranges.ang_vel_z == (0.0, 0.0)
    assert cfg.commands.ranges.heading is None
    reset = cfg.domain_rand.events.reset_base.params
    assert reset["pose_range"]["yaw"] == (0.0, 0.0)
    assert reset["velocity_range"]["yaw"] == (0.0, 0.0)


def test_standing_is_explicit_certificate_not_applicable() -> None:
    command = torch.tensor(
        ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, -0.2, 0.0))
    )
    assert matched_command_standing_mask(command).tolist() == [True, False, False]
    result = plane_v1_touchdown_reward(
        None,
        6,
        -3.0,
        touchdown_index=5,
        terrain_plane_valid=False,
        solver_valid=False,
        enabled=True,
        intentional_not_applicable=True,
    )
    assert result.total == 0.0
    assert not result.update_previous_phi


def test_standing_nominal_lookup_has_explicit_direction() -> None:
    assert command_direction(0.0, 0.0, 0.0) == ("standing", 0.0, "")


def test_push_sampling_matches_baseline_uniform_distribution() -> None:
    generator = torch.Generator().manual_seed(7)
    sample = -1.0 + 2.0 * torch.rand((10_000, 2), generator=generator)
    assert float(sample.min()) >= -1.0 and float(sample.max()) <= 1.0
    assert torch.all(sample.mean(dim=0).abs() < 0.025)
    expected_std = 1.0 / math.sqrt(3.0)
    assert torch.all((sample.std(dim=0) - expected_std).abs() < 0.015)
