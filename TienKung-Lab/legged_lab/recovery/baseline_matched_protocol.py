"""Pure helpers for the baseline-matched Plane V1 training protocol."""

from __future__ import annotations

import torch


CURRICULUM_REFERENCE_TILE_LENGTH = 8.0
CARDINAL_MIN_SPEED = 0.2
CARDINAL_MAX_SPEEDS = (1.0, 0.6, 0.5, 0.5)


def configure_matched_command_and_reset(cfg) -> None:
    """Apply the shared cardinal command and heading-aligned reset contract."""

    cfg.commands.resampling_time_range = (10.0, 10.0)
    cfg.commands.rel_standing_envs = 0.2
    cfg.commands.rel_heading_envs = 0.0
    cfg.commands.heading_command = False
    cfg.commands.ranges.lin_vel_x = (-0.6, 1.0)
    cfg.commands.ranges.lin_vel_y = (-0.5, 0.5)
    cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.ranges.heading = None

    reset_base = cfg.domain_rand.events.reset_base
    reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    reset_base.params["velocity_range"]["yaw"] = (0.0, 0.0)


def terrain_curriculum_decisions(
    distance: torch.Tensor,
    command_speed: torch.Tensor,
    max_episode_length_s: float,
    reference_tile_length: float = CURRICULUM_REFERENCE_TILE_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the same move-up/down decisions as BaseEnv, with an 8 m reference."""

    move_up = distance > float(reference_tile_length) / 2.0
    move_down = distance < command_speed * float(max_episode_length_s) * 0.5
    move_down &= ~move_up
    return move_up, move_down


def lookup_matched_slope(
    slope_table: torch.Tensor,
    terrain_levels: torch.Tensor,
    terrain_types: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index signed alpha by both row and column with explicit validity."""

    if slope_table.ndim != 2 or terrain_levels.shape != terrain_types.shape:
        raise ValueError("slope table must be 2-D and terrain index shapes must match")
    valid = (terrain_levels >= 0) & (terrain_levels < slope_table.shape[0])
    valid &= (terrain_types >= 0) & (terrain_types < slope_table.shape[1])
    levels = terrain_levels.clamp(0, slope_table.shape[0] - 1)
    types = terrain_types.clamp(0, slope_table.shape[1] - 1)
    return slope_table[levels, types], valid


def sample_baseline_matched_cardinal_commands(
    count: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    standing_probability: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample standing/cardinal commands and return commands, standing, directions.

    Direction indices are ``0:+x, 1:-x, 2:+y, 3:-y``.  Standing rows retain
    their sampled direction only for distribution diagnostics.
    """

    if count < 0 or not 0.0 <= standing_probability <= 1.0:
        raise ValueError("invalid command sample count or standing probability")
    standing = torch.rand(count, device=device, generator=generator) < standing_probability
    directions = torch.randint(0, 4, (count,), device=device, generator=generator)
    maxima = torch.tensor(CARDINAL_MAX_SPEEDS, device=device)
    speed = CARDINAL_MIN_SPEED + torch.rand(
        count, device=device, generator=generator
    ) * (maxima[directions] - CARDINAL_MIN_SPEED)
    speed = torch.where(standing, torch.zeros_like(speed), speed)
    command = torch.zeros((count, 3), device=device)
    command[:, 0] = torch.where(
        directions == 0, speed, torch.where(directions == 1, -speed, 0.0)
    )
    command[:, 1] = torch.where(
        directions == 2, speed, torch.where(directions == 3, -speed, 0.0)
    )
    return command, standing, directions


def matched_command_standing_mask(
    command: torch.Tensor, tolerance: float = 1.0e-6
) -> torch.Tensor:
    """Return commands for which the moving certificate is intentionally N/A."""

    if command.ndim != 2 or command.shape[1] < 2:
        raise ValueError("command must have shape [N, >=2]")
    return torch.linalg.vector_norm(command[:, :2], dim=1) <= float(tolerance)


__all__ = [
    "CARDINAL_MIN_SPEED",
    "CARDINAL_MAX_SPEEDS",
    "CURRICULUM_REFERENCE_TILE_LENGTH",
    "configure_matched_command_and_reset",
    "sample_baseline_matched_cardinal_commands",
    "matched_command_standing_mask",
    "lookup_matched_slope",
    "terrain_curriculum_decisions",
]
