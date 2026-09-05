"""Shared command generator for all seven baseline-matched tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand

from legged_lab.recovery.baseline_matched_protocol import (
    sample_baseline_matched_cardinal_commands,
)


class BaselineMatchedCardinalVelocityCommand(UniformVelocityCommand):
    """Sample 20% standing and otherwise uniformly cardinal moving commands."""

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        command, standing, _ = sample_baseline_matched_cardinal_commands(
            ids.numel(),
            device=self.device,
            standing_probability=float(self.cfg.rel_standing_envs),
        )
        self.vel_command_b[ids] = command
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = standing


__all__ = ["BaselineMatchedCardinalVelocityCommand"]
