"""Environment classes that decouple 64 m planes from the 8 m curriculum rule."""

from __future__ import annotations

import torch

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.g1.g1_dwaq_env import G1DwaqEnv
from legged_lab.recovery.baseline_matched_protocol import terrain_curriculum_decisions
from legged_lab.terrains import make_plane_baseline_matched_terrain_cfg


class _MatchedTerrainCurriculumMixin:
    def __init__(self, cfg, headless):
        cfg.scene.terrain_generator = make_plane_baseline_matched_terrain_cfg(
            int(cfg.scene.seed)
        )
        super().__init__(cfg, headless)

    def update_terrain_levels(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        distance = torch.linalg.vector_norm(
            self.robot.data.root_pos_w[ids, :2] - self.scene.env_origins[ids, :2], dim=1
        )
        command_speed = torch.linalg.vector_norm(
            self.command_generator.command[ids, :2], dim=1
        )
        move_up, move_down = terrain_curriculum_decisions(
            distance,
            command_speed,
            self.max_episode_length_s,
            self.cfg.curriculum_reference_tile_length,
        )
        self.scene.terrain.update_env_origins(ids, move_up, move_down)
        return {
            "Curriculum/terrain_levels": self.scene.terrain.terrain_levels.float().mean(),
            "Curriculum/reference_move_up_distance_m": 4.0,
        }


class G1SlopeBaselineMatchedEnv(_MatchedTerrainCurriculumMixin, BaseEnv):
    pass


class G1DwaqSlopeBaselineMatchedEnv(_MatchedTerrainCurriculumMixin, G1DwaqEnv):
    pass


__all__ = ["G1DwaqSlopeBaselineMatchedEnv", "G1SlopeBaselineMatchedEnv"]
