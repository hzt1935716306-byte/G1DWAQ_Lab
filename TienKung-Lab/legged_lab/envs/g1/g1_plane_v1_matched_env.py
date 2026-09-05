"""Runtime for baseline-matched continuous-plane Plane V1 tasks."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg

from legged_lab.envs.g1.g1_matched_command import BaselineMatchedCardinalVelocityCommand
from legged_lab.envs.g1.g1_plane_v1_env import G1PlaneV1Env
from legged_lab.recovery.baseline_matched_protocol import (
    lookup_matched_slope,
    terrain_curriculum_decisions,
)
from legged_lab.terrains import (
    make_plane_baseline_matched_slope_table,
    make_plane_baseline_matched_terrain_cfg,
)


class G1PlaneV1BaselineMatchedEnv(G1PlaneV1Env):
    """Use baseline performance curriculum with exact row/column plane geometry."""

    def __init__(self, cfg, headless):
        # CLI seed overrides happen after config construction.  Rebuild here so
        # TerrainGenerator and the metadata replay always use the same seed.
        terrain_seed = int(cfg.scene.seed)
        cfg.scene.terrain_generator = make_plane_baseline_matched_terrain_cfg(terrain_seed)
        self._matched_slope_table_numpy = make_plane_baseline_matched_slope_table(
            terrain_seed
        )
        super().__init__(cfg, headless)
        self._matched_slope_table = torch.as_tensor(
            self._matched_slope_table_numpy,
            dtype=torch.float32,
            device=self.device,
        )

    def _create_command_generator(self, command_cfg: UniformVelocityCommandCfg):
        return BaselineMatchedCardinalVelocityCommand(cfg=command_cfg, env=self)

    def get_recovery_plane_geometry(self):
        terrain = self.scene.terrain
        levels = getattr(terrain, "terrain_levels", None)
        types = getattr(terrain, "terrain_types", None)
        valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if levels is None or types is None:
            alpha = torch.zeros(self.num_envs, device=self.device)
            valid.zero_()
        else:
            alpha, valid = lookup_matched_slope(
                self._matched_slope_table, levels, types
            )
        normal_world = torch.stack(
            (-torch.sin(alpha), torch.zeros_like(alpha), torch.cos(alpha)), dim=-1
        )
        return normal_world, self.scene.env_origins.clone(), valid

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
            "Curriculum/reference_move_up_distance_m": float(
                self.cfg.curriculum_reference_tile_length / 2.0
            ),
        }

    def _terrain_log(self, level: int) -> dict[str, float]:
        del level
        terrain = self.scene.terrain
        levels = terrain.terrain_levels
        types = terrain.terrain_types
        alpha = self._matched_slope_table[levels, types]
        return {
            "TerrainCurriculum/mean_level": float(levels.float().mean().item()),
            "TerrainCurriculum/max_abs_slope_deg": float(
                torch.rad2deg(alpha.abs()).max().item()
            ),
            "TerrainCurriculum/P_flat": float((alpha.abs() < 1.0e-8).float().mean().item()),
            "TerrainCurriculum/P_uphill": float((alpha > 1.0e-8).float().mean().item()),
            "TerrainCurriculum/P_downhill": float((alpha < -1.0e-8).float().mean().item()),
        }


__all__ = ["BaselineMatchedCardinalVelocityCommand", "G1PlaneV1BaselineMatchedEnv"]
