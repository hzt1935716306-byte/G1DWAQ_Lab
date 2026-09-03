"""Plane-generalized G1 recovery environment.

The environment is intentionally limited to x-aligned coplanar terrain and
cardinal heading-frame commands.  The legacy flat-only environment remains in
``g1_recovery_env.py`` unchanged in meaning.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg

from legged_lab.envs.g1.g1_recovery_env import G1RecoveryEnv
from legged_lab.recovery.plane_certificate_runtime import (
    PlaneCalibratedG1CertificateEvaluator,
)
from legged_lab.recovery.plane_nominal_params import PlaneNominalParameterTable
from legged_lab.recovery.state_extractor import G1PrivilegedStateExtractor, G1StateExtractorCfg


class CardinalVelocityCommand(UniformVelocityCommand):
    """Sample exactly one of +x, -x, +y, -y with zero yaw command."""

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        direction = torch.randint(0, 4, (ids.numel(),), device=self.device)
        minimum = float(self._env.cfg.plane_recovery.minimum_command_speed)
        ranges = self.cfg.ranges
        maxima = torch.tensor(
            (
                float(ranges.lin_vel_x[1]),
                abs(float(ranges.lin_vel_x[0])),
                float(ranges.lin_vel_y[1]),
                abs(float(ranges.lin_vel_y[0])),
            ),
            device=self.device,
        )
        selected_maxima = maxima[direction]
        if torch.any(selected_maxima < minimum):
            raise ValueError("cardinal command maximum is below minimum_command_speed")
        speed = minimum + torch.rand(ids.numel(), device=self.device) * (
            selected_maxima - minimum
        )
        self.vel_command_b[ids] = 0.0
        self.vel_command_b[ids, 0] = torch.where(
            direction == 0,
            speed,
            torch.where(direction == 1, -speed, torch.zeros_like(speed)),
        )
        self.vel_command_b[ids, 1] = torch.where(
            direction == 2,
            speed,
            torch.where(direction == 3, -speed, torch.zeros_like(speed)),
        )
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = False


class G1PlaneRecoveryEnv(G1RecoveryEnv):
    """Use projected flat capability and plane-dependent nominal parameters."""

    def _create_command_generator(self, command_cfg: UniformVelocityCommandCfg):
        return CardinalVelocityCommand(cfg=command_cfg, env=self)

    def _create_state_extractor(self):
        return G1PrivilegedStateExtractor(
            self,
            G1StateExtractorCfg(
                h_eff=None,
                use_terrain_plane_geometry=True,
                slope_alignment_tolerance=self.cfg.plane_recovery.slope_alignment_tolerance,
            ),
        )

    def _create_certificate_evaluator(self):
        cfg = self.cfg.stage2_reward
        plane_cfg = self.cfg.plane_recovery
        return PlaneCalibratedG1CertificateEvaluator(
            cfg.certificate_parameters_path,
            plane_cfg.nominal_parameters_path,
            workers=cfg.certificate_workers,
            executor_type=cfg.certificate_executor,
            failure_window_size=cfg.certificate_failure_window_size,
            failure_rate_threshold=cfg.certificate_failure_rate_threshold,
            z_sole=plane_cfg.z_sole,
        )

    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        count = self.num_envs
        self._plane_nominal_table = PlaneNominalParameterTable.from_yaml(
            cfg.plane_recovery.nominal_parameters_path
        )
        self._nominal_cache_key = torch.full((count, 3), torch.nan, device=self.device)
        self._nominal_cache_valid = torch.zeros(count, dtype=torch.bool, device=self.device)
        self._nominal_cache_attitude = torch.zeros((count, 2), device=self.device)
        self._nominal_cache_thresholds = torch.zeros((count, 3), device=self.device)

    def get_recovery_plane_geometry(self):
        """Return exact configured plane normals and one point per environment."""

        terrain = self.scene.terrain
        terrain_types = getattr(terrain, "terrain_types", None)
        slopes = tuple(float(value) for value in self.cfg.plane_recovery.slopes_degrees)
        valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if terrain_types is None:
            slope_degrees = torch.zeros(self.num_envs, device=self.device)
            valid.zero_()
        else:
            slope_table = torch.tensor(slopes, dtype=torch.float32, device=self.device)
            valid &= (terrain_types >= 0) & (terrain_types < len(slopes))
            safe_types = torch.clamp(terrain_types, min=0, max=max(len(slopes) - 1, 0))
            slope_degrees = slope_table[safe_types]
        alpha = torch.deg2rad(slope_degrees)
        normal_world = torch.stack(
            (-torch.sin(alpha), torch.zeros_like(alpha), torch.cos(alpha)), dim=-1
        )
        return normal_world, self.scene.env_origins.clone(), valid

    def _refresh_nominal_cache(self, state) -> None:
        key = torch.stack(
            (
                state.signed_slope,
                state.command_velocity[:, 0],
                state.command_velocity[:, 1],
            ),
            dim=-1,
        )
        changed = torch.any(~torch.isclose(key, self._nominal_cache_key, atol=1.0e-8, rtol=0.0), dim=1)
        changed |= ~state.terrain_plane_valid
        ids = changed.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        for env_id in ids:
            command = state.command_velocity[env_id].detach().cpu().tolist()
            alpha = float(state.signed_slope[env_id].item())
            lookup = (
                self._plane_nominal_table.lookup_command(alpha, command)
                if bool(state.terrain_plane_valid[env_id].item())
                else None
            )
            valid = bool(lookup is not None and lookup.valid and lookup.value is not None)
            self._nominal_cache_valid[env_id] = valid
            if valid:
                value = lookup.value
                assert value is not None
                self._nominal_cache_attitude[env_id] = torch.tensor(
                    (value.roll_star, value.pitch_star), device=self.device
                )
                self._nominal_cache_thresholds[env_id] = torch.tensor(
                    (
                        value.mean_velocity_error_threshold,
                        value.mean_abs_roll_error_threshold,
                        value.mean_abs_pitch_error_threshold,
                    ),
                    device=self.device,
                )
            else:
                self._nominal_cache_attitude[env_id] = 0.0
                self._nominal_cache_thresholds[env_id] = 0.0
        self._nominal_cache_key.copy_(key)

    def _practical_errors(self, state):
        self._refresh_nominal_cache(state)
        velocity_error = torch.linalg.vector_norm(
            state.com_velocity[:, :2] - state.command_velocity[:, :2], dim=1
        )
        attitude_error = torch.abs(state.root_roll_pitch - self._nominal_cache_attitude)
        invalid = ~self._nominal_cache_valid
        velocity_error[invalid] = math.inf
        attitude_error[invalid] = math.inf
        return velocity_error, attitude_error

    def _practical_thresholds(self, state, env_ids: torch.Tensor):
        self._refresh_nominal_cache(state)
        values = self._nominal_cache_thresholds[env_ids]
        return values[:, 0], values[:, 1], values[:, 2]


__all__ = ["CardinalVelocityCommand", "G1PlaneRecoveryEnv"]
