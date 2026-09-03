"""Privileged G1 state extraction for the LIPM/DCM certificate.

The extractor is deliberately read-only: it does not add observations, rewards,
or policy state.  All horizontal quantities are expressed in the robot heading
frame (yaw only), and all body masses are read from the running simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat


@dataclass(frozen=True)
class G1StateExtractorCfg:
    gravity: float = 9.81
    h_eff: float | None = None
    contact_force_threshold: float = 5.0
    min_touchdown_interval: float = 0.08
    fallback_step_period: float = 0.60
    minimum_com_height: float = 0.20
    left_foot_body_name: str = "left_ankle_roll_link"
    right_foot_body_name: str = "right_ankle_roll_link"
    use_terrain_plane_geometry: bool = False
    slope_alignment_tolerance: float = 0.05

    def __post_init__(self):
        if self.gravity <= 0.0:
            raise ValueError("gravity must be positive")
        if self.h_eff is not None and self.h_eff <= 0.0:
            raise ValueError("h_eff must be positive when supplied")
        if self.contact_force_threshold <= 0.0:
            raise ValueError("contact_force_threshold must be positive")
        if self.min_touchdown_interval < 0.0 or self.fallback_step_period <= 0.0:
            raise ValueError("touchdown debounce and fallback step period are invalid")
        if self.slope_alignment_tolerance < 0.0:
            raise ValueError("slope_alignment_tolerance must be non-negative")


@dataclass(frozen=True)
class G1PrivilegedRecoveryState:
    """A vectorized snapshot; one row corresponds to one Isaac Lab environment."""

    time: torch.Tensor
    episode_reset: torch.Tensor
    com_position: torch.Tensor
    com_velocity: torch.Tensor
    com_height: torch.Tensor
    left_foot_position: torch.Tensor
    right_foot_position: torch.Tensor
    left_foot_position_w: torch.Tensor
    right_foot_position_w: torch.Tensor
    heading_quat_w: torch.Tensor
    contacts: torch.Tensor
    contact_forces: torch.Tensor
    support_is_left: torch.Tensor
    swing_is_left: torch.Tensor
    touchdown: torch.Tensor
    touchdown_foot: torch.Tensor
    step_period: torch.Tensor
    step_period_is_fallback: torch.Tensor
    phase: torch.Tensor
    command_velocity: torch.Tensor
    omega: torch.Tensor
    b: torch.Tensor
    q: torch.Tensor
    root_roll_pitch: torch.Tensor
    terrain_normal_heading: torch.Tensor
    terrain_plane_point_w: torch.Tensor
    signed_slope: torch.Tensor
    terrain_plane_valid: torch.Tensor

    @property
    def support_side(self) -> list[str]:
        return ["left" if value else "right" for value in self.support_is_left.tolist()]


class G1PrivilegedStateExtractor:
    """Extract certificate inputs from a running ``BaseEnv`` using simulator truth."""

    LEFT = 0
    RIGHT = 1
    NO_TOUCHDOWN = -1

    def __init__(self, env, cfg: G1StateExtractorCfg | None = None):
        self.env = env
        self.cfg = cfg or G1StateExtractorCfg()
        self.device = env.device
        self.num_envs = env.num_envs

        foot_names = [self.cfg.left_foot_body_name, self.cfg.right_foot_body_name]
        self._robot_foot_ids, resolved_robot_names = env.robot.find_bodies(foot_names, preserve_order=True)
        self._sensor_foot_ids, resolved_sensor_names = env.contact_sensor.find_bodies(
            foot_names, preserve_order=True
        )
        if resolved_robot_names != foot_names or resolved_sensor_names != foot_names:
            raise RuntimeError(
                "Could not resolve the G1 feet in left/right order: "
                f"robot={resolved_robot_names}, sensor={resolved_sensor_names}"
            )

        masses = env.robot.root_physx_view.get_masses().clone().to(self.device)
        self._masses = masses.reshape(self.num_envs, -1)
        body_count = env.robot.data.body_com_pose_w.shape[1]
        if self._masses.shape[1] != body_count:
            raise RuntimeError(
                f"Mass/body count mismatch: {self._masses.shape[1]} masses for {body_count} bodies"
            )
        if torch.any(self._masses <= 0.0):
            raise RuntimeError("The running articulation contains a non-positive rigid-body mass")
        self._total_mass = self._masses.sum(dim=1, keepdim=True)

        force_norm = self._foot_force_norm()
        self._previous_contacts = force_norm > self.cfg.contact_force_threshold
        self._support_is_left = force_norm[:, self.LEFT] >= force_norm[:, self.RIGHT]
        self._last_touchdown_time = torch.full((self.num_envs,), torch.nan, device=self.device)
        self._step_period = torch.full(
            (self.num_envs,), self.cfg.fallback_step_period, device=self.device
        )
        self._period_is_fallback = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_episode_length = env.episode_length_buf.clone()

    @property
    def total_mass(self) -> torch.Tensor:
        """Current total simulated mass for every environment."""

        return self._total_mass.squeeze(-1)

    def _foot_force_norm(self) -> torch.Tensor:
        forces = self.env.contact_sensor.data.net_forces_w[:, self._sensor_foot_ids]
        return torch.linalg.vector_norm(forces, dim=-1)

    def _reset_history(self, reset_mask: torch.Tensor, contacts: torch.Tensor, force_norm: torch.Tensor):
        if not torch.any(reset_mask):
            return
        self._last_touchdown_time[reset_mask] = torch.nan
        self._step_period[reset_mask] = self.cfg.fallback_step_period
        self._period_is_fallback[reset_mask] = True
        self._previous_contacts[reset_mask] = contacts[reset_mask]
        self._support_is_left[reset_mask] = (
            force_norm[reset_mask, self.LEFT] >= force_norm[reset_mask, self.RIGHT]
        )

    def extract(self) -> G1PrivilegedRecoveryState:
        """Return the current vectorized privileged state.

        Call this once immediately after each ``env.step``.  The extractor uses
        its own small contact-edge/debounce history and never writes to the env.
        """

        env = self.env
        now_value = float(env.sim_step_counter) * float(env.physics_dt)
        now = torch.full((self.num_envs,), now_value, device=self.device)

        force_norm = self._foot_force_norm()
        contacts = force_norm > self.cfg.contact_force_threshold
        reset_mask = env.episode_length_buf < self._last_episode_length
        self._reset_history(reset_mask, contacts, force_norm)

        contact_edges = contacts & ~self._previous_contacts & ~reset_mask.unsqueeze(-1)
        elapsed = now - self._last_touchdown_time
        debounce_ok = torch.isnan(self._last_touchdown_time) | (elapsed >= self.cfg.min_touchdown_interval)
        candidates = contact_edges & debounce_ok.unsqueeze(-1)

        touchdown = torch.any(candidates, dim=1)
        touchdown_foot = torch.full(
            (self.num_envs,), self.NO_TOUCHDOWN, dtype=torch.long, device=self.device
        )
        only_left = candidates[:, self.LEFT] & ~candidates[:, self.RIGHT]
        only_right = candidates[:, self.RIGHT] & ~candidates[:, self.LEFT]
        both = candidates[:, self.LEFT] & candidates[:, self.RIGHT]
        touchdown_foot[only_left] = self.LEFT
        touchdown_foot[only_right] = self.RIGHT
        touchdown_foot[both] = torch.where(
            force_norm[both, self.LEFT] >= force_norm[both, self.RIGHT], self.LEFT, self.RIGHT
        )

        had_previous_touchdown = torch.isfinite(self._last_touchdown_time)
        measured_period = now - self._last_touchdown_time
        period_update = touchdown & had_previous_touchdown
        self._step_period[period_update] = measured_period[period_update]
        self._period_is_fallback[period_update] = False
        self._last_touchdown_time[touchdown] = now[touchdown]

        self._support_is_left[touchdown] = touchdown_foot[touchdown] == self.LEFT
        left_only_contact = contacts[:, self.LEFT] & ~contacts[:, self.RIGHT]
        right_only_contact = contacts[:, self.RIGHT] & ~contacts[:, self.LEFT]
        self._support_is_left[left_only_contact] = True
        self._support_is_left[right_only_contact] = False

        since_touchdown = now - self._last_touchdown_time
        episode_time = env.episode_length_buf.to(torch.float32) * float(env.step_dt)
        phase_time = torch.where(torch.isfinite(since_touchdown), since_touchdown, episode_time)
        phase = torch.clamp(phase_time / self._step_period, min=0.0, max=1.0 - 1.0e-6)

        body_com_pos_w = env.robot.data.body_com_pose_w[..., :3]
        body_com_vel_w = env.robot.data.body_com_vel_w[..., :3]
        weights = self._masses.unsqueeze(-1)
        com_pos_w = (weights * body_com_pos_w).sum(dim=1) / self._total_mass
        com_vel_w = (weights * body_com_vel_w).sum(dim=1) / self._total_mass

        heading_quat_w = yaw_quat(env.robot.data.root_quat_w)
        env_origins = env.scene.env_origins
        com_position = quat_apply_inverse(heading_quat_w, com_pos_w - env_origins)
        com_velocity = quat_apply_inverse(heading_quat_w, com_vel_w)

        foot_pos_w = env.robot.data.body_link_pos_w[:, self._robot_foot_ids]
        heading_for_feet = heading_quat_w.unsqueeze(1).expand(-1, 2, -1)
        foot_position = quat_apply_inverse(
            heading_for_feet, foot_pos_w - env_origins.unsqueeze(1)
        )
        left_foot_position = foot_position[:, self.LEFT]
        right_foot_position = foot_position[:, self.RIGHT]

        support_position = torch.where(
            self._support_is_left.unsqueeze(-1), left_foot_position, right_foot_position
        )
        swing_position = torch.where(
            self._support_is_left.unsqueeze(-1), right_foot_position, left_foot_position
        )
        if self.cfg.use_terrain_plane_geometry:
            provider = getattr(env, "get_recovery_plane_geometry", None)
            if provider is None:
                raise RuntimeError(
                    "plane state extraction requires env.get_recovery_plane_geometry()"
                )
            terrain_normal_w, terrain_plane_point_w, provider_valid = provider()
            terrain_normal_w = terrain_normal_w.to(device=self.device, dtype=com_pos_w.dtype)
            terrain_plane_point_w = terrain_plane_point_w.to(
                device=self.device, dtype=com_pos_w.dtype
            )
            provider_valid = provider_valid.to(device=self.device, dtype=torch.bool)
        else:
            terrain_normal_w = torch.zeros_like(com_pos_w)
            terrain_normal_w[:, 2] = 1.0
            terrain_plane_point_w = env_origins.clone()
            provider_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        terrain_normal_heading = quat_apply_inverse(heading_quat_w, terrain_normal_w)
        normal_norm = torch.linalg.vector_norm(terrain_normal_heading, dim=1)
        finite_geometry = (
            torch.all(torch.isfinite(terrain_normal_heading), dim=1)
            & torch.all(torch.isfinite(terrain_plane_point_w), dim=1)
            & (normal_norm > 0.0)
        )
        safe_norm = torch.clamp(normal_norm, min=1.0e-12)
        terrain_normal_heading = terrain_normal_heading / safe_norm.unsqueeze(-1)
        terrain_plane_valid = (
            provider_valid
            & finite_geometry
            & (terrain_normal_heading[:, 2] > 0.0)
            & (
                torch.abs(terrain_normal_heading[:, 1])
                <= self.cfg.slope_alignment_tolerance
            )
        )
        signed_slope = torch.atan2(
            -terrain_normal_heading[:, 0], terrain_normal_heading[:, 2]
        )
        signed_slope = torch.where(
            terrain_plane_valid, signed_slope, torch.zeros_like(signed_slope)
        )

        # Vertical height above the local plane: n·(r_com-r0)/n_z.  This is
        # intentionally not the normal distance.
        safe_normal_z = torch.where(
            terrain_normal_w[:, 2] > 0.0,
            terrain_normal_w[:, 2],
            torch.ones_like(terrain_normal_w[:, 2]),
        )
        com_height = torch.sum(
            terrain_normal_w * (com_pos_w - terrain_plane_point_w), dim=1
        ) / safe_normal_z
        omega_height = (
            torch.full_like(com_height, self.cfg.h_eff)
            if self.cfg.h_eff is not None
            else torch.clamp(com_height, min=self.cfg.minimum_com_height)
        )
        omega = torch.sqrt(self.cfg.gravity / omega_height)

        xi = com_position[:, :2] + com_velocity[:, :2] / omega.unsqueeze(-1)
        b = xi - support_position[:, :2]
        q = swing_position[:, :2] - support_position[:, :2]

        roll, pitch, _ = euler_xyz_from_quat(env.robot.data.root_quat_w)
        root_roll_pitch = torch.stack((roll, pitch), dim=-1)

        result = G1PrivilegedRecoveryState(
            time=now,
            episode_reset=reset_mask.clone(),
            com_position=com_position,
            com_velocity=com_velocity,
            com_height=com_height,
            left_foot_position=left_foot_position,
            right_foot_position=right_foot_position,
            left_foot_position_w=foot_pos_w[:, self.LEFT].clone(),
            right_foot_position_w=foot_pos_w[:, self.RIGHT].clone(),
            heading_quat_w=heading_quat_w,
            contacts=contacts,
            contact_forces=force_norm,
            support_is_left=self._support_is_left.clone(),
            swing_is_left=~self._support_is_left,
            touchdown=touchdown,
            touchdown_foot=touchdown_foot,
            step_period=self._step_period.clone(),
            step_period_is_fallback=self._period_is_fallback.clone(),
            phase=phase,
            command_velocity=env.command_generator.command.clone(),
            omega=omega,
            b=b,
            q=q,
            root_roll_pitch=root_roll_pitch,
            terrain_normal_heading=terrain_normal_heading,
            terrain_plane_point_w=terrain_plane_point_w,
            signed_slope=signed_slope,
            terrain_plane_valid=terrain_plane_valid,
        )

        self._previous_contacts.copy_(contacts)
        self._last_episode_length.copy_(env.episode_length_buf)
        return result


def theoretical_periodic_state(
    vx_cmd: float,
    vy_cmd: float,
    step_period: float,
    omega: float,
    step_width: float,
) -> dict[str, tuple[float, float]]:
    """Compute the unchanged theory's periodic ``b*`` and ``q*`` values."""

    gain = math.exp(omega * step_period)
    landing_left = torch.tensor((vx_cmd * step_period, vy_cmd * step_period - step_width), dtype=torch.float64)
    landing_right = torch.tensor((vx_cmd * step_period, vy_cmd * step_period + step_width), dtype=torch.float64)
    denominator = gain * gain - 1.0
    b_left = (gain * landing_left + landing_right) / denominator
    b_right = (landing_left + gain * landing_right) / denominator
    return {
        "b_left": tuple(float(value) for value in b_left),
        "b_right": tuple(float(value) for value in b_right),
        "q_left": tuple(float(value) for value in -landing_right),
        "q_right": tuple(float(value) for value in -landing_left),
        "landing_left": tuple(float(value) for value in landing_left),
        "landing_right": tuple(float(value) for value in landing_right),
    }
