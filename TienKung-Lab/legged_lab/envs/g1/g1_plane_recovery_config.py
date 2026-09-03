"""Flat/uphill/downhill Stage2 recovery configuration."""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from legged_lab.envs.g1.g1_recovery_config import (
    G1FlatSymmetricRecoveryAgentCfg,
    G1FlatSymmetricRecoveryEnvCfg,
    G1RecoveryContextCfg,
    G1Stage2RewardCfg,
)
from legged_lab.recovery.stage2_reward import DEFAULT_EVENT_SCALE
from legged_lab.terrains import PLANE_RECOVERY_SLOPES_DEG, PLANE_RECOVERY_TERRAINS_CFG


_DEFAULT_PLANE_NOMINAL_PARAMETERS = str(
    Path(__file__).resolve().parents[3]
    / "tools/recovery/generated/g1_plane_nominal_params.yaml"
)


@configclass
class G1PlaneRecoveryCfg:
    """Scope and geometry constants for the first x-aligned plane version."""

    slopes_degrees: tuple[float, ...] = PLANE_RECOVERY_SLOPES_DEG
    slope_alignment_tolerance: float = 0.05
    z_sole: float = -0.045
    minimum_command_speed: float = 0.2
    nominal_parameters_path: str = _DEFAULT_PLANE_NOMINAL_PARAMETERS


@configclass
class G1PlaneSymmetricStage2EnvCfg(G1FlatSymmetricRecoveryEnvCfg):
    """Shared plane task; Baseline and Ours differ only in context mode."""

    plane_recovery: G1PlaneRecoveryCfg = G1PlaneRecoveryCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = PLANE_RECOVERY_TERRAINS_CFG
        self.scene.max_init_terrain_level = 0

        # First-version scope: no active yaw and cardinal commands only.  The
        # specialized environment performs the discrete direction sampling.
        self.commands.rel_standing_envs = 0.0
        self.commands.rel_heading_envs = 0.0
        self.commands.heading_command = False
        self.commands.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.ranges.heading = (0.0, 0.0)

        reset_base = self.domain_rand.events.reset_base
        reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        reset_base.params["velocity_range"]["yaw"] = (0.0, 0.0)


@configclass
class G1PlaneSymmetricStage2BaselineEnvCfg(G1PlaneSymmetricStage2EnvCfg):
    """963-D Actor with a permanently zero three-value context."""

    stage2_reward: G1Stage2RewardCfg = G1Stage2RewardCfg(
        enabled=True,
        enable_shared_event_reward=True,
        enable_certificate_reward=False,
        enable_soft_reward_scaling=False,
        defer_certificate_reward_to_rollout_end=False,
        event_scale=DEFAULT_EVENT_SCALE,
    )
    recovery_context: G1RecoveryContextCfg = G1RecoveryContextCfg(
        enabled=True,
        mode="zero",
    )


@configclass
class G1PlaneSymmetricStage2OursEnvCfg(G1PlaneSymmetricStage2EnvCfg):
    """963-D Actor with touchdown-held ``[N_norm, margin_norm, valid]``."""

    stage2_reward: G1Stage2RewardCfg = G1Stage2RewardCfg(
        enabled=True,
        enable_shared_event_reward=True,
        enable_certificate_reward=False,
        enable_soft_reward_scaling=False,
        defer_certificate_reward_to_rollout_end=False,
        event_scale=DEFAULT_EVENT_SCALE,
    )
    recovery_context: G1RecoveryContextCfg = G1RecoveryContextCfg(
        enabled=True,
        mode="certificate",
    )


@configclass
class G1PlaneSymmetricStage2BaselineAgentCfg(G1FlatSymmetricRecoveryAgentCfg):
    experiment_name: str = "g1_plane_symmetric"
    wandb_project: str = "g1_plane_symmetric"
    run_name: str = "stage2_plane_context_zero_baseline"


@configclass
class G1PlaneSymmetricStage2OursAgentCfg(G1FlatSymmetricRecoveryAgentCfg):
    experiment_name: str = "g1_plane_symmetric"
    wandb_project: str = "g1_plane_symmetric"
    run_name: str = "stage2_plane_context_input_only"


__all__ = [
    "G1PlaneRecoveryCfg",
    "G1PlaneSymmetricStage2BaselineAgentCfg",
    "G1PlaneSymmetricStage2BaselineEnvCfg",
    "G1PlaneSymmetricStage2EnvCfg",
    "G1PlaneSymmetricStage2OursAgentCfg",
    "G1PlaneSymmetricStage2OursEnvCfg",
]
