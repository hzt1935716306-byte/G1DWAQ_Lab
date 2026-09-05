"""Baseline-matched curriculum variants of the frozen Plane V1 method."""

from __future__ import annotations

from isaaclab.utils import configclass

from legged_lab.envs.g1.g1_plane_v1_config import (
    G1PlaneV1AgentCfg,
    G1PlaneV1EnvCfg,
    G1PlaneV1RecoverabilityRewardCfg,
)
from legged_lab.recovery.baseline_matched_protocol import (
    CURRICULUM_REFERENCE_TILE_LENGTH,
    configure_matched_command_and_reset,
)
from legged_lab.terrains import make_plane_baseline_matched_terrain_cfg


@configclass
class G1PlaneV1BaselineMatchedEnvCfg(G1PlaneV1EnvCfg):
    """Shared matched task; source/reward subclasses remain the only ablations."""

    curriculum_reference_tile_length: float = CURRICULUM_REFERENCE_TILE_LENGTH

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = make_plane_baseline_matched_terrain_cfg(
            int(self.scene.seed)
        )
        self.scene.max_init_terrain_level = 5

        self.plane_recovery.minimum_command_speed = 0.2
        configure_matched_command_and_reset(self)

        # Coplanar geometry tolerates baseline x/y spawn offsets.  Yaw remains
        # the one theory-required exception to baseline randomization.
        reset_base = self.domain_rand.events.reset_base
        reset_base.params["pose_range"]["x"] = (-0.5, 0.5)
        reset_base.params["pose_range"]["y"] = (-0.5, 0.5)

        # Ordinary locomotion terms exactly match g1_slope_sys_d_matched.
        # Reward-on variants add only the separate touchdown-event channel.
        self.reward.track_lin_vel_xy_exp.weight = 1.0
        self.reward.track_ang_vel_z_exp.weight = 1.0
        self.reward.joint_deviation_hip.weight = -0.15


@configclass
class G1PlaneV1EstimatorContextNoRewardMatchedEnvCfg(G1PlaneV1BaselineMatchedEnvCfg):
    com_velocity_source: str = "estimator"
    plane_v1_reward: G1PlaneV1RecoverabilityRewardCfg = G1PlaneV1RecoverabilityRewardCfg(
        enabled=False
    )


@configclass
class G1PlaneV1EstimatorContextRewardMatchedEnvCfg(G1PlaneV1BaselineMatchedEnvCfg):
    com_velocity_source: str = "estimator"
    plane_v1_reward: G1PlaneV1RecoverabilityRewardCfg = G1PlaneV1RecoverabilityRewardCfg(
        enabled=True
    )


@configclass
class G1PlaneV1PrivilegedContextNoRewardMatchedEnvCfg(G1PlaneV1BaselineMatchedEnvCfg):
    com_velocity_source: str = "privileged"
    plane_v1_reward: G1PlaneV1RecoverabilityRewardCfg = G1PlaneV1RecoverabilityRewardCfg(
        enabled=False
    )


@configclass
class G1PlaneV1PrivilegedContextRewardMatchedEnvCfg(G1PlaneV1BaselineMatchedEnvCfg):
    com_velocity_source: str = "privileged"
    plane_v1_reward: G1PlaneV1RecoverabilityRewardCfg = G1PlaneV1RecoverabilityRewardCfg(
        enabled=True
    )


@configclass
class G1PlaneV1BaselineMatchedAgentCfg(G1PlaneV1AgentCfg):
    experiment_name: str = "g1_plane_v1_matched"
    wandb_project: str = "g1_plane_v1_matched"
    num_steps_per_env: int = 24
    max_iterations: int = 10000
    resume: bool = False


@configclass
class G1PlaneV1EstimatorContextNoRewardMatchedAgentCfg(G1PlaneV1BaselineMatchedAgentCfg):
    run_name: str = "estimator_context_no_reward_matched"


@configclass
class G1PlaneV1EstimatorContextRewardMatchedAgentCfg(G1PlaneV1BaselineMatchedAgentCfg):
    run_name: str = "estimator_context_reward_matched"


@configclass
class G1PlaneV1PrivilegedContextNoRewardMatchedAgentCfg(G1PlaneV1BaselineMatchedAgentCfg):
    run_name: str = "privileged_context_no_reward_matched"


@configclass
class G1PlaneV1PrivilegedContextRewardMatchedAgentCfg(G1PlaneV1BaselineMatchedAgentCfg):
    run_name: str = "privileged_context_reward_matched"


__all__ = [name for name in globals() if name.startswith("G1PlaneV1")]
