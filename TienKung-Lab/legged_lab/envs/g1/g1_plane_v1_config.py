"""Final Plane V1 2x2 estimator/privileged and reward-off/on tasks."""

from __future__ import annotations

from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.envs.g1.g1_plane_recovery_config import (
    G1PlaneSymmetricStage2EnvCfg,
)
from legged_lab.envs.g1.g1_config import G1FlatSymmetricAgentCfg
from legged_lab.envs.g1.g1_recovery_config import (
    G1PushCurriculumCfg,
    G1RecoveryContextCfg,
    G1Stage2RewardCfg,
)
from legged_lab.terrains import PLANE_RECOVERY_TERRAINS_CFG


_FINAL_NOMINAL_PARAMETERS = str(
    Path(__file__).resolve().parents[3]
    / "tools/recovery/generated/g1_plane_nominal_params_g1_slope_sys_d_candidate.yaml"
)


@configclass
class G1PlaneV1RecoverabilityRewardCfg:
    """Direct, unscaled final touchdown-event reward coefficients."""

    enabled: bool = False
    certificate_progress_weight: float = 0.25
    certificate_delta_phi_clip: float = 2.0
    unrecovered_touchdown_cost: float = -0.05
    td5_unrecovered_penalty: float = -0.25
    certificate_target_n_max: int = 1
    certificate_horizon_touchdowns: int = 5


@configclass
class G1PlaneV1EnvCfg(G1PlaneSymmetricStage2EnvCfg):
    """Shared final environment; subclasses change only source and reward flag."""

    com_velocity_source: str = "estimator"
    estimator_checkpoint_path: str = ""
    estimator_imu_acceleration_scale: float = 0.05
    terrain_level_1_iteration: int = 2000
    terrain_level_2_iteration: int = 4000
    push_mode: str = "fixed_full_range"
    plane_v1_reward: G1PlaneV1RecoverabilityRewardCfg = G1PlaneV1RecoverabilityRewardCfg()
    push_curriculum: G1PushCurriculumCfg = G1PushCurriculumCfg(
        enable_push_curriculum=False,
        adaptive_upgrades_enabled=False,
        easy_sample_probability=0.0,
    )
    # Disable every legacy Stage2 channel.  The final environment owns the
    # independent TD0--TD5 reward implementation below this config layer.
    stage2_reward: G1Stage2RewardCfg = G1Stage2RewardCfg(
        enabled=False,
        enable_shared_event_reward=False,
        enable_certificate_reward=False,
        enable_soft_reward_scaling=False,
        defer_certificate_reward_to_rollout_end=False,
    )
    recovery_context: G1RecoveryContextCfg = G1RecoveryContextCfg(
        enabled=True,
        mode="certificate",
    )

    def __post_init__(self):
        super().__post_init__()
        if self.com_velocity_source not in ("estimator", "privileged"):
            raise ValueError("com_velocity_source must be 'estimator' or 'privileged'")
        if self.terrain_level_1_iteration < 0:
            raise ValueError("terrain_level_1_iteration must be non-negative")
        if self.terrain_level_2_iteration <= self.terrain_level_1_iteration:
            raise ValueError("terrain Level 2 must start after Level 1")
        if self.push_mode != "fixed_full_range":
            raise ValueError("Plane V1 supports only fixed_full_range push mode")

        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = PLANE_RECOVERY_TERRAINS_CFG
        self.scene.max_init_terrain_level = 0
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            offset=ImuCfg.OffsetCfg(pos=(0.04525, 0.0, -0.08339)),
            update_period=0.0,
            gravity_bias=(0.0, 0.0, 9.81),
        )

        self.robot.actor_obs_history_length = 5
        self.robot.critic_obs_history_length = 10
        self.plane_recovery.nominal_parameters_path = _FINAL_NOMINAL_PARAMETERS
        self.plane_recovery.minimum_command_speed = 0.2
        self.commands.rel_standing_envs = 0.0
        self.commands.rel_heading_envs = 0.0
        self.commands.heading_command = False
        self.commands.ranges.lin_vel_x = (-0.4, 0.4)
        self.commands.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.ranges.heading = None

        self.domain_rand.events.push_robot = EventTerm(
            func=mdp.fixed_full_range_push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={},
        )

        # The final locomotion reward is G1SymmetricRewardCfg with only these
        # explicitly requested coefficient changes.
        self.reward.track_lin_vel_xy_exp.weight = 2.0
        self.reward.track_ang_vel_z_exp.weight = 2.0
        self.reward.joint_deviation_hip.weight = -0.30


@configclass
class G1PlaneV1EstimatorContextNoRewardEnvCfg(G1PlaneV1EnvCfg):
    com_velocity_source: str = "estimator"
    plane_v1_reward: G1PlaneV1RecoverabilityRewardCfg = G1PlaneV1RecoverabilityRewardCfg(
        enabled=False
    )


@configclass
class G1PlaneV1EstimatorContextRewardEnvCfg(G1PlaneV1EnvCfg):
    com_velocity_source: str = "estimator"
    plane_v1_reward: G1PlaneV1RecoverabilityRewardCfg = G1PlaneV1RecoverabilityRewardCfg(
        enabled=True
    )


@configclass
class G1PlaneV1PrivilegedContextNoRewardEnvCfg(G1PlaneV1EnvCfg):
    com_velocity_source: str = "privileged"
    plane_v1_reward: G1PlaneV1RecoverabilityRewardCfg = G1PlaneV1RecoverabilityRewardCfg(
        enabled=False
    )


@configclass
class G1PlaneV1PrivilegedContextRewardEnvCfg(G1PlaneV1EnvCfg):
    com_velocity_source: str = "privileged"
    plane_v1_reward: G1PlaneV1RecoverabilityRewardCfg = G1PlaneV1RecoverabilityRewardCfg(
        enabled=True
    )


@configclass
class G1PlaneV1AgentCfg(G1FlatSymmetricAgentCfg):
    experiment_name: str = "g1_plane_v1"
    wandb_project: str = "g1_plane_v1"
    max_iterations: int = 10000
    resume: bool = False


@configclass
class G1PlaneV1EstimatorContextNoRewardAgentCfg(G1PlaneV1AgentCfg):
    run_name: str = "estimator_context_no_reward"


@configclass
class G1PlaneV1EstimatorContextRewardAgentCfg(G1PlaneV1AgentCfg):
    run_name: str = "estimator_context_reward"


@configclass
class G1PlaneV1PrivilegedContextNoRewardAgentCfg(G1PlaneV1AgentCfg):
    run_name: str = "privileged_context_no_reward"


@configclass
class G1PlaneV1PrivilegedContextRewardAgentCfg(G1PlaneV1AgentCfg):
    run_name: str = "privileged_context_reward"


__all__ = [name for name in globals() if name.startswith("G1PlaneV1")]
