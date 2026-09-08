"""Additive reward ablations; the original matched tasks remain unchanged."""
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.envs.g1.g1_config import G1SymmetricRewardCfg
from legged_lab.envs.g1.g1_dwaq_nosys_config import G1DwaqNoSysRewardCfg
from legged_lab.envs.g1.g1_plane_v1_matched_config import (
    G1PlaneV1EstimatorContextNoRewardMatchedEnvCfg,
    G1PlaneV1EstimatorContextNoRewardMatchedAgentCfg,
)
from legged_lab.envs.g1.g1_slope_matched_config import (
    G1DwaqSlopeNoSysDMatchedEnvCfg, G1DwaqSlopeNoSysDMatchedAgentCfg,
)


@configclass
class G1PlaneContextIdleSwingRewardCfg(G1SymmetricRewardCfg):
    idle_penalty = RewTerm(
        func=mdp.idle_when_commanded, weight=-0.1,
        params={"cmd_threshold": 0.2, "vel_threshold": 0.1},
    )
    feet_swing_height = RewTerm(
        func=mdp.feet_swing_height, weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "target_height": 0.08,
        },
    )


@configclass
class G1PlaneV1EstimatorContextNoRewardMatchedV2EnvCfg(G1PlaneV1EstimatorContextNoRewardMatchedEnvCfg):
    reward = G1PlaneContextIdleSwingRewardCfg()


@configclass
class G1PlaneV1EstimatorContextNoRewardMatchedV2AgentCfg(G1PlaneV1EstimatorContextNoRewardMatchedAgentCfg):
    experiment_name: str = "g1_plane_v1_estimator_context_no_reward_matched_v2"
    wandb_project: str = "g1_plane_v1_estimator_context_no_reward_matched_v2"
    run_name: str = "estimator_context_no_reward_matched_v2_idle01_swing02"
    resume: bool = False


@configclass
class G1DwaqNoIdleRewardCfg(G1DwaqNoSysRewardCfg):
    idle_penalty = None


@configclass
class G1DwaqNoSwingHeightRewardCfg(G1DwaqNoSysRewardCfg):
    feet_swing_height = None


@configclass
class G1DwaqSlopeNoSysDMatchedV2EnvCfg(G1DwaqSlopeNoSysDMatchedEnvCfg):
    reward = G1DwaqNoIdleRewardCfg()


@configclass
class G1DwaqSlopeNoSysDMatchedV3EnvCfg(G1DwaqSlopeNoSysDMatchedEnvCfg):
    reward = G1DwaqNoSwingHeightRewardCfg()


@configclass
class G1DwaqSlopeNoSysDMatchedV2AgentCfg(G1DwaqSlopeNoSysDMatchedAgentCfg):
    experiment_name: str = "g1_dwaq_slope_nosys_d_matched_v2"
    wandb_project: str = "g1_dwaq_slope_nosys_d_matched_v2"
    run_name: str = "dwaq_nosys_matched_no_idle"
    resume: bool = False


@configclass
class G1DwaqSlopeNoSysDMatchedV3AgentCfg(G1DwaqSlopeNoSysDMatchedAgentCfg):
    experiment_name: str = "g1_dwaq_slope_nosys_d_matched_v3"
    wandb_project: str = "g1_dwaq_slope_nosys_d_matched_v3"
    run_name: str = "dwaq_nosys_matched_no_swing_height"
    resume: bool = False
