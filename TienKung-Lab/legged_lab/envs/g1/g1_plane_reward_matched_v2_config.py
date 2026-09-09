"""Reward-on Plane ablation; retain the original recoverability reward config."""
from isaaclab.utils import configclass
from legged_lab.envs.g1.g1_plane_v1_matched_config import (
    G1PlaneV1EstimatorContextRewardMatchedEnvCfg,
    G1PlaneV1EstimatorContextRewardMatchedAgentCfg,
)
from legged_lab.envs.g1.g1_reward_shaping_ablation_config import G1PlaneContextIdleSwingRewardCfg


@configclass
class G1PlaneV1EstimatorContextRewardMatchedV2EnvCfg(G1PlaneV1EstimatorContextRewardMatchedEnvCfg):
    reward = G1PlaneContextIdleSwingRewardCfg()


@configclass
class G1PlaneV1EstimatorContextRewardMatchedV2AgentCfg(G1PlaneV1EstimatorContextRewardMatchedAgentCfg):
    experiment_name: str = 'g1_plane_v1_estimator_context_reward_matched_v2'
    wandb_project: str = 'g1_plane_v1_estimator_context_reward_matched_v2'
    run_name: str = 'estimator_context_reward_matched_v2_idle01_swing02'
    resume: bool = False
