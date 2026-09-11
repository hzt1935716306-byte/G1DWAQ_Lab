"""Actor-visible certificate-context ablations for matched Plane V1."""

from isaaclab.utils import configclass

from legged_lab.envs.g1.g1_reward_shaping_ablation_config import (
    G1PlaneV1EstimatorContextNoRewardMatchedV2AgentCfg,
    G1PlaneV1EstimatorContextNoRewardMatchedV2EnvCfg,
)


@configclass
class G1PlaneV1EstimatorContextNoRewardMatchedV2NValidEnvCfg(
    G1PlaneV1EstimatorContextNoRewardMatchedV2EnvCfg
):
    actor_recovery_context_fields: tuple[str, ...] = ("n_norm", "valid")


@configclass
class G1PlaneV1EstimatorContextNoRewardMatchedV2MarginValidEnvCfg(
    G1PlaneV1EstimatorContextNoRewardMatchedV2EnvCfg
):
    actor_recovery_context_fields: tuple[str, ...] = ("margin_norm", "valid")


@configclass
class G1PlaneV1EstimatorContextNoRewardMatchedV2NValidAgentCfg(
    G1PlaneV1EstimatorContextNoRewardMatchedV2AgentCfg
):
    experiment_name: str = "g1_plane_v1_estimator_context_no_reward_matched_v2_n_valid"
    wandb_project: str = "g1_plane_v1_estimator_context_no_reward_matched_v2_n_valid"
    run_name: str = "context_ablation_v2_n_valid"
    resume: bool = False


@configclass
class G1PlaneV1EstimatorContextNoRewardMatchedV2MarginValidAgentCfg(
    G1PlaneV1EstimatorContextNoRewardMatchedV2AgentCfg
):
    experiment_name: str = "g1_plane_v1_estimator_context_no_reward_matched_v2_margin_valid"
    wandb_project: str = "g1_plane_v1_estimator_context_no_reward_matched_v2_margin_valid"
    run_name: str = "context_ablation_v2_margin_valid"
    resume: bool = False


__all__ = [
    name
    for name in globals()
    if name.startswith("G1PlaneV1EstimatorContextNoRewardMatchedV2")
]
