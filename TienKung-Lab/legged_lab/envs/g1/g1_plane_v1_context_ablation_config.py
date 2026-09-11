"""Actor-visible certificate-context ablations for matched Plane V1."""

from isaaclab.utils import configclass

from legged_lab.envs.g1.g1_plane_v1_matched_config import (
    G1PlaneV1EstimatorContextNoRewardMatchedAgentCfg,
    G1PlaneV1EstimatorContextNoRewardMatchedEnvCfg,
)


@configclass
class G1PlaneV1EstimatorContextNoRewardMatchedNValidEnvCfg(
    G1PlaneV1EstimatorContextNoRewardMatchedEnvCfg
):
    actor_recovery_context_fields: tuple[str, ...] = ("n_norm", "valid")


@configclass
class G1PlaneV1EstimatorContextNoRewardMatchedMarginValidEnvCfg(
    G1PlaneV1EstimatorContextNoRewardMatchedEnvCfg
):
    actor_recovery_context_fields: tuple[str, ...] = ("margin_norm", "valid")


@configclass
class G1PlaneV1EstimatorContextNoRewardMatchedNValidAgentCfg(
    G1PlaneV1EstimatorContextNoRewardMatchedAgentCfg
):
    experiment_name: str = "g1_plane_v1_estimator_context_no_reward_matched_n_valid"
    wandb_project: str = "g1_plane_v1_estimator_context_no_reward_matched_n_valid"
    run_name: str = "context_ablation_n_valid"
    resume: bool = False


@configclass
class G1PlaneV1EstimatorContextNoRewardMatchedMarginValidAgentCfg(
    G1PlaneV1EstimatorContextNoRewardMatchedAgentCfg
):
    experiment_name: str = "g1_plane_v1_estimator_context_no_reward_matched_margin_valid"
    wandb_project: str = "g1_plane_v1_estimator_context_no_reward_matched_margin_valid"
    run_name: str = "context_ablation_margin_valid"
    resume: bool = False


__all__ = [name for name in globals() if name.startswith("G1PlaneV1EstimatorContextNoRewardMatched")]
