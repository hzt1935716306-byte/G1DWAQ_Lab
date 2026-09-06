"""RL-only ablation for the baseline-matched Plane V1 training protocol."""

from isaaclab.utils import configclass

from legged_lab.envs.g1.g1_plane_v1_matched_config import (
    G1PlaneV1BaselineMatchedAgentCfg,
)
from legged_lab.envs.g1.g1_slope_matched_config import G1SlopeSysDMatchedEnvCfg


@configclass
class G1PlaneV1RLOnlyMatchedEnvCfg(G1SlopeSysDMatchedEnvCfg):
    """Five-frame symmetric locomotion policy without recoverability machinery."""

    def __post_init__(self):
        super().__post_init__()

        # Match the Plane V1 temporal inputs while keeping the ordinary BaseEnv
        # observation path: 5 * 96 = 480 policy inputs and 10 * 101 = 1010
        # critic inputs. No three-value context is appended.
        self.robot.actor_obs_history_length = 5
        self.robot.critic_obs_history_length = 10

        # Match the final Plane V1 ordinary locomotion objective exactly.
        self.reward.track_lin_vel_xy_exp.weight = 2.0
        self.reward.track_ang_vel_z_exp.weight = 2.0
        self.reward.joint_deviation_hip.weight = -0.30


@configclass
class G1PlaneV1RLOnlyMatchedAgentCfg(G1PlaneV1BaselineMatchedAgentCfg):
    """Reuse the exact Plane V1 matched PPO and symmetry configuration."""

    run_name: str = "rl_only_matched"


__all__ = [
    "G1PlaneV1RLOnlyMatchedAgentCfg",
    "G1PlaneV1RLOnlyMatchedEnvCfg",
]
