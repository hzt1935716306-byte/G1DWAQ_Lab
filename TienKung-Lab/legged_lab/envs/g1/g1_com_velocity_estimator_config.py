"""Environment registration configs for standalone G1 CoM velocity estimators."""

from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass

from legged_lab.envs.g1.g1_slope_training_config import (
    G1SlopeSysDAgentCfg,
    G1SlopeSysDEnvCfg,
)


@configclass
class G1ComVelocityEstimatorEnvCfg(G1SlopeSysDEnvCfg):
    """The frozen teacher's environment, inherited without behavioural changes."""

    pass


@configclass
class G1ComVelocityEstimatorAgentCfg(G1SlopeSysDAgentCfg):
    """Teacher architecture used only to reconstruct and load the frozen policy."""

    experiment_name: str = "g1_com_velocity_estimator"
    wandb_project: str = "g1_com_velocity_estimator"


@configclass
class G1ComVelocityEstimatorV2EnvCfg(G1SlopeSysDEnvCfg):
    """Frozen-teacher environment augmented only with the deployable pelvis IMU."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            offset=ImuCfg.OffsetCfg(pos=(0.04525, 0.0, -0.08339)),
            update_period=self.sim.decimation * self.sim.dt,
            gravity_bias=(0.0, 0.0, 9.81),
        )


@configclass
class G1ComVelocityEstimatorV2AgentCfg(G1SlopeSysDAgentCfg):
    """Unchanged teacher architecture used while collecting V2 supervision."""

    experiment_name: str = "g1_com_velocity_estimator_v2"
    wandb_project: str = "g1_com_velocity_estimator_v2"


__all__ = [
    "G1ComVelocityEstimatorAgentCfg",
    "G1ComVelocityEstimatorEnvCfg",
    "G1ComVelocityEstimatorV2AgentCfg",
    "G1ComVelocityEstimatorV2EnvCfg",
]
