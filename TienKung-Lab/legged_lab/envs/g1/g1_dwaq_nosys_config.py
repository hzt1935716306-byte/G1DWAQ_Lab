"""G1 DWAQ configuration without a fixed gait-phase system."""

from isaaclab.utils import configclass

from legged_lab.envs.g1.g1_dwaq_config import G1DwaqAgentCfg, G1DwaqEnvCfg, G1DwaqRewardCfg
from legged_lab.terrains import G1_DWAQ_SLOPE_TERRAINS_CFG


@configclass
class G1DwaqNoSysRewardCfg(G1DwaqRewardCfg):
    """Use the DWAQ rewards without phase/contact tracking."""

    gait_phase_contact = None


@configclass
class G1DwaqNoSysEnvCfg(G1DwaqEnvCfg):
    """Use a learned gait timing on the flat-ground and slope curriculum."""

    reward = G1DwaqNoSysRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_generator = G1_DWAQ_SLOPE_TERRAINS_CFG
        self.robot.gait_phase.enable = False


@configclass
class G1DwaqNoSysAgentCfg(G1DwaqAgentCfg):
    """Store no-phase-system runs separately from regular DWAQ runs."""

    experiment_name: str = "g1_dwaq_nosys"
    wandb_project: str = "g1_dwaq_nosys"
