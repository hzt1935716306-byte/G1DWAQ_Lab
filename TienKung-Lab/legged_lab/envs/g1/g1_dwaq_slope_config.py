"""G1 DWAQ configuration for the flat-ground and slope curriculum."""

from isaaclab.utils import configclass

from legged_lab.envs.g1.g1_dwaq_config import G1DwaqAgentCfg, G1DwaqEnvCfg
from legged_lab.terrains import G1_DWAQ_SLOPE_TERRAINS_CFG


@configclass
class G1DwaqSlopeEnvCfg(G1DwaqEnvCfg):
    """Reuse g1_dwaq unchanged, except for its terrain curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_generator = G1_DWAQ_SLOPE_TERRAINS_CFG


@configclass
class G1DwaqSlopeAgentCfg(G1DwaqAgentCfg):
    """Store slope-curriculum runs separately from regular g1_dwaq runs."""

    experiment_name: str = "g1_dwaq_slope"
    wandb_project: str = "g1_dwaq_slope"
