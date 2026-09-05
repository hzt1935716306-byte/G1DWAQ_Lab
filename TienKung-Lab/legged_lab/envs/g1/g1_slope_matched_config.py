"""Strict ±15 degree continuous-plane counterparts of the three baselines."""

from isaaclab.utils import configclass

from legged_lab.envs.g1.g1_slope_training_config import (
    G1DwaqSlopeNoSysDAgentCfg,
    G1DwaqSlopeNoSysDEnvCfg,
    G1SlopeNoSysDAgentCfg,
    G1SlopeNoSysDEnvCfg,
    G1SlopeSysDAgentCfg,
    G1SlopeSysDEnvCfg,
)
from legged_lab.recovery.baseline_matched_protocol import (
    CURRICULUM_REFERENCE_TILE_LENGTH,
    configure_matched_command_and_reset,
)
from legged_lab.terrains import make_plane_baseline_matched_terrain_cfg


def _configure_matched_plane(cfg) -> None:
    cfg.scene.terrain_type = "generator"
    cfg.scene.terrain_generator = make_plane_baseline_matched_terrain_cfg(
        int(cfg.scene.seed)
    )
    cfg.scene.max_init_terrain_level = 5
    cfg.curriculum_reference_tile_length = CURRICULUM_REFERENCE_TILE_LENGTH

    configure_matched_command_and_reset(cfg)


@configclass
class G1SlopeNoSysDMatchedEnvCfg(G1SlopeNoSysDEnvCfg):
    curriculum_reference_tile_length: float = CURRICULUM_REFERENCE_TILE_LENGTH

    def __post_init__(self):
        super().__post_init__()
        _configure_matched_plane(self)


@configclass
class G1SlopeSysDMatchedEnvCfg(G1SlopeSysDEnvCfg):
    curriculum_reference_tile_length: float = CURRICULUM_REFERENCE_TILE_LENGTH

    def __post_init__(self):
        super().__post_init__()
        _configure_matched_plane(self)


@configclass
class G1DwaqSlopeNoSysDMatchedEnvCfg(G1DwaqSlopeNoSysDEnvCfg):
    curriculum_reference_tile_length: float = CURRICULUM_REFERENCE_TILE_LENGTH

    def __post_init__(self):
        super().__post_init__()
        _configure_matched_plane(self)


@configclass
class G1SlopeNoSysDMatchedAgentCfg(G1SlopeNoSysDAgentCfg):
    experiment_name: str = "g1_slope_nosys_d_matched"
    wandb_project: str = "g1_slope_nosys_d_matched"


@configclass
class G1SlopeSysDMatchedAgentCfg(G1SlopeSysDAgentCfg):
    experiment_name: str = "g1_slope_sys_d_matched"
    wandb_project: str = "g1_slope_sys_d_matched"


@configclass
class G1DwaqSlopeNoSysDMatchedAgentCfg(G1DwaqSlopeNoSysDAgentCfg):
    experiment_name: str = "g1_dwaq_slope_nosys_d_matched"
    wandb_project: str = "g1_dwaq_slope_nosys_d_matched"


__all__ = [name for name in globals() if "Matched" in name]
