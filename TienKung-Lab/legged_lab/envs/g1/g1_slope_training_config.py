"""G1 flat/uphill/downhill training ablations without a push curriculum."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.envs.g1.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1FlatSymmetricAgentCfg,
    G1FlatSymmetricEnvCfg,
)
from legged_lab.envs.g1.g1_dwaq_nosys_config import (
    G1DwaqNoSysAgentCfg,
    G1DwaqNoSysEnvCfg,
)
from legged_lab.envs.g1.g1_recovery_config import (
    G1FlatSymmetricRecoveryEnvCfg,
    G1PushCurriculumCfg,
    G1RecoveryContextCfg,
    G1Stage2RewardCfg,
)
from legged_lab.terrains import G1_DWAQ_SLOPE_TERRAINS_CFG


_FIXED_PUSH_RANGE = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}


def _configure_slope_curriculum_without_turning(cfg) -> None:
    """Select the shared slope terrain and retain only translational commands."""

    cfg.scene.terrain_type = "generator"
    cfg.scene.terrain_generator = G1_DWAQ_SLOPE_TERRAINS_CFG
    cfg.commands.heading_command = False
    cfg.commands.rel_heading_envs = 0.0
    cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.ranges.heading = None


def _enable_fixed_velocity_jump(cfg) -> None:
    """Enable the full velocity-jump range from the start of training."""

    cfg.domain_rand.events.push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": _FIXED_PUSH_RANGE},
    )


@configclass
class G1SlopeNoSysDEnvCfg(G1FlatEnvCfg):
    """Plain G1 policy with slope curriculum and fixed random pushes."""

    def __post_init__(self):
        super().__post_init__()
        _configure_slope_curriculum_without_turning(self)
        _enable_fixed_velocity_jump(self)


@configclass
class G1SlopeNoSysDAgentCfg(G1FlatAgentCfg):
    experiment_name: str = "g1_slope_nosys_d"
    wandb_project: str = "g1_slope_nosys_d"
    max_iterations: int = 10000


@configclass
class G1SlopeSysNdEnvCfg(G1FlatSymmetricRecoveryEnvCfg):
    """Symmetric policy with a reserved, permanently zero 3-D actor context."""

    push_curriculum: G1PushCurriculumCfg = G1PushCurriculumCfg(
        enable_push_curriculum=False,
        adaptive_upgrades_enabled=False,
        easy_sample_probability=0.0,
    )
    stage2_reward: G1Stage2RewardCfg = G1Stage2RewardCfg(enabled=False)
    recovery_context: G1RecoveryContextCfg = G1RecoveryContextCfg(
        enabled=True,
        mode="zero",
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_slope_curriculum_without_turning(self)
        self.domain_rand.events.push_robot = None


@configclass
class G1SlopeSysNdAgentCfg(G1FlatSymmetricAgentCfg):
    experiment_name: str = "g1_slope_sys_nd"
    wandb_project: str = "g1_slope_sys_nd"
    max_iterations: int = 10000


@configclass
class G1SlopeSysDEnvCfg(G1FlatSymmetricEnvCfg):
    """Symmetric policy with slope curriculum and fixed random pushes."""

    def __post_init__(self):
        super().__post_init__()
        _configure_slope_curriculum_without_turning(self)
        _enable_fixed_velocity_jump(self)


@configclass
class G1SlopeSysDAgentCfg(G1FlatSymmetricAgentCfg):
    experiment_name: str = "g1_slope_sys_d"
    wandb_project: str = "g1_slope_sys_d"
    max_iterations: int = 10000


@configclass
class G1DwaqSlopeNoSysDEnvCfg(G1DwaqNoSysEnvCfg):
    """No-system DWAQ policy with slope curriculum and fixed random pushes."""

    def __post_init__(self):
        super().__post_init__()
        _configure_slope_curriculum_without_turning(self)
        _enable_fixed_velocity_jump(self)


@configclass
class G1DwaqSlopeNoSysDAgentCfg(G1DwaqNoSysAgentCfg):
    experiment_name: str = "g1_dwaq_slope_nosys_d"
    wandb_project: str = "g1_dwaq_slope_nosys_d"
    max_iterations: int = 10000


__all__ = [
    "G1DwaqSlopeNoSysDAgentCfg",
    "G1DwaqSlopeNoSysDEnvCfg",
    "G1SlopeNoSysDAgentCfg",
    "G1SlopeNoSysDEnvCfg",
    "G1SlopeSysDAgentCfg",
    "G1SlopeSysDEnvCfg",
    "G1SlopeSysNdAgentCfg",
    "G1SlopeSysNdEnvCfg",
]
