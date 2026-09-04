# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.tienkung.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from legged_lab.envs.tienkung.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)

from legged_lab.envs.g1.g1_env import G1Env
from legged_lab.envs.g1.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1FlatSymmetricAgentCfg,
    G1FlatSymmetricEnvCfg,
    G1RoughAgentCfg,
    G1RoughEnvCfg,
)
from legged_lab.envs.g1.g1_recovery_config import (
    G1FlatSymmetricRecoveryAgentCfg,
    G1FlatSymmetricRecoveryEnvCfg,
    G1FlatSymmetricStage2BaselineAgentCfg,
    G1FlatSymmetricStage2BaselineEnvCfg,
    G1FlatSymmetricStage2OursAgentCfg,
    G1FlatSymmetricStage2OursEnvCfg,
)
from legged_lab.envs.g1.g1_recovery_env import G1RecoveryEnv
from legged_lab.envs.g1.g1_plane_recovery_config import (
    G1PlaneSymmetricStage2BaselineAgentCfg,
    G1PlaneSymmetricStage2BaselineEnvCfg,
    G1PlaneSymmetricStage2OursAgentCfg,
    G1PlaneSymmetricStage2OursEnvCfg,
)
from legged_lab.envs.g1.g1_plane_recovery_env import G1PlaneRecoveryEnv
from legged_lab.envs.g1.g1_plane_v1_config import (
    G1PlaneV1EstimatorContextNoRewardAgentCfg,
    G1PlaneV1EstimatorContextNoRewardEnvCfg,
    G1PlaneV1EstimatorContextRewardAgentCfg,
    G1PlaneV1EstimatorContextRewardEnvCfg,
    G1PlaneV1PrivilegedContextNoRewardAgentCfg,
    G1PlaneV1PrivilegedContextNoRewardEnvCfg,
    G1PlaneV1PrivilegedContextRewardAgentCfg,
    G1PlaneV1PrivilegedContextRewardEnvCfg,
)
from legged_lab.envs.g1.g1_plane_v1_env import G1PlaneV1Env

from legged_lab.envs.g1.g1_dwaq_env import G1DwaqEnv
from legged_lab.envs.g1.g1_dwaq_config import (
    G1DwaqAgentCfg,
    G1DwaqEnvCfg,
)
from legged_lab.envs.g1.g1_dwaq_nosys_config import (
    G1DwaqNoSysAgentCfg,
    G1DwaqNoSysEnvCfg,
)
from legged_lab.envs.g1.g1_dwaq_slope_config import (
    G1DwaqSlopeAgentCfg,
    G1DwaqSlopeEnvCfg,
)
from legged_lab.envs.g1.g1_slope_training_config import (
    G1DwaqSlopeNoSysDAgentCfg,
    G1DwaqSlopeNoSysDEnvCfg,
    G1SlopeNoSysDAgentCfg,
    G1SlopeNoSysDEnvCfg,
    G1SlopeSysDAgentCfg,
    G1SlopeSysDEnvCfg,
    G1SlopeSysNdAgentCfg,
    G1SlopeSysNdEnvCfg,
)
from legged_lab.envs.g1.g1_com_velocity_estimator_config import (
    G1ComVelocityEstimatorAgentCfg,
    G1ComVelocityEstimatorEnvCfg,
    G1ComVelocityEstimatorV2AgentCfg,
    G1ComVelocityEstimatorV2EnvCfg,
)

from legged_lab.envs.h1.h1_config import (
    H1FlatAgentCfg,
    H1FlatEnvCfg,
    H1RoughAgentCfg,
    H1RoughEnvCfg,
)
from legged_lab.utils.task_registry import task_registry


task_registry.register("walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungWalkAgentCfg())
task_registry.register("run", TienKungEnv, TienKungRunFlatEnvCfg(), TienKungRunAgentCfg())
task_registry.register(
    "walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg(), TienKungWalkWithSensorAgentCfg()
)
task_registry.register(
    "run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg(), TienKungRunWithSensorAgentCfg()
)
task_registry.register("h1_flat", BaseEnv, H1FlatEnvCfg(), H1FlatAgentCfg())
task_registry.register("h1_rough", BaseEnv, H1RoughEnvCfg(), H1RoughAgentCfg())


task_registry.register("g1_flat", BaseEnv, G1FlatEnvCfg(), G1FlatAgentCfg())
task_registry.register(
    "g1_flat_symmetric", BaseEnv, G1FlatSymmetricEnvCfg(), G1FlatSymmetricAgentCfg()
)
task_registry.register(
    "g1_flat_symmetric_recovery",
    G1RecoveryEnv,
    G1FlatSymmetricRecoveryEnvCfg(),
    G1FlatSymmetricRecoveryAgentCfg(),
)
task_registry.register(
    "g1_flat_symmetric_stage2_baseline_old",
    G1RecoveryEnv,
    G1FlatSymmetricStage2BaselineEnvCfg(),
    G1FlatSymmetricStage2BaselineAgentCfg(),
)
task_registry.register(
    "g1_flat_symmetric_stage2_ours_old",
    G1RecoveryEnv,
    G1FlatSymmetricStage2OursEnvCfg(),
    G1FlatSymmetricStage2OursAgentCfg(),
)
# Deprecated aliases retained only for old scripts/checkpoints.  They still
# point to the LEGACY FLAT-ONLY IMPLEMENTATION and never to the plane tasks.
task_registry.register(
    "g1_flat_symmetric_stage2_baseline",
    G1RecoveryEnv,
    G1FlatSymmetricStage2BaselineEnvCfg(),
    G1FlatSymmetricStage2BaselineAgentCfg(),
)
task_registry.register(
    "g1_flat_symmetric_stage2_ours",
    G1RecoveryEnv,
    G1FlatSymmetricStage2OursEnvCfg(),
    G1FlatSymmetricStage2OursAgentCfg(),
)
task_registry.register(
    "g1_plane_symmetric_stage2_baseline",
    G1PlaneRecoveryEnv,
    G1PlaneSymmetricStage2BaselineEnvCfg(),
    G1PlaneSymmetricStage2BaselineAgentCfg(),
)
task_registry.register(
    "g1_plane_symmetric_stage2_ours",
    G1PlaneRecoveryEnv,
    G1PlaneSymmetricStage2OursEnvCfg(),
    G1PlaneSymmetricStage2OursAgentCfg(),
)
task_registry.register(
    "g1_plane_v1_estimator_context_no_reward",
    G1PlaneV1Env,
    G1PlaneV1EstimatorContextNoRewardEnvCfg(),
    G1PlaneV1EstimatorContextNoRewardAgentCfg(),
)
task_registry.register(
    "g1_plane_v1_estimator_context_reward",
    G1PlaneV1Env,
    G1PlaneV1EstimatorContextRewardEnvCfg(),
    G1PlaneV1EstimatorContextRewardAgentCfg(),
)
task_registry.register(
    "g1_plane_v1_privileged_context_no_reward",
    G1PlaneV1Env,
    G1PlaneV1PrivilegedContextNoRewardEnvCfg(),
    G1PlaneV1PrivilegedContextNoRewardAgentCfg(),
)
task_registry.register(
    "g1_plane_v1_privileged_context_reward",
    G1PlaneV1Env,
    G1PlaneV1PrivilegedContextRewardEnvCfg(),
    G1PlaneV1PrivilegedContextRewardAgentCfg(),
)
task_registry.register("g1_rough", G1Env, G1RoughEnvCfg(), G1RoughAgentCfg())
task_registry.register("g1_dwaq", G1DwaqEnv, G1DwaqEnvCfg(), G1DwaqAgentCfg())
task_registry.register(
    "g1_dwaq_slope",
    G1DwaqEnv,
    G1DwaqSlopeEnvCfg(),
    G1DwaqSlopeAgentCfg(),
)
task_registry.register(
    "g1_dwaq_nosys",
    G1DwaqEnv,
    G1DwaqNoSysEnvCfg(),
    G1DwaqNoSysAgentCfg(),
)
task_registry.register(
    "g1_slope_nosys_d",
    BaseEnv,
    G1SlopeNoSysDEnvCfg(),
    G1SlopeNoSysDAgentCfg(),
)
task_registry.register(
    "g1_slope_sys_nd",
    G1RecoveryEnv,
    G1SlopeSysNdEnvCfg(),
    G1SlopeSysNdAgentCfg(),
)
task_registry.register(
    "g1_slope_sys_d",
    BaseEnv,
    G1SlopeSysDEnvCfg(),
    G1SlopeSysDAgentCfg(),
)
task_registry.register(
    "g1_com_velocity_estimator",
    BaseEnv,
    G1ComVelocityEstimatorEnvCfg(),
    G1ComVelocityEstimatorAgentCfg(),
)
task_registry.register(
    "g1_com_velocity_estimator_v2",
    BaseEnv,
    G1ComVelocityEstimatorV2EnvCfg(),
    G1ComVelocityEstimatorV2AgentCfg(),
)
task_registry.register(
    "g1_dwaq_slope_nosys_d",
    G1DwaqEnv,
    G1DwaqSlopeNoSysDEnvCfg(),
    G1DwaqSlopeNoSysDAgentCfg(),
)
