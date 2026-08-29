# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Stage2 symmetric flat-ground recovery curriculum configuration."""

from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.envs.g1.g1_config import G1FlatSymmetricAgentCfg, G1FlatSymmetricEnvCfg
from legged_lab.recovery.stage2_reward import DEFAULT_EVENT_SCALE


_DEFAULT_CERTIFICATE_PARAMETERS = str(
    Path(__file__).resolve().parents[3] / "tools/recovery/generated/g1_recovery_params.yaml"
)


@configclass
class G1PushCurriculumCfg:
    """Configurable six-level curriculum ending at the Stage1B push range."""

    enable_push_curriculum: bool = True
    adaptive_upgrades_enabled: bool = True
    level_ratios: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
    stage1b_abs_delta_v_xy: tuple[float, float] = (1.0, 1.0)
    k_min_iterations: int = 500
    k_max_iterations: int = 1800
    statistics_window_episodes: int = 500
    p5_threshold: float = 0.85
    median_enter_step_threshold: float = 4.0
    required_consecutive_pass_windows: int = 2
    easy_sample_probability: float = 0.20
    num_steps_per_iteration: int = 24

    # Reuse the already validated practical-gait definition.  These are the
    # Gate 1 thresholds recorded in the current recovery validation report.
    mean_velocity_error_threshold: float = 0.14522231240183686
    mean_abs_roll_threshold: float = 0.02584471284877509
    mean_abs_pitch_threshold: float = 0.042438490772619845
    max_recovery_touchdowns: int = 5

    # Legacy RewardManager path.  Keep it off because Stage2 one-shot events
    # are injected after RewardManager.compute and must not receive dt scaling.
    recovery_reward_weight: float = 0.0


@configclass
class G1Stage2RewardCfg:
    """Shared Stage2 reward and recovery-only soft-constraint settings."""

    enabled: bool = False
    enable_certificate_reward: bool = False
    event_scale: float = DEFAULT_EVENT_SCALE
    certificate_parameters_path: str = _DEFAULT_CERTIFICATE_PARAMETERS
    certificate_workers: int = 16
    certificate_failure_window_size: int = 4096
    certificate_failure_rate_threshold: float = 0.01
    soft_reward_min_multipliers: dict[str, float] = {
        "joint_deviation_arms": 0.25,
        "joint_deviation_hip": 0.40,
        "joint_deviation_legs": 0.50,
        "action_rate_l2": 0.50,
        "energy": 0.50,
        "dof_acc_l2": 0.50,
        "feet_air_time_symmetry": 0.25,
        "feet_contact_time_symmetry": 0.25,
        "feet_sagittal_symmetry": 0.25,
        "body_orientation_l2": 0.70,
        "flat_orientation_l2": 0.70,
    }


@configclass
class G1FlatSymmetricRecoveryEnvCfg(G1FlatSymmetricEnvCfg):
    """Stage2: Stage1A policy plus velocity-jump curriculum and recovery logging."""

    push_curriculum: G1PushCurriculumCfg = G1PushCurriculumCfg()
    stage2_reward: G1Stage2RewardCfg = G1Stage2RewardCfg()

    def __post_init__(self):
        super().__post_init__()

        # Keep exactly the same plane setup as Stage 1A.
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None

        # Keep the Stage1B physical event form and interval.  Only the sampled
        # component-wise maximum is selected by the curriculum wrapper.
        self.domain_rand.events.push_robot = EventTerm(
            func=mdp.curriculum_push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={},
        )


@configclass
class G1FlatSymmetricRecoveryAgentCfg(G1FlatSymmetricAgentCfg):
    """Reuse the Stage 1A policy architecture and log root for checkpoint compatibility."""

    experiment_name: str = "g1_flat_symmetric"
    run_name: str = "stage2_push_curriculum"


@configclass
class G1FlatSymmetricStage2BaselineEnvCfg(G1FlatSymmetricRecoveryEnvCfg):
    """Shared Stage2 task with generic event rewards and no certificate progress."""

    stage2_reward: G1Stage2RewardCfg = G1Stage2RewardCfg(
        enabled=True,
        enable_certificate_reward=False,
    )


@configclass
class G1FlatSymmetricStage2OursEnvCfg(G1FlatSymmetricRecoveryEnvCfg):
    """The identical Stage2 task with the certificate potential channel enabled."""

    stage2_reward: G1Stage2RewardCfg = G1Stage2RewardCfg(
        enabled=True,
        enable_certificate_reward=True,
    )


@configclass
class G1FlatSymmetricStage2BaselineAgentCfg(G1FlatSymmetricRecoveryAgentCfg):
    run_name: str = "stage2_baseline"


@configclass
class G1FlatSymmetricStage2OursAgentCfg(G1FlatSymmetricRecoveryAgentCfg):
    run_name: str = "stage2_ours"
