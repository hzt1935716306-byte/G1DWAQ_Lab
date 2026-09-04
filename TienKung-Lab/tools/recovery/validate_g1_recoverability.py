#!/usr/bin/env python3
"""Run nominal and disturbed recoverability Gates for the fixed Stage1B G1 policy."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import os
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS = PROJECT_DIR / "tools/recovery/generated/g1_recovery_params.yaml"
DEFAULT_REPORT = PROJECT_DIR / "tools/recovery/generated/g1_recoverability_report.yaml"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="g1_flat_symmetric_recovery")
parser.add_argument("--validation_mode", choices=("flat", "plane"), default="flat")
parser.add_argument(
    "--plane_nominal_params",
    type=Path,
    default=PROJECT_DIR / "tools/recovery/generated/g1_plane_validation_frozen_policy_params.yaml",
)
parser.add_argument("--slope_degrees", type=float, default=0.0)
parser.add_argument(
    "--runner_type", choices=("auto", "on_policy", "dwaq"), default="auto"
)
parser.add_argument(
    "--plane_repeats",
    type=int,
    default=10,
    help="Repeats per slope/push-direction/magnitude/phase cell in plane mode.",
)
parser.add_argument(
    "--spearman_bootstrap_resamples",
    type=int,
    default=2000,
    help="Bootstrap resamples for the two formal Gate B terminal correlations.",
)
parser.add_argument(
    "--estimator_diagnostic",
    action="store_true",
    help=(
        "Measure the frozen DWAQ mean-velocity head and recompute plane certificates "
        "with only the CoM velocity contribution replaced."
    ),
)
parser.add_argument(
    "--estimator_certificate_diagnostic",
    action="store_true",
    help=(
        "Recompute paired plane certificates after replacing only true whole-body "
        "CoM velocity XY with the frozen standalone 5x96 estimator output."
    ),
)
parser.add_argument(
    "--com_velocity_estimator_checkpoint",
    type=Path,
    default=(
        PROJECT_DIR
        / "logs/g1_com_velocity_estimator/2026-09-04_19-51-28/com_velocity_estimator_best.pt"
    ),
)
parser.add_argument(
    "--estimator_nominal_frames",
    type=int,
    default=1000,
    help="Valid env-frame samples retained for nominal estimator statistics.",
)
parser.add_argument(
    "--estimator_repeats",
    type=int,
    default=2,
    help="Recovery repeats per direction/magnitude/phase in estimator diagnostic mode.",
)
parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
parser.add_argument("--checkpoint_path", type=Path, default=None)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--warmup_steps", type=int, default=250)
parser.add_argument("--nominal_touchdowns_per_speed", type=int, default=30)
parser.add_argument(
    "--plane_nominal_sampling_envs",
    type=int,
    default=2,
    help=(
        "Number of env trajectories used for plane nominal sanity. Keeping this "
        "small avoids treating synchronized deterministic clones as independent gait samples."
    ),
)
parser.add_argument("--max_gate1_steps", type=int, default=12000)
parser.add_argument("--max_gate1_over_horizon_fraction", type=float, default=0.10)
parser.add_argument(
    "--reuse_gate1_report",
    type=Path,
    default=None,
    help="Reuse Gate 1 thresholds from a prior report for large disturbed-test batches.",
)
parser.add_argument("--validation_speed", type=float, default=0.6)
parser.add_argument("--push_levels", type=float, nargs="+", default=(0.5, 1.0, 1.25, 1.5))
parser.add_argument("--direction_count", type=int, default=4)
parser.add_argument("--push_phases", type=float, nargs="+", default=(0.0, 0.5))
parser.add_argument("--trials_per_condition", type=int, default=1)
parser.add_argument("--trial_timeout_s", type=float, default=8.0)
parser.add_argument(
    "--random_trials",
    type=int,
    default=0,
    help="Run this many randomized trials instead of the fixed push grid.",
)
parser.add_argument(
    "--in_range_fraction",
    type=float,
    default=0.90,
    help="Fraction of randomized trials inside the trained per-axis push range.",
)
parser.add_argument(
    "--trained_abs_delta_v_xy",
    type=float,
    default=1.0,
    help="Absolute per-axis velocity-jump limit used during training.",
)
parser.add_argument(
    "--outside_abs_delta_v_xy",
    type=float,
    default=1.5,
    help="Absolute per-axis sampling limit for out-of-training-range trials.",
)
parser.add_argument(
    "--random_command_speed_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Random forward command range; defaults to the calibrated command range.",
)
parser.add_argument(
    "--certificate_workers",
    type=int,
    default=min(16, os.cpu_count() or 1),
    help="CPU workers used to solve saved touchdown certificate queries after simulation.",
)
parser.add_argument(
    "--recovery_manager_validation",
    action="store_true",
    help=(
        "Keep each disturbed rollout through five new touchdowns (or a real fall), "
        "then replay its event/touchdown certificates through RecoveryManager."
    ),
)
parser.add_argument(
    "--q_memory_diagnostic",
    action="store_true",
    help="Continue diagnostic logging through TD8 without changing the TD5 state-machine timeout.",
)
parser.add_argument(
    "--q_memory_sample_count",
    type=int,
    default=30,
    help="Maximum TD5 N=1 TIMEOUT trajectories and nominal touchdowns used by the diagnostic.",
)
parser.add_argument(
    "--passive_log_touchdowns",
    type=int,
    default=5,
    help="Keep passive labels after the unchanged TD5 state-machine exit; allowed range is 5--8.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--output",
    type=Path,
    default=DEFAULT_REPORT,
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
from rsl_rl.runners import DWAQOnPolicyRunner, OnPolicyRunner  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.estimation.com_velocity_estimator import (  # noqa: E402
    ComVelocityEstimator,
    EstimatorFrameHistory,
    ResetWarmupMask,
    extract_recent_actor_history,
    latest_actor_frame,
)
from legged_lab.recovery.certificate import (  # noqa: E402
    CertificateState,
    HalfspaceRegion2D,
    RecoverabilityConfig,
    certify_recoverability,
    terminal_contains,
)
from legged_lab.recovery.dwaq_estimator_diagnostic import (  # noqa: E402
    certificate_agreement,
    dcm_velocity_error_statistics,
    query_with_replaced_com_velocity,
    terminal_ordering,
    velocity_error_statistics,
)
from legged_lab.recovery.plane_certificate_runtime import (  # noqa: E402
    PlaneCalibratedG1CertificateEvaluator,
    PlaneCertificateQuery,
)
from legged_lab.recovery.plane_nominal_params import (  # noqa: E402
    PRACTICAL_METRIC_INTERVAL_MEAN_V1,
    PlaneNominalParameterTable,
)
from legged_lab.recovery.recovery_manager import (  # noqa: E402
    RecoveryExitReason,
    RecoveryManager,
)
from legged_lab.recovery.state_extractor import (  # noqa: E402
    G1PrivilegedStateExtractor,
    G1StateExtractorCfg,
    theoretical_periodic_state,
)
from legged_lab.terrains import make_plane_recovery_terrain_cfg  # noqa: E402
from legged_lab.utils import task_registry  # noqa: E402


def _native(value):
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _load_yaml(path: Path) -> dict:
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _disable_evaluation_randomization(env_cfg):
    events = env_cfg.domain_rand.events
    events.push_robot = None
    events.physics_material = None
    events.add_base_mass = None
    for name in ("randomize_dome_light", "randomize_distant_light"):
        if hasattr(events, name):
            setattr(events, name, None)
    for key in events.reset_base.params["pose_range"]:
        events.reset_base.params["pose_range"][key] = (0.0, 0.0)
    for key in events.reset_base.params["velocity_range"]:
        events.reset_base.params["velocity_range"][key] = (0.0, 0.0)
    events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def _make_env_policy(parameters: dict):
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_cfg.scene.num_envs = max(args.num_envs, len(parameters["provenance"]["commands_vx"]))
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.scene.terrain_type = "plane"
    env_cfg.scene.terrain_generator = None
    env_cfg.noise.add_noise = False
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.debug_vis = False
    env_cfg.commands.resampling_time_range = (1.0e9, 1.0e9)
    command_speeds = parameters["provenance"]["commands_vx"]
    env_cfg.commands.ranges.lin_vel_x = (min(command_speeds), max(command_speeds))
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = None
    _disable_evaluation_randomization(env_cfg)

    env = task_registry.get_task_class(args.task)(env_cfg, args.headless)
    checkpoint = (
        args.checkpoint_path.expanduser().resolve()
        if args.checkpoint_path is not None
        else Path(parameters["provenance"]["checkpoint"]).expanduser().resolve()
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    return env, runner.get_inference_policy(device=env.device), checkpoint


def _disable_plane_evaluation_randomization(env_cfg) -> None:
    """Match the frozen-policy collector while leaving the flat path unchanged."""

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.action_delay.enable = False
    events = env_cfg.domain_rand.events
    for name in (
        "push_robot",
        "physics_material",
        "add_base_mass",
        "randomize_actuator_gains",
        "randomize_com",
        "randomize_dome_light",
        "randomize_distant_light",
    ):
        if hasattr(events, name):
            setattr(events, name, None)
    for key in events.reset_base.params["pose_range"]:
        events.reset_base.params["pose_range"][key] = (0.0, 0.0)
    for key in events.reset_base.params["velocity_range"]:
        events.reset_base.params["velocity_range"][key] = (0.0, 0.0)
    events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def _attach_validation_plane_provider(env, slope_degrees: float) -> None:
    def provider(self):
        alpha = torch.full(
            (self.num_envs,),
            math.radians(slope_degrees),
            dtype=torch.float32,
            device=self.device,
        )
        normal = torch.stack(
            (-torch.sin(alpha), torch.zeros_like(alpha), torch.cos(alpha)), dim=-1
        )
        valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return normal, self.scene.env_origins.clone(), valid

    env.get_recovery_plane_geometry = MethodType(provider, env)


def _make_plane_env_policy():
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.seed = args.seed
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.scene.terrain_type = "generator"
    env_cfg.scene.terrain_generator = make_plane_recovery_terrain_cfg(
        (args.slope_degrees,)
    )
    env_cfg.scene.max_init_terrain_level = 0
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.debug_vis = False
    env_cfg.commands.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.ranges.lin_vel_x = (0.4, 0.4)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = None
    _disable_plane_evaluation_randomization(env_cfg)
    if args.estimator_certificate_diagnostic:
        # Match standalone-estimator training: the five actor frames include
        # BaseEnv's configured observation noise and normalization/scaling.
        env_cfg.noise.add_noise = True
    if hasattr(args, "device"):
        env_cfg.device = args.device
        agent_cfg.device = args.device

    checkpoint = args.checkpoint_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    env = task_registry.get_task_class(args.task)(env_cfg, args.headless)
    _attach_validation_plane_provider(env, args.slope_degrees)
    runner_type = args.runner_type
    if runner_type == "auto":
        runner_type = "dwaq" if "dwaq" in args.task.lower() else "on_policy"
    if runner_type == "dwaq":
        runner = DWAQOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(checkpoint), load_optimizer=False)
        runner.eval_mode()
        runner.alg.policy.requires_grad_(False)
        policy = runner.alg.policy.act_inference
    else:
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(checkpoint), load_optimizer=False)
        runner.eval_mode()
        runner.alg.policy.requires_grad_(False)
        policy = runner.get_inference_policy(device=env.device)
    locomotion_policy = runner.alg.policy
    actor_linear = [
        module for module in locomotion_policy.actor if isinstance(module, torch.nn.Linear)
    ]
    if not actor_linear:
        raise RuntimeError("frozen locomotion policy has no actor Linear layers")
    current_actor_obs, _ = env.compute_current_observations()
    policy_contract = {
        "actor_input_dim": int(actor_linear[0].in_features),
        "actor_history_length": int(env.cfg.robot.actor_obs_history_length),
        "per_frame_actor_observation_dim": int(current_actor_obs.shape[1]),
        "action_dim": int(actor_linear[-1].out_features),
        "eval_mode": not bool(locomotion_policy.training),
        "requires_grad": any(
            parameter.requires_grad for parameter in locomotion_policy.parameters()
        ),
    }
    expected_contract = {
        "actor_input_dim": 960,
        "actor_history_length": 10,
        "per_frame_actor_observation_dim": 96,
        "action_dim": 29,
        "eval_mode": True,
        "requires_grad": False,
    }
    if policy_contract != expected_contract:
        raise RuntimeError(
            f"frozen locomotion policy contract mismatch: "
            f"{policy_contract} != {expected_contract}"
        )
    return env, policy, checkpoint, runner_type, runner, policy_contract


def _set_plane_commands(env) -> None:
    env.command_generator.command[:, 0] = 0.4
    env.command_generator.command[:, 1:] = 0.0
    env.command_generator.is_standing_env[:] = False
    env.command_generator.is_heading_env[:] = False


def _plane_policy_step(env, policy, runner_type, obs, obs_hist):
    with torch.inference_mode():
        actions = policy(obs, obs_hist) if runner_type == "dwaq" else policy(obs)
        obs, _, dones, extras = env.step(actions)
        if runner_type == "dwaq":
            obs_hist = extras["observations"]["obs_hist"]
    return obs, obs_hist, dones


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _estimator_sensitivity_enabled() -> bool:
    return bool(args.estimator_diagnostic or args.estimator_certificate_diagnostic)


def _load_com_velocity_estimator(
    env,
    locomotion_runner,
    locomotion_checkpoint: Path,
) -> tuple[ComVelocityEstimator, ResetWarmupMask, EstimatorFrameHistory | None, dict]:
    checkpoint = args.com_velocity_estimator_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CoM velocity estimator checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location=env.device, weights_only=False)
    input_dim = int(payload.get("input_dim", -1))
    is_v2 = input_dim == 495
    expected = {
        "input_dim": 495 if is_v2 else 480,
        "per_frame_obs_dim": 99 if is_v2 else 96,
        "history_length": 5,
        "hidden_dims": [256, 128, 64],
        "output_dim": 2,
        "output_frame": "heading",
        "output_quantity": "whole_body_com_velocity_xy",
        "output_unit": "m/s",
    }
    mismatches = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"CoM estimator semantic contract mismatch: {mismatches}")
    input_history = None
    if is_v2:
        v2_expected = {
            "actor_per_frame_obs_dim": 96,
            "imu_input_dim": 3,
            "imu_quantity": "deployable_pelvis_specific_force_body_xyz",
        }
        v2_mismatches = {
            key: {"actual": payload.get(key), "expected": value}
            for key, value in v2_expected.items()
            if payload.get(key) != value
        }
        if v2_mismatches:
            raise RuntimeError(f"V2 CoM estimator semantic mismatch: {v2_mismatches}")
        if "imu" not in env.scene.sensors:
            raise RuntimeError("V2 CoM estimator requires the deployable pelvis IMU task")
        input_history = EstimatorFrameHistory(
            env.num_envs,
            history_length=5,
            actor_frame_dim=96,
            imu_dim=3,
            imu_acceleration_scale=float(payload["imu_acceleration_scale"]),
            device=env.device,
        )
    locomotion_hash = _sha256(locomotion_checkpoint)
    if payload.get("teacher_checkpoint_hash") != locomotion_hash:
        raise RuntimeError(
            "CoM estimator was not trained from this frozen locomotion checkpoint: "
            f"{payload.get('teacher_checkpoint_hash')} != {locomotion_hash}"
        )
    model = ComVelocityEstimator(
        input_dim=payload["input_dim"],
        hidden_dims=payload["hidden_dims"],
        output_dim=payload["output_dim"],
    ).to(env.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)

    locomotion_policy = locomotion_runner.alg.policy
    actor_linear = [
        module for module in locomotion_policy.actor if isinstance(module, torch.nn.Linear)
    ]
    current_actor_obs, _ = env.compute_current_observations()
    teacher_contract = {
        "actor_input_dim": int(actor_linear[0].in_features),
        "actor_history_length": int(env.cfg.robot.actor_obs_history_length),
        "per_frame_actor_observation_dim": int(current_actor_obs.shape[1]),
        "action_dim": int(actor_linear[-1].out_features),
    }
    expected_teacher = {
        "actor_input_dim": 960,
        "actor_history_length": 10,
        "per_frame_actor_observation_dim": 96,
        "action_dim": 29,
    }
    if teacher_contract != expected_teacher:
        raise RuntimeError(
            f"frozen locomotion teacher contract mismatch: {teacher_contract} != {expected_teacher}"
        )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CoM estimator must be frozen in eval mode")
    if locomotion_policy.training or any(
        parameter.requires_grad for parameter in locomotion_policy.parameters()
    ):
        raise RuntimeError("locomotion policy must be frozen in eval mode")
    metadata = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_training_step": int(payload.get("training_step", -1)),
        "strict_state_dict_load": True,
        "eval_mode": True,
        "requires_grad": False,
        "estimator_version": "V2" if is_v2 else "V1",
        "architecture": [input_dim, 256, 128, 64, 2],
        "input_semantics": (
            "latest 5 frames of [scaled/noisy 96-D Actor proprioception, deployable pelvis "
            "IMU specific-force xyz] in oldest-to-newest order"
            if is_v2
            else "latest 5 frames from BaseEnv actor history; each frame is scaled/noisy "
            "96-D actor proprioception in oldest-to-newest buffer order"
        ),
        "imu_acceleration_scale": payload.get("imu_acceleration_scale") if is_v2 else None,
        "reset_warmup_policy_steps": 5,
        "output_frame": payload["output_frame"],
        "output_quantity": payload["output_quantity"],
        "output_unit": payload["output_unit"],
        "teacher_checkpoint_hash": payload["teacher_checkpoint_hash"],
        "locomotion_checkpoint_hash": locomotion_hash,
        "teacher_contract": teacher_contract,
        "observation_noise_enabled": bool(env.cfg.noise.add_noise),
        "only_replaced_certificate_quantity": "whole_body_com_velocity_xy",
    }
    return model, ResetWarmupMask(env.num_envs, 5, env.device), input_history, metadata


def _com_velocity_estimator_tensors(
    env,
    model: ComVelocityEstimator,
    state,
    actor_observations: torch.Tensor,
    reset_mask: torch.Tensor,
    input_history: EstimatorFrameHistory | None,
) -> dict[str, torch.Tensor]:
    if input_history is None:
        estimator_input = extract_recent_actor_history(
            actor_observations,
            teacher_history_length=10,
            estimator_history_length=5,
            per_frame_obs_dim=96,
        )
    else:
        estimator_input = input_history.append(
            latest_actor_frame(actor_observations),
            env.scene.sensors["imu"].data.lin_acc_b,
            reset_mask,
        )
    with torch.inference_mode():
        estimate_xy = model(estimator_input)
    # The trained estimator has no z output.  z is copied from GT only so the
    # existing 3-D velocity report remains well formed; certificates use XY only.
    estimate_heading = torch.cat((estimate_xy, state.com_velocity[:, 2:3]), dim=1)
    return {
        "direct_com_est_heading": estimate_heading,
        "com_GT_heading": state.com_velocity,
    }


def _dwaq_estimator_metadata(env, runner, checkpoint: Path) -> dict:
    policy = runner.alg.policy
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state = state["model_state_dict"]
    velocity_keys = sorted(
        key
        for key in model_state
        if key.startswith("encoder.")
        or key.startswith("encode_mean_vel.")
        or key.startswith("encode_logvar_vel.")
    )
    encoder_input = int(policy.encoder[0].in_features)
    observation_dimension = int(policy.obs_dim)
    if encoder_input % observation_dimension != 0:
        raise RuntimeError(
            f"encoder input {encoder_input} is not divisible by obs_dim {observation_dimension}"
        )
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_iteration": int(state.get("iter", -1)),
        "checkpoint_keys": sorted(state),
        "velocity_weight_keys": velocity_keys,
        "weight_shapes": {
            key: list(model_state[key].shape) for key in velocity_keys
        },
        "observation_dimension": observation_dimension,
        "history_length": encoder_input // observation_dimension,
        "encoder_input_dimension": encoder_input,
        "encoder_output_dimension": int(policy.encoder[2].out_features),
        "velocity_head_output_dimension": int(policy.encode_mean_vel.out_features),
        "deterministic_estimator_path": "encode_mean_vel(encoder(obs_history))",
        "stochastic_code_vel_used_for_diagnostic": False,
        "actor_inference_unchanged": True,
        "observation_normalization": {
            "runner_empirical_normalization": bool(runner.empirical_normalization),
            "checkpoint_has_obs_normalizer": "obs_norm_state_dict" in state,
            "diagnostic_history_preprocessing": "raw env obs_history (same frozen DWAQ pipeline)",
        },
        "velocity_target": {
            "source": "critic_obs[:, obs_dim:obs_dim+3] = root_lin_vel_b * obs_scales.lin_vel",
            "frame": "root/base body frame",
            "quantity": "root rigid-body CoM linear velocity, not whole-robot CoM velocity",
            "obs_scales_lin_vel": float(env.obs_scales.lin_vel),
            "physical_output_conversion": "mean_vel / obs_scales.lin_vel",
            "time_alignment": "latest obs_history frame and same post-step simulator state",
        },
    }


def _dwaq_velocity_tensors(env, runner, state, obs_hist) -> dict[str, torch.Tensor]:
    """Return deterministic estimator/GT velocities at the same post-step instant."""

    policy = runner.alg.policy
    with torch.inference_mode():
        encoded = policy.encoder(obs_hist)
        estimate_scaled_body = policy.encode_mean_vel(encoded)
    lin_vel_scale = float(env.obs_scales.lin_vel)
    if not math.isfinite(lin_vel_scale) or lin_vel_scale == 0.0:
        raise RuntimeError(f"invalid obs_scales.lin_vel={lin_vel_scale}")
    estimate_body = estimate_scaled_body / lin_vel_scale
    base_gt_body = env.robot.data.root_lin_vel_b
    estimate_world = quat_apply(env.robot.data.root_quat_w, estimate_body)
    base_gt_world = quat_apply(env.robot.data.root_quat_w, base_gt_body)
    estimate_heading = quat_apply_inverse(state.heading_quat_w, estimate_world)
    base_gt_heading = quat_apply_inverse(state.heading_quat_w, base_gt_world)
    return {
        "base_est_body": estimate_body,
        "base_GT_body": base_gt_body,
        "direct_com_est_heading": estimate_heading,
        "base_GT_heading": base_gt_heading,
        "com_GT_heading": state.com_velocity,
    }


def _velocity_row(tensors: dict[str, torch.Tensor], env_id: int, omega: float) -> dict:
    return {
        key: value[env_id].detach().cpu().numpy().astype(np.float64)
        for key, value in tensors.items()
    } | {"omega": float(omega)}


def _velocity_accuracy_report(rows: list[dict]) -> dict:
    if not rows:
        return {"sample_count": 0}

    def stack(key: str) -> np.ndarray:
        return np.stack([row[key] for row in rows], axis=0)

    direct = stack("direct_com_est_heading")
    com_gt = stack("com_GT_heading")
    report = {
        "sample_count": len(rows),
        "estimated_whole_body_com_heading_frame": velocity_error_statistics(
            direct, com_gt
        ),
        "DCM_velocity_induced_error": dcm_velocity_error_statistics(
            direct[:, :2],
            com_gt[:, :2],
            [row["omega"] for row in rows],
        ),
    }
    if all("base_est_body" in row for row in rows):
        report.update(
            {
                "base_estimator_body_frame": velocity_error_statistics(
                    stack("base_est_body"), stack("base_GT_body")
                ),
                "direct_base_estimate_as_whole_body_com_heading_frame": report[
                    "estimated_whole_body_com_heading_frame"
                ],
                "GT_base_vs_GT_whole_body_com_heading_frame": velocity_error_statistics(
                    stack("base_GT_heading"), com_gt
                ),
            }
        )
    return report


def _set_commands(env, speeds):
    command = env.command_generator.command
    command[:, 0] = torch.as_tensor(speeds, device=env.device)
    command[:, 1:] = 0.0
    env.command_generator.is_standing_env[:] = False


def _period_at(parameters: dict, speed: float) -> float:
    model = parameters["step_period"]
    if model["type"] == "linear":
        return max(0.10, float(model["intercept"] + model["slope"] * speed))
    return float(model["value"])


def _bounds(region: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    return tuple(region["x"]), tuple(region["y"])


def _certificate_config(parameters: dict, speed: float) -> RecoverabilityConfig:
    period = _period_at(parameters, speed)
    omega = float(parameters["omega"])
    theory = theoretical_periodic_state(speed, 0.0, period, omega, float(parameters["w"]))
    c_left_x, c_left_y = _bounds(parameters["C_left"])
    c_right_x, c_right_y = _bounds(parameters["C_right"])
    l_left_x, l_left_y = _bounds(parameters["L_left"])
    l_right_x, l_right_y = _bounds(parameters["L_right"])
    return RecoverabilityConfig(
        gravity=9.81,
        h_eff=float(parameters["h_eff"]),
        step_period=period,
        max_steps=5,
        cop_left=HalfspaceRegion2D.box(c_left_x, c_left_y),
        cop_right=HalfspaceRegion2D.box(c_right_x, c_right_y),
        landing_left=HalfspaceRegion2D.box(l_left_x, l_left_y),
        landing_right=HalfspaceRegion2D.box(l_right_x, l_right_y),
        swing_velocity_limits=(float(parameters["v_max"]["x"]), float(parameters["v_max"]["y"])),
        nominal_cop_left=(0.0, 0.0),
        nominal_cop_right=(0.0, 0.0),
        nominal_step_left=theory["landing_left"],
        nominal_step_right=theory["landing_right"],
        nominal_b_left=theory["b_left"],
        nominal_b_right=theory["b_right"],
        nominal_q_left=theory["q_left"],
        nominal_q_right=theory["q_right"],
        epsilon_b=(float(parameters["epsilon_b"]["x"]), float(parameters["epsilon_b"]["y"])),
        epsilon_q=(float(parameters["epsilon_q"]["x"]), float(parameters["epsilon_q"]["y"])),
    )


def _support_position(state, env_id: int) -> torch.Tensor:
    return (
        state.left_foot_position[env_id]
        if bool(state.support_is_left[env_id].item())
        else state.right_foot_position[env_id]
    )


def _certificate_query(state, env_id: int, parameters: dict, delta_v_heading: np.ndarray | None = None):
    speed = float(state.command_velocity[env_id, 0].item())
    omega = float(parameters["omega"])
    support = _support_position(state, env_id)
    b = (
        state.com_position[env_id, :2]
        + state.com_velocity[env_id, :2] / omega
        - support[:2]
    ).detach().cpu().numpy()
    if delta_v_heading is not None:
        b = b + np.asarray(delta_v_heading) / omega
    q = state.q[env_id].detach().cpu().numpy()
    side = "left" if bool(state.support_is_left[env_id].item()) else "right"
    return {
        "command_vx": speed,
        "b": b,
        "q": q,
        "support_side": side,
        "phase": float(state.phase[env_id].item()),
    }


def _solve_certificate_query(query: dict, parameters: dict):
    config = _certificate_config(parameters, float(query["command_vx"]))
    return certify_recoverability(
        CertificateState(
            b=np.asarray(query["b"]),
            q=np.asarray(query["q"]),
            support_side=query["support_side"],
            phase=float(query["phase"]),
            step_period=config.step_period,
            omega=config.omega,
        ),
        config,
    )


def _annotate_q_memory_sample(sample: dict, query: dict, parameters: dict) -> None:
    """Attach model errors without changing the certificate or its inputs."""
    config = _certificate_config(parameters, float(query["command_vx"]))
    nominal_b, nominal_q = config.terminal_nominal(query["support_side"])
    b_real = np.asarray(query["b"], dtype=np.float64)
    q_real = np.asarray(query["q"], dtype=np.float64)
    e_b = b_real - nominal_b
    e_q = q_real - nominal_q
    # At touchdown q_k is the old support foot relative to the new support,
    # hence q_k = -l_{k-1} in the certificate sign convention.
    previous_landing_actual = -q_real
    previous_landing_nominal = -nominal_q
    previous_landing_error = previous_landing_actual - previous_landing_nominal
    sample.update(
        {
            "e_b_xy": e_b,
            "e_q_xy": e_q,
            "previous_landing_actual_xy": previous_landing_actual,
            "previous_landing_nominal_xy": previous_landing_nominal,
            "previous_landing_error_xy": previous_landing_error,
            "q_plus_previous_landing_error_xy": e_q + previous_landing_error,
        }
    )


def _certify(state, env_id: int, parameters: dict, delta_v_heading: np.ndarray | None = None):
    return _solve_certificate_query(
        _certificate_query(state, env_id, parameters, delta_v_heading),
        parameters,
    )


def _per_step_metric(state, env_id: int) -> tuple[float, np.ndarray]:
    velocity_error = torch.linalg.vector_norm(
        state.com_velocity[env_id, :2] - state.command_velocity[env_id, :2]
    )
    return float(velocity_error.item()), np.abs(state.root_roll_pitch[env_id].detach().cpu().numpy())


def _touchdown_diagnostic_snapshot(state, env_id: int) -> dict:
    velocity_error = (
        state.com_velocity[env_id, :2] - state.command_velocity[env_id, :2]
    ).detach().cpu().numpy()
    return {
        "velocity_tracking_error_xy": velocity_error,
        "root_roll_pitch": state.root_roll_pitch[env_id].detach().cpu().numpy(),
        "good_cycle": None,
        "practical_entered": False,
        "practical_confirmed": False,
        "practical_enter_step": None,
        "practical_confirmed_step": None,
    }


def _gate1(env, policy, extractor, parameters):
    commands = [float(value) for value in parameters["provenance"]["commands_vx"]]
    assigned = np.asarray([commands[index % len(commands)] for index in range(env.num_envs)])
    targets = {speed: args.nominal_touchdowns_per_speed for speed in commands}
    counts = {speed: 0 for speed in commands}
    n_counts = {label: 0 for label in (0, 1, 2, 3, 4, 5, 6)}
    margins = []
    solver_failures = 0
    cycle_velocity_error = []
    cycle_abs_roll_pitch = []
    last_touchdown_side: list[int | None] = [None] * env.num_envs
    interval_velocity: list[list[float]] = [[] for _ in range(env.num_envs)]
    interval_tilt: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]

    _set_commands(env, assigned)
    obs, _ = env.get_observations()
    for step in range(args.max_gate1_steps):
        _set_commands(env, assigned)
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            state = extractor.extract()

        for env_id in range(env.num_envs):
            velocity_error, tilt = _per_step_metric(state, env_id)
            interval_velocity[env_id].append(velocity_error)
            interval_tilt[env_id].append(tilt)

        for env_id in (dones | state.episode_reset).nonzero(as_tuple=False).flatten().tolist():
            last_touchdown_side[env_id] = None
            interval_velocity[env_id].clear()
            interval_tilt[env_id].clear()

        for env_id in state.touchdown.nonzero(as_tuple=False).flatten().tolist():
            side = int(state.touchdown_foot[env_id].item())
            alternating = last_touchdown_side[env_id] is not None and side != last_touchdown_side[env_id]
            if step >= args.warmup_steps and alternating and interval_velocity[env_id]:
                cycle_velocity_error.append(float(np.mean(interval_velocity[env_id])))
                cycle_abs_roll_pitch.append(np.mean(np.asarray(interval_tilt[env_id]), axis=0))
            interval_velocity[env_id].clear()
            interval_tilt[env_id].clear()
            last_touchdown_side[env_id] = side

            speed = float(assigned[env_id])
            if step >= args.warmup_steps and counts[speed] < targets[speed]:
                result = _certify(state, env_id, parameters)
                if result.n_min is None or result.margin is None:
                    solver_failures += 1
                else:
                    n_counts[int(result.n_min)] += 1
                    margins.append(float(result.margin))
                counts[speed] += 1
        if step > 0 and step % 100 == 0:
            print(f"[INFO] Gate 1 step={step}, touchdown_counts={counts}", flush=True)
        if all(counts[speed] >= targets[speed] for speed in commands):
            break
    else:
        print(f"[WARN] Gate 1 reached {args.max_gate1_steps} steps: counts={counts}")

    if not cycle_velocity_error or not cycle_abs_roll_pitch:
        raise RuntimeError("No complete alternating nominal cycles were collected for recovery thresholds")
    cycle_velocity_error = np.asarray(cycle_velocity_error)
    cycle_abs_roll_pitch = np.asarray(cycle_abs_roll_pitch)
    thresholds = {
        "mean_velocity_error": float(np.quantile(cycle_velocity_error, 0.95) * 1.25 + 0.05),
        "mean_abs_roll": float(np.quantile(cycle_abs_roll_pitch[:, 0], 0.95) * 1.25 + 0.01),
        "mean_abs_pitch": float(np.quantile(cycle_abs_roll_pitch[:, 1], 0.95) * 1.25 + 0.01),
        "derivation": "1.25 * nominal complete-cycle p95 + small numerical allowance",
        "baseline_cycle_count": int(cycle_velocity_error.size),
    }
    total = sum(n_counts.values())
    fractions = {str(key) if key < 6 else ">5": value / max(total, 1) for key, value in n_counts.items()}
    over_fraction = fractions[">5"]
    passed = total > 0 and solver_failures == 0 and over_fraction <= args.max_gate1_over_horizon_fraction
    report = {
        "passed": passed,
        "criterion": f"P(N>5) <= {args.max_gate1_over_horizon_fraction} and no solver failures",
        "touchdowns_by_command": {str(key): value for key, value in counts.items()},
        "N_min_count": {str(key) if key < 6 else ">5": value for key, value in n_counts.items()},
        "N_min_probability": fractions,
        "margin": {
            "count": len(margins),
            "mean": float(np.mean(margins)) if margins else None,
            "median": float(np.median(margins)) if margins else None,
            "p05": float(np.quantile(margins, 0.05)) if margins else None,
            "p95": float(np.quantile(margins, 0.95)) if margins else None,
            "min": float(np.min(margins)) if margins else None,
            "max": float(np.max(margins)) if margins else None,
        },
        "solver_failures": solver_failures,
        "actual_recovery_thresholds": thresholds,
        "baseline_cycle_statistics": {
            "mean_velocity_error_p50": float(np.median(cycle_velocity_error)),
            "mean_velocity_error_p95": float(np.quantile(cycle_velocity_error, 0.95)),
            "mean_abs_roll_p95": float(np.quantile(cycle_abs_roll_pitch[:, 0], 0.95)),
            "mean_abs_pitch_p95": float(np.quantile(cycle_abs_roll_pitch[:, 1], 0.95)),
        },
    }
    return report, thresholds, obs


def _random_trial_plans(parameters: dict) -> list[dict]:
    if args.random_trials <= 0:
        return []
    if not 0.0 <= args.in_range_fraction <= 1.0:
        raise ValueError("in_range_fraction must be in [0, 1]")
    if args.trained_abs_delta_v_xy <= 0.0:
        raise ValueError("trained_abs_delta_v_xy must be positive")
    if args.outside_abs_delta_v_xy <= args.trained_abs_delta_v_xy:
        raise ValueError("outside_abs_delta_v_xy must exceed trained_abs_delta_v_xy")

    calibrated_speeds = [float(value) for value in parameters["provenance"]["commands_vx"]]
    speed_min, speed_max = (
        tuple(args.random_command_speed_range)
        if args.random_command_speed_range is not None
        else (min(calibrated_speeds), max(calibrated_speeds))
    )
    if not 0.0 < speed_min <= speed_max:
        raise ValueError("random command speeds must satisfy 0 < MIN <= MAX")

    rng = np.random.default_rng(args.seed)
    in_range_count = int(round(args.random_trials * args.in_range_fraction))
    scopes = np.asarray(
        ["in_training_range"] * in_range_count
        + ["outside_training_range"] * (args.random_trials - in_range_count),
        dtype=object,
    )
    rng.shuffle(scopes)
    plans = []
    for trial_index, scope in enumerate(scopes.tolist()):
        if scope == "in_training_range":
            delta = rng.uniform(
                -args.trained_abs_delta_v_xy,
                args.trained_abs_delta_v_xy,
                size=2,
            )
        else:
            # Sample from the larger square conditioned on at least one axis
            # exceeding the trained component-wise limit.
            while True:
                delta = rng.uniform(
                    -args.outside_abs_delta_v_xy,
                    args.outside_abs_delta_v_xy,
                    size=2,
                )
                if np.any(np.abs(delta) > args.trained_abs_delta_v_xy):
                    break
        level = math.hypot(float(delta[0]), float(delta[1]))
        plans.append(
            {
                "trial_index": trial_index,
                "push_scope": scope,
                "level": level,
                "direction_index": None,
                "angle_rad": float(math.atan2(delta[1], delta[0])),
                "delta_v_heading_xy": delta,
                "target_phase": float(rng.uniform(0.0, 1.0)),
                "command_vx": float(rng.uniform(speed_min, speed_max)),
                "repeat": 0,
            }
        )
    return plans


def _trial_plans(parameters: dict) -> list[dict]:
    if args.random_trials > 0:
        return _random_trial_plans(parameters)
    plans = []
    for level in args.push_levels:
        for direction_index in range(args.direction_count):
            angle = 2.0 * math.pi * direction_index / args.direction_count
            for phase in args.push_phases:
                if not 0.0 <= phase < 1.0:
                    raise ValueError(f"push phase must be in [0, 1): {phase}")
                for repeat in range(args.trials_per_condition):
                    plans.append(
                        {
                            "level": float(level),
                            "direction_index": direction_index,
                            "angle_rad": angle,
                            "target_phase": float(phase),
                            "repeat": repeat,
                        }
                    )
    rng = np.random.default_rng(args.seed)
    rng.shuffle(plans)
    return plans


def _spearman_summary(
    first: np.ndarray,
    second: np.ndarray,
    bootstrap_resamples: int = 0,
) -> dict:
    if np.unique(first).size < 2 or np.unique(second).size < 2:
        return {
            "rho": None,
            "p_value": None,
            "reason": "undefined because at least one variable is constant",
        }
    rho, p_value = spearmanr(first, second)
    result = {"rho": float(rho), "p_value": float(p_value)}
    if bootstrap_resamples > 0:
        rng = np.random.default_rng(args.seed + 104729)
        values = []
        for _ in range(bootstrap_resamples):
            indices = rng.integers(0, len(first), size=len(first))
            first_sample = first[indices]
            second_sample = second[indices]
            if np.unique(first_sample).size < 2 or np.unique(second_sample).size < 2:
                continue
            sample_rho, _ = spearmanr(first_sample, second_sample)
            if np.isfinite(sample_rho):
                values.append(float(sample_rho))
        result["bootstrap_95_CI"] = (
            [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if values
            else None
        )
        result["bootstrap_requested_resamples"] = int(bootstrap_resamples)
        result["bootstrap_valid_resamples"] = len(values)
    return result


def _start_trial(env, state, env_id: int, plan: dict, parameters: dict, step: int) -> dict:
    delta_heading = np.asarray(
        plan.get(
            "delta_v_heading_xy",
            (plan["level"] * math.cos(plan["angle_rad"]), plan["level"] * math.sin(plan["angle_rad"])),
        ),
        dtype=np.float64,
    )
    initial_query = _certificate_query(state, env_id, parameters, delta_heading)
    delta_heading_tensor = torch.tensor(
        [[delta_heading[0], delta_heading[1], 0.0]], dtype=torch.float32, device=env.device
    )
    delta_world = quat_apply(state.heading_quat_w[env_id].unsqueeze(0), delta_heading_tensor)[0]
    ids = torch.tensor([env_id], dtype=torch.long, device=env.device)
    root_velocity = env.robot.data.root_vel_w[ids].clone()
    root_velocity[:, :2] += delta_world[:2]
    env.robot.write_root_velocity_to_sim(root_velocity, env_ids=ids)
    starts_at_touchdown = bool(plan["target_phase"] == 0.0 and state.touchdown[env_id].item())
    support_foot = 0 if bool(state.support_is_left[env_id].item()) else 1
    start_time = float(state.time[env_id].item())
    return {
        **plan,
        "env_id": env_id,
        "start_step": step,
        "start_time": start_time,
        "applied_phase": float(state.phase[env_id].item()),
        "delta_v_heading_xy": delta_heading,
        "delta_v_world_xy": delta_world[:2].detach().cpu().numpy(),
        "N_theory": None,
        "margin": None,
        "certificate_status": "pending",
        "success": False,
        "practical_entered": False,
        "practical_enter_step": None,
        "N_actual": None,
        "N_confirmation": None,
        "t_recovery": None,
        "t_confirmation": None,
        "failure_reason": None,
        "touchdowns": 0,
        "last_touchdown_foot": support_foot if starts_at_touchdown else None,
        "interval_started_after_touchdown": starts_at_touchdown,
        "interval_start_touchdown": 0 if starts_at_touchdown else None,
        "interval_start_time": start_time if starts_at_touchdown else None,
        "interval_velocity": [],
        "interval_tilt": [],
        "consecutive_good_cycles": 0,
        "recovery_onset_touchdown": None,
        "recovery_onset_time": None,
        "cycle_metrics": [],
        "certificate_trace": [
            {
                "touchdown": 0,
                "time_after_push": 0.0,
                "N_theory": None,
                "margin": None,
                "good_cycle": None,
                "practical_entered": False,
                "practical_confirmed": False,
                "practical_enter_step": None,
                "practical_confirmed_step": None,
                "status": "pending",
                "_query": initial_query,
            }
        ],
    }


def _finalize_trial(trial: dict) -> dict:
    internal_fields = {
        "interval_velocity",
        "interval_tilt",
        "consecutive_good_cycles",
        "interval_started_after_touchdown",
        "interval_start_touchdown",
        "interval_start_time",
        "recovery_onset_touchdown",
        "recovery_onset_time",
    }
    finalized = {
        key: value
        for key, value in trial.items()
        if key not in internal_fields
    }
    if finalized["success"]:
        recovery_touchdown = int(finalized["N_actual"])
        finalized["certificate_trace"] = [
            {
                **sample,
                "N_actual_remaining": max(recovery_touchdown - int(sample["touchdown"]), 0),
            }
            for sample in finalized["certificate_trace"]
        ]
    return finalized


def _resolve_saved_certificates(trials: list[dict], parameters: dict) -> None:
    jobs = []
    for trial_index, trial in enumerate(trials):
        if args.recovery_manager_validation:
            trace_limit = 8 if args.q_memory_diagnostic else args.passive_log_touchdowns
            trial["certificate_trace"] = [
                sample for sample in trial["certificate_trace"] if sample["touchdown"] <= trace_limit
            ]
        elif trial["success"]:
            trial["certificate_trace"] = [
                sample
                for sample in trial["certificate_trace"]
                if sample["touchdown"] <= trial["N_actual"]
            ]
        else:
            trial["certificate_trace"] = trial["certificate_trace"][:1]
        for sample_index, sample in enumerate(trial["certificate_trace"]):
            jobs.append((trial_index, sample_index, sample["_query"]))

    print(
        f"[INFO] Solving {len(jobs)} saved certificate states with "
        f"{args.certificate_workers} CPU workers",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.certificate_workers) as executor:
        results = list(
            executor.map(
                lambda job: _solve_certificate_query(job[2], parameters),
                jobs,
            )
        )
    for (trial_index, sample_index, _), result in zip(jobs, results):
        sample = trials[trial_index]["certificate_trace"][sample_index]
        query = sample.pop("_query")
        sample.setdefault("support_side", query["support_side"])
        sample["N_theory"] = result.n_min
        sample["margin"] = result.margin
        sample["status"] = result.status.value
        sample["orbit_recovered"] = result.n_min == 0
        if args.q_memory_diagnostic:
            _annotate_q_memory_sample(sample, query, parameters)

    for trial in trials:
        initial = trial["certificate_trace"][0]
        trial["N_theory"] = initial["N_theory"]
        trial["margin"] = initial["margin"]
        trial["certificate_status"] = initial["status"]


def _resolve_nominal_diagnostic_samples(samples: list[dict], parameters: dict) -> None:
    for sample in samples:
        query = sample.pop("_query")
        result = _solve_certificate_query(query, parameters)
        sample["N_theory"] = result.n_min
        sample["margin"] = result.margin
        sample["status"] = result.status.value
        sample["orbit_recovered"] = result.n_min == 0
        _annotate_q_memory_sample(sample, query, parameters)


def _q_memory_diagnostic_summary(trials: list[dict], nominal_samples: list[dict]) -> dict:
    candidates = []
    for trial in sorted(trials, key=lambda item: int(item.get("trial_index", 0))):
        episode = trial.get("recovery_state_machine")
        if episode is None or episode["exit_reason"] != RecoveryExitReason.TIMEOUT.value:
            continue
        by_touchdown = {int(sample["touchdown"]): sample for sample in trial["certificate_trace"]}
        if by_touchdown.get(5, {}).get("N_theory") != 1:
            continue
        if not all(touchdown in by_touchdown for touchdown in range(5, 9)):
            continue
        candidates.append((trial, by_touchdown))
    selected = candidates[: args.q_memory_sample_count]
    nominal = nominal_samples[: args.q_memory_sample_count]
    if not selected:
        raise RuntimeError("q-memory diagnostic found no complete TD5 N=1 TIMEOUT trajectories")
    if not nominal:
        raise RuntimeError("q-memory diagnostic collected no nominal touchdown samples")

    def percentile_summary(values) -> dict:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "median": float(np.median(array)),
            "p95": float(np.quantile(array, 0.95)),
        }

    touchdown_summary = {}
    relation_residuals = []
    for touchdown in range(5, 9):
        samples = [by_touchdown[touchdown] for _, by_touchdown in selected]
        for sample in samples:
            relation_residuals.extend(np.abs(sample["q_plus_previous_landing_error_xy"]))
        velocity = np.abs(np.asarray([sample["velocity_tracking_error_xy"] for sample in samples]))
        tilt = np.abs(np.asarray([sample["root_roll_pitch"] for sample in samples]))
        valid_good = [sample["good_cycle"] for sample in samples if sample["good_cycle"] is not None]
        touchdown_summary[f"TD{touchdown}"] = {
            "abs_e_q_x": percentile_summary(
                [abs(float(sample["e_q_xy"][0])) for sample in samples]
            ),
            "P_N_equals_0": float(np.mean([sample["N_theory"] == 0 for sample in samples])),
            "N_distribution": {
                (str(value) if value < 6 else ">5"): sum(sample["N_theory"] == value for sample in samples)
                for value in sorted(set(int(sample["N_theory"]) for sample in samples))
            },
            "abs_velocity_tracking_error": {
                "x_median": float(np.median(velocity[:, 0])),
                "x_p95": float(np.quantile(velocity[:, 0], 0.95)),
                "y_median": float(np.median(velocity[:, 1])),
                "y_p95": float(np.quantile(velocity[:, 1], 0.95)),
            },
            "abs_root_tilt": {
                "roll_median": float(np.median(tilt[:, 0])),
                "roll_p95": float(np.quantile(tilt[:, 0], 0.95)),
                "pitch_median": float(np.median(tilt[:, 1])),
                "pitch_p95": float(np.quantile(tilt[:, 1], 0.95)),
            },
            "good_cycle_fraction": float(np.mean(valid_good)) if valid_good else None,
            "practical_entered_fraction": float(
                np.mean([sample["practical_entered"] for sample in samples])
            ),
            "practical_confirmed_fraction": float(
                np.mean([sample["practical_confirmed"] for sample in samples])
            ),
        }

    nominal_abs_e_q_x = [abs(float(sample["e_q_xy"][0])) for sample in nominal]
    enter_steps = [
        trial["practical_enter_step"]
        for trial, _ in selected
        if trial["practical_enter_step"] is not None
    ]
    confirmed_steps = [
        trial["N_confirmation"] for trial, _ in selected if trial["N_confirmation"] is not None
    ]
    trajectories = []
    for trial, by_touchdown in selected:
        trajectories.append(
            {
                "trial_index": trial.get("trial_index"),
                "push_scope": trial.get("push_scope"),
                "delta_v_heading_xy": trial["delta_v_heading_xy"],
                "applied_phase": trial["applied_phase"],
                "practical_enter_step": trial["practical_enter_step"],
                "practical_confirmed_step": trial["N_confirmation"],
                "samples": [by_touchdown[touchdown] for touchdown in range(5, 9)],
            }
        )
    return {
        "definition": (
            "Offline logging continuation after the unchanged formal TD5 TIMEOUT; "
            "RecoveryManager is not reactivated."
        ),
        "eligible_complete_TD5_N1_timeout_count": len(candidates),
        "selected_trial_count": len(selected),
        "nominal_sample_count": len(nominal),
        "nominal_abs_e_q_x": percentile_summary(nominal_abs_e_q_x),
        "touchdowns": touchdown_summary,
        "q_equals_negative_previous_landing_error_check": {
            "definition": "e_q_k + e_l_(k-1) should be zero",
            "median_abs_residual": float(np.median(relation_residuals)),
            "max_abs_residual": float(np.max(relation_residuals)),
        },
        "practical_enter_step_distribution": {
            str(value): enter_steps.count(value) for value in sorted(set(enter_steps))
        },
        "practical_confirmed_step_distribution": {
            str(value): confirmed_steps.count(value) for value in sorted(set(confirmed_steps))
        },
        "not_practical_confirmed_by_TD8_count": sum(
            trial["N_confirmation"] is None for trial, _ in selected
        ),
        "trajectories": trajectories,
    }


def _attach_recovery_state_machine(trials: list[dict]) -> dict:
    """Replay saved event/touchdown certificates through the minimal manager."""
    exit_counts = {reason.value: 0 for reason in RecoveryExitReason}
    confirmed_exit_counts = {reason.value: 0 for reason in RecoveryExitReason}
    orbit_exit_counts = {reason.value: 0 for reason in RecoveryExitReason}
    incomplete_count = 0
    alternating_checks = 0
    alternation_violations = 0
    duplicate_touchdowns = 0
    all_rewards_zero = True
    reset_failures = 0
    old_confirmed_timeout_new_entered_success = 0
    practical_success_without_orbit = 0
    practical_enter_step_distribution = {str(step): 0 for step in range(1, 6)}
    success_n_distribution = {"0": 0, "1": 0, ">=2": 0}
    timeout_n_distribution = {str(value): 0 for value in range(6)} | {">5": 0}

    for trial in trials:
        trace = trial["certificate_trace"]
        if not trace or trace[0]["N_theory"] is None or trace[0]["margin"] is None:
            trial["recovery_state_machine"] = None
            incomplete_count += 1
            continue

        manager = RecoveryManager(max_touchdowns=5)
        initial = trace[0]
        manager.on_push(
            n=int(initial["N_theory"]),
            margin=float(initial["margin"]),
            delta_v_xy=tuple(float(value) for value in trial["delta_v_heading_xy"]),
            phase_at_push=float(trial["applied_phase"]),
            policy_step=int(trial["start_step"]),
        )
        completed_episode = None
        confirmed_exit_reason = None
        orbit_exit_reason = None
        for sample in trace[1:]:
            if sample["N_theory"] is None or sample["margin"] is None:
                continue
            update = manager.on_touchdown(
                n=int(sample["N_theory"]),
                margin=float(sample["margin"]),
                practical_entered=bool(sample.get("practical_entered", False)),
                practical_confirmed=bool(sample.get("practical_confirmed", False)),
                touchdown_token=int(sample["touchdown"]),
                support_side=sample.get("support_side"),
                time_after_push=float(sample["time_after_push"]),
            )
            duplicate_touchdowns += int(update.duplicate_touchdown)
            if update.transition is not None and update.transition.support_alternating is not None:
                alternating_checks += 1
                alternation_violations += int(not update.transition.support_alternating)
            if update.completed_episode is not None:
                completed_episode = update.completed_episode
            if confirmed_exit_reason is None:
                if bool(sample.get("practical_confirmed", False)):
                    confirmed_exit_reason = RecoveryExitReason.SUCCESS
                elif int(sample["touchdown"]) >= 5:
                    confirmed_exit_reason = RecoveryExitReason.TIMEOUT
            if orbit_exit_reason is None:
                if int(sample["N_theory"]) == 0:
                    orbit_exit_reason = RecoveryExitReason.SUCCESS
                elif int(sample["touchdown"]) >= 5:
                    orbit_exit_reason = RecoveryExitReason.TIMEOUT

        if manager.recovery_active and trial.get("failure_reason") == "fall_or_illegal_contact":
            completed_episode = manager.on_fall()
        if confirmed_exit_reason is None and trial.get("failure_reason") == "fall_or_illegal_contact":
            confirmed_exit_reason = RecoveryExitReason.FALL
        if orbit_exit_reason is None and trial.get("failure_reason") == "fall_or_illegal_contact":
            orbit_exit_reason = RecoveryExitReason.FALL
        if completed_episode is None and manager.completed_episodes:
            completed_episode = manager.completed_episodes[-1]

        if completed_episode is None:
            trial["recovery_state_machine"] = None
            incomplete_count += 1
        else:
            manager.record_actual_recovery(
                enter_step=(int(trial["N_actual"]) if trial["N_actual"] is not None else None),
                confirmed_step=(
                    int(trial["N_confirmation"])
                    if trial["N_confirmation"] is not None
                    else None
                ),
                recovery_time=(
                    float(trial["t_recovery"])
                    if trial["t_recovery"] is not None
                    else None
                ),
                practical_enter_step=(
                    int(trial["practical_enter_step"])
                    if trial["practical_enter_step"] is not None
                    else None
                ),
                episode=completed_episode,
            )
            trial["recovery_state_machine"] = completed_episode.to_dict()
            exit_counts[completed_episode.exit_reason.value] += 1
            all_rewards_zero = all_rewards_zero and completed_episode.recovery_reward == 0.0
            practical_success_without_orbit += int(
                completed_episode.exit_reason is RecoveryExitReason.SUCCESS
                and not completed_episode.orbit_recovered
            )
            if confirmed_exit_reason is not None:
                confirmed_exit_counts[confirmed_exit_reason.value] += 1
                old_confirmed_timeout_new_entered_success += int(
                    confirmed_exit_reason is RecoveryExitReason.TIMEOUT
                    and completed_episode.exit_reason is RecoveryExitReason.SUCCESS
                )
            if orbit_exit_reason is not None:
                orbit_exit_counts[orbit_exit_reason.value] += 1
            trial["recovery_state_machine_comparison"] = {
                "old_practical_confirmed_exit_reason": (
                    confirmed_exit_reason.value if confirmed_exit_reason is not None else None
                ),
                "new_practical_entered_exit_reason": completed_episode.exit_reason.value,
            }
            if completed_episode.exit_reason is RecoveryExitReason.SUCCESS:
                assert completed_episode.practical_enter_step is not None
                practical_enter_step_distribution[str(completed_episode.practical_enter_step)] += 1
                exit_n = completed_episode.transitions[-1].n_current
                success_n_distribution["0" if exit_n == 0 else "1" if exit_n == 1 else ">=2"] += 1
            elif completed_episode.exit_reason is RecoveryExitReason.TIMEOUT:
                exit_n = completed_episode.transitions[-1].n_current
                timeout_n_distribution[str(exit_n) if exit_n <= 5 else ">5"] += 1

        manager.reset()
        reset_failures += int(manager.recovery_active or manager.current_episode is not None)

    return {
        "definition": (
            "Known velocity perturbation enters RECOVERY; certificates are evaluated at the "
            "push and once per new touchdown only."
        ),
        "success_condition": "first completed practical good gait cycle (practical_entered)",
        "timeout_condition": "five new touchdowns without practical_entered",
        "orbit_recovered_definition": "N_min == 0; logged independently and does not control exit",
        "exit_counts": exit_counts,
        "old_practical_confirmed_exit_counts_on_same_trajectories": confirmed_exit_counts,
        "orbit_exit_counts_on_same_trajectories": orbit_exit_counts,
        "old_confirmed_timeout_new_entered_success_count": (
            old_confirmed_timeout_new_entered_success
        ),
        "practical_success_without_orbit_count": practical_success_without_orbit,
        "practical_enter_step_distribution": practical_enter_step_distribution,
        "success_N_at_exit_distribution": success_n_distribution,
        "timeout_N_at_TD5_distribution": timeout_n_distribution,
        "practical_confirmed_final_count": sum(bool(trial["success"]) for trial in trials),
        "practical_confirmed_final_fraction": (
            sum(bool(trial["success"]) for trial in trials) / len(trials) if trials else None
        ),
        "incomplete_count": incomplete_count,
        "max_new_touchdowns": 5,
        "passive_logging_through_touchdown": (
            8 if args.q_memory_diagnostic else args.passive_log_touchdowns
        ),
        "touchdown_duplicate_count": duplicate_touchdowns,
        "support_alternation_check_count": alternating_checks,
        "support_alternation_violation_count": alternation_violations,
        "reset_state_failure_count": reset_failures,
        "all_recovery_rewards_zero": all_rewards_zero,
        "timeout_terminates_environment": False,
    }


def _save_raw_disturbed_trials(
    trials: list[dict], parameters: dict, thresholds: dict, nominal_samples: list[dict]
) -> Path:
    output = args.output.expanduser().resolve()
    raw_output = output.with_name(f"{output.stem}_raw{output.suffix}")
    raw_report = {
        "schema_version": 1,
        "parameters": str(args.params.expanduser().resolve()),
        "checkpoint": parameters["provenance"]["checkpoint"],
        "actual_recovery_thresholds": thresholds,
        "trial_count": len(trials),
        "trials": trials,
        "q_memory_nominal_samples": nominal_samples if args.q_memory_diagnostic else None,
    }
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with raw_output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(_native(raw_report), stream, sort_keys=False, allow_unicode=True)
    print(f"[INFO] Saved raw disturbed trials to {raw_output}", flush=True)
    return raw_output


def _gate2(env, policy, extractor, parameters, thresholds, obs):
    if args.q_memory_diagnostic and not args.recovery_manager_validation:
        raise ValueError("--q_memory_diagnostic requires --recovery_manager_validation")
    if args.q_memory_sample_count <= 0:
        raise ValueError("--q_memory_sample_count must be positive")
    if not 5 <= args.passive_log_touchdowns <= 8:
        raise ValueError("--passive_log_touchdowns must be between 5 and 8")
    plans = _trial_plans(parameters)
    pending = list(plans)
    worker_env_count = min(env.num_envs, len(plans))
    active: list[dict | None] = [None] * env.num_envs
    prepared: list[dict | None] = [None] * env.num_envs
    forced_reset_pending = [True] * env.num_envs
    prepared_since = np.full(env.num_envs, -1, dtype=np.int64)
    cooldown_until = np.full(env.num_envs, args.warmup_steps, dtype=np.int64)
    stable_touchdowns = np.zeros(env.num_envs, dtype=np.int64)
    last_phase = np.zeros(env.num_envs)
    completed = []
    nominal_diagnostic_samples = []
    nominal_sampled_envs = set()
    timeout_steps = int(math.ceil(args.trial_timeout_s / env.step_dt))
    assigned_speeds = np.full(env.num_envs, args.validation_speed)

    for step in range(args.max_gate1_steps + max(1000, len(plans) * timeout_steps)):
        _set_commands(env, assigned_speeds)
        step_start = time.perf_counter()
        with torch.inference_mode():
            if step == 0:
                print("[INFO] Gate 2 step 0: starting policy inference", flush=True)
            actions = policy(obs)
            policy_done = time.perf_counter()
            if step == 0:
                print(
                    f"[INFO] Gate 2 step 0: policy finished in {policy_done - step_start:.3f}s; "
                    "starting env.step",
                    flush=True,
                )
            obs, _, dones, _ = env.step(actions)
            env_done = time.perf_counter()
            if step == 0:
                print(
                    f"[INFO] Gate 2 step 0: env.step finished in {env_done - policy_done:.3f}s; "
                    "starting state extraction",
                    flush=True,
                )
            state = extractor.extract()
            extractor_done = time.perf_counter()
        # One batched transfer avoids thousands of synchronizing CUDA .item()
        # calls when validation is run with large environment counts.
        phase_values = state.phase.detach().cpu().numpy()
        if step < 3:
            print(
                f"[INFO] Gate 2 timing step={step}: policy={policy_done - step_start:.3f}s, "
                f"env_step={env_done - policy_done:.3f}s, "
                f"extractor={extractor_done - env_done:.3f}s",
                flush=True,
            )

        reset_mask = dones | state.episode_reset
        for env_id in reset_mask.nonzero(as_tuple=False).flatten().tolist():
            if active[env_id] is not None:
                active[env_id]["failure_reason"] = "fall_or_illegal_contact"
                completed.append(_finalize_trial(active[env_id]))
                active[env_id] = None
            if prepared[env_id] is not None:
                pending.append(prepared[env_id])
                prepared[env_id] = None
                prepared_since[env_id] = -1
            stable_touchdowns[env_id] = 0
            cooldown_until[env_id] = step + args.warmup_steps

        for env_id in state.touchdown.nonzero(as_tuple=False).flatten().tolist():
            stable_touchdowns[env_id] += 1
            trial = active[env_id]
            if (
                args.q_memory_diagnostic
                and trial is None
                and prepared[env_id] is not None
                and stable_touchdowns[env_id] >= 3
                and env_id not in nominal_sampled_envs
                and len(nominal_diagnostic_samples) < args.q_memory_sample_count
            ):
                nominal_diagnostic_samples.append(
                    {
                        "sample_index": len(nominal_diagnostic_samples),
                        "env_id": env_id,
                        "support_side": (
                            "left" if int(state.touchdown_foot[env_id].item()) == 0 else "right"
                        ),
                        "_query": _certificate_query(state, env_id, parameters),
                        **_touchdown_diagnostic_snapshot(state, env_id),
                    }
                )
                nominal_sampled_envs.add(env_id)
            if trial is None:
                continue
            foot = int(state.touchdown_foot[env_id].item())
            touchdown_time = float(state.time[env_id].item())
            trial["touchdowns"] += 1
            trial["certificate_trace"].append(
                {
                    "touchdown": trial["touchdowns"],
                    "time_after_push": touchdown_time - trial["start_time"],
                    "support_side": "left" if foot == 0 else "right",
                    "N_theory": None,
                    "margin": None,
                    "good_cycle": None,
                    "practical_entered": bool(trial["practical_entered"]),
                    "practical_confirmed": bool(trial["success"]),
                    "practical_enter_step": trial["practical_enter_step"],
                    "practical_confirmed_step": trial["N_confirmation"],
                    "status": "pending",
                    "_query": _certificate_query(state, env_id, parameters),
                    **(
                        _touchdown_diagnostic_snapshot(state, env_id)
                        if args.q_memory_diagnostic
                        else {}
                    ),
                }
            )
            alternating = trial["last_touchdown_foot"] is None or foot != trial["last_touchdown_foot"]
            good = None
            if trial["interval_started_after_touchdown"] and trial["interval_velocity"]:
                mean_velocity = float(np.mean(trial["interval_velocity"]))
                mean_tilt = np.mean(np.asarray(trial["interval_tilt"]), axis=0)
                good = bool(
                    alternating
                    and mean_velocity <= thresholds["mean_velocity_error"]
                    and mean_tilt[0] <= thresholds["mean_abs_roll"]
                    and mean_tilt[1] <= thresholds["mean_abs_pitch"]
                )
                trial["cycle_metrics"].append(
                    {
                        "mean_velocity_error": mean_velocity,
                        "mean_abs_roll": float(mean_tilt[0]),
                        "mean_abs_pitch": float(mean_tilt[1]),
                        "alternating": alternating,
                        "good": good,
                        "start_touchdown": trial["interval_start_touchdown"],
                        "end_touchdown": trial["touchdowns"],
                    }
                )
                if good:
                    if trial["consecutive_good_cycles"] == 0:
                        # The first good complete cycle shows that normal gait
                        # had resumed at its starting touchdown.  Keep waiting
                        # for a second good cycle only to confirm stability.
                        trial["recovery_onset_touchdown"] = trial["interval_start_touchdown"]
                        trial["recovery_onset_time"] = trial["interval_start_time"] - trial["start_time"]
                        if not trial["practical_entered"]:
                            trial["practical_entered"] = True
                            trial["practical_enter_step"] = trial["touchdowns"]
                    trial["consecutive_good_cycles"] += 1
                else:
                    trial["consecutive_good_cycles"] = 0
                    trial["recovery_onset_touchdown"] = None
                    trial["recovery_onset_time"] = None
            trial["certificate_trace"][-1]["good_cycle"] = good
            trial["interval_started_after_touchdown"] = True
            trial["interval_start_touchdown"] = trial["touchdowns"]
            trial["interval_start_time"] = touchdown_time
            trial["last_touchdown_foot"] = foot
            trial["interval_velocity"].clear()
            trial["interval_tilt"].clear()
            if trial["consecutive_good_cycles"] >= 2 and not trial["success"]:
                trial["success"] = True
                trial["N_actual"] = trial["recovery_onset_touchdown"]
                trial["N_confirmation"] = trial["touchdowns"]
                trial["t_recovery"] = trial["recovery_onset_time"]
                trial["t_confirmation"] = touchdown_time - trial["start_time"]
                if not args.recovery_manager_validation:
                    completed.append(_finalize_trial(trial))
                    active[env_id] = None
                    cooldown_until[env_id] = step + 100
                    stable_touchdowns[env_id] = 0
            trial["certificate_trace"][-1].update(
                {
                    "practical_entered": bool(trial["practical_entered"]),
                    "practical_confirmed": bool(trial["success"]),
                    "practical_enter_step": trial["practical_enter_step"],
                    "practical_confirmed_step": trial["N_confirmation"],
                }
            )
            if (
                args.recovery_manager_validation
                and active[env_id] is trial
                and trial["touchdowns"]
                >= (8 if args.q_memory_diagnostic else args.passive_log_touchdowns)
            ):
                completed.append(_finalize_trial(trial))
                active[env_id] = None
                cooldown_until[env_id] = step + 100
                stable_touchdowns[env_id] = 0

        for env_id in range(worker_env_count):
            trial = active[env_id]
            if trial is not None:
                velocity_error, tilt = _per_step_metric(state, env_id)
                trial["interval_velocity"].append(velocity_error)
                trial["interval_tilt"].append(tilt)
                if step - trial["start_step"] >= timeout_steps:
                    trial["failure_reason"] = "recovery_timeout"
                    completed.append(_finalize_trial(trial))
                    active[env_id] = None
                    cooldown_until[env_id] = step + 100
                    stable_touchdowns[env_id] = 0

            if (
                active[env_id] is None
                and prepared[env_id] is None
                and pending
                and step >= cooldown_until[env_id]
            ):
                prepared[env_id] = pending.pop()
                prepared_since[env_id] = step
                assigned_speeds[env_id] = prepared[env_id].get("command_vx", args.validation_speed)
                stable_touchdowns[env_id] = 0

            if (
                active[env_id] is None
                and prepared[env_id] is not None
                and step - prepared_since[env_id] >= max(500, 2 * args.warmup_steps)
            ):
                pending.append(prepared[env_id])
                prepared[env_id] = None
                prepared_since[env_id] = -1
                stable_touchdowns[env_id] = 0

            if active[env_id] is None and prepared[env_id] is not None and stable_touchdowns[env_id] >= 3:
                target_phase = prepared[env_id]["target_phase"]
                phase_now = float(phase_values[env_id])
                crossed = (
                    (target_phase == 0.0 and phase_now < last_phase[env_id])
                    or (last_phase[env_id] < target_phase <= phase_now)
                )
                if crossed:
                    active[env_id] = _start_trial(env, state, env_id, prepared[env_id], parameters, step)
                    prepared[env_id] = None
                    prepared_since[env_id] = -1
                    stable_touchdowns[env_id] = 0
            last_phase[env_id] = phase_values[env_id]

        if (
            not pending
            and all(plan is None for plan in prepared[:worker_env_count])
            and all(trial is None for trial in active[:worker_env_count])
        ):
            break
        if step > 0 and step % 100 == 0:
            print(
                f"[INFO] Gate 2 step={step}, completed={len(completed)}, pending={len(pending)}, "
                f"prepared={sum(plan is not None for plan in prepared)}, "
                f"active={sum(trial is not None for trial in active)}",
                flush=True,
            )
    else:
        print(f"[WARN] Gate 2 stopped with {len(pending)} pending plans")

    if args.random_trials > 0:
        _save_raw_disturbed_trials(
            completed, parameters, thresholds, nominal_diagnostic_samples
        )
    _resolve_saved_certificates(completed, parameters)
    if args.q_memory_diagnostic:
        _resolve_nominal_diagnostic_samples(nominal_diagnostic_samples, parameters)
    state_machine_summary = (
        _attach_recovery_state_machine(completed)
        if args.recovery_manager_validation
        else None
    )
    q_memory_summary = (
        _q_memory_diagnostic_summary(completed, nominal_diagnostic_samples)
        if args.q_memory_diagnostic
        else None
    )
    valid = [trial for trial in completed if trial["N_theory"] is not None and trial["margin"] is not None]
    if not valid:
        raise RuntimeError("Gate 2 produced no trials with a valid certificate result")
    max_success_steps = max((trial["N_actual"] for trial in valid if trial["success"]), default=5)
    theory = np.asarray([trial["N_theory"] for trial in valid], dtype=float)
    actual_difficulty = np.asarray(
        [trial["N_actual"] if trial["success"] else max_success_steps + 1 for trial in valid], dtype=float
    )
    margin = np.asarray([trial["margin"] for trial in valid], dtype=float)
    success = np.asarray([trial["success"] for trial in valid], dtype=float)

    grouped = {}
    for n_value in sorted(set(int(value) for value in theory)):
        group = [trial for trial in valid if int(trial["N_theory"]) == n_value]
        successful = [trial for trial in group if trial["success"]]
        label = str(n_value) if n_value < 6 else ">5"
        grouped[label] = {
            "count": len(group),
            "success_rate": len(successful) / len(group),
            "mean_N_actual_success_only": float(np.mean([item["N_actual"] for item in successful])) if successful else None,
            "median_N_actual_success_only": float(np.median([item["N_actual"] for item in successful])) if successful else None,
            "mean_t_recovery_success_only": float(np.mean([item["t_recovery"] for item in successful])) if successful else None,
            "median_t_recovery_success_only": float(np.median([item["t_recovery"] for item in successful])) if successful else None,
            "mean_margin": float(np.mean([item["margin"] for item in group])),
        }

    def summarize(group):
        successful = [trial for trial in group if trial["success"]]
        return {
            "count": len(group),
            "success_rate": len(successful) / len(group) if group else None,
            "mean_N_actual_success_only": (
                float(np.mean([trial["N_actual"] for trial in successful])) if successful else None
            ),
            "median_N_actual_success_only": (
                float(np.median([trial["N_actual"] for trial in successful])) if successful else None
            ),
            "mean_t_recovery_success_only": (
                float(np.mean([trial["t_recovery"] for trial in successful])) if successful else None
            ),
        }

    scope_groups = {}
    for scope in sorted(set(trial.get("push_scope", "grid") for trial in valid)):
        scope_groups[scope] = summarize(
            [trial for trial in valid if trial.get("push_scope", "grid") == scope]
        )

    magnitude_groups = {}
    magnitude_edges = (0.0, 0.5, 1.0, 1.5, 2.0, math.inf)
    for lower, upper in zip(magnitude_edges[:-1], magnitude_edges[1:]):
        group = [trial for trial in valid if lower <= float(trial["level"]) < upper]
        if group:
            label = f"[{lower:.1f}, {'inf' if math.isinf(upper) else f'{upper:.1f}'})"
            magnitude_groups[label] = summarize(group)

    command_speeds = np.asarray([float(trial.get("command_vx", args.validation_speed)) for trial in valid])

    theory_trace_pairs = []
    nonincreasing_count = 0
    transition_count = 0
    monotonic_trials = 0
    trace_trials = 0
    theory_at_recovery = []
    for trial in valid:
        if not trial["success"]:
            continue
        samples = [
            sample
            for sample in trial["certificate_trace"]
            if sample["N_theory"] is not None and sample["touchdown"] <= trial["N_actual"]
        ]
        if not samples:
            continue
        trace_trials += 1
        encoded = [min(int(sample["N_theory"]), 6) for sample in samples]
        remaining = [int(sample["N_actual_remaining"]) for sample in samples]
        theory_trace_pairs.extend(zip(encoded, remaining))
        transitions = list(zip(encoded[:-1], encoded[1:]))
        transition_count += len(transitions)
        trial_nonincreasing = sum(next_value <= value for value, next_value in transitions)
        nonincreasing_count += trial_nonincreasing
        if trial_nonincreasing == len(transitions):
            monotonic_trials += 1
        recovery_samples = [
            sample for sample in trial["certificate_trace"] if sample["touchdown"] == trial["N_actual"]
        ]
        if recovery_samples and recovery_samples[0]["N_theory"] is not None:
            theory_at_recovery.append(min(int(recovery_samples[0]["N_theory"]), 6))

    if theory_trace_pairs:
        trace_theory = np.asarray([pair[0] for pair in theory_trace_pairs], dtype=float)
        trace_remaining = np.asarray([pair[1] for pair in theory_trace_pairs], dtype=float)
        trace_correlation = _spearman_summary(trace_theory, trace_remaining)
        trace_mae = float(np.mean(np.abs(trace_theory - trace_remaining)))
        trace_exact = float(np.mean(trace_theory == trace_remaining))
    else:
        trace_correlation = {"rho": None, "p_value": None, "reason": "no valid trajectory pairs"}
        trace_mae = None
        trace_exact = None

    n_actual_distribution = {
        str(value): sum(trial["success"] and int(trial["N_actual"]) == value for trial in valid)
        for value in sorted(set(int(trial["N_actual"]) for trial in valid if trial["success"]))
    }
    n_theory_distribution = {
        (str(value) if value < 6 else ">5"): sum(int(trial["N_theory"]) == value for trial in valid)
        for value in sorted(set(int(trial["N_theory"]) for trial in valid))
    }
    if args.random_trials > 0:
        conditions = {
            "mode": "randomized",
            "requested_trial_count": args.random_trials,
            "in_range_fraction": args.in_range_fraction,
            "trained_delta_v_component_range": [
                -args.trained_abs_delta_v_xy,
                args.trained_abs_delta_v_xy,
            ],
            "outside_delta_v_component_limit": args.outside_abs_delta_v_xy,
            "command_vx_range": [float(command_speeds.min()), float(command_speeds.max())],
            "push_phase_range": [0.0, 1.0],
            "frame": "heading",
            "seed": args.seed,
            "note": "Forward command only because the current theory parameters were calibrated on straight walking.",
        }
    else:
        conditions = {
            "mode": "fixed_grid",
            "velocity_jump_magnitudes": list(args.push_levels),
            "direction_count": args.direction_count,
            "target_phases": list(args.push_phases),
            "trials_per_condition": args.trials_per_condition,
            "frame": "heading",
            "training_delta_v_component_range": [-1.0, 1.0],
            "note": "Levels above 1.0 m/s deliberately test beyond the trained per-axis range.",
        }
    return {
        "trial_count": len(completed),
        "valid_certificate_trial_count": len(valid),
        "conditions": conditions,
        "metric_definitions": {
            "N_actual": "touchdowns before the start of the first of two consecutive good complete cycles",
            "N_confirmation": "touchdowns observed through the end of the second consecutive good cycle",
            "t_recovery": "time to the start of the first confirmed-good cycle",
            "t_confirmation": "time through the end of the second confirmed-good cycle",
        },
        "recovery_state_machine": state_machine_summary,
        "q_memory_diagnostic": q_memory_summary,
        "spearman": {
            "N_theory_vs_actual_difficulty": _spearman_summary(theory, actual_difficulty),
            "margin_vs_actual_difficulty": _spearman_summary(margin, actual_difficulty),
            "margin_vs_success": _spearman_summary(margin, success),
            "failed_trial_actual_difficulty_encoding": int(max_success_steps + 1),
        },
        "step_distributions": {
            "N_theory_initial": n_theory_distribution,
            "N_actual_success_only": n_actual_distribution,
        },
        "trajectory_consistency": {
            "definition": "Initial state and each touchdown through the measured recovery onset.",
            "successful_trials_with_trace": trace_trials,
            "transition_count": transition_count,
            "nonincreasing_transition_fraction": (
                nonincreasing_count / transition_count if transition_count else None
            ),
            "fully_nonincreasing_trial_fraction": monotonic_trials / trace_trials if trace_trials else None,
            "N_theory_vs_N_actual_remaining_spearman": trace_correlation,
            "encoded_over_horizon_value": 6,
            "mean_absolute_step_error": trace_mae,
            "exact_step_match_fraction": trace_exact,
            "N_theory_at_actual_recovery_distribution": {
                (str(value) if value < 6 else ">5"): theory_at_recovery.count(value)
                for value in sorted(set(theory_at_recovery))
            },
        },
        "by_N_theory": grouped,
        "by_push_scope": scope_groups,
        "by_push_magnitude": magnitude_groups,
        "command_vx_statistics": {
            "min": float(command_speeds.min()),
            "mean": float(command_speeds.mean()),
            "median": float(np.median(command_speeds)),
            "max": float(command_speeds.max()),
        },
        "trials": completed,
    }


def _plane_nominal_gate(
    env,
    policy,
    runner_type,
    extractor,
    evaluator,
    nominal,
    estimator_runner=None,
    com_estimator=None,
    estimator_warmup=None,
    estimator_input_history=None,
) -> tuple[dict, dict, object, object]:
    """Estimate practical cycle thresholds from this exact policy/slope/command."""

    target = max(1, int(args.nominal_touchdowns_per_speed))
    nominal_sampling_envs = min(
        env.num_envs, max(1, int(args.plane_nominal_sampling_envs))
    )
    cycles: list[dict] = []
    interval_velocity = [[] for _ in range(env.num_envs)]
    interval_roll = [[] for _ in range(env.num_envs)]
    interval_pitch = [[] for _ in range(env.num_envs)]
    previous_foot: list[int | None] = [None] * env.num_envs
    segment_reset_pending = [False] * env.num_envs
    estimator_rows: list[dict] = []
    estimator_target = max(1, int(args.estimator_nominal_frames))
    _set_plane_commands(env)
    obs, obs_hist = env.get_observations()
    for step in range(args.max_gate1_steps):
        _set_plane_commands(env)
        obs, obs_hist, dones = _plane_policy_step(
            env, policy, runner_type, obs, obs_hist
        )
        state = extractor.extract()
        estimator_eligible = None
        if com_estimator is not None:
            if estimator_warmup is None:
                raise RuntimeError("standalone estimator diagnostic requires reset warm-up state")
            reset_mask = dones | state.episode_reset
            estimator_eligible = estimator_warmup.eligible_after_step(reset_mask)
            estimator_tensors = _com_velocity_estimator_tensors(
                env,
                com_estimator,
                state,
                obs,
                reset_mask,
                estimator_input_history,
            )
        else:
            estimator_tensors = (
                _dwaq_velocity_tensors(env, estimator_runner, state, obs_hist)
                if estimator_runner is not None
                else None
            )
        reset_ids = (dones | state.episode_reset).nonzero(as_tuple=False).flatten().tolist()
        for env_id in reset_ids:
            segment_reset_pending[env_id] = False
            previous_foot[env_id] = None
            interval_velocity[env_id].clear()
            interval_roll[env_id].clear()
            interval_pitch[env_id].clear()
        for env_id in (~state.terrain_plane_valid).nonzero(
            as_tuple=False
        ).flatten().tolist():
            if segment_reset_pending[env_id]:
                continue
            previous_foot[env_id] = None
            interval_velocity[env_id].clear()
            interval_roll[env_id].clear()
            interval_pitch[env_id].clear()
            segment_reset_pending[env_id] = True
            env.episode_length_buf[env_id] = env.max_episode_length
        for env_id in state.touchdown.nonzero(as_tuple=False).flatten().tolist():
            if segment_reset_pending[env_id]:
                continue
            foot = int(state.touchdown_foot[env_id].item())
            if (
                env_id < nominal_sampling_envs
                and
                step >= args.warmup_steps
                and previous_foot[env_id] is not None
                and foot != previous_foot[env_id]
                and interval_velocity[env_id]
                and len(cycles) < target
            ):
                cycles.append(
                    {
                        "velocity": np.asarray(interval_velocity[env_id]),
                        "roll": np.asarray(interval_roll[env_id]),
                        "pitch": np.asarray(interval_pitch[env_id]),
                        "_pending_certificate": evaluator.submit(
                            state,
                            torch.tensor(
                                [env_id], dtype=torch.long, device=state.b.device
                            ),
                        ),
                    }
                )
            previous_foot[env_id] = foot
            interval_velocity[env_id].clear()
            interval_roll[env_id].clear()
            interval_pitch[env_id].clear()
        if step >= args.warmup_steps:
            if estimator_tensors is not None and len(estimator_rows) < estimator_target:
                eligible = (
                    ~dones
                    & ~state.episode_reset
                    & state.terrain_plane_valid
                )
                if estimator_eligible is not None:
                    eligible &= estimator_eligible
                eligible = eligible.nonzero(as_tuple=False).flatten().tolist()
                remaining = estimator_target - len(estimator_rows)
                estimator_rows.extend(
                    _velocity_row(estimator_tensors, env_id, nominal.omega)
                    for env_id in eligible[:remaining]
                )
            velocity_error = torch.linalg.vector_norm(
                state.com_velocity[:, :2] - state.command_velocity[:, :2], dim=1
            ).detach().cpu().numpy()
            roll_pitch = state.root_roll_pitch.detach().cpu().numpy()
            for env_id in range(env.num_envs):
                if (
                    env_id >= nominal_sampling_envs
                    or
                    previous_foot[env_id] is None
                    or segment_reset_pending[env_id]
                    or not bool(state.terrain_plane_valid[env_id].item())
                ):
                    continue
                interval_velocity[env_id].append(float(velocity_error[env_id]))
                interval_roll[env_id].append(float(roll_pitch[env_id, 0]))
                interval_pitch[env_id].append(float(roll_pitch[env_id, 1]))
        estimator_complete = (
            (estimator_runner is None and com_estimator is None)
            or len(estimator_rows) >= estimator_target
        )
        if len(cycles) >= target and estimator_complete:
            break
        if step > 0 and step % 250 == 0:
            print(
                f"[plane] nominal slope={args.slope_degrees:+g} step={step} "
                f"cycles={len(cycles)}/{target}",
                flush=True,
            )
    if len(cycles) < target:
        raise RuntimeError(f"collected only {len(cycles)}/{target} nominal cycles")

    all_roll = np.concatenate([cycle["roll"] for cycle in cycles])
    all_pitch = np.concatenate([cycle["pitch"] for cycle in cycles])
    roll_star = float(np.median(all_roll))
    pitch_star = float(np.median(all_pitch))
    mean_velocity = np.asarray([np.mean(cycle["velocity"]) for cycle in cycles])
    mean_abs_roll = np.asarray(
        [np.mean(np.abs(cycle["roll"] - roll_star)) for cycle in cycles]
    )
    mean_abs_pitch = np.asarray(
        [np.mean(np.abs(cycle["pitch"] - pitch_star)) for cycle in cycles]
    )
    nominal_certificate_counts = {str(value): 0 for value in range(7)}
    nominal_certificate_invalid = 0
    nominal_margins = []
    for cycle in cycles:
        pending = cycle.pop("_pending_certificate")
        n_min, margin, valid = evaluator.resolve_with_validity(pending)
        if bool(valid.item()):
            value = int(n_min.item())
            nominal_certificate_counts[str(value)] += 1
            nominal_margins.append(float(margin.item()))
        else:
            nominal_certificate_invalid += 1
    nominal_certificate_valid = len(cycles) - nominal_certificate_invalid
    thresholds = {
        "mean_velocity_error": float(np.quantile(mean_velocity, 0.95)),
        "mean_abs_roll_error": float(np.quantile(mean_abs_roll, 0.95)),
        "mean_abs_pitch_error": float(np.quantile(mean_abs_pitch, 0.95)),
        "roll_star": roll_star,
        "pitch_star": pitch_star,
        "derivation": "p95 over complete nominal gait cycles from the same frozen policy/slope/command",
        "practical_metric_version": PRACTICAL_METRIC_INTERVAL_MEAN_V1,
        "baseline_cycle_count": len(cycles),
    }
    report = {
        "passed": True,
        "slope_degrees": float(args.slope_degrees),
        "command": [0.4, 0.0, 0.0],
        "nominal_node": {
            "T": nominal.step_period,
            "h_eff": nominal.h_eff,
            "omega": nominal.omega,
            "w": nominal.step_width,
            "epsilon_b": nominal.epsilon_b,
            "epsilon_q": nominal.epsilon_q,
            "sample_count": nominal.sample_count,
            "calibration_policy_id": nominal.calibration_policy_id,
            "practical_metric_version": nominal.practical_metric_version,
        },
        "actual_recovery_thresholds": thresholds,
        "nominal_cycle_statistics": {
            "mean_velocity_error_median": float(np.median(mean_velocity)),
            "mean_velocity_error_p95": float(np.quantile(mean_velocity, 0.95)),
            "mean_abs_roll_error_median": float(np.median(mean_abs_roll)),
            "mean_abs_roll_error_p95": float(np.quantile(mean_abs_roll, 0.95)),
            "mean_abs_pitch_error_median": float(np.median(mean_abs_pitch)),
            "mean_abs_pitch_error_p95": float(np.quantile(mean_abs_pitch, 0.95)),
        },
        "nominal_certificate_sanity": {
            "state_source": "privileged whole-body CoM position/velocity",
            "sampling_env_count": nominal_sampling_envs,
            "sampling_semantics": (
                "longitudinal gait-cycle samples from a small number of trajectories; "
                "synchronized deterministic clones are not counted as independent samples"
            ),
            "sample_count": len(cycles),
            "valid_count": nominal_certificate_valid,
            "invalid_count": nominal_certificate_invalid,
            "N_distribution_count": nominal_certificate_counts,
            "N_distribution_fraction": {
                key: count / nominal_certificate_valid
                if nominal_certificate_valid
                else None
                for key, count in nominal_certificate_counts.items()
            },
            "margin_median": (
                float(np.median(nominal_margins)) if nominal_margins else None
            ),
        },
    }
    if estimator_runner is not None or com_estimator is not None:
        report_key = (
            "CoM_velocity_estimator"
            if com_estimator is not None
            else "DWAQ_velocity_estimator"
        )
        report[report_key] = {
            "collection_semantics": (
                "post-warmup valid env-frame samples; estimator and simulator state "
                "share the same post-step time index"
            ),
            **_velocity_accuracy_report(estimator_rows),
        }
    return report, thresholds, obs, obs_hist


def _plane_trial_plans() -> list[dict]:
    repeats = args.estimator_repeats if _estimator_sensitivity_enabled() else args.plane_repeats
    if repeats <= 0:
        option = "--estimator_repeats" if _estimator_sensitivity_enabled() else "--plane_repeats"
        raise ValueError(f"{option} must be positive")
    plans = []
    directions = (
        ("+x", np.asarray((1.0, 0.0))),
        ("-x", np.asarray((-1.0, 0.0))),
        ("+y", np.asarray((0.0, 1.0))),
        ("-y", np.asarray((0.0, -1.0))),
    )
    for direction, unit in directions:
        for magnitude in (0.25, 0.50, 0.75, 1.00):
            for phase in (0.25, 0.75):
                for repeat in range(repeats):
                    plans.append(
                        {
                            "push_direction": direction,
                            "push_magnitude": magnitude,
                            "target_phase": phase,
                            "repeat": repeat,
                            "delta_v_heading_xy": magnitude * unit,
                        }
                    )
    np.random.default_rng(args.seed).shuffle(plans)
    return plans


def _plane_apply_push(
    env,
    state,
    env_id: int,
    plan: dict,
    step: int,
    velocity_diagnostic: dict | None = None,
) -> dict:
    delta_heading = np.asarray(plan["delta_v_heading_xy"], dtype=np.float64)
    delta_tensor = torch.tensor(
        [[delta_heading[0], delta_heading[1], 0.0]],
        dtype=torch.float32,
        device=env.device,
    )
    delta_world = quat_apply(state.heading_quat_w[env_id].unsqueeze(0), delta_tensor)[0]
    ids = torch.tensor([env_id], dtype=torch.long, device=env.device)
    root_velocity = env.robot.data.root_vel_w[ids].clone()
    root_velocity[:, :2] += delta_world[:2]
    env.robot.write_root_velocity_to_sim(root_velocity, env_ids=ids)
    return {
        **plan,
        "env_id": env_id,
        "push_step": step,
        "push_time": float(state.time[env_id].item()),
        "applied_phase": float(state.phase[env_id].item()),
        "delta_v_heading_xy": delta_heading,
        "delta_v_world_xy": delta_world[:2].detach().cpu().numpy(),
        "reference_established": False,
        "reference_time": None,
        "touchdown_index": -1,
        "last_touchdown_foot": None,
        "interval_velocity": [],
        "interval_roll_error": [],
        "interval_pitch_error": [],
        "N_actual_terminal": None,
        "N_actual_practical": None,
        "fall": False,
        "timeout": False,
        "trial_status": "awaiting_reference_touchdown",
        "failure_reason": None,
        "certificate_trace": [],
        "cycle_metrics": [],
        "pre_push_velocity_diagnostic": velocity_diagnostic,
    }


def _plane_finalize_trial(
    trial: dict,
    *,
    fall: bool = False,
    reason: str | None = None,
    applicability_exit: bool = False,
) -> dict:
    trial["fall"] = bool(fall)
    if applicability_exit:
        trial["trial_status"] = reason
        trial["timeout"] = False
    elif fall:
        trial["trial_status"] = reason or "fall"
        trial["N_actual_terminal"] = 6
        trial["N_actual_practical"] = 6
    else:
        trial["trial_status"] = "completed_in_applicability_domain"
        if trial["N_actual_terminal"] is None:
            trial["N_actual_terminal"] = 6
        if trial["N_actual_practical"] is None:
            trial["N_actual_practical"] = 6
        trial["timeout"] = bool(trial["N_actual_terminal"] == 6)
    trial["failure_reason"] = reason
    internal = {
        "reference_established",
        "last_touchdown_foot",
        "interval_velocity",
        "interval_roll_error",
        "interval_pitch_error",
    }
    return {key: value for key, value in trial.items() if key not in internal}


def _plane_add_touchdown(
    evaluator,
    terminal_config,
    state,
    env_id: int,
    trial: dict,
    thresholds: dict,
    estimator_tensors: dict[str, torch.Tensor] | None = None,
    nominal=None,
    estimate_name: str = "direct",
) -> bool:
    """Add TD0..TD5 and return True once the fixed observation horizon ends."""

    foot = int(state.touchdown_foot[env_id].item())
    touchdown_time = float(state.time[env_id].item())
    good_cycle = None
    cycle_metrics = None
    if not trial["reference_established"]:
        trial["reference_established"] = True
        trial["reference_time"] = touchdown_time
        trial["touchdown_index"] = 0
        trial["trial_status"] = "reference_touchdown_established"
    else:
        trial["touchdown_index"] += 1
        alternating = foot != trial["last_touchdown_foot"]
        if trial["interval_velocity"]:
            cycle_metrics = {
                "mean_velocity_error": float(np.mean(trial["interval_velocity"])),
                "mean_abs_roll_error": float(np.mean(trial["interval_roll_error"])),
                "mean_abs_pitch_error": float(np.mean(trial["interval_pitch_error"])),
                "alternating": alternating,
            }
            good_cycle = bool(
                alternating
                and cycle_metrics["mean_velocity_error"] <= thresholds["mean_velocity_error"]
                and cycle_metrics["mean_abs_roll_error"] <= thresholds["mean_abs_roll_error"]
                and cycle_metrics["mean_abs_pitch_error"] <= thresholds["mean_abs_pitch_error"]
            )
            cycle_metrics["good"] = good_cycle
            cycle_metrics["end_touchdown"] = trial["touchdown_index"]
            trial["cycle_metrics"].append(cycle_metrics)
            if good_cycle and trial["N_actual_practical"] is None:
                trial["N_actual_practical"] = trial["touchdown_index"]

    ids = torch.tensor([env_id], dtype=torch.long, device=state.b.device)
    pending_certificate = evaluator.submit(state, ids)
    query = pending_certificate.queries[0]
    velocity_diagnostic = None
    pending_estimate = None
    if estimator_tensors is not None:
        if nominal is None:
            raise RuntimeError("estimator touchdown diagnostic requires nominal omega")
        velocity_diagnostic = _velocity_row(
            estimator_tensors, env_id, nominal.omega
        )
        estimate_query = query_with_replaced_com_velocity(
            query,
            velocity_diagnostic["direct_com_est_heading"][:2],
            velocity_diagnostic["com_GT_heading"][:2],
            nominal.omega,
        )
        pending_estimate = evaluator.submit_queries(
            (estimate_query,), state.b.device, env_ids=(env_id,)
        )
    is_terminal = terminal_contains(
        query.b, query.q, query.support_side, terminal_config
    )
    if is_terminal and trial["N_actual_terminal"] is None:
        trial["N_actual_terminal"] = trial["touchdown_index"]
    trial["certificate_trace"].append(
        {
            "touchdown": trial["touchdown_index"],
            "time_after_push": touchdown_time - trial["push_time"],
            "b": query.b,
            "q": query.q,
            "support": query.support_side,
            "alpha": query.alpha,
            "N_theory": None,
            "margin": None,
            "certificate_valid": None,
            "velocity_diagnostic": velocity_diagnostic,
            "N_GT": None,
            "margin_GT": None,
            "certificate_valid_GT": None,
            f"N_{estimate_name}": None,
            f"margin_{estimate_name}": None,
            f"certificate_valid_{estimate_name}": None,
            "estimate_name": estimate_name if velocity_diagnostic is not None else None,
            "terminal_contains": bool(is_terminal),
            "practical_cycle_result": good_cycle,
            "cycle_metrics": cycle_metrics,
            "_pending_certificate": pending_certificate,
            "_pending_certificate_estimate": pending_estimate,
        }
    )
    trial["last_touchdown_foot"] = foot
    trial["interval_velocity"].clear()
    trial["interval_roll_error"].clear()
    trial["interval_pitch_error"].clear()
    return trial["touchdown_index"] >= 5


def _resolve_plane_certificates(evaluator, trials: list[dict]) -> None:
    count = sum(len(trial["certificate_trace"]) for trial in trials)
    print(f"[plane] resolving {count} saved touchdown certificate queries", flush=True)
    for trial in trials:
        for sample in trial["certificate_trace"]:
            pending = sample.pop("_pending_certificate")
            n_min, margin, valid = evaluator.resolve_with_validity(pending)
            sample["N_theory"] = int(n_min.item()) if bool(valid.item()) else None
            sample["margin"] = float(margin.item()) if bool(valid.item()) else None
            sample["certificate_valid"] = bool(valid.item())
            if sample.get("velocity_diagnostic") is not None:
                sample["N_GT"] = sample["N_theory"]
                sample["margin_GT"] = sample["margin"]
                sample["certificate_valid_GT"] = sample["certificate_valid"]
                estimate_name = sample.pop("estimate_name")
                estimate_pending = sample.pop("_pending_certificate_estimate")
                estimate_n, estimate_margin, estimate_valid = evaluator.resolve_with_validity(
                    estimate_pending
                )
                sample[f"N_{estimate_name}"] = (
                    int(estimate_n.item()) if bool(estimate_valid.item()) else None
                )
                sample[f"margin_{estimate_name}"] = (
                    float(estimate_margin.item()) if bool(estimate_valid.item()) else None
                )
                sample[f"certificate_valid_{estimate_name}"] = bool(estimate_valid.item())
            else:
                sample.pop("_pending_certificate_estimate", None)
                sample.pop("estimate_name", None)


def _summarize_estimator_gate2(
    trials: list[dict], nominal, *, estimate_name: str, estimator_label: str
) -> dict:
    pre_push_rows = [
        trial["pre_push_velocity_diagnostic"]
        for trial in trials
        if trial.get("pre_push_velocity_diagnostic") is not None
    ]
    samples = []
    for trial in trials:
        for sample in trial["certificate_trace"]:
            if sample.get("velocity_diagnostic") is None:
                continue
            samples.append(
                {
                    **sample,
                    **sample["velocity_diagnostic"],
                    "N_actual_terminal": trial["N_actual_terminal"],
                    "push_magnitude": trial["push_magnitude"],
                    "push_direction": trial["push_direction"],
                    "target_phase": trial["target_phase"],
                }
            )

    def group(predicate) -> list[dict]:
        return [sample for sample in samples if predicate(sample)]

    touchdown_groups = {
        "TD0": group(lambda sample: sample["touchdown"] == 0),
        "TD1": group(lambda sample: sample["touchdown"] == 1),
        "TD2": group(lambda sample: sample["touchdown"] == 2),
        "TD3_plus": group(lambda sample: 3 <= sample["touchdown"] <= 5),
        "TD0_to_TD5": samples,
    }
    by_touchdown = {
        f"TD{index}": _velocity_accuracy_report(
            group(lambda sample, index=index: sample["touchdown"] == index)
        )
        for index in range(6)
    }
    by_magnitude_td0 = {
        f"{magnitude:.2f}": _velocity_accuracy_report(
            group(
                lambda sample, magnitude=magnitude: sample["touchdown"] == 0
                and abs(sample["push_magnitude"] - magnitude) <= 1.0e-9
            )
        )
        for magnitude in (0.25, 0.50, 0.75, 1.00)
    }
    paired_certificate_samples = [
        sample
        for sample in samples
        if sample.get("certificate_valid_GT", False)
        and sample.get(f"certificate_valid_{estimate_name}", False)
    ]
    gt_ordering = terminal_ordering(paired_certificate_samples, "GT")
    estimated_ordering = terminal_ordering(paired_certificate_samples, estimate_name)

    def degradation(metric: str) -> float | None:
        before = gt_ordering.get(metric)
        after = estimated_ordering.get(metric)
        if before is None or after is None:
            return None
        return float(after - before)

    return {
        "sample_counts": {
            "pre_push": len(pre_push_rows),
            "touchdown_total": len(samples),
            **{name: len(rows) for name, rows in touchdown_groups.items()},
        },
        "pre_push_velocity_accuracy": _velocity_accuracy_report(pre_push_rows),
        "post_push_velocity_accuracy": {
            name: _velocity_accuracy_report(rows)
            for name, rows in touchdown_groups.items()
        },
        "by_touchdown_velocity_accuracy": by_touchdown,
        "TD0_by_push_magnitude_velocity_accuracy": by_magnitude_td0,
        "certificate_agreement": {
            "all_legal_touchdowns": certificate_agreement(samples, estimate_name),
            "TD0": certificate_agreement(touchdown_groups["TD0"], estimate_name),
            "TD1": certificate_agreement(touchdown_groups["TD1"], estimate_name),
            "TD2": certificate_agreement(touchdown_groups["TD2"], estimate_name),
            "TD3_plus": certificate_agreement(touchdown_groups["TD3_plus"], estimate_name),
        },
        "Gate_B_terminal_ordering": {
            "GT_certificate": gt_ordering,
            f"{estimator_label}_certificate": estimated_ordering,
            "estimate_minus_GT_spearman": {
                "N_vs_terminal": degradation("N_vs_terminal_spearman"),
                "margin_vs_terminal": degradation("margin_vs_terminal_spearman"),
            },
        },
        "epsilon_b_reference_cm": {
            "x": float(nominal.epsilon_b[0] * 100.0),
            "y": float(nominal.epsilon_b[1] * 100.0),
            "note": "comparison only; no epsilon was modified",
        },
        "kinematic_corrected_CoM": {
            "status": "pending",
            "reason": (
                "No existing deployment-faithful whole-body CoM Jacobian or centroidal "
                "kinematics API was found in this repository/IsaacLab articulation wrapper."
            ),
        },
    }


def _percentile(values: list[int], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _summarize_plane_gate(trials: list[dict]) -> dict:
    reference_trials = [
        trial
        for trial in trials
        if trial["certificate_trace"]
        and trial["certificate_trace"][0]["N_theory"] is not None
    ]
    if not reference_trials:
        raise RuntimeError("plane Gate B produced no valid reference TD0 certificates")
    correlation_trials = [
        trial
        for trial in reference_trials
        if trial["trial_status"] != "left_theory_applicability_domain"
    ]
    if not correlation_trials:
        raise RuntimeError("plane Gate B produced no applicability-valid correlation trials")
    theory = np.asarray(
        [trial["certificate_trace"][0]["N_theory"] for trial in correlation_trials],
        dtype=float,
    )
    margin = np.asarray(
        [trial["certificate_trace"][0]["margin"] for trial in correlation_trials],
        dtype=float,
    )
    terminal = np.asarray(
        [trial["N_actual_terminal"] for trial in correlation_trials], dtype=float
    )
    practical = np.asarray(
        [trial["N_actual_practical"] for trial in correlation_trials], dtype=float
    )
    by_n = {}
    for n_value in range(7):
        group = [
            trial
            for trial in correlation_trials
            if trial["certificate_trace"][0]["N_theory"] == n_value
        ]
        terminal_values = [trial["N_actual_terminal"] for trial in group]
        practical_values = [trial["N_actual_practical"] for trial in group]
        margins = [trial["certificate_trace"][0]["margin"] for trial in group]
        by_n[str(n_value)] = {
            "count": len(group),
            "terminal_success_rate_P5": (
                float(np.mean([value <= 5 for value in terminal_values])) if group else None
            ),
            "practical_success_rate_P5": (
                float(np.mean([value <= 5 for value in practical_values])) if group else None
            ),
            "fall_rate": float(np.mean([trial["fall"] for trial in group])) if group else None,
            "timeout_rate": float(np.mean([trial["timeout"] for trial in group])) if group else None,
            "median_N_actual_terminal": _percentile(terminal_values, 0.5),
            "P25_N_actual_terminal": _percentile(terminal_values, 0.25),
            "P75_N_actual_terminal": _percentile(terminal_values, 0.75),
            "median_N_actual_practical": _percentile(practical_values, 0.5),
            "P25_N_actual_practical": _percentile(practical_values, 0.25),
            "P75_N_actual_practical": _percentile(practical_values, 0.75),
            "median_margin": float(np.median(margins)) if margins else None,
        }

    trajectory_rho = []
    pooled_trajectory_theory = []
    pooled_trajectory_remaining = []
    nonincreasing = 0
    same = 0
    decrease = 0
    increase = 0
    transitions = 0
    for trial in correlation_trials:
        if trial["fall"] or trial["N_actual_terminal"] > 5:
            continue
        end = int(trial["N_actual_terminal"])
        samples = [
            sample
            for sample in trial["certificate_trace"]
            if sample["N_theory"] is not None and sample["touchdown"] <= end
        ]
        encoded = np.asarray([sample["N_theory"] for sample in samples], dtype=float)
        remaining = np.asarray(
            [max(end - int(sample["touchdown"]), 0) for sample in samples], dtype=float
        )
        if len(samples) >= 2:
            summary = _spearman_summary(encoded, remaining)
            if summary["rho"] is not None:
                trajectory_rho.append(summary["rho"])
            pooled_trajectory_theory.extend(encoded.tolist())
            pooled_trajectory_remaining.extend(remaining.tolist())
        for first, second in zip(encoded[:-1], encoded[1:]):
            transitions += 1
            nonincreasing += int(second <= first)
            same += int(second == first)
            decrease += int(second < first)
            increase += int(second > first)

    return {
        "trial_count": len(trials),
        "reference_touchdown_trial_count": len(reference_trials),
        "valid_correlation_trial_count": len(correlation_trials),
        "fall_before_reference_touchdown_count": sum(
            trial["failure_reason"] == "fall_before_reference_touchdown" for trial in trials
        ),
        "applicability": {
            "total_trials": len(trials),
            "valid_reference_trials": len(reference_trials),
            "valid_correlation_trials": len(correlation_trials),
            "invalid_before_reference_touchdown": sum(
                trial["trial_status"] == "invalid_before_reference_touchdown"
                for trial in trials
            ),
            "applicability_exit_after_reference": sum(
                trial["trial_status"] == "left_theory_applicability_domain"
                for trial in trials
            ),
            "fall_count": sum(trial["fall"] for trial in trials),
            "timeout_count": sum(trial["timeout"] for trial in trials),
            "practical_success_count": sum(
                trial["N_actual_practical"] is not None
                and trial["N_actual_practical"] <= 5
                and not trial["fall"]
                for trial in correlation_trials
            ),
            "terminal_success_count": sum(
                trial["N_actual_terminal"] is not None
                and trial["N_actual_terminal"] <= 5
                and not trial["fall"]
                for trial in correlation_trials
            ),
            "applicability_valid_fraction": len(correlation_trials) / len(trials),
            "fall_rate": float(np.mean([trial["fall"] for trial in trials])),
            "timeout_rate": float(
                np.mean([trial["timeout"] for trial in correlation_trials])
            ),
            "practical_success_rate_P5": float(
                np.mean([trial["N_actual_practical"] <= 5 for trial in correlation_trials])
            ),
            "terminal_success_rate_P5": float(
                np.mean([trial["N_actual_terminal"] <= 5 for trial in correlation_trials])
            ),
        },
        "conditions": {
            "command": [0.4, 0.0, 0.0],
            "slope_degrees": float(args.slope_degrees),
            "push_directions": ["+x", "-x", "+y", "-y"],
            "push_magnitudes": [0.25, 0.50, 0.75, 1.00],
            "target_gait_phases": [0.25, 0.75],
            "repeats": (
                args.estimator_repeats
                if _estimator_sensitivity_enabled()
                else args.plane_repeats
            ),
            "disturbance_type": "heading-frame root velocity jump",
            "certificate_reference": "first_post_push_touchdown",
        },
        "metric_definitions": {
            "N_actual_terminal": "TD0=0 then first touchdown inside the plane-node terminal; 6 means >5 or fall",
            "N_actual_practical": "first complete good cycle after TD0; 6 means >5 or fall",
        },
        "spearman": {
            "N_theory_0_vs_N_actual_terminal": _spearman_summary(
                theory, terminal, args.spearman_bootstrap_resamples
            ),
            "margin_0_vs_N_actual_terminal": _spearman_summary(
                margin, terminal, args.spearman_bootstrap_resamples
            ),
            "N_theory_0_vs_N_actual_practical": _spearman_summary(theory, practical),
            "margin_0_vs_N_actual_practical": _spearman_summary(margin, practical),
            "failure_and_over_horizon_encoding": 6,
        },
        "by_N_theory_0": by_n,
        "trajectory_consistency": {
            "successful_trial_spearman_rho": trajectory_rho,
            "median_successful_trial_spearman_rho": (
                float(np.median(trajectory_rho)) if trajectory_rho else None
            ),
            "nonincreasing_transition_fraction": (
                nonincreasing / transitions if transitions else None
            ),
            "same_transition_fraction": same / transitions if transitions else None,
            "decrease_transition_fraction": (
                decrease / transitions if transitions else None
            ),
            "increase_transition_fraction": (
                increase / transitions if transitions else None
            ),
            "transition_count": transitions,
            "pooled_N_theory_vs_actual_remaining_touchdowns_spearman": (
                _spearman_summary(
                    np.asarray(pooled_trajectory_theory, dtype=float),
                    np.asarray(pooled_trajectory_remaining, dtype=float),
                )
                if pooled_trajectory_theory
                else {"rho": None, "p_value": None, "count": 0}
            ),
        },
        "trials": trials,
    }


def _plane_gate2(
    env,
    policy,
    runner_type,
    extractor,
    evaluator,
    terminal_config,
    thresholds,
    obs,
    obs_hist,
    estimator_runner=None,
    com_estimator=None,
    estimator_warmup=None,
    estimator_input_history=None,
    nominal=None,
) -> dict:
    plans = _plane_trial_plans()
    pending = list(plans)
    active: list[dict | None] = [None] * env.num_envs
    prepared: list[dict | None] = [None] * env.num_envs
    forced_reset_pending = [True] * env.num_envs
    stable_touchdowns = np.zeros(env.num_envs, dtype=np.int64)
    cooldown_until = np.full(env.num_envs, args.warmup_steps, dtype=np.int64)
    last_phase = np.zeros(env.num_envs)
    completed: list[dict] = []
    trial_timeout_steps = int(math.ceil(args.trial_timeout_s / env.step_dt))
    per_env_trials = math.ceil(len(plans) / env.num_envs)
    max_steps = max(args.max_gate1_steps, args.warmup_steps + per_env_trials * 800)
    env.episode_length_buf[:] = env.max_episode_length

    def finish_and_request_reset(env_id: int, trial: dict) -> None:
        completed.append(trial)
        active[env_id] = None
        prepared[env_id] = None
        stable_touchdowns[env_id] = 0
        forced_reset_pending[env_id] = True
        env.episode_length_buf[env_id] = env.max_episode_length

    for step in range(max_steps):
        _set_plane_commands(env)
        obs, obs_hist, dones = _plane_policy_step(
            env, policy, runner_type, obs, obs_hist
        )
        state = extractor.extract()
        if com_estimator is not None:
            if estimator_warmup is None:
                raise RuntimeError("standalone estimator diagnostic requires reset warm-up state")
            reset_mask = dones | state.episode_reset
            estimator_eligible = estimator_warmup.eligible_after_step(reset_mask)
            estimator_tensors = _com_velocity_estimator_tensors(
                env,
                com_estimator,
                state,
                obs,
                reset_mask,
                estimator_input_history,
            )
        else:
            estimator_eligible = None
            estimator_tensors = (
                _dwaq_velocity_tensors(env, estimator_runner, state, obs_hist)
                if estimator_runner is not None
                else None
            )
        phases = state.phase.detach().cpu().numpy()
        reset_ids = (dones | state.episode_reset).nonzero(as_tuple=False).flatten().tolist()
        for env_id in reset_ids:
            if forced_reset_pending[env_id]:
                forced_reset_pending[env_id] = False
                stable_touchdowns[env_id] = 0
                cooldown_until[env_id] = step
                continue
            trial = active[env_id]
            if trial is not None:
                reason = (
                    "fall_before_reference_touchdown"
                    if not trial["reference_established"]
                    else "fall_after_reference_touchdown"
                )
                completed.append(_plane_finalize_trial(trial, fall=True, reason=reason))
                active[env_id] = None
            if prepared[env_id] is not None:
                pending.append(prepared[env_id])
                prepared[env_id] = None
            stable_touchdowns[env_id] = 0
            cooldown_until[env_id] = step + args.warmup_steps

        invalid_plane_ids = (~state.terrain_plane_valid).nonzero(
            as_tuple=False
        ).flatten().tolist()
        for env_id in invalid_plane_ids:
            if forced_reset_pending[env_id]:
                continue
            trial = active[env_id]
            if trial is not None:
                reason = (
                    "left_theory_applicability_domain"
                    if trial["reference_established"]
                    else "invalid_before_reference_touchdown"
                )
                finish_and_request_reset(
                    env_id,
                    _plane_finalize_trial(
                        trial,
                        reason=reason,
                        applicability_exit=True,
                    ),
                )
            else:
                stable_touchdowns[env_id] = 0
                forced_reset_pending[env_id] = True
                env.episode_length_buf[env_id] = env.max_episode_length

        for env_id in state.touchdown.nonzero(as_tuple=False).flatten().tolist():
            stable_touchdowns[env_id] += 1
            trial = active[env_id]
            if trial is None or forced_reset_pending[env_id]:
                continue
            finished = _plane_add_touchdown(
                evaluator,
                terminal_config,
                state,
                env_id,
                trial,
                thresholds,
                estimator_tensors=(
                    estimator_tensors
                    if estimator_eligible is None
                    or bool(estimator_eligible[env_id].item())
                    else None
                ),
                nominal=nominal,
                estimate_name=("EST" if com_estimator is not None else "direct"),
            )
            if finished:
                reason = (
                    "terminal_timeout_at_TD5"
                    if trial["N_actual_terminal"] is None
                    else "practical_only_timeout_at_TD5"
                    if trial["N_actual_practical"] is None
                    else None
                )
                finish_and_request_reset(
                    env_id, _plane_finalize_trial(trial, reason=reason)
                )

        velocity_error = torch.linalg.vector_norm(
            state.com_velocity[:, :2] - state.command_velocity[:, :2], dim=1
        ).detach().cpu().numpy()
        roll_pitch = state.root_roll_pitch.detach().cpu().numpy()
        for env_id, trial in enumerate(active):
            if (
                trial is None
                or forced_reset_pending[env_id]
                or not trial["reference_established"]
            ):
                continue
            trial["interval_velocity"].append(float(velocity_error[env_id]))
            trial["interval_roll_error"].append(
                abs(float(roll_pitch[env_id, 0]) - thresholds["roll_star"])
            )
            trial["interval_pitch_error"].append(
                abs(float(roll_pitch[env_id, 1]) - thresholds["pitch_star"])
            )
            if step - trial["push_step"] >= trial_timeout_steps:
                finish_and_request_reset(
                    env_id,
                    _plane_finalize_trial(
                        trial, reason="wall_time_recovery_timeout"
                    ),
                )

        for env_id in range(env.num_envs):
            if (
                active[env_id] is None
                and prepared[env_id] is None
                and not forced_reset_pending[env_id]
                and pending
                and step >= cooldown_until[env_id]
            ):
                prepared[env_id] = pending.pop()
                stable_touchdowns[env_id] = 0
            if (
                active[env_id] is None
                and prepared[env_id] is not None
                and not forced_reset_pending[env_id]
                and stable_touchdowns[env_id] >= 3
                and bool(state.terrain_plane_valid[env_id].item())
            ):
                target = prepared[env_id]["target_phase"]
                if last_phase[env_id] < target <= phases[env_id]:
                    active[env_id] = _plane_apply_push(
                        env,
                        state,
                        env_id,
                        prepared[env_id],
                        step,
                        velocity_diagnostic=(
                            _velocity_row(estimator_tensors, env_id, nominal.omega)
                            if estimator_tensors is not None
                            and (
                                estimator_eligible is None
                                or bool(estimator_eligible[env_id].item())
                            )
                            else None
                        ),
                    )
                    prepared[env_id] = None
                    stable_touchdowns[env_id] = 0
            last_phase[env_id] = phases[env_id]

        if not pending and not any(prepared) and not any(active):
            break
        if step > 0 and step % 250 == 0:
            print(
                f"[plane] Gate B slope={args.slope_degrees:+g} step={step} "
                f"completed={len(completed)}/{len(plans)} pending={len(pending)}",
                flush=True,
            )
    else:
        raise RuntimeError(
            f"plane Gate B exhausted {max_steps} steps with {len(completed)}/{len(plans)} trials"
        )
    _resolve_plane_certificates(evaluator, completed)
    report = _summarize_plane_gate(completed)
    if estimator_runner is not None or com_estimator is not None:
        report_key = (
            "CoM_velocity_estimator_sensitivity"
            if com_estimator is not None
            else "DWAQ_velocity_estimator_sensitivity"
        )
        report[report_key] = _summarize_estimator_gate2(
            completed,
            nominal,
            estimate_name=("EST" if com_estimator is not None else "direct"),
            estimator_label=(
                "standalone_CoM_estimator"
                if com_estimator is not None
                else "direct_DWAQ"
            ),
        )
    return report


def _run_plane_validation(parameters: dict) -> dict:
    if args.checkpoint_path is None:
        raise ValueError("plane validation requires --checkpoint_path")
    nominal_path = args.plane_nominal_params.expanduser().resolve()
    if not nominal_path.is_file():
        raise FileNotFoundError(f"Plane validation nominal table not found: {nominal_path}")
    nominal_document = _load_yaml(nominal_path)
    table = PlaneNominalParameterTable.from_yaml(nominal_path)
    node_label = f"alpha={args.slope_degrees:+g},direction=+x,speed=0.4"
    current_policy_id = nominal_document.get("collection", {}).get(
        "calibration_policy_id"
    )
    node_report = nominal_document.get("collection", {}).get("node_reports", {}).get(
        node_label
    )
    current_nodes = [
        node
        for node in nominal_document.get("nominal_plane_gait", {}).get("nodes", ())
        if abs(float(node["slope_degrees"]) - args.slope_degrees) <= 1.0e-9
        and str(node["direction"]) == "+x"
        and abs(float(node["speed"]) - 0.4) <= 1.0e-9
        and str(node.get("calibration_policy_id")) == str(current_policy_id)
    ]
    lookup = table.lookup_command(
        math.radians(args.slope_degrees), (0.4, 0.0, 0.0)
    )
    current_node_valid = bool(
        node_report is not None
        and node_report.get("valid", False)
        and len(current_nodes) == 1
    )
    nominal_available = lookup.valid and lookup.value is not None
    if (not _estimator_sensitivity_enabled() and not current_node_valid) or not nominal_available:
        reason = (
            node_report.get("reason", "node was not collected in the current run")
            if node_report is not None
            else "current collection has no report for this node"
        )
        return {
            "schema_version": 4,
            "validation_mode": "plane",
            "slope_degrees": float(args.slope_degrees),
            "stopped_before_gate_b": True,
            "stop_reason": f"missing current frozen-policy validation node: {reason}",
            "parameters": str(args.params.expanduser().resolve()),
            "plane_nominal_params": str(nominal_path),
            "checkpoint": str(args.checkpoint_path.expanduser().resolve()),
        }

    checkpoint_hash = _sha256(args.checkpoint_path.expanduser().resolve())
    expected_checkpoint_hash = nominal_document.get("teacher", {}).get(
        "checkpoint_sha256"
    )
    if expected_checkpoint_hash != checkpoint_hash:
        raise RuntimeError(
            "formal Plane Gate B checkpoint does not match candidate calibration: "
            f"{checkpoint_hash} != {expected_checkpoint_hash}"
        )
    env, policy, checkpoint, runner_type, runner, policy_contract = (
        _make_plane_env_policy()
    )
    if args.estimator_diagnostic and args.estimator_certificate_diagnostic:
        raise ValueError(
            "--estimator_diagnostic and --estimator_certificate_diagnostic are mutually exclusive"
        )
    if args.estimator_diagnostic and runner_type != "dwaq":
        raise ValueError("--estimator_diagnostic requires a DWAQ runner/task")
    if args.estimator_certificate_diagnostic and runner_type != "on_policy":
        raise ValueError(
            "--estimator_certificate_diagnostic requires a standard OnPolicyRunner task"
        )
    com_estimator = None
    estimator_warmup = None
    estimator_input_history = None
    if args.estimator_diagnostic:
        estimator_metadata = _dwaq_estimator_metadata(env, runner, checkpoint)
    elif args.estimator_certificate_diagnostic:
        (
            com_estimator,
            estimator_warmup,
            estimator_input_history,
            estimator_metadata,
        ) = _load_com_velocity_estimator(env, runner, checkpoint)
    else:
        estimator_metadata = None
    evaluator = PlaneCalibratedG1CertificateEvaluator(
        args.params,
        nominal_path,
        workers=args.certificate_workers,
        executor_type="subprocess",
    )
    try:
        terminal_config = evaluator._plane_config(
            (0.4, 0.0, 0.0), math.radians(args.slope_degrees), lookup.value
        )
        extractor = G1PrivilegedStateExtractor(
            env,
            G1StateExtractorCfg(
                h_eff=None,
                fallback_step_period=lookup.value.step_period,
                use_terrain_plane_geometry=True,
                slope_alignment_tolerance=math.radians(5.0),
            ),
        )
        gate1, thresholds, obs, obs_hist = _plane_nominal_gate(
            env,
            policy,
            runner_type,
            extractor,
            evaluator,
            lookup.value,
            estimator_runner=runner if args.estimator_diagnostic else None,
            com_estimator=com_estimator,
            estimator_warmup=estimator_warmup,
            estimator_input_history=estimator_input_history,
        )
        gate2 = _plane_gate2(
            env,
            policy,
            runner_type,
            extractor,
            evaluator,
            terminal_config,
            thresholds,
            obs,
            obs_hist,
            estimator_runner=runner if args.estimator_diagnostic else None,
            com_estimator=com_estimator,
            estimator_warmup=estimator_warmup,
            estimator_input_history=estimator_input_history,
            nominal=lookup.value if _estimator_sensitivity_enabled() else None,
        )
    finally:
        evaluator.close()
    return {
        "schema_version": 6 if args.estimator_certificate_diagnostic else (5 if args.estimator_diagnostic else 4),
        "validation_mode": (
            "plane_com_velocity_estimator_certificate_sensitivity"
            if args.estimator_certificate_diagnostic
            else "plane_dwaq_estimator_diagnostic"
            if args.estimator_diagnostic
            else "plane"
        ),
        "parameters": str(args.params.expanduser().resolve()),
        "plane_nominal_params": str(nominal_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "plane_nominal_params_sha256": _sha256(nominal_path),
        "calibration_git_commit": nominal_document.get("git_commit"),
        "runner_type": runner_type,
        "frozen_policy_contract": policy_contract,
        "certificate_state_source": (
            "G1PrivilegedStateExtractor simulator ground-truth whole-body CoM "
            "position and velocity; no learned estimator"
        ),
        "estimator_diagnostic": estimator_metadata,
        "nominal_reference_policy_identity": {
            "calibration_policy_id": current_policy_id,
            "matches_diagnostic_checkpoint": (
                str(current_policy_id).endswith(_sha256(checkpoint)[:12])
                if current_policy_id is not None
                else False
            ),
            "semantics": (
                "Velocity-only sensitivity reference. Production nominal parameters "
                "were read unchanged and were not recalibrated for this checkpoint."
                if _estimator_sensitivity_enabled()
                else "frozen-policy validation node"
            ),
        },
        "gate1_nominal": gate1,
        "gate2_disturbed": gate2,
        "stopped_before_gate_b": False,
    }


def _save_report(report):
    output = args.output.expanduser().resolve()
    if _estimator_sensitivity_enabled() and output == DEFAULT_REPORT.resolve():
        slope_label = f"{args.slope_degrees:+g}".replace("+", "plus").replace("-", "minus")
        filename = (
            f"g1_com_velocity_estimator_certificate_sensitivity_slope_{slope_label}.yaml"
            if args.estimator_certificate_diagnostic
            else f"g1_dwaq_estimator_diagnostic_slope_{slope_label}.yaml"
        )
        output = PROJECT_DIR / "tools/recovery/generated" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(_native(report), stream, sort_keys=False, allow_unicode=True)
    print(f"[INFO] Saved G1 recoverability report to {output}")
    trial_count = (report.get("gate2_disturbed") or {}).get("trial_count", 0)
    if trial_count <= 100 and not _estimator_sensitivity_enabled():
        print(yaml.safe_dump(_native(report), sort_keys=False, allow_unicode=True))
    else:
        summary = dict(report)
        summary["gate2_disturbed"] = dict(report["gate2_disturbed"])
        summary["gate2_disturbed"].pop("trials", None)
        print(yaml.safe_dump(_native(summary), sort_keys=False, allow_unicode=True))


def main():
    try:
        parameters = _load_yaml(args.params)
        if args.validation_mode == "plane":
            report = _run_plane_validation(parameters)
            _save_report(report)
            return
        env, policy, checkpoint = _make_env_policy(parameters)
        extractor = G1PrivilegedStateExtractor(
            env, G1StateExtractorCfg(h_eff=float(parameters["h_eff"]), fallback_step_period=float(parameters["T"]))
        )
        if args.reuse_gate1_report is not None:
            baseline_report_path = args.reuse_gate1_report.expanduser().resolve()
            baseline_report = _load_yaml(baseline_report_path)
            gate1 = baseline_report["gate1_nominal"]
            thresholds = gate1["actual_recovery_thresholds"]
            obs, _ = env.get_observations()
            print(f"[INFO] Reusing Gate 1 results from {baseline_report_path}", flush=True)
        else:
            gate1, thresholds, obs = _gate1(env, policy, extractor, parameters)
        report = {
            "schema_version": 3,
            "parameters": str(args.params.expanduser().resolve()),
            "checkpoint": str(checkpoint),
            "gate1_nominal": gate1,
            "gate2_disturbed": None,
        }
        if not gate1["passed"]:
            report["stopped_after_gate1"] = True
            report["stop_reason"] = "Nominal gait failed the configured certificate sanity criterion."
            _save_report(report)
            return
        report["gate2_disturbed"] = _gate2(env, policy, extractor, parameters, thresholds, obs)
        report["stopped_after_gate1"] = False
        _save_report(report)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
