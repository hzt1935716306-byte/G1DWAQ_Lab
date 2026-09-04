#!/usr/bin/env python3
"""Gate 1: evaluate the frozen V2 estimator under final Plane V1 conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from isaaclab.app import AppLauncher


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER = PROJECT_DIR / "logs/g1_slope_sys_d.pt"
DEFAULT_ESTIMATOR = (
    PROJECT_DIR
    / "logs/g1_com_velocity_estimator/v2_iteration_long_5000_random_init_fixed"
    / "com_velocity_estimator_v2_long_best.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "tools/recovery/generated/g1_plane_v1_estimator_gate1_long_best.json"
)
SLOPES_DEGREES = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
DIRECTIONS = ("+x", "-x", "+y", "-y")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher_checkpoint", type=Path, default=DEFAULT_TEACHER)
parser.add_argument("--estimator_checkpoint", type=Path, default=DEFAULT_ESTIMATOR)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--policy_steps", type=int, default=1000)
parser.add_argument("--speed_range", type=float, nargs=2, default=(0.2, 0.4))
parser.add_argument("--seed", type=int, default=20260905)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.sensors import ImuCfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
import legged_lab.mdp as mdp  # noqa: E402
from legged_lab.estimation import (  # noqa: E402
    ErrorMetricAccumulator,
    EstimatorFrameHistory,
    ResetWarmupMask,
    TouchdownAfterTransientTracker,
    extract_com_velocity_target,
    latest_actor_frame,
    load_com_velocity_estimator_for_inference,
)
from legged_lab.terrains import make_plane_recovery_terrain_cfg  # noqa: E402
from legged_lab.utils import task_registry  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _strict_checkpoint_metadata(path: Path) -> tuple[torch.nn.Module, dict]:
    model, payload = load_com_velocity_estimator_for_inference(path, device=args.device)
    expected = {
        "input_dim": 495,
        "history_length": 5,
        "actor_per_frame_obs_dim": 96,
        "per_frame_obs_dim": 99,
        "hidden_dims": [256, 128, 64],
        "output_dim": 2,
        "output_frame": "heading",
        "output_quantity": "whole_body_com_velocity_xy",
        "output_unit": "m/s",
        "imu_acceleration_scale": 0.05,
    }
    mismatch = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"final estimator contract mismatch: {mismatch}")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("final estimator is not frozen in eval mode")
    metadata = {
        key: payload.get(key)
        for key in (
            "schema_version",
            "training_format",
            "current_iteration",
            "best_iteration",
            "input_dim",
            "history_length",
            "actor_per_frame_obs_dim",
            "per_frame_obs_dim",
            "imu_input_dim",
            "hidden_dims",
            "output_dim",
            "activation",
            "output_frame",
            "output_quantity",
            "output_unit",
            "imu_acceleration_scale",
            "teacher_task",
            "teacher_checkpoint",
            "teacher_checkpoint_hash",
            "git_commit",
        )
    }
    return model, metadata


def _configure_final_plane_gate1(env_cfg) -> None:
    """Use final Plane V1 sensor, terrain, reset and push semantics."""

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.seed = args.seed
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.scene.terrain_type = "generator"
    env_cfg.scene.terrain_generator = make_plane_recovery_terrain_cfg(SLOPES_DEGREES)
    env_cfg.scene.max_init_terrain_level = 0
    env_cfg.scene.imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=ImuCfg.OffsetCfg(pos=(0.04525, 0.0, -0.08339)),
        update_period=0.0,
        gravity_bias=(0.0, 0.0, 9.81),
    )
    env_cfg.noise.add_noise = True
    env_cfg.robot.actor_obs_history_length = 10
    env_cfg.robot.critic_obs_history_length = 10
    env_cfg.recovery_context.enabled = False
    env_cfg.recovery_context.mode = "zero"
    env_cfg.stage2_reward.enabled = False
    env_cfg.stage2_reward.enable_shared_event_reward = False
    env_cfg.stage2_reward.enable_certificate_reward = False
    env_cfg.stage2_reward.enable_soft_reward_scaling = False
    env_cfg.stage2_reward.defer_certificate_reward_to_rollout_end = False
    env_cfg.push_curriculum.enable_push_curriculum = False
    env_cfg.push_curriculum.adaptive_upgrades_enabled = False
    env_cfg.push_curriculum.easy_sample_probability = 0.0
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.debug_vis = False
    env_cfg.commands.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.ranges.lin_vel_x = (-0.4, 0.4)
    env_cfg.commands.ranges.lin_vel_y = (-0.4, 0.4)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = None
    reset_base = env_cfg.domain_rand.events.reset_base
    reset_base.params["pose_range"]["x"] = (0.0, 0.0)
    reset_base.params["pose_range"]["y"] = (0.0, 0.0)
    reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    reset_base.params["velocity_range"]["yaw"] = (0.0, 0.0)
    env_cfg.domain_rand.events.push_robot = EventTerm(
        func=mdp.fixed_full_range_push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={},
    )


def _balanced_commands(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign all four cardinal directions within every realized slope column."""

    terrain_types = env.scene.terrain.terrain_types.to(dtype=torch.long)
    direction_index = torch.empty_like(terrain_types)
    for slope_index in range(len(SLOPES_DEGREES)):
        env_ids = (terrain_types == slope_index).nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            raise RuntimeError(f"no environment was assigned to slope index {slope_index}")
        direction_index[env_ids] = torch.arange(
            env_ids.numel(), device=env.device, dtype=torch.long
        ) % len(DIRECTIONS)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed + 17)
    speed_min, speed_max = (float(value) for value in args.speed_range)
    if not 0.0 < speed_min <= speed_max <= 0.4:
        raise ValueError("Gate 1 speed range must satisfy 0 < min <= max <= 0.4")
    speeds = speed_min + (speed_max - speed_min) * torch.rand(
        env.num_envs, generator=generator, device=env.device
    )
    commands = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    commands[direction_index == 0, 0] = speeds[direction_index == 0]
    commands[direction_index == 1, 0] = -speeds[direction_index == 1]
    commands[direction_index == 2, 1] = speeds[direction_index == 2]
    commands[direction_index == 3, 1] = -speeds[direction_index == 3]
    return terrain_types, direction_index, commands


def _pin_commands(env, commands: torch.Tensor) -> None:
    env.command_generator.command.copy_(commands)
    env.command_generator.is_standing_env[:] = False
    env.command_generator.is_heading_env[:] = False
    original_reset = env.command_generator.reset

    def reset_with_pinned_commands(env_ids):
        result = original_reset(env_ids)
        env.command_generator.command[env_ids] = commands[env_ids]
        env.command_generator.is_standing_env[env_ids] = False
        env.command_generator.is_heading_env[env_ids] = False
        return result

    env.command_generator.reset = reset_with_pinned_commands


def _summary_from_errors(error: torch.Tensor) -> dict[str, float | int | None]:
    if error.numel() == 0:
        return {
            "count": 0,
            "rmse_x": None,
            "rmse_y": None,
            "vector_rmse": None,
            "p95_vector_error": None,
            "bias_x": None,
            "bias_y": None,
        }
    squared = torch.square(error)
    vector = torch.linalg.vector_norm(error, dim=1)
    return {
        "count": int(error.shape[0]),
        "rmse_x": float(torch.sqrt(squared[:, 0].mean())),
        "rmse_y": float(torch.sqrt(squared[:, 1].mean())),
        "vector_rmse": float(torch.sqrt(squared.sum(dim=1).mean())),
        "p95_vector_error": float(torch.quantile(vector, 0.95)),
        "bias_x": float(error[:, 0].mean()),
        "bias_y": float(error[:, 1].mean()),
    }


def _gate_result(metrics: dict, combinations: dict) -> tuple[str, list[str]]:
    failures = []
    limits = {
        ("overall", "vector_rmse"): 0.10,
        ("transient", "vector_rmse"): 0.20,
        ("td0", "vector_rmse"): 0.15,
        ("td0", "p95_vector_error"): 0.30,
    }
    for (group, name), limit in limits.items():
        value = metrics[group][name]
        if value is None or value > limit:
            failures.append(f"{group}.{name}={value} exceeds {limit}")
    for name in ("bias_x", "bias_y"):
        value = metrics["overall"][name]
        if value is None or abs(value) > 0.03:
            failures.append(f"overall.{name}={value} exceeds abs limit 0.03")
    catastrophic = [
        name
        for name, values in combinations.items()
        if values["vector_rmse"] is None or values["vector_rmse"] > 0.30
    ]
    if catastrophic:
        failures.append(f"catastrophic slope/direction combinations: {catastrophic}")
    return ("PASS" if not failures else "FAIL"), failures


def evaluate() -> dict:  # noqa: C901,PLR0915
    if args.num_envs < len(SLOPES_DEGREES) * len(DIRECTIONS):
        raise ValueError("num_envs must cover all 28 slope/direction combinations")
    if args.policy_steps <= 0:
        raise ValueError("policy_steps must be positive")
    teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    estimator_checkpoint = args.estimator_checkpoint.expanduser().resolve()
    for path in (teacher_checkpoint, estimator_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    estimator, estimator_metadata = _strict_checkpoint_metadata(estimator_checkpoint)
    env_cfg, agent_cfg = task_registry.get_cfgs("g1_slope_sys_nd")
    _configure_final_plane_gate1(env_cfg)
    env_cfg.device = args.device
    agent_cfg.device = args.device
    env = task_registry.get_task_class("g1_slope_sys_nd")(
        env_cfg, headless=args.headless
    )
    terrain_types, direction_index, commands = _balanced_commands(env)
    _pin_commands(env, commands)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(teacher_checkpoint), load_optimizer=False)
    runner.eval_mode()
    runner.alg.policy.requires_grad_(False)
    teacher = runner.get_inference_policy(device=args.device)
    observations, _ = env.get_observations()
    actor_layers = [
        module for module in runner.alg.policy.actor if isinstance(module, torch.nn.Linear)
    ]
    teacher_contract = {
        "actor_input_dim": int(actor_layers[0].in_features),
        "actor_observation_dim": int(observations.shape[1]),
        "actor_frame_dim": int(env.compute_current_observations()[0].shape[1]),
        "actor_history_length": int(env.cfg.robot.actor_obs_history_length),
        "action_dim": int(actor_layers[-1].out_features),
        "eval_mode": not runner.alg.policy.training,
        "requires_grad": any(
            parameter.requires_grad for parameter in runner.alg.policy.parameters()
        ),
    }
    expected_teacher = {
        "actor_input_dim": 960,
        "actor_observation_dim": 960,
        "actor_frame_dim": 96,
        "actor_history_length": 10,
        "action_dim": 29,
        "eval_mode": True,
        "requires_grad": False,
    }
    if teacher_contract != expected_teacher:
        raise RuntimeError(f"teacher contract mismatch: {teacher_contract}")

    history = EstimatorFrameHistory(
        env.num_envs,
        history_length=5,
        actor_frame_dim=96,
        imu_dim=3,
        imu_acceleration_scale=0.05,
        device=args.device,
    )
    warmup = ResetWarmupMask(env.num_envs, 5, args.device)
    tracker = TouchdownAfterTransientTracker(env.num_envs, args.device)
    previous_target = torch.zeros((env.num_envs, 2), device=env.device)
    previous_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    metric_groups = {
        name: ErrorMetricAccumulator()
        for name in ("overall", "transient", "touchdown", "td0", "td1")
    }
    all_errors = []
    all_slopes = []
    all_directions = []
    push_count = 0
    reset_count = 0

    for step in range(args.policy_steps):
        with torch.inference_mode():
            actions = teacher(observations)
            observations, _, dones, _ = env.step(actions)
        # The reset hook pins reset rows; this copy also guards against an
        # unexpected command-timer resample in the remaining rows.
        env.command_generator.command.copy_(commands)
        state = env._last_recovery_state
        reset = dones.to(dtype=torch.bool) | state.episode_reset
        target = extract_com_velocity_target(state)
        estimator_input = history.append(
            latest_actor_frame(observations, history_length=10, per_frame_obs_dim=96),
            env.scene.sensors["imu"].data.lin_acc_b,
            reset,
        )
        eligible = warmup.eligible_after_step(reset)
        transient = (
            previous_valid
            & ~reset
            & (torch.linalg.vector_norm(target - previous_target, dim=1) > 0.15)
        )
        td0, td1 = tracker.update(transient, state.touchdown, reset)
        with torch.inference_mode():
            prediction = estimator(estimator_input)
        metric_groups["overall"].add(prediction, target, eligible)
        metric_groups["transient"].add(prediction, target, eligible & transient)
        metric_groups["touchdown"].add(prediction, target, eligible & state.touchdown)
        metric_groups["td0"].add(prediction, target, eligible & td0)
        metric_groups["td1"].add(prediction, target, eligible & td1)
        selected = eligible.nonzero(as_tuple=False).flatten()
        if selected.numel():
            all_errors.append((prediction[selected] - target[selected]).detach().cpu())
            all_slopes.append(terrain_types[selected].detach().cpu())
            all_directions.append(direction_index[selected].detach().cpu())
        push_count += int(env._last_push_started_mask.sum().item())
        reset_count += int(reset.sum().item())
        previous_target.copy_(target)
        previous_valid.copy_(~reset)
        if (step + 1) % 100 == 0 or step + 1 == args.policy_steps:
            print(
                f"[PlaneV1EstimatorGate1] step={step + 1}/{args.policy_steps} "
                f"pushes={push_count} resets={reset_count}",
                flush=True,
            )

    error = torch.cat(all_errors, dim=0)
    slope_ids = torch.cat(all_slopes, dim=0)
    direction_ids = torch.cat(all_directions, dim=0)
    by_slope = {
        f"{slope:+g}": _summary_from_errors(error[slope_ids == index])
        for index, slope in enumerate(SLOPES_DEGREES)
    }
    by_direction = {
        direction: _summary_from_errors(error[direction_ids == index])
        for index, direction in enumerate(DIRECTIONS)
    }
    combinations = {
        f"slope={slope:+g},direction={direction}": _summary_from_errors(
            error[(slope_ids == slope_index) & (direction_ids == direction_index_value)]
        )
        for slope_index, slope in enumerate(SLOPES_DEGREES)
        for direction_index_value, direction in enumerate(DIRECTIONS)
    }
    metrics = {name: accumulator.summary() for name, accumulator in metric_groups.items()}
    status, failures = _gate_result(metrics, combinations)
    return {
        "schema_version": 1,
        "gate": "Plane_V1_estimator_deployment_Gate_1",
        "status": status,
        "failures": failures,
        "git_commit": _git_commit(),
        "seed": args.seed,
        "environment": {
            "task_scaffold": "g1_slope_sys_nd",
            "semantics": "final Plane V1 terrain, IMU, reset, noise/domain randomization and fixed-full-range push; 10x96 history retained only for frozen teacher action generation",
            "num_envs": args.num_envs,
            "policy_steps": args.policy_steps,
            "slopes_degrees": list(SLOPES_DEGREES),
            "directions": list(DIRECTIONS),
            "speed_range_m_per_s": list(args.speed_range),
            "yaw_command": 0.0,
            "push_delta_v_xy_m_per_s": [[-1.0, 1.0], [-1.0, 1.0]],
            "push_interval_s": [10.0, 15.0],
            "push_event_count": push_count,
            "reset_count": reset_count,
            "imu": {
                "prim_path": "/Robot/pelvis",
                "offset_m": [0.04525, 0.0, -0.08339],
                "update_period_s": 0.0,
                "gravity_bias_m_per_s2": [0.0, 0.0, 9.81],
            },
        },
        "teacher_checkpoint": {
            "path": str(teacher_checkpoint),
            "sha256": _sha256(teacher_checkpoint),
            "contract": teacher_contract,
        },
        "estimator_checkpoint": {
            "path": str(estimator_checkpoint),
            "sha256": _sha256(estimator_checkpoint),
            "metadata": estimator_metadata,
            "eval_mode": not estimator.training,
            "requires_grad": any(parameter.requires_grad for parameter in estimator.parameters()),
        },
        "metric_semantics": {
            "target": "privileged mass-weighted whole-body CoM velocity XY in yaw-only heading frame",
            "transient": "norm(GT velocity_t - GT velocity_t-1) > 0.15 m/s",
            "touchdown": "G1PrivilegedStateExtractor contact rising edge",
            "td0_td1": "first and second touchdown after a transient",
            "reset_warmup": "first five policy transitions excluded",
        },
        "metrics": metrics,
        "by_slope": by_slope,
        "by_direction": by_direction,
        "by_slope_direction": combinations,
        "acceptance_limits": {
            "overall_vector_rmse_max": 0.10,
            "transient_vector_rmse_max": 0.20,
            "td0_vector_rmse_max": 0.15,
            "td0_p95_vector_error_max": 0.30,
            "overall_abs_bias_xy_max": 0.03,
            "catastrophic_combination_vector_rmse_max": 0.30,
        },
    }


def main() -> None:
    try:
        report = evaluate()
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[PlaneV1EstimatorGate1] report={output}", flush=True)
        print(json.dumps({"status": report["status"], "metrics": report["metrics"]}, indent=2))
        if report["status"] != "PASS":
            raise SystemExit(2)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
