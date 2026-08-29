#!/usr/bin/env python3
"""Calibrate the first G1 recoverability parameters from privileged simulation data."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUN = "2026-08-27_22-50-24_push_original"
DEFAULT_CHECKPOINT = "model_9998.pt"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="g1_flat_symmetric_recovery")
parser.add_argument("--load_run", default=DEFAULT_RUN)
parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
parser.add_argument("--checkpoint_path", type=Path, default=None)
parser.add_argument("--speeds", type=float, nargs="+", default=(0.2, 0.4, 0.6, 0.8, 1.0))
parser.add_argument("--warmup_steps", type=int, default=250)
parser.add_argument("--touchdowns_per_speed", type=int, default=40)
parser.add_argument("--max_nominal_steps", type=int, default=10000)
parser.add_argument("--disturbed_trials_per_speed", type=int, default=12)
parser.add_argument("--recovery_touchdowns", type=int, default=4)
parser.add_argument("--max_disturbed_steps", type=int, default=30000)
parser.add_argument("--push_max_xy", type=float, nargs=2, default=(1.0, 1.0))
parser.add_argument("--push_interval_s", type=float, nargs=2, default=(10.0, 15.0))
parser.add_argument("--geometry_shrink", type=float, default=0.10)
parser.add_argument("--landing_shrink", type=float, default=0.05)
parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_DIR / "tools/recovery/generated/g1_recovery_params.yaml",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.recovery.state_extractor import (  # noqa: E402
    G1PrivilegedStateExtractor,
    G1StateExtractorCfg,
    theoretical_periodic_state,
)
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
    if isinstance(value, Path):
        return str(value)
    return value


def _disable_evaluation_randomization(env_cfg):
    events = env_cfg.domain_rand.events
    events.push_robot = None
    events.physics_material = None
    events.add_base_mass = None
    for name in ("randomize_dome_light", "randomize_distant_light"):
        if hasattr(events, name):
            setattr(events, name, None)

    pose_range = events.reset_base.params["pose_range"]
    velocity_range = events.reset_base.params["velocity_range"]
    for key in pose_range:
        pose_range[key] = (0.0, 0.0)
    for key in velocity_range:
        velocity_range[key] = (0.0, 0.0)
    events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def _resolve_checkpoint(agent_cfg) -> Path:
    if args.checkpoint_path is not None:
        path = args.checkpoint_path.expanduser().resolve()
    else:
        log_root = PROJECT_DIR / "logs" / agent_cfg.experiment_name
        path = Path(get_checkpoint_path(str(log_root), args.load_run, args.checkpoint)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _make_env_and_policy():
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    speeds = tuple(float(value) for value in args.speeds)
    env_cfg.scene.num_envs = len(speeds)
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.scene.terrain_type = "plane"
    env_cfg.scene.terrain_generator = None
    env_cfg.noise.add_noise = False
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.debug_vis = False
    env_cfg.commands.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.ranges.lin_vel_x = (min(speeds), max(speeds))
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = None
    _disable_evaluation_randomization(env_cfg)

    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, args.headless)
    checkpoint = _resolve_checkpoint(agent_cfg)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    return env, policy, checkpoint, speeds


def _set_commands(env, speeds):
    command = env.command_generator.command
    command[:, 0] = torch.as_tensor(speeds, device=env.device)
    command[:, 1:] = 0.0
    env.command_generator.is_standing_env[:] = False


def _touchdown_snapshot(state, env_id: int) -> dict:
    support_left = bool(state.support_is_left[env_id].item())
    support_position = state.left_foot_position[env_id] if support_left else state.right_foot_position[env_id]
    support_position_w = (
        state.left_foot_position_w[env_id] if support_left else state.right_foot_position_w[env_id]
    )
    return {
        "env_id": env_id,
        "time": float(state.time[env_id].item()),
        "command": state.command_velocity[env_id].detach().cpu().numpy(),
        "support_side": "left" if support_left else "right",
        "com_position": state.com_position[env_id].detach().cpu().numpy(),
        "com_velocity": state.com_velocity[env_id].detach().cpu().numpy(),
        "com_height": float(state.com_height[env_id].item()),
        "support_position": support_position.detach().cpu().numpy(),
        "support_position_w": support_position_w.detach().clone(),
        "heading_quat_w": state.heading_quat_w[env_id].detach().clone(),
        "q": state.q[env_id].detach().cpu().numpy(),
        "root_roll_pitch": state.root_roll_pitch[env_id].detach().cpu().numpy(),
    }


def _transition(previous: dict, current: dict) -> dict | None:
    if previous["support_side"] == current["support_side"]:
        return None
    duration = current["time"] - previous["time"]
    if not 0.10 <= duration <= 1.50:
        return None
    displacement_w = (current["support_position_w"] - previous["support_position_w"]).unsqueeze(0)
    landing = quat_apply_inverse(previous["heading_quat_w"].unsqueeze(0), displacement_w)[0, :2]
    landing = landing.detach().cpu().numpy()
    return {
        "env_id": previous["env_id"],
        "command": previous["command"].copy(),
        "support_side": previous["support_side"],
        "duration": duration,
        "q_start": previous["q"].copy(),
        "landing": landing,
        "swing_displacement": landing - previous["q"],
    }


def _collect_nominal(env, policy, extractor, speeds):
    obs, _ = env.get_observations()
    previous = [None] * env.num_envs
    touchdown_records: list[dict] = []
    transition_records: list[dict] = []
    counts = np.zeros(env.num_envs, dtype=np.int64)

    for step in range(args.max_nominal_steps):
        _set_commands(env, speeds)
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            state = extractor.extract()
        reset_ids = (dones | state.episode_reset).nonzero(as_tuple=False).flatten().tolist()
        for env_id in reset_ids:
            previous[env_id] = None
        for env_id in state.touchdown.nonzero(as_tuple=False).flatten().tolist():
            current = _touchdown_snapshot(state, env_id)
            if step >= args.warmup_steps and previous[env_id] is not None:
                record = _transition(previous[env_id], current)
                if record is not None and counts[env_id] < args.touchdowns_per_speed:
                    current["step_period"] = record["duration"]
                    touchdown_records.append(current)
                    transition_records.append(record)
                    counts[env_id] += 1
            previous[env_id] = current
        if step > 0 and step % 500 == 0:
            print(f"[INFO] nominal step={step}, touchdown_counts={counts.tolist()}", flush=True)
        if np.all(counts >= args.touchdowns_per_speed):
            break
    else:
        print(f"[WARN] nominal collection reached {args.max_nominal_steps} steps: counts={counts.tolist()}")
    return touchdown_records, transition_records, obs


def _apply_velocity_jump(env, env_id: int) -> np.ndarray:
    maximum = torch.as_tensor(args.push_max_xy, device=env.device)
    delta = (2.0 * torch.rand(2, device=env.device) - 1.0) * maximum
    ids = torch.tensor([env_id], dtype=torch.long, device=env.device)
    root_velocity = env.robot.data.root_vel_w[ids].clone()
    root_velocity[:, :2] += delta
    env.robot.write_root_velocity_to_sim(root_velocity, env_ids=ids)
    return delta.detach().cpu().numpy()


def _collect_disturbed(env, policy, extractor, speeds, obs):
    previous = [None] * env.num_envs
    active: list[dict | None] = [None] * env.num_envs
    cooldown_until = np.full(env.num_envs, args.warmup_steps, dtype=np.int64)
    successes = np.zeros(env.num_envs, dtype=np.int64)
    failures = np.zeros(env.num_envs, dtype=np.int64)
    successful_transitions: list[dict] = []
    push_records: list[dict] = []
    max_trial_steps = max(300, int(8.0 / env.step_dt))
    rng = np.random.default_rng(42)

    def next_push_step(step: int) -> int:
        interval = rng.uniform(args.push_interval_s[0], args.push_interval_s[1])
        return step + int(math.ceil(interval / env.step_dt))

    for step in range(args.max_disturbed_steps):
        _set_commands(env, speeds)
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            state = extractor.extract()

        reset_mask = dones | state.episode_reset
        for env_id in reset_mask.nonzero(as_tuple=False).flatten().tolist():
            previous[env_id] = None
            if active[env_id] is not None:
                failures[env_id] += 1
                active[env_id] = None
                cooldown_until[env_id] = next_push_step(step)

        for env_id in state.touchdown.nonzero(as_tuple=False).flatten().tolist():
            current = _touchdown_snapshot(state, env_id)
            record = _transition(previous[env_id], current) if previous[env_id] is not None else None
            if record is not None and active[env_id] is not None:
                active[env_id]["transitions"].append(record)
                if len(active[env_id]["transitions"]) >= args.recovery_touchdowns:
                    successful_transitions.extend(active[env_id]["transitions"])
                    active[env_id]["success"] = True
                    push_records.append(active[env_id])
                    successes[env_id] += 1
                    active[env_id] = None
                    cooldown_until[env_id] = next_push_step(step)
            previous[env_id] = current

        for env_id in range(env.num_envs):
            trial = active[env_id]
            if trial is not None and step - trial["start_step"] > max_trial_steps:
                trial["success"] = False
                push_records.append(trial)
                failures[env_id] += 1
                active[env_id] = None
                cooldown_until[env_id] = next_push_step(step)

            if (
                active[env_id] is None
                and successes[env_id] < args.disturbed_trials_per_speed
                and step >= cooldown_until[env_id]
                and previous[env_id] is not None
            ):
                delta = _apply_velocity_jump(env, env_id)
                active[env_id] = {
                    "env_id": env_id,
                    "command_speed": float(speeds[env_id]),
                    "start_step": step,
                    "delta_v_world_xy": delta,
                    "transitions": [],
                    "success": False,
                }

        if step > 0 and step % 1000 == 0:
            print(
                f"[INFO] disturbed step={step}, successes={successes.tolist()}, "
                f"failures={failures.tolist()}",
                flush=True,
            )
        if np.all(successes >= args.disturbed_trials_per_speed):
            break
    else:
        print(
            f"[WARN] disturbed collection reached {args.max_disturbed_steps} steps: "
            f"successes={successes.tolist()}, failures={failures.tolist()}"
        )
    return successful_transitions, push_records, successes, failures


def _summary(values: np.ndarray) -> dict:
    return {
        "count": int(values.shape[0]),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _axis_error_summary(errors: np.ndarray) -> dict:
    return {
        "mean": np.mean(errors, axis=0),
        "median": np.median(errors, axis=0),
        "rmse": np.sqrt(np.mean(errors**2, axis=0)),
        "p95_abs": np.quantile(np.abs(errors), 0.95, axis=0),
        "max_abs": np.max(np.abs(errors), axis=0),
    }


def _period_model(records: list[dict], speeds) -> tuple[dict, dict[float, float]]:
    per_command = {}
    medians = []
    for speed in speeds:
        values = np.asarray(
            [record["step_period"] for record in records if abs(record["command"][0] - speed) < 1.0e-4]
        )
        if values.size == 0:
            raise RuntimeError(f"No nominal step-period samples at vx={speed}")
        per_command[float(speed)] = _summary(values)
        medians.append(float(np.median(values)))
    all_values = np.asarray([record["step_period"] for record in records])
    global_median = float(np.median(all_values))
    relative_span = (max(medians) - min(medians)) / max(global_median, 1.0e-6)
    period_by_speed: dict[float, float]
    if relative_span <= 0.05:
        model = {"type": "constant", "value": global_median, "relative_median_span": relative_span}
        period_by_speed = {float(speed): global_median for speed in speeds}
    else:
        slope, intercept = np.polyfit(np.asarray(speeds), np.asarray(medians), 1)
        model = {
            "type": "linear",
            "value": global_median,
            "slope": float(slope),
            "intercept": float(intercept),
            "relative_median_span": relative_span,
        }
        period_by_speed = {
            float(speed): max(0.10, float(intercept + slope * speed)) for speed in speeds
        }
    model["overall"] = _summary(all_values)
    model["per_command"] = {str(speed): per_command[float(speed)] for speed in speeds}
    return model, period_by_speed


def _rotation_matrix_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _foot_geometry_box(urdf_path: Path, link_name: str, shrink: float) -> list[list[float]]:
    root = ET.parse(urdf_path).getroot()
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        raise RuntimeError(f"Foot link {link_name!r} was not found in {urdf_path}")
    points = []
    for collision in link.findall("collision"):
        box = collision.find("./geometry/box")
        if box is None:
            continue
        size = np.fromstring(box.attrib["size"], sep=" ")
        origin = collision.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rotation = _rotation_matrix_from_rpy(rpy)
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                for sz in (-0.5, 0.5):
                    points.append(xyz + rotation @ (size * np.asarray((sx, sy, sz))))
    if not points:
        raise RuntimeError(f"No box collision geometry found for {link_name!r}")
    points = np.asarray(points)
    lower = np.min(points[:, :2], axis=0)
    upper = np.max(points[:, :2], axis=0)
    inset = 0.5 * shrink * (upper - lower)
    return [[float(lower[0] + inset[0]), float(upper[0] - inset[0])],
            [float(lower[1] + inset[1]), float(upper[1] - inset[1])]]


def _conservative_landing_box(samples: np.ndarray, shrink: float) -> list[list[float]]:
    lower = np.quantile(samples, 0.005, axis=0)
    upper = np.quantile(samples, 0.995, axis=0)
    inset = shrink * (upper - lower)
    if np.any(lower + inset >= upper - inset):
        raise RuntimeError("Landing samples do not span a valid two-dimensional box")
    return [[float(lower[0] + inset[0]), float(upper[0] - inset[0])],
            [float(lower[1] + inset[1]), float(upper[1] - inset[1])]]


def _calibrate(nominal, nominal_transitions, disturbed, checkpoint, speeds, extractor, push_records, successes, failures):
    period_model, periods = _period_model(nominal, speeds)
    heights = np.asarray([record["com_height"] for record in nominal])
    h_eff = float(np.mean(heights))
    omega = math.sqrt(9.81 / h_eff)

    urdf = PROJECT_DIR / "legged_lab/assets/unitree/g1/mjcf/g1_29dof.urdf"
    cop_left = _foot_geometry_box(urdf, "left_ankle_roll_link", args.geometry_shrink)
    cop_right = _foot_geometry_box(urdf, "right_ankle_roll_link", args.geometry_shrink)

    landing_source = disturbed
    landing_temporary = False
    source_label = "successful_stage1b_disturbed_touchdowns"
    side_counts = {
        side: sum(record["support_side"] == side for record in landing_source)
        for side in ("left", "right")
    }
    if min(side_counts.values(), default=0) < 20:
        landing_source = nominal_transitions
        landing_temporary = True
        source_label = "nominal_fallback_due_to_insufficient_successful_disturbed_samples"

    side_samples = {
        side: np.asarray([record["landing"] for record in landing_source if record["support_side"] == side])
        for side in ("left", "right")
    }
    if any(samples.shape[0] < 4 for samples in side_samples.values()):
        raise RuntimeError("Insufficient alternating touchdown samples to calibrate L_left/L_right")
    landing_left = _conservative_landing_box(side_samples["left"], args.landing_shrink)
    landing_right = _conservative_landing_box(side_samples["right"], args.landing_shrink)

    swing_rates = np.asarray(
        [np.abs(record["swing_displacement"]) / record["duration"] for record in landing_source]
    )
    velocity_limit = np.maximum(1.0e-3, 0.9 * np.quantile(swing_rates, 0.95, axis=0))
    step_width = float(np.median(np.abs(np.asarray([record["q"][1] for record in nominal]))))

    errors_b = []
    errors_q = []
    theory_by_command = {}
    for speed in speeds:
        theory_by_command[str(speed)] = theoretical_periodic_state(
            float(speed), 0.0, periods[float(speed)], omega, step_width
        )
    for record in nominal:
        speed = min(speeds, key=lambda candidate: abs(candidate - record["command"][0]))
        side = record["support_side"]
        theory = theory_by_command[str(speed)]
        b_real = (
            record["com_position"][:2]
            + record["com_velocity"][:2] / omega
            - record["support_position"][:2]
        )
        errors_b.append(b_real - np.asarray(theory[f"b_{side}"]))
        errors_q.append(record["q"] - np.asarray(theory[f"q_{side}"]))
    errors_b = np.asarray(errors_b)
    errors_q = np.asarray(errors_q)
    epsilon_b = np.maximum(1.0e-4, np.quantile(np.abs(errors_b), 0.95, axis=0))
    epsilon_q = np.maximum(1.0e-4, np.quantile(np.abs(errors_q), 0.95, axis=0))
    characteristic_scale = np.asarray(
        (
            max(0.05, np.median(np.abs([record["landing"][0] for record in nominal_transitions]))),
            max(0.05, step_width),
        )
    )
    mismatch_ratios = np.maximum(epsilon_b, epsilon_q) / characteristic_scale
    model_mismatch = bool(np.any(mismatch_ratios > 0.5))

    result = {
        "schema_version": 1,
        "T": period_model["value"],
        "step_period": period_model,
        "h_eff": h_eff,
        "omega": omega,
        "w": step_width,
        "C_left": {"x": cop_left[0], "y": cop_left[1]},
        "C_right": {"x": cop_right[0], "y": cop_right[1]},
        "L_left": {"x": landing_left[0], "y": landing_left[1]},
        "L_right": {"x": landing_right[0], "y": landing_right[1]},
        "v_max": {"x": velocity_limit[0], "y": velocity_limit[1]},
        "epsilon_b": {"x": epsilon_b[0], "y": epsilon_b[1]},
        "epsilon_q": {"x": epsilon_q[0], "y": epsilon_q[1]},
        "theoretical_nominal": theory_by_command,
        "nominal_theory_error": {
            "b": _axis_error_summary(errors_b),
            "q": _axis_error_summary(errors_q),
            "model_mismatch": model_mismatch,
            "mismatch_ratio_by_axis": mismatch_ratios,
            "criterion": "max(epsilon_b, epsilon_q) exceeds 50% of the nominal landing scale on an axis",
        },
        "provenance": {
            "task": args.task,
            "checkpoint": str(checkpoint),
            "commands_vx": list(speeds),
            "commands_vy": 0.0,
            "commands_yaw_rate": 0.0,
            "nominal_touchdown_samples": len(nominal),
            "nominal_transition_samples": len(nominal_transitions),
            "successful_disturbed_transition_samples": len(disturbed),
            "successful_disturbed_trials_by_speed": successes,
            "failed_disturbed_trials_by_speed": failures,
            "push_trials_recorded": len(push_records),
            "disturbed_delta_v_world_xy_bounds": [
                [-args.push_max_xy[0], args.push_max_xy[0]],
                [-args.push_max_xy[1], args.push_max_xy[1]],
            ],
            "disturbed_push_interval_s": list(args.push_interval_s),
            "simulated_total_mass": _summary(extractor.total_mass.detach().cpu().numpy()),
        },
        "initialization": {
            "h_eff": {
                "temporary": True,
                "source": "mean_full_body_com_height_above_flat_support_plane",
                "reason": "reliable CoP/contact-point truth is not enabled by the current ContactSensor",
                "measured_height": _summary(heights),
            },
            "C_sigma": {
                "temporary": False,
                "source": str(urdf),
                "method": "foot_collision_box_in_ankle_frame",
                "safety_shrink_fraction": args.geometry_shrink,
                "nominal_cop_containment_checked": False,
                "note": "CoP samples were unavailable; geometry is the requested first initialization.",
            },
            "L_sigma": {
                "temporary": landing_temporary,
                "source": source_label,
                "outlier_trim_each_tail": 0.005,
                "safety_shrink_fraction": args.landing_shrink,
                "sample_counts": {side: int(values.shape[0]) for side, values in side_samples.items()},
                "meaning": "conservative policy-achieved dynamic landing region, not a mechanical limit",
            },
            "v_max": {
                "temporary": landing_temporary,
                "source": source_label,
                "method": "0.9 * 95th percentile of abs(landing-q_start)/swing_time",
            },
        },
    }
    return _native(result)


def main():
    env = None
    try:
        env, policy, checkpoint, speeds = _make_env_and_policy()
        _set_commands(env, speeds)
        extractor = G1PrivilegedStateExtractor(env, G1StateExtractorCfg())
        nominal, nominal_transitions, obs = _collect_nominal(env, policy, extractor, speeds)
        disturbed, push_records, successes, failures = _collect_disturbed(
            env, policy, extractor, speeds, obs
        )
        parameters = _calibrate(
            nominal,
            nominal_transitions,
            disturbed,
            checkpoint,
            speeds,
            extractor,
            push_records,
            successes,
            failures,
        )
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(parameters, stream, sort_keys=False, allow_unicode=True)
        print(f"[INFO] Saved G1 recovery parameters to {output}")
        print(yaml.safe_dump(parameters, sort_keys=False, allow_unicode=True))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
