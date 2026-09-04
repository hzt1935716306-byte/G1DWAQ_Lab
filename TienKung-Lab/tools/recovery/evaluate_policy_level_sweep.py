#!/usr/bin/env python3
"""Compare one native policy on a deterministic Stage2 Level 1--6 push sweep.

The flat symmetric policies and the DWAQ policy keep their native observation
pipelines.  Physics randomization, observation noise, curriculum upgrades, and
training rewards are irrelevant here.  Every invocation reconstructs the same
seeded trial list so reports from different policy families are directly
comparable by ``trial_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


POLICY_TASKS = {
    "baseline": "g1_flat_symmetric",
    "flat": "g1_flat",
    "slope": "g1_slope_nosys_d",
    "slope_sys_d": "g1_slope_sys_d",
    "dwaq_slope": "g1_dwaq_slope_nosys_d",
    "ours": "g1_flat_symmetric",
    "dwaq": "g1_dwaq",
    "stage2_baseline": "g1_flat_symmetric_stage2_baseline",
    "stage2_input": "g1_flat_symmetric_stage2_ours",
}
DWAQ_POLICIES = {"dwaq", "dwaq_slope"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", choices=tuple(POLICY_TASKS), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--levels", type=int, nargs="+", default=(1, 2, 3, 4, 5))
parser.add_argument(
    "--velocity_magnitudes",
    type=float,
    nargs="+",
    default=None,
    help=(
        "Optional fixed root-velocity jump magnitudes in m/s. When set, this "
        "replaces the six curriculum levels and balances fixed directions/commands."
    ),
)
parser.add_argument("--direction_count", type=int, choices=(4, 8), default=8)
parser.add_argument(
    "--command_mode",
    choices=("benchmark", "slope_forward"),
    default="benchmark",
    help="Use the common eight-command benchmark or the slope policy's supported +0.4 m/s command.",
)
parser.add_argument("--slope_degrees", type=float, default=0.0)
parser.add_argument("--episodes_per_level", type=int, default=256)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--prepare_steps", type=int, default=50)
parser.add_argument("--max_recovery_time_s", type=float, default=10.0)
parser.add_argument(
    "--survival_horizon_s",
    type=float,
    default=None,
    help="If set, keep every post-jump rollout alive for this fixed horizon and only score FALL/SURVIVED.",
)
parser.add_argument("--max_steps", type=int, default=12000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--disturbance_pattern",
    choices=("random", "cardinal"),
    default="random",
    help="Random component-wise pushes or balanced fixed +/-x and +/-y pushes.",
)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DWAQOnPolicyRunner, OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.recovery.state_extractor import (  # noqa: E402
    G1PrivilegedStateExtractor,
    G1StateExtractorCfg,
)
from legged_lab.terrains import make_plane_recovery_terrain_cfg  # noqa: E402
from legged_lab.utils import task_registry  # noqa: E402


COMMANDS = (
    (0.4, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (-0.3, 0.0, 0.0),
    (0.4, 0.25, 0.0),
    (0.4, -0.25, 0.0),
    (0.4, 0.0, 0.5),
    (0.4, 0.0, -0.5),
    (0.0, 0.0, 0.0),
)


def _quantiles(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _disable_randomization(env_cfg) -> None:
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

    reset_base = getattr(events, "reset_base", None)
    if reset_base is not None:
        for key in reset_base.params["pose_range"]:
            reset_base.params["pose_range"][key] = (0.0, 0.0)
        for key in reset_base.params["velocity_range"]:
            reset_base.params["velocity_range"][key] = (0.0, 0.0)
    reset_joints = getattr(events, "reset_robot_joints", None)
    if reset_joints is not None:
        reset_joints.params["position_range"] = (1.0, 1.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)


def _make_plans(levels, episodes_per_level: int, level_ratios, maxima, seed: int):
    rng = np.random.default_rng(seed)
    plans = []
    # Reuse the same normalized disturbance at every requested level.  Scaling
    # only by the curriculum ratio makes the level response interpretable.  In
    # random mode this reproduces independent uniform training components;
    # cardinal mode instead provides an explicitly balanced direction test.
    if args.disturbance_pattern == "random":
        normalized_deltas = rng.uniform(-1.0, 1.0, size=(episodes_per_level, 2))
    else:
        cardinal = np.asarray(
            (
                (-0.5, 0.0),
                (0.5, 0.0),
                (0.0, -0.5),
                (0.0, 0.5),
                (-1.0, 0.0),
                (1.0, 0.0),
                (0.0, -1.0),
                (0.0, 1.0),
            ),
            dtype=np.float64,
        )
        normalized_deltas = cardinal[
            np.arange(episodes_per_level, dtype=np.int64) % len(cardinal)
        ]
    for level in levels:
        bound = np.asarray(maxima, dtype=np.float64) * float(level_ratios[level - 1])
        for sample_index in range(episodes_per_level):
            normalized_delta = normalized_deltas[sample_index]
            delta = normalized_delta * bound
            command_index = (
                sample_index % len(COMMANDS)
                if args.disturbance_pattern == "random"
                else (sample_index // 8) % len(COMMANDS)
            )
            plans.append(
                {
                    "trial_id": f"L{level}-{sample_index:05d}",
                    "level": int(level),
                    "level_ratio": float(level_ratios[level - 1]),
                    "command_velocity": list(COMMANDS[command_index]),
                    "normalized_delta_xy": [
                        float(normalized_delta[0]), float(normalized_delta[1])
                    ],
                    "delta_v_world_xy": [float(delta[0]), float(delta[1])],
                }
            )
    rng.shuffle(plans)
    serialized = json.dumps(plans, sort_keys=True, separators=(",", ":")).encode()
    return plans, hashlib.sha256(serialized).hexdigest()


def _make_velocity_magnitude_plans(
    magnitudes: tuple[float, ...],
    episodes_per_magnitude: int,
    direction_count: int,
    commands: tuple[tuple[float, float, float], ...],
    seed: int,
):
    cells_per_repeat = direction_count * len(commands)
    if episodes_per_magnitude % cells_per_repeat != 0:
        raise ValueError(
            "episodes_per_level must be divisible by direction_count * command_count "
            f"({cells_per_repeat}) in velocity-magnitude mode"
        )
    directions = [
        (math.cos(2.0 * math.pi * index / direction_count),
         math.sin(2.0 * math.pi * index / direction_count))
        for index in range(direction_count)
    ]
    plans = []
    for magnitude in magnitudes:
        for sample_index in range(episodes_per_magnitude):
            direction_index = sample_index % direction_count
            command_index = (sample_index // direction_count) % len(commands)
            repeat_index = sample_index // cells_per_repeat
            unit = np.asarray(directions[direction_index], dtype=np.float64)
            delta = float(magnitude) * unit
            plans.append(
                {
                    "trial_id": f"V{magnitude:.3f}-{sample_index:05d}",
                    "push_magnitude_mps": float(magnitude),
                    "direction_index": int(direction_index),
                    "direction_world_xy": [float(unit[0]), float(unit[1])],
                    "command_velocity": list(commands[command_index]),
                    "onset_offset_steps": int(6 * repeat_index),
                    "delta_v_world_xy": [float(delta[0]), float(delta[1])],
                }
            )
    rng = np.random.default_rng(seed)
    rng.shuffle(plans)
    serialized = json.dumps(plans, sort_keys=True, separators=(",", ":")).encode()
    return plans, hashlib.sha256(serialized).hexdigest()


def _set_commands(env, slots) -> None:
    command = env.command_generator.command
    command.zero_()
    for env_id, slot in enumerate(slots):
        if slot is not None:
            command[env_id, :3] = torch.as_tensor(
                slot["plan"]["command_velocity"], dtype=command.dtype, device=env.device
            )
    env.command_generator.is_standing_env[:] = False


def _apply_push(env, env_id: int, delta_v_xy) -> None:
    ids = torch.tensor([env_id], dtype=torch.long, device=env.device)
    root_velocity = env.robot.data.root_vel_w[ids].clone()
    root_velocity[:, :2] += torch.as_tensor(
        delta_v_xy, dtype=root_velocity.dtype, device=env.device
    )
    env.robot.write_root_velocity_to_sim(root_velocity, env_ids=ids)


def _new_slot(plan: dict, prepare_steps: int) -> dict:
    return {
        "plan": plan,
        "status": "preparing",
        "prepare_remaining": int(prepare_steps + plan.get("onset_offset_steps", 0)),
        "start_step": None,
        "start_time": None,
        "touchdowns": 0,
        "last_touchdown_foot": -1,
        "interval_started": False,
        "sample_count": 0,
        "velocity_error_sum": 0.0,
        "abs_tilt_sum": np.zeros(2, dtype=np.float64),
    }


def _complete(slot: dict, outcome: str, state, env_id: int, policy_step: int, reason: str) -> dict:
    episode = dict(slot["plan"])
    sample_count = int(slot["sample_count"])
    mean_abs_tilt = (
        slot["abs_tilt_sum"] / sample_count
        if sample_count > 0
        else np.asarray((math.nan, math.nan), dtype=np.float64)
    )
    episode.update(
        {
            "env_id": int(env_id),
            "outcome": outcome,
            "completion_reason": reason,
            "practical_enter_step": int(slot["touchdowns"]) if outcome == "SUCCESS" else None,
            "touchdown_count": int(slot["touchdowns"]),
            "start_policy_step": int(slot["start_step"]),
            "end_policy_step": int(policy_step),
            "recovery_time_s": float(state.time[env_id].item()) - float(slot["start_time"]),
            "active_sample_count": sample_count,
            "mean_velocity_tracking_error_mps": (
                float(slot["velocity_error_sum"] / sample_count)
                if sample_count > 0
                else None
            ),
            "mean_abs_roll_rad": (
                float(mean_abs_tilt[0]) if sample_count > 0 else None
            ),
            "mean_abs_pitch_rad": (
                float(mean_abs_tilt[1]) if sample_count > 0 else None
            ),
        }
    )
    return episode


def _performance(episodes: list[dict]) -> dict:
    total = len(episodes)
    counts = {
        outcome: sum(item["outcome"] == outcome for item in episodes)
        for outcome in ("SUCCESS", "TIMEOUT", "FALL", "SURVIVED")
    }
    success_steps = [
        item["practical_enter_step"]
        for item in episodes
        if item["outcome"] == "SUCCESS"
    ]
    return {
        "episode_count": total,
        "outcome_counts": counts,
        **{
            f"P{touchdown}": (
                sum(step <= touchdown for step in success_steps) / total if total else None
            )
            for touchdown in range(1, 6)
        },
        "success_rate_P5": counts["SUCCESS"] / total if total else None,
        "non_fall_rate": 1.0 - counts["FALL"] / total if total else None,
        "full_horizon_survival_rate": counts["SURVIVED"] / total if total else None,
        "timeout_rate": counts["TIMEOUT"] / total if total else None,
        "fall_rate": counts["FALL"] / total if total else None,
        "practical_enter_step": _quantiles(success_steps),
        "practical_enter_step_distribution": {
            str(step): success_steps.count(step) for step in range(1, 6)
        },
        "recovery_time_s_all": _quantiles([item["recovery_time_s"] for item in episodes]),
        "recovery_time_s_success": _quantiles(
            [item["recovery_time_s"] for item in episodes if item["outcome"] == "SUCCESS"]
        ),
        "mean_velocity_tracking_error_mps": _quantiles(
            [
                item["mean_velocity_tracking_error_mps"]
                for item in episodes
                if item.get("mean_velocity_tracking_error_mps") is not None
            ]
        ),
    }


def _policy_environment():
    task = POLICY_TASKS[args.policy]
    env_cfg, agent_cfg = task_registry.get_cfgs(task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.seed = args.seed
    env_cfg.scene.max_episode_length_s = 1000.0
    if math.isclose(args.slope_degrees, 0.0, abs_tol=1.0e-12):
        env_cfg.scene.terrain_type = "plane"
        env_cfg.scene.terrain_generator = None
    else:
        env_cfg.scene.terrain_type = "generator"
        env_cfg.scene.terrain_generator = make_plane_recovery_terrain_cfg(
            (float(args.slope_degrees),)
        )
        env_cfg.scene.max_init_terrain_level = 0
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.debug_vis = False
    env_cfg.commands.resampling_time_range = (1.0e9, 1.0e9)
    _disable_randomization(env_cfg)
    if hasattr(env_cfg, "stage2_reward"):
        env_cfg.stage2_reward.enabled = False
        env_cfg.stage2_reward.enable_shared_event_reward = False
        env_cfg.stage2_reward.enable_certificate_reward = False
        env_cfg.stage2_reward.certificate_workers = min(8, args.num_envs)
    if hasattr(args, "device"):
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device

    env = task_registry.get_task_class(task)(env_cfg, args.headless)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.policy in DWAQ_POLICIES:
        runner = DWAQOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    runner.eval_mode()
    return env, runner, checkpoint


def main() -> None:
    levels = tuple(args.levels)
    magnitude_mode = args.velocity_magnitudes is not None
    magnitudes = tuple(sorted(set(args.velocity_magnitudes or ())))
    if not magnitude_mode and (
        not levels
        or any(level < 1 or level > 6 for level in levels)
        or len(set(levels)) != len(levels)
    ):
        raise ValueError("--levels must contain unique values from 1 through 6")
    if magnitude_mode and (not magnitudes or any(value < 0.0 for value in magnitudes)):
        raise ValueError("--velocity_magnitudes must contain unique non-negative values")
    if args.episodes_per_level <= 0 or args.num_envs <= 0 or args.prepare_steps < 0:
        raise ValueError("episode, environment, and preparation counts are invalid")
    if not magnitude_mode and args.disturbance_pattern == "cardinal" and args.episodes_per_level % 64 != 0:
        raise ValueError(
            "cardinal episodes_per_level must be divisible by 64 to balance "
            "8 disturbance cases across 8 commands"
        )
    if args.max_recovery_time_s <= 0.0 or args.max_steps <= 0:
        raise ValueError("time and step limits must be positive")
    if args.survival_horizon_s is not None and args.survival_horizon_s <= 0.0:
        raise ValueError("survival_horizon_s must be positive")

    recovery_cfg, _ = task_registry.get_cfgs("g1_flat_symmetric_stage2_baseline")
    curriculum_cfg = recovery_cfg.push_curriculum
    active_commands = (
        ((0.4, 0.0, 0.0),) if args.command_mode == "slope_forward" else COMMANDS
    )
    if magnitude_mode:
        plans, plan_hash = _make_velocity_magnitude_plans(
            magnitudes,
            args.episodes_per_level,
            args.direction_count,
            active_commands,
            args.seed,
        )
    else:
        plans, plan_hash = _make_plans(
            levels,
            args.episodes_per_level,
            curriculum_cfg.level_ratios,
            curriculum_cfg.stage1b_abs_delta_v_xy,
            args.seed,
        )
    pending = list(reversed(plans))
    env, runner, checkpoint = _policy_environment()
    extractor = G1PrivilegedStateExtractor(
        env, G1StateExtractorCfg(h_eff=0.6884990671277046)
    )
    slots: list[dict | None] = [None] * env.num_envs
    for env_id in range(env.num_envs):
        if pending:
            slots[env_id] = _new_slot(pending.pop(), args.prepare_steps)
    _set_commands(env, slots)

    if args.policy in DWAQ_POLICIES:
        obs, obs_hist = env.get_observations()
        inference_policy = runner.alg.policy.act_inference
    else:
        obs, _ = env.get_observations()
        obs_hist = None
        inference_policy = runner.get_inference_policy(device=env.device)
    completed: list[dict] = []
    nominal_reset_count = 0

    for policy_step in range(args.max_steps):
        _set_commands(env, slots)
        with torch.inference_mode():
            if args.policy in DWAQ_POLICIES:
                actions = inference_policy(obs, obs_hist)
            else:
                actions = inference_policy(obs)
            obs, _, dones, extras = env.step(actions)
            state = extractor.extract()
            if args.policy in DWAQ_POLICIES:
                obs_hist = extras["observations"]["obs_hist"]

        done_mask = dones | state.episode_reset
        for env_id, slot in enumerate(slots):
            if slot is None:
                continue
            if bool(done_mask[env_id].item()):
                if slot["status"] == "active":
                    completed.append(
                        _complete(slot, "FALL", state, env_id, policy_step, "environment_reset")
                    )
                    slots[env_id] = None
                else:
                    nominal_reset_count += 1
                    slot["prepare_remaining"] = args.prepare_steps + int(
                        slot["plan"].get("onset_offset_steps", 0)
                    )

        touchdown_ids = state.touchdown.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        for env_id in touchdown_ids:
            slot = slots[env_id]
            if slot is None or slot["status"] != "active":
                continue
            slot["touchdowns"] += 1
            if args.survival_horizon_s is not None:
                continue
            sample_count = int(slot["sample_count"])
            has_interval = bool(slot["interval_started"] and sample_count > 0)
            alternating = slot["last_touchdown_foot"] < 0 or (
                int(state.touchdown_foot[env_id].item()) != slot["last_touchdown_foot"]
            )
            good_cycle = False
            if has_interval:
                mean_velocity_error = slot["velocity_error_sum"] / sample_count
                mean_abs_tilt = slot["abs_tilt_sum"] / sample_count
                good_cycle = bool(
                    alternating
                    and mean_velocity_error <= curriculum_cfg.mean_velocity_error_threshold
                    and mean_abs_tilt[0] <= curriculum_cfg.mean_abs_roll_threshold
                    and mean_abs_tilt[1] <= curriculum_cfg.mean_abs_pitch_threshold
                )
            slot["last_touchdown_foot"] = int(state.touchdown_foot[env_id].item())
            slot["interval_started"] = True
            slot["sample_count"] = 0
            slot["velocity_error_sum"] = 0.0
            slot["abs_tilt_sum"][:] = 0.0
            if good_cycle:
                completed.append(
                    _complete(slot, "SUCCESS", state, env_id, policy_step, "practical_good_cycle")
                )
                slots[env_id] = None
            elif slot["touchdowns"] >= curriculum_cfg.max_recovery_touchdowns:
                completed.append(
                    _complete(slot, "TIMEOUT", state, env_id, policy_step, "five_touchdowns")
                )
                slots[env_id] = None

        for env_id, slot in enumerate(slots):
            if slot is None or slot["status"] != "active":
                continue
            elapsed = float(state.time[env_id].item()) - float(slot["start_time"])
            if args.survival_horizon_s is not None and elapsed >= args.survival_horizon_s:
                completed.append(
                    _complete(slot, "SURVIVED", state, env_id, policy_step, "full_survival_horizon")
                )
                slots[env_id] = None
                continue
            if args.survival_horizon_s is None and elapsed >= args.max_recovery_time_s:
                completed.append(
                    _complete(slot, "TIMEOUT", state, env_id, policy_step, "wall_time_limit")
                )
                slots[env_id] = None
                continue
            velocity_error = torch.linalg.vector_norm(
                state.com_velocity[env_id, :2] - state.command_velocity[env_id, :2]
            )
            slot["sample_count"] += 1
            slot["velocity_error_sum"] += float(velocity_error.item())
            slot["abs_tilt_sum"] += np.abs(
                state.root_roll_pitch[env_id].detach().cpu().numpy()
            )

        for env_id, slot in enumerate(slots):
            if slot is None and pending:
                slots[env_id] = _new_slot(pending.pop(), args.prepare_steps)
                slot = slots[env_id]
            if slot is None or slot["status"] != "preparing":
                continue
            slot["prepare_remaining"] -= 1
            if slot["prepare_remaining"] <= 0:
                _apply_push(env, env_id, slot["plan"]["delta_v_world_xy"])
                slot["status"] = "active"
                slot["start_step"] = policy_step
                slot["start_time"] = float(state.time[env_id].item())

        if (policy_step + 1) % 250 == 0:
            if magnitude_mode:
                counts = {
                    magnitude: sum(
                        math.isclose(item["push_magnitude_mps"], magnitude) for item in completed
                    )
                    for magnitude in magnitudes
                }
            else:
                counts = {
                    level: sum(item["level"] == level for item in completed) for level in levels
                }
            print(
                f"[policy-level-sweep] policy={args.policy} step={policy_step + 1}/{args.max_steps} "
                f"completed={len(completed)}/{len(plans)} per_group={counts}",
                flush=True,
            )
        if len(completed) == len(plans):
            break

    if magnitude_mode:
        per_group = {
            f"{magnitude:g}": _performance(
                [
                    item
                    for item in completed
                    if math.isclose(item["push_magnitude_mps"], magnitude)
                ]
            )
            for magnitude in magnitudes
        }
    else:
        per_group = {
            str(level): _performance([item for item in completed if item["level"] == level])
            for level in levels
        }
    report = {
        "schema_version": 1,
        "policy": args.policy,
        "native_task": POLICY_TASKS[args.policy],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "inference_only": True,
        "common_protocol": {
            "levels": None if magnitude_mode else list(levels),
            "level_ratios": list(curriculum_cfg.level_ratios),
            "stage1b_abs_delta_v_xy": list(curriculum_cfg.stage1b_abs_delta_v_xy),
            "velocity_magnitudes_mps": list(magnitudes) if magnitude_mode else None,
            "velocity_jump_frame": "world",
            "direction_count": args.direction_count if magnitude_mode else None,
            "episodes_per_level": args.episodes_per_level,
            "commands": [list(command) for command in active_commands],
            "command_mode": args.command_mode,
            "slope_degrees": float(args.slope_degrees),
            "seed": args.seed,
            "disturbance_pattern": (
                "fixed_magnitude_balanced_directions"
                if magnitude_mode
                else args.disturbance_pattern
            ),
            "trial_plan_sha256": plan_hash,
            "flat_plane": math.isclose(args.slope_degrees, 0.0, abs_tol=1.0e-12),
            "terrain_geometry": (
                "flat plane"
                if math.isclose(args.slope_degrees, 0.0, abs_tol=1.0e-12)
                else "continuous x-aligned plane"
            ),
            "observation_noise": False,
            "physics_randomization": False,
            "max_recovery_touchdowns": curriculum_cfg.max_recovery_touchdowns,
            "survival_horizon_s": args.survival_horizon_s,
            "mean_velocity_error_threshold": curriculum_cfg.mean_velocity_error_threshold,
            "mean_abs_roll_threshold": curriculum_cfg.mean_abs_roll_threshold,
            "mean_abs_pitch_threshold": curriculum_cfg.mean_abs_pitch_threshold,
        },
        "num_envs": args.num_envs,
        "prepare_steps": args.prepare_steps,
        "max_recovery_time_s": args.max_recovery_time_s,
        "policy_steps_executed": policy_step + 1,
        "planned_episode_count": len(plans),
        "completed_episode_count": len(completed),
        "pending_episode_count": len(plans) - len(completed),
        "nominal_reset_count": nominal_reset_count,
        "actor_observation_shape": list(obs.shape),
        "overall_performance": _performance(completed),
        "performance_by_velocity_magnitude": per_group if magnitude_mode else None,
        "performance_by_level": None if magnitude_mode else per_group,
        "episodes": completed,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(report)
    summary.pop("episodes")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[policy-level-sweep] wrote {output}", flush=True)
    evaluator = getattr(env, "_certificate_evaluator", None)
    if evaluator is not None:
        evaluator.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    os._exit(0)
