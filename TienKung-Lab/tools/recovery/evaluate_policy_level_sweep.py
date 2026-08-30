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
from pathlib import Path

from isaaclab.app import AppLauncher


POLICY_TASKS = {
    "baseline": "g1_flat_symmetric",
    "ours": "g1_flat_symmetric",
    "dwaq": "g1_dwaq",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", choices=tuple(POLICY_TASKS), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--levels", type=int, nargs="+", default=(1, 2, 3, 4, 5))
parser.add_argument("--episodes_per_level", type=int, default=256)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--prepare_steps", type=int, default=50)
parser.add_argument("--max_recovery_time_s", type=float, default=10.0)
parser.add_argument("--max_steps", type=int, default=12000)
parser.add_argument("--seed", type=int, default=42)
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
    # only by the curriculum ratio makes the level response interpretable and
    # still reproduces the training distribution (independent uniform x/y).
    normalized_deltas = rng.uniform(-1.0, 1.0, size=(episodes_per_level, 2))
    for level in levels:
        bound = np.asarray(maxima, dtype=np.float64) * float(level_ratios[level - 1])
        for sample_index in range(episodes_per_level):
            normalized_delta = normalized_deltas[sample_index]
            delta = normalized_delta * bound
            plans.append(
                {
                    "trial_id": f"L{level}-{sample_index:05d}",
                    "level": int(level),
                    "level_ratio": float(level_ratios[level - 1]),
                    "command_velocity": list(COMMANDS[sample_index % len(COMMANDS)]),
                    "normalized_delta_xy": [
                        float(normalized_delta[0]), float(normalized_delta[1])
                    ],
                    "delta_v_world_xy": [float(delta[0]), float(delta[1])],
                }
            )
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
        "prepare_remaining": int(prepare_steps),
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
        }
    )
    return episode


def _performance(episodes: list[dict]) -> dict:
    total = len(episodes)
    counts = {
        outcome: sum(item["outcome"] == outcome for item in episodes)
        for outcome in ("SUCCESS", "TIMEOUT", "FALL")
    }
    success_steps = [
        item["practical_enter_step"]
        for item in episodes
        if item["outcome"] == "SUCCESS"
    ]
    return {
        "episode_count": total,
        "outcome_counts": counts,
        "success_rate_P5": counts["SUCCESS"] / total if total else None,
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
    }


def _policy_environment():
    task = POLICY_TASKS[args.policy]
    env_cfg, agent_cfg = task_registry.get_cfgs(task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.seed = args.seed
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.scene.terrain_type = "plane"
    env_cfg.scene.terrain_generator = None
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.debug_vis = False
    env_cfg.commands.resampling_time_range = (1.0e9, 1.0e9)
    _disable_randomization(env_cfg)
    if hasattr(args, "device"):
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device

    env = task_registry.get_task_class(task)(env_cfg, args.headless)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.policy == "dwaq":
        runner = DWAQOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    runner.eval_mode()
    return env, runner, checkpoint


def main() -> None:
    levels = tuple(args.levels)
    if not levels or any(level < 1 or level > 6 for level in levels) or len(set(levels)) != len(levels):
        raise ValueError("--levels must contain unique values from 1 through 6")
    if args.episodes_per_level <= 0 or args.num_envs <= 0 or args.prepare_steps < 0:
        raise ValueError("episode, environment, and preparation counts are invalid")
    if args.max_recovery_time_s <= 0.0 or args.max_steps <= 0:
        raise ValueError("time and step limits must be positive")

    recovery_cfg, _ = task_registry.get_cfgs("g1_flat_symmetric_stage2_baseline")
    curriculum_cfg = recovery_cfg.push_curriculum
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

    if args.policy == "dwaq":
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
            if args.policy == "dwaq":
                actions = inference_policy(obs, obs_hist)
            else:
                actions = inference_policy(obs)
            obs, _, dones, extras = env.step(actions)
            state = extractor.extract()
            if args.policy == "dwaq":
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
                    slot["prepare_remaining"] = args.prepare_steps

        touchdown_ids = state.touchdown.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        for env_id in touchdown_ids:
            slot = slots[env_id]
            if slot is None or slot["status"] != "active":
                continue
            slot["touchdowns"] += 1
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
            if float(state.time[env_id].item()) - float(slot["start_time"]) >= args.max_recovery_time_s:
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
            counts = {
                level: sum(item["level"] == level for item in completed) for level in levels
            }
            print(
                f"[policy-level-sweep] policy={args.policy} step={policy_step + 1}/{args.max_steps} "
                f"completed={len(completed)}/{len(plans)} per_level={counts}",
                flush=True,
            )
        if len(completed) == len(plans):
            break

    per_level = {
        str(level): _performance([item for item in completed if item["level"] == level])
        for level in levels
    }
    report = {
        "schema_version": 1,
        "policy": args.policy,
        "native_task": POLICY_TASKS[args.policy],
        "checkpoint": str(checkpoint),
        "inference_only": True,
        "common_protocol": {
            "levels": list(levels),
            "level_ratios": list(curriculum_cfg.level_ratios),
            "stage1b_abs_delta_v_xy": list(curriculum_cfg.stage1b_abs_delta_v_xy),
            "episodes_per_level": args.episodes_per_level,
            "commands": [list(command) for command in COMMANDS],
            "seed": args.seed,
            "trial_plan_sha256": plan_hash,
            "flat_plane": True,
            "observation_noise": False,
            "physics_randomization": False,
            "max_recovery_touchdowns": curriculum_cfg.max_recovery_touchdowns,
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
        "performance_by_level": per_level,
        "episodes": completed,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(report)
    summary.pop("episodes")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[policy-level-sweep] wrote {output}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
