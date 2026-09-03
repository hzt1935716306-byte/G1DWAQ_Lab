#!/usr/bin/env python3
"""Inference-only robustness evaluation at one fixed Stage2 curriculum level."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import traceback
from types import MethodType

from isaaclab.app import AppLauncher


PROJECT_DIR = Path(__file__).resolve().parents[2]
TASKS = (
    "g1_flat_symmetric_stage2_baseline",
    "g1_flat_symmetric_stage2_ours",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=TASKS, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--level", type=int, choices=range(1, 7), default=5)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--push_interval_s", type=float, nargs=2, default=(0.25, 0.35))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--diagnostic_trace", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.utils import task_registry  # noqa: E402


EPS = 1.0e-12


def _quantiles(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "p90": None}
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


class _EpisodeCapture:
    def __init__(self, env) -> None:
        self.env = env
        self.active: dict[int, dict] = {}
        self.completed: list[dict] = []
        self.push_count = 0
        self._attach()

    @staticmethod
    def _ids(tensor: torch.Tensor) -> list[int]:
        return [int(value) for value in tensor.detach().cpu().tolist()]

    def _attach(self) -> None:
        env = self.env
        original_push = env._on_curriculum_push
        original_outcome = env._record_outcomes
        capture = self

        def wrapped_push(this, env_ids, delta_v_xy, sampled_level_indices):
            original_push(env_ids, delta_v_xy, sampled_level_indices)
            ids = capture._ids(env_ids)
            levels = capture._ids(sampled_level_indices)
            deltas = delta_v_xy.detach().cpu().tolist()
            commands = this.command_generator.command[env_ids, :3].detach().cpu().tolist()
            start_step = int(this.sim_step_counter // this.cfg.sim.decimation)
            for env_id, level_index, delta, command in zip(ids, levels, deltas, commands):
                capture.push_count += 1
                capture.active[env_id] = {
                    "episode_id": capture.push_count,
                    "env_id": env_id,
                    "level": level_index + 1,
                    "delta_v_xy": [float(delta[0]), float(delta[1])],
                    "command_velocity": [float(value) for value in command],
                    "start_policy_step": start_step,
                }

        def wrapped_outcome(
            this,
            env_ids,
            outcome,
            practical_enter_steps=None,
            clear_event_buffers=False,
        ):
            ids = capture._ids(env_ids)
            enter_steps = (
                [None] * len(ids)
                if practical_enter_steps is None
                else capture._ids(practical_enter_steps)
            )
            end_step = int(this.sim_step_counter // this.cfg.sim.decimation)
            touchdowns = capture._ids(this._recovery_touchdowns[env_ids])
            for env_id, enter_step, touchdown_count in zip(ids, enter_steps, touchdowns):
                episode = capture.active.pop(env_id, None)
                if episode is None:
                    continue
                elapsed_steps = max(0, end_step - int(episode["start_policy_step"]))
                episode.update(
                    {
                        "outcome": outcome.value,
                        "practical_enter_step": enter_step,
                        "touchdown_count": touchdown_count,
                        "end_policy_step": end_step,
                        "recovery_time_s": elapsed_steps * float(this.step_dt),
                    }
                )
                capture.completed.append(episode)
            return original_outcome(
                env_ids,
                outcome,
                practical_enter_steps,
                clear_event_buffers=clear_event_buffers,
            )

        env._on_curriculum_push = MethodType(wrapped_push, env)
        env._record_outcomes = MethodType(wrapped_outcome, env)


def _performance(episodes: list[dict]) -> dict:
    total = len(episodes)
    counts = {
        name: sum(episode["outcome"] == name for episode in episodes)
        for name in ("SUCCESS", "TIMEOUT", "FALL")
    }
    success_steps = [
        int(episode["practical_enter_step"])
        for episode in episodes
        if episode["outcome"] == "SUCCESS"
    ]
    p_by_touchdown = {
        f"P{touchdown}": (
            sum(step <= touchdown for step in success_steps) / total if total else None
        )
        for touchdown in range(1, 6)
    }
    return {
        "episode_count": total,
        "outcome_counts": counts,
        **p_by_touchdown,
        "success_rate_P5": counts["SUCCESS"] / total if total else None,
        "timeout_rate": counts["TIMEOUT"] / total if total else None,
        "fall_rate": counts["FALL"] / total if total else None,
        "practical_enter_step": _quantiles(success_steps),
        "practical_enter_step_distribution": {
            str(step): success_steps.count(step) for step in range(1, 6)
        },
        "recovery_time_s_all": _quantiles(
            [episode["recovery_time_s"] for episode in episodes]
        ),
        "recovery_time_s_success": _quantiles(
            [
                episode["recovery_time_s"]
                for episode in episodes
                if episode["outcome"] == "SUCCESS"
            ]
        ),
    }


def _signed_axis_performance(episodes: list[dict], axis: int) -> dict[str, dict]:
    """Split outcomes by the sign of one world-frame push component."""

    labels = ("negative", "positive")
    return {
        label: _performance(
            [
                episode
                for episode in episodes
                if (episode["delta_v_xy"][axis] < 0.0) == (label == "negative")
            ]
        )
        for label in labels
    }


def _stratified_performance(episodes: list[dict], bound: float) -> list[dict]:
    bins = ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.000001))
    result = []
    for low, high in bins:
        selected = []
        for episode in episodes:
            normalized = max(abs(value) for value in episode["delta_v_xy"]) / bound
            if low <= normalized < high:
                selected.append(episode)
        performance = _performance(selected)
        performance.update({"normalized_max_abs_delta_v_min": low, "normalized_max_abs_delta_v_max": high})
        result.append(performance)
    return result


def main() -> None:
    if args.diagnostic_trace:
        faulthandler.enable()
        faulthandler.dump_traceback_later(60.0, repeat=True)
    if args.push_interval_s[0] <= 0.0 or args.push_interval_s[1] < args.push_interval_s[0]:
        raise ValueError("push interval must be positive and ordered")

    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.seed = args.seed
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.domain_rand.events.push_robot.interval_range_s = tuple(args.push_interval_s)
    env_cfg.push_curriculum.adaptive_upgrades_enabled = False
    env_cfg.push_curriculum.easy_sample_probability = 0.0
    env_cfg.stage2_reward.certificate_workers = min(8, args.num_envs)
    # Rewards are irrelevant during inference.  Disable them equally for both
    # policies while preserving the real certificate evaluator required by the
    # Input-only actor context.
    env_cfg.stage2_reward.enabled = False
    env_cfg.stage2_reward.enable_shared_event_reward = False
    env_cfg.stage2_reward.enable_certificate_reward = False
    if hasattr(args, "device"):
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device

    env_class = task_registry.get_task_class(args.task)
    print("[fixed-level-eval] before environment construction", flush=True)
    env = env_class(env_cfg, args.headless)
    print("[fixed-level-eval] after environment construction", flush=True)
    env.push_curriculum.level_index = args.level - 1
    env.push_curriculum.level_start_iteration = 0
    env.push_curriculum.current_learning_iteration = 0
    env.push_curriculum_level = env.push_curriculum.level
    env.push_curriculum_level_ratio = env.push_curriculum.level_ratio
    env.push_curriculum_max_xy = torch.tensor(
        env.push_curriculum.current_abs_delta_v_xy,
        dtype=torch.float32,
        device=env.device,
    )
    capture = _EpisodeCapture(env)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    print("[fixed-level-eval] before runner construction", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print("[fixed-level-eval] before checkpoint load", flush=True)
    runner.load(str(checkpoint), load_optimizer=False)
    print("[fixed-level-eval] after checkpoint load", flush=True)
    policy = runner.get_inference_policy(device=env.device)
    obs, extras = env.get_observations()

    for step in range(args.steps):
        with torch.inference_mode():
            obs, _, _, extras = env.step(policy(obs))
        if (step + 1) % 250 == 0:
            print(
                f"[fixed-level-eval] task={args.task} level={args.level} "
                f"step={step + 1}/{args.steps} completed={len(capture.completed)} "
                f"active={len(capture.active)}",
                flush=True,
            )

    episodes = capture.completed
    bound = float(env.push_curriculum.bounds_for_level_index(args.level - 1)[0])
    deltas = np.asarray([episode["delta_v_xy"] for episode in episodes], dtype=np.float64)
    report = {
        "schema_version": 1,
        "task": args.task,
        "checkpoint": str(checkpoint),
        "fixed_curriculum_level": args.level,
        "level_ratio": env.push_curriculum.level_ratios[args.level - 1],
        "abs_delta_v_xy_bound": list(
            env.push_curriculum.bounds_for_level_index(args.level - 1)
        ),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "push_interval_s": list(args.push_interval_s),
        "seed": args.seed,
        "inference_only": True,
        "actor_observation_shape": list(obs.shape),
        "critic_observation_shape": list(extras["observations"]["critic"].shape),
        "certificate_evaluator_enabled": env._certificate_evaluator is not None,
        "completed_episode_count": len(episodes),
        "incomplete_episode_count_at_stop": len(capture.active),
        "performance": _performance(episodes),
        "performance_by_normalized_max_abs_delta_v": _stratified_performance(episodes, bound),
        "performance_by_push_direction": {
            "delta_v_x": _signed_axis_performance(episodes, axis=0),
            "delta_v_y": _signed_axis_performance(episodes, axis=1),
        },
        "disturbance": {
            "delta_v_x": _quantiles(deltas[:, 0] if deltas.size else []),
            "delta_v_y": _quantiles(deltas[:, 1] if deltas.size else []),
            "abs_delta_v_x": _quantiles(np.abs(deltas[:, 0]) if deltas.size else []),
            "abs_delta_v_y": _quantiles(np.abs(deltas[:, 1]) if deltas.size else []),
            "norm_delta_v_xy": _quantiles(
                np.linalg.norm(deltas, axis=1) if deltas.size else []
            ),
            "max_abs_delta_v_xy": _quantiles(
                np.max(np.abs(deltas), axis=1) if deltas.size else []
            ),
        },
        "episodes": episodes,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(report)
    summary.pop("episodes")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[fixed-level-eval] wrote {output}", flush=True)
    if env._certificate_evaluator is not None:
        env._certificate_evaluator.close()
    if args.diagnostic_trace:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Isaac shutdown may block and hide the actual evaluation error.  Emit
        # it first, then let the standalone process release resources on exit.
        traceback.print_exc()
        os._exit(1)
    # This standalone diagnostic has already persisted its report and closed
    # the certificate workers.  Isaac's shutdown callback can otherwise block
    # indefinitely after a successful headless run.
    os._exit(0)
