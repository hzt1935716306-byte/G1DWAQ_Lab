#!/usr/bin/env python3
"""Small Isaac Lab integration smoke test for the Stage2 curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.utils import task_registry  # noqa: E402


def main() -> None:
    env_cfg, agent_cfg = task_registry.get_cfgs("g1_flat_symmetric_recovery")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.max_episode_length_s = 100.0
    env_cfg.domain_rand.events.push_robot.interval_range_s = (0.20, 0.30)
    curriculum_cfg = env_cfg.push_curriculum
    curriculum_cfg.k_min_iterations = 2
    curriculum_cfg.k_max_iterations = 5
    curriculum_cfg.statistics_window_episodes = 8

    env_class = task_registry.get_task_class("g1_flat_symmetric_recovery")
    env = env_class(env_cfg, args.headless)
    # Test-only mapping: one environment step is one curriculum iteration.
    env.set_num_steps_per_learning_iteration(1)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint.expanduser().resolve()), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    obs, extras = env.get_observations()
    actor_shape = tuple(obs.shape)
    critic_shape = tuple(extras["observations"]["critic"].shape)
    reward_finite = True
    observed_levels = [env.push_curriculum.level]
    for _ in range(args.steps):
        with torch.inference_mode():
            obs, rewards, _, extras = env.step(policy(obs))
        reward_finite = reward_finite and bool(torch.all(torch.isfinite(rewards)).item())
        observed_levels.append(env.push_curriculum.level)

    controller = env.push_curriculum
    level_stats = [item.to_dict() for item in controller.level_statistics]
    completed_episodes = sum(item["recovery_episodes"] for item in level_stats)
    max_abs_recorded_delta = float(torch.max(torch.abs(env.last_push_delta_v_xy)).item())
    report = {
        "checkpoint_loaded": str(args.checkpoint.expanduser().resolve()),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "actor_observation_shape": actor_shape,
        "critic_observation_shape": critic_shape,
        "action_size": env.num_actions,
        "push_sample_count": controller.push_sample_count,
        "easy_push_sample_count": controller.easy_push_sample_count,
        "easy_push_sample_fraction": controller.easy_sample_fraction,
        "max_abs_recorded_delta_v": max_abs_recorded_delta,
        "observed_levels": sorted(set(observed_levels)),
        "final_level": controller.level,
        "upgrade_history": controller.upgrade_history,
        "completed_recovery_episodes": completed_episodes,
        "level_recovery_statistics": level_stats,
        "active_recoveries_at_end": int(env.recovery_active.sum().item()),
        "recovery_reward_weight": curriculum_cfg.recovery_reward_weight,
        "logged_recovery_reward": extras["log"]["Recovery/reward"],
        "policy_rewards_finite": reward_finite,
    }
    if controller.push_sample_count <= 0:
        raise RuntimeError("curriculum event did not produce any physical push samples")
    if max_abs_recorded_delta <= 0.0 or max_abs_recorded_delta > 1.0 + 1.0e-6:
        raise RuntimeError(f"recorded velocity increment is invalid: {max_abs_recorded_delta}")
    if completed_episodes <= 0:
        raise RuntimeError("recovery state machine completed no episodes")
    if curriculum_cfg.recovery_reward_weight != 0.0 or report["logged_recovery_reward"] != 0.0:
        raise RuntimeError("recovery reward must stay zero")
    if not reward_finite:
        raise RuntimeError("policy reward contained a non-finite value")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
