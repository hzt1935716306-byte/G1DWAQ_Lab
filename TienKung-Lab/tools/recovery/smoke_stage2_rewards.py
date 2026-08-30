#!/usr/bin/env python3
"""Minimal online rollout smoke test for one Stage2 A/B reward task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


TASKS = (
    "g1_flat_symmetric_stage2_baseline",
    "g1_flat_symmetric_stage2_ours",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=TASKS, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--checkpoint", type=Path, required=True)
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


def main() -> None:
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.max_episode_length_s = 100.0
    env_cfg.domain_rand.events.push_robot.interval_range_s = (0.20, 0.30)
    env_cfg.push_curriculum.k_min_iterations = 2
    env_cfg.push_curriculum.k_max_iterations = 5
    env_cfg.push_curriculum.statistics_window_episodes = 8
    env_cfg.stage2_reward.certificate_workers = min(16, args.num_envs)

    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, args.headless)
    env.set_num_steps_per_learning_iteration(1)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    obs, extras = env.get_observations()
    actor_shape = tuple(obs.shape)
    critic_shape = tuple(extras["observations"]["critic"].shape)
    reward_finite = True
    max_abs_policy_reward = 0.0
    shared_event_values: list[float] = []
    certificate_event_values: list[float] = []
    episode_ratios: list[float] = []
    absolute_episode_ratios: list[float] = []
    recovery_event_values: list[float] = []
    outcome_counts = {"SUCCESS": 0, "TIMEOUT": 0, "FALL": 0}
    one_shot_buffers_clear = True
    push_event_reward_zero = True
    completion_closes_recovery = True
    soft_reward_scaling_observed = False
    observed_levels = [env.push_curriculum.level]

    for _ in range(args.steps):
        with torch.inference_mode():
            obs, rewards, _, extras = env.step(policy(obs))
        reward_finite &= bool(torch.all(torch.isfinite(rewards)).item())
        max_abs_policy_reward = max(max_abs_policy_reward, float(torch.max(torch.abs(rewards)).item()))
        observed_levels.append(env.push_curriculum.level)

        event = env._last_event_rewards
        shared = event["recovery_shared_total"]
        certificate = event["recovery_certificate"]
        shared_event_values.extend(shared[shared != 0.0].detach().cpu().tolist())
        certificate_event_values.extend(certificate[certificate != 0.0].detach().cpu().tolist())
        recovery_total = event["recovery_total"]
        recovery_event_values.extend(
            recovery_total[recovery_total != 0.0].detach().cpu().tolist()
        )
        if torch.any(env._last_push_started_mask):
            push_event_reward_zero &= bool(
                torch.all(event["recovery_total"][env._last_push_started_mask] == 0.0).item()
            )
        one_shot_buffers_clear &= all(
            bool(torch.all(buffer == 0.0).item())
            for buffer in (
                env._event_touchdown_cost,
                env._event_success,
                env._event_timeout,
                env._event_certificate,
            )
        )

        soft_mask = env._last_soft_scaling_recovery_mask
        for multipliers in env._last_soft_reward_multipliers.values():
            if torch.any(soft_mask):
                soft_reward_scaling_observed |= bool(
                    torch.any(multipliers[soft_mask] < 1.0).item()
                )

        for episode in extras.get("recovery_episode_rewards", []):
            episode_ratios.append(episode["recovery_to_locomotion_abs_ratio"])
            absolute_episode_ratios.append(
                episode["absolute_recovery_to_locomotion_ratio"]
            )
            completion_closes_recovery &= not bool(
                env.recovery_active[int(episode["env_id"])].item()
            )

    is_ours = args.task.endswith("_ours")
    curriculum_snapshot = env.push_curriculum.snapshot()
    outcome_counts = {
        outcome: sum(
            int(level_stats[field])
            for level_stats in curriculum_snapshot["all_level_statistics"]
        )
        for outcome, field in (
            ("SUCCESS", "success"),
            ("TIMEOUT", "timeout"),
            ("FALL", "fall"),
        )
    }
    baseline_solver_disabled = env._certificate_evaluator is None if not is_ours else None
    ours_solver_enabled = env._certificate_evaluator is not None if is_ours else None
    certificate_always_zero = len(certificate_event_values) == 0
    certificate_nonzero = len(certificate_event_values) > 0
    absolute_ratio_median = (
        float(np.median(absolute_episode_ratios)) if absolute_episode_ratios else None
    )
    if float(env_cfg.stage2_reward.event_scale) != 0.50:
        raise RuntimeError("the certificate event_scale must be exactly 0.50")
    if env.push_curriculum.push_sample_count <= 0:
        raise RuntimeError("curriculum produced no physical push")
    if sum(outcome_counts.values()) <= 0:
        raise RuntimeError("recovery state machine completed no episode")
    if not reward_finite or max_abs_policy_reward > 250.0:
        raise RuntimeError("policy reward was non-finite or exploded")
    if not one_shot_buffers_clear or not push_event_reward_zero:
        raise RuntimeError("one-shot event buffer or push-zero invariant failed")
    if soft_reward_scaling_observed or env._soft_reward_term_indices:
        raise RuntimeError("the clean A/B must use the original locomotion reward")
    if not completion_closes_recovery:
        raise RuntimeError("SUCCESS/TIMEOUT/FALL did not close recovery immediately")
    if is_ours and not certificate_nonzero:
        raise RuntimeError("Ours produced no certificate reward")
    if not is_ours and not certificate_always_zero:
        raise RuntimeError("Baseline produced certificate reward")
    if is_ours and not ours_solver_enabled:
        raise RuntimeError("Ours certificate evaluator is disabled")
    if not is_ours and not baseline_solver_disabled:
        raise RuntimeError("Baseline must not construct or call a certificate evaluator")
    if shared_event_values:
        raise RuntimeError("touchdown/success/timeout rewards must be zero for both tasks")
    if is_ours and (
        not env._stage2_reward_enabled
        or env._shared_event_reward_enabled
        or not env._certificate_reward_enabled
        or env._soft_reward_scaling_enabled
    ):
        raise RuntimeError("Ours is not configured as original reward plus certificate only")
    if not is_ours and (
        env._stage2_reward_enabled
        or env._shared_event_reward_enabled
        or env._certificate_reward_enabled
        or env._soft_reward_scaling_enabled
    ):
        raise RuntimeError("Baseline is not using the untouched original reward")

    report = {
        "task": args.task,
        "checkpoint_loaded": str(checkpoint),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "certificate_event_scale": float(env_cfg.stage2_reward.event_scale),
        "actor_observation_shape": actor_shape,
        "critic_observation_shape": critic_shape,
        "action_size": env.num_actions,
        "push_sample_count": env.push_curriculum.push_sample_count,
        "observed_levels": sorted(set(observed_levels)),
        "final_level": env.push_curriculum.level,
        "upgrade_history": env.push_curriculum.upgrade_history,
        "outcome_counts": outcome_counts,
        "shared_event_count": len(shared_event_values),
        "certificate_event_count": env._certificate_event_count,
        "certificate_nonzero_event_count": env._certificate_nonzero_event_count,
        "certificate_solver_statistics": (
            env._certificate_evaluator.statistics
            if env._certificate_evaluator is not None
            else None
        ),
        "shared_event_abs_max": max((abs(value) for value in shared_event_values), default=0.0),
        "certificate_event_abs_max": max(
            (abs(value) for value in certificate_event_values), default=0.0
        ),
        "max_abs_single_recovery_event_reward": max(
            (abs(value) for value in recovery_event_values), default=0.0
        ),
        "max_abs_policy_reward": max_abs_policy_reward,
        "episode_recovery_to_locomotion_abs_ratio": {
            "count": len(episode_ratios),
            "mean": float(np.mean(episode_ratios)) if episode_ratios else None,
            "median": float(np.median(episode_ratios)) if episode_ratios else None,
            "min": min(episode_ratios) if episode_ratios else None,
            "max": max(episode_ratios) if episode_ratios else None,
        },
        "episode_absolute_reward_dominance_ratio": {
            "definition": "sum(abs(recovery_event_reward)) / "
            "(sum(abs(locomotion_reward_during_recovery)) + eps)",
            "count": len(absolute_episode_ratios),
            "median": absolute_ratio_median,
            "p75": (
                float(np.quantile(absolute_episode_ratios, 0.75))
                if absolute_episode_ratios
                else None
            ),
            "p90": (
                float(np.quantile(absolute_episode_ratios, 0.90))
                if absolute_episode_ratios
                else None
            ),
        },
        "checks": {
            "checkpoint_loaded": True,
            "curriculum_operational": env.push_curriculum.push_sample_count > 0,
            "state_machine_completed_episode": sum(outcome_counts.values()) > 0,
            "original_locomotion_reward_unchanged": not soft_reward_scaling_observed
            and not env._soft_reward_term_indices,
            "shared_touchdown_success_timeout_always_zero": not shared_event_values,
            "event_buffers_one_shot": one_shot_buffers_clear,
            "push_event_reward_zero": push_event_reward_zero,
            "completion_closes_recovery": completion_closes_recovery,
            "baseline_certificate_always_zero": certificate_always_zero if not is_ours else None,
            "baseline_certificate_solver_disabled": baseline_solver_disabled,
            "ours_certificate_nonzero": certificate_nonzero if is_ours else None,
            "ours_certificate_solver_enabled": ours_solver_enabled,
            "policy_rewards_finite": reward_finite,
            "reward_magnitude_not_exploded": max_abs_policy_reward <= 250.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if env._certificate_evaluator is not None:
        env._certificate_evaluator.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
