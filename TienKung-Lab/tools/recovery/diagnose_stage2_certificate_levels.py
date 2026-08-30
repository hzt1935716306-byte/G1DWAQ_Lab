#!/usr/bin/env python3
"""Collect Stage2 certificate diagnostics at one fixed curriculum level.

This is an inference-only diagnostic.  It attaches temporary wrappers to the
existing recovery environment so that the already-computed touchdown
certificate and reward values can be copied before the one-shot buffers are
cleared.  It does not change the policy, reward, certificate, state machine, or
curriculum implementation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import MethodType

from isaaclab.app import AppLauncher


PROJECT_DIR = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="g1_flat_symmetric_stage2_ours")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--level", type=int, choices=(1, 2), required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--push_interval_s", type=float, nargs=2, default=(0.25, 0.35))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_DIR / "tools/recovery/generated/stage2_certificate_level_diagnostic.json",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.recovery.certificate import RecoverabilityConfig  # noqa: E402
from legged_lab.utils import task_registry  # noqa: E402


EPS = 1.0e-12


def _native(value):
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _quantiles(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "p05": None, "p25": None, "p75": None, "p90": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _dominance_quantiles(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "p75": None, "p90": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _fraction(count: int, total: int) -> float | None:
    return count / total if total else None


def _spearman(x, y) -> dict:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.size < 2:
        return {"count": int(x_array.size), "rho": None, "pvalue": None, "reason": "fewer_than_two_samples"}
    if np.unique(x_array).size < 2 or np.unique(y_array).size < 2:
        return {
            "count": int(x_array.size),
            "rho": None,
            "pvalue": None,
            "reason": "one_input_is_constant",
        }
    result = spearmanr(x_array, y_array)
    return {
        "count": int(x_array.size),
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
        "reason": None,
    }


class _TouchdownCapture:
    """Copy diagnostic values around existing environment methods."""

    def __init__(self, env) -> None:
        self.env = env
        self.active: dict[int, dict] = {}
        self.completed: list[dict] = []
        self.push_count = 0
        self._attach()

    @staticmethod
    def _floats(tensor: torch.Tensor) -> list[float]:
        return [float(value) for value in tensor.detach().cpu().tolist()]

    @staticmethod
    def _ints(tensor: torch.Tensor) -> list[int]:
        return [int(value) for value in tensor.detach().cpu().tolist()]

    def _attach(self) -> None:
        env = self.env
        original_push = env._on_curriculum_push
        original_initial = env._initialize_pending_certificates
        original_touchdown = env._queue_touchdown_rewards
        original_outcome = env._record_outcomes

        capture = self

        def wrapped_push(this, env_ids, delta_v_xy, sampled_level_indices):
            original_push(env_ids, delta_v_xy, sampled_level_indices)
            ids = capture._ints(env_ids)
            levels = capture._ints(sampled_level_indices)
            deltas = delta_v_xy.detach().cpu().tolist()
            for env_id, level_index, delta in zip(ids, levels, deltas):
                capture.push_count += 1
                capture.active[env_id] = {
                    "episode_id": capture.push_count,
                    "env_id": env_id,
                    "level": level_index + 1,
                    "delta_v_xy": [float(delta[0]), float(delta[1])],
                    "initial": None,
                    "touchdowns": [],
                    "certificate_abs_sum": 0.0,
                    "shared_abs_sum": 0.0,
                    "total_event_abs_sum": 0.0,
                }

        def wrapped_initial(this, state, dones):
            pending = this._pending_initial_certificate & this.recovery_active & ~dones
            ids = pending.nonzero(as_tuple=False).flatten()
            original_initial(state, dones)
            for env_id in capture._ints(ids):
                episode = capture.active.get(env_id)
                if episode is None or int(this._certificate_n[env_id].item()) < 0:
                    continue
                episode["initial"] = {
                    "touchdown": 0,
                    "N": int(this._certificate_n[env_id].item()),
                    "margin": float(this._certificate_margin[env_id].item()),
                    "phi": float(this._certificate_phi_previous[env_id].item()),
                }

        def wrapped_touchdown(this, state, env_ids, entered_ids, timeout_ids):
            ids = capture._ints(env_ids)
            old_n = capture._ints(this._certificate_n[env_ids])
            old_margin = capture._floats(this._certificate_margin[env_ids])
            old_phi = capture._floats(this._certificate_phi_previous[env_ids])
            touchdown_numbers = capture._ints(this._recovery_touchdowns[env_ids])
            entered = set(capture._ints(entered_ids))
            timed_out = set(capture._ints(timeout_ids))

            original_touchdown(state, env_ids, entered_ids, timeout_ids)

            new_n = capture._ints(this._certificate_n[env_ids])
            new_margin = capture._floats(this._certificate_margin[env_ids])
            new_phi = capture._floats(this._certificate_phi_previous[env_ids])
            certificate_rewards = capture._floats(this._event_certificate[env_ids])
            shared_rewards = capture._floats(
                this._event_touchdown_cost[env_ids]
                + this._event_success[env_ids]
                + this._event_timeout[env_ids]
            )

            for values in zip(
                ids,
                touchdown_numbers,
                old_n,
                old_margin,
                old_phi,
                new_n,
                new_margin,
                new_phi,
                certificate_rewards,
                shared_rewards,
            ):
                (
                    env_id,
                    touchdown,
                    n_old,
                    margin_old,
                    phi_old,
                    n_new,
                    margin_new,
                    phi_new,
                    certificate_reward,
                    shared_reward,
                ) = values
                episode = capture.active.get(env_id)
                if episode is None:
                    continue
                total_reward = shared_reward + certificate_reward
                episode["touchdowns"].append(
                    {
                        "touchdown": touchdown,
                        "N_old": n_old,
                        "margin_old": margin_old,
                        "phi_old": phi_old,
                        "N_new": n_new,
                        "margin_new": margin_new,
                        "phi_new": phi_new,
                        "delta_phi": phi_new - phi_old,
                        "r_certificate": certificate_reward,
                        "r_shared": shared_reward,
                        "r_total_event": total_reward,
                        "practical_success": env_id in entered,
                        "timeout": env_id in timed_out,
                    }
                )
                episode["certificate_abs_sum"] += abs(certificate_reward)
                episode["shared_abs_sum"] += abs(shared_reward)
                episode["total_event_abs_sum"] += abs(total_reward)

        def wrapped_outcome(
            this,
            env_ids,
            outcome,
            practical_enter_steps=None,
            clear_event_buffers=False,
        ):
            ids = capture._ints(env_ids)
            enter_steps = (
                [None] * len(ids)
                if practical_enter_steps is None
                else capture._ints(practical_enter_steps)
            )
            for env_id, enter_step in zip(ids, enter_steps):
                episode = capture.active.pop(env_id, None)
                if episode is None:
                    continue
                episode.update(
                    {
                        "outcome": outcome.value,
                        "practical_enter_step": enter_step,
                        "locomotion_signed_sum": float(
                            this._episode_locomotion_reward[env_id].item()
                        ),
                        "locomotion_abs_sum": float(
                            this._episode_locomotion_reward_abs_sum[env_id].item()
                        ),
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
        env._initialize_pending_certificates = MethodType(wrapped_initial, env)
        env._queue_touchdown_rewards = MethodType(wrapped_touchdown, env)
        env._record_outcomes = MethodType(wrapped_outcome, env)


def _analyze(capture: _TouchdownCapture, metadata: dict) -> dict:
    episodes = [episode for episode in capture.completed if episode["initial"] is not None]
    touchdowns = [sample for episode in episodes for sample in episode["touchdowns"]]
    n_values = [int(sample["N_new"]) for sample in touchdowns]
    margins = [float(sample["margin_new"]) for sample in touchdowns]
    finite_margins = [margin for n, margin in zip(n_values, margins) if n <= 5]
    over_margins = [margin for n, margin in zip(n_values, margins) if n > 5]

    n_counts = {str(n): n_values.count(n) for n in range(6)}
    n_counts[">5"] = sum(n > 5 for n in n_values)
    n_distribution = {
        key: {"count": count, "probability": _fraction(count, len(n_values))}
        for key, count in n_counts.items()
    }

    cap_cfg = RecoverabilityConfig()
    margin_report = {
        "N_le_5_positive_rho": _quantiles(finite_margins),
        "N_gt_5_negative_eta": _quantiles(over_margins),
        "rho_solver_cap": cap_cfg.rho_max,
        "eta_solver_cap": cap_cfg.eta_max,
        "rho_at_cap_fraction": (
            float(np.mean(np.isclose(finite_margins, cap_cfg.rho_max, atol=1.0e-6)))
            if finite_margins
            else None
        ),
        "eta_at_solver_cap_fraction": (
            float(np.mean(np.isclose(over_margins, -cap_cfg.eta_max, atol=1.0e-6)))
            if over_margins
            else None
        ),
        # Phi clips margin / 2 to -1 for N>5, so all eta <= -2 carry
        # the same potential-margin information even though the LP cap is -3.
        "eta_at_phi_clip_fraction": (
            float(np.mean(np.asarray(over_margins) <= -2.0 + 1.0e-9))
            if over_margins
            else None
        ),
    }

    certificate_rewards = np.asarray(
        [sample["r_certificate"] for sample in touchdowns], dtype=np.float64
    )
    abs_certificate_rewards = np.abs(certificate_rewards)
    zero_mask = np.isclose(certificate_rewards, 0.0, atol=EPS, rtol=0.0)
    reward_report = _quantiles(abs_certificate_rewards)
    reward_report.update(
        {
            "mean": float(np.mean(abs_certificate_rewards)) if abs_certificate_rewards.size else None,
            "positive_fraction": (
                float(np.mean(certificate_rewards > EPS)) if certificate_rewards.size else None
            ),
            "zero_fraction": float(np.mean(zero_mask)) if certificate_rewards.size else None,
            "negative_fraction": (
                float(np.mean(certificate_rewards < -EPS)) if certificate_rewards.size else None
            ),
        }
    )

    transitions = [
        (int(sample["N_old"]), int(sample["N_new"]))
        for sample in touchdowns
        if int(sample["N_old"]) >= 0 and int(sample["N_new"]) >= 0
    ]
    decreasing = sum(new < old for old, new in transitions)
    unchanged = sum(new == old for old, new in transitions)
    increasing = sum(new > old for old, new in transitions)
    transition_report = {
        "count": len(transitions),
        "N_new_lt_N_old": {"count": decreasing, "fraction": _fraction(decreasing, len(transitions))},
        "N_new_eq_N_old": {"count": unchanged, "fraction": _fraction(unchanged, len(transitions))},
        "N_new_gt_N_old": {"count": increasing, "fraction": _fraction(increasing, len(transitions))},
        "over_horizon": {
            "gt5_to_gt5": {
                "count": sum(old > 5 and new > 5 for old, new in transitions),
                "fraction": _fraction(
                    sum(old > 5 and new > 5 for old, new in transitions), len(transitions)
                ),
            },
            "gt5_to_le5": {
                "count": sum(old > 5 and new <= 5 for old, new in transitions),
                "fraction": _fraction(
                    sum(old > 5 and new <= 5 for old, new in transitions), len(transitions)
                ),
            },
            "le5_to_gt5": {
                "count": sum(old <= 5 and new > 5 for old, new in transitions),
                "fraction": _fraction(
                    sum(old <= 5 and new > 5 for old, new in transitions), len(transitions)
                ),
            },
        },
    }

    successes = [episode for episode in episodes if episode["outcome"] == "SUCCESS"]
    patterns = {
        "3_to_4": lambda old, new: old == 3 and new == 4,
        "3_to_gt5": lambda old, new: old == 3 and new > 5,
        "4_to_gt5": lambda old, new: old == 4 and new > 5,
        "2_to_3": lambda old, new: old == 2 and new == 3,
    }
    success_rise_count = 0
    pattern_counts = {name: 0 for name in patterns}
    success_delta_phi = []
    success_actual_progress = []
    phi_states = []
    negative_actual_remaining_states = []
    sign_progress = {"positive": [], "zero": [], "negative": []}
    for episode in successes:
        pairs = [
            (int(sample["N_old"]), int(sample["N_new"]))
            for sample in episode["touchdowns"]
        ]
        success_rise_count += int(any(new > old for old, new in pairs))
        for name, predicate in patterns.items():
            pattern_counts[name] += int(any(predicate(old, new) for old, new in pairs))

        final_touchdown = int(episode["practical_enter_step"])
        initial = episode["initial"]
        phi_states.append(float(initial["phi"]))
        negative_actual_remaining_states.append(-float(final_touchdown))
        for sample in episode["touchdowns"]:
            touchdown = int(sample["touchdown"])
            if touchdown > final_touchdown:
                continue
            actual_remaining_old = final_touchdown - (touchdown - 1)
            actual_remaining_new = final_touchdown - touchdown
            actual_progress = actual_remaining_old - actual_remaining_new
            delta_phi = float(sample["delta_phi"])
            success_delta_phi.append(delta_phi)
            success_actual_progress.append(float(actual_progress))
            sign = "positive" if delta_phi > EPS else "negative" if delta_phi < -EPS else "zero"
            sign_progress[sign].append(float(actual_progress))
            phi_states.append(float(sample["phi_new"]))
            negative_actual_remaining_states.append(-float(actual_remaining_new))

    success_report = {
        "episode_count": len(successes),
        "episodes_with_any_N_rise": {
            "count": success_rise_count,
            "fraction": _fraction(success_rise_count, len(successes)),
        },
        "patterns": {
            name: {"count": count, "fraction": _fraction(count, len(successes))}
            for name, count in pattern_counts.items()
        },
    }
    progress_report = {
        "definition": "actual_remaining_steps_old - actual_remaining_steps_new on successful trajectories",
        "delta_phi_vs_actual_progress_spearman": _spearman(
            success_delta_phi, success_actual_progress
        ),
        "actual_progress_by_delta_phi_sign": {
            sign: {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
            }
            for sign, values in sign_progress.items()
        },
        "metric_limitation": (
            "For a successful trajectory measured only at adjacent touchdowns, "
            "actual_remaining_steps decreases by exactly one at every transition; "
            "therefore the requested Spearman correlation is undefined."
        ),
        "auxiliary_phi_vs_negative_actual_remaining_spearman": _spearman(
            phi_states, negative_actual_remaining_states
        ),
    }

    complete_reward_episodes = [
        episode for episode in episodes if episode["locomotion_abs_sum"] > EPS
    ]
    dominance = {
        "certificate_abs_sum": _quantiles(
            [episode["certificate_abs_sum"] for episode in complete_reward_episodes]
        ),
        "shared_abs_sum": _quantiles(
            [episode["shared_abs_sum"] for episode in complete_reward_episodes]
        ),
        "locomotion_abs_sum": _quantiles(
            [episode["locomotion_abs_sum"] for episode in complete_reward_episodes]
        ),
        "certificate_over_locomotion": _dominance_quantiles(
            [
                episode["certificate_abs_sum"] / (episode["locomotion_abs_sum"] + EPS)
                for episode in complete_reward_episodes
            ]
        ),
        "certificate_over_shared": _dominance_quantiles(
            [
                episode["certificate_abs_sum"] / (episode["shared_abs_sum"] + EPS)
                for episode in complete_reward_episodes
            ]
        ),
        "total_recovery_over_locomotion": _dominance_quantiles(
            [
                episode["total_event_abs_sum"] / (episode["locomotion_abs_sum"] + EPS)
                for episode in complete_reward_episodes
            ]
        ),
    }

    outcomes = {
        name: sum(episode["outcome"] == name for episode in episodes)
        for name in ("SUCCESS", "TIMEOUT", "FALL")
    }
    return {
        "schema_version": 1,
        **metadata,
        "completed_episode_count": len(episodes),
        "incomplete_episode_count_at_stop": len(capture.active),
        "outcome_counts": outcomes,
        "touchdown_count": len(touchdowns),
        "N_distribution": n_distribution,
        "P_N_gt_5": _fraction(n_counts[">5"], len(n_values)),
        "margin": margin_report,
        "certificate_reward": reward_report,
        "N_transition": transition_report,
        "successful_trajectories": success_report,
        "actual_progress": progress_report,
        "actor_certificate_context": {
            "present": False,
            "reason": (
                "G1RecoveryEnv keeps certificate state simulator-privileged and does not "
                "append N/margin/valid to actor or critic observations."
            ),
            "Var_N_norm": None,
            "Var_margin_norm": None,
            "N_norm_equals_1_fraction": None,
        },
        "reward_dominance": dominance,
        "certificate_solver_statistics": (
            capture.env._certificate_evaluator.statistics
            if capture.env._certificate_evaluator is not None
            else None
        ),
        "episodes": episodes,
    }


def main() -> None:
    if not args.task.endswith("_ours"):
        raise ValueError("the touchdown certificate diagnostic requires the Ours task")
    if args.push_interval_s[0] <= 0.0 or args.push_interval_s[1] < args.push_interval_s[0]:
        raise ValueError("push interval must be positive and ordered")

    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.seed = args.seed
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.domain_rand.events.push_robot.interval_range_s = tuple(args.push_interval_s)
    env_cfg.push_curriculum.adaptive_upgrades_enabled = False
    env_cfg.push_curriculum.easy_sample_probability = 0.0
    env_cfg.stage2_reward.certificate_workers = min(16, args.num_envs)
    if hasattr(args, "device"):
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device

    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, args.headless)
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
    capture = _TouchdownCapture(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    obs, extras = env.get_observations()

    for step in range(args.steps):
        with torch.inference_mode():
            obs, _, _, extras = env.step(policy(obs))
        if (step + 1) % 200 == 0:
            print(
                f"[diagnostic] level={args.level} step={step + 1}/{args.steps} "
                f"completed={len(capture.completed)} active={len(capture.active)}",
                flush=True,
            )

    report = _analyze(
        capture,
        {
            "task": args.task,
            "checkpoint": str(checkpoint),
            "fixed_curriculum_level": args.level,
            "level_ratio": env.push_curriculum.level_ratios[args.level - 1],
            "num_envs": args.num_envs,
            "steps": args.steps,
            "push_interval_s": list(args.push_interval_s),
            "seed": args.seed,
            "event_scale": float(env_cfg.stage2_reward.event_scale),
            "actor_observation_shape": list(obs.shape),
            "critic_observation_shape": list(extras["observations"]["critic"].shape),
            "inference_only": True,
            "algorithms_modified": False,
        },
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_native(report), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(report)
    summary.pop("episodes")
    print(json.dumps(_native(summary), ensure_ascii=False, indent=2), flush=True)
    print(f"[diagnostic] wrote {output}", flush=True)
    if env._certificate_evaluator is not None:
        env._certificate_evaluator.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
