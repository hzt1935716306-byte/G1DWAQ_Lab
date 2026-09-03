#!/usr/bin/env python3
"""Short inference smoke/performance test for the 963-D recovery-context actor."""

from __future__ import annotations

import argparse
import faulthandler
import json
from pathlib import Path
import time
from types import SimpleNamespace

from isaaclab.app import AppLauncher


TASKS = (
    "g1_flat_symmetric_stage2_baseline",
    "g1_flat_symmetric_stage2_ours",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=TASKS, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mirror_samples", type=int, default=0)
parser.add_argument("--certificate_workers", type=int, default=1)
parser.add_argument(
    "--certificate_executor",
    choices=("sequential", "thread", "subprocess"),
    default="subprocess",
)
parser.add_argument("--diagnostic_trace", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.envs.g1.g1_symmetry import compute_symmetric_states  # noqa: E402
from legged_lab.recovery.checkpoint_migration import warm_start_context_policy  # noqa: E402
from legged_lab.utils import task_registry  # noqa: E402


def _old_actor_mean(model_state: dict[str, torch.Tensor], observations: torch.Tensor) -> torch.Tensor:
    value = observations
    layer_indices = (0, 2, 4, 6)
    for layer_index in layer_indices:
        value = F.linear(
            value,
            model_state[f"actor.{layer_index}.weight"].to(value.device),
            model_state[f"actor.{layer_index}.bias"].to(value.device),
        )
        if layer_index != layer_indices[-1]:
            value = F.elu(value)
    return value


def _mirror_report(env, samples: list[dict]) -> dict:
    if not samples:
        return {"sample_count": 0, "valid_pair_count": 0}
    b = torch.tensor([item["b"] for item in samples], dtype=torch.float32, device=env.device)
    q = torch.tensor([item["q"] for item in samples], dtype=torch.float32, device=env.device)
    command = torch.tensor(
        [item["command"] for item in samples], dtype=torch.float32, device=env.device
    )
    phase = torch.tensor(
        [item["phase"] for item in samples], dtype=torch.float32, device=env.device
    )
    support = torch.tensor(
        [item["support_is_left"] for item in samples], dtype=torch.bool, device=env.device
    )
    b[:, 1] *= -1.0
    q[:, 1] *= -1.0
    command[:, 1:] *= -1.0
    mirrored_state = SimpleNamespace(
        b=b,
        q=q,
        command_velocity=command,
        phase=phase,
        support_is_left=~support,
    )
    ids = torch.arange(len(samples), device=env.device)
    mirror_n, mirror_margin, mirror_valid = env._certificate_evaluator.evaluate_with_validity(
        mirrored_state, ids
    )
    original_n = torch.tensor(
        [item["n_min"] for item in samples], dtype=torch.long, device=env.device
    )
    original_margin = torch.tensor(
        [item["margin"] for item in samples], dtype=torch.float32, device=env.device
    )
    original_valid = torch.tensor(
        [item["valid"] for item in samples], dtype=torch.bool, device=env.device
    )
    valid_pair = original_valid & mirror_valid
    n_diff = original_n[valid_pair] != mirror_n[valid_pair]
    margin_diff = torch.abs(original_margin[valid_pair] - mirror_margin[valid_pair])
    return {
        "sample_count": len(samples),
        "valid_pair_count": int(valid_pair.sum().item()),
        "N_mismatch_rate": float(n_diff.float().mean().item()) if n_diff.numel() else None,
        "margin_abs_difference_median": (
            float(torch.quantile(margin_diff, 0.5).item()) if margin_diff.numel() else None
        ),
        "margin_abs_difference_p90": (
            float(torch.quantile(margin_diff, 0.9).item()) if margin_diff.numel() else None
        ),
    }


def _distribution(values, *, scale: float = 1.0) -> dict[str, float | int | None]:
    """Return compact smoke-test distribution statistics for JSON reporting."""

    array = np.asarray(tuple(values), dtype=np.float64) * scale
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def main() -> None:
    if args.diagnostic_trace:
        faulthandler.enable()
        faulthandler.dump_traceback_later(30.0, repeat=True)
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.seed = args.seed
    env_cfg.scene.max_episode_length_s = 60.0
    env_cfg.noise.add_noise = False
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = (0.5, 0.5)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    # Test-only short interval so a minimal smoke observes push-without-touchdown
    # leakage cases.  The training configuration remains unchanged at 10--15 s.
    env_cfg.domain_rand.events.push_robot.interval_range_s = (0.02, 0.04)
    env_cfg.push_curriculum.adaptive_upgrades_enabled = False
    env_cfg.push_curriculum.easy_sample_probability = 0.0
    env_cfg.stage2_reward.certificate_workers = args.certificate_workers
    env_cfg.stage2_reward.certificate_executor = args.certificate_executor
    env_cfg.sim.device = args.device
    agent_cfg.device = args.device

    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, args.headless)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    migration = warm_start_context_policy(runner.alg.policy, args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)
    obs, extras = env.get_observations()

    if obs.shape != (args.num_envs, 963):
        raise RuntimeError(f"actor observation shape mismatch: {tuple(obs.shape)}")
    if extras["observations"]["critic"].shape != (args.num_envs, 1010):
        raise RuntimeError(
            f"critic observation shape mismatch: {tuple(extras['observations']['critic'].shape)}"
        )
    if torch.count_nonzero(env._recovery_context).item() != 0:
        raise RuntimeError("recovery context is not zero after initialization")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with torch.inference_mode():
        old_mean = _old_actor_mean(saved["model_state_dict"], obs[:, :960])
        new_mean = runner.alg.policy.actor(obs)
    action_mean_max_abs_error = float(torch.max(torch.abs(old_mean - new_mean)).item())
    if action_mean_max_abs_error > 5.0e-5:
        raise RuntimeError(
            f"Stage1A warm-start action mismatch: {action_mean_max_abs_error:.3e}"
        )

    mirrored, _ = compute_symmetric_states(env, obs=obs, obs_type="policy")
    if mirrored.shape != (2 * args.num_envs, 963):
        raise RuntimeError(f"symmetry output shape mismatch: {tuple(mirrored.shape)}")
    torch.testing.assert_close(mirrored[args.num_envs :, -3:], obs[:, -3:])

    baseline = env._recovery_context_mode == "zero"
    if baseline and env._certificate_evaluator is not None:
        raise RuntimeError("zero-context baseline instantiated a certificate evaluator")

    mirror_samples: list[dict] = []
    refresh_events = 0
    valid_refresh_events = 0
    held_step_rows = 0
    push_without_touchdown = 0
    multiple_refresh_envs = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    context_changes_at_refresh = 0
    started = time.perf_counter()
    for step_index in range(args.steps):
        if args.diagnostic_trace:
            print(f"[ContextSmoke] step={step_index} before_policy", flush=True)
        context_before = env._recovery_context.clone()
        with torch.inference_mode():
            actions = policy(obs)
            if args.diagnostic_trace:
                print(f"[ContextSmoke] step={step_index} before_env_step", flush=True)
            obs, _, dones, _ = env.step(actions)
            if args.diagnostic_trace:
                print(
                    f"[ContextSmoke] step={step_index} after_env_step "
                    f"refresh={int(env._context_refresh_mask.sum().item())}",
                    flush=True,
                )
        context_after = env._recovery_context
        refresh_mask = env._context_refresh_mask.clone()
        touchdown_mask = env._context_touchdown_mask.clone()
        push_mask = env._last_push_started_mask.clone()

        if baseline:
            if torch.count_nonzero(context_after).item() != 0:
                raise RuntimeError("zero-context baseline produced a nonzero context")
            if torch.any(refresh_mask):
                raise RuntimeError("zero-context baseline attempted a certificate refresh")
        else:
            if torch.any(context_after[:, 0] < 0.0) or torch.any(context_after[:, 0] > 1.0):
                raise RuntimeError("N_norm left [0, 1]")
            if torch.any(context_after[:, 1] < -1.0) or torch.any(context_after[:, 1] > 1.0):
                raise RuntimeError("margin_norm left [-1, 1]")
            hold_mask = ~refresh_mask & ~dones
            if not torch.equal(context_after[hold_mask], context_before[hold_mask]):
                raise RuntimeError("recovery context changed between touchdowns")
            held_step_rows += int(hold_mask.sum().item())
            refresh_events += int(refresh_mask.sum().item())
            valid_refresh_events += int(env.current_certificate_valid[refresh_mask].sum().item())
            multiple_refresh_envs += refresh_mask.to(torch.long)
            context_changes_at_refresh += int(
                torch.any(context_after[refresh_mask] != context_before[refresh_mask], dim=-1)
                .sum()
                .item()
            )
            if len(mirror_samples) < args.mirror_samples and torch.any(refresh_mask):
                state = env._last_recovery_state
                for env_id in refresh_mask.nonzero(as_tuple=False).flatten().tolist():
                    if len(mirror_samples) >= args.mirror_samples:
                        break
                    mirror_samples.append(
                        {
                            "b": state.b[env_id].detach().cpu().tolist(),
                            "q": state.q[env_id].detach().cpu().tolist(),
                            "command": state.command_velocity[env_id].detach().cpu().tolist(),
                            "phase": float(state.phase[env_id].item()),
                            "support_is_left": bool(state.support_is_left[env_id].item()),
                            "n_min": int(env._touchdown_certificate_cache_n[env_id].item()),
                            "margin": float(
                                env._touchdown_certificate_cache_margin[env_id].item()
                            ),
                            "valid": bool(
                                env._touchdown_certificate_cache_valid[env_id].item()
                            ),
                        }
                    )

        push_only = push_mask & ~touchdown_mask
        if torch.any(push_only & refresh_mask):
            raise RuntimeError("a mid-step push refreshed the actor context")
        push_without_touchdown += int(push_only.sum().item())
        if torch.any(dones) and torch.count_nonzero(context_after[dones]).item() != 0:
            raise RuntimeError("done/reset did not clear recovery context")

    elapsed = time.perf_counter() - started
    if not baseline and refresh_events == 0:
        raise RuntimeError("input-only smoke did not observe a touchdown refresh")
    if push_without_touchdown == 0:
        raise RuntimeError("smoke did not observe a push without a simultaneous touchdown")

    reset_candidates = env.current_certificate_valid.nonzero(as_tuple=False).flatten()[:8]
    explicit_reset_checked = int(reset_candidates.numel())
    if reset_candidates.numel():
        if args.diagnostic_trace:
            print("[ContextSmoke] before_reset_cleanup", flush=True)
        # Exercise the exact cleanup invoked by G1RecoveryEnv.reset without an
        # extra scene.reset()/sim.forward() at the end of a performance smoke.
        # Full physics resets are already part of BaseEnv's normal done path.
        env._clear_recovery_context(reset_candidates)
        if torch.count_nonzero(env._recovery_context[reset_candidates]).item() != 0:
            raise RuntimeError("reset cleanup did not clear recovery context")
        if torch.any(env.current_certificate_valid[reset_candidates]):
            raise RuntimeError("reset cleanup did not clear certificate validity")
        if torch.any(env.current_n_min[reset_candidates] != -1):
            raise RuntimeError("reset cleanup did not clear raw N_min")
        if torch.count_nonzero(env.current_margin[reset_candidates]).item() != 0:
            raise RuntimeError("reset cleanup did not clear raw margin")

    if args.diagnostic_trace:
        print("[ContextSmoke] before_mirror_report", flush=True)
    mirror = _mirror_report(env, mirror_samples) if not baseline else {"sample_count": 0}
    evaluator_stats = (
        dict(env._certificate_evaluator.statistics)
        if env._certificate_evaluator is not None
        else {}
    )
    policy_step_seconds = float(env._policy_step_total_seconds)
    certificate_seconds = float(env._context_refresh_total_seconds)
    report = {
        "task": args.task,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "actor_observation_dim": int(obs.shape[-1]),
        "critic_observation_dim": int(extras["observations"]["critic"].shape[-1]),
        "context_mode": env._recovery_context_mode,
        "certificate_reward_enabled": env._certificate_reward_enabled,
        "shared_event_reward_enabled": env._shared_event_reward_enabled,
        "certificate_evaluator_initialized": env._certificate_evaluator is not None,
        "warm_start": migration,
        "action_mean_max_abs_error": action_mean_max_abs_error,
        "symmetry_passed": True,
        "refresh_events": refresh_events,
        "valid_refresh_events": valid_refresh_events,
        "zero_order_hold_checked_rows": held_step_rows,
        "envs_with_multiple_refreshes": int((multiple_refresh_envs >= 2).sum().item()),
        "context_changes_at_refresh": context_changes_at_refresh,
        "push_without_touchdown_events": push_without_touchdown,
        "explicit_reset_checked_envs": explicit_reset_checked,
        "wall_seconds": elapsed,
        "environment_steps_per_second": args.num_envs * args.steps / elapsed,
        "policy_step_wall_time_ms": _distribution(
            env._policy_step_latencies_s, scale=1000.0
        ),
        "certificate_context_performance": {
            "refresh_batches": env._context_refresh_batches,
            "touchdown_count": refresh_events,
            "evaluations": env._context_refresh_evaluations,
            "valid_evaluations": valid_refresh_events,
            "valid_rate": (
                valid_refresh_events / refresh_events if refresh_events else None
            ),
            "total_solve_seconds": env._context_refresh_total_seconds,
            "evaluations_per_second": (
                env._context_refresh_evaluations / env._context_refresh_total_seconds
                if env._context_refresh_total_seconds > 0.0
                else None
            ),
            "mean_batch_latency_ms": (
                1000.0 * env._context_refresh_total_seconds / env._context_refresh_batches
                if env._context_refresh_batches
                else None
            ),
            "batch_size": _distribution(env._context_refresh_batch_sizes),
            "batch_latency_ms": _distribution(
                env._context_refresh_latencies_s, scale=1000.0
            ),
            "certificate_wall_time_fraction": (
                certificate_seconds / policy_step_seconds
                if policy_step_seconds > 0.0
                else None
            ),
            "retry_count": int(evaluator_stats.get("retried_results", 0)),
            "fallback_count": int(evaluator_stats.get("fallbacks", 0)),
            "evaluator_statistics": evaluator_stats,
        },
        "certificate_mirror_diagnostic": mirror,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if env._certificate_evaluator is not None:
        if args.diagnostic_trace:
            print("[ContextSmoke] before_evaluator_close", flush=True)
        env._certificate_evaluator.close()
    if args.diagnostic_trace:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
