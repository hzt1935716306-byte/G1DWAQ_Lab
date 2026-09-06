#!/usr/bin/env python3
"""Direct-step and config-diff smoke for the Plane V1 RL-only ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.envs.g1.g1_symmetry import compute_symmetric_states  # noqa: E402
from legged_lab.recovery.rl_only_matched_contract import (  # noqa: E402
    RL_ONLY_EXPECTED_ENV_DIFFERENCE_PREFIXES,
    unexpected_config_differences,
)
from legged_lab.utils import task_registry  # noqa: E402


TASK = "g1_plane_v1_rl_only_matched"
REFERENCE_TASK = "g1_plane_v1_estimator_context_no_reward_matched"


def _certificate_worker_children() -> tuple[int, ...]:
    """Return live clean-solver descendants without importing process libraries."""

    workers = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            status = (entry / "status").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        parent_line = next((line for line in status.splitlines() if line.startswith("PPid:")), "")
        parent = int(parent_line.split()[1]) if parent_line else -1
        if parent == os.getpid() and "legged_lab.recovery.certificate_worker" in command:
            workers.append(int(entry.name))
    return tuple(sorted(workers))


def _assert_push_semantics(reference_cfg, rl_only_cfg) -> None:
    reference = reference_cfg.domain_rand.events.push_robot
    rl_only = rl_only_cfg.domain_rand.events.push_robot
    assert tuple(reference.interval_range_s) == (10.0, 15.0)
    assert tuple(rl_only.interval_range_s) == (10.0, 15.0)
    assert reference_cfg.push_curriculum.enable_push_curriculum is False
    assert reference_cfg.push_curriculum.adaptive_upgrades_enabled is False
    assert reference_cfg.push_curriculum.easy_sample_probability == 0.0
    assert tuple(reference_cfg.push_curriculum.stage1b_abs_delta_v_xy) == (1.0, 1.0)
    assert rl_only.params["velocity_range"] == {
        "x": (-1.0, 1.0),
        "y": (-1.0, 1.0),
    }


def main() -> None:
    rl_only_cfg, rl_only_agent_cfg = task_registry.get_cfgs(TASK)
    reference_cfg, reference_agent_cfg = task_registry.get_cfgs(REFERENCE_TASK)

    env_differences = unexpected_config_differences(
        reference_cfg,
        rl_only_cfg,
        allowed_prefixes=RL_ONLY_EXPECTED_ENV_DIFFERENCE_PREFIXES,
    )
    agent_differences = unexpected_config_differences(
        reference_agent_cfg,
        rl_only_agent_cfg,
        allowed_prefixes=("run_name",),
    )
    if env_differences or agent_differences:
        raise RuntimeError(
            "unexpected RL-only config differences: "
            f"env={env_differences}, agent={agent_differences}"
        )
    _assert_push_semantics(reference_cfg, rl_only_cfg)

    assert rl_only_cfg.robot.actor_obs_history_length == 5
    assert reference_cfg.robot.actor_obs_history_length == 5
    assert rl_only_cfg.robot.critic_obs_history_length == 10
    assert reference_cfg.robot.critic_obs_history_length == 10
    assert list(rl_only_agent_cfg.policy.actor_hidden_dims) == [512, 256, 128]
    assert list(rl_only_agent_cfg.policy.critic_hidden_dims) == [512, 256, 128]
    assert rl_only_agent_cfg.num_steps_per_env == 24
    assert rl_only_agent_cfg.max_iterations == 10000
    assert rl_only_agent_cfg.resume is False
    assert not hasattr(rl_only_cfg, "recovery_context")
    assert not hasattr(rl_only_cfg, "stage2_reward")
    assert not hasattr(rl_only_cfg, "plane_v1_reward")
    assert not hasattr(rl_only_cfg, "estimator_checkpoint_path")
    assert rl_only_cfg.scene.imu is None

    rl_only_cfg.scene.num_envs = args.num_envs
    rl_only_cfg.scene.seed = args.seed
    rl_only_cfg.device = args.device
    rl_only_agent_cfg.device = args.device
    env_class = task_registry.get_task_class(TASK)
    env = env_class(rl_only_cfg, args.headless)
    if _certificate_worker_children():
        raise RuntimeError("RL-only created certificate worker processes")
    forbidden_runtime_attributes = (
        "_certificate_evaluator",
        "_estimator",
        "_recovery_context",
        "_v1_event_total",
    )
    for name in forbidden_runtime_attributes:
        if hasattr(env, name):
            raise RuntimeError(f"RL-only unexpectedly initialized {name}")
    if "imu" in env.scene.sensors:
        raise RuntimeError("RL-only unexpectedly initialized the estimator IMU")
    recovery_reward_terms = tuple(
        name
        for name in env.reward_manager.active_terms
        if "recovery" in name.lower() or "certificate" in name.lower()
    )
    if recovery_reward_terms:
        raise RuntimeError(f"RL-only initialized recovery rewards: {recovery_reward_terms}")

    runner = OnPolicyRunner(
        env,
        rl_only_agent_cfg.to_dict(),
        log_dir=None,
        device=rl_only_agent_cfg.device,
    )
    policy = runner.get_inference_policy(device=env.device)
    actor_input = int(runner.alg.policy.actor[0].in_features)
    critic_input = int(runner.alg.policy.critic[0].in_features)
    if actor_input != 480 or critic_input != 1010:
        raise RuntimeError(f"network input mismatch: actor={actor_input}, critic={critic_input}")

    obs, extras = env.get_observations()
    actor_shape = tuple(obs.shape)
    critic_shape = tuple(extras["observations"]["critic"].shape)
    if actor_shape != (args.num_envs, 480):
        raise RuntimeError(f"actor observation shape mismatch: {actor_shape}")
    if critic_shape != (args.num_envs, 1010):
        raise RuntimeError(f"critic observation shape mismatch: {critic_shape}")

    mirrored_obs, _ = compute_symmetric_states(env, obs=obs, obs_type="policy")
    if mirrored_obs.shape != (2 * args.num_envs, 480):
        raise RuntimeError(f"symmetry observation shape mismatch: {tuple(mirrored_obs.shape)}")

    reset_count = 0
    for _ in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
            if not torch.all(torch.isfinite(actions)):
                raise RuntimeError("RL-only action contains NaN/Inf")
            obs, rewards, dones, extras = env.step(actions)
        if not torch.all(torch.isfinite(obs)):
            raise RuntimeError("RL-only observation contains NaN/Inf")
        if not torch.all(torch.isfinite(rewards)):
            raise RuntimeError("RL-only reward contains NaN/Inf")
        critic = extras["observations"]["critic"]
        if not torch.all(torch.isfinite(critic)):
            raise RuntimeError("RL-only critic observation contains NaN/Inf")
        reset_count += int(dones.sum().item())

    if _certificate_worker_children():
        raise RuntimeError("RL-only submitted work to certificate worker processes")
    report = {
        "task": TASK,
        "reference_task": REFERENCE_TASK,
        "config_diff_unexpected_count": 0,
        "expected_differences": list(RL_ONLY_EXPECTED_ENV_DIFFERENCE_PREFIXES),
        "actor_observation_shape": list(actor_shape),
        "critic_observation_shape": list(critic_shape),
        "actor_input_dim": actor_input,
        "critic_input_dim": critic_input,
        "action_dim": int(actions.shape[-1]),
        "actor_hidden_dims": list(rl_only_agent_cfg.policy.actor_hidden_dims),
        "critic_hidden_dims": list(rl_only_agent_cfg.policy.critic_hidden_dims),
        "num_steps_per_env": int(rl_only_agent_cfg.num_steps_per_env),
        "max_iterations": int(rl_only_agent_cfg.max_iterations),
        "resume": bool(rl_only_agent_cfg.resume),
        "steps": args.steps,
        "reset_count": reset_count,
        "certificate_workers": 0,
        "certificate_submissions": 0,
        "estimator_loaded": False,
        "recovery_context_dim": 0,
        "recovery_reward": False,
        "finite": True,
        "symmetry": {
            "data_augmentation": bool(
                rl_only_agent_cfg.algorithm.symmetry_cfg.use_data_augmentation
            ),
            "mirror_loss": bool(rl_only_agent_cfg.algorithm.symmetry_cfg.use_mirror_loss),
            "mirror_loss_coeff": float(
                rl_only_agent_cfg.algorithm.symmetry_cfg.mirror_loss_coeff
            ),
        },
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
