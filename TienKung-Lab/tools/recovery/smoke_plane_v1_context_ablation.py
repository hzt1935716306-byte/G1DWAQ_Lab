#!/usr/bin/env python3
"""Native 32-env, two-PPO-iteration smoke for Plane V1 context ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import traceback

from isaaclab.app import AppLauncher


TASK_FIELDS = {
    "g1_plane_v1_estimator_context_no_reward_matched_v2_n_valid": (
        "n_norm",
        "valid",
    ),
    "g1_plane_v1_estimator_context_no_reward_matched_v2_margin_valid": (
        "margin_norm",
        "valid",
    ),
}
REFERENCE_TASK = "g1_plane_v1_estimator_context_no_reward_matched_v2"
EXPECTED_ESTIMATOR_SHA256 = (
    "8574d845f28cbc7908e437250de46ba865898056483741e8fb72a81281f9e319"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=tuple(TASK_FIELDS), required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--extra_steps", type=int, default=40)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--estimator_checkpoint_path", type=Path, required=True)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.envs.g1.g1_symmetry import compute_symmetric_states  # noqa: E402
from legged_lab.recovery.baseline_matched_protocol import (  # noqa: E402
    matched_command_standing_mask,
)
from legged_lab.recovery.context_ablation_contract import (  # noqa: E402
    config_differences,
    unexpected_config_differences,
)
from legged_lab.recovery.plane_v1 import plane_v1_context_held_mask  # noqa: E402
from legged_lab.utils import task_registry  # noqa: E402


ENV_ALLOWED = ("actor_recovery_context_fields",)
AGENT_ALLOWED = ("experiment_name", "run_name", "wandb_project")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_exact_config_diff(reference_cfg, candidate_cfg, reference_agent, candidate_agent):
    env_differences = config_differences(reference_cfg, candidate_cfg)
    agent_differences = config_differences(reference_agent, candidate_agent)
    env_paths = tuple(difference.path for difference in env_differences)
    agent_paths = tuple(difference.path for difference in agent_differences)
    if env_paths != ENV_ALLOWED:
        raise RuntimeError(f"unexpected environment config diff: {env_differences}")
    if set(agent_paths) != set(AGENT_ALLOWED):
        raise RuntimeError(f"unexpected agent config diff: {agent_differences}")
    if unexpected_config_differences(
        reference_cfg, candidate_cfg, allowed_paths=ENV_ALLOWED
    ):
        raise RuntimeError("environment config diff allow-list audit failed")
    if unexpected_config_differences(
        reference_agent, candidate_agent, allowed_paths=AGENT_ALLOWED
    ):
        raise RuntimeError("agent config diff allow-list audit failed")
    return env_differences, agent_differences


def _assert_reward_contract(env, cfg) -> None:
    if bool(cfg.plane_v1_reward.enabled):
        raise RuntimeError("recoverability reward must be disabled")
    forbidden = {"alive"}
    present = forbidden.intersection(env.reward_manager.active_terms)
    if present:
        raise RuntimeError(f"forbidden reward terms are active: {sorted(present)}")
    if env._plane_v1_reward_enabled:
        raise RuntimeError("Plane V1 runtime recovery reward is enabled")
    idle = env.reward_manager.get_term_cfg("idle_penalty")
    swing = env.reward_manager.get_term_cfg("feet_swing_height")
    if float(idle.weight) != -0.1 or idle.params != {
        "cmd_threshold": 0.2,
        "vel_threshold": 0.1,
    }:
        raise RuntimeError("V2 idle reward contract changed")
    if float(swing.weight) != -0.2 or float(swing.params["target_height"]) != 0.08:
        raise RuntimeError("V2 swing-height reward contract changed")


def main() -> None:
    expected_fields = TASK_FIELDS[args.task]
    checkpoint = args.estimator_checkpoint_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != EXPECTED_ESTIMATOR_SHA256:
        raise RuntimeError(
            f"estimator SHA mismatch: expected={EXPECTED_ESTIMATOR_SHA256}, actual={checkpoint_sha}"
        )

    cfg, agent_cfg = task_registry.get_cfgs(args.task)
    reference_cfg, reference_agent_cfg = task_registry.get_cfgs(REFERENCE_TASK)
    env_differences, agent_differences = _assert_exact_config_diff(
        reference_cfg,
        cfg,
        reference_agent_cfg,
        agent_cfg,
    )
    if tuple(cfg.actor_recovery_context_fields) != expected_fields:
        raise RuntimeError("Actor-visible context field order is incorrect")
    if cfg.robot.actor_obs_history_length != 5 or cfg.robot.critic_obs_history_length != 10:
        raise RuntimeError("observation history contract changed")
    if agent_cfg.seed != 42 or agent_cfg.num_steps_per_env != 24:
        raise RuntimeError("seed or rollout length contract changed")
    if agent_cfg.max_iterations != 10000 or agent_cfg.resume:
        raise RuntimeError("training-budget/random-initialization contract changed")
    if list(agent_cfg.policy.actor_hidden_dims) != [512, 256, 128]:
        raise RuntimeError("Actor hidden dimensions changed")
    if list(agent_cfg.policy.critic_hidden_dims) != [512, 256, 128]:
        raise RuntimeError("Critic hidden dimensions changed")
    symmetry_cfg = agent_cfg.algorithm.symmetry_cfg
    if not symmetry_cfg.use_data_augmentation or not symmetry_cfg.use_mirror_loss:
        raise RuntimeError("symmetry augmentation or mirror loss is disabled")
    if float(symmetry_cfg.mirror_loss_coeff) != 0.1:
        raise RuntimeError("mirror loss coefficient changed")

    cfg.estimator_checkpoint_path = str(checkpoint)
    cfg.scene.num_envs = args.num_envs
    cfg.scene.seed = args.seed
    cfg.device = args.device
    agent_cfg.device = args.device
    agent_cfg.seed = args.seed
    env_class = task_registry.get_task_class(args.task)
    env = None
    runner = None
    try:
        env = env_class(cfg, args.headless)
        if tuple(env.actor_recovery_context_fields) != expected_fields:
            raise RuntimeError("runtime Actor context field order changed")
        if env.actor_recovery_context_dim != 2:
            raise RuntimeError("runtime Actor context is not genuinely two-dimensional")
        if env._recovery_context.shape != (args.num_envs, 3):
            raise RuntimeError("internal full certificate context is no longer three-dimensional")
        if env._estimator is None or env._estimator.training:
            raise RuntimeError("Estimator was not loaded in eval mode")
        if any(parameter.requires_grad for parameter in env._estimator.parameters()):
            raise RuntimeError("Estimator parameters are not frozen")
        if env._certificate_evaluator is None:
            raise RuntimeError("certificate evaluator was not created")
        _assert_reward_contract(env, cfg)

        # This RSL-RL fork always snapshots git state in ``learn()`` and thus
        # requires a log directory.  Keep smoke artifacts outside the project
        # and remove them automatically so they cannot become resume sources.
        smoke_log = tempfile.TemporaryDirectory(prefix="plane_context_ablation_smoke_")
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=smoke_log.name,
            device=agent_cfg.device,
        )
        actor_input = int(runner.alg.policy.actor[0].in_features)
        critic_input = int(runner.alg.policy.critic[0].in_features)
        actor_output = int(runner.alg.policy.actor[-1].out_features)
        if (actor_input, critic_input, actor_output) != (482, 1010, 29):
            raise RuntimeError(
                f"network dimensions changed: actor={actor_input}, critic={critic_input}, action={actor_output}"
            )

        obs, extras = env.get_observations()
        if tuple(obs.shape) != (args.num_envs, 482):
            raise RuntimeError(f"Actor observation shape mismatch: {tuple(obs.shape)}")
        if tuple(extras["observations"]["critic"].shape) != (args.num_envs, 1010):
            raise RuntimeError("Critic observation shape mismatch")
        torch.testing.assert_close(obs[:, -2:], env._actor_visible_recovery_context())

        original_full = env._recovery_context[0].clone()
        env._recovery_context[0] = torch.tensor([0.5, -0.3, 1.0], device=env.device)
        expected_artificial = (
            torch.tensor([0.5, 1.0], device=env.device)
            if expected_fields[0] == "n_norm"
            else torch.tensor([-0.3, 1.0], device=env.device)
        )
        torch.testing.assert_close(env._actor_visible_recovery_context()[0], expected_artificial)
        env._recovery_context[0] = original_full

        mirrored, _ = compute_symmetric_states(env, obs=obs, obs_type="policy")
        if tuple(mirrored.shape) != (2 * args.num_envs, 482):
            raise RuntimeError("482-D symmetry augmentation shape mismatch")
        torch.testing.assert_close(mirrored[args.num_envs :, -2:], obs[:, -2:])

        # Two genuine PPO iterations with random policy initialization.  With
        # log_dir=None this cannot create a checkpoint used by formal training.
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)

        policy = runner.get_inference_policy(device=env.device)
        obs, extras = env.get_observations()
        refresh_count = 0
        held_count = 0
        standing_zero_count = 0
        invalid_zero_count = 0
        reset_count = 0
        for _ in range(args.extra_steps):
            context_before = env._recovery_context.clone()
            visible_before = env._actor_visible_recovery_context().clone()
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)
            if actions.shape != (args.num_envs, 29):
                raise RuntimeError("action shape mismatch")
            tensors = (actions, obs, rewards, extras["observations"]["critic"])
            if any(not torch.all(torch.isfinite(tensor)) for tensor in tensors):
                raise RuntimeError("native smoke produced NaN/Inf")
            torch.testing.assert_close(obs[:, -2:], env._actor_visible_recovery_context())
            standing = matched_command_standing_mask(
                env._last_certificate_state.command_velocity
            )
            held = plane_v1_context_held_mask(env._context_touchdown_mask, dones, standing)
            torch.testing.assert_close(env._recovery_context[held], context_before[held])
            torch.testing.assert_close(
                env._actor_visible_recovery_context()[held], visible_before[held]
            )
            held_count += int(held.sum().item())
            refresh_count += int(env._context_refresh_mask.sum().item())
            if torch.any(standing):
                standing_zero_count += int(standing.sum().item())
                if torch.count_nonzero(env._recovery_context[standing]).item() != 0:
                    raise RuntimeError("standing context was not cleared")
            invalid = ~env.current_certificate_valid
            if torch.any(invalid):
                invalid_zero_count += int(invalid.sum().item())
                if torch.count_nonzero(env._recovery_context[invalid]).item() != 0:
                    raise RuntimeError("invalid context did not use full zero encoding")
            reset_count += int(dones.sum().item())

        if refresh_count == 0 or env._certificate_evaluator.statistics["evaluations"] == 0:
            raise RuntimeError("native smoke did not execute certificate queries")
        if held_count == 0 or standing_zero_count == 0 or invalid_zero_count == 0:
            raise RuntimeError("held/standing/invalid context paths were not exercised")

        reset_ids = torch.arange(min(4, args.num_envs), device=env.device)
        # Isaac state tensors were advanced under inference mode above; reset
        # them under the same mode to avoid creating an in-place update mismatch.
        with torch.inference_mode():
            env.reset(reset_ids)
        if torch.count_nonzero(env._recovery_context[reset_ids]).item() != 0:
            raise RuntimeError("explicit reset did not clear full context")
        if torch.count_nonzero(env._actor_visible_recovery_context()[reset_ids]).item() != 0:
            raise RuntimeError("explicit reset did not clear visible context")

        report = {
            "task": args.task,
            "reference_task": REFERENCE_TASK,
            "context_fields": list(expected_fields),
            "internal_context_dim": 3,
            "actor_visible_context_dim": 2,
            "actor_input_dim": actor_input,
            "critic_input_dim": critic_input,
            "action_dim": actor_output,
            "ppo_iterations": 2,
            "random_initialization": True,
            "certificate_evaluations": int(
                env._certificate_evaluator.statistics["evaluations"]
            ),
            "context_refresh_count": refresh_count,
            "held_frame_env_count": held_count,
            "standing_zero_count": standing_zero_count,
            "invalid_zero_count": invalid_zero_count,
            "natural_reset_count": reset_count,
            "explicit_reset_checked": int(reset_ids.numel()),
            "estimator_sha256": checkpoint_sha,
            "estimator_eval": not env._estimator.training,
            "estimator_frozen": True,
            "recovery_reward_enabled": False,
            "idle_penalty_weight": -0.1,
            "feet_swing_height_weight": -0.2,
            "feet_swing_target_height": 0.08,
            "symmetry_data_augmentation": True,
            "mirror_loss": True,
            "mirror_loss_coeff": 0.1,
            "finite": True,
            "environment_diff": [difference.path for difference in env_differences],
            "agent_diff": [difference.path for difference in agent_differences],
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if env is not None and getattr(env, "_certificate_evaluator", None) is not None:
            env._certificate_evaluator.close()
        if "smoke_log" in locals():
            smoke_log.cleanup()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
