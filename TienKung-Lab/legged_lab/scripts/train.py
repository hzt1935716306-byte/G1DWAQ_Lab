# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import math

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner, DWAQOnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--defer_certificate_reward",
    action="store_true",
    help="Overlap Ours certificate solves across a rollout and backfill rewards before GAE.",
)
parser.add_argument(
    "--certificate_event_scale",
    type=float,
    default=None,
    help="Override the certificate-only Stage2 event scale for the Ours task.",
)
parser.add_argument(
    "--disable_push_curriculum",
    action="store_true",
    help="Disable Stage2 push progression and use the full random Stage1B push range.",
)
parser.add_argument(
    "--resume_curriculum_level",
    type=int,
    default=None,
    help="Resume a Stage2 checkpoint at this curriculum level (requires iterations-in-level).",
)
parser.add_argument(
    "--resume_curriculum_iterations_in_level",
    type=int,
    default=None,
    help="Number of completed iterations in the resumed Stage2 curriculum level.",
)
parser.add_argument(
    "--stage1a_context_warm_start",
    action="store_true",
    help=(
        "Strictly migrate a 960-D Stage1A actor to the 963-D context actor; "
        "model weights are loaded and the Stage2 optimizer starts fresh."
    ),
)
parser.add_argument(
    "--estimator_checkpoint_path",
    type=str,
    default=None,
    help=(
        "Strict V2 CoM-estimator checkpoint for Plane V1 estimator-source tasks. "
        "The formal config default is intentionally empty."
    ),
)
parser.add_argument(
    "--plane_v1_smoke",
    action="store_true",
    help="Use short integration-only Plane V1 push/terrain timing for a 3-iteration smoke.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering for tasks that require RGB/depth sensing
if args_cli.task and ("sensor" in args_cli.task or "rgb" in args_cli.task or "depth" in args_cli.task):
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.recovery.checkpoint_migration import warm_start_context_policy
from legged_lab.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    runner: OnPolicyRunner | AmpOnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.estimator_checkpoint_path is not None:
        if not hasattr(env_cfg, "estimator_checkpoint_path"):
            raise ValueError("--estimator_checkpoint_path requires a Plane V1 task")
        env_cfg.estimator_checkpoint_path = args_cli.estimator_checkpoint_path
    if args_cli.plane_v1_smoke:
        if not hasattr(env_cfg, "plane_v1_reward"):
            raise ValueError("--plane_v1_smoke requires a final Plane V1 task")
        env_cfg.domain_rand.events.push_robot.interval_range_s = (0.10, 0.20)
        env_cfg.stage2_reward.certificate_executor = "sequential"
        env_cfg.stage2_reward.certificate_workers = 1
        env_cfg.terrain_level_1_iteration = 1
        env_cfg.terrain_level_2_iteration = 2
    if args_cli.certificate_event_scale is not None:
        if not math.isfinite(args_cli.certificate_event_scale) or args_cli.certificate_event_scale <= 0.0:
            raise ValueError("--certificate_event_scale must be finite and positive")
        if not hasattr(env_cfg, "stage2_reward"):
            raise ValueError("--certificate_event_scale requires a Stage2 task")
        reward_cfg = env_cfg.stage2_reward
        if not reward_cfg.enabled or not reward_cfg.enable_certificate_reward:
            raise ValueError("--certificate_event_scale requires the Ours certificate task")
        if reward_cfg.enable_shared_event_reward or reward_cfg.enable_soft_reward_scaling:
            raise ValueError(
                "--certificate_event_scale requires certificate-only reward: "
                "shared events and soft locomotion scaling must both be disabled"
            )
        reward_cfg.event_scale = float(args_cli.certificate_event_scale)
        print(
            "[INFO] Certificate-only Stage2 reward: "
            f"event_scale={reward_cfg.event_scale}, shared_events=False, soft_scaling=False"
        )
    if args_cli.defer_certificate_reward and hasattr(env_cfg, "stage2_reward"):
        if not env_cfg.stage2_reward.enable_certificate_reward:
            raise ValueError("--defer_certificate_reward is only valid for the Ours task")
        env_cfg.stage2_reward.defer_certificate_reward_to_rollout_end = True

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    resume_level = args_cli.resume_curriculum_level
    resume_iterations = args_cli.resume_curriculum_iterations_in_level
    if args_cli.disable_push_curriculum:
        if resume_level is not None or resume_iterations is not None:
            raise ValueError("--disable_push_curriculum cannot be combined with explicit curriculum restoration")
        if not hasattr(env_cfg, "push_curriculum"):
            raise ValueError("--disable_push_curriculum requires a Stage2 push task")
        curriculum_cfg = env_cfg.push_curriculum
        curriculum_cfg.enable_push_curriculum = False
        curriculum_cfg.adaptive_upgrades_enabled = False
        curriculum_cfg.easy_sample_probability = 0.0
        print(
            "[INFO] Stage2 push curriculum disabled: "
            f"fixed_level={len(curriculum_cfg.level_ratios)}, "
            f"abs_delta_v_xy={curriculum_cfg.stage1b_abs_delta_v_xy}, "
            "easy_sample_probability=0.0"
        )
    if (resume_level is None) != (resume_iterations is None):
        raise ValueError(
            "--resume_curriculum_level and --resume_curriculum_iterations_in_level must be provided together"
        )
    if resume_level is not None:
        if not agent_cfg.resume:
            raise ValueError("explicit curriculum restoration requires --resume True")
        if not hasattr(env_cfg, "push_curriculum") or not env_cfg.push_curriculum.enable_push_curriculum:
            raise ValueError("explicit curriculum restoration requires an enabled Stage2 push curriculum")
        curriculum_cfg = env_cfg.push_curriculum
        if not 1 <= resume_level <= len(curriculum_cfg.level_ratios):
            raise ValueError("--resume_curriculum_level is outside the configured curriculum range")
        if resume_iterations < 0:
            raise ValueError("--resume_curriculum_iterations_in_level must be non-negative")
        if resume_level < len(curriculum_cfg.level_ratios) and resume_iterations >= curriculum_cfg.k_max_iterations:
            raise ValueError(
                "--resume_curriculum_iterations_in_level must be below K_max before the final level"
            )
        curriculum_cfg.initial_level = int(resume_level)
        curriculum_cfg.initial_iterations_in_level = int(resume_iterations)
        print(
            "[INFO] Restoring Stage2 curriculum: "
            f"level={curriculum_cfg.initial_level}, "
            f"iterations_in_level={curriculum_cfg.initial_iterations_in_level}"
        )
    if args_cli.stage1a_context_warm_start:
        if not agent_cfg.resume:
            raise ValueError("--stage1a_context_warm_start requires --resume True")
        if resume_level is not None:
            raise ValueError(
                "Stage1A context warm start cannot restore a Stage2 curriculum position"
            )
    env_cfg.scene.seed = agent_cfg.seed

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    env = env_class(env_cfg, args_cli.headless)
    if hasattr(env, "set_num_steps_per_learning_iteration"):
        env.set_num_steps_per_learning_iteration(agent_cfg.num_steps_per_env)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    if hasattr(env, "configure_curriculum_logging"):
        env.configure_curriculum_logging(log_dir)

    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.stage1a_context_warm_start:
            if not getattr(env, "_recovery_context_enabled", False):
                raise ValueError("Stage1A context warm start requires a 963-D recovery-context task")
            migration = warm_start_context_policy(runner.alg.policy, resume_path)
            print(
                "[INFO] Strict Stage1A context warm start: "
                f"actor={migration['actor_input_before']}->{migration['actor_input_after']}, "
                f"source_iteration={migration['source_iteration']}, optimizer_loaded=False"
            )
        else:
            # A same-shape Stage2 resume preserves the model, optimizer, and iteration.
            runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train()
    simulation_app.close()
