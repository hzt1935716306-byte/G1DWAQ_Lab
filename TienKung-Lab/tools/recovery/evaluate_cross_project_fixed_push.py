#!/usr/bin/env python3
"""Evaluate a native TienKung or unitree_rl_lab G1 policy at a fixed push bound.

The policies keep their native observation histories, actor networks, robot
assets, actuator gains, and termination logic.  The trial plan, flat terrain,
commands, velocity jumps, recovery thresholds, and outcome bookkeeping are
shared.  This makes the result a native-system comparison, not a pure actor
ablation across identical physics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_UNITREE_ROOT = Path("/home/zt/project/g1_base/unitree_rl_lab")
COMMANDS = (
    (0.4, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (-0.3, 0.0, 0.0),
    (0.4, 0.25, 0.0),
    (0.4, -0.25, 0.0),
    (0.4, 0.0, 0.2),
    (0.4, 0.0, -0.2),
    (0.0, 0.0, 0.0),
)
MEAN_VELOCITY_ERROR_THRESHOLD = 0.14522231240183686
MEAN_ABS_ROLL_THRESHOLD = 0.02584471284877509
MEAN_ABS_PITCH_THRESHOLD = 0.042438490772619845
MAX_RECOVERY_TOUCHDOWNS = 5

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", choices=("ours", "unitree"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--bound", type=float, default=0.5)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--prepare_steps", type=int, default=50)
parser.add_argument("--max_recovery_time_s", type=float, default=10.0)
parser.add_argument("--max_steps", type=int, default=12000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--unitree_root", type=Path, default=DEFAULT_UNITREE_ROOT)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--force_exit_after_report",
    action="store_true",
    help="Exit after the JSON is flushed; useful when an Isaac environment hangs during teardown.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.recovery.state_extractor import (  # noqa: E402
    G1PrivilegedStateExtractor,
    G1StateExtractorCfg,
)
from legged_lab.utils import task_registry  # noqa: E402


def _quantiles(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {key: None for key in ("mean", "median", "p75", "p90", "min", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_plans(episodes: int, bound: float, seed: int):
    rng = np.random.default_rng(seed)
    normalized = rng.uniform(-1.0, 1.0, size=(episodes, 2))
    plans = []
    for sample_index in range(episodes):
        delta = normalized[sample_index] * bound
        plans.append(
            {
                "trial_id": f"B{bound:.3f}-{sample_index:05d}",
                "sample_index": sample_index,
                "command_velocity": list(COMMANDS[sample_index % len(COMMANDS)]),
                "normalized_delta_xy": normalized[sample_index].tolist(),
                "delta_v_world_xy": delta.tolist(),
            }
        )
    rng.shuffle(plans)
    serialized = json.dumps(plans, sort_keys=True, separators=(",", ":")).encode()
    return plans, hashlib.sha256(serialized).hexdigest()


def _disable_tienkung_randomization(env_cfg) -> None:
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


class _UnitreeStateAdapter:
    """Expose the small interface used by G1PrivilegedStateExtractor."""

    def __init__(self, env):
        self._env = env
        self.robot = env.scene["robot"]
        self.contact_sensor = env.scene["contact_forces"]
        self.command_generator = env.command_manager.get_term("base_velocity")
        self.device = env.device
        self.num_envs = env.num_envs
        self.scene = env.scene

    @property
    def episode_length_buf(self):
        return self._env.episode_length_buf

    @property
    def sim_step_counter(self):
        physics_steps_per_policy_step = round(float(self.step_dt) / float(self.physics_dt))
        return int(self._env.common_step_counter) * physics_steps_per_policy_step

    @property
    def physics_dt(self):
        return self._env.physics_dt

    @property
    def step_dt(self):
        return self._env.step_dt


class _Actor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, obs):
        return self.actor(obs)


class _Runtime:
    def __init__(self, kind, env, robot, command_term, state_env, obs, policy, checkpoint, iteration):
        self.kind = kind
        self.env = env
        self.robot = robot
        self.command_term = command_term
        self.state_env = state_env
        self.obs = obs
        self.policy = policy
        self.checkpoint = checkpoint
        self.iteration = iteration
        self.num_envs = state_env.num_envs
        self.device = state_env.device

    def set_commands(self, slots) -> None:
        command = self.command_term.command
        command.zero_()
        for env_id, slot in enumerate(slots):
            if slot is not None:
                command[env_id, :3] = torch.as_tensor(
                    slot["plan"]["command_velocity"], dtype=command.dtype, device=self.device
                )
        self.command_term.is_standing_env[:] = False

    def apply_push(self, env_id: int, delta_v_xy) -> None:
        ids = torch.tensor([env_id], dtype=torch.long, device=self.device)
        velocity = self.robot.data.root_vel_w[ids].clone()
        velocity[:, :2] += torch.as_tensor(delta_v_xy, dtype=velocity.dtype, device=self.device)
        self.robot.write_root_velocity_to_sim(velocity, env_ids=ids)

    def step(self):
        with torch.inference_mode():
            actions = self.policy(self.obs)
            if self.kind == "ours":
                obs, _, dones, _ = self.env.step(actions)
            else:
                obs_dict, _, terminated, truncated, _ = self.env.step(actions)
                obs = obs_dict["policy"]
                dones = terminated | truncated
        self.obs = obs
        return dones

    def close(self):
        self.env.close()


def _create_ours(checkpoint: Path) -> _Runtime:
    print("[cross-fixed-push] creating TienKung environment", flush=True)
    env_cfg, agent_cfg = task_registry.get_cfgs("g1_flat_symmetric")
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
    _disable_tienkung_randomization(env_cfg)
    env_cfg.sim.device = args.device
    agent_cfg.device = args.device
    env = task_registry.get_task_class("g1_flat_symmetric")(env_cfg, args.headless)
    print("[cross-fixed-push] TienKung environment ready; loading checkpoint", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    runner.eval_mode()
    obs, _ = env.get_observations()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return _Runtime(
        "ours",
        env,
        env.robot,
        env.command_generator,
        env,
        obs,
        runner.get_inference_policy(device=env.device),
        checkpoint,
        int(saved.get("iter", -1)),
    )


def _create_unitree(checkpoint: Path) -> _Runtime:
    print("[cross-fixed-push] importing unitree task", flush=True)
    source = args.unitree_root.expanduser().resolve() / "source/unitree_rl_lab"
    if not source.is_dir():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source))
    import unitree_rl_lab.tasks  # noqa: F401
    module = importlib.import_module(
        "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
    )
    RobotEnvCfg = module.RobotEnvCfg

    env_cfg = RobotEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.events.push_robot = None
    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base.params["pose_range"] = {
        key: (0.0, 0.0) for key in env_cfg.events.reset_base.params["pose_range"]
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        key: (0.0, 0.0) for key in env_cfg.events.reset_base.params["velocity_range"]
    }
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.episode_length_s = 1000.0
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed

    print("[cross-fixed-push] creating Unitree environment", flush=True)
    gym_env = gym.make("Unitree-G1-29dof-Velocity", cfg=env_cfg)
    print("[cross-fixed-push] Unitree environment created; resetting", flush=True)
    env = gym_env.unwrapped
    obs_dict, _ = gym_env.reset()
    print("[cross-fixed-push] Unitree environment ready; loading checkpoint", flush=True)
    obs = obs_dict["policy"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    print("[cross-fixed-push] Unitree checkpoint loaded", flush=True)
    state_dict = saved["model_state_dict"]
    input_dim = int(state_dict["actor.0.weight"].shape[1])
    output_dim = int(state_dict["actor.6.weight"].shape[0])
    actor = _Actor(input_dim, output_dim)
    print("[cross-fixed-push] Unitree actor allocated on CPU", flush=True)
    actor.load_state_dict({name: value for name, value in state_dict.items() if name.startswith("actor.")})
    print("[cross-fixed-push] Unitree actor weights loaded on CPU; moving to GPU", flush=True)
    actor.to(env.device)
    print("[cross-fixed-push] Unitree actor ready on GPU", flush=True)
    actor.eval()
    if obs.shape[-1] != input_dim or env.action_manager.total_action_dim != output_dim:
        raise RuntimeError(
            f"unitree checkpoint/env mismatch: obs={obs.shape}, actions={env.action_manager.total_action_dim}, "
            f"actor={input_dim}->{output_dim}"
        )
    adapter = _UnitreeStateAdapter(env)
    return _Runtime(
        "unitree",
        gym_env,
        env.scene["robot"],
        env.command_manager.get_term("base_velocity"),
        adapter,
        obs,
        actor,
        checkpoint,
        int(saved.get("iter", -1)),
    )


def _new_slot(plan: dict) -> dict:
    return {
        "plan": plan,
        "status": "preparing",
        "prepare_remaining": args.prepare_steps,
        "start_step": None,
        "start_time": None,
        "touchdowns": 0,
        "last_touchdown_foot": -1,
        "interval_started": False,
        "sample_count": 0,
        "velocity_error_sum": 0.0,
        "abs_tilt_sum": np.zeros(2, dtype=np.float64),
    }


def _complete(slot, outcome, state, env_id, policy_step, reason):
    episode = dict(slot["plan"])
    episode.update(
        {
            "env_id": env_id,
            "outcome": outcome,
            "completion_reason": reason,
            "practical_enter_step": slot["touchdowns"] if outcome == "SUCCESS" else None,
            "touchdown_count": slot["touchdowns"],
            "start_policy_step": slot["start_step"],
            "end_policy_step": policy_step,
            "recovery_time_s": float(state.time[env_id].item()) - float(slot["start_time"]),
        }
    )
    return episode


def _performance(episodes):
    total = len(episodes)
    counts = {key: sum(item["outcome"] == key for item in episodes) for key in ("SUCCESS", "TIMEOUT", "FALL")}
    successes = [item for item in episodes if item["outcome"] == "SUCCESS"]
    return {
        "episode_count": total,
        "outcome_counts": counts,
        "success_rate_P5": counts["SUCCESS"] / total if total else None,
        "non_fall_rate": 1.0 - counts["FALL"] / total if total else None,
        "timeout_rate": counts["TIMEOUT"] / total if total else None,
        "fall_rate": counts["FALL"] / total if total else None,
        "practical_enter_step": _quantiles([item["practical_enter_step"] for item in successes]),
        "recovery_time_s_success": _quantiles([item["recovery_time_s"] for item in successes]),
    }


def main() -> None:
    if args.bound <= 0.0 or args.episodes <= 0 or args.num_envs <= 0 or args.prepare_steps < 0:
        raise ValueError("bound, episode, environment, and preparation arguments are invalid")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    plans, plan_hash = _make_plans(args.episodes, args.bound, args.seed)
    pending = list(reversed(plans))
    runtime = _create_ours(checkpoint) if args.policy == "ours" else _create_unitree(checkpoint)
    print("[cross-fixed-push] initializing shared state extractor", flush=True)
    extractor = G1PrivilegedStateExtractor(
        runtime.state_env, G1StateExtractorCfg(h_eff=0.6884990671277046)
    )
    print("[cross-fixed-push] starting evaluation loop", flush=True)
    slots = [None] * runtime.num_envs
    for env_id in range(runtime.num_envs):
        if pending:
            slots[env_id] = _new_slot(pending.pop())
    runtime.set_commands(slots)
    completed = []
    nominal_reset_count = 0

    for policy_step in range(args.max_steps):
        runtime.set_commands(slots)
        dones = runtime.step()
        state = extractor.extract()
        done_mask = dones.bool() | state.episode_reset

        for env_id, slot in enumerate(slots):
            if slot is None or not bool(done_mask[env_id].item()):
                continue
            if slot["status"] == "active":
                completed.append(_complete(slot, "FALL", state, env_id, policy_step, "environment_reset"))
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
            count = slot["sample_count"]
            alternating = slot["last_touchdown_foot"] < 0 or (
                int(state.touchdown_foot[env_id].item()) != slot["last_touchdown_foot"]
            )
            good = False
            if slot["interval_started"] and count > 0:
                velocity_error = slot["velocity_error_sum"] / count
                abs_tilt = slot["abs_tilt_sum"] / count
                good = bool(
                    alternating
                    and velocity_error <= MEAN_VELOCITY_ERROR_THRESHOLD
                    and abs_tilt[0] <= MEAN_ABS_ROLL_THRESHOLD
                    and abs_tilt[1] <= MEAN_ABS_PITCH_THRESHOLD
                )
            slot["last_touchdown_foot"] = int(state.touchdown_foot[env_id].item())
            slot["interval_started"] = True
            slot["sample_count"] = 0
            slot["velocity_error_sum"] = 0.0
            slot["abs_tilt_sum"][:] = 0.0
            if good:
                completed.append(_complete(slot, "SUCCESS", state, env_id, policy_step, "practical_good_cycle"))
                slots[env_id] = None
            elif slot["touchdowns"] >= MAX_RECOVERY_TOUCHDOWNS:
                completed.append(_complete(slot, "TIMEOUT", state, env_id, policy_step, "five_touchdowns"))
                slots[env_id] = None

        for env_id, slot in enumerate(slots):
            if slot is None or slot["status"] != "active":
                continue
            if float(state.time[env_id].item()) - float(slot["start_time"]) >= args.max_recovery_time_s:
                completed.append(_complete(slot, "TIMEOUT", state, env_id, policy_step, "wall_time_limit"))
                slots[env_id] = None
                continue
            velocity_error = torch.linalg.vector_norm(
                state.com_velocity[env_id, :2] - state.command_velocity[env_id, :2]
            )
            slot["sample_count"] += 1
            slot["velocity_error_sum"] += float(velocity_error.item())
            slot["abs_tilt_sum"] += np.abs(state.root_roll_pitch[env_id].detach().cpu().numpy())

        for env_id, slot in enumerate(slots):
            if slot is None and pending:
                slots[env_id] = _new_slot(pending.pop())
                slot = slots[env_id]
            if slot is None or slot["status"] != "preparing":
                continue
            slot["prepare_remaining"] -= 1
            if slot["prepare_remaining"] <= 0:
                runtime.apply_push(env_id, slot["plan"]["delta_v_world_xy"])
                slot["status"] = "active"
                slot["start_step"] = policy_step
                slot["start_time"] = float(state.time[env_id].item())

        if (policy_step + 1) % 250 == 0:
            print(
                f"[cross-fixed-push] policy={args.policy} step={policy_step + 1}/{args.max_steps} "
                f"completed={len(completed)}/{len(plans)}",
                flush=True,
            )
        if len(completed) == len(plans):
            break

    report = {
        "schema_version": 1,
        "policy": args.policy,
        "native_environment": "g1_flat_symmetric" if args.policy == "ours" else "Unitree-G1-29dof-Velocity",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_iteration": runtime.iteration,
        "inference_only": True,
        "comparison_scope": "native policy + native robot asset/actuator/termination stack",
        "common_protocol": {
            "seed": args.seed,
            "component_bound_mps": args.bound,
            "episodes": args.episodes,
            "commands": [list(command) for command in COMMANDS],
            "trial_plan_sha256": plan_hash,
            "flat_plane": True,
            "observation_noise": False,
            "physics_randomization": False,
            "max_recovery_touchdowns": MAX_RECOVERY_TOUCHDOWNS,
            "max_recovery_time_s": args.max_recovery_time_s,
            "mean_velocity_error_threshold": MEAN_VELOCITY_ERROR_THRESHOLD,
            "mean_abs_roll_threshold": MEAN_ABS_ROLL_THRESHOLD,
            "mean_abs_pitch_threshold": MEAN_ABS_PITCH_THRESHOLD,
        },
        "num_envs": runtime.num_envs,
        "actor_observation_shape": list(runtime.obs.shape),
        "planned_episode_count": len(plans),
        "completed_episode_count": len(completed),
        "pending_episode_count": len(plans) - len(completed),
        "nominal_reset_count": nominal_reset_count,
        "policy_steps_executed": policy_step + 1,
        "overall_performance": _performance(completed),
        "episodes": completed,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(report)
    summary.pop("episodes")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[cross-fixed-push] wrote {output}", flush=True)
    if args.force_exit_after_report:
        print("[cross-fixed-push] forcing clean process exit after persisted report", flush=True)
        os._exit(0)
    runtime.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
