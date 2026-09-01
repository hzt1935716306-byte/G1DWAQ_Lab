#!/usr/bin/env python3
"""Evaluate one policy on the paired Stage2 Phase-1 disturbance suite."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher


POLICY_TASKS = {
    "baseline": "g1_flat_symmetric",
    "ours": "g1_flat_symmetric",
    "dwaq": "g1_dwaq",
    "unitree": "Unitree-G1-29dof-Velocity",
}
FAMILIES = (
    "velocity_ood",
    "force_pulse",
    "constant_force",
    "repeated_impulse",
    "random_force",
    "wrench_pulse",
)
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
CROSS_PROJECT_COMMANDS = (
    (0.4, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (-0.3, 0.0, 0.0),
    (0.4, 0.25, 0.0),
    (0.4, -0.25, 0.0),
    (0.4, 0.0, 0.2),
    (0.4, 0.0, -0.2),
    (0.0, 0.0, 0.0),
)
CONTROL_DT = 0.02
DEFAULT_UNITREE_ROOT = Path("/home/zt/project/g1_base/unitree_rl_lab")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", choices=tuple(POLICY_TASKS), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--family", choices=(*FAMILIES, "all"), required=True)
parser.add_argument("--episodes_per_condition", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--prepare_steps", type=int, default=50)
parser.add_argument("--onset_jitter_steps", type=int, default=40)
parser.add_argument("--max_recovery_time_s", type=float, default=10.0)
parser.add_argument("--max_steps", type=int, default=30000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--limit_trials", type=int, default=None)
parser.add_argument("--cross_project_protocol", action="store_true")
parser.add_argument("--fixed_survival_horizon", action="store_true")
parser.add_argument("--unitree_root", type=Path, default=DEFAULT_UNITREE_ROOT)
parser.add_argument("--force_exit_after_report", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from rsl_rl.runners import DWAQOnPolicyRunner, OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.recovery.state_extractor import (  # noqa: E402
    G1PrivilegedStateExtractor,
    G1StateExtractorCfg,
)
from legged_lab.utils import task_registry  # noqa: E402


def _commands() -> tuple[tuple[float, float, float], ...]:
    return CROSS_PROJECT_COMMANDS if args.cross_project_protocol else COMMANDS


class _UnitreeActor(nn.Module):
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

    def forward(self, observations):
        return self.actor(observations)


class _UnitreeRunnerAdapter:
    def __init__(self, policy):
        self.policy = policy

    def get_inference_policy(self, device=None):
        return self.policy


class _UnitreeEnvAdapter:
    """Present the interface consumed by the shared disturbance evaluator."""

    def __init__(self, gym_env, observations):
        self._gym_env = gym_env
        self._env = gym_env.unwrapped
        self._observations = observations
        self.robot = self._env.scene["robot"]
        self.contact_sensor = self._env.scene["contact_forces"]
        self.command_generator = self._env.command_manager.get_term("base_velocity")
        self.device = self._env.device
        self.num_envs = self._env.num_envs
        self.scene = self._env.scene
        self.step_dt = self._env.step_dt
        self.physics_dt = self._env.physics_dt
        # Unitree uses no runner-side action clipping.  This value is used only
        # to keep the optional action-saturation diagnostic well-defined.
        self.clip_actions = 100.0

    @property
    def episode_length_buf(self):
        return self._env.episode_length_buf

    @property
    def sim_step_counter(self):
        physics_steps_per_policy_step = round(float(self.step_dt) / float(self.physics_dt))
        return int(self._env.common_step_counter) * physics_steps_per_policy_step

    def get_observations(self):
        return self._observations, None

    def step(self, actions):
        observations, rewards, terminated, truncated, extras = self._gym_env.step(actions)
        self._observations = observations["policy"]
        return self._observations, rewards, terminated | truncated, extras

    def close(self):
        self._gym_env.close()


def _default_episodes_per_condition(family: str) -> int:
    return 256 if family == "velocity_ood" else 128


def _selected_families(family: str) -> tuple[str, ...]:
    return FAMILIES if family == "all" else (family,)


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


def _condition_specs(family: str) -> list[dict]:
    if family == "velocity_ood":
        return [
            {"condition_id": f"square_s{severity:.2f}", "component_bound_mps": severity}
            for severity in (1.25, 1.50, 2.00)
        ]
    if family == "force_pulse":
        return [
            {
                "condition_id": f"veq{veq:.2f}_t{duration:.2f}",
                "equivalent_delta_v_mps": veq,
                "nominal_duration_s": duration,
            }
            for veq in (0.50, 1.00, 1.50)
            for duration in (0.05, 0.20, 0.50)
        ]
    if family == "constant_force":
        return [
            {
                "condition_id": f"a{acceleration:.2f}_t5.00",
                "acceleration_mps2": acceleration,
                "nominal_duration_s": 5.0,
            }
            for acceleration in (0.25, 0.50, 0.75, 1.00, 1.50)
        ]
    if family == "repeated_impulse":
        return [
            {
                "condition_id": f"dv{delta_v:.2f}_period{period:.2f}",
                "single_delta_v_mps": delta_v,
                "impact_period_s": period,
                "nominal_duration_s": 8.0,
            }
            for delta_v in (0.25, 0.50, 0.75)
            for period in (0.50, 1.00)
        ]
    if family == "random_force":
        return [
            {
                "condition_id": f"rms{rms:.2f}_tau{correlation:.2f}",
                "rms_acceleration_mps2": rms,
                "correlation_time_s": correlation,
                "nominal_duration_s": 10.0,
            }
            for rms in (0.25, 0.50, 1.00)
            for correlation in (0.10, 1.00)
        ]
    if family == "wrench_pulse":
        return [
            {
                "condition_id": f"{mode}_veq1.00_t0.20",
                "wrench_mode": mode,
                "equivalent_delta_v_mps": 1.0,
                "nominal_duration_s": 0.20,
                "moment_arm_m": 0.20 if mode == "upper_pitch" else 0.15,
            }
            for mode in ("com", "upper_pitch", "lateral_yaw")
        ]
    raise ValueError(family)


def _direction(sample_index: int) -> tuple[float, float]:
    direction_index = (sample_index // len(_commands())) % 8
    angle = direction_index * (math.pi / 4.0)
    return math.cos(angle), math.sin(angle)


def _make_plans(family: str, episodes_per_condition: int | None, seed: int):
    rng = np.random.default_rng(seed)
    plans = []
    for selected_family in _selected_families(family):
        family_episode_count = (
            episodes_per_condition
            if episodes_per_condition is not None
            else _default_episodes_per_condition(selected_family)
        )
        for condition in _condition_specs(selected_family):
            for sample_index in range(family_episode_count):
                plan = {
                    "trial_id": (
                        f"{selected_family}-{condition['condition_id']}-{sample_index:05d}"
                    ),
                    "family": selected_family,
                    **condition,
                    "sample_index": sample_index,
                    "command_velocity": list(_commands()[sample_index % len(_commands())]),
                    "direction_xy": list(_direction(sample_index)),
                    "onset_jitter_steps": int(
                        rng.integers(0, args.onset_jitter_steps + 1)
                    ),
                }
                if selected_family == "velocity_ood":
                    bound = float(condition["component_bound_mps"])
                    plan["delta_v_world_xy"] = rng.uniform(-bound, bound, size=2).tolist()
                elif selected_family == "repeated_impulse":
                    interval_steps = max(
                        1, round(condition["impact_period_s"] / CONTROL_DT)
                    )
                    duration_steps = math.ceil(
                        condition["nominal_duration_s"] / CONTROL_DT
                    )
                    impact_count = math.ceil(duration_steps / interval_steps)
                    impact_angles = rng.uniform(0.0, 2.0 * math.pi, size=impact_count)
                    magnitude = float(condition["single_delta_v_mps"])
                    plan["impact_delta_v_world_xy"] = [
                        [magnitude * math.cos(angle), magnitude * math.sin(angle)]
                        for angle in impact_angles
                    ]
                elif selected_family == "random_force":
                    plan["waveform_seed"] = int(rng.integers(0, 2**31 - 1))
                if selected_family == "wrench_pulse":
                    plan["torque_sign"] = -1.0 if sample_index % 2 else 1.0
                plans.append(plan)
    rng.shuffle(plans)
    if args.limit_trials is not None:
        if args.limit_trials <= 0:
            raise ValueError("--limit_trials must be positive")
        plans = plans[: args.limit_trials]
    serialized = json.dumps(plans, sort_keys=True, separators=(",", ":")).encode()
    return plans, hashlib.sha256(serialized).hexdigest()


def _policy_environment():
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.policy == "unitree":
        source = args.unitree_root.expanduser().resolve() / "source/unitree_rl_lab"
        if not source.is_dir():
            raise FileNotFoundError(source)
        sys.path.insert(0, str(source))
        import unitree_rl_lab.tasks  # noqa: F401

        module = importlib.import_module(
            "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
        )
        env_cfg = module.RobotEnvCfg()
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
        gym_env = gym.make("Unitree-G1-29dof-Velocity", cfg=env_cfg)
        observations, _ = gym_env.reset()
        env = _UnitreeEnvAdapter(gym_env, observations["policy"])
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = saved["model_state_dict"]
        input_dim = int(state_dict["actor.0.weight"].shape[1])
        output_dim = int(state_dict["actor.6.weight"].shape[0])
        actor = _UnitreeActor(input_dim, output_dim)
        actor.load_state_dict(
            {name: value for name, value in state_dict.items() if name.startswith("actor.")}
        )
        actor.to(env.device).eval()
        if env.get_observations()[0].shape[-1] != input_dim:
            raise RuntimeError(
                f"unitree observation/checkpoint mismatch: {env.get_observations()[0].shape} vs {input_dim}"
            )
        return env, _UnitreeRunnerAdapter(actor), checkpoint

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
    if not math.isclose(float(env.step_dt), CONTROL_DT, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"control dt mismatch: expected {CONTROL_DT}, got {env.step_dt}")
    runner_cls = DWAQOnPolicyRunner if args.policy == "dwaq" else OnPolicyRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    runner.eval_mode()
    return env, runner, checkpoint


def _set_commands(env, slots) -> None:
    command = env.command_generator.command
    values = np.zeros((env.num_envs, 3), dtype=np.float32)
    for env_id, slot in enumerate(slots):
        if slot is not None:
            values[env_id] = slot["plan"]["command_velocity"]
    command[:, :3].copy_(torch.as_tensor(values, dtype=command.dtype, device=env.device))
    if command.shape[1] > 3:
        command[:, 3:].zero_()
    env.command_generator.is_standing_env[:] = False


def _apply_velocity_jumps(env, env_ids: list[int], delta_v_xy: list[list[float]]) -> None:
    if not env_ids:
        return
    env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    velocity = env.robot.data.root_vel_w[env_ids_tensor].clone()
    velocity[:, :2] += torch.as_tensor(
        delta_v_xy, dtype=velocity.dtype, device=env.device
    )
    env.robot.write_root_velocity_to_sim(velocity, env_ids=env_ids_tensor)


def _duration_steps(plan: dict) -> int:
    return math.ceil(float(plan.get("nominal_duration_s", 0.0)) / CONTROL_DT)


def _new_slot(plan: dict) -> dict:
    return {
        "plan": plan,
        "status": "preparing",
        "prepare_remaining": args.prepare_steps + int(plan["onset_jitter_steps"]),
        "disturb_step": 0,
        "impact_index": 0,
        "ou_state": np.zeros(2, dtype=np.float64),
        "ou_rng": np.random.default_rng(plan.get("waveform_seed", 0)),
        "disturbance_start_time": None,
        "release_time": None,
        "survived_disturbance": None,
        "survived_full_horizon": False,
        "functional_hold": None,
        "p5_outcome": None,
        "p5_enter_step": None,
        "p5_completion_time": None,
        "recovery_touchdowns": 0,
        "disturbance_touchdowns": 0,
        "last_touchdown_foot": -1,
        "interval_started": False,
        "interval_sample_count": 0,
        "interval_velocity_error_sum": 0.0,
        "interval_abs_tilt_sum": np.zeros(2, dtype=np.float64),
        "disturbance_sample_count": 0,
        "disturbance_velocity_error_sum": 0.0,
        "disturbance_velocity_error_sq_sum": 0.0,
        "response_sample_count": 0,
        "response_velocity_error_sum": 0.0,
        "response_velocity_error_sq_sum": 0.0,
        "max_abs_roll": 0.0,
        "max_abs_pitch": 0.0,
        "max_com_speed": 0.0,
        "max_contact_force": 0.0,
        "max_com_displacement": 0.0,
        "disturbance_start_com": None,
        "foot_slip_distance": 0.0,
        "previous_foot_positions": None,
        "last_hold_samples": [],
        "action_sample_count": 0,
        "action_saturation_count": 0,
        "max_abs_action": 0.0,
    }


def _force_for_slot(slot: dict, total_mass: float):
    plan = slot["plan"]
    family = plan["family"]
    direction = np.asarray(plan["direction_xy"], dtype=np.float64)
    force = np.zeros(3, dtype=np.float64)
    torque = np.zeros(3, dtype=np.float64)
    if family in ("force_pulse", "wrench_pulse"):
        actual_duration = _duration_steps(plan) * CONTROL_DT
        acceleration = float(plan["equivalent_delta_v_mps"]) / actual_duration
        force[:2] = total_mass * acceleration * direction
        if family == "wrench_pulse":
            mode = plan["wrench_mode"]
            arm = float(plan["moment_arm_m"])
            if mode == "upper_pitch":
                torque[:2] = arm * np.asarray((-force[1], force[0]))
            elif mode == "lateral_yaw":
                torque[2] = float(plan["torque_sign"]) * arm * np.linalg.norm(force[:2])
    elif family == "constant_force":
        force[:2] = total_mass * float(plan["acceleration_mps2"]) * direction
    elif family == "random_force":
        correlation = float(plan["correlation_time_s"])
        alpha = math.exp(-CONTROL_DT / correlation)
        axis_std = float(plan["rms_acceleration_mps2"]) / math.sqrt(2.0)
        noise_scale = axis_std * math.sqrt(1.0 - alpha * alpha)
        slot["ou_state"] = alpha * slot["ou_state"] + noise_scale * slot["ou_rng"].normal(size=2)
        force[:2] = total_mass * slot["ou_state"]
    return force, torque


def _write_wrenches(env, slots, total_mass: np.ndarray, torso_body_id: int) -> None:
    forces = np.zeros((env.num_envs, 1, 3), dtype=np.float32)
    torques = np.zeros_like(forces)
    for env_id, slot in enumerate(slots):
        if slot is None or slot["status"] != "disturbing":
            continue
        if slot["plan"]["family"] not in ("force_pulse", "constant_force", "random_force", "wrench_pulse"):
            continue
        force, torque = _force_for_slot(slot, float(total_mass[env_id]))
        forces[env_id, 0] = force
        torques[env_id, 0] = torque
    env.robot.permanent_wrench_composer.set_forces_and_torques(
        forces=torch.as_tensor(forces, device=env.device),
        torques=torch.as_tensor(torques, device=env.device),
        body_ids=[torso_body_id],
        is_global=True,
    )


def _quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("mean", "median", "p75", "p90", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _state_snapshot(state, actions) -> np.ndarray:
    velocity_error = torch.linalg.vector_norm(
        state.com_velocity[:, :2] - state.command_velocity[:, :2], dim=1, keepdim=True
    )
    com_speed = torch.linalg.vector_norm(state.com_velocity[:, :2], dim=1, keepdim=True)
    contact_force = torch.max(state.contact_forces, dim=1, keepdim=True).values
    return torch.cat(
        (
            velocity_error,
            com_speed,
            contact_force,
            torch.abs(state.root_roll_pitch),
            state.com_position[:, :2],
            state.left_foot_position_w,
            state.right_foot_position_w,
            state.contacts.to(torch.float32),
            torch.abs(actions),
        ),
        dim=1,
    ).detach().cpu().numpy()


def _record_response_sample(
    slot: dict,
    snapshot: np.ndarray,
    env_id: int,
    action_clip: float,
    *,
    during_disturbance: bool,
) -> None:
    row = snapshot[env_id]
    velocity_error = float(row[0])
    com_speed = float(row[1])
    contact_force = float(row[2])
    abs_tilt = row[3:5]
    com_position = row[5:7]
    if slot["disturbance_start_com"] is None:
        slot["disturbance_start_com"] = com_position.copy()
    displacement = float(np.linalg.norm(com_position - slot["disturbance_start_com"]))
    foot_positions = row[7:13].reshape(2, 3)
    if slot["previous_foot_positions"] is not None:
        delta = np.linalg.norm(foot_positions[:, :2] - slot["previous_foot_positions"][:, :2], axis=1)
        contacts = row[13:15]
        slot["foot_slip_distance"] += float(np.sum(delta * contacts))
    slot["previous_foot_positions"] = foot_positions
    slot["response_sample_count"] += 1
    slot["response_velocity_error_sum"] += velocity_error
    slot["response_velocity_error_sq_sum"] += velocity_error * velocity_error
    if during_disturbance:
        slot["disturbance_sample_count"] += 1
        slot["disturbance_velocity_error_sum"] += velocity_error
        slot["disturbance_velocity_error_sq_sum"] += velocity_error * velocity_error
    slot["max_abs_roll"] = max(slot["max_abs_roll"], float(abs_tilt[0]))
    slot["max_abs_pitch"] = max(slot["max_abs_pitch"], float(abs_tilt[1]))
    slot["max_com_speed"] = max(slot["max_com_speed"], com_speed)
    slot["max_contact_force"] = max(slot["max_contact_force"], contact_force)
    slot["max_com_displacement"] = max(slot["max_com_displacement"], displacement)
    if during_disturbance:
        slot["last_hold_samples"].append(
            (velocity_error, float(abs_tilt[0]), float(abs_tilt[1]))
        )
        hold_count = round(1.0 / CONTROL_DT)
        if len(slot["last_hold_samples"]) > hold_count:
            slot["last_hold_samples"] = slot["last_hold_samples"][-hold_count:]
    abs_actions = row[15:]
    slot["action_sample_count"] += int(abs_actions.size)
    slot["action_saturation_count"] += int(np.count_nonzero(abs_actions >= 0.99 * action_clip))
    slot["max_abs_action"] = max(slot["max_abs_action"], float(np.max(abs_actions)))


def _functional_hold(slot: dict, curriculum_cfg) -> bool:
    samples = np.asarray(slot["last_hold_samples"], dtype=np.float64)
    if samples.size == 0:
        return False
    means = np.mean(samples, axis=0)
    return bool(
        means[0] <= curriculum_cfg.mean_velocity_error_threshold
        and means[1] <= curriculum_cfg.mean_abs_roll_threshold
        and means[2] <= curriculum_cfg.mean_abs_pitch_threshold
    )


def _complete(
    slot: dict, outcome: str, current_time: float, env_id: int, policy_step: int, reason: str
):
    episode = dict(slot["plan"])
    sample_count = slot["disturbance_sample_count"]
    response_sample_count = slot["response_sample_count"]
    disturbance_start_time = slot["disturbance_start_time"]
    release_time = slot["release_time"]
    episode.update(
        {
            "env_id": env_id,
            "outcome": outcome,
            "completion_reason": reason,
            "survived_disturbance": bool(slot["survived_disturbance"]),
            "survived_full_horizon": bool(slot["survived_full_horizon"]),
            "functional_hold": bool(slot["functional_hold"]) if slot["functional_hold"] is not None else False,
            "composite_success": outcome == "SUCCESS" and bool(slot["survived_disturbance"]),
            "p5_outcome_before_horizon": slot["p5_outcome"],
            "recovery_touchdown_count": slot["recovery_touchdowns"],
            "disturbance_touchdown_count": slot["disturbance_touchdowns"],
            "practical_enter_step": (
                slot["p5_enter_step"]
                if args.fixed_survival_horizon and outcome == "SUCCESS"
                else slot["recovery_touchdowns"] if outcome == "SUCCESS" else None
            ),
            "disturbance_duration_s": (
                float(release_time - disturbance_start_time)
                if release_time is not None and disturbance_start_time is not None
                else None
            ),
            "post_release_recovery_time_s": (
                float(slot["p5_completion_time"]) - float(release_time)
                if args.fixed_survival_horizon
                and outcome == "SUCCESS"
                and release_time is not None
                and slot["p5_completion_time"] is not None
                else current_time - float(release_time)
                if release_time is not None
                else None
            ),
            "end_policy_step": policy_step,
            "disturbance_velocity_error_mean": (
                slot["disturbance_velocity_error_sum"] / sample_count if sample_count else None
            ),
            "disturbance_velocity_error_rms": (
                math.sqrt(slot["disturbance_velocity_error_sq_sum"] / sample_count)
                if sample_count
                else None
            ),
            "response_velocity_error_mean": (
                slot["response_velocity_error_sum"] / response_sample_count
                if response_sample_count
                else None
            ),
            "response_velocity_error_rms": (
                math.sqrt(slot["response_velocity_error_sq_sum"] / response_sample_count)
                if response_sample_count
                else None
            ),
            "max_abs_roll": slot["max_abs_roll"],
            "max_abs_pitch": slot["max_abs_pitch"],
            "max_com_speed": slot["max_com_speed"],
            "max_contact_force": slot["max_contact_force"],
            "max_com_displacement": slot["max_com_displacement"],
            "foot_slip_distance": slot["foot_slip_distance"],
            "action_saturation_fraction": (
                slot["action_saturation_count"] / slot["action_sample_count"]
                if slot["action_sample_count"]
                else None
            ),
            "max_abs_action": slot["max_abs_action"],
        }
    )
    return episode


def _performance(episodes):
    count = len(episodes)
    outcomes = Counter(item["outcome"] for item in episodes)
    successes = [item for item in episodes if item["outcome"] == "SUCCESS"]
    numeric_keys = (
        "disturbance_velocity_error_mean",
        "disturbance_velocity_error_rms",
        "response_velocity_error_mean",
        "response_velocity_error_rms",
        "max_abs_roll",
        "max_abs_pitch",
        "max_com_speed",
        "max_contact_force",
        "max_com_displacement",
        "foot_slip_distance",
        "action_saturation_fraction",
        "max_abs_action",
    )
    return {
        "episode_count": count,
        "outcome_counts": dict(outcomes),
        "survival_rate": (
            sum(item["survived_full_horizon"] for item in episodes) / count
            if args.fixed_survival_horizon and count
            else sum(item["survived_disturbance"] for item in episodes) / count if count else None
        ),
        "full_horizon_survival_rate": (
            sum(item["survived_full_horizon"] for item in episodes) / count if count else None
        ),
        "functional_hold_rate": sum(item["functional_hold"] for item in episodes) / count if count else None,
        "post_release_P5": outcomes["SUCCESS"] / count if count else None,
        "composite_success_rate": sum(item["composite_success"] for item in episodes) / count if count else None,
        "recovery_touchdowns_success": _quantiles(
            [item["practical_enter_step"] for item in successes]
        ),
        "post_release_recovery_time_s_success": _quantiles(
            [item["post_release_recovery_time_s"] for item in successes]
        ),
        "continuous_metrics": {
            key: _quantiles([item[key] for item in episodes if item[key] is not None])
            for key in numeric_keys
        },
    }


def main() -> None:
    if args.num_envs <= 0 or args.prepare_steps < 0 or args.onset_jitter_steps < 0:
        raise ValueError("invalid environment/preparation counts")
    if args.max_recovery_time_s <= 0.0 or args.max_steps <= 0:
        raise ValueError("invalid time/step limits")
    episodes_per_condition = args.episodes_per_condition
    if episodes_per_condition is not None and episodes_per_condition <= 0:
        raise ValueError("episodes per condition must be positive")

    env, runner, checkpoint = _policy_environment()
    plans, plan_hash = _make_plans(args.family, episodes_per_condition, args.seed)
    pending = list(reversed(plans))
    extractor = G1PrivilegedStateExtractor(env, G1StateExtractorCfg(h_eff=0.6884990671277046))
    total_mass = extractor.total_mass.detach().cpu().numpy()
    recovery_cfg, _ = task_registry.get_cfgs("g1_flat_symmetric_stage2_baseline")
    curriculum_cfg = recovery_cfg.push_curriculum
    torso_ids, torso_names = env.robot.find_bodies(["torso_link"], preserve_order=True)
    if torso_names != ["torso_link"] or len(torso_ids) != 1:
        raise RuntimeError(f"could not resolve torso_link: ids={torso_ids}, names={torso_names}")
    torso_body_id = int(torso_ids[0])

    slots = [None] * env.num_envs
    for env_id in range(env.num_envs):
        if pending:
            slots[env_id] = _new_slot(pending.pop())
    _set_commands(env, slots)
    if args.policy == "dwaq":
        obs, obs_hist = env.get_observations()
        inference_policy = runner.alg.policy.act_inference
    else:
        obs, _ = env.get_observations()
        obs_hist = None
        inference_policy = runner.get_inference_policy(device=env.device)
    completed = []
    nominal_reset_count = 0

    for policy_step in range(args.max_steps):
        _set_commands(env, slots)
        _write_wrenches(env, slots, total_mass, torso_body_id)
        impact_env_ids = []
        impact_delta_v = []
        for env_id, slot in enumerate(slots):
            if slot is None or slot["status"] != "disturbing":
                continue
            if slot["plan"]["family"] == "repeated_impulse":
                interval_steps = max(1, round(slot["plan"]["impact_period_s"] / CONTROL_DT))
                if slot["disturb_step"] % interval_steps == 0:
                    impacts = slot["plan"]["impact_delta_v_world_xy"]
                    if slot["impact_index"] < len(impacts):
                        impact_env_ids.append(env_id)
                        impact_delta_v.append(impacts[slot["impact_index"]])
                        slot["impact_index"] += 1
        _apply_velocity_jumps(env, impact_env_ids, impact_delta_v)

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
        snapshot = _state_snapshot(state, actions)
        event_snapshot = torch.stack(
            (
                done_mask.to(torch.int64),
                state.touchdown.to(torch.int64),
                state.touchdown_foot,
            ),
            dim=1,
        ).detach().cpu().numpy()
        current_time = float(env.sim_step_counter) * float(env.physics_dt)
        for env_id, slot in enumerate(slots):
            if slot is None or slot["status"] not in ("disturbing", "recovering"):
                continue
            # env.step() has already reset terminated environments, so their extracted
            # state is the reset state rather than the terminal state.
            if bool(event_snapshot[env_id, 0]):
                continue
            _record_response_sample(
                slot,
                snapshot,
                env_id,
                float(env.clip_actions),
                during_disturbance=slot["status"] == "disturbing",
            )

        for env_id, slot in enumerate(slots):
            if slot is None:
                continue
            if bool(event_snapshot[env_id, 0]):
                if slot["status"] in ("disturbing", "recovering"):
                    slot["survived_disturbance"] = slot["status"] == "recovering"
                    slot["survived_full_horizon"] = False
                    completed.append(
                        _complete(
                            slot,
                            "FALL",
                            current_time,
                            env_id,
                            policy_step,
                            "environment_reset",
                        )
                    )
                    slots[env_id] = None
                else:
                    nominal_reset_count += 1
                    slot["prepare_remaining"] = args.prepare_steps + int(slot["plan"]["onset_jitter_steps"])

        touchdown_ids = np.flatnonzero(event_snapshot[:, 1]).tolist()
        for env_id in touchdown_ids:
            slot = slots[env_id]
            if slot is None:
                continue
            if slot["status"] == "disturbing":
                slot["disturbance_touchdowns"] += 1
                continue
            if slot["status"] != "recovering":
                continue
            if args.fixed_survival_horizon and slot["p5_outcome"] is not None:
                continue
            slot["recovery_touchdowns"] += 1
            count = slot["interval_sample_count"]
            has_interval = slot["interval_started"] and count > 0
            touchdown_foot = int(event_snapshot[env_id, 2])
            alternating = (
                slot["last_touchdown_foot"] < 0
                or touchdown_foot != slot["last_touchdown_foot"]
            )
            good_cycle = False
            if has_interval:
                velocity_error = slot["interval_velocity_error_sum"] / count
                abs_tilt = slot["interval_abs_tilt_sum"] / count
                good_cycle = bool(
                    alternating
                    and velocity_error <= curriculum_cfg.mean_velocity_error_threshold
                    and abs_tilt[0] <= curriculum_cfg.mean_abs_roll_threshold
                    and abs_tilt[1] <= curriculum_cfg.mean_abs_pitch_threshold
                )
            slot["last_touchdown_foot"] = touchdown_foot
            slot["interval_started"] = True
            slot["interval_sample_count"] = 0
            slot["interval_velocity_error_sum"] = 0.0
            slot["interval_abs_tilt_sum"][:] = 0.0
            if good_cycle:
                if args.fixed_survival_horizon:
                    slot["p5_outcome"] = "SUCCESS"
                    slot["p5_enter_step"] = slot["recovery_touchdowns"]
                    slot["p5_completion_time"] = current_time
                else:
                    completed.append(
                        _complete(
                            slot,
                            "SUCCESS",
                            current_time,
                            env_id,
                            policy_step,
                            "practical_good_cycle",
                        )
                    )
                    slots[env_id] = None
            elif slot["recovery_touchdowns"] >= curriculum_cfg.max_recovery_touchdowns:
                if args.fixed_survival_horizon:
                    slot["p5_outcome"] = "TIMEOUT"
                    slot["p5_completion_time"] = current_time
                else:
                    completed.append(
                        _complete(
                            slot,
                            "TIMEOUT",
                            current_time,
                            env_id,
                            policy_step,
                            "five_touchdowns",
                        )
                    )
                    slots[env_id] = None

        for env_id, slot in enumerate(slots):
            if slot is None:
                continue
            if slot["status"] == "disturbing":
                slot["disturb_step"] += 1
                if slot["disturb_step"] >= _duration_steps(slot["plan"]):
                    slot["survived_disturbance"] = True
                    slot["functional_hold"] = _functional_hold(slot, curriculum_cfg)
                    slot["release_time"] = current_time
                    slot["status"] = "recovering"
                    slot["recovery_touchdowns"] = 0
                    slot["last_touchdown_foot"] = -1
                    slot["interval_started"] = False
                    slot["interval_sample_count"] = 0
                    slot["interval_velocity_error_sum"] = 0.0
                    slot["interval_abs_tilt_sum"][:] = 0.0
            elif slot["status"] == "recovering":
                if current_time - float(slot["release_time"]) >= args.max_recovery_time_s:
                    if args.fixed_survival_horizon:
                        slot["survived_full_horizon"] = True
                        if slot["p5_outcome"] is None:
                            slot["p5_outcome"] = "TIMEOUT"
                            slot["p5_completion_time"] = current_time
                        completed.append(
                            _complete(
                                slot,
                                slot["p5_outcome"],
                                current_time,
                                env_id,
                                policy_step,
                                "full_survival_horizon",
                            )
                        )
                        slots[env_id] = None
                        continue
                    completed.append(
                        _complete(
                            slot,
                            "TIMEOUT",
                            current_time,
                            env_id,
                            policy_step,
                            "wall_time_limit",
                        )
                    )
                    slots[env_id] = None
                    continue
                slot["interval_sample_count"] += 1
                slot["interval_velocity_error_sum"] += float(snapshot[env_id, 0])
                slot["interval_abs_tilt_sum"] += snapshot[env_id, 3:5]

        jump_env_ids = []
        jump_delta_v = []
        for env_id, slot in enumerate(slots):
            if slot is None and pending:
                slots[env_id] = _new_slot(pending.pop())
                slot = slots[env_id]
            if slot is None or slot["status"] != "preparing":
                continue
            slot["prepare_remaining"] -= 1
            if slot["prepare_remaining"] <= 0:
                slot["disturbance_start_time"] = current_time
                if slot["plan"]["family"] == "velocity_ood":
                    jump_env_ids.append(env_id)
                    jump_delta_v.append(slot["plan"]["delta_v_world_xy"])
                    slot["survived_disturbance"] = True
                    slot["functional_hold"] = False
                    slot["release_time"] = current_time
                    slot["status"] = "recovering"
                else:
                    slot["status"] = "disturbing"
        _apply_velocity_jumps(env, jump_env_ids, jump_delta_v)

        if (policy_step + 1) % 250 == 0:
            print(
                f"[disturbance-suite] policy={args.policy} family={args.family} "
                f"step={policy_step + 1}/{args.max_steps} completed={len(completed)}/{len(plans)}",
                flush=True,
            )
        if len(completed) == len(plans):
            break

    _write_wrenches(env, [None] * env.num_envs, total_mass, torso_body_id)
    condition_specs = [
        {"family": family, **condition}
        for family in _selected_families(args.family)
        for condition in _condition_specs(family)
    ]
    per_condition = {
        f"{condition['family']}/{condition['condition_id']}": _performance(
            [
                item
                for item in completed
                if item["family"] == condition["family"]
                and item["condition_id"] == condition["condition_id"]
            ]
        )
        for condition in condition_specs
    }
    per_family = {
        family: _performance([item for item in completed if item["family"] == family])
        for family in _selected_families(args.family)
    }
    report = {
        "schema_version": 1,
        "policy": args.policy,
        "native_task": POLICY_TASKS[args.policy],
        "checkpoint": str(checkpoint),
        "inference_only": True,
        "family": args.family,
        "common_protocol": {
            "seed": args.seed,
            "trial_plan_sha256": plan_hash,
            "commands": [list(command) for command in _commands()],
            "control_dt": CONTROL_DT,
            "flat_plane": True,
            "observation_noise": False,
            "physics_randomization": False,
            "force_body": "torso_link",
            "force_frame": "world",
            "peak_and_action_metric_window": "disturbance onset through outcome",
            "disturbance_velocity_metric_window": "active disturbance only",
            "action_clip": float(env.clip_actions),
            "action_saturation_threshold": 0.99 * float(env.clip_actions),
            "max_recovery_touchdowns": curriculum_cfg.max_recovery_touchdowns,
            "max_recovery_time_s": args.max_recovery_time_s,
            "fixed_survival_horizon": args.fixed_survival_horizon,
            "survival_definition": (
                "no native termination through disturbance and max_recovery_time_s after release"
                if args.fixed_survival_horizon
                else "no native termination during active disturbance"
            ),
            "functional_thresholds": {
                "mean_velocity_error": curriculum_cfg.mean_velocity_error_threshold,
                "mean_abs_roll": curriculum_cfg.mean_abs_roll_threshold,
                "mean_abs_pitch": curriculum_cfg.mean_abs_pitch_threshold,
            },
        },
        "condition_specs": condition_specs,
        "episodes_per_condition": (
            episodes_per_condition
            if episodes_per_condition is not None
            else {
                family: _default_episodes_per_condition(family)
                for family in _selected_families(args.family)
            }
        ),
        "num_envs": args.num_envs,
        "prepare_steps": args.prepare_steps,
        "onset_jitter_steps": args.onset_jitter_steps,
        "planned_episode_count": len(plans),
        "completed_episode_count": len(completed),
        "pending_episode_count": len(plans) - len(completed),
        "policy_steps_executed": policy_step + 1,
        "nominal_reset_count": nominal_reset_count,
        "actor_observation_shape": list(obs.shape),
        "total_mass_kg": _quantiles(extractor.total_mass.detach().cpu().numpy()),
        "overall_performance": _performance(completed),
        "performance_by_family": per_family,
        "performance_by_condition": per_condition,
        "episodes": completed,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(report)
    summary.pop("episodes")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[disturbance-suite] wrote {output}", flush=True)
    if args.force_exit_after_report:
        print("[disturbance-suite] forcing process exit after persisted report", flush=True)
        os._exit(0)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
