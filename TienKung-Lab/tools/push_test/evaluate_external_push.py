#!/usr/bin/env python3
"""Evaluate standing G1 policies with body-heading continuous or pulse forces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


CONTROL_DT = 0.02
DEFAULT_UNITREE_ROOT = Path("/home/zt/project/g1_base/unitree_rl_lab")
POLICY_TASKS = {"ours": "g1_flat_symmetric", "unitree": "Unitree-G1-29dof-Velocity"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", choices=tuple(POLICY_TASKS), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--mode", choices=("continuous", "impulse"), required=True)
parser.add_argument("--force", type=float, nargs="+", required=True, help="Force levels in newtons.")
parser.add_argument("--force_duration", type=float, default=None)
parser.add_argument("--push_direction_body", type=float, nargs=3, default=(1.0, 0.0, 0.0))
parser.add_argument("--trials_per_force", type=int, default=10)
parser.add_argument("--application_link", default="torso_link")
parser.add_argument("--application_point", choices=("body_com", "link_origin"), default="body_com")
parser.add_argument("--application_offset_body", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("--stabilization_time", type=float, default=5.0)
parser.add_argument("--post_time", type=float, default=None)
parser.add_argument("--step_displacement_threshold", type=float, default=0.03)
parser.add_argument("--airborne_min_time", type=float, default=0.06)
parser.add_argument("--airborne_min_height", type=float, default=0.015)
parser.add_argument("--settle_linear_speed", type=float, default=0.10)
parser.add_argument("--settle_angular_speed", type=float, default=0.20)
parser.add_argument("--settle_tilt", type=float, default=0.10)
parser.add_argument("--settle_hold_time", type=float, default=0.20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--unitree_root", type=Path, default=DEFAULT_UNITREE_ROOT)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--no_traces", action="store_true")
parser.add_argument("--force_exit_after_report", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, yaw_quat  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.recovery.state_extractor import (  # noqa: E402
    G1PrivilegedStateExtractor,
    G1StateExtractorCfg,
)
from legged_lab.utils import task_registry  # noqa: E402


TRACE_FIELDS = (
    "policy",
    "trial_id",
    "trial_index",
    "mode",
    "force_N",
    "duration_s",
    "impulse_Ns",
    "time_s",
    "phase",
    "force_world_x",
    "force_world_y",
    "force_world_z",
    "application_point_world_x",
    "application_point_world_y",
    "application_point_world_z",
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
    "base_lin_vel_x",
    "base_lin_vel_y",
    "base_lin_vel_z",
    "base_ang_vel_x",
    "base_ang_vel_y",
    "base_ang_vel_z",
    "roll",
    "pitch",
    "yaw",
    "com_pos_x",
    "com_pos_y",
    "com_pos_z",
    "com_vel_x",
    "com_vel_y",
    "com_vel_z",
    "left_foot_x",
    "left_foot_y",
    "left_foot_z",
    "right_foot_x",
    "right_foot_y",
    "right_foot_z",
    "left_contact",
    "right_contact",
    "touchdown",
    "touchdown_foot",
    "step_event",
    "step_flag",
    "fall_event",
    "state_is_post_reset",
)

RESULT_FIELDS = (
    "policy",
    "checkpoint",
    "trial_id",
    "trial_index",
    "env_id",
    "seed",
    "mode",
    "force_N",
    "force_duration_s",
    "impulse_Ns",
    "direction_body",
    "application_link",
    "application_point",
    "application_offset_body",
    "force_start_time_s",
    "force_end_time_s",
    "stabilization_time_s",
    "post_time_s",
    "first_force_world",
    "last_force_world",
    "mean_force_world",
    "first_application_point_world",
    "initial_left_foot_world",
    "initial_right_foot_world",
    "final_base_position_world",
    "final_base_linear_velocity_world",
    "final_base_angular_velocity_world",
    "final_rpy",
    "final_com_position_heading",
    "final_com_velocity_heading",
    "fall_flag",
    "fall_phase",
    "step_flag_force",
    "step_flag_post",
    "step_count",
    "max_touchdown_displacement_m",
    "recovery_time_s",
    "success_no_step",
    "success_no_fall",
)


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

    @property
    def episode_length_buf(self):
        return self._env.episode_length_buf

    @property
    def sim_step_counter(self):
        ratio = round(float(self.step_dt) / float(self.physics_dt))
        return int(self._env.common_step_counter) * ratio

    def get_observations(self):
        return self._observations, None

    def step(self, actions):
        observations, rewards, terminated, truncated, extras = self._gym_env.step(actions)
        self._observations = observations["policy"]
        return self._observations, rewards, terminated | truncated, extras

    def close(self):
        self._gym_env.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _disable_non_reset_randomization(events) -> None:
    for name in (
        "push_robot",
        "physics_material",
        "add_base_mass",
        "randomize_actuator_gains",
        "randomize_com",
        "randomize_dome_light",
        "randomize_distant_light",
        "base_external_force_torque",
    ):
        if hasattr(events, name):
            setattr(events, name, None)


def _policy_environment(num_envs: int):
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
        env_cfg.scene.num_envs = num_envs
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        _disable_non_reset_randomization(env_cfg.events)
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.commands.base_velocity.rel_standing_envs = 1.0
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
            raise RuntimeError("Unitree observation/checkpoint mismatch")
        return env, _UnitreeRunnerAdapter(actor), checkpoint

    env_cfg, agent_cfg = task_registry.get_cfgs("g1_flat_symmetric")
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.seed = args.seed
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.scene.terrain_type = "plane"
    env_cfg.scene.terrain_generator = None
    env_cfg.commands.rel_standing_envs = 1.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.debug_vis = False
    env_cfg.commands.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.action_delay.enable = False
    _disable_non_reset_randomization(env_cfg.domain_rand.events)
    env_cfg.sim.device = args.device
    agent_cfg.device = args.device
    env = task_registry.get_task_class("g1_flat_symmetric")(env_cfg, args.headless)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    runner.eval_mode()
    return env, runner, checkpoint


def _set_standing_command(env) -> None:
    env.command_generator.command.zero_()
    env.command_generator.is_standing_env[:] = True


def _as_json(values) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def _make_plans(forces: list[float], duration: float) -> list[dict]:
    plans = []
    for force in forces:
        for trial_index in range(args.trials_per_force):
            plans.append(
                {
                    "trial_id": f"{args.mode}-F{force:g}-trial{trial_index:03d}",
                    "trial_index": trial_index,
                    "force_N": float(force),
                    "duration_s": duration,
                    "impulse_Ns": float(force) * duration,
                }
            )
    return plans


def _phase(step: int, stabilization_steps: int, force_steps: int) -> str:
    if step < stabilization_steps:
        return "stabilization"
    if step < stabilization_steps + force_steps:
        return "force"
    return "post"


def _summary_by_force(results: list[dict]) -> list[dict]:
    grouped = []
    for force in sorted({float(row["force_N"]) for row in results}):
        rows = [row for row in results if float(row["force_N"]) == force]
        count = len(rows)
        grouped.append(
            {
                "force_N": force,
                "impulse_Ns": force * float(rows[0]["force_duration_s"]),
                "trials": count,
                "success_no_step_count": sum(bool(row["success_no_step"]) for row in rows),
                "success_no_fall_count": sum(bool(row["success_no_fall"]) for row in rows),
                "success_rate_no_step": sum(bool(row["success_no_step"]) for row in rows) / count,
                "success_rate_no_fall": sum(bool(row["success_no_fall"]) for row in rows) / count,
            }
        )
    return grouped


def _threshold(grouped: list[dict], field: str, threshold: float) -> tuple[float | None, bool]:
    passing = [row for row in grouped if float(row[field]) >= threshold]
    if not passing:
        return None, False
    best = max(passing, key=lambda row: float(row["force_N"]))
    highest_tested = max(float(row["force_N"]) for row in grouped)
    return float(best["force_N"]), math.isclose(float(best["force_N"]), highest_tested)


def main() -> None:
    if args.trials_per_force <= 0:
        raise ValueError("--trials_per_force must be positive")
    forces = sorted(set(float(value) for value in args.force))
    if not forces or any(value < 0.0 for value in forces):
        raise ValueError("force levels must be non-negative")
    duration = args.force_duration
    if duration is None:
        duration = 10.0 if args.mode == "continuous" else 0.1
    post_time = args.post_time
    if post_time is None:
        post_time = 3.0 if args.mode == "continuous" else 5.0
    if duration <= 0.0 or args.stabilization_time < 0.0 or post_time < 0.0:
        raise ValueError("invalid protocol timing")
    direction = np.asarray(args.push_direction_body, dtype=np.float64)
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("horizontal push direction must be non-zero")
    direction /= norm

    plans = _make_plans(forces, duration)
    env, runner, checkpoint = _policy_environment(len(plans))
    if not math.isclose(float(env.step_dt), CONTROL_DT, abs_tol=1.0e-9):
        raise RuntimeError(f"control dt mismatch: {env.step_dt}")
    extractor = G1PrivilegedStateExtractor(
        env,
        G1StateExtractorCfg(
            h_eff=0.6884990671277046,
            contact_force_threshold=5.0,
            min_touchdown_interval=0.08,
        ),
    )
    body_ids, body_names = env.robot.find_bodies([args.application_link], preserve_order=True)
    if body_names != [args.application_link] or len(body_ids) != 1:
        raise RuntimeError(f"application link resolution failed: {body_ids}, {body_names}")
    body_id = int(body_ids[0])
    offset_body = torch.tensor(args.application_offset_body, dtype=torch.float32, device=env.device)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.csv"
    trace_path = output_dir / "traces.csv"
    result_stream = result_path.open("w", encoding="utf-8", newline="")
    result_writer = csv.DictWriter(result_stream, fieldnames=RESULT_FIELDS)
    result_writer.writeheader()
    trace_stream = None
    trace_writer = None
    if not args.no_traces:
        trace_stream = trace_path.open("w", encoding="utf-8", newline="")
        trace_writer = csv.DictWriter(trace_stream, fieldnames=TRACE_FIELDS)
        trace_writer.writeheader()

    count = len(plans)
    alive = np.ones(count, dtype=bool)
    fall_flag = np.zeros(count, dtype=bool)
    fall_phase = np.full(count, "", dtype=object)
    step_force = np.zeros(count, dtype=bool)
    step_post = np.zeros(count, dtype=bool)
    step_count = np.zeros(count, dtype=np.int64)
    max_touchdown_displacement = np.zeros(count, dtype=np.float64)
    initial_feet = np.full((count, 2, 3), np.nan, dtype=np.float64)
    airborne_counter = np.zeros((count, 2), dtype=np.int64)
    airborne_qualified = np.zeros((count, 2), dtype=bool)
    recovery_time = np.full(count, np.nan, dtype=np.float64)
    settle_counter = np.zeros(count, dtype=np.int64)
    force_start_time = np.full(count, np.nan, dtype=np.float64)
    force_end_time = np.full(count, np.nan, dtype=np.float64)
    first_force_world = np.full((count, 3), np.nan, dtype=np.float64)
    last_force_world = np.full((count, 3), np.nan, dtype=np.float64)
    force_world_sum = np.zeros((count, 3), dtype=np.float64)
    force_sample_count = np.zeros(count, dtype=np.int64)
    first_point_world = np.full((count, 3), np.nan, dtype=np.float64)
    final_snapshot = [None] * count

    stabilization_steps = math.ceil(args.stabilization_time / CONTROL_DT)
    force_steps = math.ceil(duration / CONTROL_DT)
    post_steps = math.ceil(post_time / CONTROL_DT)
    actual_duration = force_steps * CONTROL_DT
    airborne_min_steps = max(1, math.ceil(args.airborne_min_time / CONTROL_DT))
    settle_hold_steps = max(1, math.ceil(args.settle_hold_time / CONTROL_DT))
    total_steps = stabilization_steps + force_steps + post_steps

    obs, _ = env.get_observations()
    inference_policy = runner.get_inference_policy(device=env.device)
    _set_standing_command(env)
    previous_state = extractor.extract()

    try:
        for policy_step in range(total_steps):
            phase = _phase(policy_step, stabilization_steps, force_steps)
            _set_standing_command(env)
            root_quat = env.robot.data.root_quat_w
            heading = yaw_quat(root_quat)
            local_force = torch.zeros((count, 3), dtype=torch.float32, device=env.device)
            if phase == "force":
                magnitudes = torch.tensor(
                    [plan["force_N"] for plan in plans], dtype=torch.float32, device=env.device
                )
                local_direction = torch.tensor(direction, dtype=torch.float32, device=env.device)
                local_force = magnitudes.unsqueeze(-1) * local_direction.unsqueeze(0)
                local_force[torch.as_tensor(~alive, device=env.device)] = 0.0
            world_force = quat_apply(heading, local_force)

            if args.application_point == "body_com":
                point_world = env.robot.data.body_com_pos_w[:, body_id, :3].clone()
            else:
                point_world = env.robot.data.body_link_pos_w[:, body_id, :3].clone()
            if torch.any(offset_body != 0.0):
                link_quat = env.robot.data.body_link_quat_w[:, body_id]
                point_world += quat_apply(link_quat, offset_body.unsqueeze(0).expand(count, -1))
            zeros = torch.zeros_like(world_force).unsqueeze(1)
            env.robot.permanent_wrench_composer.set_forces_and_torques(
                forces=world_force.unsqueeze(1),
                torques=zeros,
                positions=point_world.unsqueeze(1),
                body_ids=[body_id],
                is_global=True,
            )

            if phase == "force":
                world_force_np_pre = world_force.detach().cpu().numpy()
                point_world_np_pre = point_world.detach().cpu().numpy()
                newly_started = np.isnan(force_start_time) & alive
                current_pre_time = float(env.sim_step_counter) * float(env.physics_dt)
                force_start_time[newly_started] = current_pre_time
                first_force_world[newly_started] = world_force_np_pre[newly_started]
                first_point_world[newly_started] = point_world_np_pre[newly_started]
                last_force_world[alive] = world_force_np_pre[alive]
                force_world_sum[alive] += world_force_np_pre[alive]
                force_sample_count[alive] += 1

            with torch.inference_mode():
                actions = inference_policy(obs)
                obs, _, dones, _ = env.step(actions)
                state = extractor.extract()

            done_mask = (dones | state.episode_reset).detach().cpu().numpy().astype(bool)
            contacts = state.contacts.detach().cpu().numpy().astype(bool)
            touchdown = state.touchdown.detach().cpu().numpy().astype(bool)
            touchdown_foot = state.touchdown_foot.detach().cpu().numpy()
            left_foot = state.left_foot_position_w.detach().cpu().numpy()
            right_foot = state.right_foot_position_w.detach().cpu().numpy()
            feet = np.stack((left_foot, right_foot), axis=1)
            root_pos = env.robot.data.root_pos_w.detach().cpu().numpy()
            root_lin_vel = env.robot.data.root_lin_vel_w.detach().cpu().numpy()
            root_ang_vel = env.robot.data.root_ang_vel_w.detach().cpu().numpy()
            roll, pitch, yaw = euler_xyz_from_quat(env.robot.data.root_quat_w)
            rpy = torch.stack((roll, pitch, yaw), dim=-1).detach().cpu().numpy()
            com_pos = state.com_position.detach().cpu().numpy()
            com_vel = state.com_velocity.detach().cpu().numpy()
            world_force_np = world_force.detach().cpu().numpy()
            point_world_np = point_world.detach().cpu().numpy()
            current_time = float(state.time[0].item())

            if policy_step == stabilization_steps - 1:
                initial_feet[:] = feet
            if policy_step == stabilization_steps + force_steps - 1:
                force_end_time[alive] = current_time

            step_event = np.zeros(count, dtype=bool)
            if phase in ("force", "post"):
                lift = feet[:, :, 2] - initial_feet[:, :, 2]
                airborne_now = (~contacts) & (lift >= args.airborne_min_height) & alive[:, None]
                airborne_counter = np.where(airborne_now, airborne_counter + 1, 0)
                airborne_qualified |= airborne_counter >= airborne_min_steps
                for env_id in np.flatnonzero(touchdown & alive):
                    foot = int(touchdown_foot[env_id])
                    if foot not in (0, 1):
                        continue
                    displacement = float(
                        np.linalg.norm(feet[env_id, foot, :2] - initial_feet[env_id, foot, :2])
                    )
                    max_touchdown_displacement[env_id] = max(
                        max_touchdown_displacement[env_id], displacement
                    )
                    if airborne_qualified[env_id, foot] and displacement > args.step_displacement_threshold:
                        step_event[env_id] = True
                        step_count[env_id] += 1
                        if phase == "force":
                            step_force[env_id] = True
                        else:
                            step_post[env_id] = True
                    airborne_qualified[env_id, foot] = False
                    airborne_counter[env_id, foot] = 0

            new_falls = done_mask & alive
            fall_flag[new_falls] = True
            fall_phase[new_falls] = phase
            alive[new_falls] = False

            if phase == "post":
                horizontal_speed = np.linalg.norm(com_vel[:, :2], axis=1)
                angular_speed = np.linalg.norm(root_ang_vel, axis=1)
                tilt = np.max(np.abs(rpy[:, :2]), axis=1)
                stable = (
                    (horizontal_speed <= args.settle_linear_speed)
                    & (angular_speed <= args.settle_angular_speed)
                    & (tilt <= args.settle_tilt)
                    & alive
                )
                settle_counter = np.where(stable, settle_counter + 1, 0)
                newly_settled = (settle_counter >= settle_hold_steps) & np.isnan(recovery_time)
                release = np.where(np.isnan(force_end_time), current_time, force_end_time)
                recovery_time[newly_settled] = current_time - release[newly_settled]

            for env_id in range(count):
                if alive[env_id]:
                    final_snapshot[env_id] = {
                        "root_pos": root_pos[env_id].copy(),
                        "root_lin_vel": root_lin_vel[env_id].copy(),
                        "root_ang_vel": root_ang_vel[env_id].copy(),
                        "rpy": rpy[env_id].copy(),
                        "com_pos": com_pos[env_id].copy(),
                        "com_vel": com_vel[env_id].copy(),
                    }

            if trace_writer is not None:
                rows = []
                for env_id, plan in enumerate(plans):
                    if not alive[env_id] and not new_falls[env_id]:
                        continue
                    rows.append(
                        {
                            "policy": args.policy,
                            "trial_id": plan["trial_id"],
                            "trial_index": plan["trial_index"],
                            "mode": args.mode,
                            "force_N": plan["force_N"],
                            "duration_s": actual_duration,
                            "impulse_Ns": plan["force_N"] * actual_duration,
                            "time_s": current_time,
                            "phase": phase,
                            "force_world_x": float(world_force_np[env_id, 0]),
                            "force_world_y": float(world_force_np[env_id, 1]),
                            "force_world_z": float(world_force_np[env_id, 2]),
                            "application_point_world_x": float(point_world_np[env_id, 0]),
                            "application_point_world_y": float(point_world_np[env_id, 1]),
                            "application_point_world_z": float(point_world_np[env_id, 2]),
                            "base_pos_x": float(root_pos[env_id, 0]),
                            "base_pos_y": float(root_pos[env_id, 1]),
                            "base_pos_z": float(root_pos[env_id, 2]),
                            "base_lin_vel_x": float(root_lin_vel[env_id, 0]),
                            "base_lin_vel_y": float(root_lin_vel[env_id, 1]),
                            "base_lin_vel_z": float(root_lin_vel[env_id, 2]),
                            "base_ang_vel_x": float(root_ang_vel[env_id, 0]),
                            "base_ang_vel_y": float(root_ang_vel[env_id, 1]),
                            "base_ang_vel_z": float(root_ang_vel[env_id, 2]),
                            "roll": float(rpy[env_id, 0]),
                            "pitch": float(rpy[env_id, 1]),
                            "yaw": float(rpy[env_id, 2]),
                            "com_pos_x": float(com_pos[env_id, 0]),
                            "com_pos_y": float(com_pos[env_id, 1]),
                            "com_pos_z": float(com_pos[env_id, 2]),
                            "com_vel_x": float(com_vel[env_id, 0]),
                            "com_vel_y": float(com_vel[env_id, 1]),
                            "com_vel_z": float(com_vel[env_id, 2]),
                            "left_foot_x": float(feet[env_id, 0, 0]),
                            "left_foot_y": float(feet[env_id, 0, 1]),
                            "left_foot_z": float(feet[env_id, 0, 2]),
                            "right_foot_x": float(feet[env_id, 1, 0]),
                            "right_foot_y": float(feet[env_id, 1, 1]),
                            "right_foot_z": float(feet[env_id, 1, 2]),
                            "left_contact": int(contacts[env_id, 0]),
                            "right_contact": int(contacts[env_id, 1]),
                            "touchdown": int(touchdown[env_id]),
                            "touchdown_foot": int(touchdown_foot[env_id]),
                            "step_event": int(step_event[env_id]),
                            "step_flag": int(step_force[env_id] or step_post[env_id]),
                            "fall_event": int(new_falls[env_id]),
                            "state_is_post_reset": int(new_falls[env_id]),
                        }
                    )
                trace_writer.writerows(rows)

            if (policy_step + 1) % 100 == 0 or policy_step + 1 == total_steps:
                print(
                    f"[push-test] mode={args.mode} step={policy_step + 1}/{total_steps} "
                    f"alive={int(alive.sum())}/{count}",
                    flush=True,
                )

        zero_forces = torch.zeros((count, 1, 3), device=env.device)
        env.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=zero_forces,
            torques=zero_forces,
            body_ids=[body_id],
            is_global=True,
        )

        results = []
        for env_id, plan in enumerate(plans):
            snapshot = final_snapshot[env_id]
            if snapshot is None:
                snapshot = {
                    "root_pos": np.full(3, np.nan),
                    "root_lin_vel": np.full(3, np.nan),
                    "root_ang_vel": np.full(3, np.nan),
                    "rpy": np.full(3, np.nan),
                    "com_pos": np.full(3, np.nan),
                    "com_vel": np.full(3, np.nan),
                }
            no_step_failure = step_force[env_id] or (
                args.mode == "impulse" and step_post[env_id]
            )
            success_no_fall = not fall_flag[env_id]
            result = {
                "policy": args.policy,
                "checkpoint": str(checkpoint),
                "trial_id": plan["trial_id"],
                "trial_index": plan["trial_index"],
                "env_id": env_id,
                "seed": args.seed,
                "mode": args.mode,
                "force_N": plan["force_N"],
                "force_duration_s": actual_duration,
                "impulse_Ns": plan["force_N"] * actual_duration,
                "direction_body": _as_json(direction),
                "application_link": args.application_link,
                "application_point": args.application_point,
                "application_offset_body": _as_json(args.application_offset_body),
                "force_start_time_s": float(force_start_time[env_id]),
                "force_end_time_s": float(force_end_time[env_id]),
                "stabilization_time_s": args.stabilization_time,
                "post_time_s": post_time,
                "first_force_world": _as_json(first_force_world[env_id]),
                "last_force_world": _as_json(last_force_world[env_id]),
                "mean_force_world": _as_json(
                    force_world_sum[env_id] / max(1, force_sample_count[env_id])
                ),
                "first_application_point_world": _as_json(first_point_world[env_id]),
                "initial_left_foot_world": _as_json(initial_feet[env_id, 0]),
                "initial_right_foot_world": _as_json(initial_feet[env_id, 1]),
                "final_base_position_world": _as_json(snapshot["root_pos"]),
                "final_base_linear_velocity_world": _as_json(snapshot["root_lin_vel"]),
                "final_base_angular_velocity_world": _as_json(snapshot["root_ang_vel"]),
                "final_rpy": _as_json(snapshot["rpy"]),
                "final_com_position_heading": _as_json(snapshot["com_pos"]),
                "final_com_velocity_heading": _as_json(snapshot["com_vel"]),
                "fall_flag": bool(fall_flag[env_id]),
                "fall_phase": str(fall_phase[env_id]),
                "step_flag_force": bool(step_force[env_id]),
                "step_flag_post": bool(step_post[env_id]),
                "step_count": int(step_count[env_id]),
                "max_touchdown_displacement_m": float(max_touchdown_displacement[env_id]),
                "recovery_time_s": None
                if np.isnan(recovery_time[env_id])
                else float(recovery_time[env_id]),
                "success_no_step": bool(success_no_fall and not no_step_failure),
                "success_no_fall": bool(success_no_fall),
            }
            results.append(result)
            result_writer.writerow(result)

        grouped = _summary_by_force(results)
        thresholds = {}
        for rate_name in ("success_rate_no_step", "success_rate_no_fall"):
            for threshold_name, threshold_value in (("90", 0.90), ("100", 1.0)):
                force_value, lower_bound = _threshold(grouped, rate_name, threshold_value)
                key = f"F_max_{'no_step' if rate_name.endswith('no_step') else 'no_fall'}_{threshold_name}pct_N"
                thresholds[key] = force_value
                thresholds[f"{key}_is_lower_bound"] = lower_bound
                if args.mode == "impulse":
                    impulse_key = key.replace("F_max", "J_max").replace("_N", "_Ns")
                    thresholds[impulse_key] = None if force_value is None else force_value * actual_duration
                    thresholds[f"{impulse_key}_is_lower_bound"] = lower_bound

        report = {
            "schema_version": 1,
            "policy": args.policy,
            "native_task": POLICY_TASKS[args.policy],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "mode": args.mode,
            "inference_only": True,
            "protocol": {
                "seed": args.seed,
                "standing_command": [0.0, 0.0, 0.0],
                "force_levels_N": forces,
                "trials_per_force": args.trials_per_force,
                "nominal_force_duration_s": duration,
                "actual_force_duration_s": actual_duration,
                "stabilization_time_s": args.stabilization_time,
                "post_time_s": post_time,
                "push_direction_body": direction.tolist(),
                "force_direction_update": "body heading (yaw) converted to world every control step",
                "application_link": args.application_link,
                "application_point": args.application_point,
                "application_offset_body": list(args.application_offset_body),
                "external_force_api": "robot.permanent_wrench_composer.set_forces_and_torques",
                "force_is_velocity_jump": False,
                "control_dt_s": CONTROL_DT,
                "step_displacement_threshold_m": args.step_displacement_threshold,
                "airborne_min_time_s": args.airborne_min_time,
                "airborne_min_height_m": args.airborne_min_height,
                "touchdown_contact_force_threshold_N": 5.0,
                "touchdown_debounce_s": 0.08,
                "fall_criterion": "native task termination",
                "continuous_no_step_window": "active 10 s force only",
                "impulse_no_step_window": "pulse plus post-pulse recovery",
                "no_fall_window": "stabilization, active force, and post-force observation",
                "observation_noise": False,
                "physics_randomization": False,
                "native_reset_randomization": True,
                "initial_state_pairing": "same seed and env/trial ordering within each native policy stack",
            },
            "planned_trial_count": len(plans),
            "completed_trial_count": len(results),
            "thresholds": thresholds,
            "performance_by_force": grouped,
            "files": {
                "results_csv": str(result_path),
                "traces_csv": None if args.no_traces else str(trace_path),
            },
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f"[push-test] wrote {result_path}", flush=True)
        print(f"[push-test] wrote {summary_path}", flush=True)
    finally:
        result_stream.close()
        if trace_stream is not None:
            trace_stream.close()
        # Some Isaac Sim 5.1 headless runs block indefinitely while closing the
        # environment. Batch evaluation has already flushed all CSV/JSON data
        # here, so let process teardown release simulator resources directly.
        if args.force_exit_after_report:
            os._exit(0)
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
