"""Isaac Lab adapter for the controlled protocol-v1.3 physical environment.

Simulator imports are intentionally lazy.  This module depends on native
training tasks for policy observations/inference, but never on the retired
evaluation system.
"""
from __future__ import annotations

import copy
import importlib.metadata
import math
from pathlib import Path
from typing import Any

import numpy as np

from .certificate_offline import OfflineCertificate
from .common import LAB, digest, sha256_file
from .disturbances import (
    apply_velocity_jump, clear_wrenches, requested_wrench,
    set_world_wrenches_at_link_origins,
)
from .model_loader import ModelIdentity, load_native_policy, validate_context_ablation_configs


def runtime_versions() -> dict[str, str]:
    values = {}
    for label, package in (("IsaacLab", "isaaclab"), ("IsaacSim", "isaacsim"), ("torch", "torch")):
        try:
            values[label] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[label] = "unknown"
    return values


def close_environment(env: Any | None) -> None:
    """Close native workers and stop simulation before process teardown.

    Legged Lab's ``BaseEnv`` is a ``VecEnv`` rather than an Isaac DirectRLEnv,
    so it deliberately has no ``close()`` method.  The owning worker process
    is the final resource boundary (see ``run.py``); this helper first closes
    every child worker and stops physics so all persisted output is settled.
    """
    if env is None:
        return
    evaluator = getattr(env, "_certificate_evaluator", None)
    if evaluator is not None:
        evaluator.close()
    # BaseEnv exposes neither close() nor a non-blocking simulator shutdown in
    # Isaac Sim 5.1.  The dedicated worker process is therefore the simulator
    # ownership boundary after native certificate children have been joined.


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if callable(value):
        return f"{value.__module__}:{value.__name__}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return [value.start, value.stop, value.step]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        if math.isnan(float(value)):
            return "NaN"
        return "+Infinity" if float(value) > 0 else "-Infinity"
    return value


def evaluation_environment_seed(protocol: dict[str, Any], stage: str) -> int:
    stages = protocol.get("stages", {})
    if stage not in stages:
        raise ValueError(f"unknown evaluation stage: {stage}")
    return int(stages[stage]["seed_namespace"])


def make_environment(identity: ModelIdentity, protocol: dict[str, Any], *, stage: str, num_envs: int,
                     device: str, headless: bool = True) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    import legged_lab.envs  # noqa: F401 - performs task registration after AppLauncher
    from isaaclab.envs.mdp.commands import UniformVelocityCommand
    from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, quat_from_euler_xyz, yaw_quat
    from legged_lab.terrains import make_plane_recovery_terrain_cfg
    from legged_lab.utils import task_registry

    validate_context_ablation_configs(task_registry)
    env_cfg, agent_cfg = (copy.deepcopy(value) for value in task_registry.get_cfgs(identity.task_name))
    env_class = task_registry.get_task_class(identity.task_name)
    if identity.estimator_path is not None:
        env_cfg.estimator_checkpoint_path = str(identity.estimator_path)
    expected_context = tuple(identity.actor_certificate_context)
    actual_context = tuple(getattr(env_cfg, "actor_recovery_context_fields", ("n_norm", "margin_norm", "valid")))
    if expected_context != actual_context and expected_context:
        raise ValueError(f"checkpoint/task context mismatch: registered={expected_context}, task={actual_context}")
    if float(env_cfg.sim.dt) != protocol["physics"]["dt_phys_s"] or int(env_cfg.sim.decimation) != protocol["physics"]["decimation"]:
        raise ValueError("native task does not satisfy the frozen 200/50 Hz timing")
    regex = protocol["termination"]["termination_body_regex"]
    if env_cfg.robot.terminate_contacts_body_names != [regex]:
        raise ValueError("native termination regex differs from protocol")

    env_cfg.device = device
    agent_cfg.device = device
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.scene.seed = evaluation_environment_seed(protocol, stage)
    env_cfg.scene.max_init_terrain_level = 0
    env_cfg.scene.max_episode_length_s = 60.0
    env_cfg.scene.terrain_type = "generator"
    env_cfg.scene.terrain_generator = make_plane_recovery_terrain_cfg(tuple(protocol["conditions"]["slopes_deg"]))
    env_cfg.scene.terrain_generator.size = tuple(protocol["physics"]["terrain_tile_size_m"])
    env_cfg.scene.terrain_generator.seed = env_cfg.scene.seed
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.action_delay.enable = False
    for name in list(vars(env_cfg.domain_rand.events)):
        if not name.startswith("_"):
            setattr(env_cfg.domain_rand.events, name, None)
    env_cfg.commands.heading_command = False
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.debug_vis = False
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    if hasattr(env_cfg, "push_curriculum"):
        env_cfg.push_curriculum.enable_push_curriculum = False
    if hasattr(env_cfg, "debug_save_num"):
        env_cfg.debug_save_num = 0

    slopes = [float(value) for value in protocol["conditions"]["slopes_deg"]]
    termination = protocol["termination"]

    class FixedCommand(UniformVelocityCommand):
        def _resample_command(self, env_ids):
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            self.vel_command_b[ids] = self._env.eval_commands[ids]
            self.is_heading_env[ids] = False
            self.is_standing_env[ids] = False

        def _update_command(self):
            self._resample_command(torch.arange(self.num_envs, device=self.device))

    class EvaluationEnvironment(env_class):
        def __init__(self, cfg, is_headless):
            self.eval_commands = torch.zeros((cfg.scene.num_envs, 3), device=cfg.device)
            self.eval_plans = None
            self.eval_slopes = torch.zeros(cfg.scene.num_envs, device=cfg.device)
            self.eval_capture = None
            self.eval_runtime = None
            self.eval_physical_failure_mask = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=cfg.device)
            self.eval_controller_failure_mask = torch.zeros_like(self.eval_physical_failure_mask)
            self.eval_last_actions = torch.zeros((cfg.scene.num_envs, 29), device=cfg.device)
            super().__init__(cfg, is_headless)
            self.cfg.scene.terrain_generator.curriculum = False
            self.scene.terrain.cfg.terrain_generator.curriculum = False
            self.eval_origin_table = self.scene.terrain.terrain_origins.clone()
            self.eval_masses = self.robot.root_physx_view.get_masses().clone().to(self.device)
            self.eval_total_mass = self.eval_masses.sum(dim=1, keepdim=True)
            self.eval_foot_ids, foot_names = self.contact_sensor.find_bodies(
                ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            )
            if foot_names != ["left_ankle_roll_link", "right_ankle_roll_link"]:
                raise ValueError(f"foot body resolution failed: {foot_names}")
            self.eval_termination_ids, self.eval_termination_names = self.contact_sensor.find_bodies(regex, preserve_order=True)
            if not self.eval_termination_names:
                raise ValueError(f"termination regex resolved no bodies: {regex}")
            torso_ids, torso_names = self.robot.find_bodies("torso_link", preserve_order=True)
            if torso_names != ["torso_link"]:
                raise ValueError(f"wrench target must resolve uniquely: {torso_names}")
            self.eval_torso_id = int(torso_ids[0])
            asset = Path(self.cfg.scene.robot.spawn.usd_path).resolve(strict=True)
            self.eval_asset_hash = sha256_file(asset)
            if abs(self.physics_dt - protocol["physics"]["dt_phys_s"]) > 1.0e-10 or abs(self.step_dt - protocol["physics"]["dt_policy_s"]) > 1.0e-10:
                raise ValueError("realized simulator timing differs from protocol")

        def _create_command_generator(self, command_cfg):
            return FixedCommand(command_cfg, self)

        def update_terrain_levels(self, env_ids):
            return {}

        def _terrain_log(self, level):
            return {"EvaluationTerrain/curriculum_enabled": 0.0}

        def get_recovery_plane_geometry(self):
            alpha = torch.deg2rad(self.eval_slopes)
            normal = torch.stack((-torch.sin(alpha), torch.zeros_like(alpha), torch.cos(alpha)), dim=-1)
            valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return normal, self.scene.env_origins.clone(), valid

        def reset(self, env_ids):
            if self.eval_runtime is not None:
                self.eval_runtime.before_reset(env_ids)
            super().reset(env_ids)
            if self.eval_plans is None or len(env_ids) == 0:
                return
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            state = self.robot.data.default_root_state[ids].clone()
            scales, velocities = [], []
            for local, env_id in enumerate(ids.tolist()):
                plan = self.eval_plans[env_id]
                pose = plan["initial_pose"]
                velocity = plan["initial_velocity"]
                state[local, :3] += self.scene.env_origins[env_id]
                state[local, :3] += torch.tensor([
                    pose["x"], pose["y"], pose["z"] + math.tan(math.radians(plan["slope_deg"])) * pose["x"]
                ], device=self.device)
                angles = [torch.tensor([pose[key]], device=self.device) for key in ("roll", "pitch", "yaw")]
                state[local, 3:7] = quat_from_euler_xyz(*angles)[0]
                state[local, 7:13] += torch.tensor([velocity[key] for key in ("x", "y", "z", "roll", "pitch", "yaw")], device=self.device)
                scales.append(plan["initial_joint_position_scale"])
                velocities.append(plan["initial_joint_velocity"])
            joint_position = self.robot.data.default_joint_pos[ids] * torch.tensor(scales, device=self.device)
            limits = self.robot.data.soft_joint_pos_limits[ids]
            joint_position = joint_position.clamp(limits[..., 0], limits[..., 1])
            joint_velocity = torch.tensor(velocities, device=self.device)
            self.robot.write_root_state_to_sim(state, env_ids=ids)
            self.robot.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=ids)
            self.command_generator._resample_command(ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.eval_physical_failure_mask[ids] = False
            self.eval_controller_failure_mask[ids] = False

        def begin_batch(self, plans):
            if len(plans) != self.num_envs:
                raise ValueError("batch plans must be padded to exactly num_envs")
            self.eval_plans = plans
            ids = torch.arange(self.num_envs, device=self.device)
            terrain_types = torch.tensor([slopes.index(float(plan["slope_deg"])) for plan in plans], device=self.device)
            self.scene.terrain.terrain_levels[:] = 0
            self.scene.terrain.terrain_types[:] = terrain_types
            origins = self.eval_origin_table[0, terrain_types]
            self.scene.terrain.env_origins[:] = origins
            self.scene.env_origins[:] = origins
            self.eval_slopes[:] = torch.tensor([plan["slope_deg"] for plan in plans], device=self.device)
            self.eval_commands[:] = torch.tensor([
                [plan["command_vx"], plan["command_vy"], plan["command_yaw"]] for plan in plans
            ], device=self.device)
            # Manifest-controlled material is set per articulation/environment.
            material = self.robot.root_physx_view.get_material_properties().clone()
            for index, plan in enumerate(plans):
                material[index, :, 0] = plan["friction"]["static"]
                material[index, :, 1] = plan["friction"]["dynamic"]
                material[index, :, 2] = plan["friction"]["restitution"]
            self.robot.root_physx_view.set_material_properties(material, torch.arange(self.num_envs, device="cpu"))
            self.reset(ids)
            observation, extras = self.get_observations()
            return observation, extras

        def snapshot_rows(self, *, fell=None, timeout=None):
            data = self.robot.data
            heading = yaw_quat(data.root_quat_w)
            weighted_velocity = (self.eval_masses.unsqueeze(-1) * data.body_com_vel_w[..., :3]).sum(1) / self.eval_total_mass
            weighted_position = (self.eval_masses.unsqueeze(-1) * data.body_com_pose_w[..., :3]).sum(1) / self.eval_total_mass
            com_heading = quat_apply_inverse(heading, weighted_velocity)
            root_heading = quat_apply_inverse(heading, data.root_lin_vel_w)
            roll, pitch, yaw = euler_xyz_from_quat(data.root_quat_w)
            wrap = lambda values: torch.atan2(torch.sin(values), torch.cos(values))
            relative = data.root_pos_w - self.scene.env_origins
            alpha = torch.deg2rad(self.eval_slopes)
            clearance = relative[:, 2] - torch.tan(alpha) * relative[:, 0]
            body_up = quat_apply(data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1))
            tilt = torch.acos(torch.clamp(body_up[:, 2], -1.0, 1.0))
            foot_forces = torch.linalg.vector_norm(self.contact_sensor.data.net_forces_w[:, self.eval_foot_ids], dim=-1)
            termination_forces = self.contact_sensor.data.net_forces_w[:, self.eval_termination_ids]
            outside = relative[:, :2].abs().amax(dim=-1) > protocol["physics"]["test_area_half_extent_m"]
            fell_value = fell if fell is not None else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            timeout_value = timeout if timeout is not None else torch.zeros_like(fell_value)
            values = {
                "root_position": relative, "root_quaternion_wxyz": data.root_quat_w,
                "root_velocity_world": data.root_vel_w, "root_velocity_heading": root_heading,
                "com_position_world": weighted_position, "com_velocity_world": weighted_velocity,
                "com_velocity_heading": com_heading, "joint_position": data.joint_pos,
                "joint_velocity": data.joint_vel, "actions": self.eval_last_actions,
                "contact_forces_world": self.contact_sensor.data.net_forces_w,
                "foot_force_norms_N": foot_forces, "termination_contact_forces_world": termination_forces,
                "roll_pitch": torch.stack((wrap(roll), wrap(pitch)), dim=-1), "yaw": wrap(yaw),
                "gravity_tilt_rad": tilt, "angular_velocity_world_z": data.root_vel_w[:, 5],
                "pelvis_clearance_m": clearance, "command": self.command_generator.command,
                "out_of_area": outside, "fell": fell_value, "timeout": timeout_value,
            }
            valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            for key, value in values.items():
                if hasattr(value, "dtype") and value.dtype != torch.bool:
                    valid &= torch.isfinite(value).reshape(self.num_envs, -1).all(dim=-1)
            values["data_valid"] = valid
            cpu = {key: value.detach().cpu().numpy().copy() for key, value in values.items()}
            return [{key: (array[index].tolist() if array[index].ndim else array[index].item())
                     for key, array in cpu.items()} for index in range(self.num_envs)]

        def check_reset(self):
            native, timeout = super().check_reset()
            done = native | self.eval_physical_failure_mask | self.eval_controller_failure_mask
            if self.eval_plans is not None:
                self.eval_capture = self.snapshot_rows(fell=done & ~timeout, timeout=timeout)
            return done, timeout

    env = EvaluationEnvironment(env_cfg, headless)
    runner, policy = load_native_policy(env, agent_cfg, identity, device)
    actual_physics = {
        "dt_phys_s": float(env.physics_dt), "dt_policy_s": float(env.step_dt),
        "masses": env.eval_masses[0].detach().cpu().tolist(),
        "joint_stiffness": env.robot.data.joint_stiffness[0].detach().cpu().tolist(),
        "joint_damping": env.robot.data.joint_damping[0].detach().cpu().tolist(),
        "termination_body_names": env.eval_termination_names,
        "asset_hash": env.eval_asset_hash,
    }
    versions = runtime_versions()
    effective = {
        "task_name": identity.task_name, "model_id": identity.model_id,
        "environment": _plain(env_cfg.to_dict()), "actual_physics": actual_physics,
        "actual_physics_hash": digest(actual_physics), "asset_hash": env.eval_asset_hash,
        "versions": versions, "simulator_version": f"IsaacSim={versions['IsaacSim']};IsaacLab={versions['IsaacLab']}",
        "termination_body_names": env.eval_termination_names,
        "actor_certificate_context": list(actual_context if identity.actor_certificate_context else ()),
    }
    return env, runner, policy, effective


class DisturbanceRuntime:
    """Inject disturbances and monitor physical failure at every 200 Hz substep."""

    def __init__(self, env: Any, identity: ModelIdentity, protocol: dict[str, Any], machines: list[Any | None],
                 *, certificate_enabled: bool):
        import torch

        self.torch = torch
        self.env = env
        self.identity = identity
        self.protocol = protocol
        self.machines = machines
        self.base_counter = int(env.sim_step_counter)
        self.dt = float(env.physics_dt)
        self.original_write = env.scene.write_data_to_sim
        self.original_update = env.scene.update
        self.original_reset = env.reset
        self.released: set[int] = set()
        self.hpush_yaw: dict[int, float] = {}
        self.last_wrench_update: dict[int, int | None] = {}
        self.env.scene.write_data_to_sim = self.write_data_to_sim
        self.env.scene.update = self.update_scene
        self.env.eval_runtime = self
        capability = LAB / "tools/recovery/generated/g1_recovery_params.yaml"
        nominal = LAB / "tools/recovery/generated/g1_plane_nominal_params_g1_slope_sys_d_candidate.yaml"
        self.certificate = OfflineCertificate(env, capability_path=capability, nominal_path=nominal) if certificate_enabled else None

    def before_reset(self, env_ids: Any) -> None:
        clear_wrenches(self.env.robot.permanent_wrench_composer, env_ids)

    def _time_start(self) -> float:
        return (int(self.env.sim_step_counter) - self.base_counter - 1) * self.dt

    def _time_end(self) -> float:
        return (int(self.env.sim_step_counter) - self.base_counter) * self.dt

    def _start(self, index: int, machine: Any, timestamp: float) -> None:
        torch = self.torch
        env = self.env
        disturbance = machine.plan["disturbance"]
        _, _, yaw = __import__("isaaclab.utils.math", fromlist=["euler_xyz_from_quat"]).euler_xyz_from_quat(
            env.robot.data.root_quat_w[index:index + 1]
        )
        frozen_yaw = float(torch.atan2(torch.sin(yaw), torch.cos(yaw))[0].item())
        self.hpush_yaw[index] = frozen_yaw
        metadata: dict[str, Any] = {"Hpush_yaw": frozen_yaw, "target_link": disturbance["target_link"]}
        if disturbance["type"] == "velocity_jump":
            view = env.robot.root_physx_view
            before_root = view.get_root_velocities()[index].detach().cpu().numpy().copy()
            before_pose = view.get_root_transforms()[index].detach().cpu().numpy().copy()
            before_joints = view.get_dof_positions()[index].detach().cpu().numpy().copy()
            before_joint_vel = view.get_dof_velocities()[index].detach().cpu().numpy().copy()
            before_link_velocities = view.get_link_velocities()[index, :, :3]
            before_com = ((env.eval_masses[index, :, None] * before_link_velocities).sum(0)
                          / env.eval_total_mass[index]).detach().cpu().numpy().copy()
            expected = apply_velocity_jump(before_root, disturbance, frozen_yaw)
            ids = torch.tensor([index], dtype=torch.long, device=env.device)
            env.robot.write_root_velocity_to_sim(torch.tensor(expected[None], dtype=torch.float32, device=env.device), env_ids=ids)
            actual = view.get_root_velocities()[index].detach().cpu().numpy().copy()
            after_link_velocities = view.get_link_velocities()[index, :, :3]
            after_com = ((env.eval_masses[index, :, None] * after_link_velocities).sum(0)
                         / env.eval_total_mass[index]).detach().cpu().numpy().copy()
            if not np.allclose(actual, expected, atol=5.0e-6, rtol=0.0):
                raise ValueError("declared velocity jump differs from physics state before integration")
            invariants = (
                (view.get_root_transforms()[index].detach().cpu().numpy(), before_pose, "root pose"),
                (view.get_dof_positions()[index].detach().cpu().numpy(), before_joints, "joint position"),
                (view.get_dof_velocities()[index].detach().cpu().numpy(), before_joint_vel, "joint velocity"),
            )
            for after_value, before_value, label in invariants:
                if not np.array_equal(after_value, before_value):
                    raise ValueError(f"velocity jump unexpectedly changed {label} before integration")
            delta = actual[:3] - before_root[:3]
            if self.identity.actor_certificate_context and hasattr(env, "_on_curriculum_push"):
                active = bool(env.recovery_active[index]) if hasattr(env, "recovery_active") else False
                if not active:
                    env._on_curriculum_push(ids, torch.tensor(delta[:2][None], dtype=torch.float32, device=env.device),
                                            torch.zeros(1, dtype=torch.long, device=env.device))
                    if not np.array_equal(env.robot.data.root_vel_w[index].detach().cpu().numpy(), actual):
                        raise ValueError("native certificate notification changed physics a second time")
            metadata.update({
                "root_velocity_before": before_root.tolist(), "root_velocity_after": actual.tolist(),
                "com_velocity_before": before_com.tolist(), "com_velocity_after": after_com.tolist(),
                "actual_delta": (actual - before_root).tolist(),
                "pose_joint_invariants_verified": True,
            })
        else:
            metadata.update({"wrench_frame": "world/Hpush converted to current link frame each physics substep",
                             "application_point": None})
        machine.on_disturbance_start(timestamp, metadata)

    def write_data_to_sim(self):
        try:
            timestamp = self._time_start()
            env = self.env
            forces = np.zeros((env.num_envs, 3), dtype=np.float64)
            torques = np.zeros_like(forces)
            for index, machine in enumerate(self.machines):
                if machine is None or machine.status:
                    continue
                planned = float(machine.plan.get("planned_disturbance_start_s", math.inf))
                if not machine.push_applied and timestamp + 1.0e-9 >= planned:
                    self._start(index, machine, timestamp)
                if not machine.push_applied or machine.disturbance_end is None:
                    continue
                disturbance = machine.plan["disturbance"]
                if disturbance["type"] == "velocity_jump":
                    continue
                if timestamp >= machine.disturbance_end - 1.0e-9:
                    if index not in self.released:
                        machine.on_disturbance_end(timestamp)
                        self.released.add(index)
                    continue
                force, torque, update_index = requested_wrench(
                    disturbance, self.hpush_yaw[index], timestamp - machine.disturbance_start
                )
                forces[index], torques[index] = force, torque
                if update_index != self.last_wrench_update.get(index):
                    machine.events.append({"event": "wrench_update", "timestamp": timestamp,
                                           "update_index": update_index, "force_world_N": force.tolist(),
                                           "torque_world_Nm": torque.tolist()})
                    machine.disturbance_metadata.setdefault("force_vector", []).append(force.tolist())
                    machine.disturbance_metadata.setdefault("torque_vector", []).append(torque.tolist())
                    self.last_wrench_update[index] = update_index
            quaternions = env.robot.data.body_link_quat_w[:, env.eval_torso_id].detach().cpu().numpy()
            set_world_wrenches_at_link_origins(
                env.robot.permanent_wrench_composer, forces, torques, quaternions,
                env.eval_torso_id, env.device,
            )
            return self.original_write()
        except BaseException:
            clear_wrenches(self.env.robot.permanent_wrench_composer)
            raise

    def update_scene(self, *args, **kwargs):
        result = self.original_update(*args, **kwargs)
        env, torch = self.env, self.torch
        timestamp = self._time_end()
        foot = torch.linalg.vector_norm(env.contact_sensor.data.net_forces_w[:, env.eval_foot_ids], dim=-1)
        termination_force = torch.linalg.vector_norm(
            env.contact_sensor.data.net_forces_w_history[:, :, env.eval_termination_ids], dim=-1
        ).amax(dim=(1, 2))
        from isaaclab.utils.math import euler_xyz_from_quat
        roll, pitch, _ = euler_xyz_from_quat(env.robot.data.root_quat_w)
        roll = torch.atan2(torch.sin(roll), torch.cos(roll)).abs()
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch)).abs()
        relative = env.robot.data.root_pos_w - env.scene.env_origins
        outside = relative[:, :2].abs().amax(dim=-1) > self.protocol["physics"]["test_area_half_extent_m"]
        finite = torch.isfinite(env.robot.data.root_state_w).all(dim=-1)
        reasons: list[str | None] = [None] * env.num_envs
        for index in range(env.num_envs):
            if termination_force[index] > self.protocol["termination"]["termination_force_N"]:
                reasons[index] = "TERMINATION_BODY_CONTACT"
            elif roll[index] > self.protocol["termination"]["roll_limit_rad"]:
                reasons[index] = "ROLL_LIMIT"
            elif pitch[index] > self.protocol["termination"]["pitch_limit_rad"]:
                reasons[index] = "PITCH_LIMIT"
            elif outside[index]:
                reasons[index] = "OUT_OF_TEST_AREA"
            elif not finite[index]:
                reasons[index] = "CONTROLLER_NUMERICAL_FAILURE"
        certificate_ids = []
        certificate_machines = []
        force_cpu = foot.detach().cpu().numpy()
        for index, machine in enumerate(self.machines):
            if machine is None or machine.status:
                continue
            previous_td0 = machine.td0_time
            machine.feed_physics(timestamp, force_cpu[index], reasons[index])
            if reasons[index] is not None:
                env.eval_physical_failure_mask[index] = True
                clear_wrenches(env.robot.permanent_wrench_composer, [index])
            if self.certificate is not None and previous_td0 is None and machine.td0_time is not None:
                certificate_ids.append(index)
                certificate_machines.append(machine)
        if certificate_ids:
            # Extract the exact current physics state only on the TD0 substep.
            state = self.certificate.extract()
            ids = torch.tensor(certificate_ids, dtype=torch.long, device=env.device)
            for machine, certificate in zip(certificate_machines, self.certificate.evaluate(state, ids)):
                machine.set_certificate(**certificate)
        # Capture exact non-policy-aligned observation endpoints using simulator timestamps.
        endpoint_indices = [index for index, machine in enumerate(self.machines)
                            if machine is not None and not machine.status and machine.disturbance_end is not None
                            and abs(timestamp - (machine.disturbance_end + 10.0)) <= 1.0e-8]
        if endpoint_indices:
            rows = env.snapshot_rows()
            for index in endpoint_indices:
                rows[index]["timestamp"] = timestamp
                self.machines[index].feed_policy(rows[index])
        return result

    def close(self) -> None:
        try:
            clear_wrenches(self.env.robot.permanent_wrench_composer)
            if self.certificate is not None:
                self.certificate.close()
        finally:
            self.env.scene.write_data_to_sim = self.original_write
            self.env.scene.update = self.original_update
            self.env.eval_runtime = None
