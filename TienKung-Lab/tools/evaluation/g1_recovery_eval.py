#!/usr/bin/env python3
"""G1 Recovery Eval: offline commands are independent of Isaac Sim and GPU availability."""
from __future__ import annotations
import argparse
import copy
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_recovery_protocol import (
    LAB, DEFAULT_PROTOCOL, TASKS, RunStore, atomic_write, canonical, code_identity, csv_bytes, digest,
    evaluation_key, import_results, inspect_checkpoint, jsonl, load_prepared, load_yaml, lock,
    prepare, read_json, read_jsonl, register_model, register_result, sha256, write_json,
    RESOURCE_FIELDS, native_contract, validate_resource_identity, is_common, COMMON_PROTOCOL,
)
from g1_recovery_metrics import TrialMachine, make_trial_machine, additive_velocity, summarize
from g1_common_task_detector import METRICS_VERSION as COMMON_METRICS_VERSION
from g1_development_validation import (DEVELOPMENT_MANIFEST, DEVELOPMENT_MANIFEST_SHA256,
    select_development_assignment, realized_environment_payload, measure_realized_slopes, TERRAIN_SOURCES)


def runtime_environment_versions():
    """Called only after the existing AppLauncher, never by offline preparation."""
    from pxr import Usd
    import omni.kit.app
    versions = {}
    for key, package in (('IsaacLab', 'isaaclab'), ('IsaacSim', 'isaacsim'), ('trimesh', 'trimesh'), ('numpy', 'numpy')):
        try:
            versions[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if key == 'IsaacLab':
                import isaaclab
                versions[key] = isaaclab.__version__
            elif key == 'IsaacSim':
                from isaacsim.core.version import get_version
                versions[key] = '.'.join(map(str, get_version()))
            else:
                raise ValueError('Missing terrain/runtime package version: ' + package)
    versions['USD'] = '.'.join(map(str, Usd.GetVersion()))
    app = omni.kit.app.get_app()
    versions['PhysX'] = app.get_extension_manager().get_enabled_extension_id('omni.physx')
    versions['Kit'] = app.get_build_version()
    return versions


def apply_trial_intervention(env, machine, index, frame, identity):
    """One causal trigger; sham never reaches the physics write or native push hook."""
    import numpy as np
    import torch
    before = env.robot.data.root_vel_w[index].detach().cpu().numpy().copy()
    if getattr(machine, 'sham', False):
        after = env.robot.data.root_vel_w[index].detach().cpu().numpy().copy()
        machine.apply_sham_marker(before.tolist(), after.tolist())
        return
    ids = torch.tensor([index], device=env.device)
    after = additive_velocity(before, frame['yaw'], machine.plan['push_direction_heading'], machine.plan['push_magnitude'])
    env.robot.write_root_velocity_to_sim(torch.tensor(after[None], dtype=torch.float32, device=env.device), env_ids=ids)
    actual = env.robot.data.root_vel_w[index].detach().cpu().numpy().copy()
    machine.push_applied(before.tolist(), actual.tolist(), frame['yaw'])
    if identity['method'].startswith('context'):
        delta = torch.tensor([machine.plan['push_direction_heading']], device=env.device) * machine.plan['push_magnitude']
        env._on_curriculum_push(ids, delta, torch.zeros(1, dtype=torch.long, device=env.device))
        if not np.allclose(env.robot.data.root_vel_w[index].cpu().numpy(), actual):
            raise ValueError('Plane hook applied a second physical jump')


def snapshot_checkpoint(root, identity):
    """Copy before simulation, verify source stability and deserialization, make immutable."""
    import torch
    variant = digest({k: identity.get(k) for k in ('agent_config_sha256', 'env_config_sha256',
                     'native_nominal_sha256', 'native_capability_sha256', 'estimator_sha256',
                     'native_configuration_sha256')})[:24]
    dest = Path(root) / 'checkpoints' / identity['checkpoint_sha256'] / variant
    with lock(Path(root) / '.snapshot.lock'):
        dest.mkdir(parents=True, exist_ok=True)
        source = getattr(identity, 'inputs', {}).get('checkpoint', LAB / identity['checkpoint_path'])
        target = dest / 'model.pt'
        if not target.exists():
            atomic_write(target, source.read_bytes())
        if sha256(target) != identity['checkpoint_sha256'] or sha256(source) != identity['checkpoint_sha256']:
            raise ValueError('Checkpoint snapshot SHA mismatch/source changed')
        torch.load(target, map_location='cpu', weights_only=False)
        target.chmod(0o444)
        for filename, key in [('agent.yaml', 'agent_config_sha256'), ('env.yaml', 'env_config_sha256')]:
            src = source.parent / 'params' / filename
            if sha256(src) != identity[key]:
                raise ValueError('Training config changed during snapshot')
            output = dest / 'params' / filename
            if output.exists() and sha256(output) != identity[key]:
                raise ValueError('Same weights have conflicting training configurations')
            if not output.exists():
                atomic_write(output, src.read_bytes())
            if sha256(output) != identity[key] or sha256(src) != identity[key]:
                raise ValueError('Training config SHA mismatch during snapshot')
            output.chmod(0o444)
        for role, resource in identity.get('resources', {}).items():
            source_resource = identity.inputs[role]
            output = dest / resource['snapshot']
            data = source_resource.read_bytes()
            if sha256(source_resource) != resource['sha256']:
                raise ValueError(f'{role}: source changed during snapshot')
            if not output.exists():
                atomic_write(output, data)
            if sha256(output) != resource['sha256']:
                raise ValueError(f'{role}: snapshot SHA mismatch')
            output.chmod(0o444)
        config_path = dest / 'native_configuration.json'
        if config_path.exists() and read_json(config_path) != identity['native_configuration']:
            raise ValueError('Native configuration snapshot conflict')
        if not config_path.exists():
            write_json(config_path, identity['native_configuration'])
        config_path.chmod(0o444)
        verify_native_snapshot(identity, target)
        identity['checkpoint_snapshot'] = target.relative_to(Path(root)).as_posix()
        if not source.is_relative_to(LAB):
            identity['checkpoint_path'] = identity['checkpoint_snapshot']
            identity['checkpoint_path_base'] = 'output_root'
    return target


def verify_native_snapshot(identity, snapshot):
    validate_resource_identity(identity)
    for path, key in [(snapshot, 'checkpoint_sha256'), (snapshot.parent / 'params/env.yaml', 'env_config_sha256'),
                      (snapshot.parent / 'params/agent.yaml', 'agent_config_sha256')]:
        if sha256(path) != identity[key]:
            raise ValueError(f'{key}: snapshot SHA mismatch')
    paths = {}
    for role, resource in identity.get('resources', {}).items():
        path = (snapshot.parent / resource['snapshot']).resolve(strict=True)
        if not path.is_relative_to(snapshot.parent.resolve()) or sha256(path) != resource['sha256']:
            raise ValueError(f'{role}: snapshot SHA mismatch or unsafe path')
        paths[role] = path
    config = read_json(snapshot.parent / 'native_configuration.json')
    if digest(config) != identity['native_configuration_sha256'] or config != identity['native_configuration']:
        raise ValueError('Native configuration snapshot mismatch')
    return paths


def bind_native_inputs(cfg, identity, snapshot):
    """Called before constructing the environment (and therefore any worker)."""
    paths = verify_native_snapshot(identity, snapshot)
    if not identity['method'].startswith('context'):
        return paths
    if native_contract(cfg.to_dict()) != identity['native_configuration']:
        raise ValueError('Native solver/context configuration differs from training snapshot')
    for role, (section, field) in RESOURCE_FIELDS.items():
        setattr(getattr(cfg, section), field, str(paths[role]))
    cfg.estimator_checkpoint_path = str(paths['estimator'])
    assert_native_inputs(cfg, identity)
    return paths


def assert_native_inputs(cfg, identity):
    validate_resource_identity(identity)
    for role, (section, field) in RESOURCE_FIELDS.items():
        if sha256(getattr(getattr(cfg, section), field)) != identity['resources'][role]['sha256']:
            raise ValueError(f'{role}: actual solver input SHA differs from identity')
    if sha256(cfg.estimator_checkpoint_path) != identity['estimator_sha256']:
        raise ValueError('Actual estimator SHA differs from identity')


def make_evaluation_environment(args, p, identity, snapshot):
    """Lazy simulator adapter; bypass only matched terrain constructors, retain native stepping.

    Source: BaseEnv/G1DwaqEnv/G1PlaneV1Env and state_extractor mass-weighted CoM
    calculation. No training environment or global factory is patched.
    """
    import numpy as np
    import torch
    import legged_lab.envs  # task registration, after AppLauncher only
    from legged_lab.utils import task_registry
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.g1.g1_dwaq_env import G1DwaqEnv
    from legged_lab.envs.g1.g1_plane_v1_env import G1PlaneV1Env
    from isaaclab.envs.mdp.commands import UniformVelocityCommand
    from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat, quat_apply_inverse, yaw_quat
    from legged_lab.terrains import make_plane_recovery_terrain_cfg
    from rsl_rl.runners import OnPolicyRunner, DWAQOnPolicyRunner

    method = identity['method']
    native = G1DwaqEnv if method == 'dwaq' else G1PlaneV1Env if method.startswith('context') else BaseEnv
    cfg, agent = (copy.deepcopy(c) for c in task_registry.get_cfgs(args.task))
    native_paths = bind_native_inputs(cfg, identity, snapshot)
    saved_env, saved_agent = load_yaml(snapshot.parent / 'params/env.yaml'), load_yaml(snapshot.parent / 'params/agent.yaml')
    # Keep native input semantics; reject config drift rather than silently loading a new input chain.
    current = cfg.to_dict()
    def plain(v):
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        if callable(v):
            return v.__module__ + ':' + v.__name__
        if isinstance(v, slice):
            return [v.start, v.stop, v.step]
        if isinstance(v, Path):
            v = str(v)
        if isinstance(v, str) and Path(v).is_absolute():
            candidate = Path(v)
            try:
                return candidate.relative_to(LAB).as_posix()
            except ValueError:
                return f'external/{candidate.name}'
        return v
    def asset_contract(value):
        value = copy.deepcopy(plain(value))
        # Repository relocation is not a change to the native input contract.
        # Actual USD contents are independently hashed in the effective profile.
        value['spawn']['usd_path'] = Path(value['spawn']['usd_path']).name
        return value
    common_cfg = task_registry.get_cfgs(TASKS['ppo_plain'])[0]
    if asset_contract(current['scene']['robot']) != asset_contract(saved_env['scene']['robot']):
        raise ValueError('Native robot asset differs from training snapshot')
    if plain(cfg.scene.robot.to_dict()) != plain(common_cfg.scene.robot.to_dict()):
        raise ValueError('Robot asset differs from the common controlled physics profile')
    if cfg.sim.dt != common_cfg.sim.dt or cfg.sim.decimation != common_cfg.sim.decimation:
        raise ValueError('Physics timestep/decimation differs from common profile')
    for key in ('robot', 'normalization'):
        if plain(current[key]) != plain(saved_env[key]):
            raise ValueError(f'Native {key} input configuration differs from training snapshot')
    for key in ('enable_height_scan', 'critic_only'):
        if current['scene']['height_scanner'].get(key) != saved_env['scene']['height_scanner'].get(key):
            raise ValueError('Height scan observation contract changed')
    for key in ('policy', 'empirical_normalization', 'runner_class_name'):
        if plain(agent.to_dict()[key]) != plain(saved_agent[key]):
            raise ValueError(f'Native agent {key} differs from training snapshot')
    cfg.device = args.device
    agent.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.scene.seed = p['manifest_seed']
    cfg.scene.max_init_terrain_level = 0
    cfg.scene.max_episode_length_s = 60
    cfg.scene.terrain_type = 'generator'
    cfg.scene.terrain_generator = make_plane_recovery_terrain_cfg(tuple(p['slopes_deg']))
    cfg.scene.terrain_generator.size = tuple(p['physics']['tile_size_m'])
    cfg.scene.terrain_generator.seed = p['manifest_seed']
    cfg.noise.add_noise = False
    # Disable lag sampling while preserving native action history for action_rate_l2.
    cfg.domain_rand.action_delay.enable = False
    for name in list(vars(cfg.domain_rand.events)):
        if not name.startswith('_'):
            setattr(cfg.domain_rand.events, name, None)
    cfg.commands.heading_command = False
    cfg.commands.rel_heading_envs = 0
    cfg.commands.resampling_time_range = (1e9, 1e9)
    cfg.commands.debug_vis = False
    cfg.scene.height_scanner.drift_range = (0, 0)
    if hasattr(cfg, 'debug_save_num'):
        cfg.debug_save_num = 0
    if method.startswith('context'):
        cfg.plane_recovery.slopes_degrees = tuple(p['slopes_deg'])
        cfg.push_curriculum.enable_push_curriculum = False
        # Preserve context/reward/certificate cadence, including reward-off certificate.
    metrics_cfg = p['metrics']

    class FixedCommand(UniformVelocityCommand):
        def _resample_command(self, env_ids):
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            self.vel_command_b[ids] = self._env.eval_commands[ids]
            self.is_heading_env[ids] = False
            self.is_standing_env[ids] = torch.linalg.vector_norm(self.vel_command_b[ids, :2], dim=-1) < 1e-8

        def _update_command(self):
            self._resample_command(torch.arange(self.num_envs, device=self.device))

    class EvaluationEnv(native):
        def _create_certificate_evaluator(self):
            assert_native_inputs(self.cfg, identity)
            return super()._create_certificate_evaluator()

        def __init__(self, cfg, headless):
            self.eval_commands = torch.zeros((cfg.scene.num_envs, 3), device=cfg.device)
            self.eval_plans = None
            self.eval_capture = None
            self.eval_history_calls = 0
            self.eval_slopes = torch.zeros(cfg.scene.num_envs, device=cfg.device)
            super().__init__(cfg, headless)
            # Curriculum=True is needed for deterministic column construction only.
            self.cfg.scene.terrain_generator.curriculum = False
            self.scene.terrain.cfg.terrain_generator.curriculum = False
            self.eval_mesh = self._read_mesh()
            self.eval_origin_table = self.scene.terrain.terrain_origins.clone()
            self.eval_masses = self.robot.root_physx_view.get_masses().to(self.device)
            self.eval_foot_ids, names = self.contact_sensor.find_bodies(metrics_cfg['feet'], preserve_order=True)
            if names != metrics_cfg['feet']:
                raise ValueError('Foot names/order mismatch')
            self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
            material = self.robot.root_physx_view.get_material_properties().clone()
            material[..., 0] = p['physics']['material']['static_friction']
            material[..., 1] = p['physics']['material']['dynamic_friction']
            material[..., 2] = p['physics']['material']['restitution']
            self.robot.root_physx_view.set_material_properties(material, torch.arange(self.num_envs, device='cpu'))
            self.eval_slopes[:] = torch.tensor(p['slopes_deg'], device=self.device)[self.scene.terrain.terrain_types]
            self.assert_terrain()
            if abs(self.step_dt - 1 / p['physics']['policy_hz']) > 1e-8:
                raise ValueError('Trace/policy frequency is not 50 Hz')

        def _read_mesh(self):
            # Read the actual USD collision mesh after TerrainImporter placement.
            from pxr import UsdGeom
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if is_common(p) and UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
                raise ValueError('Common task measurement contract requires world +Z up')
            meshes = []
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith('/World/ground') and prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
                    points = np.array([transform.Transform(x) for x in mesh.GetPointsAttr().Get()], dtype=float)
                    meshes.append(points)
            if not meshes:
                raise ValueError('No actual terrain USD mesh found')
            return np.concatenate(meshes)

        def assert_terrain(self):
            origins = self.scene.env_origins
            if not torch.allclose(origins, self.scene.terrain.env_origins):
                raise ValueError('Scene/terrain origins diverged')
            for i in range(self.num_envs):
                origin = origins[i].cpu().numpy()
                points = self.eval_mesh
                # Corners of this 64 m tile must satisfy its signed plane equation.
                corners = points[(np.abs(np.abs(points[:, 0] - origin[0]) - 32) < 1e-3)
                                 & (np.abs(np.abs(points[:, 1] - origin[1]) - 32) < 1e-3)]
                a = float(self.eval_slopes[i])
                residual = corners[:, 2] - origin[2] - math.tan(math.radians(a)) * (corners[:, 0] - origin[0])
                # Adjacent tile seam may contain corners at different heights; require four coplanar vertices.
                if np.count_nonzero(np.abs(residual) < 2e-3) < 4:
                    raise ValueError(f'Mesh/origin/signed slope mismatch for env {i}')
            normal, point, valid = self.get_recovery_plane_geometry()
            if not bool(valid.all()) or not torch.allclose(point, origins):
                raise ValueError('Plane metadata/origin mismatch')
            measured = torch.rad2deg(torch.atan2(-normal[:, 0], normal[:, 2]))
            if not torch.allclose(measured, self.eval_slopes, atol=1e-4):
                raise ValueError('Certificate normal/signed slope mismatch')
            if self.cfg.scene.terrain_generator.curriculum:
                raise ValueError('Curriculum must be disabled after mesh generation')

        def _create_command_generator(self, command_cfg):
            return FixedCommand(command_cfg, self)

        def update_terrain_levels(self, env_ids):
            return {}  # Evaluation-only, including the constructor's first reset.

        def get_recovery_plane_geometry(self):
            a = torch.deg2rad(self.eval_slopes)
            return torch.stack((-torch.sin(a), torch.zeros_like(a), torch.cos(a)), -1), self.scene.env_origins.clone(), torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        def compute_observations(self):
            self.eval_history_calls += 1
            return super().compute_observations()

        def reset(self, env_ids):
            super().reset(env_ids)
            if self.eval_plans is None or len(env_ids) == 0:
                return
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            state = self.robot.data.default_root_state[ids].clone()
            pos_scale, vel_joint = [], []
            for j, i in enumerate(ids.tolist()):
                plan = self.eval_plans[i]
                pose, vel = plan['initial_pose_parameters'], plan['initial_velocity_parameters']
                state[j, :3] += self.scene.env_origins[i]
                state[j, :3] += torch.tensor([pose['x'], pose['y'], pose['z'] + math.tan(math.radians(plan['slope_deg'])) * pose['x']], device=self.device)
                angles = [torch.tensor([pose[k]], device=self.device) for k in ('roll', 'pitch', 'yaw')]
                state[j, 3:7] = quat_from_euler_xyz(*angles)[0]
                state[j, 7:13] += torch.tensor([vel[k] for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')], device=self.device)
                pos_scale.append(plan['initial_joint_parameters']['position_scale'])
                vel_joint.append(plan['initial_joint_parameters']['velocity'])
            joint_pos = self.robot.data.default_joint_pos[ids] * torch.tensor(pos_scale, device=self.device)
            limits = self.robot.data.soft_joint_pos_limits[ids]
            joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])  # same reset_joints_by_scale contract
            joint_vel = torch.tensor(vel_joint, device=self.device)
            self.robot.write_root_state_to_sim(state, env_ids=ids)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
            self.command_generator._resample_command(ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.assert_terrain()

        def begin_batch(self, plans):
            self.eval_plans = plans
            ids = torch.arange(self.num_envs, device=self.device)
            types = torch.tensor([p['slopes_deg'].index(r['slope_deg']) for r in plans], device=self.device)
            self.scene.terrain.terrain_levels[:] = 0
            self.scene.terrain.terrain_types[:] = types
            origins = self.eval_origin_table[0, types]
            self.scene.terrain.env_origins[:] = origins
            self.scene.env_origins[:] = origins
            if not torch.allclose(self.scene.terrain.env_origins, self.scene.env_origins):
                raise ValueError('Scene/terrain origins diverged after fixed-slope selection')
            self.eval_slopes[:] = torch.tensor([r['slope_deg'] for r in plans], device=self.device)
            self.eval_commands[:] = torch.tensor([[r['command_vx'], r['command_vy'], r['command_yaw']] for r in plans], device=self.device)
            self.reset(ids)
            before = self.eval_history_calls
            initial = self.get_observations()
            if self.eval_history_calls != before + 1:
                raise ValueError('Initial observations must advance history once')
            return initial[0], self.extras

        def check_reset(self):
            done, timeout = super().check_reset()
            # Native check uses the same torso contact threshold; assert shared configuration.
            if hasattr(self, 'eval_masses') and self.eval_plans is not None:
                self.eval_capture = self.physical_snapshot(done & ~timeout, timeout)
            return done, timeout

        def physical_snapshot(self, fell=None, timeout=None):
            data = self.robot.data
            q = yaw_quat(data.root_quat_w)
            weighted = (self.eval_masses.unsqueeze(-1) * data.body_com_vel_w[..., :3]).sum(1) / self.eval_masses.sum(1, keepdim=True)
            com = quat_apply_inverse(q, weighted)
            root = quat_apply_inverse(q, data.root_lin_vel_w)
            roll, pitch, yaw = euler_xyz_from_quat(data.root_quat_w)
            wrap = lambda x: torch.atan2(torch.sin(x), torch.cos(x))
            forces = torch.linalg.vector_norm(self.contact_sensor.data.net_forces_w[:, self.eval_foot_ids], dim=-1)
            relative = data.root_pos_w - self.scene.env_origins
            outside = relative[:, :2].abs().amax(-1) > metrics_cfg['test_area_half_extent_m']
            values = {'root_velocity': root, 'root_velocity_world': data.root_vel_w,
                      'com_velocity': com, 'roll_pitch': torch.stack((wrap(roll), wrap(pitch)), -1), 'yaw': wrap(yaw),
                      'forces': forces, 'root_position': relative,
                      'fell': fell if fell is not None else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                      'timeout': timeout if timeout is not None else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                      'out_of_test_area': outside}
            if p['metrics']['version'] == COMMON_METRICS_VERSION:
                # This plane is checked against the actual USD tile in assert_terrain.
                # All positions below use the same env-origin-relative frame.
                # Read data tensors only: never advance observations/actor history.
                alpha = torch.deg2rad(self.eval_slopes)
                plane_normal = torch.stack((-torch.sin(alpha), torch.zeros_like(alpha), torch.cos(alpha)), -1)
                clearance = relative[:, 2] - torch.tan(alpha) * relative[:, 0]
                values.update(root_quaternion_wxyz=data.root_quat_w,
                              angular_velocity_world_z=data.root_vel_w[:, 5],
                              command=self.command_generator.command.clone(),
                              root_clearance_m=clearance,
                              local_plane_normal=plane_normal,
                              local_plane_point=torch.zeros_like(relative))
                physical = ('com_velocity', 'root_velocity', 'root_velocity_world',
                            'root_quaternion_wxyz', 'command', 'root_position', 'forces',
                            'angular_velocity_world_z', 'root_clearance_m', 'local_plane_normal')
                valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                for key in physical:
                    valid &= torch.isfinite(values[key]).reshape(self.num_envs, -1).all(-1)
                values['data_valid'] = valid
            cpu = {k: v.detach().cpu().numpy().copy() for k, v in values.items()}
            return [{k: (v[i].tolist() if v[i].ndim else v[i].item()) for k, v in cpu.items()} for i in range(self.num_envs)]

        def plane_diagnostics(self, i):
            if not method.startswith('context'):
                return None
            valid = bool(self.current_certificate_valid[i])
            standing = bool(self._v1_intentional_not_applicable[i])
            diagnostic = self._v1_context_diagnostics[i]
            reason = 'standing' if standing else diagnostic['category'] if not valid else ''
            return {'context_valid': valid, 'standing_na': standing, 'invalid_reason': reason,
                    'N': int(self.current_n_min[i]) if valid else None,
                    'margin': float(self.current_margin[i]) if valid else None,
                    'query_id': diagnostic['query_id'],
                    'query_event': bool(self._context_refresh_mask[i]) and diagnostic['query_id'] is not None,
                    'query_failed': diagnostic['failed'], 'query_category': diagnostic['category'],
                    'query_status': diagnostic.get('status'), 'query_message': diagnostic.get('message'),
                    **{out: float(getattr(self, attr)[i]) for out, attr in
                       [('reward_progress', '_v1_event_progress'), ('reward_step_cost', '_v1_event_step_cost'),
                        ('reward_td5', '_v1_event_td5'), ('reward_total', '_v1_event_total')]}}

    env = EvaluationEnv(cfg, args.headless)
    if cfg.domain_rand.action_delay.enable is not False:
        raise ValueError('Evaluation requires action_delay.enable to be False.')
    if env.action_buffer._circular_buffer.max_length < 2:
        raise ValueError('Evaluation requires at least two action-buffer frames because '
                         'action_rate_l2 uses current and previous actions.')
    if is_common(p) and env.robot.body_names[0] != 'pelvis':
        raise ValueError('Common orientation requires the G1 root/pelvis with asset +Z up')
    if metrics_cfg['termination_force_n'] != 1.0:
        raise ValueError('Native termination threshold is 1 N; changing it requires a new adapter')
    if cfg.robot.terminate_contacts_body_names != metrics_cfg['termination_bodies']:
        raise ValueError('Native termination bodies differ from common judge')
    runner_cls = DWAQOnPolicyRunner if method == 'dwaq' else OnPolicyRunner
    runner = runner_cls(env, agent.to_dict(), log_dir=None, device=args.device)
    payload = torch.load(snapshot, map_location=args.device, weights_only=False)
    runner.alg.policy.load_state_dict(payload['model_state_dict'], strict=True)
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(payload['obs_norm_state_dict'], strict=True)
        runner.privileged_obs_normalizer.load_state_dict(payload['privileged_obs_norm_state_dict'], strict=True)
    runner.eval_mode()
    for module in (runner.alg.policy, runner.obs_normalizer, runner.privileged_obs_normalizer):
        module.eval()
        module.requires_grad_(False)
    before_weights = {k: v.detach().cpu().clone() for k, v in runner.alg.policy.state_dict().items()}
    if method == 'dwaq':
        def policy(obs, extra):
            # Native DWAQ act_inference samples the VAE even in eval mode. Isolate
            # each trial/step RNG so resume and batch size cannot change its draws.
            actions = []
            devices = [torch.device(args.device).index or 0] if args.device.startswith('cuda') else []
            for i, plan in enumerate(env.eval_plans):
                seed = int(digest([plan['reset_seed'], int(env.episode_length_buf[i]), 'native_dwaq_v1'])[:15], 16)
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    actions.append(runner.alg.policy.act_inference(obs[i:i + 1], extra['observations']['obs_hist'][i:i + 1]))
            return torch.cat(actions, 0)
    else:
        native_policy = runner.get_inference_policy(device=args.device)
        policy = lambda obs, extra: native_policy(obs)
    env_cfg = plain(cfg.to_dict())
    # Persist logical content locations, independent of checkout/output root names.
    for role, (section, field) in RESOURCE_FIELDS.items():
        if role in native_paths:
            env_cfg[section][field] = identity['resources'][role]['snapshot']
    if 'estimator' in native_paths:
        env_cfg['estimator_checkpoint_path'] = identity['resources']['estimator']['snapshot']
    # Serialize actual physical quantities separately from native inference configuration.
    view = env.robot.root_physx_view
    asset_path = Path(cfg.scene.robot.spawn.usd_path).resolve(strict=True)
    asset_config = plain(cfg.scene.robot.to_dict())
    asset_config['spawn']['usd_path'] = 'sha256:' + sha256(asset_path)
    actual = {'robot_asset': asset_config, 'masses': view.get_masses()[0].tolist(),
              'inertias': view.get_inertias()[0].tolist(), 'materials': view.get_material_properties()[0].tolist(),
              'joint_stiffness': env.robot.data.joint_stiffness[0].tolist(),
              'joint_damping': env.robot.data.joint_damping[0].tolist(), 'joint_names': env.robot.joint_names,
              'physics_dt': env.physics_dt, 'step_dt': env.step_dt,
              'ground_material': plain(env.scene.terrain.cfg.physics_material.to_dict())}
    effective = {'environment': env_cfg, 'actual_physics': actual, 'actual_physics_hash': digest(actual),
                 'native_inputs': {k: sha256(v) for k, v in native_paths.items()},
                 'native_configuration_sha256': identity['native_configuration_sha256'],
                 'terrain_mesh_sha256': digest(env.eval_mesh.tolist()), 'terrain_origins': env.eval_origin_table.cpu().tolist(),
                 'slopes_deg': p['slopes_deg'], 'native_inference': identity['runner_type'],
                 'normalization': saved_env['normalization']}
    if is_common(p):
        effective['common_measurement_contract'] = {
            'body': env.robot.body_names[0], 'up_axis_body': '+Z', 'quaternion_order': 'wxyz',
            'quaternion_transform': 'body_to_world', 'velocity': 'mass_weighted_body_com_vel_w_to_current_yaw_heading',
            'angular_velocity': 'root_vel_w[5] world-Z, not Euler yaw derivative',
            'clearance': 'vertical_above_USD_verified_local_plane', 'history_reads_added': 0}
    return env, runner, policy, before_weights, effective


def execute_run(args):
    import numpy as np
    import torch
    root = Path(args.output_root).resolve()
    development = getattr(args, 'command', None) == 'run-development'
    if development:
        requested = load_yaml(args.protocol)
        if not is_common(requested) or requested['protocol_version'] != '2.0-dev':
            raise ValueError('Development validation requires common_task_window_v1 / 2.0-dev')
        if root.is_relative_to((LAB / 'experiments/g1_recovery_eval').resolve()):
            raise ValueError('Development validation cannot write the protected historical root')
    if not (root / 'prepared.json').exists():
        prepare(root, args.protocol)
    p, manifest, prepared = load_prepared(root)
    if development and digest(p) != digest(requested):
        raise ValueError('Development CLI protocol differs from prepared protocol')
    full_manifest = manifest
    identity = inspect_checkpoint(args.task, args.checkpoint, args.model_alias, args.checkpoint_stage,
                                  args.estimator_checkpoint, args.identity_manifest)
    if development:
        manifest, development_fields = select_development_assignment(p, args.task, identity['method'],
            args.development_manifest, args.development_manifest_sha256)
        if args.trial_limit:
            raise ValueError('Development assignments cannot be truncated or resampled')
        identity.update(development_fields)
    snapshot = snapshot_checkpoint(root, identity)
    register_model(root, identity)
    identity.update(prepared)
    if development:
        identity['manifest_hash'] = digest(manifest)
        identity['subset'] = 'registered_development_8'
    elif args.trial_limit:
        if args.trial_limit < 1:
            raise ValueError('--trial_limit must be positive')
        # Fixed prefix of each experiment includes E0 and E1 even in a small acceptance run.
        manifest = [r for e in ('E0', 'E1') for r in [x for x in manifest if x['experiment'] == e][:args.trial_limit]]
        identity['manifest_hash'] = digest(manifest)
        identity['subset'] = f'first_{args.trial_limit}_per_experiment'
    else:
        identity['subset'] = 'lite_full'
    if p['protocol_version'] == '1.0':
        gate_path = root / 'detector_validation.json'
        if not gate_path.exists():
            raise ValueError('Frozen protocol requires manually approved detector validation evidence')
        gate = read_json(gate_path)
        from g1_recovery_report import validate_detector_gate
        validate_detector_gate(root, p, full_manifest, prepared, identity, gate)
    identity['trials'] = len(manifest)
    identity['realized_environment_schema_version'] = 1
    key = evaluation_key(identity)
    with lock(root / '.run-selection.lock'):
        attempts = sorted((root / 'runs').glob(key + '-attempt-*'))
        if not args.new_attempt:
            for old in attempts:
                if (old / 'completion.json').exists():
                    RunStore(old).validate()
                    from g1_recovery_report import build_report
                    build_report(root)
                    return {'status': 'ALREADY_COMPLETE', 'evaluation_id': old.name}
            run = attempts[-1] if attempts else root / 'runs' / (key + '-attempt-0001')
        else:
            run = root / 'runs' / (key + f'-attempt-{len(attempts) + 1:04d}')
        run.mkdir(parents=True, exist_ok=True)
    identity['attempt_index'] = int(run.name.rsplit('-', 1)[1])
    with lock(run / '.run.lock'):
        store = RunStore(run)
        if (run / 'completion.json').exists():
            store.validate()
            return {'status': 'ALREADY_COMPLETE', 'evaluation_id': run.name}
        if not (run / 'identity.json').exists():
            write_json(run / 'identity.json', identity)
            atomic_write(run / 'protocol_snapshot.yaml', (root / 'protocol.yaml').read_bytes())
            judge_file = 'common_detector_config.yaml' if is_common(p) else 'metrics_reference.yaml'
            atomic_write(run / judge_file, (root / judge_file).read_bytes())
            atomic_write(run / 'manifest_snapshot.jsonl', jsonl(manifest))
            if development:
                if sha256(args.development_manifest) != identity['development_manifest_sha256']:
                    raise ValueError('Development manifest changed before snapshot')
                atomic_write(run / 'development_manifest_snapshot.json', Path(args.development_manifest).read_bytes())
            for role, resource in identity['resources'].items():
                if role != 'estimator':
                    atomic_write(run / resource['snapshot'], (snapshot.parent / resource['snapshot']).read_bytes())
            atomic_write(run / 'native_configuration.json', (snapshot.parent / 'native_configuration.json').read_bytes())
        else:
            store.validate(require_complete=False)
        committed = {r['trial_id'] for r in store.records()}
        remaining = [r for r in manifest if r['trial_id'] not in committed]
        if not remaining:
            summary = summarize(store.records(), manifest)
            store.complete(summary)
            register_result(root, run)
            from g1_recovery_report import build_report
            build_report(root)
            return {'status': 'COMPLETE', 'evaluation_id': run.name}
        handler = logging.FileHandler(run / 'run.log')
        logger = logging.getLogger('g1_recovery_eval')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        app = None
        try:
            # This is the only simulator launch in all four CLI commands.
            os.chdir(LAB)
            sys.path.insert(0, str(LAB))
            sys.path.insert(0, str(LAB / 'rsl_rl'))
            from isaaclab.app import AppLauncher
            launcher = AppLauncher(headless=args.headless, device=args.device)
            app = launcher.app
            env, runner, policy, weights, effective = make_evaluation_environment(args, p, identity, snapshot)
            effective['environment_versions'] = runtime_environment_versions()
            effective['realized_slopes_deg'] = measure_realized_slopes(env.eval_mesh, effective['terrain_origins'], effective['slopes_deg'])
            effective['terrain_source_sha256'] = {k: v for k, v in identity['evaluation_runtime_sources'].items()
                if k in TERRAIN_SOURCES}
            effective['realized_environment_hash'] = digest(realized_environment_payload(effective))
            effective_path = run / 'effective_env_config.yaml'
            if effective_path.exists() and load_yaml(effective_path) != effective:
                raise ValueError('Effective physical/native configuration changed across resume')
            import yaml
            atomic_write(effective_path, yaml.safe_dump(effective, sort_keys=False))
            identity['actual_physics_hash'] = effective['actual_physics_hash']
            identity['realized_environment_hash'] = effective['realized_environment_hash']
            identity['software'].update({k: effective['environment_versions'][k] for k in ('IsaacLab', 'IsaacSim')})
            identity['hardware']['GPU'] = torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else 'CPU'
            write_json(run / 'identity.json', identity)
            nodes = {} if is_common(p) else read_json(root / 'metrics_nodes.json')
            with torch.inference_mode():
                for offset in range(0, len(remaining), args.num_envs):
                    selected = remaining[offset:offset + args.num_envs]
                    plans = selected + [selected[-1]] * (args.num_envs - len(selected))
                    obs, extra = env.begin_batch(plans)
                    machines = [make_trial_machine(r, p, nodes.get(f"{r['slope_deg']}:{r['command_name']}")) for r in selected]
                    frames = env.physical_snapshot()
                    for i, machine in enumerate(machines):
                        frame = {**frames[i], 'time': 0, 'plane': env.plane_diagnostics(i)}
                        machine.feed(frame)
                    step = 0
                    while any(not m.status for m in machines):
                        before_history = env.eval_history_calls
                        actions = policy(obs, extra)
                        if not torch.isfinite(actions).all():
                            raise ValueError('Non-finite actor action')
                        obs, _, dones, extra = env.step(actions)
                        if env.eval_history_calls != before_history + 1:
                            raise ValueError('Native history advanced more than once per policy step')
                        if not torch.allclose(env.command_generator.command, env.eval_commands):
                            raise ValueError('Fixed command changed during step')
                        step += 1
                        if env.eval_capture is None:
                            raise ValueError('Missing pre-reset physical frame')
                        for i, machine in enumerate(machines):
                            if machine.status:
                                continue
                            frame = {**env.eval_capture[i], 'time': step * env.step_dt,
                                     'plane': env.plane_diagnostics(i) if not bool(dones[i]) else None}
                            if machine.feed(frame):
                                apply_trial_intervention(env, machine, i, frame, identity)
                            if machine.status:
                                store.save_trial(machine.result(), machine.trace(), machine.all_events())
                                logger.info('%s %s', machine.plan['trial_id'], machine.status)
                        if step * env.step_dt > 30:
                            raise ValueError('Trial state machine exceeded bounded observation duration')
                if any(not torch.equal(v.cpu(), weights[k]) for k, v in runner.alg.policy.state_dict().items()):
                    raise ValueError('Policy weights changed during evaluation')
            summary = summarize(store.records(), manifest)
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
            store.complete(summary)
            register_result(root, run)
        except Exception as exc:
            if 'machines' in locals():
                saved_ids = {r['trial_id'] for r in store.records()}
                for machine in machines:
                    if machine.plan['trial_id'] not in saved_ids and machine.frames:
                        machine.finish('EVALUATION_ERROR')
                        machine.events.append({'event': 'execution_error', 'time': machine.t, 'error': str(exc)})
                        try:
                            store.save_trial(machine.result(include_metrics=False), machine.trace(), machine.all_events())
                        except Exception as cleanup_exc:
                            logger.exception('Could not preserve failed trial: %s', cleanup_exc)
            logger.exception('Evaluation failed: %s', exc)
            write_json(run / 'failure.json', {'status': 'INVALID', 'error': str(exc), 'completed_trials': len(store.records())})
            write_json(run / 'summary.json', {**summarize(store.records(), manifest), 'status': 'INVALID'})
            if args.update_report:
                from g1_recovery_report import build_report
                build_report(root)
            raise
        finally:
            handler.close()
            logger.removeHandler(handler)
            if app is not None:
                app.close()
        from g1_recovery_report import build_report
        build_report(root)
        if args.wandb:
            from g1_recovery_report import upload_wandb
            upload_wandb(run, p)
        return {'status': 'COMPLETE', 'evaluation_id': run.name}


def reanalyze(args):
    """Create an immutable offline analysis; never rewrite original trials/metrics."""
    from g1_recovery_report import analyze_traces
    return analyze_traces(Path(args.source), Path(args.protocol))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'report', 'import-results', 'register', 'reanalyze', 'detector-validation', 'diagnose-common', 'run-development'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--output_root', default=None)
        if name in ('prepare', 'run', 'reanalyze', 'diagnose-common', 'run-development'):
            cmd.add_argument('--protocol', default=str(COMMON_PROTOCOL if name in ('diagnose-common', 'run-development') else DEFAULT_PROTOCOL))
        if name in ('run', 'run-development', 'register'):
            cmd.add_argument('--task', required=True, choices=TASKS.values())
            cmd.add_argument('--checkpoint', required=True)
            cmd.add_argument('--model_alias', required=True)
            cmd.add_argument('--checkpoint_stage', choices=('intermediate', 'final', 'unknown'), default='unknown')
            cmd.add_argument('--estimator_checkpoint')
            cmd.add_argument('--identity_manifest', help='SHA-bound legacy task confirmation, resource mapping and training provenance YAML')
        if name in ('run', 'run-development'):
            cmd.add_argument('--suite', choices=['lite'], default='lite')
            cmd.add_argument('--num_envs', type=int, default=1 if name == 'run-development' else 32)
            cmd.add_argument('--headless', action='store_true')
            cmd.add_argument('--device', default='cuda:0')
            cmd.add_argument('--trial_limit', type=int, help='Development prefix PER experiment; distinct manifest hash')
            cmd.add_argument('--new_attempt', action='store_true')
            cmd.add_argument('--wandb', action='store_true')
        if name == 'run-development':
            cmd.add_argument('--development_manifest', default=str(DEVELOPMENT_MANIFEST))
            cmd.add_argument('--development_manifest_sha256', default=DEVELOPMENT_MANIFEST_SHA256)
        if name in ('run', 'run-development', 'import-results'):
            cmd.add_argument('--update_report', action='store_true')
        if name in ('import-results', 'reanalyze', 'diagnose-common'):
            cmd.add_argument('--source', required=True)
        if name == 'report':
            cmd.add_argument('--format', nargs='+', choices=['md', 'docx'], default=['md', 'docx'])
    args = parser.parse_args(argv)
    if args.output_root is None:
        common = hasattr(args, 'protocol') and is_common(load_yaml(args.protocol))
        args.output_root = str(LAB / 'experiments' / ('g1_recovery_eval_v2' if common else 'g1_recovery_eval'))
    if args.command == 'prepare':
        result = prepare(args.output_root, args.protocol)
    elif args.command == 'register':
        identity = inspect_checkpoint(args.task, args.checkpoint, args.model_alias, args.checkpoint_stage,
                                      args.estimator_checkpoint, args.identity_manifest)
        snapshot_checkpoint(args.output_root, identity)
        register_model(args.output_root, identity)
        result = identity
    elif args.command in ('run', 'run-development'):
        if args.num_envs < 1:
            parser.error('--num_envs must be positive')
        result = execute_run(args)
    elif args.command == 'report':
        from g1_recovery_report import build_report
        result = build_report(args.output_root, args.format)
    elif args.command == 'reanalyze':
        result = reanalyze(args)
    elif args.command == 'detector-validation':
        from g1_recovery_report import detector_validation_report
        result = detector_validation_report(args.output_root)
    elif args.command == 'diagnose-common':
        from g1_common_task_diagnostics import offline_acceptance
        result = offline_acceptance(args.source, args.protocol, args.output_root)
    else:
        if not (Path(args.output_root) / 'prepared.json').exists():
            prepare(args.output_root)
        result = import_results(args.source, args.output_root)
        if args.update_report:
            from g1_recovery_report import build_report
            build_report(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
