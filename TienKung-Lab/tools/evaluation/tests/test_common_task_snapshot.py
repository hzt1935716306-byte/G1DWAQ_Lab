"""Exercise the real snapshot method on CPU tensors without importing Isaac."""
import ast
from types import SimpleNamespace as NS

import numpy as np
import torch

from g1_common_task_detector import METRICS_VERSION, CommonTaskMeasurements
from g1_recovery_protocol import LAB, EVALUATION_RUNTIME_SOURCES


def snapshot(version):
    tree = ast.parse((LAB / 'tools/evaluation/g1_recovery_eval.py').read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'physical_snapshot')
    namespace = dict(torch=torch, np=np, p={'metrics': {'version': version}},
                     COMMON_METRICS_VERSION=METRICS_VERSION,
                     metrics_cfg={'test_area_half_extent_m': 24},
                     yaw_quat=lambda q: q,
                     quat_apply_inverse=lambda q, v: v,
                     euler_xyz_from_quat=lambda q: (torch.zeros(len(q)),)*3)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'snapshot_contract', 'exec'), namespace)
    robot = NS(root_quat_w=torch.tensor([[1., 0, 0, 0]]),
               root_vel_w=torch.tensor([[.4, 0, 0, 0, 0, .12]]),
               root_lin_vel_w=torch.tensor([[.4, 0, 0]]),
               body_com_vel_w=torch.tensor([[[.3, 0, 0], [.5, 0, 0]]]),
               root_pos_w=torch.tensor([[11., 20, 3.8]]))
    env = NS(robot=NS(data=robot), eval_masses=torch.tensor([[1., 3.]]),
             contact_sensor=NS(data=NS(net_forces_w=torch.tensor([[[0., 0, 20], [0., 0, 30]]]))),
             eval_foot_ids=[0, 1], scene=NS(env_origins=torch.tensor([[10., 20, 3.]])),
             eval_slopes=torch.tensor([-10.]), num_envs=1, device='cpu', eval_history_calls=7,
             command_generator=NS(command=torch.tensor([[.4, 0, 0]])))
    result = namespace['physical_snapshot'](env)[0]
    assert env.eval_history_calls == 7
    return result


def test_common_snapshot_reads_weighted_gt_and_plane_without_history_advance():
    f = snapshot(METRICS_VERSION)
    f['time'] = 0.
    m = CommonTaskMeasurements.from_frame(f)
    np.testing.assert_allclose(m.com_velocity_xy, [.45, 0], atol=1e-7)
    np.testing.assert_allclose(m.root_velocity_xy, [.4, 0], atol=1e-7)
    np.testing.assert_allclose(m.root_clearance_m, .8-np.tan(np.deg2rad(-10)), atol=1e-6)
    np.testing.assert_allclose(m.angular_velocity_world_z, .12, atol=1e-7)
    assert m.data_valid
    assert 'local_plane_normal' in f and 'local_plane_point' in f


def test_legacy_snapshot_field_contract_unchanged():
    f = snapshot('practical_interval_confirm2_v1')
    assert set(f) == {'root_velocity', 'root_velocity_world', 'com_velocity', 'roll_pitch',
                      'yaw', 'forces', 'root_position', 'fell', 'timeout', 'out_of_test_area'}


def test_common_detector_is_bound_in_runtime_allowlist():
    assert 'tools/evaluation/g1_common_task_detector.py' in EVALUATION_RUNTIME_SOURCES
