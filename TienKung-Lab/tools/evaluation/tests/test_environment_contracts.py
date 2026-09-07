"""Exercise the actual evaluation subclass against CPU native stubs, no Isaac launch.

These test hooks/geometry contracts; real physics/contact behavior remains pending
simulation acceptance. AST extraction executes the original nested class body.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from g1_recovery_protocol import LAB


def extracted_class(name, namespace):
    tree = ast.parse((LAB / 'tools/evaluation/g1_recovery_eval.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, 'actual_evaluation_class', 'exec'), namespace)
    return namespace[name]


def env_class():
    class Native:
        def compute_observations(self):
            self.history += 1
            return self.history
        def check_reset(self):
            return torch.tensor([True]), torch.tensor([False])
        def reset(self, ids):
            self.pose = 'RESET_POSE'
    return extracted_class('EvaluationEnv', dict(native=Native, torch=torch, np=np, math=math,
                           p={'slopes_deg': [-10, 0, 10]}, metrics_cfg={'test_area_half_extent_m': 24}))


def test_history_one_append_and_terminal_capture_precedes_reset():
    cls = env_class(); env = cls.__new__(cls)
    env.eval_history_calls = 0; env.history = 0
    assert env.compute_observations() == 1
    assert env.eval_history_calls == env.history == 1
    env.eval_masses = torch.ones(1); env.eval_plans = [{}]; env.pose = 'FALLING_POSE'
    env.physical_snapshot = lambda fell, timeout: {'pose': env.pose, 'fell': bool(fell[0])}
    done, timeout = env.check_reset()
    env.eval_plans = None
    env.reset([0])
    assert env.pose == 'RESET_POSE'
    assert env.eval_capture == {'pose': 'FALLING_POSE', 'fell': True}
    assert env.history == 1  # metric capture does not call observations


@pytest.mark.parametrize('slope', [-10, 0, 10])
def test_mesh_origin_normal_consistency_and_curriculum_disabled(slope):
    cls = env_class(); env = cls.__new__(cls)
    env.num_envs = 1; env.device = 'cpu'; env.eval_slopes = torch.tensor([float(slope)])
    origins = torch.tensor([[4., 5., 2.]])
    env.scene = NS(env_origins=origins, terrain=NS(env_origins=origins))
    env.cfg = NS(scene=NS(terrain_generator=NS(curriculum=False)))
    env.eval_mesh = np.array([[x + 4, y + 5, 2 + math.tan(math.radians(slope)) * x]
                              for x, y in [(-32, -32), (32, -32), (32, 32), (-32, 32)]])
    env.assert_terrain()
    assert env.update_terrain_levels([0]) == {}
    np.testing.assert_allclose(env.scene.env_origins.numpy(), [[4, 5, 2]])
    env.eval_mesh[0, 2] += 1
    with pytest.raises(ValueError, match='Mesh/origin'):
        env.assert_terrain()
    env.eval_mesh[0, 2] -= 1
    env.cfg.scene.terrain_generator.curriculum = True
    with pytest.raises(ValueError, match='Curriculum'):
        env.assert_terrain()


def test_commands_are_reapplied_without_sampling():
    cls = extracted_class('FixedCommand', {'UniformVelocityCommand': object, 'torch': torch})
    cmd = cls(); cmd.device = 'cpu'; cmd.num_envs = 2
    cmd._env = NS(eval_commands=torch.tensor([[.4, 0, 0], [0, 0, 0]]))
    cmd.vel_command_b = torch.ones(2, 3)
    cmd.is_heading_env = torch.ones(2, dtype=torch.bool)
    cmd.is_standing_env = torch.zeros(2, dtype=torch.bool)
    cmd._update_command()
    assert torch.equal(cmd.vel_command_b, cmd._env.eval_commands)
    assert cmd.is_standing_env.tolist() == [False, True]
    assert not cmd.is_heading_env.any()
    cmd.vel_command_b[:] = 42
    cmd._resample_command([0, 1])
    assert torch.equal(cmd.vel_command_b, cmd._env.eval_commands)


@pytest.mark.parametrize('training_level', [0, 1, 2])
def test_fixed_plane_logging_does_not_require_training_curriculum_slopes(training_level):
    cls = env_class(); env = cls.__new__(cls)
    native_slopes = (-15., -10., -5., 0., 5., 10., 15.)
    env.cfg = NS(plane_recovery=NS(slopes_degrees=native_slopes))
    env.scene = NS(terrain=NS(terrain_types=torch.tensor([0, 0, 1, 2])),
                   env_origins=torch.zeros(4, 3))
    env.num_envs = 4; env.device = 'cpu'
    env.eval_slopes = torch.tensor([-10., -10., 0., 10.])
    env.eval_history_calls = 7
    log = env._terrain_log(training_level)
    assert log == {'EvaluationTerrain/curriculum_enabled': 0.,
                   'EvaluationTerrain/max_abs_slope_deg': 10.,
                   'EvaluationTerrain/P_slope_-10_deg': .5,
                   'EvaluationTerrain/P_slope_0_deg': .25,
                   'EvaluationTerrain/P_slope_10_deg': .25}
    assert env.cfg.plane_recovery.slopes_degrees == native_slopes
    assert env.eval_history_calls == 7
    normals, _, valid = env.get_recovery_plane_geometry()
    assert valid.all()
    torch.testing.assert_close(torch.rad2deg(torch.atan2(-normals[:, 0], normals[:, 2])), env.eval_slopes)
    # Exercise the production configuration overrides, not a copied fixture.
    tree = ast.parse((LAB / 'tools/evaluation/g1_recovery_eval.py').read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == 'make_evaluation_environment')
    context = next(n for n in function.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "method.startswith('context')")
    cfg = NS(plane_recovery=NS(slopes_degrees=native_slopes),
             push_curriculum=NS(enable_push_curriculum=True))
    exec(compile(ast.Module(body=[context], type_ignores=[]), 'actual_context_overrides', 'exec'),
         dict(cfg=cfg, method='context_reward', p={'slopes_deg': [-10, 0, 10]}))
    assert cfg.plane_recovery.slopes_degrees == native_slopes
    assert cfg.push_curriculum.enable_push_curriculum is False


def test_gt_metrics_are_not_actor_arguments():
    tree = ast.parse((LAB / 'tools/evaluation/g1_recovery_eval.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'act_inference']
    assert calls
    for call in calls:
        args = ast.unparse(call)
        assert 'com_velocity' not in args and 'physical' not in args and 'eval_capture' not in args
        assert 'obs' in args
