"""Exercise play's reset override without starting Isaac or importing its app."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_node(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('generated', [False, True])
def test_play_initial_and_later_reset_do_not_use_training_curriculum(generated):
    reset = load_node('legged_lab/envs/base/base_env.py', 'reset', {})
    calls = []
    noop = lambda *args, **kwargs: None

    class TrainingEnv:
        def __init__(self, cfg, headless):
            self.cfg = cfg
            # Matched constructors rebuild and enable the training generator.
            cfg.scene.terrain_generator = NS(curriculum=True)
            terrain = NS(terrain_levels=np.array([3])) if generated else NS()
            self.scene = NS(terrain=terrain, reset=lambda ids: calls.append(tuple(ids)), write_data_to_sim=noop)
            self.extras = {}
            self.event_manager = NS(available_modes=[])
            self.reward_manager = NS(reset=lambda ids: {})
            self.time_out_buf = np.zeros(1, dtype=bool)
            self.episode_length_buf = np.ones(1)
            for attr in ('command_generator', 'actor_obs_buffer', 'critic_obs_buffer', 'action_buffer'):
                setattr(self, attr, NS(reset=noop))
            self.sim = NS(forward=noop)
            self.reset([0])

        def update_terrain_levels(self, ids):
            raise AssertionError('Training curriculum must not run during play')

    TrainingEnv.reset = reset
    play_class = load_node('legged_lab/scripts/play.py', 'PlayEnv', {'env_class': TrainingEnv})
    env = play_class(NS(scene=NS(terrain_generator=None)), False)
    env.reset([0])
    assert calls == [(0,), (0,)]
    assert env.episode_length_buf[0] == 0
    assert env.extras['log'] == {}
    if generated:
        assert env.scene.terrain.terrain_levels.tolist() == [3]
    else:
        assert not hasattr(env.scene.terrain, 'terrain_levels')


def test_play_plain_floor_has_valid_geometry_without_tile_indices():
    class NativeEnv:
        def get_recovery_plane_geometry(self):
            return 'native geometry'

    cls = load_node('legged_lab/scripts/play.py', 'PlayEnv', {'env_class': NativeEnv, 'torch': torch})
    env = cls()
    env.cfg = NS(scene=NS(terrain_type='plane'))
    env.device = 'cpu'
    env.num_envs = 2
    env.scene = NS(env_origins=torch.tensor([[0., 0., 0.], [6., 0., 0.]]), terrain=NS())
    normal, point, valid = env.get_recovery_plane_geometry()
    assert normal.tolist() == [[0., 0., 1.], [0., 0., 1.]]
    assert valid.tolist() == [True, True]
    assert torch.equal(point, env.scene.env_origins)
    env.cfg.scene.terrain_type = 'generator'
    assert env.get_recovery_plane_geometry() == 'native geometry'


def test_play_per_step_terrain_log_handles_plane_and_preserves_generated_log():
    native_log = load_node('legged_lab/envs/g1/g1_plane_v1_matched_env.py', '_terrain_log', {'torch': torch})
    native = type('NativeEnv', (), {'_terrain_log': native_log})
    cls = load_node('legged_lab/scripts/play.py', 'PlayEnv', {'env_class': native, 'torch': torch})
    env = cls()
    env.cfg = NS(scene=NS(terrain_type='plane'))
    env.scene = NS(terrain=NS())
    assert env._terrain_log(0) == {'Play/flat_terrain': 1.0}
    env.cfg.scene.terrain_type = 'generator'
    env.scene.terrain = NS(terrain_levels=torch.tensor([0]), terrain_types=torch.tensor([0]))
    env._matched_slope_table = torch.tensor([[0.1]])
    assert env._terrain_log(0) == native_log(env, 0)
