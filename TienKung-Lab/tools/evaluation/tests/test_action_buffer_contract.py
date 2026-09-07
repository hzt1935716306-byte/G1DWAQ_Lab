"""Run evaluation/BaseEnv source with real Isaac Lab buffers on CPU, without Sim."""
import ast
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from g1_recovery_protocol import LAB


def function_node(path, name):
    tree = ast.parse((LAB / path).read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def execute(nodes, namespace):
    future = ast.parse('from __future__ import annotations').body
    module = ast.Module(body=future + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'action_buffer_contract', 'exec'), namespace)


def evaluation_config(max_delay=5):
    cfg = NS(domain_rand=NS(action_delay=NS(enable=True, params={'min_delay': 0, 'max_delay': max_delay})))
    original = copy.deepcopy(cfg.domain_rand.action_delay.params)
    node = function_node('tools/evaluation/g1_recovery_eval.py', 'make_evaluation_environment')
    overrides = [n for n in node.body if isinstance(n, ast.Assign)
                 and any(ast.unparse(t).startswith('cfg.domain_rand.action_delay') for t in n.targets)]
    assert overrides
    execute(overrides, {'cfg': cfg})
    assert cfg.domain_rand.action_delay.enable is False
    assert cfg.domain_rand.action_delay.params == original
    return cfg


@pytest.fixture
def delay_buffer(monkeypatch):
    # Import only the installed buffer sources: isaaclab.utils imports simulator modules.
    spec = importlib.util.find_spec('isaaclab')
    if spec is None:
        pytest.skip('Real buffer contract requires an installed Isaac Lab environment')
    directory = Path(next(iter(spec.submodule_search_locations))) / 'utils/buffers'
    package = ModuleType('_evaluation_test_buffers')
    package.__path__ = [str(directory)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name in ('circular_buffer', 'delay_buffer'):
        module_spec = importlib.util.spec_from_file_location(f'{package.__name__}.{name}', directory / f'{name}.py')
        module = importlib.util.module_from_spec(module_spec)
        monkeypatch.setitem(sys.modules, module_spec.name, module)
        module_spec.loader.exec_module(module)
    return module.DelayBuffer


def native_action_buffer(cfg, delay_buffer):
    env = NS(cfg=cfg, num_envs=2, num_actions=29, device='cpu')
    node = function_node('legged_lab/envs/base/base_env.py', 'init_buffers')
    start = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                 and any(ast.unparse(t) == 'self.action_buffer' for t in n.targets))
    end = next(i for i, n in enumerate(node.body) if isinstance(n, ast.If)
               and ast.unparse(n.test) == 'self.cfg.domain_rand.action_delay.enable')
    execute(node.body[start:end + 1], {'self': env, 'DelayBuffer': delay_buffer, 'torch': torch})
    return env


def validate_environment(env):
    node = function_node('tools/evaluation/g1_recovery_eval.py', 'make_evaluation_environment')
    construction = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                        and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'EvaluationEnv')
    # Execute the guards at their actual position, immediately after construction.
    guards = node.body[construction + 1:construction + 3]
    assert all(isinstance(n, ast.If) for n in guards)
    execute(guards, {'cfg': env.cfg, 'env': env})


def test_evaluation_disables_delay_without_overwriting_native_capacity():
    assert evaluation_config().domain_rand.action_delay.params['max_delay'] == 5


@pytest.mark.parametrize('max_delay', [1, 5])
def test_native_history_supports_reward_without_delaying_actions(delay_buffer, max_delay):
    env = native_action_buffer(evaluation_config(max_delay), delay_buffer)
    validate_environment(env)
    assert env.action_buffer._circular_buffer.max_length == max_delay + 1
    assert env.action_buffer._circular_buffer.buffer.shape == (2, max_delay + 1, 29)
    assert torch.count_nonzero(env.action_buffer.time_lags) == 0
    reward = function_node('legged_lab/mdp/rewards.py', 'action_rate_l2')
    namespace = {'torch': torch}
    execute([reward], namespace)
    previous = torch.zeros(2, 29)
    # Includes the first two actions and wraparound of the six-frame native buffer.
    for step in range(1, 9):
        current = torch.full((2, 29), float(step * step))
        assert torch.equal(env.action_buffer.compute(current), current)
        history = env.action_buffer._circular_buffer.buffer
        assert torch.equal(history[:, -1, :], current)
        assert torch.equal(history[:, -2, :], previous)
        torch.testing.assert_close(namespace['action_rate_l2'](env), ((current - previous) ** 2).sum(dim=1))
        previous = current


def test_evaluation_rejects_single_frame_buffer(delay_buffer):
    env = native_action_buffer(evaluation_config(0), delay_buffer)
    with pytest.raises(ValueError, match='Evaluation requires at least two action-buffer frames because '
                                        'action_rate_l2 uses current and previous actions.'):
        validate_environment(env)


def test_evaluation_rejects_enabled_delay(delay_buffer):
    env = native_action_buffer(evaluation_config(), delay_buffer)
    env.cfg.domain_rand.action_delay.enable = True
    with pytest.raises(ValueError, match='Evaluation requires action_delay.enable to be False'):
        validate_environment(env)


def test_capacity_and_applied_lag_are_independent(delay_buffer):
    cfg = evaluation_config()
    cfg.domain_rand.action_delay.enable = True
    cfg.domain_rand.action_delay.params['min_delay'] = 5
    env = native_action_buffer(cfg, delay_buffer)
    assert env.action_buffer._circular_buffer.max_length == 6
    assert env.action_buffer.time_lags.tolist() == [5, 5]
    for step in range(1, 7):
        current = torch.full((2, 29), float(step))
        torch.testing.assert_close(env.action_buffer.compute(current), torch.full((2, 29), float(max(0, step - 5))))
