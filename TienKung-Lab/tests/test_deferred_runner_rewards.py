from __future__ import annotations

from types import SimpleNamespace

import torch

from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.storage import RolloutStorage


class _DeferredEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.num_actions = 1
        self.max_episode_length = 100
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.device = "cpu"
        self.unwrapped = self
        self._step = 0
        self.corrections = torch.tensor(
            [[0.0, 0.25], [-0.5, 0.0], [0.75, -0.25]], dtype=torch.float32
        )

    def get_observations(self):
        obs = torch.zeros(2, 1)
        return obs, {"observations": {"critic": obs.clone()}}

    def begin_deferred_reward_rollout(self, num_steps: int) -> bool:
        assert num_steps == 3
        self._step = 0
        return True

    def step(self, _actions):
        self._step += 1
        obs = torch.full((2, 1), float(self._step))
        rewards = torch.tensor([float(self._step), 10.0 + self._step])
        dones = torch.zeros(2, dtype=torch.bool)
        infos = {"observations": {"critic": obs.clone()}}
        return obs, rewards, dones, infos

    def resolve_deferred_reward_rollout(self) -> dict:
        assert self._step == 3
        return {"reward_corrections": self.corrections, "episode_rewards": [], "log": {}}


class _Algorithm:
    def __init__(self, expected: torch.Tensor) -> None:
        self.rnd = None
        self.storage = RolloutStorage(
            "rl", 2, 3, [1], [1], [1], device="cpu"
        )
        self.expected = expected
        self.compute_returns_called = False

    def act(self, _obs, _critic_obs):
        return torch.zeros(2, 1)

    def process_env_step(self, rewards, _dones, _infos):
        self.storage.rewards[self.storage.step, :, 0] = rewards
        self.storage.step += 1

    def compute_returns(self, _critic_obs):
        assert torch.equal(self.storage.rewards[..., 0], self.expected)
        self.compute_returns_called = True

    def update(self):
        return {"value_function": 0.0, "surrogate": 0.0, "entropy": 0.0}


def test_runner_backfills_before_compute_returns() -> None:
    env = _DeferredEnv()
    base = torch.tensor([[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.env = env
    runner.alg = _Algorithm(base + env.corrections)
    runner.device = "cpu"
    runner.training_type = "rl"
    runner.privileged_obs_type = "critic"
    runner.obs_normalizer = torch.nn.Identity()
    runner.privileged_obs_normalizer = torch.nn.Identity()
    runner.num_steps_per_env = 3
    runner.log_dir = None
    runner.writer = None
    runner.disable_logs = True
    runner.is_distributed = False
    runner.current_learning_iteration = 0
    runner.save_interval = 100
    runner.train_mode = lambda: None
    runner.cfg = SimpleNamespace()

    runner.learn(1)

    assert runner.alg.compute_returns_called
