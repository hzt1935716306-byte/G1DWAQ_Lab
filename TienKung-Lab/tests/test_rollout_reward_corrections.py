from __future__ import annotations

import pytest
import torch

from rsl_rl.storage import RolloutStorage


def _storage() -> RolloutStorage:
    storage = RolloutStorage(
        "rl",
        num_envs=3,
        num_transitions_per_env=2,
        obs_shape=[1],
        privileged_obs_shape=[1],
        actions_shape=[1],
        device="cpu",
    )
    storage.step = 2
    return storage


def test_reward_corrections_are_added_at_original_rollout_indices() -> None:
    storage = _storage()
    storage.rewards[..., 0] = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    correction = torch.tensor([[0.0, 0.25, 0.0], [-0.5, 0.0, 0.75]])

    storage.add_reward_corrections(correction)

    assert torch.equal(
        storage.rewards[..., 0],
        torch.tensor([[1.0, 2.25, 3.0], [3.5, 5.0, 6.75]]),
    )


def test_reward_corrections_require_a_complete_finite_rollout() -> None:
    storage = _storage()
    storage.step = 1
    with pytest.raises(RuntimeError, match="complete rollout"):
        storage.add_reward_corrections(torch.zeros(2, 3))

    storage.step = 2
    with pytest.raises(ValueError, match="shape"):
        storage.add_reward_corrections(torch.zeros(1, 3))
    invalid = torch.zeros(2, 3)
    invalid[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        storage.add_reward_corrections(invalid)
