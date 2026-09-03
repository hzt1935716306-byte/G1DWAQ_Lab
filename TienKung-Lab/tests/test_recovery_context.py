"""Tensor-only tests for actor context normalization and strict warm start."""

from __future__ import annotations

import pytest
import torch

from legged_lab.recovery.checkpoint_migration import migrate_stage1a_model_state_dict
from legged_lab.recovery.recovery_context import normalize_recovery_context


def test_recovery_context_normalization_preserves_valid_margin() -> None:
    n_min = torch.tensor([0, 1, 3, 6, 6])
    margin = torch.tensor([0.475, -1.0, 1.9, -3.0, -3.0])
    valid = torch.tensor([True, True, True, True, False])

    context = normalize_recovery_context(n_min, margin, valid)

    torch.testing.assert_close(context[:, 0], torch.tensor([0.0, 1 / 6, 0.5, 1.0, 0.0]))
    torch.testing.assert_close(context[:, 1], torch.tensor([0.5, -0.5, 1.0, -1.0, 0.0]))
    torch.testing.assert_close(context[:, 2], torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]))


def _checkpoint_states():
    source = {
        "actor.0.weight": torch.randn(8, 960),
        "actor.0.bias": torch.randn(8),
        "actor.2.weight": torch.randn(4, 8),
        "critic.0.weight": torch.randn(8, 1010),
    }
    target = {
        "actor.0.weight": torch.randn(8, 963),
        "actor.0.bias": torch.randn(8),
        "actor.2.weight": torch.randn(4, 8),
        "critic.0.weight": torch.randn(8, 1010),
    }
    return source, target


def test_stage1a_migration_changes_only_three_actor_columns() -> None:
    source, target = _checkpoint_states()
    migrated = migrate_stage1a_model_state_dict(source, target)

    torch.testing.assert_close(migrated["actor.0.weight"][:, :960], source["actor.0.weight"])
    assert torch.count_nonzero(migrated["actor.0.weight"][:, 960:]).item() == 0
    for key in source.keys() - {"actor.0.weight"}:
        torch.testing.assert_close(migrated[key], source[key])

    observations = torch.randn(13, 960)
    old_output = observations @ source["actor.0.weight"].T + source["actor.0.bias"]
    context_observations = torch.cat((observations, torch.zeros(13, 3)), dim=-1)
    new_output = context_observations @ migrated["actor.0.weight"].T + migrated["actor.0.bias"]
    torch.testing.assert_close(new_output, old_output, atol=2.0e-5, rtol=2.0e-5)


def test_stage1a_migration_rejects_unexpected_mismatch() -> None:
    source, target = _checkpoint_states()
    source["critic.0.weight"] = torch.randn(8, 1009)
    with pytest.raises(RuntimeError, match="Unexpected Stage1A checkpoint shape mismatch"):
        migrate_stage1a_model_state_dict(source, target)
