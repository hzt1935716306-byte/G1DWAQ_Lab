"""Tensor-only tests for Plane V1 Actor-visible context projection."""

import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "legged_lab/recovery/context_projection.py"
SPEC = importlib.util.spec_from_file_location("context_projection_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

project_actor_recovery_context = MODULE.project_actor_recovery_context


def test_n_valid_projection_really_removes_margin_dimension() -> None:
    full = torch.tensor([[0.5, -0.3, 1.0]])
    projected = project_actor_recovery_context(full, ("n_norm", "valid"))

    assert projected.shape == (1, 2)
    torch.testing.assert_close(projected, torch.tensor([[0.5, 1.0]]))


def test_margin_valid_projection_really_removes_n_dimension() -> None:
    full = torch.tensor([[0.5, -0.3, 1.0]])
    projected = project_actor_recovery_context(full, ("margin_norm", "valid"))

    assert projected.shape == (1, 2)
    torch.testing.assert_close(projected, torch.tensor([[-0.3, 1.0]]))


def test_full_projection_preserves_existing_three_dimensional_context() -> None:
    full = torch.tensor([[0.5, -0.3, 1.0]])
    projected = project_actor_recovery_context(
        full, ("n_norm", "margin_norm", "valid")
    )

    assert projected.shape == (1, 3)
    torch.testing.assert_close(projected, full)


def test_projection_preserves_batch_dtype_and_device() -> None:
    full = torch.randn(3, 4, 3, dtype=torch.float64)
    projected = project_actor_recovery_context(full, ("n_norm", "valid"))

    assert projected.shape == (3, 4, 2)
    assert projected.dtype == full.dtype
    assert projected.device == full.device


@pytest.mark.parametrize(
    "fields",
    ((), ("n_norm", "n_norm"), ("unknown", "valid")),
)
def test_invalid_projection_contract_is_rejected(fields) -> None:
    with pytest.raises(ValueError):
        project_actor_recovery_context(torch.zeros(1, 3), fields)


def test_projection_requires_full_internal_three_dimensional_context() -> None:
    with pytest.raises(ValueError, match="full 3-D recovery context"):
        project_actor_recovery_context(
            torch.zeros(1, 2), ("n_norm", "valid")
        )
