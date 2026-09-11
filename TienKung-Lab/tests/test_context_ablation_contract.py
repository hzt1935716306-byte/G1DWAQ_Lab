"""Unit tests for the strict Plane V1 context-ablation diff helper."""

from legged_lab.recovery.context_ablation_contract import (
    config_differences,
    unexpected_config_differences,
)


def test_config_differences_reports_exact_leaf_path() -> None:
    differences = config_differences(
        {"actor": {"context": ("n_norm", "margin_norm", "valid")}, "seed": 42},
        {"actor": {"context": ("n_norm", "valid")}, "seed": 42},
    )

    assert len(differences) == 1
    assert differences[0].path == "actor.context"


def test_allowed_path_does_not_silently_allow_sibling_differences() -> None:
    unexpected = unexpected_config_differences(
        {"context": ("n_norm", "margin_norm", "valid"), "reward": {"weight": 1.0}},
        {"context": ("n_norm", "valid"), "reward": {"weight": 2.0}},
        allowed_paths=("context",),
    )

    assert [difference.path for difference in unexpected] == ["reward.weight"]
