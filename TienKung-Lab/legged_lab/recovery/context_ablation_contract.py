"""Strict config-diff contract for Plane V1 Actor-context ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from legged_lab.recovery.rl_only_matched_contract import flatten_config


@dataclass(frozen=True)
class ConfigDifference:
    path: str
    reference: Any
    candidate: Any


def config_differences(reference: Any, candidate: Any) -> tuple[ConfigDifference, ...]:
    """Return every differing flattened config leaf."""

    reference_flat = flatten_config(reference)
    candidate_flat = flatten_config(candidate)
    differences = []
    for path in sorted(set(reference_flat) | set(candidate_flat)):
        reference_value = reference_flat.get(path, "<absent>")
        candidate_value = candidate_flat.get(path, "<absent>")
        if reference_value != candidate_value:
            differences.append(ConfigDifference(path, reference_value, candidate_value))
    return tuple(differences)


def unexpected_config_differences(
    reference: Any,
    candidate: Any,
    *,
    allowed_paths: tuple[str, ...],
) -> tuple[ConfigDifference, ...]:
    """Return differences outside an exact allow-list (no prefix matching)."""

    allowed = set(allowed_paths)
    return tuple(
        difference
        for difference in config_differences(reference, candidate)
        if difference.path not in allowed
    )


__all__ = [
    "ConfigDifference",
    "config_differences",
    "unexpected_config_differences",
]
