"""Config-diff helpers for the Plane V1 RL-only matched ablation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


RL_ONLY_EXPECTED_ENV_DIFFERENCE_PREFIXES = (
    "com_velocity_source",
    "estimator_checkpoint_path",
    "estimator_imu_acceleration_scale",
    "plane_recovery",
    "plane_v1_reward",
    "push_curriculum",
    "push_mode",
    "recovery_context",
    "scene.imu",
    "stage2_reward",
    "terrain_level_1_iteration",
    "terrain_level_2_iteration",
    # The two events have identical timing and delta-v distributions, but the
    # Plane runtime wrapper additionally reports the known push to its recovery
    # bookkeeping. RL-only deliberately uses Isaac Lab's direct event.
    "domain_rand.events.push_robot",
)


@dataclass(frozen=True)
class ConfigDifference:
    path: str
    reference: Any
    rl_only: Any


def _leaf_value(value: Any) -> Any:
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config dictionaries into stable dotted paths."""

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_config(value[key], path))
        return flattened
    if isinstance(value, (tuple, list)):
        return {prefix: tuple(_leaf_value(item) for item in value)}
    return {prefix: _leaf_value(value)}


def unexpected_config_differences(
    reference: Any,
    rl_only: Any,
    *,
    allowed_prefixes: tuple[str, ...],
) -> tuple[ConfigDifference, ...]:
    """Return every leaf difference outside the explicitly allowed prefixes."""

    reference_flat = flatten_config(reference)
    rl_only_flat = flatten_config(rl_only)
    paths = sorted(set(reference_flat) | set(rl_only_flat))
    differences = []
    for path in paths:
        if any(path == allowed or path.startswith(f"{allowed}.") for allowed in allowed_prefixes):
            continue
        reference_value = reference_flat.get(path, "<absent>")
        rl_only_value = rl_only_flat.get(path, "<absent>")
        if reference_value != rl_only_value:
            differences.append(ConfigDifference(path, reference_value, rl_only_value))
    return tuple(differences)


__all__ = [
    "ConfigDifference",
    "RL_ONLY_EXPECTED_ENV_DIFFERENCE_PREFIXES",
    "flatten_config",
    "unexpected_config_differences",
]
