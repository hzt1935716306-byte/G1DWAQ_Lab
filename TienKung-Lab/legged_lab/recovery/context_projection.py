"""Actor-visible projections of the full Plane V1 certificate context."""

from __future__ import annotations

import torch


FULL_RECOVERY_CONTEXT_FIELDS = ("n_norm", "margin_norm", "valid")
RECOVERY_CONTEXT_FIELD_INDICES = {
    "n_norm": 0,
    "margin_norm": 1,
    "valid": 2,
}


def validate_actor_recovery_context_fields(fields) -> tuple[str, ...]:
    fields = tuple(fields)
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("actor recovery context fields must be non-empty and unique")
    unknown = set(fields) - set(RECOVERY_CONTEXT_FIELD_INDICES)
    if unknown:
        raise ValueError(f"unknown actor recovery context fields: {sorted(unknown)}")
    return fields


def project_actor_recovery_context(
    full_context: torch.Tensor,
    fields,
) -> torch.Tensor:
    """Select real Actor input columns from full ``[N, margin, valid]`` context."""

    fields = validate_actor_recovery_context_fields(fields)
    if full_context.shape[-1] != len(FULL_RECOVERY_CONTEXT_FIELDS):
        raise ValueError(
            f"expected full 3-D recovery context, got {full_context.shape[-1]} dimensions"
        )
    indices = tuple(RECOVERY_CONTEXT_FIELD_INDICES[field] for field in fields)
    return full_context[..., indices]


__all__ = [
    "FULL_RECOVERY_CONTEXT_FIELDS",
    "RECOVERY_CONTEXT_FIELD_INDICES",
    "project_actor_recovery_context",
    "validate_actor_recovery_context_fields",
]
