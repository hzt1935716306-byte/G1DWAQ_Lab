"""Recoverability context normalization for the Stage2 actor observation."""

from __future__ import annotations

import torch


RECOVERY_CONTEXT_DIM = 3
RECOVERY_CONTEXT_MODES = ("zero", "certificate")


def normalize_recovery_context(
    n_min: torch.Tensor,
    margin: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return ``[N_norm, margin_norm, valid]`` without changing certificate semantics."""

    if n_min.shape != margin.shape or n_min.shape != valid.shape:
        raise ValueError("N_min, margin, and valid must have identical shapes")
    if valid.dtype is not torch.bool:
        raise ValueError("certificate validity must be a boolean tensor")

    n_norm = torch.clamp(n_min.to(torch.float32), min=0.0, max=6.0) / 6.0
    margin = margin.to(torch.float32)
    margin_norm = torch.where(
        margin >= 0.0,
        torch.clamp(margin / 0.95, min=0.0, max=1.0),
        torch.clamp(margin / 2.0, min=-1.0, max=0.0),
    )
    context = torch.stack((n_norm, margin_norm, valid.to(torch.float32)), dim=-1)
    return torch.where(valid.unsqueeze(-1), context, torch.zeros_like(context))
