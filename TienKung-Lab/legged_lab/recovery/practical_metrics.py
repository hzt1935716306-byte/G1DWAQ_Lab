"""Shared practical-gait metric definitions for runtime and calibration."""

from __future__ import annotations

import numpy as np
import torch


def practical_frame_errors(
    com_velocity_xy: torch.Tensor,
    command_velocity_xy: torch.Tensor,
    roll_pitch: torch.Tensor,
    nominal_roll_pitch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-frame velocity norm and absolute attitude errors."""

    velocity_error = torch.linalg.vector_norm(
        com_velocity_xy - command_velocity_xy,
        dim=-1,
    )
    attitude_error = torch.abs(roll_pitch - nominal_roll_pitch)
    return velocity_error, attitude_error


def practical_interval_means_from_sums(
    velocity_error_sum,
    absolute_attitude_error_sum,
    sample_count,
):
    """Reduce complete-interval sums with the same semantics in all callers."""

    if isinstance(velocity_error_sum, torch.Tensor):
        safe_count = torch.clamp(sample_count, min=1).to(velocity_error_sum.dtype)
        return (
            velocity_error_sum / safe_count,
            absolute_attitude_error_sum / safe_count.unsqueeze(-1),
        )
    safe_count = max(int(sample_count), 1)
    return (
        float(velocity_error_sum) / safe_count,
        np.asarray(absolute_attitude_error_sum, dtype=np.float64) / safe_count,
    )


__all__ = ["practical_frame_errors", "practical_interval_means_from_sums"]
