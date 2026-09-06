"""Small, simulator-independent helpers for the final Plane V1 method."""

from __future__ import annotations

from dataclasses import replace

import torch


PLANE_V1_SLOPES_DEG = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)


def plane_v1_context_held_mask(
    touchdown_mask: torch.Tensor,
    dones: torch.Tensor,
    standing_mask: torch.Tensor,
) -> torch.Tensor:
    """Select moving rows whose context must remain unchanged under ZOH.

    Standing commands are intentionally certificate-N/A and clear their
    context immediately, including when a command resample occurs between two
    touchdowns.
    """

    if touchdown_mask.shape != dones.shape or touchdown_mask.shape != standing_mask.shape:
        raise ValueError("touchdown, done, and standing masks must have identical shapes")
    if any(mask.dtype is not torch.bool for mask in (touchdown_mask, dones, standing_mask)):
        raise ValueError("touchdown, done, and standing masks must be boolean tensors")
    return ~touchdown_mask & ~dones & ~standing_mask


def plane_v1_learning_iteration(policy_step: int, steps_per_iteration: int) -> int:
    """Map completed policy steps to the rollout iteration currently executing."""

    if policy_step < 0 or steps_per_iteration <= 0:
        raise ValueError("policy step must be non-negative and rollout length positive")
    if policy_step == 0:
        return 0
    return (policy_step - 1) // steps_per_iteration


def plane_v1_terrain_level(
    learning_iteration: int,
    *,
    level_1_iteration: int = 2000,
    level_2_iteration: int = 4000,
) -> int:
    """Return the deterministic global terrain level for one PPO iteration."""

    if level_1_iteration < 0 or level_2_iteration <= level_1_iteration:
        raise ValueError("terrain curriculum thresholds must satisfy 0 <= L1 < L2")
    if learning_iteration < 0:
        raise ValueError("learning_iteration must be non-negative")
    if learning_iteration >= level_2_iteration:
        return 2
    if learning_iteration >= level_1_iteration:
        return 1
    return 0


def plane_v1_allowed_slopes(level: int) -> tuple[float, ...]:
    """Return the exact signed-slope support at one terrain level."""

    if level == 0:
        return (-5.0, 0.0, 5.0)
    if level == 1:
        return (-10.0, -5.0, 0.0, 5.0, 10.0)
    if level == 2:
        return PLANE_V1_SLOPES_DEG
    raise ValueError("Plane V1 terrain level must be 0, 1, or 2")


def plane_v1_allowed_type_indices(
    level: int,
    slopes_degrees: tuple[float, ...] = PLANE_V1_SLOPES_DEG,
) -> tuple[int, ...]:
    """Map the allowed slope values to the existing terrain column indices."""

    index = {float(slope): position for position, slope in enumerate(slopes_degrees)}
    try:
        return tuple(index[slope] for slope in plane_v1_allowed_slopes(level))
    except KeyError as exc:
        raise ValueError(f"configured terrain is missing Plane V1 slope {exc.args[0]}") from exc


def replace_com_velocity_for_certificate(state, estimated_velocity_xy: torch.Tensor):
    """Replace only CoM XY velocity and its derived DCM offset ``b``.

    Every geometric, gait, command and nominal-lookup field remains the exact
    physical-state value produced by :class:`G1PrivilegedStateExtractor`.
    """

    if estimated_velocity_xy.shape != state.com_velocity[:, :2].shape:
        raise ValueError("estimated CoM velocity must have shape [num_envs, 2]")
    velocity = state.com_velocity.clone()
    velocity[:, :2] = estimated_velocity_xy.to(
        device=velocity.device, dtype=velocity.dtype
    )
    support_position = torch.where(
        state.support_is_left.unsqueeze(-1),
        state.left_foot_position,
        state.right_foot_position,
    )
    xi_xy = state.com_position[:, :2] + velocity[:, :2] / state.omega.unsqueeze(-1)
    b = xi_xy - support_position[:, :2]
    return replace(state, com_velocity=velocity, b=b)


__all__ = [
    "PLANE_V1_SLOPES_DEG",
    "plane_v1_allowed_slopes",
    "plane_v1_allowed_type_indices",
    "plane_v1_context_held_mask",
    "plane_v1_learning_iteration",
    "plane_v1_terrain_level",
    "replace_com_velocity_for_certificate",
]
