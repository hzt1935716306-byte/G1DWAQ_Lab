# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry transforms for the Unitree G1 velocity task."""

from __future__ import annotations

from functools import lru_cache

import torch

__all__ = ["compute_symmetric_states"]


def _unwrap_env(env):
    """Return the underlying environment when RSL-RL passes a wrapper."""
    return getattr(env, "unwrapped", env)


def _opposite_side_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name.removeprefix("left_")
    if name.startswith("right_"):
        return "left_" + name.removeprefix("right_")
    return name


@lru_cache(maxsize=None)
def _joint_mirror_spec(joint_names: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Build the permutation and signs for sagittal-plane reflection.

    Joint axes in the G1 model follow the usual convention: roll joints rotate
    around x, pitch/knee/elbow joints around y, and yaw joints around z. Under
    reflection across y=0, axial-vector x/z components change sign while y does
    not. Left/right joint pairs are swapped at the same time.
    """
    index_by_name = {name: index for index, name in enumerate(joint_names)}
    if len(index_by_name) != len(joint_names):
        raise ValueError("G1 symmetry requires unique joint names.")

    mirror_indices: list[int] = []
    mirror_signs: list[float] = []
    for name in joint_names:
        opposite_name = _opposite_side_name(name)
        if opposite_name not in index_by_name:
            raise ValueError(f"Missing mirrored G1 joint for {name!r}: {opposite_name!r}")
        mirror_indices.append(index_by_name[opposite_name])

        if "_roll_joint" in name or "_yaw_joint" in name:
            mirror_signs.append(-1.0)
        elif "_pitch_joint" in name or "_knee_joint" in name or "_elbow_joint" in name:
            mirror_signs.append(1.0)
        else:
            raise ValueError(f"Unknown G1 joint axis for symmetry transform: {name!r}")

    return tuple(mirror_indices), tuple(mirror_signs)


@lru_cache(maxsize=None)
def _body_mirror_indices(body_names: tuple[str, ...]) -> tuple[int, ...]:
    """Build a left/right permutation for a list of paired body names."""
    index_by_name = {name: index for index, name in enumerate(body_names)}
    indices: list[int] = []
    for name in body_names:
        opposite_name = _opposite_side_name(name)
        if opposite_name not in index_by_name:
            raise ValueError(f"Missing mirrored G1 body for {name!r}: {opposite_name!r}")
        indices.append(index_by_name[opposite_name])
    return tuple(indices)


def _mirror_joint_data(joint_data: torch.Tensor, joint_names: tuple[str, ...]) -> torch.Tensor:
    indices, signs = _joint_mirror_spec(joint_names)
    index_tensor = torch.tensor(indices, device=joint_data.device, dtype=torch.long)
    sign_tensor = joint_data.new_tensor(signs)
    return joint_data.index_select(-1, index_tensor) * sign_tensor


def _mirror_current_actor_frame(frame: torch.Tensor, joint_names: tuple[str, ...]) -> torch.Tensor:
    """Mirror one or more actor-observation frames along their last axis."""
    num_joints = len(joint_names)
    expected_dim = 9 + 3 * num_joints
    if frame.shape[-1] != expected_dim:
        raise ValueError(f"Expected a {expected_dim}-D G1 actor frame, got {frame.shape[-1]} dimensions.")

    mirrored = frame.clone()
    # Angular velocity is an axial vector; gravity is a polar vector.
    mirrored[..., 0:3] = frame[..., 0:3] * frame.new_tensor((-1.0, 1.0, -1.0))
    mirrored[..., 3:6] = frame[..., 3:6] * frame.new_tensor((1.0, -1.0, 1.0))
    # Command layout: forward velocity, lateral velocity, yaw rate.
    mirrored[..., 6:9] = frame[..., 6:9] * frame.new_tensor((1.0, -1.0, -1.0))

    offset = 9
    for _ in range(3):  # joint position, joint velocity, previous action
        mirrored[..., offset : offset + num_joints] = _mirror_joint_data(
            frame[..., offset : offset + num_joints], joint_names
        )
        offset += num_joints
    return mirrored


def _feet_body_names(env) -> tuple[str, ...]:
    body_ids = env.feet_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = range(*body_ids.indices(len(env.contact_sensor.body_names)))
    # SceneEntityCfg body IDs are resolved in the contact sensor's body-name
    # space, which is not guaranteed to match the articulation body ordering.
    return tuple(env.contact_sensor.body_names[index] for index in body_ids)


def _mirror_observations(env, obs: torch.Tensor, obs_type: str) -> torch.Tensor:
    env = _unwrap_env(env)
    joint_names = tuple(env.robot.joint_names)
    num_joints = len(joint_names)
    actor_frame_dim = 9 + 3 * num_joints

    if obs_type == "policy":
        history_length = env.cfg.robot.actor_obs_history_length
        frame_dim = actor_frame_dim
        feet_names: tuple[str, ...] = ()
        context_cfg = getattr(env.cfg, "recovery_context", None)
        context_dim = 3 if context_cfg is not None and bool(context_cfg.enabled) else 0
    elif obs_type == "critic":
        history_length = env.cfg.robot.critic_obs_history_length
        feet_names = _feet_body_names(env)
        frame_dim = actor_frame_dim + 3 + len(feet_names)
        context_dim = 0
    else:
        raise ValueError(f"Unsupported G1 symmetry observation type: {obs_type!r}")

    history_dim = history_length * frame_dim
    expected_dim = history_dim + context_dim
    if obs.ndim != 2 or obs.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected {obs_type} observations shaped [batch, {expected_dim}], got {tuple(obs.shape)}. "
            "The G1 symmetry mapping must be updated when the observation layout changes."
        )

    history = obs[..., :history_dim]
    context = obs[..., history_dim:]
    frames = history.reshape(obs.shape[0], history_length, frame_dim)
    mirrored = frames.clone()
    mirrored[..., :actor_frame_dim] = _mirror_current_actor_frame(frames[..., :actor_frame_dim], joint_names)

    if obs_type == "critic":
        # Root linear velocity is a polar vector.
        root_velocity_start = actor_frame_dim
        mirrored[..., root_velocity_start : root_velocity_start + 3] = (
            frames[..., root_velocity_start : root_velocity_start + 3]
            * frames.new_tensor((1.0, -1.0, 1.0))
        )
        contact_start = root_velocity_start + 3
        contact_indices = torch.tensor(
            _body_mirror_indices(feet_names), device=obs.device, dtype=torch.long
        )
        mirrored[..., contact_start:] = frames[..., contact_start:].index_select(-1, contact_indices)

    mirrored_history = mirrored.reshape(obs.shape[0], history_dim)
    if context_dim:
        return torch.cat((mirrored_history, context), dim=-1)
    return mirrored_history


@torch.no_grad()
def compute_symmetric_states(
    env,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return the original batch followed by its left-right mirrored batch."""
    env = _unwrap_env(env)

    if obs is not None:
        mirrored_obs = _mirror_observations(env, obs, obs_type)
        obs_aug = torch.cat((obs, mirrored_obs), dim=0)
    else:
        obs_aug = None

    if actions is not None:
        joint_names = tuple(env.robot.joint_names)
        if actions.ndim != 2 or actions.shape[-1] != len(joint_names):
            raise ValueError(
                f"Expected G1 actions shaped [batch, {len(joint_names)}], got {tuple(actions.shape)}."
            )
        mirrored_actions = _mirror_joint_data(actions, joint_names)
        actions_aug = torch.cat((actions, mirrored_actions), dim=0)
    else:
        actions_aug = None

    return obs_aug, actions_aug
