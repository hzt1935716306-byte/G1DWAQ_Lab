"""Protocol-v1.2 physical disturbance algebra and Isaac wrench bridge."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def rotation_world_from_link(quaternion_wxyz: Any) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape[-1:] != (4,) or not np.isfinite(q).all():
        raise ValueError("finite WXYZ quaternion required")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(np.abs(norm - 1.0) > 2.0e-6):
        raise ValueError("unit WXYZ quaternion required")
    q = q / norm
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def rotate(rotation: np.ndarray, vector: Any) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape[-1:] != (3,) or not np.isfinite(value).all():
        raise ValueError("finite three-vector required")
    return np.einsum("...ij,...j->...i", rotation, value)


def heading_vector(yaw: float, angle_deg: float, magnitude: float, *, axis: str = "horizontal") -> np.ndarray:
    angle = float(yaw) + math.radians(float(angle_deg))
    if axis == "horizontal":
        return np.asarray([magnitude * math.cos(angle), magnitude * math.sin(angle), 0.0], dtype=np.float64)
    if axis == "roll":
        return np.asarray([magnitude * math.cos(yaw), magnitude * math.sin(yaw), 0.0], dtype=np.float64)
    if axis == "pitch":
        return np.asarray([-magnitude * math.sin(yaw), magnitude * math.cos(yaw), 0.0], dtype=np.float64)
    if axis == "yaw":
        return np.asarray([0.0, 0.0, magnitude], dtype=np.float64)
    raise ValueError(f"unknown Hpush axis: {axis}")


def apply_velocity_jump(before_root_velocity_world: Any, disturbance: dict[str, Any], hpush_yaw: float) -> np.ndarray:
    before = np.asarray(before_root_velocity_world, dtype=np.float64)
    if before.shape != (6,) or not np.isfinite(before).all():
        raise ValueError("velocity jump requires a finite 6-D physics root velocity")
    after = before.copy()
    after[:3] += heading_vector(hpush_yaw, disturbance["direction_Hpush_deg"], disturbance["linear_speed_mps"])
    angular = heading_vector(
        hpush_yaw, 0.0,
        float(disturbance["angular_sign"]) * float(disturbance["angular_speed_radps"]),
        axis=disturbance["angular_axis"],
    )
    after[3:] += angular
    return after


def requested_wrench(disturbance: dict[str, Any], hpush_yaw: float, elapsed_s: float) -> tuple[np.ndarray, np.ndarray, int | None]:
    duration = float(disturbance["duration_s"])
    if elapsed_s < -1.0e-9 or elapsed_s >= duration - 1.0e-9:
        return np.zeros(3), np.zeros(3), None
    index = min(int(math.floor(max(0.0, elapsed_s) + 1.0e-9)), len(disturbance["updates"]) - 1)
    update = disturbance["updates"][index]
    force = heading_vector(hpush_yaw, update["direction_Hpush_deg"], update["force_norm_N"])
    torque = heading_vector(hpush_yaw, 0.0,
                            update["torque_sign"] * update["torque_norm_Nm"], axis=update["torque_axis"])
    return force, torque, index


def world_wrench_at_link_origin(force_world: Any, free_torque_world: Any,
                                link_quaternion_wxyz: Any) -> tuple[np.ndarray, np.ndarray]:
    rotation = rotation_world_from_link(link_quaternion_wxyz)
    inverse = np.swapaxes(rotation, -1, -2)
    return rotate(inverse, force_world), rotate(inverse, free_torque_world)


def set_world_wrenches_at_link_origins(composer: Any, force_world: Any, torque_world: Any,
                                       link_quaternions_wxyz: Any, body_id: int, device: str) -> list[dict[str, Any]]:
    import torch

    forces = np.asarray(force_world, dtype=np.float64)
    torques = np.asarray(torque_world, dtype=np.float64)
    quaternions = np.asarray(link_quaternions_wxyz, dtype=np.float64)
    force_link, torque_link = world_wrench_at_link_origin(forces, torques, quaternions)
    composer.reset()
    try:
        composer.set_forces_and_torques(
            forces=torch.as_tensor(force_link[:, None, :], dtype=torch.float32, device=device),
            torques=torch.as_tensor(torque_link[:, None, :], dtype=torch.float32, device=device),
            positions=None,
            body_ids=[body_id],
            is_global=False,
        )
        stored_force = composer.composed_force_as_torch[:, body_id].detach().cpu().numpy()
        stored_torque = composer.composed_torque_as_torch[:, body_id].detach().cpu().numpy()
        if not np.allclose(stored_force, force_link, atol=2.0e-5, rtol=0.0):
            raise ValueError("composer local force differs from requested force")
        if not np.allclose(stored_torque, torque_link, atol=2.0e-5, rtol=0.0):
            raise ValueError("composer local torque differs from requested torque")
    except BaseException:
        composer.reset()
        raise
    reconstructed_force = rotate(rotation_world_from_link(quaternions), stored_force)
    reconstructed_torque = rotate(rotation_world_from_link(quaternions), stored_torque)
    if not np.allclose(reconstructed_force, forces, atol=2.0e-5, rtol=0.0):
        composer.reset()
        raise ValueError("world/link force frame verification failed")
    if not np.allclose(reconstructed_torque, torques, atol=2.0e-5, rtol=0.0):
        composer.reset()
        raise ValueError("world/link torque frame verification failed")
    return [{
        "force_world_N": forces[index].tolist(),
        "torque_world_Nm": torques[index].tolist(),
        "force_link_N": stored_force[index].astype(float).tolist(),
        "torque_link_Nm": stored_torque[index].astype(float).tolist(),
        "position": None,
        "composer_is_global": False,
    } for index in range(len(forces))]


def clear_wrenches(composer: Any, env_ids: Any = None) -> None:
    composer.reset(env_ids=env_ids)
    for name in ("composed_force_as_torch", "composed_torque_as_torch"):
        values = getattr(composer, name)
        if env_ids is not None:
            values = values[env_ids]
        if bool((values != 0).any()):
            raise ValueError(f"residual external wrench after cleanup: {name}")
