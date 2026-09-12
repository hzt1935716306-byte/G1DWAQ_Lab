import math

import numpy as np

from paper_eval_final.src.disturbances import (
    apply_velocity_jump, heading_vector, requested_wrench,
    rotation_world_from_link, rotate, world_wrench_at_link_origin,
)


def test_velocity_jump_changes_physics_velocity_and_preserves_non_targets():
    before = np.asarray([1, 2, 3, 4, 5, 6], dtype=float)
    disturbance = {"direction_Hpush_deg": 90, "linear_speed_mps": 2,
                   "angular_speed_radps": 0.5, "angular_axis": "pitch", "angular_sign": -1}
    after = apply_velocity_jump(before, disturbance, 0.0)
    assert np.allclose(after[:3], [1, 4, 3])
    assert np.allclose(after[3:], [4, 4.5, 6])


def test_world_link_wrench_roundtrip_at_link_origin():
    yaw = math.pi / 3
    quaternion = np.asarray([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    force, torque = np.asarray([10, -3, 2.]), np.asarray([1, 4, -2.])
    local_force, local_torque = world_wrench_at_link_origin(force, torque, quaternion)
    rotation = rotation_world_from_link(quaternion)
    assert np.allclose(rotate(rotation, local_force), force)
    assert np.allclose(rotate(rotation, local_torque), torque)


def test_wrench_release_is_zero_and_150_is_vector_norm():
    disturbance = {
        "duration_s": 0.5,
        "updates": [{"force_norm_N": 150.0, "torque_norm_Nm": 150.0,
                     "direction_Hpush_deg": 90, "torque_axis": "yaw", "torque_sign": -1}],
    }
    force, torque, index = requested_wrench(disturbance, 0.2, 0.1)
    assert math.isclose(np.linalg.norm(force), 150.0)
    assert math.isclose(np.linalg.norm(torque), 150.0)
    assert index == 0
    force, torque, index = requested_wrench(disturbance, 0.2, 0.5)
    assert np.array_equal(force, np.zeros(3)) and np.array_equal(torque, np.zeros(3)) and index is None
