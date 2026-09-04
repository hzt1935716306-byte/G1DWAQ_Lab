"""Tensor-only tests for the Unitree G1 left-right symmetry mapping."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "legged_lab/envs/g1/g1_symmetry.py"
SPEC = importlib.util.spec_from_file_location("g1_symmetry_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SYMMETRY_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYMMETRY_MODULE)
compute_symmetric_states = SYMMETRY_MODULE.compute_symmetric_states


JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@pytest.fixture
def fake_env():
    robot = SimpleNamespace(
        joint_names=JOINT_NAMES,
        body_names=("pelvis", "left_ankle_roll_link", "right_ankle_roll_link"),
    )
    robot_cfg = SimpleNamespace(actor_obs_history_length=10, critic_obs_history_length=10)
    return SimpleNamespace(
        robot=robot,
        contact_sensor=SimpleNamespace(
            body_names=("pelvis", "left_ankle_roll_link", "right_ankle_roll_link")
        ),
        cfg=SimpleNamespace(robot=robot_cfg),
        feet_cfg=SimpleNamespace(body_ids=[1, 2]),
    )


@pytest.mark.parametrize("obs_type,frame_dim", [("policy", 96), ("critic", 101)])
def test_observation_mirror_is_an_involution(fake_env, obs_type, frame_dim):
    observations = torch.randn(7, 10 * frame_dim)
    augmented, _ = compute_symmetric_states(fake_env, obs=observations, obs_type=obs_type)

    assert augmented.shape == (14, 10 * frame_dim)
    torch.testing.assert_close(augmented[:7], observations)

    mirrored = augmented[7:]
    mirrored_twice, _ = compute_symmetric_states(fake_env, obs=mirrored, obs_type=obs_type)
    torch.testing.assert_close(mirrored_twice[7:], observations)


def test_action_mirror_swaps_sides_and_flips_roll_yaw(fake_env):
    actions = torch.arange(1, 30, dtype=torch.float32).unsqueeze(0)
    _, augmented = compute_symmetric_states(fake_env, actions=actions)
    mirrored = augmented[1]

    # Left pitch takes right pitch unchanged; left roll/yaw take negated right values.
    assert mirrored[0] == actions[0, 6]
    assert mirrored[1] == -actions[0, 7]
    assert mirrored[2] == -actions[0, 8]
    # Center yaw/roll change sign while center pitch does not.
    assert mirrored[12] == -actions[0, 12]
    assert mirrored[13] == -actions[0, 13]
    assert mirrored[14] == actions[0, 14]

    _, mirrored_twice = compute_symmetric_states(fake_env, actions=mirrored.unsqueeze(0))
    torch.testing.assert_close(mirrored_twice[1], actions[0])


def test_actor_vector_components_have_correct_reflection_signs(fake_env):
    observations = torch.zeros(1, 10 * 96)
    frame = observations.reshape(1, 10, 96)
    frame[..., 0:3] = torch.tensor([1.0, 2.0, 3.0])  # angular velocity
    frame[..., 3:6] = torch.tensor([4.0, 5.0, 6.0])  # projected gravity
    frame[..., 6:9] = torch.tensor([7.0, 8.0, 9.0])  # velocity command

    augmented, _ = compute_symmetric_states(fake_env, obs=observations, obs_type="policy")
    mirrored_frame = augmented[1].reshape(10, 96)[0]

    torch.testing.assert_close(mirrored_frame[0:3], torch.tensor([-1.0, 2.0, -3.0]))
    torch.testing.assert_close(mirrored_frame[3:6], torch.tensor([4.0, -5.0, 6.0]))
    torch.testing.assert_close(mirrored_frame[6:9], torch.tensor([7.0, -8.0, -9.0]))


def test_critic_mirror_swaps_foot_contacts(fake_env):
    observations = torch.zeros(1, 10 * 101)
    frame = observations.reshape(1, 10, 101)
    frame[..., 96:99] = torch.tensor([1.0, 2.0, 3.0])
    frame[..., 99:101] = torch.tensor([1.0, 0.0])

    augmented, _ = compute_symmetric_states(fake_env, obs=observations, obs_type="critic")
    mirrored_frame = augmented[1].reshape(10, 101)[0]

    torch.testing.assert_close(mirrored_frame[96:99], torch.tensor([1.0, -2.0, 3.0]))
    torch.testing.assert_close(mirrored_frame[99:101], torch.tensor([0.0, 1.0]))


def test_actor_recovery_context_is_mirror_invariant(fake_env):
    fake_env.cfg.recovery_context = SimpleNamespace(enabled=True, mode="certificate")
    history = torch.randn(4, 10 * 96)
    context = torch.tensor(
        [[0.0, -1.0, 1.0], [0.5, 0.25, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    )
    observations = torch.cat((history, context), dim=-1)

    augmented, _ = compute_symmetric_states(fake_env, obs=observations, obs_type="policy")

    assert augmented.shape == (8, 963)
    torch.testing.assert_close(augmented[4:, -3:], context)
    mirrored_twice, _ = compute_symmetric_states(
        fake_env, obs=augmented[4:], obs_type="policy"
    )
    torch.testing.assert_close(mirrored_twice[4:], observations)


def test_five_frame_actor_history_with_context_is_mirror_invariant(fake_env):
    fake_env.cfg.robot.actor_obs_history_length = 5
    fake_env.cfg.recovery_context = SimpleNamespace(enabled=True, mode="certificate")
    observations = torch.randn(6, 5 * 96 + 3)
    observations[:, -3:] = torch.tensor([0.6, -0.2, 1.0])

    augmented, _ = compute_symmetric_states(fake_env, obs=observations, obs_type="policy")

    assert augmented.shape == (12, 483)
    torch.testing.assert_close(augmented[6:, -3:], observations[:, -3:])
    mirrored_twice, _ = compute_symmetric_states(
        fake_env, obs=augmented[6:], obs_type="policy"
    )
    torch.testing.assert_close(mirrored_twice[6:], observations)
