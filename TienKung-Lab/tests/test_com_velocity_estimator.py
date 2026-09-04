from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from legged_lab.estimation.com_velocity_estimator import (
    ComVelocityEstimator,
    ComVelocityEstimatorV2TrainCfg,
    EstimatorFrameHistory,
    ResetWarmupMask,
    TouchdownAfterTransientTracker,
    extract_com_velocity_target,
    extract_recent_actor_history,
    latest_actor_frame,
    partitioned_recovery_group_mask,
    weighted_velocity_mse,
)
from rsl_rl.modules import ActorCritic


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEACHER_CHECKPOINT = REPOSITORY_ROOT / "logs/g1_slope_sys_d.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_teacher_checkpoint_strictly_loads_and_is_frozen() -> None:
    checkpoint = torch.load(TEACHER_CHECKPOINT, map_location="cpu", weights_only=False)
    teacher = ActorCritic(
        960,
        1010,
        29,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.eval().requires_grad_(False)
    assert teacher.actor[0].in_features == 10 * 96
    assert teacher.actor[-1].out_features == 29
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_recent_history_is_newest_five_frames() -> None:
    history = torch.arange(2 * 10 * 96, dtype=torch.float32).reshape(2, 10, 96)
    extracted = extract_recent_actor_history(history.reshape(2, -1))
    assert extracted.shape == (2, 5 * 96)
    torch.testing.assert_close(extracted.reshape(2, 5, 96), history[:, 5:])


def test_reset_warmup_excludes_first_five_policy_steps() -> None:
    warmup = ResetWarmupMask(num_envs=2, warmup_steps=5, device="cpu")
    no_reset = torch.tensor([False, False])
    for _ in range(5):
        assert not torch.any(warmup.eligible_after_step(no_reset))
    assert torch.all(warmup.eligible_after_step(no_reset))
    eligible = warmup.eligible_after_step(torch.tensor([True, False]))
    assert not eligible[0] and eligible[1]
    for _ in range(5):
        assert not warmup.eligible_after_step(no_reset)[0]
    assert warmup.eligible_after_step(no_reset)[0]


def test_estimator_output_shape_and_architecture() -> None:
    estimator = ComVelocityEstimator()
    assert estimator(torch.zeros(7, 480)).shape == (7, 2)
    linear_shapes = [
        (layer.in_features, layer.out_features)
        for layer in estimator.network
        if isinstance(layer, torch.nn.Linear)
    ]
    assert linear_shapes == [(480, 256), (256, 128), (128, 64), (64, 2)]


def test_v2_contract_and_frame_history_order() -> None:
    cfg = ComVelocityEstimatorV2TrainCfg()
    cfg.validate()
    assert cfg.estimator_frame_dim == 99
    assert cfg.input_dim == 495
    model = ComVelocityEstimator(input_dim=cfg.input_dim)
    assert model(torch.zeros(7, 495)).shape == (7, 2)

    history = EstimatorFrameHistory(
        2, device="cpu", imu_acceleration_scale=cfg.imu_acceleration_scale
    )
    for index in range(1, 7):
        actor = torch.full((2, 96), float(index))
        imu = torch.full((2, 3), float(index * 10))
        flattened = history.append(actor, imu, torch.tensor([False, index == 6]))
    frames = flattened.reshape(2, 5, 99)
    torch.testing.assert_close(frames[0, :, 0], torch.tensor([2, 3, 4, 5, 6.0]))
    torch.testing.assert_close(frames[0, :, 96], torch.tensor([1, 1.5, 2, 2.5, 3.0]))
    assert torch.count_nonzero(frames[1, :-1]) == 0
    assert frames[1, -1, 0] == 6.0


def test_latest_actor_frame_and_partitioned_recovery_groups() -> None:
    actor = torch.arange(4 * 10 * 96, dtype=torch.float32).reshape(4, 10, 96)
    torch.testing.assert_close(latest_actor_frame(actor.reshape(4, -1)), actor[:, -1])
    mask = partitioned_recovery_group_mask(10, 8, 0.5, "cpu")
    torch.testing.assert_close(
        mask, torch.tensor([False, False, False, False, True, True, True, True, False, True])
    )


def test_td0_td1_tracker_restarts_on_new_transient_and_reset() -> None:
    tracker = TouchdownAfterTransientTracker(2, "cpu")
    no = torch.tensor([False, False])
    td0, td1 = tracker.update(torch.tensor([True, False]), no, no)
    assert not td0.any() and not td1.any()
    td0, td1 = tracker.update(no, torch.tensor([True, False]), no)
    assert td0.tolist() == [True, False] and not td1.any()
    td0, td1 = tracker.update(no, torch.tensor([True, False]), no)
    assert not td0.any() and td1.tolist() == [True, False]
    tracker.update(torch.tensor([True, True]), no, no)
    td0, td1 = tracker.update(no, torch.tensor([True, True]), torch.tensor([True, False]))
    assert td0.tolist() == [False, True] and not td1.any()


def test_label_uses_whole_body_com_heading_velocity() -> None:
    state = SimpleNamespace(
        com_velocity=torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]]),
        root_lin_vel_b=torch.full((2, 3), 999.0),
    )
    target = extract_com_velocity_target(state)
    torch.testing.assert_close(target, torch.tensor([[1.0, -2.0], [4.0, 5.0]]))


def test_validation_partition_cannot_change_gradient_or_update() -> None:
    torch.manual_seed(7)
    first = ComVelocityEstimator()
    second = ComVelocityEstimator()
    second.load_state_dict(first.state_dict())
    inputs = torch.randn(8, 480)
    train_target = torch.randn(6, 2)
    validation_a = torch.randn(2, 2)
    validation_b = validation_a + 1000.0

    def update(model: ComVelocityEstimator, validation_target: torch.Tensor) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        all_targets = torch.cat((train_target, validation_target), dim=0)
        prediction = model(inputs[:6])
        loss = weighted_velocity_mse(
            prediction,
            all_targets[:6],
            torch.zeros(6, dtype=torch.bool),
            transient_weight=4.0,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    update(first, validation_a)
    update(second, validation_b)
    for first_value, second_value in zip(first.state_dict().values(), second.state_dict().values()):
        torch.testing.assert_close(first_value, second_value, rtol=0.0, atol=0.0)


def test_semantic_checkpoint_reloads_with_identical_inference(tmp_path: Path) -> None:
    model = ComVelocityEstimator()
    model.eval()
    inputs = torch.randn(4, 480)
    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": 480,
        "per_frame_obs_dim": 96,
        "history_length": 5,
        "hidden_dims": [256, 128, 64],
        "output_dim": 2,
        "output_frame": "heading",
        "output_quantity": "whole_body_com_velocity_xy",
        "output_unit": "m/s",
    }
    checkpoint = tmp_path / "estimator.pt"
    torch.save(payload, checkpoint)
    loaded_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    loaded = ComVelocityEstimator()
    loaded.load_state_dict(loaded_payload["model_state_dict"], strict=True)
    loaded.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs), loaded(inputs), rtol=0.0, atol=0.0)


def test_protected_theory_and_original_task_files_are_unchanged() -> None:
    expected = {
        "legged_lab/recovery/certificate.py": (
            "7fbef67ba3faa4bc6fdaa4d6b0de0262f85cb178d0964afe8d2a385c105234ef"
        ),
        "legged_lab/envs/g1/g1_slope_training_config.py": (
            "6ecd0564bddcf80490e2cf066c7dc32574090e03a3ee3366a786d54df9d82137"
        ),
    }
    for relative_path, expected_hash in expected.items():
        assert _sha256(REPOSITORY_ROOT / relative_path) == expected_hash
