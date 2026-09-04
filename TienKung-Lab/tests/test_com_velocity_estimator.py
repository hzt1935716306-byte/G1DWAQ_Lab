from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from legged_lab.estimation.com_velocity_estimator import (
    ComVelocityEstimator,
    ComVelocityEstimatorV2TrainCfg,
    EstimatorFrameHistory,
    EstimatorRolloutBuffer,
    ManualPushAlignmentDiagnostic,
    ResetWarmupMask,
    TouchdownAfterTransientTracker,
    extract_com_velocity_target,
    extract_recent_actor_history,
    fixed_length_rollout_indices,
    iteration_checkpoint_filename,
    latest_actor_frame,
    load_v2_training_checkpoint,
    partitioned_recovery_group_mask,
    velocity_estimator_selection_score,
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
    first = ComVelocityEstimator(input_dim=495)
    second = ComVelocityEstimator(input_dim=495)
    second.load_state_dict(first.state_dict())
    inputs = torch.randn(8, 495)
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


def test_two_iterations_of_three_steps_call_environment_exactly_six_times() -> None:
    class FakeEnvironment:
        step_count = 0

        def step(self) -> None:
            self.step_count += 1

    environment = FakeEnvironment()
    for _ in range(2):
        for _ in fixed_length_rollout_indices(3):
            environment.step()
    assert environment.step_count == 6


def test_iteration_rollout_buffer_shapes_and_exact_minibatch_traversal() -> None:
    buffer = EstimatorRolloutBuffer(3, 2, 5, device="cpu")
    assert buffer.inputs.shape == (3, 2, 5)
    assert buffer.targets.shape == (3, 2, 2)
    for step in range(3):
        identifiers = torch.tensor([2 * step, 2 * step + 1], dtype=torch.float32)
        inputs = identifiers[:, None].repeat(1, 5)
        targets = torch.zeros(2, 2)
        eligible = torch.tensor([True, step != 1])
        mask = torch.zeros(2, dtype=torch.bool)
        buffer.add(inputs, targets, eligible, mask, mask, mask, mask, mask)
    assert buffer.full
    assert buffer.eligible_count() == 5
    visited: dict[int, list[int]] = {0: [], 1: []}
    generator = torch.Generator().manual_seed(9)
    for epoch, _, inputs, _ in buffer.iter_minibatches(2, 2, generator=generator):
        visited[epoch].extend(int(value) for value in inputs[:, 0])
    assert sorted(visited[0]) == [0, 1, 2, 4, 5]
    assert sorted(visited[1]) == [0, 1, 2, 4, 5]
    assert len(visited[0]) == len(set(visited[0]))
    assert len(visited[1]) == len(set(visited[1]))


def _v2_checkpoint_payload(
    model: ComVelocityEstimator,
    teacher_hash: str,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "input_dim": 495,
        "per_frame_obs_dim": 99,
        "actor_per_frame_obs_dim": 96,
        "imu_input_dim": 3,
        "history_length": 5,
        "hidden_dims": [256, 128, 64],
        "output_dim": 2,
        "output_frame": "heading",
        "output_quantity": "whole_body_com_velocity_xy",
        "output_unit": "m/s",
        "teacher_checkpoint_hash": teacher_hash,
    }


def test_legacy_v2_checkpoint_cannot_warm_start_formal_long_run(tmp_path: Path) -> None:
    teacher_hash = "teacher"
    source = ComVelocityEstimator(input_dim=495)
    checkpoint = tmp_path / "legacy_v2.pt"
    torch.save(_v2_checkpoint_payload(source, teacher_hash), checkpoint)
    loaded = ComVelocityEstimator(input_dim=495)
    optimizer = torch.optim.Adam(loaded.parameters(), lr=1.0e-3)
    with pytest.raises(RuntimeError, match="comparison-only"):
        load_v2_training_checkpoint(
            checkpoint, loaded, optimizer, teacher_hash=teacher_hash, restore_rng=False
        )
    assert not optimizer.state_dict()["state"]


def test_iteration_checkpoint_restores_model_optimizer_and_iteration(tmp_path: Path) -> None:
    teacher_hash = "teacher"
    source = ComVelocityEstimator(input_dim=495)
    source_optimizer = torch.optim.Adam(source.parameters(), lr=3.0e-4)
    loss = source(torch.randn(8, 495)).square().mean()
    loss.backward()
    source_optimizer.step()
    payload = _v2_checkpoint_payload(source, teacher_hash) | {
        "optimizer_state_dict": source_optimizer.state_dict(),
        "current_iteration": 123,
        "global_policy_steps": 2952,
        "optimizer_updates": 738,
        "best_selection_score": 0.25,
        "best_iteration": 100,
        "training_configuration": {"max_iterations": 5000},
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": [],
    }
    checkpoint = tmp_path / "iteration_v2.pt"
    torch.save(payload, checkpoint)
    loaded = ComVelocityEstimator(input_dim=495)
    loaded_optimizer = torch.optim.Adam(loaded.parameters(), lr=1.0e-2)
    info = load_v2_training_checkpoint(
        checkpoint, loaded, loaded_optimizer, teacher_hash=teacher_hash, restore_rng=False
    )
    assert info.mode == "iteration_resume"
    assert info.start_iteration == 123
    assert info.global_policy_steps == 2952
    assert info.optimizer_updates == 738
    assert not info.optimizer_restarted
    for expected, actual in zip(source.parameters(), loaded.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)
    source_states = list(source_optimizer.state_dict()["state"].values())
    loaded_states = list(loaded_optimizer.state_dict()["state"].values())
    assert len(source_states) == len(loaded_states)
    for expected, actual in zip(source_states, loaded_states):
        torch.testing.assert_close(expected["exp_avg"], actual["exp_avg"])
        torch.testing.assert_close(expected["exp_avg_sq"], actual["exp_avg_sq"])


def test_periodic_checkpoint_name_and_selection_score_are_fixed() -> None:
    assert iteration_checkpoint_filename(100) == "checkpoint_iteration_0100.pt"
    assert iteration_checkpoint_filename(5000) == "checkpoint_iteration_5000.pt"
    metrics = {
        "td0": {"vector_rmse": 2.0},
        "td1": {"vector_rmse": 4.0},
        "overall": {"vector_rmse": 6.0},
    }
    assert velocity_estimator_selection_score(metrics) == pytest.approx(3.5)


def test_manual_push_alignment_requires_one_shared_post_step_timestamp() -> None:
    diagnostic = ManualPushAlignmentDiagnostic()
    diagnostic.record(
        pushed_env_count=4,
        global_policy_step=9,
        push_sim_step=32,
        post_step_sim_step=36,
        sim_decimation=4,
        observation_frame_sim_step=36,
        imu_frame_sim_step=36,
        target_frame_sim_step=36,
        imu_timestamp_s=0.18,
        imu_current_timestamp_s=0.18,
    )
    assert diagnostic.records[0]["aligned"] is True
    diagnostic.record(
        pushed_env_count=4,
        global_policy_step=800,
        push_sim_step=3196,
        post_step_sim_step=3200,
        sim_decimation=4,
        observation_frame_sim_step=3200,
        imu_frame_sim_step=3200,
        target_frame_sim_step=3200,
        imu_timestamp_s=16.0000019,
        imu_current_timestamp_s=16.0,
    )
    assert diagnostic.records[1]["aligned"] is True
    with pytest.raises(RuntimeError, match="one-frame alignment mismatch"):
        diagnostic.record(
            pushed_env_count=1,
            global_policy_step=10,
            push_sim_step=36,
            post_step_sim_step=40,
            sim_decimation=4,
            observation_frame_sim_step=40,
            imu_frame_sim_step=36,
            target_frame_sim_step=40,
            imu_timestamp_s=0.20,
            imu_current_timestamp_s=0.20,
        )
    with pytest.raises(RuntimeError, match="one-frame alignment mismatch"):
        diagnostic.record(
            pushed_env_count=1,
            global_policy_step=11,
            push_sim_step=40,
            post_step_sim_step=44,
            sim_decimation=4,
            observation_frame_sim_step=44,
            imu_frame_sim_step=44,
            target_frame_sim_step=44,
            imu_timestamp_s=0.20,
            imu_current_timestamp_s=0.22,
        )


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
