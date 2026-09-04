"""Tensor-only contract tests for the final Plane V1 method."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from legged_lab.estimation.com_velocity_estimator import (
    ComVelocityEstimator,
    load_com_velocity_estimator_for_inference,
)
from legged_lab.recovery.plane_v1 import (
    PLANE_V1_SLOPES_DEG,
    plane_v1_allowed_slopes,
    plane_v1_allowed_type_indices,
    plane_v1_learning_iteration,
    plane_v1_terrain_level,
    replace_com_velocity_for_certificate,
)
from legged_lab.recovery.push_curriculum import PushCurriculumController
from legged_lab.recovery.stage2_reward import (
    PlaneV1RecoveryRewardChannel,
    PlaneV1RewardParameters,
    certificate_potential,
    plane_v1_touchdown_reward,
)


def _event(previous_n: int, current_n: int, *, touchdown_index: int = 1):
    return plane_v1_touchdown_reward(
        certificate_potential(previous_n, 0.0),
        current_n,
        0.0,
        touchdown_index=touchdown_index,
        terrain_plane_valid=True,
        solver_valid=True,
        enabled=True,
    )


def test_plane_v1_reward_encourages_n_decrease_and_target_entry() -> None:
    assert _event(4, 3).total > 0.0
    assert _event(2, 1).total > 0.0


def test_plane_v1_reward_charges_unchanged_unrecovered_touchdown() -> None:
    assert _event(4, 4).total == pytest.approx(-0.05)


def test_plane_v1_reward_penalizes_n_increase_more_than_step_cost() -> None:
    assert _event(3, 4).total < -0.05


def _sequence_reward(sequence: tuple[int, ...]) -> float:
    previous_phi = certificate_potential(sequence[0], 0.0)
    total = 0.0
    for index, n_min in enumerate(sequence[1:], start=1):
        result = plane_v1_touchdown_reward(
            previous_phi,
            n_min,
            0.0,
            touchdown_index=index,
            terrain_plane_valid=True,
            solver_valid=True,
            enabled=True,
        )
        total += result.total
        assert result.phi_current is not None
        previous_phi = result.phi_current
    return total


def test_plane_v1_faster_recovery_has_larger_cumulative_reward() -> None:
    assert _sequence_reward((4, 3, 2, 1)) > _sequence_reward((4, 4, 3, 2, 1))


def test_plane_v1_td5_unrecovered_has_extra_penalty() -> None:
    result = _event(4, 4, touchdown_index=5)
    assert result.step_cost == pytest.approx(-0.05)
    assert result.td5_penalty == pytest.approx(-0.25)


def test_plane_v1_target_orbit_is_zero_but_degradation_is_negative() -> None:
    assert _event(1, 1).total == pytest.approx(0.0)
    assert _event(1, 2).total < 0.0


def test_plane_v1_delta_phi_clipping_bounds_progress() -> None:
    cfg = PlaneV1RewardParameters(certificate_delta_phi_clip=2.0)
    result = plane_v1_touchdown_reward(
        -100.0,
        1,
        0.0,
        touchdown_index=1,
        terrain_plane_valid=True,
        solver_valid=True,
        enabled=True,
        parameters=cfg,
    )
    assert abs(result.progress) <= 0.5
    assert result.progress == pytest.approx(0.5)


def test_plane_v1_invalid_geometry_and_solver_failure_are_distinct() -> None:
    geometry_invalid = plane_v1_touchdown_reward(
        None,
        6,
        -3.0,
        touchdown_index=5,
        terrain_plane_valid=False,
        solver_valid=False,
        enabled=True,
    )
    assert geometry_invalid.total == pytest.approx(-0.30)
    solver_failure = plane_v1_touchdown_reward(
        1.0,
        6,
        -3.0,
        touchdown_index=5,
        terrain_plane_valid=True,
        solver_valid=False,
        enabled=True,
    )
    assert solver_failure.total == 0.0
    assert not solver_failure.update_previous_phi


def test_plane_v1_reward_off_is_exactly_zero() -> None:
    for geometry_valid, solver_valid in ((True, True), (False, False), (True, False)):
        result = plane_v1_touchdown_reward(
            certificate_potential(4, 0.0),
            3,
            0.0,
            touchdown_index=5,
            terrain_plane_valid=geometry_valid,
            solver_valid=solver_valid,
            enabled=False,
        )
        assert result.total == 0.0


def test_plane_v1_td0_initializes_without_reward_and_td5_always_ends() -> None:
    channel = PlaneV1RecoveryRewardChannel(enabled=True)
    channel.on_push()
    td0 = channel.on_touchdown(4, 0.0, practical_entered=True)
    assert td0.total == 0.0
    assert channel.active
    assert channel.touchdown_index == 0
    td1 = channel.on_touchdown(3, 0.0, practical_entered=True)
    assert td1.total > 0.0
    assert channel.active
    for _ in range(4):
        channel.on_touchdown(1, 0.0, practical_entered=True)
    assert channel.touchdown_index == 5
    assert not channel.active


@pytest.mark.parametrize(
    "iteration,expected",
    [(0, 0), (1999, 0), (2000, 1), (3999, 1), (4000, 2), (9999, 2)],
)
def test_plane_v1_terrain_schedule(iteration: int, expected: int) -> None:
    assert plane_v1_terrain_level(iteration) == expected


def test_plane_v1_terrain_level_supports_are_exact() -> None:
    assert plane_v1_allowed_slopes(0) == (-5.0, 0.0, 5.0)
    assert set(plane_v1_allowed_slopes(1)) == {-10.0, -5.0, 0.0, 5.0, 10.0}
    assert plane_v1_allowed_slopes(2) == PLANE_V1_SLOPES_DEG
    assert plane_v1_allowed_type_indices(0) == (2, 3, 4)
    assert plane_v1_allowed_type_indices(1) == (1, 2, 3, 4, 5)
    assert plane_v1_allowed_type_indices(2) == tuple(range(7))


def test_policy_step_boundary_does_not_advance_terrain_one_step_early() -> None:
    assert plane_v1_learning_iteration(0, 24) == 0
    assert plane_v1_learning_iteration(1, 24) == 0
    assert plane_v1_learning_iteration(24, 24) == 0
    assert plane_v1_learning_iteration(25, 24) == 1


def test_disabled_push_curriculum_is_fixed_at_full_range() -> None:
    cfg = SimpleNamespace(
        enable_push_curriculum=False,
        adaptive_upgrades_enabled=False,
        initial_level=1,
        initial_iterations_in_level=0,
        level_ratios=(0.25, 0.40, 0.55, 0.70, 0.85, 1.00),
        stage1b_abs_delta_v_xy=(1.0, 1.0),
        k_min_iterations=500,
        k_max_iterations=1800,
        statistics_window_episodes=500,
        p5_threshold=0.85,
        median_enter_step_threshold=4.0,
        required_consecutive_pass_windows=2,
        easy_sample_probability=0.0,
    )
    controller = PushCurriculumController(cfg)
    assert controller.current_abs_delta_v_xy == (1.0, 1.0)
    assert torch.all(controller.sample_level_indices(1024, "cpu") == 5)
    controller.set_learning_iteration(10000)
    assert controller.current_abs_delta_v_xy == (1.0, 1.0)


@dataclass(frozen=True)
class _State:
    com_position: torch.Tensor
    com_velocity: torch.Tensor
    omega: torch.Tensor
    support_is_left: torch.Tensor
    left_foot_position: torch.Tensor
    right_foot_position: torch.Tensor
    b: torch.Tensor
    q: torch.Tensor
    signed_slope: torch.Tensor


def test_estimator_source_replaces_only_velocity_and_derived_b() -> None:
    state = _State(
        com_position=torch.tensor([[1.0, 2.0, 0.7], [3.0, 4.0, 0.8]]),
        com_velocity=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        omega=torch.tensor([2.0, 4.0]),
        support_is_left=torch.tensor([True, False]),
        left_foot_position=torch.tensor([[0.5, 1.0, 0.0], [2.0, 3.0, 0.0]]),
        right_foot_position=torch.tensor([[0.4, 1.2, 0.0], [2.5, 3.5, 0.0]]),
        b=torch.zeros(2, 2),
        q=torch.randn(2, 2),
        signed_slope=torch.tensor([0.0, 0.1]),
    )
    estimated = torch.tensor([[0.8, -0.2], [-0.4, 0.6]])
    replaced = replace_com_velocity_for_certificate(state, estimated)
    expected_support = torch.tensor([[0.5, 1.0], [2.5, 3.5]])
    expected_b = state.com_position[:, :2] + estimated / state.omega[:, None] - expected_support
    torch.testing.assert_close(replaced.b, expected_b)
    torch.testing.assert_close(replaced.com_velocity[:, :2], estimated)
    torch.testing.assert_close(replaced.com_velocity[:, 2], state.com_velocity[:, 2])
    assert replaced.q is state.q
    assert replaced.signed_slope is state.signed_slope


def _estimator_payload(model: ComVelocityEstimator) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "input_dim": 495,
        "history_length": 5,
        "actor_per_frame_obs_dim": 96,
        "per_frame_obs_dim": 99,
        "imu_input_dim": 3,
        "imu_acceleration_scale": 0.05,
        "hidden_dims": [256, 128, 64],
        "output_dim": 2,
        "output_frame": "heading",
        "output_quantity": "whole_body_com_velocity_xy",
        "output_unit": "m/s",
    }


def test_strict_estimator_inference_load_is_frozen_and_exact(tmp_path) -> None:
    torch.manual_seed(7)
    model = ComVelocityEstimator(495, [256, 128, 64], 2)
    path = tmp_path / "estimator.pt"
    torch.save(_estimator_payload(model), path)
    loaded, metadata = load_com_velocity_estimator_for_inference(path)
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    sample = torch.randn(3, 495)
    torch.testing.assert_close(model(sample), loaded(sample))
    assert metadata["output_quantity"] == "whole_body_com_velocity_xy"


def test_strict_estimator_inference_load_rejects_empty_missing_and_mismatch(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="empty"):
        load_com_velocity_estimator_for_inference("")
    with pytest.raises(RuntimeError, match="does not exist"):
        load_com_velocity_estimator_for_inference(tmp_path / "missing.pt")
    model = ComVelocityEstimator(495, [256, 128, 64], 2)
    payload = _estimator_payload(model)
    payload["history_length"] = 4
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="semantic mismatch"):
        load_com_velocity_estimator_for_inference(path)
