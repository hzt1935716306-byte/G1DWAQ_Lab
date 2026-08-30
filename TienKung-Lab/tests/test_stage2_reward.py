from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from legged_lab.recovery.stage2_reward import (
    RecoveryEventReward,
    Stage2RecoveryRewardChannel,
    certificate_potential,
    certificate_potential_tensor,
)
from legged_lab.recovery.push_curriculum import PushCurriculumController


def test_certificate_potential_definition() -> None:
    assert certificate_potential(6, -2.0) == pytest.approx(-0.25)
    assert certificate_potential(5, 0.95) == pytest.approx(1.25)
    assert certificate_potential(2, 0.95) == pytest.approx(4.25)
    assert certificate_potential(1, 0.95) == pytest.approx(5.0)
    assert certificate_potential(0, -123.0) == pytest.approx(5.0)


def test_tensor_potential_matches_scalar() -> None:
    n_min = torch.tensor([6, 5, 4, 3, 2, 1, 0])
    margin = torch.tensor([-1.0, 0.2, 0.4, 0.6, 0.95, 0.3, -0.2])
    expected = torch.tensor(
        [certificate_potential(int(n), float(m)) for n, m in zip(n_min, margin)]
    )
    assert torch.allclose(certificate_potential_tensor(n_min, margin), expected)


def test_generic_rewards_and_one_shot_consumption() -> None:
    channel = Stage2RecoveryRewardChannel(
        enable_certificate_reward=False,
        enable_shared_event_reward=True,
        event_scale=1.0,
    )
    channel.on_push(5, 0.1)
    assert channel.consume() == RecoveryEventReward()
    channel.on_touchdown(4, 0.2, practical_entered=True)
    event = channel.consume()
    assert event.touchdown_cost == pytest.approx(-0.1)
    assert event.success == pytest.approx(3.0)
    assert event.timeout == 0.0
    assert event.certificate == 0.0
    assert channel.consume() == RecoveryEventReward()


def test_timeout_total_shared_reward_is_negative() -> None:
    channel = Stage2RecoveryRewardChannel(
        enable_certificate_reward=False,
        enable_shared_event_reward=True,
        event_scale=1.0,
    )
    channel.on_push(6, -1.0)
    channel.consume()
    shared = 0.0
    for touchdown in range(1, 6):
        outcome = channel.on_touchdown(6, -1.0, practical_entered=False)
        shared += channel.consume().shared_total
    assert outcome == "TIMEOUT"
    assert shared == pytest.approx(-3.5)


def test_shared_events_are_disabled_by_default_and_certificate_scale_is_050() -> None:
    success = Stage2RecoveryRewardChannel(enable_certificate_reward=True)
    success.on_push(6, -2.0)
    success.consume()
    success.on_touchdown(5, 0.0, practical_entered=True)
    event = success.consume()
    assert event.touchdown_cost == 0.0
    assert event.success == 0.0
    assert event.certificate == pytest.approx(
        0.50 * 0.5 * (certificate_potential(5, 0.0) - certificate_potential(6, -2.0))
    )

    timeout = Stage2RecoveryRewardChannel(enable_certificate_reward=False)
    timeout.on_push(6, -1.0)
    timeout.consume()
    for _ in range(5):
        timeout.on_touchdown(6, -1.0, practical_entered=False)
        event = timeout.consume()
    assert event == RecoveryEventReward()


def test_certificate_telescopes_and_has_no_orbit_bonus() -> None:
    channel = Stage2RecoveryRewardChannel(
        enable_certificate_reward=True,
        event_scale=1.0,
    )
    channel.on_push(6, -2.0)
    channel.consume()
    certificate_sum = 0.0
    for n_min, margin in ((4, 0.3), (6, -1.0), (1, 0.2), (0, 0.8)):
        channel.on_touchdown(n_min, margin, practical_entered=False)
        certificate_sum += channel.consume().certificate
    expected = 0.5 * (certificate_potential(0, 0.8) - certificate_potential(6, -2.0))
    assert certificate_sum == pytest.approx(expected)

    orbit = Stage2RecoveryRewardChannel(
        enable_certificate_reward=True,
        event_scale=1.0,
    )
    orbit.on_push(1, 0.2)
    orbit.consume()
    orbit.on_touchdown(0, 0.8, practical_entered=False)
    assert orbit.consume().certificate == 0.0
    assert math.isfinite(certificate_sum)


def test_adaptive_curriculum_upgrades_can_be_frozen() -> None:
    cfg = SimpleNamespace(
        enable_push_curriculum=True,
        adaptive_upgrades_enabled=False,
        level_ratios=(0.25, 0.40, 0.55, 0.70, 0.85, 1.00),
        stage1b_abs_delta_v_xy=(1.0, 1.0),
        k_min_iterations=2,
        k_max_iterations=5,
        statistics_window_episodes=8,
        p5_threshold=0.85,
        median_enter_step_threshold=4.0,
        required_consecutive_pass_windows=2,
        easy_sample_probability=0.2,
    )
    controller = PushCurriculumController(cfg)
    controller.set_learning_iteration(100)
    assert controller.level == 1
    assert controller.level_ratio == pytest.approx(0.25)
