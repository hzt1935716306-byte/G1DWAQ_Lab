#!/usr/bin/env python3
"""Small deterministic checks for the Stage2 dual-trigger curriculum."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legged_lab.recovery.push_curriculum import (  # noqa: E402
    CurriculumRecoveryOutcome,
    CurriculumUpgradeReason,
    PushCurriculumController,
)


def _cfg(*, enabled: bool = True, initial_level: int = 1, initial_iterations_in_level: int = 0):
    return SimpleNamespace(
        enable_push_curriculum=enabled,
        initial_level=initial_level,
        initial_iterations_in_level=initial_iterations_in_level,
        level_ratios=(0.25, 0.40, 0.55, 0.70, 0.85, 1.00),
        stage1b_abs_delta_v_xy=(1.0, 1.0),
        k_min_iterations=2,
        k_max_iterations=5,
        statistics_window_episodes=4,
        p5_threshold=0.85,
        median_enter_step_threshold=4.0,
        required_consecutive_pass_windows=2,
        easy_sample_probability=0.20,
        recovery_reward_weight=0.0,
    )


def _record_passing_window(controller: PushCurriculumController) -> None:
    for _ in range(controller.window_size):
        controller.record_episode(
            level_index=controller.level_index,
            outcome=CurriculumRecoveryOutcome.SUCCESS,
            practical_enter_step=4,
        )


def main() -> None:
    expected_ranges = [(ratio, ratio) for ratio in (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)]

    performance = PushCurriculumController(_cfg())
    assert performance.level == 1
    performance.set_learning_iteration(2)
    _record_passing_window(performance)
    assert performance.level == 1
    _record_passing_window(performance)
    assert performance.level == 2
    assert performance.upgrade_history[-1]["upgrade_reason"] == CurriculumUpgradeReason.PERFORMANCE.value

    forced = PushCurriculumController(_cfg())
    forced.set_learning_iteration(5)
    assert forced.level == 2
    assert forced.upgrade_history[-1]["upgrade_reason"] == CurriculumUpgradeReason.MAX_ITERATIONS.value
    levels = [forced.level]
    while not forced.at_final_level:
        forced.set_learning_iteration(forced.current_learning_iteration + 5)
        levels.append(forced.level)
    assert levels == [2, 3, 4, 5, 6]
    history_size = len(forced.upgrade_history)
    _record_passing_window(forced)
    _record_passing_window(forced)
    assert forced.last_window_p5 == 1.0
    reason = forced.set_learning_iteration(forced.current_learning_iteration + 100)
    assert forced.level == 6
    assert len(forced.upgrade_history) == history_size
    assert reason is CurriculumUpgradeReason.NONE

    mixture = PushCurriculumController(_cfg())
    mixture.set_learning_iteration(5)
    mixture.set_learning_iteration(10)
    mixture.set_learning_iteration(15)
    assert mixture.level == 4
    torch.manual_seed(123)
    sampled = mixture.sample_level_indices(20_000, "cpu")
    easy_fraction = float((sampled < mixture.level_index).float().mean().item())
    assert 0.18 <= easy_fraction <= 0.22
    assert torch.all(sampled <= mixture.level_index)
    assert torch.all(sampled >= 0)

    fixed = PushCurriculumController(_cfg(enabled=False))
    assert fixed.level == 6
    assert fixed.current_abs_delta_v_xy == (1.0, 1.0)
    fixed_sampled = fixed.sample_level_indices(100, "cpu")
    assert torch.all(fixed_sampled == 5)

    resumed = PushCurriculumController(_cfg(initial_level=5, initial_iterations_in_level=3))
    assert resumed.level == 5
    assert resumed.current_learning_iteration == 0
    assert resumed.iterations_in_current_level == 3
    resumed.set_learning_iteration(1)
    assert resumed.level == 5
    assert resumed.iterations_in_current_level == 4
    resumed.set_learning_iteration(2)
    assert resumed.level == 6
    assert resumed.upgrade_history[-1]["iterations_in_level"] == 5
    assert resumed.upgrade_history[-1]["upgrade_reason"] == CurriculumUpgradeReason.MAX_ITERATIONS.value

    assert _cfg().recovery_reward_weight == 0.0
    report = {
        "level_ranges_abs_delta_v_xy": expected_ranges,
        "performance_upgrade": "passed",
        "forced_upgrade": "passed",
        "monotonic_levels": levels,
        "level6_stop": "passed",
        "easy_sample_fraction": easy_fraction,
        "fixed_mode_level": fixed.level,
        "fixed_mode_abs_delta_v_xy": fixed.current_abs_delta_v_xy,
        "resumed_level": resumed.level,
        "resumed_initial_iterations_in_level": 3,
        "recovery_reward_weight": _cfg().recovery_reward_weight,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
