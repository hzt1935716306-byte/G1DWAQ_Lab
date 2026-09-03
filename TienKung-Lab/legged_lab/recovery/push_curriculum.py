"""Global Stage2 push curriculum state and recovery statistics.

This module is independent of Isaac Lab.  It owns only curriculum progression
and aggregate logging; physical pushes and practical-gait measurements stay in
the environment integration layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
import statistics
from typing import Sequence

import torch


class CurriculumUpgradeReason(str, Enum):
    NONE = "none"
    PERFORMANCE = "performance"
    MAX_ITERATIONS = "max_iterations"


class CurriculumRecoveryOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    FALL = "FALL"


@dataclass
class LevelRecoveryStatistics:
    recovery_episodes: int = 0
    success: int = 0
    timeout: int = 0
    fall: int = 0
    practical_enter_step_counts: list[int] = field(default_factory=lambda: [0] * 6)
    success_recovery_times_s: list[float] = field(default_factory=list)

    def record(
        self,
        outcome: CurriculumRecoveryOutcome,
        practical_enter_step: int | None,
        recovery_time_s: float | None = None,
    ) -> None:
        self.recovery_episodes += 1
        if outcome is CurriculumRecoveryOutcome.SUCCESS:
            if practical_enter_step is None or not 1 <= practical_enter_step <= 5:
                raise ValueError("SUCCESS requires practical_enter_step in [1, 5]")
            self.success += 1
            self.practical_enter_step_counts[practical_enter_step] += 1
            if recovery_time_s is not None:
                if recovery_time_s < 0.0:
                    raise ValueError("recovery time cannot be negative")
                self.success_recovery_times_s.append(float(recovery_time_s))
        elif outcome is CurriculumRecoveryOutcome.TIMEOUT:
            self.timeout += 1
        elif outcome is CurriculumRecoveryOutcome.FALL:
            self.fall += 1
        else:
            raise ValueError(f"Unknown recovery outcome: {outcome}")

    @property
    def p5(self) -> float:
        return self.success / self.recovery_episodes if self.recovery_episodes else 0.0

    def p_at(self, touchdown: int) -> float:
        if not 1 <= touchdown <= 5:
            raise ValueError("touchdown probability is defined only for P1 through P5")
        successes = sum(self.practical_enter_step_counts[1 : touchdown + 1])
        return successes / self.recovery_episodes if self.recovery_episodes else 0.0

    @property
    def mean_practical_enter_step(self) -> float | None:
        if not self.success:
            return None
        weighted_sum = sum(step * count for step, count in enumerate(self.practical_enter_step_counts))
        return weighted_sum / self.success

    @property
    def median_practical_enter_step(self) -> float | None:
        if not self.success:
            return None
        targets = ((self.success - 1) // 2, self.success // 2)
        values = []
        cumulative = 0
        for step, count in enumerate(self.practical_enter_step_counts):
            previous = cumulative
            cumulative += count
            for target in targets[len(values) :]:
                if previous <= target < cumulative:
                    values.append(step)
                else:
                    break
        return 0.5 * (values[0] + values[1])

    @property
    def mean_recovery_time_s(self) -> float | None:
        return statistics.mean(self.success_recovery_times_s) if self.success_recovery_times_s else None

    @property
    def median_recovery_time_s(self) -> float | None:
        return statistics.median(self.success_recovery_times_s) if self.success_recovery_times_s else None

    def to_dict(self) -> dict:
        result = asdict(self)
        result.pop("success_recovery_times_s")
        result.update(
            {
                **{f"P{touchdown}": self.p_at(touchdown) for touchdown in range(1, 6)},
                "mean_practical_enter_step": self.mean_practical_enter_step,
                "median_practical_enter_step": self.median_practical_enter_step,
                "mean_recovery_time_s": self.mean_recovery_time_s,
                "median_recovery_time_s": self.median_recovery_time_s,
                "timeout_rate": self.timeout / self.recovery_episodes if self.recovery_episodes else 0.0,
                "fall_rate": self.fall / self.recovery_episodes if self.recovery_episodes else 0.0,
            }
        )
        return result


class PushCurriculumController:
    """Six-level monotonic curriculum with performance and timeout upgrades."""

    def __init__(self, cfg):
        self.enabled = bool(cfg.enable_push_curriculum)
        self.adaptive_upgrades_enabled = bool(getattr(cfg, "adaptive_upgrades_enabled", True))
        self.level_ratios = tuple(float(value) for value in cfg.level_ratios)
        self.stage1b_abs_delta_v_xy = tuple(float(value) for value in cfg.stage1b_abs_delta_v_xy)
        self.k_min = int(cfg.k_min_iterations)
        self.k_max = int(cfg.k_max_iterations)
        self.window_size = int(cfg.statistics_window_episodes)
        self.p5_threshold = float(cfg.p5_threshold)
        self.median_enter_step_threshold = float(cfg.median_enter_step_threshold)
        self.required_pass_windows = int(cfg.required_consecutive_pass_windows)
        self.easy_sample_probability = float(cfg.easy_sample_probability)
        self.initial_level = int(getattr(cfg, "initial_level", 1))
        self.initial_iterations_in_level = int(getattr(cfg, "initial_iterations_in_level", 0))
        self._validate()

        # Disabling the curriculum restores the fixed Stage1B maximum range.
        self.level_index = self.initial_level - 1 if self.enabled else len(self.level_ratios) - 1
        self.current_learning_iteration = 0
        self.level_start_iteration = -self.initial_iterations_in_level if self.enabled else 0
        self.consecutive_pass_windows = 0
        self.last_window_p5: float | None = None
        self.last_window_median_enter_step: float | None = None
        self.last_upgrade_reason = CurriculumUpgradeReason.NONE
        self._pending_upgrade_reason = CurriculumUpgradeReason.NONE
        self.upgrade_history: list[dict] = []
        self.level_statistics = [LevelRecoveryStatistics() for _ in self.level_ratios]
        self._window_outcomes: list[tuple[CurriculumRecoveryOutcome, int | None]] = []
        self.push_sample_count = 0
        self.easy_push_sample_count = 0
        self._upgrade_log_path: Path | None = None

    def _validate(self) -> None:
        if not self.level_ratios or any(value <= 0.0 for value in self.level_ratios):
            raise ValueError("level_ratios must contain positive values")
        if any(right <= left for left, right in zip(self.level_ratios, self.level_ratios[1:])):
            raise ValueError("level_ratios must be strictly increasing")
        if self.level_ratios[-1] != 1.0:
            raise ValueError("the final curriculum ratio must be 1.0")
        if len(self.stage1b_abs_delta_v_xy) != 2 or any(
            value <= 0.0 for value in self.stage1b_abs_delta_v_xy
        ):
            raise ValueError("stage1b_abs_delta_v_xy must contain two positive values")
        if self.k_min < 0 or self.k_max <= self.k_min:
            raise ValueError("K_max must be greater than K_min")
        if self.window_size <= 0 or self.required_pass_windows <= 0:
            raise ValueError("window size and required pass windows must be positive")
        if not 0.0 <= self.p5_threshold <= 1.0:
            raise ValueError("P5 threshold must be in [0, 1]")
        if not 1.0 <= self.median_enter_step_threshold <= 5.0:
            raise ValueError("median enter-step threshold must be in [1, 5]")
        if not 0.0 <= self.easy_sample_probability <= 1.0:
            raise ValueError("easy sample probability must be in [0, 1]")
        if not 1 <= self.initial_level <= len(self.level_ratios):
            raise ValueError("initial curriculum level is outside the configured level range")
        if self.initial_iterations_in_level < 0:
            raise ValueError("initial curriculum iterations in level must be non-negative")
        if (
            self.enabled
            and self.initial_level < len(self.level_ratios)
            and self.initial_iterations_in_level >= self.k_max
        ):
            raise ValueError("initial curriculum iterations must be below K_max before the final level")

    @property
    def level(self) -> int:
        return self.level_index + 1

    @property
    def level_ratio(self) -> float:
        return self.level_ratios[self.level_index]

    @property
    def current_abs_delta_v_xy(self) -> tuple[float, float]:
        return tuple(value * self.level_ratio for value in self.stage1b_abs_delta_v_xy)

    @property
    def iterations_in_current_level(self) -> int:
        return self.current_learning_iteration - self.level_start_iteration

    @property
    def at_final_level(self) -> bool:
        return self.level_index == len(self.level_ratios) - 1

    @property
    def easy_sample_fraction(self) -> float:
        return self.easy_push_sample_count / self.push_sample_count if self.push_sample_count else 0.0

    def bounds_for_level_index(self, level_index: int) -> tuple[float, float]:
        ratio = self.level_ratios[level_index]
        return tuple(value * ratio for value in self.stage1b_abs_delta_v_xy)

    def configure_upgrade_log(self, log_dir: str | Path) -> None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._upgrade_log_path = log_dir / "push_curriculum_upgrades.jsonl"
        metadata = {
            "enabled": self.enabled,
            "adaptive_upgrades_enabled": self.adaptive_upgrades_enabled,
            "initial_level": self.initial_level,
            "initial_iterations_in_level": self.initial_iterations_in_level,
            "level_ratios": self.level_ratios,
            "stage1b_abs_delta_v_xy": self.stage1b_abs_delta_v_xy,
            "K_min": self.k_min,
            "K_max": self.k_max,
            "statistics_window_episodes": self.window_size,
            "P5_threshold": self.p5_threshold,
            "median_enter_step_threshold": self.median_enter_step_threshold,
            "required_consecutive_pass_windows": self.required_pass_windows,
            "easy_sample_probability": self.easy_sample_probability,
        }
        with (log_dir / "push_curriculum_config.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)

    def sample_level_indices(self, count: int, device: str | torch.device) -> torch.Tensor:
        if count < 0:
            raise ValueError("sample count cannot be negative")
        sampled = torch.full((count,), self.level_index, dtype=torch.long, device=device)
        if self.enabled and self.level_index > 0 and count > 0 and self.easy_sample_probability > 0.0:
            easy_mask = torch.rand(count, device=device) < self.easy_sample_probability
            easy_count = int(easy_mask.sum().item())
            if easy_count:
                sampled[easy_mask] = torch.randint(
                    low=0,
                    high=self.level_index,
                    size=(easy_count,),
                    device=device,
                )
            self.easy_push_sample_count += easy_count
        self.push_sample_count += count
        return sampled

    def record_episode(
        self,
        *,
        level_index: int,
        outcome: CurriculumRecoveryOutcome,
        practical_enter_step: int | None,
        recovery_time_s: float | None = None,
    ) -> None:
        self.level_statistics[level_index].record(outcome, practical_enter_step, recovery_time_s)
        # Episodes that started at an older level may finish just after an
        # upgrade.  Keep them in their per-level totals, but do not contaminate
        # the new level's performance window.
        if not self.enabled or level_index != self.level_index:
            return
        self._window_outcomes.append((outcome, practical_enter_step))
        if len(self._window_outcomes) == self.window_size:
            self._close_statistics_window()

    def _close_statistics_window(self) -> None:
        success_steps = [
            step
            for outcome, step in self._window_outcomes
            if outcome is CurriculumRecoveryOutcome.SUCCESS and step is not None and step <= 5
        ]
        self.last_window_p5 = len(success_steps) / self.window_size
        self.last_window_median_enter_step = (
            float(statistics.median(success_steps)) if success_steps else None
        )
        performance_pass = (
            self.last_window_p5 >= self.p5_threshold
            and self.last_window_median_enter_step is not None
            and self.last_window_median_enter_step <= self.median_enter_step_threshold
        )
        self.consecutive_pass_windows = self.consecutive_pass_windows + 1 if performance_pass else 0
        self._window_outcomes.clear()
        self._maybe_upgrade()

    def set_learning_iteration(self, learning_iteration: int) -> CurriculumUpgradeReason:
        learning_iteration = int(learning_iteration)
        if learning_iteration < self.current_learning_iteration:
            raise ValueError("learning iteration must be monotonic")
        self.current_learning_iteration = learning_iteration
        self.last_upgrade_reason = CurriculumUpgradeReason.NONE
        reason = self._maybe_upgrade()
        if reason is not CurriculumUpgradeReason.NONE:
            self._pending_upgrade_reason = CurriculumUpgradeReason.NONE
            return reason
        if self._pending_upgrade_reason is not CurriculumUpgradeReason.NONE:
            reason = self._pending_upgrade_reason
            self._pending_upgrade_reason = CurriculumUpgradeReason.NONE
            self.last_upgrade_reason = reason
        return reason

    def _maybe_upgrade(self) -> CurriculumUpgradeReason:
        if not self.enabled or not self.adaptive_upgrades_enabled or self.at_final_level:
            return CurriculumUpgradeReason.NONE
        iterations = self.iterations_in_current_level
        if iterations >= self.k_min and self.consecutive_pass_windows >= self.required_pass_windows:
            return self._upgrade(CurriculumUpgradeReason.PERFORMANCE)
        if iterations >= self.k_max:
            return self._upgrade(CurriculumUpgradeReason.MAX_ITERATIONS)
        return CurriculumUpgradeReason.NONE

    def _upgrade(self, reason: CurriculumUpgradeReason) -> CurriculumUpgradeReason:
        old_level = self.level
        event = {
            "old_level": old_level,
            "new_level": old_level + 1,
            "learning_iteration": self.current_learning_iteration,
            "old_level_ratio": self.level_ratios[self.level_index],
            "new_level_ratio": self.level_ratios[self.level_index + 1],
            "P5": self.last_window_p5,
            "median_enter_step": self.last_window_median_enter_step,
            "iterations_in_level": self.iterations_in_current_level,
            "upgrade_reason": reason.value,
        }
        self.level_index += 1
        self.level_start_iteration = self.current_learning_iteration
        self.consecutive_pass_windows = 0
        self._window_outcomes.clear()
        self.last_upgrade_reason = reason
        self._pending_upgrade_reason = reason
        self.upgrade_history.append(event)
        print(
            "[PushCurriculum] "
            f"L{event['old_level']} -> L{event['new_level']}, "
            f"P5={event['P5']}, median_enter_step={event['median_enter_step']}, "
            f"iterations_in_level={event['iterations_in_level']}, reason={reason.value}",
            flush=True,
        )
        if self._upgrade_log_path is not None:
            with self._upgrade_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        return reason

    def snapshot(self) -> dict:
        current_stats = self.level_statistics[self.level_index]
        return {
            "curriculum_level": self.level,
            "level_ratio": self.level_ratio,
            "adaptive_upgrades_enabled": self.adaptive_upgrades_enabled,
            "current_delta_v_max_x": self.current_abs_delta_v_xy[0],
            "current_delta_v_max_y": self.current_abs_delta_v_xy[1],
            "iterations_in_current_level": self.iterations_in_current_level,
            "P5": self.last_window_p5,
            "median_enter_step": self.last_window_median_enter_step,
            "consecutive_pass_windows": self.consecutive_pass_windows,
            "upgrade_reason": self.last_upgrade_reason.value,
            "easy_sample_fraction": self.easy_sample_fraction,
            "current_level_statistics": current_stats.to_dict(),
            "all_level_statistics": [item.to_dict() for item in self.level_statistics],
        }


def record_episode_batch(
    controller: PushCurriculumController,
    level_indices: Sequence[int],
    outcomes: Sequence[CurriculumRecoveryOutcome],
    practical_enter_steps: Sequence[int | None],
    recovery_times_s: Sequence[float | None] | None = None,
) -> None:
    """Record a small batch of completed recovery episodes in deterministic order."""

    if recovery_times_s is None:
        recovery_times_s = [None] * len(level_indices)
    if not (
        len(level_indices)
        == len(outcomes)
        == len(practical_enter_steps)
        == len(recovery_times_s)
    ):
        raise ValueError("recovery episode batch fields must have equal lengths")
    for level_index, outcome, enter_step, recovery_time_s in zip(
        level_indices, outcomes, practical_enter_steps, recovery_times_s
    ):
        controller.record_episode(
            level_index=int(level_index),
            outcome=outcome,
            practical_enter_step=enter_step,
            recovery_time_s=recovery_time_s,
        )
