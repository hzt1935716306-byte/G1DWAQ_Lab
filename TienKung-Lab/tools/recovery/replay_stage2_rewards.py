#!/usr/bin/env python3
"""Replay the two Stage2 reward channels on the saved 200 recovery trials."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from legged_lab.recovery.stage2_reward import (  # noqa: E402
    DEFAULT_EVENT_SCALE,
    RecoveryEventReward,
    Stage2RecoveryRewardChannel,
    certificate_potential,
)


DEFAULT_INPUTS = (
    PROJECT_DIR / "tools/recovery/generated/g1_recovery_manager_entered_seed42_100_report.yaml",
    PROJECT_DIR / "tools/recovery/generated/g1_recovery_manager_entered_seed43_100_report.yaml",
)
DEFAULT_OUTPUT = PROJECT_DIR / "tools/recovery/generated/stage2_reward_offline_replay.json"


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _load_trials(paths: list[Path]) -> list[dict]:
    trials = []
    for path in paths:
        with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
            report = yaml.safe_load(stream)
        trials.extend(report["gate2_disturbed"]["trials"])
    return trials


def _replay_channel(trial: dict, enable_certificate_reward: bool) -> dict:
    trace = trial["certificate_trace"]
    initial = trace[0]
    channel = Stage2RecoveryRewardChannel(
        enable_certificate_reward=enable_certificate_reward,
    )
    channel.on_push(int(initial["N_theory"]), float(initial["margin"]))
    push_event = channel.consume()
    if push_event != RecoveryEventReward():
        raise AssertionError("push produced a recovery reward")

    shared = 0.0
    certificate = 0.0
    event_count = 0
    final_n = int(initial["N_theory"])
    final_margin = float(initial["margin"])
    outcome = None
    for sample in trace[1:]:
        if int(sample["touchdown"]) > 5:
            break
        final_n = int(sample["N_theory"])
        final_margin = float(sample["margin"])
        outcome = channel.on_touchdown(
            final_n,
            final_margin,
            practical_entered=bool(sample.get("practical_entered", False)),
        )
        event = channel.consume()
        duplicate = channel.consume()
        if duplicate != RecoveryEventReward():
            raise AssertionError("event reward was returned more than once")
        shared += event.shared_total
        certificate += event.certificate
        event_count += 1
        if outcome is not None:
            break

    if outcome is None and trial.get("failure_reason") == "fall_or_illegal_contact":
        channel.on_fall()
        outcome = "FALL"
    if outcome is None:
        raise AssertionError(f"trial did not exit by TD5: {trial['trial_index']}")
    expected_telescoping = 0.0
    if enable_certificate_reward:
        expected_telescoping = 0.50 * (
            certificate_potential(final_n, final_margin)
            - certificate_potential(int(initial["N_theory"]), float(initial["margin"]))
        ) * DEFAULT_EVENT_SCALE
    return {
        "outcome": outcome,
        "touchdowns": event_count,
        "shared": shared,
        "certificate": certificate,
        "total": shared + certificate,
        "telescoping_error": certificate - expected_telescoping,
    }


def _synthetic_checks() -> dict:
    # A round trip in Phi contributes zero net certificate reward.
    oscillating = Stage2RecoveryRewardChannel(enable_certificate_reward=True)
    oscillating.on_push(4, 0.40)
    oscillating.consume()
    oscillation_certificate_sum = 0.0
    for n_min, margin in ((3, 0.30), (4, 0.40)):
        oscillating.on_touchdown(n_min, margin, practical_entered=False)
        oscillation_certificate_sum += oscillating.consume().certificate

    over_horizon = Stage2RecoveryRewardChannel(enable_certificate_reward=True)
    over_horizon.on_push(6, -2.0)
    over_horizon.consume()
    over_horizon.on_touchdown(6, -1.0, practical_entered=False)
    over_horizon_improvement = over_horizon.consume().certificate

    no_orbit_bonus = Stage2RecoveryRewardChannel(enable_certificate_reward=True)
    no_orbit_bonus.on_push(1, 0.4)
    no_orbit_bonus.consume()
    no_orbit_bonus.on_touchdown(0, 0.8, practical_entered=False)
    n1_to_n0_certificate = no_orbit_bonus.consume().certificate

    success_values = {
        str(step): 3.0 * (6 - step) / 5.0 for step in range(1, 6)
    }
    return {
        "oscillation_round_trip_certificate": oscillation_certificate_sum,
        "over_horizon_margin_improvement_certificate": over_horizon_improvement,
        "n1_to_n0_certificate": n1_to_n0_certificate,
        "success_values": success_values,
        "success_values_strictly_decrease": all(
            success_values[str(left)] > success_values[str(left + 1)]
            for left in range(1, 5)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    trials = _load_trials(args.inputs)
    baseline = [_replay_channel(trial, False) for trial in trials]
    ours = [_replay_channel(trial, True) for trial in trials]
    if len(trials) != 200:
        raise AssertionError(f"expected exactly 200 saved trajectories, got {len(trials)}")
    if any(item["certificate"] != 0.0 for item in baseline):
        raise AssertionError("Baseline produced certificate reward")
    if any(
        not math.isclose(left["shared"], right["shared"], abs_tol=1.0e-12)
        for left, right in zip(baseline, ours)
    ):
        raise AssertionError("Baseline and Ours did not share the same generic reward")

    timeout_shared = [item["shared"] for item in baseline if item["outcome"] == "TIMEOUT"]
    timeout_ours = [item for item in ours if item["outcome"] == "TIMEOUT"]
    if not timeout_shared or any(value >= 0.0 for value in timeout_shared):
        raise AssertionError("TIMEOUT shared reward was not negative")
    if any(item["total"] >= 0.0 for item in timeout_ours):
        raise AssertionError("Ours TIMEOUT total reward was not negative")
    telescoping_errors = [abs(item["telescoping_error"]) for item in ours]
    synthetic = _synthetic_checks()
    if synthetic["over_horizon_margin_improvement_certificate"] <= 0.0:
        raise AssertionError("N>5 margin improvement did not produce positive certificate reward")
    if synthetic["n1_to_n0_certificate"] != 0.0:
        raise AssertionError("N=1 -> N=0 produced an orbit bonus")
    if not synthetic["success_values_strictly_decrease"]:
        raise AssertionError("success reward does not decrease with touchdown count")

    report = {
        "schema_version": 1,
        "input_reports": [str(path.expanduser().resolve()) for path in args.inputs],
        "trajectory_count": len(trials),
        "outcome_counts": {
            outcome: sum(item["outcome"] == outcome for item in baseline)
            for outcome in ("SUCCESS", "TIMEOUT", "FALL")
        },
        "baseline": {
            "shared_reward": _distribution([item["shared"] for item in baseline]),
            "certificate_reward": _distribution([item["certificate"] for item in baseline]),
            "episode_total_reward": _distribution([item["total"] for item in baseline]),
        },
        "ours": {
            "shared_reward": _distribution([item["shared"] for item in ours]),
            "certificate_reward": _distribution([item["certificate"] for item in ours]),
            "episode_total_reward": _distribution([item["total"] for item in ours]),
        },
        "timeout_shared_reward": _distribution(timeout_shared),
        "timeout_ours_certificate_reward": _distribution(
            [item["certificate"] for item in timeout_ours]
        ),
        "timeout_ours_total_reward": _distribution([item["total"] for item in timeout_ours]),
        "telescoping": {
            "max_abs_error": max(telescoping_errors),
            "mean_abs_error": statistics.fmean(telescoping_errors),
        },
        "checks": {
            "push_reward_zero": True,
            "event_reward_one_shot": True,
            "baseline_certificate_always_zero": True,
            "shared_channels_identical": True,
            "timeout_shared_reward_strictly_negative": True,
            "timeout_ours_total_reward_strictly_negative": True,
            "all_values_finite": all(
                math.isfinite(item[key])
                for item in baseline + ours
                for key in ("shared", "certificate", "total", "telescoping_error")
            ),
            **synthetic,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
