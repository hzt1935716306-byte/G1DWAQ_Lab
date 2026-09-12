"""Timestamp-based sustained-recovery detector frozen by protocol v1.3."""
from __future__ import annotations

from collections import deque
import math
from typing import Any

import numpy as np


EPS = 1.0e-9


def _time_mean(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) == 1 or times[-1] - times[0] <= EPS:
        return float(values[-1])
    return float(np.trapz(values, times) / (times[-1] - times[0]))


class RecoveryDetector:
    def __init__(self, disturbance_end: float, config: dict[str, Any]):
        self.end = float(disturbance_end)
        self.cfg = config
        self.window_s = float(config["window_s"])
        self.hold_s = float(config["confirmation_hold_s"])
        self.deadline = self.end + float(config["entry_deadline_after_disturbance_end_s"])
        self.observation_end = self.end + float(config["observation_after_disturbance_end_s"])
        self.samples: deque[dict[str, Any]] = deque()
        self.candidate_entry: float | None = None
        self.current_confirmed = False
        self.first_entry: float | None = None
        self.confirmations: list[dict[str, float]] = []
        self.relapse_events: list[dict[str, float]] = []
        self.last_time: float | None = None
        self.last_passed = False
        self.terminal = False

    def _window(self, timestamp: float) -> list[dict[str, Any]] | None:
        left = timestamp - self.window_s
        while len(self.samples) > 1 and self.samples[1]["timestamp"] <= left + EPS:
            self.samples.popleft()
        if not self.samples or self.samples[0]["timestamp"] > left + EPS:
            return None
        values = list(self.samples)
        if values[0]["timestamp"] < left - EPS:
            first, second = values[0], values[1]
            ratio = (left - first["timestamp"]) / (second["timestamp"] - first["timestamp"])
            clipped = {"timestamp": left}
            for key in ("velocity_error", "gravity_tilt", "world_z_angular_error", "clearance"):
                clipped[key] = first[key] + ratio * (second[key] - first[key])
            for key in ("physical_failure", "out_of_area", "valid"):
                clipped[key] = first[key] if ratio < 1.0 else second[key]
            values[0] = clipped
        return values

    def update(self, sample: dict[str, Any]) -> bool:
        timestamp = float(sample["timestamp"])
        if timestamp <= self.end + EPS:
            return False
        if self.last_time is not None and timestamp <= self.last_time + EPS:
            raise ValueError("recovery timestamps must strictly increase")
        if timestamp > self.observation_end + 0.021:
            raise ValueError("recovery sample exceeds observation endpoint")
        self.last_time = timestamp
        required = ("velocity_error", "gravity_tilt", "world_z_angular_error", "clearance")
        finite = all(math.isfinite(float(sample[key])) for key in required)
        entry = {**sample, "valid": bool(sample.get("valid", True) and finite)}
        self.samples.append(entry)
        window = self._window(timestamp)
        passed = False
        if window is not None and window[0]["timestamp"] >= self.end - EPS:
            times = np.asarray([row["timestamp"] for row in window], dtype=np.float64)
            velocity = np.asarray([row["velocity_error"] for row in window], dtype=np.float64)
            tilt = np.asarray([row["gravity_tilt"] for row in window], dtype=np.float64)
            angular = np.asarray([row["world_z_angular_error"] for row in window], dtype=np.float64)
            clearance = np.asarray([row["clearance"] for row in window], dtype=np.float64)
            passed = bool(
                _time_mean(times, velocity) <= self.cfg["mean_com_horizontal_velocity_error_max_mps"] + EPS
                and float(np.max(velocity)) <= self.cfg["peak_com_horizontal_velocity_error_max_mps"] + EPS
                and math.sqrt(_time_mean(times, tilt * tilt)) <= math.radians(self.cfg["gravity_tilt_rms_max_deg"]) + EPS
                and float(np.max(tilt)) <= math.radians(self.cfg["gravity_tilt_peak_max_deg"]) + EPS
                and math.sqrt(_time_mean(times, angular * angular)) <= self.cfg["world_z_angular_velocity_error_rms_max_radps"] + EPS
                and float(np.min(clearance)) >= self.cfg["pelvis_clearance_min_m"] - EPS
                and all(row["valid"] and not row.get("physical_failure") and not row.get("out_of_area") for row in window)
            )
        if passed:
            if self.candidate_entry is None:
                self.candidate_entry = timestamp
                if self.first_entry is None:
                    self.first_entry = timestamp
            if not self.current_confirmed and timestamp - self.candidate_entry >= self.hold_s - EPS:
                self.current_confirmed = True
                self.confirmations.append({"entry": self.candidate_entry, "confirmation": timestamp})
        else:
            if self.current_confirmed:
                self.relapse_events.append({"timestamp": timestamp})
            self.candidate_entry = None
            self.current_confirmed = False
        self.last_passed = passed
        self.terminal |= bool(sample.get("physical_failure") or sample.get("out_of_area") or not entry["valid"])
        return passed

    def result(self, *, survived_to_end: bool) -> dict[str, Any]:
        observed = self.last_time is not None and self.last_time >= self.observation_end - 0.021
        final_entry = self.candidate_entry if (
            survived_to_end and observed and not self.terminal and self.current_confirmed
            and self.candidate_entry is not None and self.candidate_entry <= self.deadline + EPS
        ) else None
        return {
            "first_entry": self.first_entry,
            "final_entry": final_entry,
            "confirmations": list(self.confirmations),
            "relapse_events": list(self.relapse_events),
            "recovery_time": None if final_entry is None else final_entry - self.end,
            "recovered_sustained": final_entry is not None,
            "survived_post_observation": bool(survived_to_end and observed and not self.terminal),
            "observation_end": self.observation_end,
        }
