"""Frozen 200 Hz physical-touchdown detector and derived alternating stream."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


EPS = 1.0e-9


@dataclass(frozen=True)
class TouchdownCounts:
    physical: int
    alternating: int
    same_foot: int
    simultaneous: int


class TouchdownDetector:
    def __init__(self, *, contact_on_N: float, contact_off_N: float, stable_frames: int, debounce_s: float):
        if not 0 <= contact_off_N < contact_on_N:
            raise ValueError("contact thresholds must satisfy 0 <= off < on")
        if type(stable_frames) is not int or stable_frames < 1 or debounce_s < 0:
            raise ValueError("invalid stable-frame/debounce configuration")
        self.on = float(contact_on_N)
        self.off = float(contact_off_N)
        self.stable_frames = stable_frames
        self.debounce_s = float(debounce_s)
        self.reset()

    def reset(self) -> None:
        self.contact: np.ndarray | None = None
        self.pending = [0, 0]
        self.airborne_frames = [0, 0]
        self.armed = [False, False]
        self.last_landing = [-math.inf, -math.inf]
        self.last_time: float | None = None
        self.alternating_anchor: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.physical_count = 0
        self.alternating_count = 0
        self.same_foot_count = 0
        self.simultaneous_count = 0
        self.chatter_count = 0

    def counts(self) -> TouchdownCounts:
        return TouchdownCounts(self.physical_count, self.alternating_count, self.same_foot_count, self.simultaneous_count)

    def update(self, timestamp: float, force_norms_N: list[float] | np.ndarray) -> list[dict[str, Any]]:
        time = float(timestamp)
        forces = np.asarray(force_norms_N, dtype=np.float64)
        if forces.shape != (2,) or not np.isfinite(forces).all() or np.any(forces < 0):
            raise ValueError("two finite nonnegative foot-force norms are required")
        if self.last_time is not None and time <= self.last_time + EPS:
            raise ValueError("touchdown timestamps must strictly increase")
        self.last_time = time
        if self.contact is None:
            self.contact = forces > self.on
            return []

        landed: list[int] = []
        for foot in range(2):
            if not self.contact[foot] and forces[foot] <= self.off:
                self.airborne_frames[foot] += 1
                if self.airborne_frames[foot] >= self.stable_frames:
                    self.armed[foot] = True
            else:
                self.airborne_frames[foot] = 0
            threshold = self.off if self.contact[foot] else self.on
            target = bool(forces[foot] > threshold)
            if target == bool(self.contact[foot]):
                if self.pending[foot]:
                    self.chatter_count += 1
                self.pending[foot] = 0
                continue
            self.pending[foot] += 1
            if self.pending[foot] < self.stable_frames:
                continue
            self.contact[foot] = target
            self.pending[foot] = 0
            if not target:
                self.armed[foot] = True
            elif self.armed[foot]:
                self.armed[foot] = False
                if time - self.last_landing[foot] >= self.debounce_s - EPS:
                    self.last_landing[foot] = time
                    landed.append(foot)
                else:
                    self.chatter_count += 1

        produced: list[dict[str, Any]] = []
        simultaneous = len(landed) == 2
        simultaneous_group = None
        if simultaneous:
            self.simultaneous_count += 1
            simultaneous_group = self.simultaneous_count
            self.alternating_anchor = None
        for foot_index in landed:
            self.physical_count += 1
            foot = ("left", "right")[foot_index]
            event = {
                "event": "physical_touchdown", "timestamp": time, "foot": foot,
                "physical_index": self.physical_count, "simultaneous": simultaneous,
                "simultaneous_group": simultaneous_group,
            }
            self.events.append(event)
            produced.append(event)
        if len(landed) == 1:
            foot = ("left", "right")[landed[0]]
            if self.alternating_anchor is not None and self.alternating_anchor["foot"] == foot:
                self.same_foot_count += 1
                event = {"event": "same_foot_relanding", "timestamp": time, "foot": foot}
            else:
                self.alternating_count += 1
                event = {
                    "event": "alternating_touchdown", "timestamp": time, "foot": foot,
                    "alternating_index": self.alternating_count,
                    "interval_s": None if self.alternating_anchor is None else time - self.alternating_anchor["timestamp"],
                }
            self.alternating_anchor = {"timestamp": time, "foot": foot}
            self.events.append(event)
            produced.append(event)
        elif simultaneous:
            event = {"event": "simultaneous_touchdown", "timestamp": time, "feet": ["left", "right"],
                     "simultaneous_group": simultaneous_group}
            self.events.append(event)
            produced.append(event)
        return produced
