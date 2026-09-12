"""Protocol trial lifecycle, denominator state, and terminal classification."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .recovery_detector import RecoveryDetector
from .touchdown_detector import TouchdownDetector


class TrialMachine:
    def __init__(self, plan: dict[str, Any], protocol: dict[str, Any]):
        self.plan = plan
        self.protocol = protocol
        td = protocol["metrics"]["touchdown"]
        self.touchdowns = TouchdownDetector(**td)
        self.events: list[dict[str, Any]] = []
        self.frames: list[dict[str, Any]] = []
        self.status: str | None = None
        self.termination_kind: str | None = None
        self.task_outcome: str | None = None
        self.failure_reason: str | None = None
        self.invalid_kind: str | None = None
        self.push_applied = False
        self.disturbance_start: float | None = None
        self.disturbance_end: float | None = None
        self.disturbance_metadata: dict[str, Any] = {}
        self.recovery: RecoveryDetector | None = None
        self.td0_time: float | None = None
        self.certificate_valid = False
        self.n_min: int | None = None
        self.margin: float | None = None
        self.margin_group: str | None = None
        self.cert_after_recovery = False
        self.pre_reset_snapshot: dict[str, Any] | None = None
        self.last_physics_time: float | None = None

    def feed_physics(self, timestamp: float, forces: list[float], failure_reason: str | None = None) -> list[dict[str, Any]]:
        if self.status:
            return []
        self.last_physics_time = float(timestamp)
        produced = self.touchdowns.update(timestamp, forces)
        for event in produced:
            self.events.append(event)
            if (event["event"] == "alternating_touchdown" and self.disturbance_end is not None
                    and event["timestamp"] > self.disturbance_end and self.td0_time is None):
                self.td0_time = float(event["timestamp"])
                self.events.append({"event": "TD0", "timestamp": self.td0_time})
        if failure_reason:
            reason = "PRE_PUSH_PHYSICAL_FAILURE" if self.disturbance_start is None and self.plan["experiment_id"] in (1, 3) else failure_reason
            self.finish("PHYSICAL_RESET", "FAILURE", reason)
        return produced

    def controller_numerical_failure(self, timestamp: float) -> None:
        self.last_physics_time = timestamp
        self.finish("PHYSICAL_RESET", "FAILURE", "CONTROLLER_NUMERICAL_FAILURE")

    def evaluation_invalid(self, timestamp: float, reason: str) -> None:
        self.last_physics_time = timestamp
        self.invalid_kind = "EVAL_INVALID"
        self.finish("EVALUATION_ABORT", "INVALID", reason)

    def unresolved_invalid(self, timestamp: float, reason: str) -> None:
        self.last_physics_time = timestamp
        self.invalid_kind = "UNRESOLVED_INVALID"
        self.finish("EVALUATION_ABORT", "INVALID", reason)

    def on_disturbance_start(self, timestamp: float, metadata: dict[str, Any]) -> None:
        if self.status or self.push_applied:
            raise ValueError("disturbance can be applied exactly once to an active trial")
        self.disturbance_start = float(timestamp)
        self.push_applied = True
        self.disturbance_metadata.update(metadata)
        self.events.append({"event": "disturbance_start", "timestamp": timestamp, **metadata})
        duration = float(self.plan["disturbance"]["duration_s"])
        self.disturbance_end = float(timestamp) + duration
        self.recovery = RecoveryDetector(self.disturbance_end, self.protocol["metrics"]["recovery"])
        if duration == 0:
            self.events.append({"event": "disturbance_end", "timestamp": timestamp})

    def on_disturbance_end(self, timestamp: float) -> None:
        if self.disturbance_end is None:
            raise ValueError("cannot release an intervention that never started")
        self.disturbance_end = float(timestamp)
        self.recovery = RecoveryDetector(self.disturbance_end, self.protocol["metrics"]["recovery"])
        self.events.append({"event": "disturbance_end", "timestamp": timestamp})

    def set_certificate(self, *, valid: bool, n_min: int | None, margin: float | None, diagnostic: dict[str, Any]) -> None:
        if self.td0_time is None or self.n_min is not None or self.certificate_valid:
            raise ValueError("certificate must be recorded once, at TD0")
        self.certificate_valid = bool(valid)
        self.n_min = int(n_min) if n_min is not None else None
        self.margin = float(margin) if margin is not None else None
        self.margin_group = margin_group(self.margin) if self.certificate_valid else None
        self.events.append({"event": "offline_certificate", "timestamp": self.td0_time,
                            "valid": self.certificate_valid, "Nmin": self.n_min,
                            "margin": self.margin, "diagnostic": diagnostic})

    def feed_policy(self, frame: dict[str, Any]) -> None:
        if self.status:
            return
        timestamp = float(frame["timestamp"])
        self.frames.append(frame)
        if not frame.get("data_valid", True):
            self.evaluation_invalid(timestamp, "sensor/log corruption")
            return
        experiment = self.plan["experiment_id"]
        if experiment == 2:
            if timestamp >= float(self.plan["horizon_s"]) - 1.0e-9:
                self.finish("HORIZON_REACHED", "SUCCESS", None)
            return
        if not self.push_applied:
            return
        assert self.disturbance_end is not None and self.recovery is not None
        if timestamp > self.disturbance_end + 1.0e-9:
            self.recovery.update({
                "timestamp": timestamp,
                "velocity_error": float(np.linalg.norm(np.asarray(frame["com_velocity_heading"][:2])
                                                        - np.asarray(frame["command"][:2]))),
                "gravity_tilt": float(frame["gravity_tilt_rad"]),
                "world_z_angular_error": abs(float(frame["angular_velocity_world_z"]) - float(frame["command"][2])),
                "clearance": float(frame["pelvis_clearance_m"]),
                "physical_failure": False,
                "out_of_area": bool(frame["out_of_area"]),
                "valid": bool(frame["data_valid"]),
            })
        if timestamp >= self.disturbance_end + 10.0 - 1.0e-9:
            recovered = self.recovery.result(survived_to_end=True)["recovered_sustained"]
            self.finish("HORIZON_REACHED", "SUCCESS" if recovered else "FAILURE",
                        None if recovered else "NOT_RECOVERED")

    def finish(self, termination_kind: str, outcome: str, reason: str | None) -> None:
        if self.status:
            return
        self.termination_kind = termination_kind
        self.task_outcome = outcome
        self.failure_reason = reason
        self.status = "TERMINAL"
        self.events.append({"event": "terminal", "timestamp": self.last_physics_time,
                            "termination_kind": termination_kind, "task_outcome": outcome,
                            "failure_reason": reason})

    def attach_pre_reset_snapshot(self, snapshot: dict[str, Any]) -> None:
        if self.termination_kind != "PHYSICAL_RESET":
            raise ValueError("pre-reset snapshots belong only to physical resets")
        self.pre_reset_snapshot = snapshot

    def result(self) -> dict[str, Any]:
        if not self.status:
            raise ValueError("cannot serialize an active trial")
        recovery = (self.recovery.result(survived_to_end=self.termination_kind == "HORIZON_REACHED")
                    if self.recovery else {
                        "first_entry": None, "final_entry": None, "confirmations": [], "relapse_events": [],
                        "recovery_time": None, "recovered_sustained": False,
                        "survived_post_observation": False, "observation_end": None,
                    })
        alternating = [event for event in self.touchdowns.events if event["event"] == "alternating_touchdown"]
        physical = [event for event in self.touchdowns.events if event["event"] == "physical_touchdown"]
        final_entry = recovery["final_entry"]
        if self.td0_time is not None and final_entry is not None and final_entry < self.td0_time:
            self.cert_after_recovery = True
        n_td0 = None
        if self.td0_time is not None and final_entry is not None and not self.cert_after_recovery:
            n_td0 = sum(self.td0_time < event["timestamp"] < final_entry for event in alternating)
        disturbance_end = self.disturbance_end
        krec = (sum(disturbance_end < event["timestamp"] < final_entry for event in alternating)
                if disturbance_end is not None and final_entry is not None else None)
        krec_physical = (sum(disturbance_end < event["timestamp"] < final_entry for event in physical)
                         if disturbance_end is not None and final_entry is not None else None)
        quality = trajectory_metrics(self.frames, completed=self.termination_kind == "HORIZON_REACHED")
        eval_invalid = self.invalid_kind == "EVAL_INVALID"
        cert_invalid = self.plan["experiment_id"] == 1 and not self.certificate_valid
        return {
            **self.plan,
            "termination_kind": self.termination_kind,
            "task_outcome": self.task_outcome,
            "failure_reason": self.failure_reason,
            "invalid_kind": self.invalid_kind,
            "push_applied": self.push_applied,
            "disturbance_start": self.disturbance_start,
            "disturbance_end": self.disturbance_end,
            **self.disturbance_metadata,
            "TD0_time": self.td0_time,
            "certificate_valid": self.certificate_valid,
            "Nmin": self.n_min,
            "margin": self.margin,
            "margin_group": self.margin_group,
            "CERT_AFTER_RECOVERY": self.cert_after_recovery,
            **recovery,
            "Krec": krec,
            "Krec_physical": krec_physical,
            "nTD0": n_td0,
            "scheduled": 1,
            "executed": 1,
            "valid": int(self.task_outcome != "INVALID"),
            "eval_invalid": int(eval_invalid),
            "cert_invalid": int(cert_invalid),
            **quality,
        }


def margin_group(margin: float | None) -> str | None:
    if margin is None or not 0.0 <= margin <= 0.95:
        return None
    if margin < 0.3167:
        return "LOW"
    if margin < 0.6333:
        return "MEDIUM"
    return "HIGH"


def trajectory_metrics(frames: list[dict[str, Any]], *, completed: bool) -> dict[str, Any]:
    if not completed or not frames:
        return {"RMSE_v": None, "RMSE_vx": None, "RMSE_vy": None,
                "valid_duration": None, "missing_sample_count": 0}
    times = np.asarray([frame["timestamp"] for frame in frames], dtype=np.float64)
    velocity = np.asarray([frame["com_velocity_heading"][:2] for frame in frames], dtype=np.float64)
    command = np.asarray([frame["command"][:2] for frame in frames], dtype=np.float64)
    error = velocity - command
    ex2, ey2 = error[:, 0] ** 2, error[:, 1] ** 2
    if len(times) > 1:
        duration = float(times[-1] - times[0])
        mx = float(np.trapz(ex2, times) / duration) if duration > 0 else float(ex2[-1])
        my = float(np.trapz(ey2, times) / duration) if duration > 0 else float(ey2[-1])
        gaps = np.diff(times)
        missing = int(np.sum(np.maximum(np.rint(gaps / 0.02).astype(int) - 1, 0)))
    else:
        duration, mx, my, missing = 0.0, float(ex2[0]), float(ey2[0]), 0
    return {"RMSE_v": math.sqrt(mx + my), "RMSE_vx": math.sqrt(mx), "RMSE_vy": math.sqrt(my),
            "valid_duration": duration, "missing_sample_count": missing}
