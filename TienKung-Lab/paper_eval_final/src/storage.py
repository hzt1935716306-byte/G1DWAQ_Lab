"""Append-only trial shards, resume validation, and completion seals."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import (EVALUATION_SYSTEM, ROOT, atomic_write, digest, protocol_bundle,
                     read_json, sha256_file, within, write_csv, write_json)


IDENTITY_GATE_FIELDS = (
    "protocol_version", "stage", "experiment_id", "manifest_hash",
    "metrics_config_hash", "physics_profile_hash", "evaluation_code_commit",
    "asset_hash", "simulator_version",
)

TRIAL_COLUMNS = [
    "trial_id", "experiment_id", "stage", "sequence", "repeat_id", "condition_id",
    "candidate_phase", "sampling_target_hint", "method", "model_id", "train_seed", "eval_seed",
    "checkpoint_sha256", "evaluation_code_commit", "protocol_hash", "manifest_hash",
    "slope_deg", "terrain_id", "friction", "command_vx", "command_vy", "command_yaw", "speed_group",
    "disturbance", "disturbance_start", "disturbance_end", "target_link", "force_vector", "torque_vector",
    "root_velocity_before", "root_velocity_after", "com_velocity_before", "com_velocity_after", "push_applied",
    "TD0_time", "certificate_valid", "Nmin", "margin", "margin_raw", "margin_storage_dtype",
    "margin_was_rounded", "margin_group", "margin_boundary_id", "sampling_eligible",
    "sampling_accepted", "sampling_rejection_reason", "certificate_diagnostic",
    "nmin_q1", "nmin_q2", "sample_role", "sample_roles", "calibration_selected",
    "pilot_analysis_selected", "formal_analysis_selected", "calibration_only", "margin_degenerate",
    "calibration_manifest_hash", "evaluation_manifest_hash", "certificate_calculation", "CERT_AFTER_RECOVERY",
    "termination_kind", "task_outcome", "failure_reason", "reset_time", "out_of_area", "pre_reset_snapshot_path",
    "first_entry", "final_entry", "confirmations", "relapse_events", "recovery_time", "Krec", "Krec_physical",
    "nTD0", "recovered_sustained", "survived_post_observation", "observation_end",
    "RMSE_v", "RMSE_vx", "RMSE_vy", "valid_duration", "missing_sample_count",
    "scheduled", "executed", "valid", "eval_invalid", "cert_invalid",
]


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "evaluation_system", "trial_id", "experiment_id", "stage", "method", "train_seed", "eval_seed",
        "checkpoint_sha256", "evaluation_code_commit", "protocol_version", "protocol_hash", "manifest_hash",
        "metrics_config_hash", "physics_profile_hash", "asset_hash", "simulator_version",
        "termination_kind", "task_outcome", "failure_reason", "push_applied", "scheduled", "executed",
        "valid", "eval_invalid", "cert_invalid",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"trial record missing fields: {sorted(missing)}")
    current_version = protocol_bundle()[2]["protocol_version"]
    if record["evaluation_system"] != EVALUATION_SYSTEM or record["protocol_version"] != current_version:
        raise ValueError(f"legacy/non-v{current_version} trial record refused")
    if record["termination_kind"] not in {"HORIZON_REACHED", "PHYSICAL_RESET", "PROTOCOL_STOP", "EVALUATION_ABORT"}:
        raise ValueError("unknown termination kind")
    if record["task_outcome"] not in {"SUCCESS", "FAILURE", "INVALID"}:
        raise ValueError("unknown task outcome")
    if record["task_outcome"] == "INVALID" and record["valid"]:
        raise ValueError("invalid outcome cannot be performance-valid")
    if record["eval_invalid"] and record["task_outcome"] != "INVALID":
        raise ValueError("EVAL_INVALID must have INVALID task outcome")
    if record.get("recovery_time") is not None and not record.get("recovered_sustained"):
        raise ValueError("Trec is defined only for sustained recovery")
    if record["experiment_id"] == 1 and record.get("margin_raw") is not None:
        if record.get("margin_storage_dtype") != "float64" or record.get("margin_was_rounded") is not False:
            raise ValueError("Experiment 1 raw margin must be unrounded float64")
        if not np.isfinite(float(record["margin_raw"])):
            raise ValueError("Experiment 1 raw margin must be finite when recorded")
    if record.get("margin_group") is not None:
        if record.get("margin_group") not in {"LOWER", "MIDDLE", "UPPER"}:
            raise ValueError("unknown margin group")
        if (record.get("nmin_q1") is None or record.get("nmin_q2") is None
                or not float(record["nmin_q1"]) < float(record["nmin_q2"])):
            raise ValueError("margin groups require the valid q1/q2 pair for their Nmin")
        if not record.get("margin_boundary_id") or not record.get("calibration_manifest_hash"):
            raise ValueError("margin group provenance is incomplete")


class RunStore:
    def __init__(self, path: str | Path, identity: dict[str, Any]):
        self.path = Path(path).resolve()
        results_root = (ROOT / "results").resolve()
        if not within(self.path, results_root):
            raise ValueError("new evaluator can write only under paper_eval_final/results")
        self.records = self.path / "records"
        self.traces = self.path / "traces"
        self.events = self.path / "events"
        self.snapshots = self.path / "pre_reset_snapshots"
        for folder in (self.records, self.traces, self.events, self.snapshots):
            folder.mkdir(parents=True, exist_ok=True)
        identity_path = self.path / "identity.json"
        if identity_path.exists():
            if read_json(identity_path) != identity:
                raise ValueError("existing shard identity differs; use a new shard")
        else:
            write_json(identity_path, identity)
        self.identity = identity

    def completed_ids(self) -> set[str]:
        completed = set()
        for path in self.records.glob("*.json"):
            try:
                record = read_json(path)
                validate_record(record)
                if record["trial_id"] != path.stem:
                    raise ValueError("record filename/trial ID mismatch")
                completed.add(record["trial_id"])
            except Exception:
                # A malformed record is quarantined by validation and never silently overwritten.
                raise ValueError(f"malformed terminal record requires manual review: {path}")
        return completed

    def save_trial(self, record: dict[str, Any], frames: list[dict[str, Any]], events: list[dict[str, Any]],
                   pre_reset_snapshot: dict[str, Any] | None = None) -> None:
        record = {**self.identity, **record}
        validate_record(record)
        trial_id = record["trial_id"]
        target = self.records / f"{trial_id}.json"
        if target.exists():
            if read_json(target) != record:
                raise ValueError(f"terminal record is immutable: {trial_id}")
            return
        arrays: dict[str, Any] = {}
        if frames:
            keys = sorted(set.intersection(*(set(frame) for frame in frames)))
            for key in keys:
                try:
                    arrays[key] = np.asarray([frame[key] for frame in frames])
                except (TypeError, ValueError):
                    continue
        trace_path = self.traces / f"{trial_id}.npz"
        np.savez_compressed(trace_path, **arrays)
        event_path = self.events / f"{trial_id}.json"
        write_json(event_path, events)
        record["trace_path"] = trace_path.relative_to(self.path).as_posix()
        record["trace_sha256"] = sha256_file(trace_path)
        record["events_path"] = event_path.relative_to(self.path).as_posix()
        record["events_sha256"] = sha256_file(event_path)
        if pre_reset_snapshot is not None:
            snapshot_path = self.snapshots / f"{trial_id}.npz"
            np.savez_compressed(snapshot_path, **{key: np.asarray(value) for key, value in pre_reset_snapshot.items()})
            record["pre_reset_snapshot_path"] = snapshot_path.relative_to(self.path).as_posix()
            record["pre_reset_snapshot_sha256"] = sha256_file(snapshot_path)
        else:
            record["pre_reset_snapshot_path"] = None
        write_json(target, record)

    def load_records(self) -> list[dict[str, Any]]:
        rows = [read_json(path) for path in sorted(self.records.glob("*.json"))]
        for row in rows:
            validate_record(row)
            for name in ("trace", "events"):
                if sha256_file(self.path / row[f"{name}_path"]) != row[f"{name}_sha256"]:
                    raise ValueError(f"{row['trial_id']}: {name} integrity mismatch")
            if row.get("pre_reset_snapshot_path") and sha256_file(self.path / row["pre_reset_snapshot_path"]) != row["pre_reset_snapshot_sha256"]:
                raise ValueError(f"{row['trial_id']}: pre-reset snapshot integrity mismatch")
        return rows

    def seal(self, manifest_trial_ids: Iterable[str], summary: dict[str, Any]) -> dict[str, Any]:
        rows = self.load_records()
        designated = set(manifest_trial_ids)
        recorded = {row["trial_id"] for row in rows}
        if recorded != designated:
            raise ValueError(f"cannot seal incomplete shard: missing={len(designated-recorded)}, extra={len(recorded-designated)}")
        write_csv(self.path / "trials.csv", rows, TRIAL_COLUMNS)
        write_json(self.path / "summary.json", summary)
        completion = {
            "status": "COMPLETE", "trial_count": len(rows),
            "identity_sha256": sha256_file(self.path / "identity.json"),
            "summary_sha256": sha256_file(self.path / "summary.json"),
            "csv_sha256": sha256_file(self.path / "trials.csv"),
            "record_hash": digest({row["trial_id"]: sha256_file(self.records / f"{row['trial_id']}.json") for row in rows}),
        }
        write_json(self.path / "completion.json", completion)
        return completion


def compatibility_gate(identities: list[dict[str, Any]]) -> dict[str, Any]:
    if not identities:
        raise ValueError("no shards to aggregate")
    for identity in identities:
        if identity.get("evaluation_system") != EVALUATION_SYSTEM:
            raise ValueError("legacy/new-final merge refused")
    expected = {field: identities[0].get(field) for field in IDENTITY_GATE_FIELDS}
    for identity in identities[1:]:
        for field, value in expected.items():
            if identity.get(field) != value:
                raise ValueError(f"REFUSE TO MERGE: {field} mismatch")
    return expected
