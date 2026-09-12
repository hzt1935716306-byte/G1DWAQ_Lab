"""Deterministic, model-independent common-random-number manifests."""
from __future__ import annotations

from collections import Counter
import copy
import math
from pathlib import Path
import random
from typing import Any, Iterable

from .common import ROOT, canonical_bytes, digest, git_last_commit, load_yaml, protocol_bundle, read_json, write_json


STAGES = ("screening", "pilot", "formal")
EXPERIMENTS = (1, 2, 3)


def _sample(rng: random.Random, bounds: list[float], *, high_open: bool = False) -> float:
    low, high = map(float, bounds)
    value = low + (high - low) * rng.random()
    if high_open:
        value = min(value, math.nextafter(high, low))
    return value


def _balanced_counts(cell_count: int, total: int, seed: int) -> list[int]:
    quotient, remainder = divmod(total, cell_count)
    counts = [quotient] * cell_count
    indices = list(range(cell_count))
    random.Random(seed).shuffle(indices)
    for index in indices[:remainder]:
        counts[index] += 1
    return counts


def _initial_state(rng: random.Random, protocol: dict[str, Any]) -> dict[str, Any]:
    cfg = protocol["initial_state"]
    return {
        "initial_pose": {key: _sample(rng, bounds) for key, bounds in cfg["pose"].items()},
        "initial_velocity": {key: _sample(rng, bounds) for key, bounds in cfg["velocity"].items()},
        "initial_joint_position_scale": [
            _sample(rng, cfg["joint_position_scale"]) for _ in range(cfg["joint_count"])
        ],
        "initial_joint_velocity": [
            _sample(rng, cfg["joint_velocity"]) for _ in range(cfg["joint_count"])
        ],
    }


def _base_trial(
    protocol: dict[str, Any], stage: str, experiment: int, sequence: int,
    slope: float, speed_group: str, condition_id: str, repeat: int,
) -> dict[str, Any]:
    namespace = int(protocol["stages"][stage]["seed_namespace"])
    eval_seed = namespace + experiment * 1_000_000 + sequence
    rng = random.Random(eval_seed)
    command_vx = _sample(rng, protocol["conditions"]["speed_groups"][speed_group], high_open=True)
    material = protocol["physics"]["material"]
    return {
        "trial_id": f"v1_2-{stage}-e{experiment}-{sequence:06d}",
        "condition_id": condition_id,
        "experiment_id": experiment,
        "stage": stage,
        "sequence": sequence,
        "repeat_id": repeat,
        "eval_seed": eval_seed,
        "terrain_id": f"plane_{slope:+g}deg",
        "slope_deg": float(slope),
        "speed_group": speed_group,
        "command_vx": command_vx,
        "command_vy": float(protocol["conditions"]["command_vy"]),
        "command_yaw": float(protocol["conditions"]["command_yaw"]),
        "friction": {
            "static": _sample(rng, material["static_friction_range"]),
            "dynamic": _sample(rng, material["dynamic_friction_range"]),
            "restitution": _sample(rng, material["restitution_range"]),
        },
        **_initial_state(rng, protocol),
    }


def _velocity_jump(rng: random.Random, protocol: dict[str, Any], group: str, direction: int) -> dict[str, Any]:
    cfg = protocol["disturbances"]["velocity_jump"][group]
    axis_index = rng.randrange(3)
    axis = protocol["disturbances"]["velocity_jump"]["angular_axes"][axis_index]
    return {
        "type": "velocity_jump",
        "intensity_group": group,
        "direction_Hpush_deg": int(direction),
        "linear_speed_mps": _sample(rng, cfg["linear_speed_mps"], high_open=True),
        "angular_speed_radps": _sample(rng, cfg["angular_speed_radps"], high_open=True),
        "angular_axis": axis,
        "angular_sign": -1 if rng.randrange(2) else 1,
        "duration_s": 0.0,
        "target_link": "root",
    }


def _wrench_sample(rng: random.Random, cfg: dict[str, Any], direction: int, sample_index: int) -> dict[str, Any]:
    axes = ("roll", "pitch", "yaw")
    # Rotation by sample index makes axes/signs exactly balanced over repeated cells.
    return {
        "force_norm_N": _sample(rng, cfg["force_norm_N"], high_open=True),
        "torque_norm_Nm": _sample(rng, cfg["torque_norm_Nm"], high_open=True),
        "direction_Hpush_deg": int(direction),
        "torque_axis": axes[sample_index % len(axes)],
        "torque_sign": -1 if (sample_index // len(axes)) % 2 else 1,
    }


def _wrench(rng: random.Random, protocol: dict[str, Any], kind: str, direction: int, repeat: int) -> dict[str, Any]:
    cfg = protocol["disturbances"][kind]
    count = 3 if kind == "low_frequency_random_wrench" else 1
    return {
        "type": kind,
        "intensity_group": "CONTINUOUS",
        "direction_Hpush_deg": int(direction),
        "duration_s": float(cfg["duration_s"]),
        "target_link": cfg["target_link"],
        "updates": [_wrench_sample(rng, cfg, direction, repeat + index) for index in range(count)],
    }


def _onset(rng: random.Random, protocol: dict[str, Any]) -> tuple[float, float]:
    # The sample is continuous. Realization is the first 200 Hz boundary at or after it.
    offset = _sample(rng, protocol["timing"]["onset_offset_s"], high_open=True)
    planned = float(protocol["timing"]["warmup_s"]) + offset
    return offset, planned


def generate_trials(stage: str, experiment: int, protocol: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if stage not in STAGES or experiment not in EXPERIMENTS:
        raise ValueError("Unknown stage or experiment")
    protocol = protocol or protocol_bundle()[0]
    slopes = protocol["conditions"]["slopes_deg"]
    speeds = tuple(protocol["conditions"]["speed_groups"])
    directions = protocol["disturbances"]["directions_Hpush_deg"]
    rows: list[dict[str, Any]] = []

    if experiment == 2:
        key = "experiment2_per_cell" if stage == "screening" else "experiment2_per_slope_speed"
        per_cell = int(protocol["budgets"][stage][key])
        for slope in slopes:
            for speed in speeds:
                condition = f"s{slope:+g}_{speed}"
                for repeat in range(per_cell):
                    row = _base_trial(protocol, stage, experiment, len(rows), slope, speed, condition, repeat)
                    row.update(disturbance={"type": "none", "duration_s": 0.0}, horizon_s=float(protocol["timing"]["experiment2_horizon_s"]))
                    rows.append(row)
        expected = 20 if stage == "screening" else 500 if stage == "pilot" else 2000
    else:
        families: list[tuple[str, list[tuple[float, str, str, int]]]] = []
        velocity_cells = [(s, v, impulse, d) for s in slopes for v in speeds for impulse in ("LOW", "HIGH") for d in directions]
        families.append(("velocity_jump", velocity_cells))
        if experiment == 3:
            wrench_cells = [(s, v, "CONTINUOUS", d) for s in slopes for v in speeds for d in directions]
            families.extend((kind, wrench_cells) for kind in ("low_frequency_random_wrench", "constant_wrench"))
        for family_index, (family, cells) in enumerate(families):
            if stage == "screening":
                total = len(cells)
            elif stage == "pilot":
                total = 500 if family == "velocity_jump" else 250
            else:
                total = 2000 if family == "velocity_jump" else 1000
            counts = _balanced_counts(len(cells), total, int(protocol["stages"][stage]["seed_namespace"]) + experiment * 100 + family_index)
            for cell_index, ((slope, speed, intensity, direction), count) in enumerate(zip(cells, counts)):
                condition = f"{family}_s{slope:+g}_{speed}_{intensity}_d{direction}"
                for repeat in range(count):
                    row = _base_trial(protocol, stage, experiment, len(rows), slope, speed, condition, repeat)
                    rng = random.Random(row["eval_seed"] ^ 0xA17E5)
                    offset, planned = _onset(rng, protocol)
                    disturbance = (_velocity_jump(rng, protocol, intensity, direction) if family == "velocity_jump"
                                   else _wrench(rng, protocol, family, direction, repeat + cell_index))
                    row.update(onset_offset_s=offset, planned_disturbance_start_s=planned,
                               disturbance=disturbance,
                               observation_end_s=planned + float(disturbance["duration_s"]) + 10.0)
                    rows.append(row)
        if experiment == 1:
            expected = 80 if stage == "screening" else 500 if stage == "pilot" else 2000
        else:
            expected = 160 if stage == "screening" else 1000 if stage == "pilot" else 4000
    if len(rows) != expected or len({row["trial_id"] for row in rows}) != expected:
        raise AssertionError(f"Experiment {experiment}/{stage}: expected {expected}, generated {len(rows)}")
    return rows


def manifest_payload(stage: str, experiment: int, protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol, _, identity = protocol_bundle() if protocol is None else (protocol, {}, {
        "protocol_version": str(protocol["protocol_version"]), "protocol_hash": digest(protocol)})
    trials = generate_trials(stage, experiment, protocol)
    base = {
        "evaluation_system": "paper_eval_final",
        "protocol_version": identity["protocol_version"],
        "protocol_hash": identity["protocol_hash"],
        # Model-registry/report-only commits must not change a model-independent
        # common-random-number manifest.  Record the last generator revision.
        "generator_code_commit": git_last_commit(Path(__file__)),
        "stage": stage,
        "experiment": experiment,
        "manifest_seed": int(protocol["stages"][stage]["seed_namespace"]),
        "trial_count": len(trials),
        "trials": trials,
    }
    base["manifest_sha256"] = digest(base)
    return base


def validate_manifest(payload: dict[str, Any]) -> None:
    claimed = payload.get("manifest_sha256")
    base = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if claimed != digest(base):
        raise ValueError("Manifest hash mismatch")
    if payload.get("evaluation_system") != "paper_eval_final":
        raise ValueError("Legacy/non-final manifest refused")
    trials = payload.get("trials", [])
    if payload.get("trial_count") != len(trials) or len({row["trial_id"] for row in trials}) != len(trials):
        raise ValueError("Manifest count or trial IDs are invalid")
    if any(row.get("stage") != payload["stage"] or row.get("experiment_id") != payload["experiment"] for row in trials):
        raise ValueError("Manifest trial identity mismatch")


def manifest_path(stage: str, experiment: int) -> Path:
    return ROOT / "manifests" / stage / f"experiment{experiment}.json"


def prepare_manifest(stage: str, experiment: int) -> dict[str, Any]:
    target = manifest_path(stage, experiment)
    wanted = manifest_payload(stage, experiment)
    if target.exists():
        current = read_json(target)
        validate_manifest(current)
        if current != wanted:
            raise ValueError(f"Prepared manifest is immutable and differs: {target}")
        return current
    write_json(target, wanted)
    if stage == "formal":
        target.chmod(0o444)
    return wanted


def prepare_all(stages: Iterable[str] = ("pilot", "formal"), experiments: Iterable[int] = EXPERIMENTS) -> dict[str, str]:
    result = {}
    for stage in stages:
        for experiment in experiments:
            payload = prepare_manifest(stage, int(experiment))
            result[f"{stage}/experiment{experiment}"] = payload["manifest_sha256"]
    return result


def smoke_subset(trials: list[dict[str, Any]], experiment: int) -> list[dict[str, Any]]:
    if experiment == 2:
        return [trials[0], next(row for row in trials if row["speed_group"] == "HIGH")]
    if experiment == 1:
        return [trials[0], next(row for row in trials if row["disturbance"]["intensity_group"] == "HIGH")]
    selected = []
    for family in ("velocity_jump", "low_frequency_random_wrench", "constant_wrench"):
        selected.append(next(row for row in trials if row["disturbance"]["type"] == family))
    return selected


def cell_counts(trials: list[dict[str, Any]]) -> Counter:
    return Counter(row["condition_id"] for row in trials)


def equal_effective_deficits(records: list[dict[str, Any]], trials: list[dict[str, Any]]) -> dict[str, int]:
    """Return only EVAL_INVALID replacement deficits, by comparison cell.

    Physical/controller failures are valid observations and never create a
    deficit.  This helper reports the continuation need; it never resamples or
    mutates the fixed-budget manifest.
    """
    target = cell_counts(trials)
    valid_by_cell = Counter(
        row["condition_id"] for row in records
        if row.get("invalid_kind") != "EVAL_INVALID"
    )
    return {cell: max(0, count - valid_by_cell[cell]) for cell, count in target.items()}


def select_stratified_relation_set(records: list[dict[str, Any]], *, stage: str,
                                   manifest_seed: int) -> dict[str, Any]:
    """Outcome-blind accept/reject selection for the nine N-margin cells."""
    if stage not in ("pilot", "formal"):
        raise ValueError("stratified relation set exists only for pilot/formal")
    per_cell = 40 if stage == "pilot" else 160
    layer_cap = 1 if stage == "pilot" else 2
    all_layers = sorted({row["condition_id"] for row in records})
    if len(all_layers) != 80:
        raise ValueError("Experiment 1 relation selection requires all 80 condition layers")
    selected_layers: dict[tuple[int, str], set[str]] = {}
    for n_min in (3, 4, 5):
        for group in ("LOW", "MEDIUM", "HIGH"):
            layers = list(all_layers)
            random.Random(manifest_seed + n_min * 10 + ("LOW", "MEDIUM", "HIGH").index(group)).shuffle(layers)
            selected_layers[(n_min, group)] = set(layers[:40]) if stage == "pilot" else set(layers)
    accepted: list[dict[str, Any]] = []
    occupancy: Counter = Counter()
    attempts: Counter = Counter()
    for row in sorted(records, key=lambda value: (value.get("sequence", 0), value["trial_id"])):
        n_min, group = row.get("Nmin"), row.get("margin_group")
        if not row.get("certificate_valid") or n_min not in (3, 4, 5) or group not in ("LOW", "MEDIUM", "HIGH"):
            continue
        cell = (int(n_min), str(group))
        attempts[cell] += 1
        if row["condition_id"] not in selected_layers[cell]:
            continue
        layer_key = (cell, row["condition_id"])
        if occupancy[layer_key] >= layer_cap or sum(1 for value in accepted if (value["Nmin"], value["margin_group"]) == cell) >= per_cell:
            continue
        accepted.append(row)
        occupancy[layer_key] += 1
    cells = {}
    for n_min in (3, 4, 5):
        for group in ("LOW", "MEDIUM", "HIGH"):
            cell = (n_min, group)
            rows = [row for row in accepted if (row["Nmin"], row["margin_group"]) == cell]
            cells[f"N{n_min}_{group}"] = {
                "accepted": len(rows), "target": per_cell, "missing": per_cell - len(rows),
                "attempt_count": attempts[cell], "condition_coverage": len({row["condition_id"] for row in rows}),
            }
    return {
        "dataset_role": "STRATIFIED_RELATION_SET", "stage": stage,
        "accepted_trial_ids": [row["trial_id"] for row in accepted], "cells": cells,
        "complete": all(value["missing"] == 0 for value in cells.values()),
    }
