"""Deterministic, model-independent common-random-number manifests."""
from __future__ import annotations

from collections import Counter
import copy
import math
from pathlib import Path
import random
from typing import Any, Iterable

from .common import (ROOT, canonical_bytes, digest, git_last_commit, load_yaml,
                     protocol_bundle, read_json, result_namespace, write_json)


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
    protocol_tag = str(protocol["protocol_version"]).replace(".", "_")
    return {
        "trial_id": f"v{protocol_tag}-{stage}-e{experiment}-{sequence:06d}",
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
            for row in rows:
                row["candidate_phase"] = "INITIAL"
        else:
            expected = 160 if stage == "screening" else 1000 if stage == "pilot" else 4000
    if len(rows) != expected or len({row["trial_id"] for row in rows}) != expected:
        raise AssertionError(f"Experiment {experiment}/{stage}: expected {expected}, generated {len(rows)}")
    return rows


def generate_experiment1_continuation(stage: str, start_sequence: int, count: int,
                                      records: list[dict[str, Any]],
                                      protocol: dict[str, Any] | None = None,
                                      max_candidates_per_condition_layer: int | None = None) -> list[dict[str, Any]]:
    """Generate a deterministic certificate-targeted continuation batch.

    Targeting uses only previous certificate N/margin yield and condition IDs.
    Recovery outcomes are neither read nor passed to the ranking functions.
    """
    if stage not in ("pilot", "formal") or start_sequence < 0 or count < 1:
        raise ValueError("invalid Experiment 1 continuation request")
    protocol = protocol or protocol_bundle()[0]
    initial = generate_trials(stage, 1, protocol)
    templates = {row["condition_id"]: row for row in initial}
    attempts = Counter(str(row["condition_id"]) for row in records)
    assigned: Counter[Any] = Counter()
    assigned_layers: Counter[str] = Counter()
    target_hints: list[Any] = []
    direct_condition_layers = False
    if stage == "pilot":
        from .experiment1_sampling import (
            MARGIN_GROUPS, TARGET_NMIN,
            calibration_condition_rankings, experiment1_config, load_boundaries,
            select_calibration_samples, select_pilot_analysis_samples,
        )
        config = experiment1_config()
        calibration_cfg = config["calibration_sampling"]
        calibration = select_calibration_samples(
            records, per_nmin=int(calibration_cfg["certificate_valid_per_Nmin"])
        )
        if not calibration["complete"]:
            top_layers = int(calibration_cfg["targeting_top_condition_layers"])
            deficits = {n: int(calibration["deficits"][str(n)]) for n in TARGET_NMIN}
            rankings = calibration_condition_rankings(records)
            candidate_phase = "CALIBRATION_TARGETED_CONTINUATION"
        else:
            boundaries = load_boundaries(require_frozen=True)
            analysis_cfg = config["pilot_analysis_sampling"]
            selection = select_pilot_analysis_samples(
                records, boundaries, per_cell=int(analysis_cfg["certificate_valid_per_cell"])
            )
            deficits = {
                (n, group, layer): int(missing)
                for n in TARGET_NMIN for group in MARGIN_GROUPS
                for layer, missing in selection["cells"][f"N{n}_{group}"]["missing_by_condition_layer"].items()
            }
            direct_condition_layers = True
            candidate_phase = "PILOT_LAYER_TARGETED_CONTINUATION"
        for _ in range(count):
            deficient = [
                key for key, deficit in deficits.items() if deficit > 0
                and (not direct_condition_layers or max_candidates_per_condition_layer is None
                     or attempts[key[2]] + assigned_layers[key[2]] < max_candidates_per_condition_layer)
            ]
            if not deficient:
                break
            hint = max(deficient, key=lambda key: (
                deficits[key] / (assigned[key] + 1.0), key,
            ))
            target_hints.append(hint)
            assigned[hint] += 1
            if direct_condition_layers:
                assigned_layers[hint[2]] += 1
    else:
        from .experiment1_sampling import (
            MARGIN_GROUPS, TARGET_NMIN, experiment1_config,
            load_boundaries, select_formal_samples,
        )
        boundaries = load_boundaries(require_frozen=True)
        sampling_cfg = experiment1_config()["formal_sampling"]
        target = int(sampling_cfg["certificate_valid_per_cell"])
        selection = select_formal_samples(records, boundaries, per_cell=target)
        deficits = {
            (n, group, layer): int(missing)
            for n in TARGET_NMIN for group in MARGIN_GROUPS
            for layer, missing in selection["cells"][f"N{n}_{group}"]["missing_by_condition_layer"].items()
        }
        direct_condition_layers = True
        candidate_phase = "FORMAL_LAYER_TARGETED_CONTINUATION"
        for _ in range(count):
            deficient = [
                key for key, deficit in deficits.items() if deficit > 0
                and (max_candidates_per_condition_layer is None
                     or attempts[key[2]] + assigned_layers[key[2]] < max_candidates_per_condition_layer)
            ]
            if not deficient:
                break
            hint = max(deficient, key=lambda key: (
                deficits[key] / (assigned[key] + 1.0), key,
            ))
            target_hints.append(hint)
            assigned[hint] += 1
            assigned_layers[hint[2]] += 1

    layer_cursor: Counter[Any] = Counter()
    planned_by_layer: Counter[str] = Counter()
    rows = []
    for offset, hint in enumerate(target_hints):
        if direct_condition_layers:
            condition_id = str(hint[2])
        else:
            ranked_layers = [
                layer for layer in rankings[hint]
                if max_candidates_per_condition_layer is None
                or attempts[layer] + planned_by_layer[layer] < max_candidates_per_condition_layer
            ][:top_layers]
            if not ranked_layers:
                break
            condition_id = ranked_layers[layer_cursor[hint] % len(ranked_layers)]
            layer_cursor[hint] += 1
        template = templates[condition_id]
        sequence = start_sequence + offset
        repeat = attempts[condition_id] + planned_by_layer[condition_id]
        planned_by_layer[condition_id] += 1
        row = _base_trial(
            protocol, stage, 1, sequence, float(template["slope_deg"]),
            str(template["speed_group"]), condition_id, repeat,
        )
        rng = random.Random(row["eval_seed"] ^ 0xA17E5)
        disturbance_template = template["disturbance"]
        onset_offset, planned = _onset(rng, protocol)
        disturbance = _velocity_jump(
            rng, protocol, str(disturbance_template["intensity_group"]),
            int(disturbance_template["direction_Hpush_deg"]),
        )
        hint_payload = ({"Nmin": int(hint)} if isinstance(hint, int) else {
            "Nmin": int(hint[0]), "margin_group": str(hint[1]),
            "condition_id": condition_id,
        })
        row.update(
            onset_offset_s=onset_offset,
            planned_disturbance_start_s=planned,
            disturbance=disturbance,
            observation_end_s=planned + 10.0,
            candidate_phase=candidate_phase,
            sampling_target_hint=hint_payload,
        )
        rows.append(row)
    return rows


def generate_experiment1_coverage_probe(
    protocol: dict[str, Any] | None = None, *, count: int | None = None,
) -> list[dict[str, Any]]:
    """Refuse the superseded shared-boundary N4-HIGH coverage probe."""
    raise ValueError("N4-HIGH coverage probing was retired with shared margin boundaries")
    protocol = protocol or protocol_bundle()[0]
    cfg = load_yaml(ROOT / "configs/experiment1.yaml")["coverage_probe"]
    conditions = list(cfg["conditions"])
    total = int(cfg["candidate_count"] if count is None else count)
    counts = _balanced_counts(len(conditions), total, 234_20260912)
    rows = []
    for condition_index, (condition, condition_count) in enumerate(zip(conditions, counts)):
        for repeat in range(condition_count):
            sequence = len(rows)
            condition_id = f"n4_high_probe_{condition['name']}"
            row = _base_trial(
                protocol, "pilot", 1, sequence, float(condition["slope_deg"]),
                str(condition["speed_group"]), condition_id, repeat,
            )
            rng = random.Random(row["eval_seed"] ^ 0xC04E4A6E)
            row["trial_id"] = f"v1_2-pilot-e1-n4probe-{sequence:06d}"
            row["command_vx"] = _sample(rng, condition["command_vx"], high_open=True)
            onset_offset, planned = _onset(rng, protocol)
            disturbance = {
                "type": "velocity_jump",
                "intensity_group": "N4_HIGH_COVERAGE_PROBE",
                "direction_Hpush_deg": int(condition["direction_deg"]),
                "linear_speed_mps": _sample(rng, condition["linear_speed_mps"], high_open=True),
                "angular_speed_radps": _sample(rng, condition["angular_speed_radps"], high_open=True),
                "angular_axis": protocol["disturbances"]["velocity_jump"]["angular_axes"][rng.randrange(3)],
                "angular_sign": -1 if rng.randrange(2) else 1,
                "duration_s": 0.0,
                "target_link": "root",
            }
            row.update(
                onset_offset_s=onset_offset,
                planned_disturbance_start_s=planned,
                disturbance=disturbance,
                observation_end_s=planned + 10.0,
                candidate_phase="N4_HIGH_COVERAGE_PROBE",
                sampling_target_hint={"Nmin": 4, "margin_group": "HIGH"},
                analysis_excluded=True,
                probe_condition_index=condition_index,
            )
            rows.append(row)
    return rows


def coverage_probe_manifest(
    protocol: dict[str, Any] | None = None, *, count: int | None = None,
) -> dict[str, Any]:
    protocol, _, identity = protocol_bundle() if protocol is None else (
        protocol, {}, {"protocol_version": str(protocol["protocol_version"]), "protocol_hash": digest(protocol)}
    )
    trials = generate_experiment1_coverage_probe(protocol, count=count)
    payload = {
        "evaluation_system": "paper_eval_final",
        "protocol_version": identity["protocol_version"],
        "protocol_hash": identity["protocol_hash"],
        "generator_code_commit": git_last_commit(Path(__file__)),
        "stage": "pilot",
        "experiment": 1,
        "manifest_seed": 234_20260912,
        "trial_count": len(trials),
        "dataset_role": "COVERAGE_PROBE_NONANALYSIS",
        "analysis_excluded": True,
        "trials": trials,
    }
    payload["manifest_sha256"] = digest(payload)
    return payload


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
    return ROOT / "manifests" / stage / result_namespace() / f"experiment{experiment}.json"


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
    """Compatibility entry point for the revised outcome-blind relation set.

    ``manifest_seed`` remains in the signature for compatibility, but is
    intentionally unused: selection is now fixed by candidate sequence and the
    independent pilot boundary artifact.
    """
    del manifest_seed
    from .experiment1_sampling import (
        TARGET_NMIN, experiment1_config, load_boundaries, select_calibration_samples,
        select_formal_samples, select_pilot_analysis_samples,
    )

    boundaries = load_boundaries()
    cfg = experiment1_config()
    if stage == "formal":
        return select_formal_samples(
            records, boundaries,
            per_cell=int(cfg["formal_sampling"]["certificate_valid_per_cell"]),
        )
    if stage != "pilot":
        raise ValueError("relation set exists only for pilot/formal")
    if boundaries.get("status") == "FROZEN":
        return select_pilot_analysis_samples(
            records, boundaries,
            per_cell=int(cfg["pilot_analysis_sampling"]["certificate_valid_per_cell"]),
        )
    calibration = select_calibration_samples(
        records, per_nmin=int(cfg["calibration_sampling"]["certificate_valid_per_Nmin"])
    )
    return {
        "dataset_role": "BALANCED_MARGIN_CALIBRATION",
        "stage": "pilot", "boundary_status": boundaries.get("status"),
        "accepted_trial_ids": [
            row["trial_id"] for n in TARGET_NMIN for row in calibration["selected"][n]
        ],
        "Nmin_counts": calibration["counts"], "Nmin_deficits": calibration["deficits"],
        "cells": {}, "complete": False,
    }
