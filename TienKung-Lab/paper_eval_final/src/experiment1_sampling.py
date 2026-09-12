"""Shared-margin calibration and outcome-blind Experiment 1 sampling."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .common import ROOT, atomic_write, digest, load_yaml, protocol_bundle, write_csv, write_json


TARGET_NMIN = (2, 3, 4)
MARGIN_GROUPS = ("LOW", "MEDIUM", "HIGH")
BOUNDARY_PATH = ROOT / "configs/experiment1_margin_boundaries.yaml"


def experiment1_config() -> dict[str, Any]:
    return load_yaml(ROOT / "configs/experiment1.yaml")


def load_boundaries(*, require_frozen: bool = False) -> dict[str, Any]:
    value = load_yaml(BOUNDARY_PATH)
    if require_frozen and value.get("status") != "FROZEN":
        raise ValueError("Experiment 1 shared margin boundaries are not frozen")
    if value.get("status") != "FROZEN":
        return value

    claimed = value.get("boundary_id")
    calculated = digest({key: item for key, item in value.items() if key != "boundary_id"})
    if claimed != calculated:
        raise ValueError("Experiment 1 frozen shared-margin boundary hash mismatch")
    cfg = experiment1_config()
    if tuple(int(n) for n in value.get("target_Nmin", [])) != TARGET_NMIN:
        raise ValueError("Experiment 1 frozen boundaries target a superseded Nmin scope")
    source = value.get("source_identity", {})
    current_protocol_hash = protocol_bundle()[2]["protocol_hash"]
    if value.get("source_stage") != "pilot" or source.get("stage") != "pilot":
        raise ValueError("Experiment 1 boundaries must come from the independent pilot stage")
    if source.get("protocol_hash") != current_protocol_hash:
        raise ValueError("Experiment 1 boundaries belong to a different protocol revision")
    if value.get("baseline_id") != cfg["baseline_id"] or value.get("checkpoint_sha256") != cfg["checkpoint_sha256"]:
        raise ValueError("Experiment 1 boundaries belong to a different baseline/checkpoint")
    expected = int(cfg["calibration_sampling"]["certificate_valid_per_Nmin"])
    selected = value.get("calibration_trial_ids", {})
    if any(len(selected.get(str(n), [])) != expected for n in TARGET_NMIN):
        raise ValueError("Experiment 1 boundary artifact does not contain balanced Nmin=2/3/4 calibration IDs")
    q1, q2 = value.get("shared_q1"), value.get("shared_q2")
    if q1 is None or q2 is None or not (
        math.isfinite(float(q1)) and math.isfinite(float(q2)) and float(q1) < float(q2)
    ):
        raise ValueError("Experiment 1 shared q1/q2 are invalid")
    if not value.get("calibration_manifest_hash") or not value.get("computed_at_utc"):
        raise ValueError("Experiment 1 frozen boundary provenance is incomplete")
    return value


def _raw_margin(row: dict[str, Any]) -> float | None:
    value = row.get("margin_raw")
    if value is None:
        return None
    if row.get("margin_storage_dtype") != "float64" or row.get("margin_was_rounded") is not False:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def certificate_sample_eligibility(row: dict[str, Any], *, stage: str) -> tuple[bool, str]:
    """Eligibility intentionally does not inspect any recovery/outcome field."""
    if row.get("stage") != stage:
        return False, "WRONG_STAGE"
    if row.get("invalid_kind") in {"EVAL_INVALID", "UNRESOLVED_INVALID"}:
        return False, str(row["invalid_kind"])
    if not row.get("certificate_valid"):
        return False, "CERT_INVALID"
    if row.get("Nmin") not in TARGET_NMIN:
        return False, "NMIN_OUTSIDE_TARGET"
    if _raw_margin(row) is None:
        return False, "MARGIN_NOT_RAW_FINITE_FLOAT64"
    return True, "ELIGIBLE"


def pilot_eligibility(row: dict[str, Any]) -> tuple[bool, str]:
    return certificate_sample_eligibility(row, stage="pilot")


def _ordered(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: (int(row.get("sequence", 10**18)), str(row.get("trial_id", ""))))


def select_calibration_samples(records: Iterable[dict[str, Any]], *, per_nmin: int = 120) -> dict[str, Any]:
    """Select exactly the first 120 eligible pilot candidates for every Nmin."""
    selected: dict[int, list[dict[str, Any]]] = {n: [] for n in TARGET_NMIN}
    exclusions: Counter[str] = Counter()
    for row in _ordered(records):
        eligible, reason = pilot_eligibility(row)
        if not eligible:
            exclusions[reason] += 1
            continue
        n_min = int(row["Nmin"])
        if len(selected[n_min]) < per_nmin:
            selected[n_min].append(row)
        else:
            exclusions[f"N{n_min}_CALIBRATION_QUOTA_FILLED"] += 1
    counts = {str(n): len(rows) for n, rows in selected.items()}
    return {
        "stage": "pilot",
        "dataset_role": "BALANCED_MARGIN_CALIBRATION",
        "target_per_Nmin": int(per_nmin),
        "selected": selected,
        "selected_trial_ids": {str(n): [row["trial_id"] for row in rows] for n, rows in selected.items()},
        "counts": counts,
        "deficits": {str(n): max(0, per_nmin - len(rows)) for n, rows in selected.items()},
        "exclusions": dict(sorted(exclusions.items())),
        "complete": all(len(rows) == per_nmin for rows in selected.values()),
    }


select_pilot_samples = select_calibration_samples


def _calculation_trace(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": row.get("trial_id"), "eval_seed": row.get("eval_seed"),
        "sequence": row.get("sequence"), "Nmin": row.get("Nmin"),
        "margin_raw": row.get("margin_raw"),
        "certificate_calculation": row.get("certificate_calculation"),
    }


def _shared_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 360:
        raise ValueError("shared margin calibration requires exactly 360 rows")
    margins = np.sort(np.asarray([_raw_margin(row) for row in rows], dtype=np.float64))
    m120, m121 = float(margins[119]), float(margins[120])
    m240, m241 = float(margins[239]), float(margins[240])
    q1, q2 = (m120 + m121) / 2.0, (m240 + m241) / 2.0
    unique = int(np.unique(margins).size)
    calculations = [row.get("certificate_calculation") or {} for row in rows]
    saturation = any(bool(item.get("margin_saturated")) for item in calculations)
    clamp = any(bool(item.get("margin_clamped")) for item in calculations)
    truncation = any(bool(item.get("margin_truncated")) for item in calculations)
    precision_loss = any(
        row.get("margin_storage_dtype") != "float64" or row.get("margin_was_rounded") is not False
        for row in rows
    )
    large_duplicate_block = unique < 324
    reasons = []
    if m120 == m121:
        reasons.append("M120_EQUALS_M121")
    if m240 == m241:
        reasons.append("M240_EQUALS_M241")
    if q1 >= q2:
        reasons.append("Q1_NOT_BELOW_Q2")
    if large_duplicate_block and (saturation or clamp or truncation or precision_loss):
        reasons.append("DUPLICATES_WITH_CLAMP_TRUNCATION_OR_PRECISION_LOSS")
    pooled_q05, pooled_q95 = float(np.quantile(margins, 0.05)), float(np.quantile(margins, 0.95))
    pooled_robust_range = pooled_q95 - pooled_q05
    cfg = experiment1_config().get("margin_variation_guard", {})
    per_nmin_descriptive: dict[str, dict[str, Any]] = {}
    insufficient_variation = []
    for n in TARGET_NMIN:
        values = np.asarray([_raw_margin(row) for row in rows if row["Nmin"] == n], dtype=np.float64)
        q05, q95 = float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))
        robust_range = q95 - q05
        unique_count = int(np.unique(values).size)
        checks = {
            "unique_values": unique_count >= int(cfg.get("minimum_unique_values", 12)),
            "absolute_range": float(np.ptp(values)) >= float(cfg.get("minimum_absolute_range", 1.0e-6)),
            "robust_range": robust_range >= float(cfg.get("minimum_robust_range", 1.0e-4)),
            "relative_robust_range": (
                pooled_robust_range > 0.0
                and robust_range / pooled_robust_range
                >= float(cfg.get("minimum_robust_to_pooled_ratio", 0.01))
            ),
        }
        sufficient = all(checks.values())
        if not sufficient:
            insufficient_variation.append(n)
        per_nmin_descriptive[str(n)] = {
            "count": int(values.size),
            "minimum": float(np.min(values)), "maximum": float(np.max(values)),
            "range": float(np.ptp(values)), "median": float(np.median(values)),
            "q05": q05, "q95": q95, "robust_range_q05_q95": robust_range,
            "robust_to_pooled_ratio": robust_range / pooled_robust_range if pooled_robust_range > 0 else None,
            "different_margin_value_count": unique_count,
            "variation_checks": checks,
            "variation_sufficient": sufficient,
        }
    if insufficient_variation:
        reasons.extend(f"N{n}_MARGIN_VARIATION_INSUFFICIENT" for n in insufficient_variation)
    return {
        "status": "MARGIN_DEGENERATE" if reasons else "VALID",
        "degeneracy_reasons": reasons,
        "count": 360,
        "minimum": float(margins[0]), "maximum": float(margins[-1]),
        "median": float(np.median(margins)),
        "quantile_33_3": q1, "quantile_66_7": q2,
        "order_statistics": {"m120": m120, "m121": m121, "m240": m240, "m241": m241},
        "different_margin_value_count": unique,
        "zero_fraction": float(np.mean(margins == 0.0)),
        "large_duplicate_block": large_duplicate_block,
        "margin_saturation_detected": saturation,
        "margin_clamp_detected": clamp,
        "margin_truncation_detected": truncation,
        "precision_loss_detected": precision_loss,
        "pooled_q05": pooled_q05, "pooled_q95": pooled_q95,
        "pooled_robust_range_q05_q95": pooled_robust_range,
        "margin_variation_guard": cfg,
        "insufficient_variation_Nmin": insufficient_variation,
        "per_Nmin_descriptive": per_nmin_descriptive,
        "raw_margin_and_intermediates": [_calculation_trace(row) for row in rows],
    }


def _calibration_manifest_hash(selection: dict[str, Any], source_identity: dict[str, Any]) -> str:
    ordered = [row for n in TARGET_NMIN for row in selection["selected"][n]]
    return digest({
        "dataset_role": "BALANCED_MARGIN_CALIBRATION",
        "source_manifest_hash": source_identity.get("manifest_hash"),
        "protocol_hash": source_identity.get("protocol_hash"),
        "checkpoint_sha256": source_identity.get("checkpoint_sha256"),
        "samples": [{
            "trial_id": row["trial_id"], "eval_seed": row["eval_seed"],
            "Nmin": row["Nmin"], "margin_raw": _raw_margin(row),
        } for row in ordered],
    })


def fit_pilot_boundaries(records: Iterable[dict[str, Any]], *, source_identity: dict[str, Any],
                         output_path: str | Path = BOUNDARY_PATH) -> dict[str, Any]:
    """Fit one shared q1/q2 pair, or emit a non-frozen degeneracy report."""
    cfg = experiment1_config()
    per_nmin = int(cfg["calibration_sampling"]["certificate_valid_per_Nmin"])
    selection = select_calibration_samples(records, per_nmin=per_nmin)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": 1,
        "boundary_kind": "SHARED_MARGIN_TERTILES",
        "baseline_id": cfg["baseline_id"],
        "checkpoint_sha256": cfg["checkpoint_sha256"],
        "source_stage": "pilot",
        "source_identity": source_identity,
        "calibration_selection_rule": "first eligible candidates by preassigned (sequence, trial_id); outcome-blind",
        "target_Nmin": list(TARGET_NMIN),
        "supersedes_target_Nmin": [3, 4, 5],
        "scope_change_basis": "certificate-valid sample coverage only; recovery outcomes were not inspected",
        "samples_per_Nmin": per_nmin,
        "calibration_trial_ids": selection["selected_trial_ids"],
        "selection_exclusions": selection["exclusions"],
        "shared_q1": None, "shared_q2": None,
        "calibration_manifest_hash": None, "computed_at_utc": None,
        "code_version": {
            "evaluation_code_commit": source_identity.get("evaluation_code_commit"),
            "evaluation_code_hash": source_identity.get("evaluation_code_hash"),
            "protocol_hash": source_identity.get("protocol_hash"),
        },
        "model_version": {
            "model_id": source_identity.get("model_id"),
            "train_seed": source_identity.get("train_seed"),
            "checkpoint_sha256": source_identity.get("checkpoint_sha256"),
        },
    }
    if not selection["complete"]:
        payload.update(status="COLLECTING", deficits=selection["deficits"], diagnostic={})
    else:
        calibration_rows = [row for n in TARGET_NMIN for row in selection["selected"][n]]
        diagnostic = _shared_diagnostic(calibration_rows)
        payload["diagnostic"] = diagnostic
        payload["calibration_manifest_hash"] = _calibration_manifest_hash(selection, source_identity)
        payload["computed_at_utc"] = datetime.now(timezone.utc).isoformat()
        if diagnostic["status"] == "MARGIN_DEGENERATE":
            payload["status"] = "MARGIN_DEGENERATE"
        else:
            payload["shared_q1"] = diagnostic["quantile_33_3"]
            payload["shared_q2"] = diagnostic["quantile_66_7"]
            payload["status"] = "FROZEN"
    payload["boundary_id"] = digest({key: value for key, value in payload.items() if key != "boundary_id"})
    atomic_write(output_path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
    write_json(Path(output_path).with_suffix(".diagnostics.json"), payload)
    if Path(output_path).resolve() == BOUNDARY_PATH.resolve():
        freeze_path = ROOT / "protocol/implementation_freeze.yaml"
        freeze = load_yaml(freeze_path)
        freeze["experiment1_margin_boundaries"] = {
            "file": str(Path(output_path).resolve().relative_to(ROOT)),
            "kind": "SHARED_MARGIN_TERTILES",
            "status": payload["status"],
            "target_Nmin": list(TARGET_NMIN),
            "supersedes_target_Nmin": [3, 4, 5],
            "boundary_id": payload["boundary_id"] if payload["status"] == "FROZEN" else None,
            "calibration_manifest_hash": (
                payload["calibration_manifest_hash"] if payload["status"] == "FROZEN" else None
            ),
        }
        atomic_write(freeze_path, yaml.safe_dump(freeze, allow_unicode=True, sort_keys=False))
    return payload


def classify_margin(margin_raw: float | None, boundaries: dict[str, Any]) -> str | None:
    """Classify margin with the one shared pair; Nmin is deliberately absent."""
    if boundaries.get("status") != "FROZEN" or margin_raw is None:
        return None
    margin = float(margin_raw)
    if not math.isfinite(margin):
        return None
    if margin < float(boundaries["shared_q1"]):
        return "LOW"
    if margin < float(boundaries["shared_q2"]):
        return "MEDIUM"
    return "HIGH"


def _select_analysis(records: Iterable[dict[str, Any]], boundaries: dict[str, Any], *,
                     stage: str, per_cell: int, sample_role: str) -> dict[str, Any]:
    if boundaries.get("status") != "FROZEN":
        raise ValueError("analysis selection requires frozen shared pilot boundaries")
    cells: dict[tuple[int, str], list[str]] = {
        (n, group): [] for n in TARGET_NMIN for group in MARGIN_GROUPS
    }
    attempts: Counter[tuple[int, str]] = Counter()
    exclusions: Counter[str] = Counter()
    accepted_in_order: list[str] = []
    for row in _ordered(records):
        eligible, reason = certificate_sample_eligibility(row, stage=stage)
        if not eligible:
            exclusions[reason] += 1
            continue
        group = classify_margin(_raw_margin(row), boundaries)
        assert group is not None
        cell = (int(row["Nmin"]), group)
        attempts[cell] += 1
        if len(cells[cell]) >= per_cell:
            exclusions[f"N{cell[0]}_{cell[1]}_QUOTA_FILLED"] += 1
            continue
        cells[cell].append(row["trial_id"])
        accepted_in_order.append(row["trial_id"])
    detail = {
        f"N{n}_{group}": {
            "accepted": len(cells[(n, group)]), "target": per_cell,
            "missing": max(0, per_cell - len(cells[(n, group)])),
            "attempt_count": attempts[(n, group)], "trial_ids": cells[(n, group)],
        }
        for n in TARGET_NMIN for group in MARGIN_GROUPS
    }
    complete = all(value["missing"] == 0 for value in detail.values())
    evaluation_manifest_hash = digest({
        "stage": stage, "sample_role": sample_role,
        "boundary_id": boundaries["boundary_id"], "target_per_cell": per_cell,
        "complete": complete, "accepted_trial_ids": accepted_in_order,
    })
    calibration_ids = {
        trial_id for ids in boundaries["calibration_trial_ids"].values() for trial_id in ids
    }
    accepted_ids = set(accepted_in_order)
    assignments = {}
    for row in _ordered(records):
        trial_id = row["trial_id"]
        calibration_member = stage == "pilot" and trial_id in calibration_ids
        analysis_member = trial_id in accepted_ids
        roles = (["calibration"] if calibration_member else []) + ([sample_role] if analysis_member else [])
        raw = _raw_margin(row)
        assignments[trial_id] = {
            "stage": stage,
            "sample_role": sample_role if analysis_member else "calibration" if calibration_member else None,
            "sample_roles": roles,
            "certificate_valid": bool(row.get("certificate_valid")),
            "Nmin": row.get("Nmin"), "margin_raw": row.get("margin_raw"),
            "margin_group": classify_margin(raw, boundaries) if raw is not None else None,
            "shared_q1": boundaries["shared_q1"], "shared_q2": boundaries["shared_q2"],
            "calibration_only": bool(calibration_member and not analysis_member),
            "margin_degenerate": False,
            "trial_id": trial_id, "eval_seed": row.get("eval_seed"),
            "calibration_manifest_hash": boundaries["calibration_manifest_hash"],
            "evaluation_manifest_hash": evaluation_manifest_hash,
        }
    return {
        "dataset_role": "PILOT_ANALYSIS_SET" if stage == "pilot" else "FORMAL_ANALYSIS_SET",
        "stage": stage, "sample_role": sample_role,
        "boundary_id": boundaries["boundary_id"],
        "shared_q1": boundaries["shared_q1"], "shared_q2": boundaries["shared_q2"],
        "calibration_manifest_hash": boundaries["calibration_manifest_hash"],
        "evaluation_manifest_hash": evaluation_manifest_hash,
        "accepted_trial_ids": accepted_in_order,
        "cells": detail, "exclusions": dict(sorted(exclusions.items())),
        "assignments": assignments, "complete": complete,
    }


def select_pilot_analysis_samples(records: Iterable[dict[str, Any]], boundaries: dict[str, Any],
                                  *, per_cell: int = 40) -> dict[str, Any]:
    return _select_analysis(records, boundaries, stage="pilot", per_cell=per_cell,
                            sample_role="pilot_analysis")


def select_formal_samples(records: Iterable[dict[str, Any]], boundaries: dict[str, Any],
                          *, per_cell: int = 160) -> dict[str, Any]:
    return _select_analysis(records, boundaries, stage="formal", per_cell=per_cell,
                            sample_role="formal_analysis")


def write_sampling_assignments(path: str | Path, selection: dict[str, Any]) -> None:
    """Write final derived roles without rewriting immutable raw trial records."""
    target = Path(path)
    write_json(target / "sampling_assignments.json", selection)
    fields = [
        "stage", "sample_role", "sample_roles", "certificate_valid", "Nmin", "margin_raw",
        "margin_group", "shared_q1", "shared_q2", "calibration_only", "margin_degenerate",
        "trial_id", "eval_seed", "calibration_manifest_hash", "evaluation_manifest_hash",
    ]
    write_csv(target / "sampling_assignments.csv", selection["assignments"].values(), fields)


def annotate_sampling_batch(records: list[dict[str, Any]], prior_records: Iterable[dict[str, Any]],
                            *, stage: str, boundaries: dict[str, Any] | None = None) -> None:
    """Attach all known-at-execution sampling fields; final roles live in the sidecar."""
    prior = list(prior_records)
    cfg = experiment1_config()
    if stage == "pilot":
        calibration = select_calibration_samples(
            [*prior, *records],
            per_nmin=int(cfg["calibration_sampling"]["certificate_valid_per_Nmin"]),
        )
        calibration_ids = {trial_id for ids in calibration["selected_trial_ids"].values() for trial_id in ids}
        analysis_ids: set[str] = set()
        analysis_hash = None
        if boundaries is not None and boundaries.get("status") == "FROZEN":
            analysis = select_pilot_analysis_samples(
                [*prior, *records], boundaries,
                per_cell=int(cfg["pilot_analysis_sampling"]["certificate_valid_per_cell"]),
            )
            analysis_ids = set(analysis["accepted_trial_ids"])
            analysis_hash = analysis["evaluation_manifest_hash"]
        for row in records:
            eligible, reason = pilot_eligibility(row)
            calibration_member = row["trial_id"] in calibration_ids
            analysis_member = row["trial_id"] in analysis_ids
            group = classify_margin(_raw_margin(row), boundaries) if boundaries and eligible else None
            roles = (["calibration"] if calibration_member else []) + (["pilot_analysis"] if analysis_member else [])
            chosen = analysis_member if boundaries else calibration_member
            row.update(
                sampling_eligible=eligible, sampling_accepted=chosen,
                sampling_rejection_reason=None if chosen else (
                    "NOT_SELECTED_IN_CURRENT_ROLE" if eligible else reason
                ),
                sample_role="pilot_analysis" if analysis_member else "calibration" if calibration_member else None,
                sample_roles=roles,
                calibration_selected=calibration_member,
                pilot_analysis_selected=analysis_member, formal_analysis_selected=False,
                margin_group=group,
                margin_boundary_id=boundaries.get("boundary_id") if boundaries else None,
                shared_q1=boundaries.get("shared_q1") if boundaries else None,
                shared_q2=boundaries.get("shared_q2") if boundaries else None,
                calibration_only=bool(calibration_member and boundaries and not analysis_member),
                margin_degenerate=bool(boundaries and boundaries.get("status") == "MARGIN_DEGENERATE"),
                calibration_manifest_hash=boundaries.get("calibration_manifest_hash") if boundaries else None,
                evaluation_manifest_hash=analysis_hash,
            )
        return
    if stage != "formal" or boundaries is None:
        return
    analysis = select_formal_samples(
        [*prior, *records], boundaries,
        per_cell=int(cfg["formal_sampling"]["certificate_valid_per_cell"]),
    )
    accepted = set(analysis["accepted_trial_ids"])
    for row in records:
        eligible, reason = certificate_sample_eligibility(row, stage="formal")
        analysis_member = row["trial_id"] in accepted
        group = classify_margin(_raw_margin(row), boundaries) if eligible else None
        row.update(
            sampling_eligible=eligible, sampling_accepted=analysis_member,
            sampling_rejection_reason=None if analysis_member else (
                "ANALYSIS_CELL_QUOTA_FILLED" if eligible else reason
            ),
            sample_role="formal_analysis" if analysis_member else None,
            sample_roles=["formal_analysis"] if analysis_member else [],
            calibration_selected=False, pilot_analysis_selected=False,
            formal_analysis_selected=analysis_member,
            margin_group=group, margin_boundary_id=boundaries["boundary_id"],
            shared_q1=boundaries["shared_q1"], shared_q2=boundaries["shared_q2"],
            calibration_only=False, margin_degenerate=False,
            calibration_manifest_hash=boundaries["calibration_manifest_hash"],
            evaluation_manifest_hash=analysis["evaluation_manifest_hash"],
        )


def sampling_complete(records: Iterable[dict[str, Any]], *, stage: str,
                      boundaries: dict[str, Any] | None = None) -> bool:
    if boundaries is None or boundaries.get("status") != "FROZEN":
        return False
    cfg = experiment1_config()
    if stage == "pilot":
        return select_pilot_analysis_samples(
            records, boundaries,
            per_cell=int(cfg["pilot_analysis_sampling"]["certificate_valid_per_cell"]),
        )["complete"]
    if stage == "formal":
        return select_formal_samples(
            records, boundaries,
            per_cell=int(cfg["formal_sampling"]["certificate_valid_per_cell"]),
        )["complete"]
    return True


def calibration_condition_rankings(records: Iterable[dict[str, Any]]) -> dict[int, list[str]]:
    attempts: Counter[str] = Counter()
    hits: dict[int, Counter[str]] = defaultdict(Counter)
    all_layers: set[str] = set()
    for row in records:
        layer = str(row["condition_id"])
        all_layers.add(layer)
        attempts[layer] += 1
        if row.get("certificate_valid") and row.get("Nmin") in TARGET_NMIN and _raw_margin(row) is not None:
            hits[int(row["Nmin"])][layer] += 1
    return {
        n: sorted(all_layers, key=lambda layer: (
            -(hits[n][layer] + 1.0) / (attempts[layer] + 3.0), attempts[layer], layer,
        )) for n in TARGET_NMIN
    }


pilot_continuation_condition_weights = calibration_condition_rankings


def analysis_condition_rankings(records: Iterable[dict[str, Any]], boundaries: dict[str, Any],
                                *, stage: str) -> dict[tuple[int, str], list[str]]:
    attempts: Counter[str] = Counter()
    hits: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    all_layers: set[str] = set()
    for row in records:
        layer = str(row["condition_id"])
        all_layers.add(layer)
        attempts[layer] += 1
        eligible, _ = certificate_sample_eligibility(row, stage=stage)
        if eligible:
            group = classify_margin(_raw_margin(row), boundaries)
            assert group is not None
            hits[(int(row["Nmin"]), group)][layer] += 1
    return {
        cell: sorted(all_layers, key=lambda layer: (
            -(hits[cell][layer] + 1.0) / (attempts[layer] + 3.0), attempts[layer], layer,
        )) for cell in ((n, group) for n in TARGET_NMIN for group in MARGIN_GROUPS)
    }


def formal_continuation_condition_weights(records: Iterable[dict[str, Any]],
                                          boundaries: dict[str, Any]) -> dict[tuple[int, str], list[str]]:
    return analysis_condition_rankings(records, boundaries, stage="formal")
