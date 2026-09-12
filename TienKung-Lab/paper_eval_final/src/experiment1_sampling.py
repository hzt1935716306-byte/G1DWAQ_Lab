"""Outcome-blind Experiment 1 pilot fitting and formal accept/reject sampling."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .common import ROOT, atomic_write, digest, load_yaml, write_json


TARGET_NMIN = (3, 4, 5)
MARGIN_GROUPS = ("LOW", "MEDIUM", "HIGH")
BOUNDARY_PATH = ROOT / "configs/experiment1_margin_boundaries.yaml"


def experiment1_config() -> dict[str, Any]:
    return load_yaml(ROOT / "configs/experiment1.yaml")


def load_boundaries(*, require_frozen: bool = False) -> dict[str, Any]:
    value = load_yaml(BOUNDARY_PATH)
    if require_frozen and value.get("status") != "FROZEN":
        raise ValueError("Experiment 1 margin boundaries are not frozen")
    if value.get("status") == "FROZEN":
        claimed = value.get("boundary_id")
        calculated = digest({key: item for key, item in value.items() if key != "boundary_id"})
        if claimed != calculated:
            raise ValueError("Experiment 1 frozen margin boundary hash mismatch")
        cfg = experiment1_config()
        if value.get("source_stage") != "pilot" or value.get("source_identity", {}).get("stage") != "pilot":
            raise ValueError("Experiment 1 boundaries must come from the independent pilot stage")
        if value.get("baseline_id") != cfg["baseline_id"] or value.get("checkpoint_sha256") != cfg["checkpoint_sha256"]:
            raise ValueError("Experiment 1 boundaries belong to a different baseline/checkpoint")
        expected = int(cfg["pilot_sampling"]["certificate_valid_per_Nmin"])
        for n_min in TARGET_NMIN:
            ids = value.get("selected_trial_ids", {}).get(str(n_min), [])
            bounds = value.get("boundaries", {}).get(str(n_min), {})
            q1, q2 = bounds.get("q1"), bounds.get("q2")
            if len(ids) != expected or q1 is None or q2 is None or not (
                math.isfinite(float(q1)) and math.isfinite(float(q2)) and float(q1) < float(q2)
            ):
                raise ValueError(f"Experiment 1 frozen boundary artifact is invalid for Nmin={n_min}")
    return value


def _raw_margin(row: dict[str, Any]) -> float | None:
    """Return only an explicitly unrounded, binary64 raw margin."""
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


def pilot_eligibility(row: dict[str, Any]) -> tuple[bool, str]:
    """Eligibility intentionally does not inspect any recovery/outcome field."""
    if row.get("stage") != "pilot":
        return False, "WRONG_STAGE"
    if row.get("invalid_kind") == "EVAL_INVALID":
        return False, "EVAL_INVALID"
    if not row.get("certificate_valid"):
        return False, "CERT_INVALID"
    if row.get("Nmin") not in TARGET_NMIN:
        return False, "NMIN_OUTSIDE_TARGET"
    if _raw_margin(row) is None:
        return False, "MARGIN_NOT_RAW_FINITE_FLOAT64"
    return True, "ELIGIBLE"


def select_pilot_samples(records: Iterable[dict[str, Any]], *, per_nmin: int = 120) -> dict[str, Any]:
    """Select the first eligible samples per N in deterministic candidate order.

    Recovery success, failure, times, and step counts are deliberately absent
    from both the predicate and sorting key.
    """
    ordered = sorted(records, key=lambda row: (int(row.get("sequence", 10**18)), str(row.get("trial_id", ""))))
    selected: dict[int, list[dict[str, Any]]] = {n: [] for n in TARGET_NMIN}
    exclusions: Counter[str] = Counter()
    for row in ordered:
        eligible, reason = pilot_eligibility(row)
        if not eligible:
            exclusions[reason] += 1
            continue
        n_min = int(row["Nmin"])
        if len(selected[n_min]) < per_nmin:
            selected[n_min].append(row)
        else:
            exclusions[f"N{n_min}_QUOTA_FILLED"] += 1
    counts = {str(n): len(rows) for n, rows in selected.items()}
    return {
        "stage": "pilot",
        "target_per_Nmin": int(per_nmin),
        "selected": selected,
        "selected_trial_ids": {str(n): [row["trial_id"] for row in rows] for n, rows in selected.items()},
        "counts": counts,
        "deficits": {str(n): max(0, per_nmin - len(rows)) for n, rows in selected.items()},
        "exclusions": dict(sorted(exclusions.items())),
        "complete": all(len(rows) == per_nmin for rows in selected.values()),
    }


def _calculation_trace(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": row.get("trial_id"),
        "sequence": row.get("sequence"),
        "Nmin": row.get("Nmin"),
        "margin_raw": row.get("margin_raw"),
        "certificate_calculation": row.get("certificate_calculation"),
    }


def _diagnose_nmin(n_min: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 120:
        raise ValueError(f"Nmin={n_min}: exactly 120 pilot rows are required")
    margins = np.sort(np.asarray([_raw_margin(row) for row in rows], dtype=np.float64))
    m40, m41 = float(margins[39]), float(margins[40])
    m80, m81 = float(margins[79]), float(margins[80])
    q1, q2 = (m40 + m41) / 2.0, (m80 + m81) / 2.0
    unique = int(np.unique(margins).size)
    zero_fraction = float(np.mean(margins == 0.0))
    calculations = [row.get("certificate_calculation") or {} for row in rows]
    saturation = any(bool(item.get("margin_saturated")) for item in calculations)
    clamp = any(bool(item.get("margin_clamped")) for item in calculations)
    precision_loss = any(
        row.get("margin_storage_dtype") != "float64" or row.get("margin_was_rounded") is not False
        for row in rows
    )
    # Duplicate values alone can be mathematically genuine. They become a
    # degeneracy signal when at least 10% of values repeat and the recorded
    # calculation reports saturation/clamping/precision loss.
    large_duplicate_block = unique < 108
    reasons = []
    if m40 == m41:
        reasons.append("M40_EQUALS_M41")
    if m80 == m81:
        reasons.append("M80_EQUALS_M81")
    if q1 >= q2:
        reasons.append("Q1_NOT_BELOW_Q2")
    if large_duplicate_block and (saturation or clamp or precision_loss):
        reasons.append("DUPLICATES_WITH_CLAMP_OR_PRECISION_LOSS")
    return {
        "Nmin": n_min,
        "status": "MARGIN_DEGENERATE" if reasons else "VALID",
        "degeneracy_reasons": reasons,
        "count": 120,
        "minimum": float(margins[0]),
        "maximum": float(margins[-1]),
        "median": float(np.median(margins)),
        "quantile_33_3": q1,
        "quantile_66_7": q2,
        "order_statistics": {"m40": m40, "m41": m41, "m80": m80, "m81": m81},
        "different_margin_value_count": unique,
        "zero_fraction": zero_fraction,
        "large_duplicate_block": large_duplicate_block,
        "margin_saturation_detected": saturation,
        "margin_clamp_detected": clamp,
        "precision_loss_detected": precision_loss,
        "raw_margin_and_intermediates": [_calculation_trace(row) for row in rows],
    }


def fit_pilot_boundaries(records: Iterable[dict[str, Any]], *, source_identity: dict[str, Any],
                         output_path: str | Path = BOUNDARY_PATH) -> dict[str, Any]:
    """Fit and freeze all six boundaries, or emit a non-frozen diagnostic."""
    cfg = experiment1_config()
    per_nmin = int(cfg["pilot_sampling"]["certificate_valid_per_Nmin"])
    selection = select_pilot_samples(records, per_nmin=per_nmin)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": 1,
        "baseline_id": cfg["baseline_id"],
        "checkpoint_sha256": cfg["checkpoint_sha256"],
        "source_stage": "pilot",
        "source_identity": source_identity,
        "selection_rule": "first eligible candidates by (sequence, trial_id); outcome-blind",
        "target_Nmin": list(TARGET_NMIN),
        "samples_per_Nmin": per_nmin,
        "selected_trial_ids": selection["selected_trial_ids"],
        "selection_exclusions": selection["exclusions"],
    }
    if not selection["complete"]:
        payload.update(status="COLLECTING", deficits=selection["deficits"], diagnostics={})
    else:
        diagnostics = {
            str(n): _diagnose_nmin(n, selection["selected"][n]) for n in TARGET_NMIN
        }
        degenerate = [n for n, item in diagnostics.items() if item["status"] == "MARGIN_DEGENERATE"]
        payload["diagnostics"] = diagnostics
        if degenerate:
            payload.update(status="MARGIN_DEGENERATE", degenerate_Nmin=degenerate, boundaries={})
        else:
            payload["boundaries"] = {
                str(n): {
                    "q1": diagnostics[str(n)]["quantile_33_3"],
                    "q2": diagnostics[str(n)]["quantile_66_7"],
                }
                for n in TARGET_NMIN
            }
            payload["status"] = "FROZEN"
    payload["boundary_id"] = digest({key: value for key, value in payload.items() if key != "boundary_id"})
    atomic_write(output_path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
    diagnostic_path = Path(output_path).with_suffix(".diagnostics.json")
    write_json(diagnostic_path, payload)
    if Path(output_path).resolve() == BOUNDARY_PATH.resolve():
        freeze_path = ROOT / "protocol/implementation_freeze.yaml"
        freeze = load_yaml(freeze_path)
        freeze["experiment1_margin_boundaries"] = {
            "file": str(Path(output_path).resolve().relative_to(ROOT)),
            "status": payload["status"],
            "boundary_id": payload["boundary_id"] if payload["status"] == "FROZEN" else None,
        }
        atomic_write(freeze_path, yaml.safe_dump(freeze, allow_unicode=True, sort_keys=False))
    return payload


def classify_margin(n_min: int | None, margin_raw: float | None,
                    boundaries: dict[str, Any]) -> str | None:
    if boundaries.get("status") != "FROZEN" or n_min not in TARGET_NMIN:
        return None
    if margin_raw is None or not math.isfinite(float(margin_raw)):
        return None
    values = boundaries["boundaries"][str(int(n_min))]
    margin = float(margin_raw)
    if margin < float(values["q1"]):
        return "LOW"
    if margin < float(values["q2"]):
        return "MEDIUM"
    return "HIGH"


def select_formal_samples(records: Iterable[dict[str, Any]], boundaries: dict[str, Any],
                          *, per_cell: int = 160) -> dict[str, Any]:
    """Apply frozen boundaries and outcome-blind first-arrival acceptance."""
    if boundaries.get("status") != "FROZEN":
        raise ValueError("formal selection requires frozen pilot boundaries")
    cells: dict[tuple[int, str], list[str]] = {
        (n, group): [] for n in TARGET_NMIN for group in MARGIN_GROUPS
    }
    exclusions: Counter[str] = Counter()
    ordered = sorted(records, key=lambda row: (int(row.get("sequence", 10**18)), str(row.get("trial_id", ""))))
    for row in ordered:
        if row.get("stage") != "formal":
            exclusions["WRONG_STAGE"] += 1
            continue
        if row.get("invalid_kind") == "EVAL_INVALID":
            exclusions["EVAL_INVALID"] += 1
            continue
        if not row.get("certificate_valid"):
            exclusions["CERT_INVALID"] += 1
            continue
        margin = _raw_margin(row)
        group = classify_margin(row.get("Nmin"), margin, boundaries)
        if group is None:
            exclusions["NMIN_OR_MARGIN_OUTSIDE_TARGET"] += 1
            continue
        cell = (int(row["Nmin"]), group)
        if len(cells[cell]) >= per_cell:
            exclusions[f"N{cell[0]}_{cell[1]}_QUOTA_FILLED"] += 1
            continue
        cells[cell].append(row["trial_id"])
    detail = {
        f"N{n}_{group}": {
            "accepted": len(cells[(n, group)]),
            "target": per_cell,
            "missing": max(0, per_cell - len(cells[(n, group)])),
            "trial_ids": cells[(n, group)],
        }
        for n in TARGET_NMIN for group in MARGIN_GROUPS
    }
    return {
        "dataset_role": "FROZEN_QUANTILE_RELATION_SET",
        "stage": "formal",
        "boundary_id": boundaries["boundary_id"],
        "accepted_trial_ids": [trial_id for cell in cells.values() for trial_id in cell],
        "cells": detail,
        "exclusions": dict(sorted(exclusions.items())),
        "complete": all(value["missing"] == 0 for value in detail.values()),
    }


def annotate_sampling_batch(records: list[dict[str, Any]], prior_records: Iterable[dict[str, Any]],
                            *, stage: str, boundaries: dict[str, Any] | None = None) -> None:
    """Add sampling decisions in sequence order without consulting outcomes."""
    cfg = experiment1_config()
    if stage == "pilot":
        target = int(cfg["pilot_sampling"]["certificate_valid_per_Nmin"])
        occupancy = Counter(
            int(row["Nmin"]) for row in prior_records
            if row.get("sampling_accepted") and row.get("Nmin") in TARGET_NMIN
        )
        for row in sorted(records, key=lambda item: (int(item["sequence"]), item["trial_id"])):
            eligible, reason = pilot_eligibility(row)
            accepted = bool(eligible and occupancy[int(row["Nmin"])] < target)
            row.update(
                sampling_eligible=eligible,
                sampling_accepted=accepted,
                sampling_rejection_reason=None if accepted else (
                    f"N{row['Nmin']}_QUOTA_FILLED" if eligible else reason
                ),
                margin_group=None,
                margin_boundary_id=None,
            )
            if accepted:
                occupancy[int(row["Nmin"])] += 1
        return
    if stage != "formal":
        return
    if boundaries is None or boundaries.get("status") != "FROZEN":
        raise ValueError("formal Experiment 1 annotation requires frozen boundaries")
    target = int(cfg["formal_sampling"]["certificate_valid_per_cell"])
    occupancy = Counter(
        (int(row["Nmin"]), str(row["margin_group"])) for row in prior_records
        if row.get("sampling_accepted") and row.get("Nmin") in TARGET_NMIN
        and row.get("margin_group") in MARGIN_GROUPS
    )
    for row in sorted(records, key=lambda item: (int(item["sequence"]), item["trial_id"])):
        eligible = (
            row.get("stage") == "formal"
            and row.get("invalid_kind") != "EVAL_INVALID"
            and bool(row.get("certificate_valid"))
            and row.get("Nmin") in TARGET_NMIN
            and _raw_margin(row) is not None
        )
        group = classify_margin(row.get("Nmin"), _raw_margin(row), boundaries) if eligible else None
        cell = (int(row["Nmin"]), group) if group is not None else None
        accepted = bool(cell is not None and occupancy[cell] < target)
        if not eligible:
            reason = "CERT_OR_RAW_MARGIN_INELIGIBLE"
        elif not accepted:
            reason = f"N{cell[0]}_{cell[1]}_QUOTA_FILLED"
        else:
            reason = None
        row.update(
            sampling_eligible=eligible,
            sampling_accepted=accepted,
            sampling_rejection_reason=reason,
            margin_group=group,
            margin_boundary_id=boundaries["boundary_id"],
        )
        if accepted:
            occupancy[cell] += 1


def sampling_complete(records: Iterable[dict[str, Any]], *, stage: str,
                      boundaries: dict[str, Any] | None = None) -> bool:
    rows = list(records)
    cfg = experiment1_config()
    if stage == "pilot":
        target = int(cfg["pilot_sampling"]["certificate_valid_per_Nmin"])
        counts = Counter(int(row["Nmin"]) for row in rows if row.get("sampling_accepted"))
        return all(counts[n] == target for n in TARGET_NMIN)
    if stage == "formal":
        if boundaries is None:
            return False
        return select_formal_samples(
            rows, boundaries,
            per_cell=int(cfg["formal_sampling"]["certificate_valid_per_cell"]),
        )["complete"]
    return True


def pilot_continuation_condition_weights(records: Iterable[dict[str, Any]]) -> dict[int, list[str]]:
    """Rank layers using certificate N only; never read recovery outcomes."""
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
        n: sorted(
            all_layers,
            key=lambda layer: (
                -(hits[n][layer] + 1.0) / (attempts[layer] + 3.0),
                attempts[layer],
                layer,
            ),
        )
        for n in TARGET_NMIN
    }


def formal_continuation_condition_weights(records: Iterable[dict[str, Any]],
                                          boundaries: dict[str, Any]) -> dict[tuple[int, str], list[str]]:
    """Rank condition layers from certificate-cell yield only."""
    attempts: Counter[str] = Counter()
    hits: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    all_layers: set[str] = set()
    for row in records:
        layer = str(row["condition_id"])
        all_layers.add(layer)
        attempts[layer] += 1
        if row.get("certificate_valid") and _raw_margin(row) is not None:
            group = classify_margin(row.get("Nmin"), _raw_margin(row), boundaries)
            if group is not None:
                hits[(int(row["Nmin"]), group)][layer] += 1
    return {
        cell: sorted(
            all_layers,
            key=lambda layer: (
                -(hits[cell][layer] + 1.0) / (attempts[layer] + 3.0),
                attempts[layer],
                layer,
            ),
        )
        for cell in ((n, group) for n in TARGET_NMIN for group in MARGIN_GROUPS)
    }
