"""Read-only aggregation with the mandatory identity compatibility gate."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .common import ROOT, read_json, result_namespace, write_json
from .statistics import summarize, summarize_cells
from .storage import compatibility_gate, validate_record
from .trial_generator import select_stratified_relation_set


def discover_shards(stage: str, experiment: int) -> list[Path]:
    root = (ROOT / "results" / stage / result_namespace() / f"experiment_{experiment}").resolve()
    if not root.exists():
        return []
    return sorted(path.parent for path in root.rglob("completion.json"))


def aggregate_experiment(stage: str, experiment: int) -> dict[str, Any]:
    shards = discover_shards(stage, experiment)
    identities = [read_json(path / "identity.json") for path in shards]
    gate = compatibility_gate(identities)
    records = []
    seen = set()
    for shard in shards:
        completion = read_json(shard / "completion.json")
        if completion.get("status") != "COMPLETE":
            raise ValueError(f"incomplete shard: {shard}")
        for path in sorted((shard / "records").glob("*.json")):
            row = read_json(path)
            validate_record(row)
            key = (row["model_id"], row["train_seed"], row["trial_id"])
            if key in seen:
                raise ValueError(f"duplicate completed trial: {key}")
            seen.add(key)
            records.append(row)
    by_model: dict[str, Any] = {}
    for model_id in sorted({row["model_id"] for row in records}):
        selected = [row for row in records if row["model_id"] == model_id]
        payload = {
            "summary": summarize(selected, len(selected)),
            "cells": summarize_cells(selected),
            "train_seeds": sorted({row["train_seed"] for row in selected}),
            "checkpoint_sha256": sorted({row["checkpoint_sha256"] for row in selected}),
            "formal_status": "READY" if len({row["train_seed"] for row in selected}) == 3 else "TRAIN_SEED_INCOMPLETE",
        }
        if experiment == 1:
            n_margin = {}
            for n_min in (3, 4, 5):
                for margin in ("LOW", "MEDIUM", "HIGH"):
                    rows = [row for row in selected if row.get("certificate_valid")
                            and row.get("Nmin") == n_min and row.get("margin_group") == margin]
                    recovered = sum(bool(row.get("recovered_sustained")) for row in rows)
                    n_margin[f"N{n_min}_{margin}"] = {
                        "n": len(rows), "recovered": recovered,
                        "recovery_rate": recovered / len(rows) if rows else None,
                    }
            payload["screening_detail"] = {
                "TD0_count": sum(row.get("TD0_time") is not None for row in selected),
                "certificate_valid_count": sum(bool(row.get("certificate_valid")) for row in selected),
                "Nmin_counts": {str(n_min): sum(row.get("certificate_valid") and row.get("Nmin") == n_min
                                                for row in selected) for n_min in (3, 4, 5)},
                "N_margin_occupancy": n_margin,
                "condition_layer_coverage": len({row["condition_id"] for row in selected}),
            }
            if stage in ("pilot", "formal"):
                relation_set = select_stratified_relation_set(
                    selected,
                    stage=stage,
                    manifest_seed=int(selected[0]["eval_seed"]) - int(selected[0]["sequence"]) - 1_000_000,
                )
                accepted = set(relation_set["accepted_trial_ids"])
                accepted_records = [row for row in selected if row["trial_id"] in accepted]
                relation_set["summary"] = summarize(accepted_records, len(accepted_records))
                payload["stratified_relation_set"] = relation_set
        by_model[model_id] = payload
    payload = {"compatibility_gate": gate, "shards": [str(path.relative_to(ROOT)) for path in shards],
               "models": by_model, "record_count": len(records)}
    output = ROOT / "reports" / stage / f"experiment_{experiment}_aggregate.json"
    write_json(output, payload)
    return payload


def aggregate(stage: str, experiments: Iterable[int]) -> dict[str, Any]:
    return {str(experiment): aggregate_experiment(stage, int(experiment)) for experiment in experiments}
