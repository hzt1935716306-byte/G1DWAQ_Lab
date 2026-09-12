#!/usr/bin/env python3
"""Build the small Experiment 1 pilot from immutable existing trial records."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper_eval_final.src.common import ROOT, read_json, write_csv, write_json
from paper_eval_final.src.experiment1_sampling import (
    MARGIN_GROUPS, TARGET_NMIN, classify_margin, load_boundaries,
    select_pilot_analysis_samples, write_sampling_assignments,
)
from paper_eval_final.src.plotting import write_experiment1_plots
from paper_eval_final.src.statistics import continuous_margin_diagnostics, summarize_experiment1_cell


DEFAULT_SOURCE = ROOT / "results/pilot/protocol_1_2_fe3b9cb6c6f5/experiment_1/M1/train_seed_42"
DEFAULT_OUTPUT = ROOT / "results/pilot/experiment_1_conditional_tertiles/M1/train_seed_42"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def main() -> int:
    args = parser().parse_args()
    identity = read_json(args.source / "identity.json")
    records = [read_json(path) for path in sorted((args.source / "records").glob("*.json"))]
    boundaries = load_boundaries(require_frozen=True)
    for field in ("checkpoint_sha256", "metrics_config_hash", "physics_profile_hash", "asset_hash", "simulator_version"):
        if identity.get(field) != boundaries["source_identity"].get(field):
            raise ValueError(f"source compatibility mismatch: {field}")
    calibration_ids = {trial_id for ids in boundaries["calibration_trial_ids"].values() for trial_id in ids}
    if not calibration_ids <= {row["trial_id"] for row in records}:
        raise ValueError("frozen calibration manifest trial is missing")

    selection = select_pilot_analysis_samples(records, boundaries, per_cell=40)
    if not selection["complete"]:
        print(json.dumps({"status": "PILOT_CELL_DEFICIT", "cells": selection["cells"]}, indent=2))
        return 2
    accepted = set(selection["accepted_trial_ids"])
    analysis = []
    for row in records:
        if row["trial_id"] not in accepted:
            continue
        item = dict(row)
        item["margin_group"] = classify_margin(item["Nmin"], item["margin_raw"], boundaries)
        analysis.append(item)

    cells = []
    for n_min in TARGET_NMIN:
        for group in MARGIN_GROUPS:
            rows = [row for row in analysis if row["Nmin"] == n_min and row["margin_group"] == group]
            cells.append({"cell": [n_min, group], **summarize_experiment1_cell(rows)})
    continuous = continuous_margin_diagnostics(analysis)
    calibration = {}
    by_id = {row["trial_id"]: row for row in records}
    for n_min in TARGET_NMIN:
        values = [float(by_id[trial_id]["margin_raw"])
                  for trial_id in boundaries["calibration_trial_ids"][str(n_min)]]
        group_counts = Counter(classify_margin(n_min, value, boundaries) for value in values)
        calibration[str(n_min)] = {"count": len(values), "minimum": min(values), "maximum": max(values),
                                   "q1": boundaries[f"n{n_min}_q1"], "q2": boundaries[f"n{n_min}_q2"],
                                   "group_counts": dict(group_counts)}
    summary = {
        "status": "COMPLETE", "stage": "pilot", "experiment_id": 1,
        "source_record_count": len(records), "reused_trials": len(analysis),
        "newly_scheduled_trials": 0, "analysis_trial_count": len(analysis),
        "candidate_stream": {
            "scheduled": sum(int(row.get("scheduled", 0)) for row in records),
            "eval_invalid": sum(bool(row.get("eval_invalid")) for row in records),
            "cert_invalid": sum(bool(row.get("cert_invalid")) for row in records),
            "failure_reasons": dict(Counter(row.get("failure_reason") for row in records if row.get("failure_reason"))),
        },
        "calibration": calibration, "cells": cells,
        "condition_balance": {key: value["condition_balance"] for key, value in selection["cells"].items()},
        "continuous_margin": continuous,
        "calibration_manifest_hash": boundaries["calibration_manifest_hash"],
        "evaluation_manifest_hash": selection["evaluation_manifest_hash"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_sampling_assignments(args.output, selection)
    write_json(args.output / "pilot_summary.json", summary)
    write_csv(args.output / "pilot_cells.csv", cells, [
        "cell", "scheduled", "push_applied", "push_applied_over_scheduled",
        "sustained_recovery_count", "sustained_recovery_rate", "sustained_recovery_wilson_95",
        "post_push_recovery_rate", "post_push_recovery_wilson_95", "post_push_survival_rate",
        "post_push_survival_wilson_95", "Trec", "nTD0", "Krec", "eval_invalid", "cert_invalid",
        "failure_reasons",
    ])
    write_csv(args.output / "pilot_analysis_trials.csv", analysis, [
        "trial_id", "eval_seed", "Nmin", "margin_raw", "margin_group", "slope_deg", "speed_group",
        "disturbance", "push_applied", "recovered_sustained", "survived_post_observation",
        "recovery_time", "nTD0", "Krec", "termination_kind", "failure_reason",
    ])
    plot_payload = {"models": {identity["model_id"]: {
        "cells": cells, "stratified_relation_set": selection,
    }}}
    figures = write_experiment1_plots("pilot", plot_payload)
    summary["generated_figures"] = [str(path.relative_to(ROOT)) for path in figures]
    write_json(args.output / "pilot_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
