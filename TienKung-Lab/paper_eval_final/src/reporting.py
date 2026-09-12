"""Human-readable technical reports; raw records remain immutable."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import ROOT, atomic_write


def write_experiment_report(stage: str, experiment: int, aggregate: dict[str, Any]) -> Path:
    lines = [f"# Experiment {experiment} — {stage}", "", f"Records: {aggregate['record_count']}", ""]
    for model_id, payload in aggregate["models"].items():
        summary = payload["summary"]
        lines.extend([
            f"## {model_id}", "", f"Formal seed status: `{payload['formal_status']}`", "",
            f"- Denominators: `{summary['denominators']}`",
            f"- Survival rate: `{summary['survival_rate']}`",
            f"- Sustained recovery rate: `{summary['recovery_rate']}`",
            f"- Trec: `{summary['Trec']}`",
            f"- Spearman(Nmin, nTD0): `{summary['spearman_Nmin_nTD0']}`", "",
        ])
    path = ROOT / "reports" / stage / f"experiment_{experiment}.md"
    atomic_write(path, "\n".join(lines) + "\n")
    return path


def write_screening_report(aggregate: dict[str, Any]) -> Path:
    from .model_loader import registry

    lines = ["# Experiment 1 baseline screening", "",
             "Screening is exploratory and must not be merged into pilot/formal evidence.",
             "No baseline is selected automatically; review these results and explicitly freeze one later.", ""]
    baseline_registry = registry()["screening_baselines"]
    model_to_baselines: dict[str, list[str]] = {}
    for baseline_id, row in baseline_registry.items():
        if row.get("model"):
            model_to_baselines.setdefault(row["model"], []).append(baseline_id)
    for model_id, payload in aggregate["models"].items():
        summary = payload["summary"]
        detail = payload["screening_detail"]
        occupancy = detail["N_margin_occupancy"]
        occupied = sum(value["n"] > 0 for value in occupancy.values())
        d = summary["denominators"]
        lines.extend([
            f"## {model_id} ({', '.join(model_to_baselines.get(model_id, []))})", "",
            f"- checkpoint SHA-256: `{', '.join(payload['checkpoint_sha256'])}`",
            f"- valid trials: {d['valid']}",
            f"- physical coverage: {d['valid']}/{d['executed']}",
            f"- pre-push failures: {d['pre_push_physical_failure']}",
            f"- push applied: {d['push_applied']}",
            f"- TD0 coverage: {detail['TD0_count']}/{d['valid']}",
            f"- certificate-valid coverage: {detail['certificate_valid_count']}/{detail['TD0_count']}",
            f"- N=3/4/5 counts: `{detail['Nmin_counts']}`",
            f"- occupied N-margin cells: {occupied}/9",
            f"- 9-cell occupancy: `{occupancy}`",
            f"- recovery rate: {summary['recovery_rate']}",
            f"- Trec: `{summary['Trec']}`",
            f"- nTD0: `{summary['nTD0']}`",
            f"- Spearman(Nmin,nTD0): `{summary['spearman_Nmin_nTD0']}`", "",
        ])
    missing = [(baseline_id, row) for baseline_id, row in baseline_registry.items() if not row.get("model")]
    if missing:
        lines.extend(["## Unavailable registered candidates", ""])
        for baseline_id, row in missing:
            lines.append(f"- `{baseline_id}` / `{row['task_name']}`: `{row.get('status', 'CHECKPOINT_MISSING')}`")
        lines.append("")
    lines.extend(["## Decision gate", "",
                  "STOP: choose the Experiment 1 baseline explicitly before pilot/formal execution.", ""])
    path = ROOT / "reports/screening/EXP1_BASELINE_SCREENING.md"
    atomic_write(path, "\n".join(lines) + "\n")
    return path
