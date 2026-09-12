"""Experiment 1 plots built from frozen within-Nmin tertiles."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT
from .experiment1_sampling import TARGET_NMIN


GROUPS = ("LOWER", "MIDDLE", "UPPER")


def _cell_map(payload: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (int(row["cell"][0]), str(row["cell"][1])): row
        for row in payload["cells"]
        if row.get("cell", [None, None])[0] in TARGET_NMIN
        and row.get("cell", [None, None])[1] in GROUPS
    }


def write_experiment1_plots(stage: str, aggregate: dict[str, Any]) -> list[Path]:
    """Write recovery-rate and Trec figures; never infer or refit boundaries."""
    import matplotlib.pyplot as plt

    outputs = []
    for model_id, payload in aggregate["models"].items():
        relation = payload.get("stratified_relation_set")
        if not relation or relation.get("boundary_id") is None:
            continue
        cells = _cell_map(payload)
        bounds = relation["boundaries"]
        subtitle = "\n".join(
            f"N{n}: q1={bounds[str(n)]['q1']:.6g}, q2={bounds[str(n)]['q2']:.6g}"
            for n in TARGET_NMIN
        )

        fig, axis = plt.subplots(figsize=(8.2, 5.8))
        for n_min in TARGET_NMIN:
            rows = [cells.get((n_min, group), {}) for group in GROUPS]
            rates = [row.get("sustained_recovery_rate", np.nan) for row in rows]
            intervals = [row.get("sustained_recovery_wilson_95") for row in rows]
            errors = [[rate - interval[0] for rate, interval in zip(rates, intervals)],
                      [interval[1] - rate for rate, interval in zip(rates, intervals)]]
            axis.errorbar(GROUPS, rates, yerr=errors, marker="o", capsize=3, label=f"Nmin={n_min}")
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Lower / Middle / Upper margin tertile within each Nmin")
        axis.set_ylabel("Sustained recovery rate")
        axis.set_title(f"Experiment 1 recovery rate ({stage})\n{subtitle}", fontsize=11)
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        target = ROOT / "reports" / stage / f"experiment_1_{model_id}_conditional_margin_recovery.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180)
        plt.close(fig)
        outputs.append(target)

        fig, axis = plt.subplots(figsize=(8.2, 5.8))
        for n_min in TARGET_NMIN:
            rows = [cells.get((n_min, group), {}) for group in GROUPS]
            trec = [row.get("Trec", {}) for row in rows]
            medians = [item.get("median", np.nan) for item in trec]
            errors = [[median - item.get("q1", median) for median, item in zip(medians, trec)],
                      [item.get("q3", median) - median for median, item in zip(medians, trec)]]
            axis.errorbar(GROUPS, medians, yerr=errors, marker="o", capsize=3, label=f"Nmin={n_min}")
            for group, median, item in zip(GROUPS, medians, trec):
                axis.annotate(f"n={item.get('n', 0)}", (group, median), xytext=((n_min - 3) * 24, 7),
                              textcoords="offset points", ha="center", fontsize=8)
        axis.set_xlabel("Lower / Middle / Upper margin tertile within each Nmin")
        axis.set_ylabel("Median recovery time Trec (s)")
        axis.set_title(f"Experiment 1 recovery time ({stage})\n{subtitle}", fontsize=11)
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        target = ROOT / "reports" / stage / f"experiment_1_{model_id}_conditional_margin_Trec.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        outputs.append(target)
    return outputs
