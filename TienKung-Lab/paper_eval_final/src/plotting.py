"""Experiment 1 plots built only from the frozen shared-boundary analysis set."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT


GROUPS = ("LOW", "MEDIUM", "HIGH")


def _cell_map(payload: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (int(row["cell"][0]), str(row["cell"][1])): row
        for row in payload["cells"]
        if row.get("cell", [None, None])[0] in (3, 4, 5)
        and row.get("cell", [None, None])[1] in GROUPS
    }


def write_experiment1_plots(stage: str, aggregate: dict[str, Any]) -> list[Path]:
    """Write recovery-rate and Trec figures; never infer or refit q1/q2."""
    import matplotlib.pyplot as plt

    outputs = []
    for model_id, payload in aggregate["models"].items():
        relation = payload.get("stratified_relation_set")
        if not relation or relation.get("boundary_id") is None:
            continue
        cells = _cell_map(payload)
        subtitle = f"共享边界 q1={relation['shared_q1']:.6g}, q2={relation['shared_q2']:.6g}"

        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        for n_min in (3, 4, 5):
            rates = [cells.get((n_min, group), {}).get("recovery_rate", np.nan) for group in GROUPS]
            axis.plot(GROUPS, rates, marker="o", label=f"Nmin={n_min}")
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("共享 margin 等级")
        axis.set_ylabel("持续恢复率")
        axis.set_title(f"实验一恢复率（{stage}）\n{subtitle}")
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        target = ROOT / "reports" / stage / f"experiment_1_{model_id}_shared_margin_recovery.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180)
        plt.close(fig)
        outputs.append(target)

        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        for n_min in (3, 4, 5):
            medians = [cells.get((n_min, group), {}).get("Trec", {}).get("median", np.nan)
                       for group in GROUPS]
            axis.plot(GROUPS, medians, marker="o", label=f"Nmin={n_min}")
        axis.set_xlabel("共享 margin 等级")
        axis.set_ylabel("恢复时间中位数 Trec (s)")
        axis.set_title(f"实验一恢复时间（{stage}）\n{subtitle}")
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        target = ROOT / "reports" / stage / f"experiment_1_{model_id}_shared_margin_Trec.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        outputs.append(target)
    return outputs
