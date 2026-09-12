"""Protocol denominators and compact statistical summaries."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr


DENOMINATOR_FIELDS = (
    "scheduled", "executed", "valid", "eval_invalid", "cert_invalid",
    "pre_push_physical_failure", "push_applied", "physical_reset", "horizon_reached",
    "recovered_sustained", "survived_post_observation", "n_recovered",
)


def wilson(successes: int, denominator: int) -> list[float] | None:
    if denominator == 0:
        return None
    z = 1.959963984540054
    p = successes / denominator
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    radius = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def denominators(records: list[dict[str, Any]], scheduled: int) -> dict[str, int]:
    result = {
        "scheduled": int(scheduled),
        "executed": len(records),
        "valid": sum(bool(row["valid"]) for row in records),
        "eval_invalid": sum(bool(row["eval_invalid"]) for row in records),
        "cert_invalid": sum(bool(row.get("cert_invalid")) for row in records),
        "pre_push_physical_failure": sum(row.get("failure_reason") == "PRE_PUSH_PHYSICAL_FAILURE" for row in records),
        "push_applied": sum(bool(row.get("push_applied")) for row in records),
        "physical_reset": sum(row["termination_kind"] == "PHYSICAL_RESET" for row in records),
        "horizon_reached": sum(row["termination_kind"] == "HORIZON_REACHED" for row in records),
        "recovered_sustained": sum(bool(row.get("recovered_sustained")) for row in records),
        "survived_post_observation": sum(bool(row.get("survived_post_observation")) for row in records),
        "n_recovered": sum(bool(row.get("recovered_sustained")) for row in records),
    }
    if result["valid"] + result["eval_invalid"] + sum(row.get("invalid_kind") == "UNRESOLVED_INVALID" for row in records) != len(records):
        raise ValueError("denominator closure failed")
    return result


def quantiles(values: Iterable[float | None]) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(array):
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None}
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return {"n": len(array), "median": float(median), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}


def summarize(records: list[dict[str, Any]], scheduled: int) -> dict[str, Any]:
    counts = denominators(records, scheduled)
    valid = [row for row in records if row["valid"]]
    complete = [row for row in valid if row["termination_kind"] == "HORIZON_REACHED"]
    recovered = [row for row in valid if row.get("recovered_sustained")]
    result = {
        "denominators": counts,
        "survival_rate": counts["horizon_reached"] / counts["valid"] if counts["valid"] else None,
        "survival_wilson_95": wilson(counts["horizon_reached"], counts["valid"]),
        "recovery_rate": counts["recovered_sustained"] / counts["valid"] if counts["valid"] else None,
        "recovery_wilson_95": wilson(counts["recovered_sustained"], counts["valid"]),
        "RMSE_v": quantiles(row.get("RMSE_v") for row in complete),
        "RMSE_vx": quantiles(row.get("RMSE_vx") for row in complete),
        "RMSE_vy": quantiles(row.get("RMSE_vy") for row in complete),
        "Trec": quantiles(row.get("recovery_time") for row in recovered),
        "Krec": quantiles(row.get("Krec") for row in recovered),
        "nTD0": quantiles(row.get("nTD0") for row in recovered),
    }
    relation = [row for row in valid if row.get("certificate_valid") and row.get("Nmin") in (2, 3, 4)
                and row.get("nTD0") is not None and not row.get("CERT_AFTER_RECOVERY")]
    n_values = [row["Nmin"] for row in relation]
    touchdown_values = [row["nTD0"] for row in relation]
    if len(relation) >= 2 and len(set(n_values)) >= 2 and len(set(touchdown_values)) >= 2:
        coefficient, pvalue = spearmanr(n_values, touchdown_values)
        coefficient = float(coefficient)
        pvalue = float(pvalue)
        result["spearman_Nmin_nTD0"] = {
            "n": len(relation),
            "rho": coefficient if math.isfinite(coefficient) else None,
            "pvalue": pvalue if math.isfinite(pvalue) else None,
        }
    else:
        result["spearman_Nmin_nTD0"] = {"n": len(relation), "rho": None, "pvalue": None}
    return result


def summarize_experiment1_cell(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Experiment 1 denominators; Trec is never imputed for non-recovery."""
    scheduled = len(records)
    pushed = [row for row in records if row.get("push_applied")]
    recovered = [row for row in records if row.get("recovered_sustained")]
    recovered_after_push = [row for row in pushed if row.get("recovered_sustained")]
    survived_after_push = [row for row in pushed if row.get("survived_post_observation")]
    sustained_count = len(recovered)
    return {
        "scheduled": scheduled,
        "push_applied": len(pushed),
        "push_applied_over_scheduled": len(pushed) / scheduled if scheduled else None,
        "sustained_recovery_count": sustained_count,
        "sustained_recovery_rate": sustained_count / scheduled if scheduled else None,
        "sustained_recovery_wilson_95": wilson(sustained_count, scheduled),
        "post_push_recovery_rate": len(recovered_after_push) / len(pushed) if pushed else None,
        "post_push_recovery_wilson_95": wilson(len(recovered_after_push), len(pushed)),
        "post_push_survival_rate": len(survived_after_push) / len(pushed) if pushed else None,
        "post_push_survival_wilson_95": wilson(len(survived_after_push), len(pushed)),
        "Trec": quantiles(row.get("recovery_time") for row in recovered),
        "nTD0": quantiles(row.get("nTD0") for row in records),
        "Krec": quantiles(row.get("Krec") for row in records),
        "eval_invalid": sum(bool(row.get("eval_invalid")) for row in records),
        "cert_invalid": sum(bool(row.get("cert_invalid")) for row in records),
        "failure_reasons": dict(sorted({
            reason: sum(row.get("failure_reason") == reason for row in records)
            for reason in {row.get("failure_reason") for row in records if row.get("failure_reason")}
        }.items())),
    }


def _spearman(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(x), "rho": None, "pvalue": None}
    rho, pvalue = spearmanr(x, y)
    return {"n": len(x), "rho": float(rho) if math.isfinite(rho) else None,
            "pvalue": float(pvalue) if math.isfinite(pvalue) else None}


def continuous_margin_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for n_min in (2, 3, 4):
        rows = [row for row in records if row.get("Nmin") == n_min]
        recovered = [row for row in rows if row.get("recovered_sustained")
                     and row.get("recovery_time") is not None]
        touchdown = [row for row in rows if row.get("nTD0") is not None]
        output[str(n_min)] = {
            "margin_vs_sustained_recovery": _spearman(
                [float(row["margin_raw"]) for row in rows],
                [float(bool(row.get("recovered_sustained"))) for row in rows]),
            "margin_vs_recovery_time": _spearman(
                [float(row["margin_raw"]) for row in recovered],
                [float(row["recovery_time"]) for row in recovered]),
            "margin_vs_nTD0": _spearman(
                [float(row["margin_raw"]) for row in touchdown],
                [float(row["nTD0"]) for row in touchdown]),
        }
    return output


def summarize_cells(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (row.get("Nmin"), row.get("margin_group")) if row["experiment_id"] == 1 else (
            row.get("slope_deg"), row.get("speed_group"), row.get("disturbance", {}).get("type")
        )
        groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = summarize(rows, len(rows))
        if rows[0]["experiment_id"] == 1:
            summary.update(summarize_experiment1_cell(rows))
        output.append({"cell": list(key), **summary})
    return output
