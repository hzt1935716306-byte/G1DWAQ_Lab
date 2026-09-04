"""Pure analysis helpers for the DWAQ velocity-estimator diagnostic."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr


_ABS_PERCENTILES = (0.50, 0.90, 0.95, 0.99)


def _as_rows(values: Iterable[Iterable[float]], *, columns: int) -> np.ndarray:
    rows = np.asarray(list(values), dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != columns or rows.shape[0] == 0:
        raise ValueError(f"expected a non-empty (N, {columns}) array, got {rows.shape}")
    if not np.isfinite(rows).all():
        raise ValueError("diagnostic inputs must be finite")
    return rows


def _scalar_error_statistics(error: np.ndarray) -> dict[str, float]:
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    absolute = np.abs(error)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_bias": float(np.mean(error)),
        **{
            f"absolute_error_P{int(100 * quantile)}": float(
                np.quantile(absolute, quantile)
            )
            for quantile in _ABS_PERCENTILES
        },
        "absolute_error_max": float(np.max(absolute)),
    }


def _safe_correlation(first: np.ndarray, second: np.ndarray, kind: str) -> float | None:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return None
    if kind == "pearson":
        value = np.corrcoef(first, second)[0, 1]
    elif kind == "spearman":
        value = spearmanr(first, second).statistic
    else:
        raise ValueError(f"unknown correlation kind: {kind}")
    return float(value) if np.isfinite(value) else None


def velocity_error_statistics(
    estimate: Iterable[Iterable[float]],
    reference: Iterable[Iterable[float]],
) -> dict[str, Any]:
    """Summarize component and horizontal-vector velocity errors in m/s."""

    estimate_rows = _as_rows(estimate, columns=3)
    reference_rows = _as_rows(reference, columns=3)
    if estimate_rows.shape != reference_rows.shape:
        raise ValueError("estimate and reference must have the same shape")
    error = estimate_rows - reference_rows
    vector_xy = np.linalg.norm(error[:, :2], axis=1)
    component_names = ("x", "y", "z")
    return {
        "sample_count": int(error.shape[0]),
        "components": {
            name: {
                **_scalar_error_statistics(error[:, index]),
                "pearson_estimate_vs_GT": _safe_correlation(
                    estimate_rows[:, index], reference_rows[:, index], "pearson"
                ),
                "spearman_estimate_vs_GT": _safe_correlation(
                    estimate_rows[:, index], reference_rows[:, index], "spearman"
                ),
            }
            for index, name in enumerate(component_names)
        },
        "xy_vector_error": {
            "mean": float(np.mean(vector_xy)),
            "rmse": float(np.sqrt(np.mean(np.square(vector_xy)))),
            "P50": float(np.quantile(vector_xy, 0.50)),
            "P90": float(np.quantile(vector_xy, 0.90)),
            "P95": float(np.quantile(vector_xy, 0.95)),
            "P99": float(np.quantile(vector_xy, 0.99)),
            "max": float(np.max(vector_xy)),
        },
    }


def dcm_velocity_error_statistics(
    estimate_xy: Iterable[Iterable[float]],
    reference_xy: Iterable[Iterable[float]],
    omega: Iterable[float],
) -> dict[str, Any]:
    """Summarize ``(v_est-v_GT)/omega`` in centimeters."""

    estimate_rows = _as_rows(estimate_xy, columns=2)
    reference_rows = _as_rows(reference_xy, columns=2)
    omega_rows = np.asarray(list(omega), dtype=np.float64).reshape(-1)
    if estimate_rows.shape != reference_rows.shape or omega_rows.shape[0] != estimate_rows.shape[0]:
        raise ValueError("velocity and omega sample counts must match")
    if not np.isfinite(omega_rows).all() or np.any(omega_rows <= 0.0):
        raise ValueError("omega must be finite and positive")
    error_cm = (estimate_rows - reference_rows) / omega_rows[:, None] * 100.0
    vector_cm = np.linalg.norm(error_cm, axis=1)
    return {
        "sample_count": int(error_cm.shape[0]),
        "x_cm": _scalar_error_statistics(error_cm[:, 0]),
        "y_cm": _scalar_error_statistics(error_cm[:, 1]),
        "xy_vector_cm": {
            "P50": float(np.quantile(vector_cm, 0.50)),
            "P90": float(np.quantile(vector_cm, 0.90)),
            "P95": float(np.quantile(vector_cm, 0.95)),
            "max": float(np.max(vector_cm)),
        },
    }


def query_with_replaced_com_velocity(
    query: Any,
    estimated_com_velocity_xy: Iterable[float],
    true_com_velocity_xy: Iterable[float],
    omega: float,
) -> Any:
    """Clone a certificate query while changing only its velocity contribution to b."""

    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and positive")
    estimate = np.asarray(estimated_com_velocity_xy, dtype=np.float64)
    reference = np.asarray(true_com_velocity_xy, dtype=np.float64)
    if estimate.shape != (2,) or reference.shape != (2,):
        raise ValueError("estimated and true CoM velocities must be 2D")
    replacement_b = np.asarray(query.b, dtype=np.float64) + (estimate - reference) / omega
    return replace(query, b=replacement_b)


def certificate_agreement(samples: Iterable[dict[str, Any]], estimate_name: str) -> dict[str, Any]:
    """Compare one estimated certificate stream with the GT stream."""

    n_key = f"N_{estimate_name}"
    margin_key = f"margin_{estimate_name}"
    valid_key = f"certificate_valid_{estimate_name}"
    rows = [
        row
        for row in samples
        if row.get("certificate_valid_GT", False)
        and row.get(valid_key, False)
        and row.get("N_GT") is not None
        and row.get(n_key) is not None
        and row.get("margin_GT") is not None
        and row.get(margin_key) is not None
    ]
    if not rows:
        return {"sample_count": 0}
    gt_n = np.asarray([row["N_GT"] for row in rows], dtype=np.int64)
    estimated_n = np.asarray([row[n_key] for row in rows], dtype=np.int64)
    gt_margin = np.asarray([row["margin_GT"] for row in rows], dtype=np.float64)
    estimated_margin = np.asarray([row[margin_key] for row in rows], dtype=np.float64)
    n_error = np.abs(estimated_n - gt_n)
    margin_error = estimated_margin - gt_margin
    gt_feasible_estimate_over_horizon = (gt_n <= 5) & (estimated_n == 6)
    gt_over_horizon_estimate_feasible = (gt_n == 6) & (estimated_n <= 5)
    confusion = {
        str(gt): {str(estimate): int(np.sum((gt_n == gt) & (estimated_n == estimate))) for estimate in range(7)}
        for gt in range(7)
    }
    return {
        "sample_count": len(rows),
        "N_exact_agreement": float(np.mean(estimated_n == gt_n)),
        "N_within_one_agreement": float(np.mean(n_error <= 1)),
        "N_absolute_error_median": float(np.median(n_error)),
        "N_absolute_error_mean": float(np.mean(n_error)),
        "N_confusion_matrix_GT_rows_estimate_columns": confusion,
        "margin_MAE": float(np.mean(np.abs(margin_error))),
        "margin_RMSE": float(np.sqrt(np.mean(np.square(margin_error)))),
        "margin_spearman": _safe_correlation(
            estimated_margin, gt_margin, "spearman"
        ),
        "margin_sign_agreement": float(
            np.mean(np.sign(estimated_margin) == np.sign(gt_margin))
        ),
        "false_classification": {
            "GT_feasible_estimate_over_horizon_count": int(
                np.sum(gt_feasible_estimate_over_horizon)
            ),
            "GT_feasible_estimate_over_horizon_fraction": float(
                np.mean(gt_feasible_estimate_over_horizon)
            ),
            "GT_over_horizon_estimate_feasible_count": int(
                np.sum(gt_over_horizon_estimate_feasible)
            ),
            "GT_over_horizon_estimate_feasible_fraction": float(
                np.mean(gt_over_horizon_estimate_feasible)
            ),
        },
    }


def terminal_ordering(
    samples: Iterable[dict[str, Any]],
    estimate_name: str,
) -> dict[str, Any]:
    """Compute TD0 certificate ordering against terminal recovery steps."""

    n_key = f"N_{estimate_name}"
    margin_key = f"margin_{estimate_name}"
    valid_key = f"certificate_valid_{estimate_name}"
    rows = [
        row
        for row in samples
        if row.get("touchdown") == 0
        and row.get(valid_key, False)
        and row.get(n_key) is not None
        and row.get(margin_key) is not None
        and row.get("N_actual_terminal") is not None
    ]
    if not rows:
        return {"sample_count": 0, "N_vs_terminal_spearman": None, "margin_vs_terminal_spearman": None}
    terminal = np.asarray([row["N_actual_terminal"] for row in rows], dtype=np.float64)
    return {
        "sample_count": len(rows),
        "N_vs_terminal_spearman": _safe_correlation(
            np.asarray([row[n_key] for row in rows], dtype=np.float64),
            terminal,
            "spearman",
        ),
        "margin_vs_terminal_spearman": _safe_correlation(
            np.asarray([row[margin_key] for row in rows], dtype=np.float64),
            terminal,
            "spearman",
        ),
    }


__all__ = [
    "certificate_agreement",
    "dcm_velocity_error_statistics",
    "query_with_replaced_com_velocity",
    "terminal_ordering",
    "velocity_error_statistics",
]
