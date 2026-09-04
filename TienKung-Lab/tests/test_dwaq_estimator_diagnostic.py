"""Unit tests for DWAQ estimator/certificate diagnostic helpers."""

from __future__ import annotations

import numpy as np
import pytest

from legged_lab.recovery.dwaq_estimator_diagnostic import (
    certificate_agreement,
    dcm_velocity_error_statistics,
    query_with_replaced_com_velocity,
    terminal_ordering,
    velocity_error_statistics,
)
from legged_lab.recovery.plane_certificate_runtime import PlaneCertificateQuery


def test_velocity_statistics_use_estimate_minus_gt_and_vector_norm() -> None:
    gt = np.asarray(((1.0, 2.0, 3.0), (2.0, 4.0, 6.0)))
    estimate = gt + np.asarray(((0.1, -0.2, 0.3), (-0.1, 0.2, -0.3)))
    report = velocity_error_statistics(estimate, gt)

    assert report["sample_count"] == 2
    assert report["components"]["x"]["mae"] == pytest.approx(0.1)
    assert report["components"]["x"]["mean_bias"] == pytest.approx(0.0)
    assert report["components"]["z"]["rmse"] == pytest.approx(0.3)
    assert report["xy_vector_error"]["P95"] == pytest.approx(np.hypot(0.1, 0.2))


def test_replacing_velocity_changes_only_certificate_b() -> None:
    query = PlaneCertificateQuery(
        command=np.asarray((0.4, 0.0, 0.0)),
        b=np.asarray((0.20, -0.10)),
        q=np.asarray((-0.15, 0.22)),
        support_side="left",
        phase=0.0,
        alpha=0.1,
        adapter_valid=True,
    )
    replaced = query_with_replaced_com_velocity(
        query,
        estimated_com_velocity_xy=(0.7, -0.1),
        true_com_velocity_xy=(0.4, 0.2),
        omega=3.0,
    )

    np.testing.assert_allclose(replaced.b, query.b + np.asarray((0.1, -0.1)))
    np.testing.assert_array_equal(replaced.command, query.command)
    np.testing.assert_array_equal(replaced.q, query.q)
    assert replaced.support_side == query.support_side
    assert replaced.alpha == query.alpha
    assert replaced.adapter_valid == query.adapter_valid


def test_dcm_error_is_velocity_error_divided_by_omega_in_cm() -> None:
    report = dcm_velocity_error_statistics(
        ((0.6, 0.0), (0.2, -0.4)),
        ((0.3, 0.0), (0.2, 0.2)),
        (3.0, 3.0),
    )

    assert report["x_cm"]["absolute_error_max"] == pytest.approx(10.0)
    assert report["y_cm"]["absolute_error_max"] == pytest.approx(20.0)
    assert report["xy_vector_cm"]["max"] == pytest.approx(20.0)


def test_certificate_agreement_and_terminal_ordering() -> None:
    rows = [
        {
            "touchdown": 0,
            "N_GT": 1,
            "N_direct": 1,
            "margin_GT": 0.4,
            "margin_direct": 0.3,
            "certificate_valid_GT": True,
            "certificate_valid_direct": True,
            "N_actual_terminal": 1,
        },
        {
            "touchdown": 0,
            "N_GT": 3,
            "N_direct": 4,
            "margin_GT": -0.2,
            "margin_direct": -0.1,
            "certificate_valid_GT": True,
            "certificate_valid_direct": True,
            "N_actual_terminal": 4,
        },
        {
            "touchdown": 1,
            "N_GT": 0,
            "N_direct": 2,
            "margin_GT": 0.6,
            "margin_direct": 0.2,
            "certificate_valid_GT": True,
            "certificate_valid_direct": True,
            "N_actual_terminal": 4,
        },
    ]
    agreement = certificate_agreement(rows, "direct")
    ordering = terminal_ordering(rows, "direct")

    assert agreement["N_exact_agreement"] == pytest.approx(1.0 / 3.0)
    assert agreement["N_within_one_agreement"] == pytest.approx(2.0 / 3.0)
    assert agreement["N_confusion_matrix_GT_rows_estimate_columns"]["3"]["4"] == 1
    assert agreement["margin_sign_agreement"] == 1.0
    assert agreement["false_classification"]["GT_feasible_estimate_over_horizon_count"] == 0
    assert agreement["false_classification"]["GT_over_horizon_estimate_feasible_count"] == 0
    assert ordering["sample_count"] == 2
    assert ordering["N_vs_terminal_spearman"] == pytest.approx(1.0)


def test_certificate_agreement_counts_feasibility_false_classifications() -> None:
    rows = [
        {
            "N_GT": 4,
            "N_EST": 6,
            "margin_GT": 0.1,
            "margin_EST": -0.2,
            "certificate_valid_GT": True,
            "certificate_valid_EST": True,
        },
        {
            "N_GT": 6,
            "N_EST": 5,
            "margin_GT": -0.3,
            "margin_EST": 0.1,
            "certificate_valid_GT": True,
            "certificate_valid_EST": True,
        },
    ]
    agreement = certificate_agreement(rows, "EST")
    false = agreement["false_classification"]
    assert false["GT_feasible_estimate_over_horizon_count"] == 1
    assert false["GT_over_horizon_estimate_feasible_count"] == 1
    assert false["GT_feasible_estimate_over_horizon_fraction"] == pytest.approx(0.5)
    assert false["GT_over_horizon_estimate_feasible_fraction"] == pytest.approx(0.5)
