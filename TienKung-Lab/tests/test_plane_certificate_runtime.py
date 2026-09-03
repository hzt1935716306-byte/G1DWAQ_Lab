"""Flat-equivalence and no-double-projection tests for plane runtime."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
from types import SimpleNamespace
import torch

from legged_lab.recovery.g1_certificate_runtime import CalibratedG1CertificateEvaluator
from legged_lab.recovery.plane_certificate_runtime import (
    PlaneCalibratedG1CertificateEvaluator,
    PlaneCertificateQuery,
)
from legged_lab.recovery.plane_nominal_params import PlaneNominalParameterTable


FLAT = "tools/recovery/generated/g1_recovery_params.yaml"
NOMINAL = "tools/recovery/generated/g1_plane_nominal_params.yaml"


def test_flat_old_new_certificate_N_and_margin_match() -> None:
    old = CalibratedG1CertificateEvaluator(FLAT, workers=1, executor_type="sequential")
    new = PlaneCalibratedG1CertificateEvaluator(
        FLAT, NOMINAL, workers=1, executor_type="sequential"
    )
    command = np.asarray((0.6, 0.0, 0.0))
    b = np.asarray((0.11, -0.07))
    q = np.asarray((-0.15, -0.22))
    try:
        old_result = old._solve((command, b, q, "left", 0.0))
        new_result = new._solve(
            PlaneCertificateQuery(command, b, q, "left", 0.0, 0.0, True)
        )
    finally:
        old.close()
        new.close()
    assert new_result.n_min == old_result.n_min
    assert new_result.margin == pytest.approx(old_result.margin, abs=1.0e-10)


def test_plane_query_keeps_measured_b_and_q_unprojected() -> None:
    evaluator = PlaneCalibratedG1CertificateEvaluator(
        FLAT, NOMINAL, workers=1, executor_type="sequential"
    )
    b = np.asarray((0.123, -0.234))
    q = np.asarray((-0.345, 0.210))
    query = PlaneCertificateQuery(
        np.asarray((0.6, 0.0, 0.0)), b, q, "right", 0.0, 0.0, True
    )
    try:
        np.testing.assert_array_equal(query.b, b)
        np.testing.assert_array_equal(query.q, q)
    finally:
        evaluator.close()


def test_submit_does_not_project_heading_horizontal_measurements() -> None:
    evaluator = PlaneCalibratedG1CertificateEvaluator(
        FLAT, NOMINAL, workers=1, executor_type="sequential"
    )
    alpha = np.deg2rad(10.0)
    flat_node = next(
        node
        for node in evaluator.nominal_table.nodes
        if node.alpha == 0.0 and node.direction == "+x" and node.speed == 0.2
    )
    slope_node = replace(flat_node, alpha=alpha)
    evaluator.nominal_table = PlaneNominalParameterTable((slope_node,))
    com = torch.tensor([[0.30, -0.10]])
    velocity = torch.tensor([[0.20, 0.05]])
    support = torch.tensor([[0.04, -0.12]])
    q = torch.tensor([[-0.18, 0.22]])
    state = SimpleNamespace(
        command_velocity=torch.tensor([[0.20, 0.0, 0.0]]),
        signed_slope=torch.tensor([alpha]),
        terrain_plane_valid=torch.tensor([True]),
        com_position=com,
        com_velocity=velocity,
        left_foot_position=support,
        right_foot_position=torch.zeros_like(support),
        q=q,
        support_is_left=torch.tensor([True]),
        b=torch.zeros_like(q),
    )
    try:
        pending = evaluator.submit(state, torch.tensor([0]))
        expected_b = com.numpy()[0] + velocity.numpy()[0] / slope_node.omega - support.numpy()[0]
        np.testing.assert_allclose(pending.queries[0].b, expected_b)
        np.testing.assert_array_equal(pending.queries[0].q, q.numpy()[0])
    finally:
        evaluator.close()
