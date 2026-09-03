"""Flat-equivalence and no-double-projection tests for plane runtime."""

from __future__ import annotations

from dataclasses import replace
from concurrent.futures import Future
import numpy as np
import pytest
from types import SimpleNamespace
import torch

from legged_lab.recovery.g1_certificate_runtime import CalibratedG1CertificateEvaluator
from legged_lab.recovery.certificate import CertificateResult, CertificateStatus
from legged_lab.recovery.plane_certificate_runtime import (
    PendingPlaneCertificateBatch,
    PlaneCalibratedG1CertificateEvaluator,
    PlaneCertificateQuery,
    mirror_plane_certificate_query,
    plane_periodic_state,
)
from legged_lab.recovery.plane_nominal_params import PlaneNominalParameterTable


FLAT = "tools/recovery/generated/g1_recovery_params.yaml"
NOMINAL = "tools/recovery/generated/g1_plane_nominal_params.yaml"


def _valid_slope_query() -> PlaneCertificateQuery:
    return PlaneCertificateQuery(
        command=np.asarray((0.4, 0.0, 0.0)),
        b=np.asarray((0.11, -0.07)),
        q=np.asarray((-0.15, -0.22)),
        support_side="left",
        phase=0.0,
        alpha=np.deg2rad(-10.0),
        adapter_valid=True,
    )


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


def test_plane_mirror_uses_exact_submit_query_instead_of_state_b() -> None:
    evaluator = PlaneCalibratedG1CertificateEvaluator(
        FLAT, NOMINAL, workers=1, executor_type="sequential"
    )
    alpha = np.deg2rad(-10.0)
    com = torch.tensor([[0.30, -0.10]])
    velocity = torch.tensor([[0.20, 0.05]])
    support = torch.tensor([[0.04, -0.12]])
    state = SimpleNamespace(
        command_velocity=torch.tensor([[0.40, 0.0, 0.0]]),
        signed_slope=torch.tensor([alpha]),
        terrain_plane_valid=torch.tensor([True]),
        com_position=com,
        com_velocity=velocity,
        left_foot_position=support,
        right_foot_position=torch.zeros_like(support),
        q=torch.tensor([[-0.18, 0.22]]),
        support_is_left=torch.tensor([True]),
        b=torch.tensor([[99.0, 99.0]]),
    )
    try:
        pending = evaluator.submit(state, torch.tensor([0]))
        query = pending.queries[0]
        nominal = evaluator.lookup_nominal((0.4, 0.0, 0.0), alpha).value
        assert nominal is not None
        expected_b = com.numpy()[0] + velocity.numpy()[0] / nominal.omega - support.numpy()[0]
        mirrored = mirror_plane_certificate_query(query)
    finally:
        evaluator.close()

    np.testing.assert_allclose(query.b, expected_b)
    assert not np.array_equal(query.b, state.b.numpy()[0])
    np.testing.assert_allclose(mirrored.b, (query.b[0], -query.b[1]))
    np.testing.assert_allclose(mirrored.q, (query.q[0], -query.q[1]))
    assert mirrored.support_side == "right"
    assert mirrored.alpha == query.alpha


def test_plane_config_failure_is_logged_without_secondary_exception(monkeypatch) -> None:
    evaluator = PlaneCalibratedG1CertificateEvaluator(
        FLAT, NOMINAL, workers=1, executor_type="sequential"
    )
    query = _valid_slope_query()

    def fail_config(*_args, **_kwargs):
        raise ValueError("synthetic plane config failure")

    monkeypatch.setattr(evaluator, "_plane_config", fail_config)
    result = evaluator._solve(query)
    pending = PendingPlaneCertificateBatch(
        (7,), (query,), (result,), torch.device("cpu")
    )
    try:
        n_min, margin, valid = evaluator.resolve_with_validity(pending)
    finally:
        evaluator.close()

    assert n_min.tolist() == [6]
    assert margin.tolist() == pytest.approx([-3.0])
    assert valid.tolist() == [False]
    record = evaluator.failure_records[-1]
    assert "synthetic plane config failure" in record["result"]["message"]
    assert record["configuration_lookup_error"] == {
        "type": "ValueError",
        "message": "synthetic plane config failure",
    }


def test_constraint_builder_mismatch_is_fatal_and_logged() -> None:
    evaluator = PlaneCalibratedG1CertificateEvaluator(
        FLAT, NOMINAL, workers=1, executor_type="sequential"
    )
    query = _valid_slope_query()
    mismatch = CertificateResult(
        CertificateStatus.CONSTRAINT_BUILDER_MISMATCH,
        None,
        None,
        None,
        (),
        "synthetic builder mismatch",
        diagnostic={"kind": "synthetic_builder_mismatch"},
    )
    pending = PendingPlaneCertificateBatch(
        (3,), (query,), (mismatch,), torch.device("cpu")
    )
    try:
        with pytest.raises(RuntimeError, match="constraint builder mismatch"):
            evaluator.resolve_with_validity(pending)
    finally:
        evaluator.close()
    assert evaluator.failure_records[-1]["result"]["status"] == (
        CertificateStatus.CONSTRAINT_BUILDER_MISMATCH.value
    )


def test_one_transport_failure_does_not_abort_remaining_batch() -> None:
    evaluator = PlaneCalibratedG1CertificateEvaluator(
        FLAT, NOMINAL, workers=1, executor_type="sequential"
    )
    query = _valid_slope_query()
    failed: Future = Future()
    failed.set_exception(BrokenPipeError("synthetic broken pipe"))
    normal = evaluator._solve(query)
    pending = PendingPlaneCertificateBatch(
        (0, 1), (query, query), (failed, normal), torch.device("cpu")
    )
    try:
        n_min, margin, valid = evaluator.resolve_with_validity(pending)
    finally:
        evaluator.close()

    assert n_min[0].item() == 6
    assert margin[0].item() == pytest.approx(-3.0)
    assert not valid[0].item()
    assert valid[1].item()
    assert evaluator.failure_records[0]["failure"]["kind"] == "worker_transport_failure"


def test_nondefault_z_sole_matches_sequential_and_subprocess() -> None:
    query = _valid_slope_query()
    sequential = PlaneCalibratedG1CertificateEvaluator(
        FLAT, NOMINAL, workers=1, executor_type="sequential", z_sole=-0.05
    )
    subprocess_evaluator = PlaneCalibratedG1CertificateEvaluator(
        FLAT, NOMINAL, workers=1, executor_type="subprocess", z_sole=-0.05
    )
    try:
        expected = sequential._solve(query)
        pending = subprocess_evaluator.submit_queries((query,), torch.device("cpu"))
        actual_n, actual_margin, actual_valid = subprocess_evaluator.resolve_with_validity(
            pending
        )
    finally:
        sequential.close()
        subprocess_evaluator.close()

    assert actual_valid.tolist() == [True]
    assert actual_n.item() == expected.n_min
    assert actual_margin.item() == pytest.approx(expected.margin, abs=1.0e-6)


def test_plane_query_mirror_smoke_preserves_certificate() -> None:
    evaluator = PlaneCalibratedG1CertificateEvaluator(
        FLAT, NOMINAL, workers=1, executor_type="sequential"
    )
    alpha = np.deg2rad(-10.0)
    lookup = evaluator.lookup_nominal((0.4, 0.0, 0.0), alpha)
    assert lookup.valid and lookup.value is not None
    nominal = lookup.value
    theory = plane_periodic_state(
        0.4, 0.0, nominal.step_period, nominal.omega, nominal.step_width
    )
    query = PlaneCertificateQuery(
        command=np.asarray((0.4, 0.0, 0.0)),
        b=np.asarray(theory["b_left"]),
        q=np.asarray(theory["q_left"]),
        support_side="left",
        phase=0.0,
        alpha=alpha,
        adapter_valid=True,
    )
    mirrored = mirror_plane_certificate_query(query)
    try:
        pending = evaluator.submit_queries((query, mirrored), torch.device("cpu"))
        n_min, margin, valid = evaluator.resolve_with_validity(pending)
    finally:
        evaluator.close()

    assert mirrored.alpha == query.alpha
    assert mirrored.phase == 0.0
    assert mirrored.support_side == "right"
    assert valid.tolist() == [True, True]
    assert n_min[0].item() == n_min[1].item()
    assert margin[0].item() == pytest.approx(margin[1].item(), abs=1.0e-8)
