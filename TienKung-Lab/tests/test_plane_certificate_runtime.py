"""Flat-equivalence and no-double-projection tests for plane runtime."""

from __future__ import annotations

from dataclasses import replace
from concurrent.futures import Future
import math
import numpy as np
import pytest
from types import SimpleNamespace
import torch

from legged_lab.recovery.g1_certificate_runtime import CalibratedG1CertificateEvaluator
from legged_lab.recovery.certificate import (
    CertificateResult,
    CertificateState,
    CertificateStatus,
    RecoverabilityConfig,
    certify_recoverability,
    check_witness,
)
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


def test_dcm_touchdown_map_matches_exact_lipm_solution() -> None:
    """Compare the touchdown map with an independently written COM exact solution."""

    rng = np.random.default_rng(20260903)
    for _ in range(200):
        omega = rng.uniform(2.5, 5.0)
        duration = rng.uniform(0.15, 0.85)
        x0 = rng.uniform((-0.6, -0.4), (0.6, 0.4))
        v0 = rng.uniform((-1.5, -1.0), (1.5, 1.0))
        cop = rng.uniform((-0.2, -0.1), (0.2, 0.1))
        landing = rng.uniform((-0.4, -0.35), (0.7, 0.35))

        cosh = math.cosh(omega * duration)
        sinh = math.sinh(omega * duration)
        x_t = cop + (x0 - cop) * cosh + v0 / omega * sinh
        v_t = omega * (x0 - cop) * sinh + v0 * cosh
        b_next_exact = x_t + v_t / omega - landing

        b0 = x0 + v0 / omega
        gamma = math.exp(omega * duration)
        b_next_map = gamma * b0 - (gamma - 1.0) * cop - landing
        np.testing.assert_allclose(
            b_next_exact, b_next_map, rtol=0.0, atol=1.0e-10
        )


MINIMUM_HORIZON_FIXTURES = (
    (1, (0.11468668922907482, -0.017770918712626156), (-0.29518028993545287, -0.07085552148513202), "left"),
    (2, (-0.07409775965760608, -0.016397340048072175), (-0.09942396215881749, 0.3030428883419154), "right"),
    (3, (-0.1083835224496881, 0.022192639732205498), (-0.12333928955608298, -0.1843322721393121), "left"),
    (4, (0.09450187671019977, 0.08049955191062394), (-0.3714956358822298, 0.17784993297293555), "right"),
    (5, (0.10232295491412502, 0.03053374126091995), (-0.36527460071758056, -0.14268756049907375), "left"),
)


@pytest.mark.parametrize(
    ("expected_n", "b", "q", "support_side"), MINIMUM_HORIZON_FIXTURES
)
def test_minimum_horizon_witness_decreases_by_one_after_first_step(
    expected_n: int,
    b: tuple[float, float],
    q: tuple[float, float],
    support_side: str,
) -> None:
    config = RecoverabilityConfig()
    state = CertificateState(
        np.asarray(b),
        np.asarray(q),
        support_side,
        phase=0.0,
        step_period=config.step_period,
        omega=config.omega,
    )
    result = certify_recoverability(state, config)

    assert result.n_min == expected_n
    assert result.witness is not None
    assert not result.solver_fallback
    assert not result.margin_fallback
    residual = check_witness(state, result.witness, config)
    assert residual.max_equality <= 1.0e-7
    assert residual.max_inequality_violation <= 1.0e-7
    assert residual.max_dynamics <= 1.0e-7
    assert residual.max_support_switch <= 1.0e-7

    next_state = CertificateState(
        b=result.witness.b[1],
        q=result.witness.q[1],
        support_side=result.witness.support_sides[1],
        phase=0.0,
        step_period=config.step_period,
        omega=config.omega,
    )
    next_result = certify_recoverability(next_state, config)
    assert next_result.n_min == expected_n - 1


def test_flat_old_new_monte_carlo_regression() -> None:
    rng = np.random.default_rng(20260903)
    speeds = np.asarray((0.2, 0.4, 0.6, 0.8, 1.0))
    old = CalibratedG1CertificateEvaluator(FLAT, workers=1, executor_type="sequential")
    new = PlaneCalibratedG1CertificateEvaluator(
        FLAT, NOMINAL, workers=1, executor_type="sequential"
    )
    normal_comparisons = 0
    fallback_count = 0
    n_distribution = {value: 0 for value in range(7)}
    max_margin_difference = 0.0
    try:
        for sample_index in range(200):
            speed = float(rng.choice(speeds))
            support_side = "left" if rng.random() < 0.5 else "right"
            config = old._config(speed)
            nominal_b, nominal_q = config.terminal_nominal(support_side)
            if sample_index < 20:
                b = nominal_b + rng.uniform(-0.5, 0.5, size=2) * np.asarray(
                    config.epsilon_b
                )
                q = nominal_q + rng.uniform(-0.5, 0.5, size=2) * np.asarray(
                    config.epsilon_q
                )
            else:
                b = nominal_b + rng.uniform((-0.22, -0.16), (0.22, 0.16))
                q = nominal_q + rng.uniform((-0.20, -0.16), (0.20, 0.16))
            command = np.asarray((speed, 0.0, 0.0))
            old_result = old._solve((command, b, q, support_side, 0.0))
            new_result = new._solve(
                PlaneCertificateQuery(command, b, q, support_side, 0.0, 0.0, True)
            )
            if (
                old_result.solver_fallback
                or old_result.margin_fallback
                or new_result.solver_fallback
                or new_result.margin_fallback
            ):
                fallback_count += 1
                continue
            assert old_result.n_min == new_result.n_min
            assert old_result.margin is not None and new_result.margin is not None
            difference = abs(old_result.margin - new_result.margin)
            assert difference <= 1.0e-8
            max_margin_difference = max(max_margin_difference, difference)
            normal_comparisons += 1
            assert old_result.n_min is not None
            n_distribution[int(old_result.n_min)] += 1
    finally:
        old.close()
        new.close()

    assert normal_comparisons >= 180, {
        "normal_comparisons": normal_comparisons,
        "fallback_count": fallback_count,
        "n_distribution": n_distribution,
        "max_margin_difference": max_margin_difference,
    }


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


def test_plane_v1_can_explicitly_use_recomputed_state_b_without_changing_legacy_default() -> None:
    evaluator = PlaneCalibratedG1CertificateEvaluator(
        FLAT,
        NOMINAL,
        workers=1,
        executor_type="sequential",
        use_state_b=True,
    )
    state_b = torch.tensor([[0.123, -0.456]])
    state = SimpleNamespace(
        command_velocity=torch.tensor([[0.40, 0.0, 0.0]]),
        signed_slope=torch.tensor([0.0]),
        terrain_plane_valid=torch.tensor([True]),
        com_position=torch.tensor([[9.0, 9.0]]),
        com_velocity=torch.tensor([[8.0, 8.0]]),
        left_foot_position=torch.tensor([[7.0, 7.0]]),
        right_foot_position=torch.zeros(1, 2),
        q=torch.tensor([[-0.18, 0.22]]),
        support_is_left=torch.tensor([True]),
        b=state_b,
    )
    try:
        pending = evaluator.submit(state, torch.tensor([0]))
        np.testing.assert_array_equal(pending.queries[0].b, state_b.numpy()[0])
    finally:
        evaluator.close()


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
