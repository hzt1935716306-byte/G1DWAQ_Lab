from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import OptimizeResult


@pytest.fixture()
def certificate_module():
    """Load the pure LP module without importing Isaac Lab runtime modules."""

    path = Path(__file__).resolve().parents[1] / "legged_lab/recovery/certificate.py"
    name = "_test_certificate_solver_robustness"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(name, None)


def _state(module):
    return module.CertificateState(
        b=np.array((0.0, 0.0)),
        q=np.array((0.0, 0.0)),
        support_side="left",
        phase=0.0,
    )


def _failed_result(status: int, message: str) -> OptimizeResult:
    return OptimizeResult(success=False, status=status, message=message, nit=0)


def test_normal_results_are_unchanged(certificate_module) -> None:
    module = certificate_module
    config = module.RecoverabilityConfig()

    finite = module.certify_recoverability(_state(module), config)
    assert finite.status == module.CertificateStatus.FINITE
    assert finite.n_min == 1
    assert finite.margin == pytest.approx(0.49053116461859325)
    assert finite.feasible_horizons == (False, True)
    assert not finite.margin_fallback
    assert not finite.solver_fallback
    assert not finite.solver_retried

    over_horizon = module.certify_recoverability(
        module.CertificateState(
            b=np.array((0.2, 0.1)),
            q=np.array((-0.25, -0.22)),
            support_side="left",
            phase=0.25,
        ),
        config,
    )
    assert over_horizon.status == module.CertificateStatus.OVER_HORIZON
    assert over_horizon.n_min == 6
    assert over_horizon.margin == pytest.approx(-1.1451300128228141)
    assert not over_horizon.margin_fallback
    assert not over_horizon.solver_fallback


def test_valid_rho_zero_witness_uses_margin_fallback(certificate_module, monkeypatch) -> None:
    module = certificate_module
    original = module._solve_with_margin_bound

    def fail_inset(problem, mode, bound, *, retry=False):
        if mode == "inset":
            return _failed_result(2, "forced inset infeasible")
        return original(problem, mode, bound, retry=retry)

    monkeypatch.setattr(module, "_solve_with_margin_bound", fail_inset)
    result = module.certify_recoverability(_state(module), module.RecoverabilityConfig())

    assert result.status == module.CertificateStatus.FINITE
    assert result.n_min == 1
    assert result.margin == 0.0
    assert result.margin_fallback
    assert not result.solver_fallback
    assert result.solver_retried
    assert result.witness is not None and result.witness.kind == "feasible"
    assert result.diagnostic["margin_zero_residual"]["satisfied"] is True
    assert result.diagnostic["solver"]["initial"]["status"] == 2
    assert result.diagnostic["solver"]["retry"]["status"] == 2


def test_margin_builder_mismatch_is_not_hidden_by_fallback(
    certificate_module,
    monkeypatch,
) -> None:
    module = certificate_module
    original_matrices = module._margin_problem_matrices

    def mismatched_matrices(problem, mode):
        a_ub, b_ub, a_eq, b_eq = original_matrices(problem, mode)
        if mode == "inset":
            b_ub = b_ub.copy()
            b_ub[0] = -1.0
        return a_ub, b_ub, a_eq, b_eq

    def fail_margin(*_args, **_kwargs):
        return _failed_result(2, "forced margin failure")

    monkeypatch.setattr(module, "_margin_problem_matrices", mismatched_matrices)
    monkeypatch.setattr(module, "_solve_with_margin_bound", fail_margin)
    result = module.certify_recoverability(_state(module), module.RecoverabilityConfig())

    assert result.status == module.CertificateStatus.CONSTRAINT_BUILDER_MISMATCH
    assert result.n_min is None
    assert result.margin is None
    assert not result.margin_fallback
    assert not result.solver_fallback
    assert result.diagnostic["margin_zero_residual"]["satisfied"] is False


def test_numerical_feasibility_failure_retries_then_falls_back(
    certificate_module,
    monkeypatch,
) -> None:
    module = certificate_module
    calls: list[bool] = []

    def fail_feasibility(_problem, *, retry=False):
        calls.append(retry)
        return _failed_result(4, "forced numerical failure")

    monkeypatch.setattr(module, "_solve_feasibility", fail_feasibility)
    config = module.RecoverabilityConfig()
    result = module.certify_recoverability(_state(module), config)

    assert calls == [False, True]
    assert result.status == module.CertificateStatus.OVER_HORIZON
    assert result.n_min == 6
    assert result.margin == -config.eta_max
    assert not result.margin_fallback
    assert result.solver_fallback
    assert result.solver_retried
    assert result.diagnostic["kind"] == "feasibility_solver_failure"


def test_retry_success_preserves_the_normal_certificate(certificate_module, monkeypatch) -> None:
    module = certificate_module
    original = module._solve_feasibility

    def fail_once(problem, *, retry=False):
        if not retry:
            return _failed_result(4, "forced first-attempt failure")
        return original(problem, retry=True)

    monkeypatch.setattr(module, "_solve_feasibility", fail_once)
    result = module.certify_recoverability(_state(module), module.RecoverabilityConfig())

    assert result.status == module.CertificateStatus.FINITE
    assert result.n_min == 1
    assert result.margin == pytest.approx(0.49053116461859325)
    assert result.solver_retried
    assert not result.margin_fallback
    assert not result.solver_fallback
