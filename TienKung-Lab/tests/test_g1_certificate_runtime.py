from __future__ import annotations

import importlib
import math
from pathlib import Path
import threading
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch


@pytest.fixture()
def runtime_module():
    """Import the runtime adapter without loading Isaac Lab's simulation modules."""

    fake_extractor = ModuleType("legged_lab.recovery.state_extractor")
    fake_extractor.G1PrivilegedRecoveryState = object

    def theoretical_periodic_state(speed, vy, period, omega, width):
        gain = math.exp(omega * period)
        landing_left = (speed * period, vy * period - width)
        landing_right = (speed * period, vy * period + width)
        denominator = gain * gain - 1.0
        b_left = tuple(
            (gain * landing_left[axis] + landing_right[axis]) / denominator
            for axis in range(2)
        )
        b_right = tuple(
            (landing_left[axis] + gain * landing_right[axis]) / denominator
            for axis in range(2)
        )
        return {
            "landing_left": landing_left,
            "landing_right": landing_right,
            "b_left": b_left,
            "b_right": b_right,
            "q_left": tuple(-value for value in landing_right),
            "q_right": tuple(-value for value in landing_left),
        }

    fake_extractor.theoretical_periodic_state = theoretical_periodic_state
    module_name = "legged_lab.recovery.g1_certificate_runtime"
    previous_extractor = sys.modules.get(fake_extractor.__name__)
    previous_runtime = sys.modules.pop(module_name, None)
    sys.modules[fake_extractor.__name__] = fake_extractor
    try:
        yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if previous_runtime is not None:
            sys.modules[module_name] = previous_runtime
        if previous_extractor is None:
            sys.modules.pop(fake_extractor.__name__, None)
        else:
            sys.modules[fake_extractor.__name__] = previous_extractor


def _state():
    return SimpleNamespace(
        b=torch.tensor([[0.1, -0.2]], dtype=torch.float32),
        q=torch.tensor([[-0.1, -0.22]], dtype=torch.float32),
        command_velocity=torch.tensor([[0.6, 0.0, 0.1]], dtype=torch.float32),
        phase=torch.tensor([0.25], dtype=torch.float32),
        support_is_left=torch.tensor([True]),
    )


def _state_batch(seed: int, count: int = 8):
    generator = torch.Generator().manual_seed(seed)
    command_velocity = torch.zeros(count, 3, dtype=torch.float32)
    command_velocity[:, 0] = 2.0 * torch.rand(count, generator=generator) - 1.0
    return SimpleNamespace(
        b=1.2 * torch.rand(count, 2, generator=generator) - 0.6,
        q=0.8 * torch.rand(count, 2, generator=generator) - 0.4,
        command_velocity=command_velocity,
        phase=0.95 * torch.rand(count, generator=generator),
        support_is_left=torch.rand(count, generator=generator) > 0.5,
    )


def _raw_solver_failure(module):
    return module.CertificateResult(
        module.CertificateStatus.SOLVER_FAILURE,
        None,
        None,
        None,
        (),
        "forced solver failure",
        solver_retried=True,
        diagnostic={
            "kind": "forced_solver_failure",
            "attempted_horizon": 3,
            "solver": {
                "initial": {"success": False, "status": 4, "message": "failed", "nit": 0},
                "retry": {"success": False, "status": 4, "message": "failed", "nit": 0},
            },
            "witness": None,
            "witness_residual": None,
            "margin_zero_residual": None,
        },
    )


def test_one_environment_failure_falls_back_and_writes_reproducer(
    runtime_module,
    monkeypatch,
    tmp_path,
) -> None:
    module = runtime_module
    evaluator = module.CalibratedG1CertificateEvaluator(
        Path("tools/recovery/generated/g1_recovery_params.yaml"),
        workers=1,
        failure_window_size=4,
        failure_rate_threshold=0.5,
    )
    evaluator.configure_diagnostics(tmp_path)
    monkeypatch.setattr(evaluator, "_solve", lambda _query: _raw_solver_failure(module))

    n_min, margin = evaluator.evaluate(_state(), torch.tensor([0]))

    assert n_min.tolist() == [6]
    assert margin.tolist() == pytest.approx([-3.0])
    assert evaluator.statistics["fallbacks"] == 1
    assert evaluator.statistics["solver_fallbacks"] == 1
    record = evaluator.failure_records[0]
    assert record["input"]["b"] == pytest.approx([0.1, -0.2])
    assert record["input"]["q"] == pytest.approx([-0.1, -0.22])
    assert record["input"]["phase"] == pytest.approx(0.25)
    assert record["input"]["T"] > 0.0
    assert record["input"]["support"] == "left"
    assert record["input"]["command"] == pytest.approx([0.6, 0.0, 0.1])
    assert record["result"]["N"] == 6
    assert record["failure"]["solver"]["retry"]["status"] == 4
    assert (tmp_path / "certificate_solver_fallbacks.jsonl").is_file()
    evaluator.close()


def test_global_error_requires_elevated_window_failure_rate(
    runtime_module,
    monkeypatch,
) -> None:
    module = runtime_module
    evaluator = module.CalibratedG1CertificateEvaluator(
        Path("tools/recovery/generated/g1_recovery_params.yaml"),
        workers=1,
        failure_window_size=3,
        failure_rate_threshold=0.5,
    )
    monkeypatch.setattr(evaluator, "_solve", lambda _query: _raw_solver_failure(module))

    evaluator.evaluate(_state(), torch.tensor([0]))
    evaluator.evaluate(_state(), torch.tensor([0]))
    with pytest.raises(RuntimeError, match="fallback rate exceeded"):
        evaluator.evaluate(_state(), torch.tensor([0]))
    assert evaluator.statistics["fallbacks"] == 3
    evaluator.close()


def test_submit_returns_before_worker_finishes_and_resolve_preserves_order(
    runtime_module,
    monkeypatch,
) -> None:
    module = runtime_module
    evaluator = module.CalibratedG1CertificateEvaluator(
        Path("tools/recovery/generated/g1_recovery_params.yaml"),
        workers=2,
    )
    started = threading.Event()
    release = threading.Event()

    def blocked_result(_query):
        started.set()
        if not release.wait(timeout=2.0):
            raise RuntimeError("test did not release the certificate worker")
        return module.CertificateResult(
            module.CertificateStatus.FINITE,
            3,
            0.2,
            None,
            (False, False, True),
        )

    monkeypatch.setattr(evaluator, "_solve", blocked_result)
    try:
        pending = evaluator.submit(_state(), torch.tensor([0]))
        assert started.wait(timeout=1.0)
        assert not pending.jobs[0].done()
        release.set()
        n_min, margin = evaluator.resolve(pending)
        assert n_min.tolist() == [3]
        assert margin.tolist() == pytest.approx([0.2])
        assert evaluator.statistics["evaluations"] == 1
    finally:
        release.set()
        evaluator.close()


def test_synchronous_evaluate_uses_submit_resolve_result(runtime_module) -> None:
    module = runtime_module
    direct = module.CalibratedG1CertificateEvaluator(
        Path("tools/recovery/generated/g1_recovery_params.yaml"),
        workers=2,
    )
    deferred = module.CalibratedG1CertificateEvaluator(
        Path("tools/recovery/generated/g1_recovery_params.yaml"),
        workers=2,
    )
    try:
        expected_n, expected_margin = direct.evaluate(_state(), torch.tensor([0]))
        pending = deferred.submit(_state(), torch.tensor([0]))
        actual_n, actual_margin = deferred.resolve(pending)
        assert torch.equal(actual_n, expected_n)
        assert torch.equal(actual_margin, expected_margin)
        assert deferred.statistics == direct.statistics
    finally:
        direct.close()
        deferred.close()


def test_many_submitted_batches_match_synchronous_results(runtime_module) -> None:
    module = runtime_module
    direct = module.CalibratedG1CertificateEvaluator(
        Path("tools/recovery/generated/g1_recovery_params.yaml"),
        workers=4,
    )
    deferred = module.CalibratedG1CertificateEvaluator(
        Path("tools/recovery/generated/g1_recovery_params.yaml"),
        workers=4,
    )
    states = [_state_batch(seed) for seed in range(6)]
    env_ids = torch.arange(8)
    try:
        expected = [direct.evaluate(state, env_ids) for state in states]
        pending = [deferred.submit(state, env_ids) for state in states]
        actual = [deferred.resolve(batch) for batch in pending]
        for (expected_n, expected_margin), (actual_n, actual_margin) in zip(expected, actual):
            assert torch.equal(actual_n, expected_n)
            assert torch.equal(actual_margin, expected_margin)
        assert deferred.statistics == direct.statistics
    finally:
        direct.close()
        deferred.close()
