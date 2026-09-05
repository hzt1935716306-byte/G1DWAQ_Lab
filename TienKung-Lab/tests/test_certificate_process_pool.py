"""Worker lifecycle regressions for the clean certificate subprocess pool."""

from __future__ import annotations

import time

import numpy as np
import pytest

import legged_lab.recovery.certificate_process_pool as process_pool_module
from legged_lab.recovery.certificate import CertificateStatus
from legged_lab.recovery.certificate_process_pool import CertificateProcessPool
from legged_lab.recovery.plane_certificate_runtime import PlaneCertificateQuery


FLAT = "tools/recovery/generated/g1_recovery_params.yaml"
NOMINAL = "tools/recovery/generated/g1_plane_nominal_params.yaml"


def _query() -> PlaneCertificateQuery:
    return PlaneCertificateQuery(
        command=np.asarray((0.4, 0.0, 0.0)),
        b=np.asarray((0.11, -0.07)),
        q=np.asarray((-0.15, -0.22)),
        support_side="left",
        phase=0.0,
        alpha=np.deg2rad(-10.0),
        adapter_valid=True,
    )


def test_dead_worker_is_restarted_before_submit() -> None:
    pool = CertificateProcessPool(
        FLAT,
        1,
        worker_mode="plane",
        nominal_parameters_path=NOMINAL,
    )
    worker = pool._workers[0]
    old_pid = worker.process.pid
    try:
        worker.process.kill()
        worker.process.wait(timeout=5.0)
        result = pool.submit(_query()).result(timeout=15.0)
        assert worker.process.pid != old_pid
        assert worker.process.poll() is None
        assert result.status in (CertificateStatus.FINITE, CertificateStatus.OVER_HORIZON)
    finally:
        pool.close()


def test_timeout_fails_current_future_and_replaces_worker(monkeypatch) -> None:
    pool = CertificateProcessPool(
        FLAT,
        1,
        worker_mode="plane",
        nominal_parameters_path=NOMINAL,
    )
    worker = pool._workers[0]
    old_pid = worker.process.pid
    old_thread = worker.thread
    real_select = process_pool_module.select.select

    def timeout_once(*_args, **_kwargs):
        monkeypatch.setattr(process_pool_module.select, "select", real_select)
        return ([], [], [])

    monkeypatch.setattr(process_pool_module.select, "select", timeout_once)
    try:
        with pytest.raises(TimeoutError):
            pool.submit(_query()).result(timeout=15.0)
        deadline = time.monotonic() + 5.0
        while worker.process.pid == old_pid and time.monotonic() < deadline:
            time.sleep(0.01)
        assert worker.process.pid != old_pid
        assert worker.process.poll() is None
        assert worker.thread is not old_thread
        assert not old_thread.is_alive()

        result = pool.submit(_query()).result(timeout=15.0)
        assert result.status in (CertificateStatus.FINITE, CertificateStatus.OVER_HORIZON)
    finally:
        pool.close()


def test_dynamic_micro_batches_restore_original_query_order() -> None:
    pool = CertificateProcessPool(
        FLAT,
        4,
        worker_mode="plane",
        nominal_parameters_path=NOMINAL,
        exact_alpha_cache=True,
    )
    queries = tuple(
        (index, _query()) for index in (8, 1, 9, 2, 7, 3, 6, 4, 5, 0)
    )
    try:
        response, chunks = pool.submit_batches(
            queries, chunk_size=2, profile_enabled=True, dynamic_dispatch=True
        ).result(timeout=30.0)
    finally:
        pool.close()
    assert [index for index, _ in response.items] == sorted(index for index, _ in queries)
    assert len(chunks) == 5
    assert all(
        result.status in (CertificateStatus.FINITE, CertificateStatus.OVER_HORIZON)
        for _, result in response.items
    )
