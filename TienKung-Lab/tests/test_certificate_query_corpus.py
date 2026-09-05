from __future__ import annotations

import numpy as np

from legged_lab.recovery.certificate_query_corpus import (
    PlaneCertificateQueryRecorder,
    RecordedPlaneCertificateQuery,
    load_plane_query_corpus,
)
from legged_lab.recovery.plane_certificate_runtime import PlaneCertificateQuery


def test_query_corpus_preserves_exact_values_and_dtypes(tmp_path) -> None:
    query = PlaneCertificateQuery(
        command=np.asarray((0.4, 0.0, 0.0), dtype=np.float64),
        b=np.asarray((0.123456789012345, -0.2), dtype=np.float64),
        q=np.asarray((-0.1, 0.2), dtype=np.float64),
        support_side="right",
        phase=0.0,
        alpha=float(np.float32(0.1234567)),
        adapter_valid=True,
    )
    path = tmp_path / "queries.bin"
    recorder = PlaneCertificateQueryRecorder(path)
    recorder.append((RecordedPlaneCertificateQuery(9, 42, query),))
    recorder.close()
    actual = load_plane_query_corpus(path)[0]
    assert actual.query_index == 9
    assert actual.env_id == 42
    assert actual.query.alpha == query.alpha
    assert actual.query.command.dtype == query.command.dtype
    assert np.array_equal(actual.query.b, query.b)
    assert np.array_equal(actual.query.q, query.q)
