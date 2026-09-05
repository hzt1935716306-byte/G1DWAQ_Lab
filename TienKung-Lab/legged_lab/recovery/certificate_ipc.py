"""Wire records for optional batched certificate subprocess execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IndexedCertificateQuery:
    query_index: int
    query: Any


@dataclass(frozen=True)
class CertificateBatchRequest:
    items: tuple[IndexedCertificateQuery, ...]
    enqueued_ns: int
    profile_enabled: bool = False


@dataclass(frozen=True)
class CertificateWorkerProfile:
    worker_wait_ms: float
    worker_solve_ms: float
    result_serialize_ms: float
    query_count: int
    capability_cache_size: int = 0
    capability_cache_hits: int = 0
    capability_cache_misses: int = 0


@dataclass(frozen=True)
class CertificateTransportProfile:
    serialize_ms: float
    ipc_dispatch_ms: float
    ipc_receive_ms: float
    result_rebuild_ms: float
    request_bytes: int
    response_bytes: int


@dataclass(frozen=True)
class CertificateBatchResponse:
    items: tuple[tuple[int, Any], ...]
    worker_profile: CertificateWorkerProfile | None = None
    transport_profile: CertificateTransportProfile | None = None
