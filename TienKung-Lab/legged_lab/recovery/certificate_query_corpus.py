"""Lossless framed recorder/reader for production Plane certificate queries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import struct
import threading
from typing import Any, Iterable, Iterator


_HEADER = struct.Struct("!Q")


@dataclass(frozen=True)
class RecordedPlaneCertificateQuery:
    query_index: int
    env_id: int
    query: Any


class PlaneCertificateQueryRecorder:
    """Append exact query objects without rounding, quantizing, or dtype conversion."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("ab")
        self._lock = threading.Lock()
        self.count = 0
        self.valid_count = 0

    def append(self, records: Iterable[RecordedPlaneCertificateQuery]) -> None:
        batch = tuple(records)
        if not batch:
            return
        payload = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
        with self._lock:
            self._stream.write(_HEADER.pack(len(payload)))
            self._stream.write(payload)
            self._stream.flush()
        self.count += len(batch)
        self.valid_count += sum(record.query.adapter_valid for record in batch)

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


def iter_plane_query_corpus(path: str | Path) -> Iterator[RecordedPlaneCertificateQuery]:
    with Path(path).expanduser().resolve().open("rb") as stream:
        while True:
            header = stream.read(_HEADER.size)
            if not header:
                return
            if len(header) != _HEADER.size:
                raise EOFError("truncated Plane certificate corpus header")
            size = _HEADER.unpack(header)[0]
            payload = stream.read(size)
            if len(payload) != size:
                raise EOFError("truncated Plane certificate corpus payload")
            yield from pickle.loads(payload)


def load_plane_query_corpus(
    path: str | Path,
    *,
    valid_only: bool = False,
    limit: int | None = None,
) -> list[RecordedPlaneCertificateQuery]:
    result = []
    for record in iter_plane_query_corpus(path):
        if valid_only and not record.query.adapter_valid:
            continue
        result.append(record)
        if limit is not None and len(result) >= limit:
            break
    return result


__all__ = [
    "PlaneCertificateQueryRecorder",
    "RecordedPlaneCertificateQuery",
    "iter_plane_query_corpus",
    "load_plane_query_corpus",
]
