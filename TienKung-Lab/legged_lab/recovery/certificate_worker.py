"""Clean subprocess entry point for calibrated exact certificate queries."""

from __future__ import annotations

import argparse
import pickle
import struct
import sys
import traceback
import time

from .g1_certificate_runtime import CalibratedG1CertificateEvaluator
from .certificate_ipc import (
    CertificateBatchRequest,
    CertificateBatchResponse,
    CertificateWorkerProfile,
)


_HEADER = struct.Struct("!Q")


def _read_exact(size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sys.stdin.buffer.read(size - len(chunks))
        if not chunk:
            raise EOFError("parent closed the certificate request pipe")
        chunks.extend(chunk)
    return bytes(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--mode", choices=("flat", "plane"), default="flat")
    parser.add_argument("--nominal-parameters")
    parser.add_argument("--z-sole", type=float, default=-0.045)
    parser.add_argument("--exact-alpha-cache", action="store_true")
    parser.add_argument("--exact-alpha-cache-max-entries", type=int, default=8192)
    args = parser.parse_args()
    if args.mode == "plane":
        if not args.nominal_parameters:
            parser.error("--nominal-parameters is required in plane mode")
        from .plane_certificate_runtime import PlaneCalibratedG1CertificateEvaluator

        evaluator = PlaneCalibratedG1CertificateEvaluator(
            args.parameters,
            args.nominal_parameters,
            workers=1,
            executor_type="sequential",
            z_sole=args.z_sole,
            exact_alpha_cache=args.exact_alpha_cache,
            exact_alpha_cache_max_entries=args.exact_alpha_cache_max_entries,
        )
    else:
        evaluator = CalibratedG1CertificateEvaluator(
            args.parameters,
            workers=1,
            executor_type="sequential",
        )
    while True:
        size = _HEADER.unpack(_read_exact(_HEADER.size))[0]
        if size == 0:
            break
        query = pickle.loads(_read_exact(size))
        try:
            if isinstance(query, CertificateBatchRequest):
                received_ns = time.perf_counter_ns()
                cache_before = getattr(
                    evaluator,
                    "capability_cache_statistics",
                    {"size": 0, "hits": 0, "misses": 0},
                )
                solve_started = time.perf_counter_ns()
                items = tuple(
                    (item.query_index, evaluator._solve(item.query)) for item in query.items
                )
                solve_ended = time.perf_counter_ns()
                result_serialize_ms = 0.0
                if query.profile_enabled:
                    serialization_started = time.perf_counter_ns()
                    pickle.dumps(items, protocol=pickle.HIGHEST_PROTOCOL)
                    result_serialize_ms = (time.perf_counter_ns() - serialization_started) / 1.0e6
                cache = getattr(
                    evaluator,
                    "capability_cache_statistics",
                    {"size": 0, "hits": 0, "misses": 0},
                )
                payload = CertificateBatchResponse(
                    items,
                    CertificateWorkerProfile(
                        worker_wait_ms=(received_ns - query.enqueued_ns) / 1.0e6,
                        worker_solve_ms=(solve_ended - solve_started) / 1.0e6,
                        result_serialize_ms=result_serialize_ms,
                        query_count=len(items),
                        capability_cache_size=int(cache["size"]),
                        capability_cache_hits=int(cache["hits"] - cache_before["hits"]),
                        capability_cache_misses=int(cache["misses"] - cache_before["misses"]),
                    ) if query.profile_enabled else None,
                )
                response = (True, payload)
            else:
                response = (True, evaluator._solve(query))
        except BaseException:
            response = (False, traceback.format_exc())
        payload = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stdout.buffer.write(_HEADER.pack(len(payload)))
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
