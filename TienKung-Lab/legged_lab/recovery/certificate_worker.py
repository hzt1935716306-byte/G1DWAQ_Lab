"""Clean subprocess entry point for calibrated exact certificate queries."""

from __future__ import annotations

import argparse
import pickle
import struct
import sys
import traceback

from .g1_certificate_runtime import CalibratedG1CertificateEvaluator


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
            response = (True, evaluator._solve(query))
        except BaseException:
            response = (False, traceback.format_exc())
        payload = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stdout.buffer.write(_HEADER.pack(len(payload)))
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
