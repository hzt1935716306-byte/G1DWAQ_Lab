"""Persistent clean-process executor for exact certificate solves.

Isaac/CUDA and SciPy/HiGHS can conflict when HiGHS is called in the simulator
process, even sequentially.  These workers are launched as independent Python
interpreters and never import Isaac Lab.  Parent-side threads only perform
blocking pipe I/O; no LP is solved in a thread.
"""

from __future__ import annotations

from concurrent.futures import Future
import os
import pickle
from pathlib import Path
from queue import Queue
import select
import struct
import subprocess
import sys
import threading
import time
from typing import Any


_HEADER = struct.Struct("!Q")
_STOP = object()
_RESPONSE_TIMEOUT_S = 30.0


def _read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("certificate worker closed its output pipe")
        chunks.extend(chunk)
    return bytes(chunks)


class _Worker:
    def __init__(self, parameters_path: Path, index: int) -> None:
        # Do not pass Isaac/Kit's dynamic-library environment into the clean
        # solver interpreter.  In particular, inherited LD_LIBRARY_PATH values
        # can make SciPy/HiGHS load incompatible simulator-side runtimes.
        environment = {
            name: os.environ[name]
            for name in (
                "HOME",
                "LANG",
                "LC_ALL",
                "PATH",
                "PYTHONPATH",
                "USER",
                "CONDA_PREFIX",
                "VIRTUAL_ENV",
            )
            if name in os.environ
        }
        environment.update(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONUNBUFFERED="1",
        )
        self.process = subprocess.Popen(
            (
                sys.executable,
                "-m",
                "legged_lab.recovery.certificate_worker",
                "--parameters",
                str(parameters_path),
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            bufsize=0,
        )
        self.queue: Queue[Any] = Queue()
        self.thread = threading.Thread(
            target=self._serve,
            name=f"certificate-process-io-{index}",
            daemon=True,
        )
        self.thread.start()

    def submit(self, query) -> Future:
        future: Future = Future()
        self.queue.put((future, query))
        return future

    def _serve(self) -> None:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        while True:
            item = self.queue.get()
            if item is _STOP:
                break
            future, query = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                request = pickle.dumps(query, protocol=pickle.HIGHEST_PROTOCOL)
                self.process.stdin.write(_HEADER.pack(len(request)))
                self.process.stdin.write(request)
                self.process.stdin.flush()
                readable, _, _ = select.select(
                    (self.process.stdout,), (), (), _RESPONSE_TIMEOUT_S
                )
                if not readable:
                    self.process.terminate()
                    self.process.wait(timeout=5.0)
                    diagnostic = self.stderr().strip()
                    raise TimeoutError(
                        "clean certificate worker did not respond within "
                        f"{_RESPONSE_TIMEOUT_S:.0f}s; returncode={self.process.returncode}; "
                        f"stderr={diagnostic!r}"
                    )
                response_size = _HEADER.unpack(_read_exact(self.process.stdout, _HEADER.size))[0]
                success, payload = pickle.loads(_read_exact(self.process.stdout, response_size))
                if success:
                    future.set_result(payload)
                else:
                    future.set_exception(RuntimeError(payload))
            except BaseException as exc:
                future.set_exception(exc)

    def request_close(self) -> None:
        self.queue.put(_STOP)
        self.thread.join(timeout=0.5)
        if self.thread.is_alive() and self.process.poll() is None:
            # A worker may have completed its Future while its I/O thread is
            # still unwinding the pipe operation.  Never let diagnostic or
            # training shutdown block indefinitely on that thread.
            self.process.terminate()
            self.thread.join(timeout=1.0)
        if self.process.poll() is None:
            assert self.process.stdin is not None
            try:
                self.process.stdin.write(_HEADER.pack(0))
                self.process.stdin.flush()
            except BrokenPipeError:
                pass

    def stderr(self) -> str:
        if self.process.stderr is None or self.process.poll() is None:
            return ""
        return self.process.stderr.read().decode("utf-8", errors="replace")


class CertificateProcessPool:
    """Small round-robin pool exposing the subset of Executor used at runtime."""

    def __init__(self, parameters_path: str | Path, workers: int) -> None:
        if workers <= 0:
            raise ValueError("certificate process workers must be positive")
        path = Path(parameters_path).expanduser().resolve()
        self._workers = tuple(_Worker(path, index) for index in range(workers))
        self._next_worker = 0
        self._closed = False

    def submit(self, query) -> Future:
        if self._closed:
            raise RuntimeError("certificate process pool is closed")
        worker = self._workers[self._next_worker]
        self._next_worker = (self._next_worker + 1) % len(self._workers)
        return worker.submit(query)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Notify every worker first, then wait against one shared deadline.
        # Sequential per-process waits made eight idle workers look like a
        # 40-second shutdown hang even though the report was already complete.
        for worker in self._workers:
            worker.request_close()
        deadline = time.monotonic() + 2.0
        for worker in self._workers:
            if worker.process.poll() is not None:
                continue
            try:
                worker.process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        remaining = [worker for worker in self._workers if worker.process.poll() is None]
        for worker in remaining:
            worker.process.terminate()
        for worker in remaining:
            try:
                worker.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait(timeout=1.0)
