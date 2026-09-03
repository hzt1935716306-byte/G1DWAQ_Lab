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
    def __init__(
        self,
        parameters_path: Path,
        index: int,
        worker_mode: str = "flat",
        nominal_parameters_path: Path | None = None,
        z_sole: float = -0.045,
    ) -> None:
        # Do not pass Isaac/Kit's dynamic-library environment into the clean
        # solver interpreter.  In particular, inherited LD_LIBRARY_PATH values
        # can make SciPy/HiGHS load incompatible simulator-side runtimes.
        self._environment = {
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
        self._environment.update(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONUNBUFFERED="1",
        )
        self._parameters_path = parameters_path
        self._index = int(index)
        self._worker_mode = worker_mode
        self._nominal_parameters_path = nominal_parameters_path
        self._z_sole = float(z_sole)
        self._process_lock = threading.Lock()
        self._closing = False
        self._thread_generation = 0
        self.process = self._start_process()
        self.queue: Queue[Any] = Queue()
        self.thread = self._new_io_thread()
        self.thread.start()

    def _new_io_thread(self) -> threading.Thread:
        self._thread_generation += 1
        return threading.Thread(
            target=self._serve,
            name=(
                f"certificate-process-io-{self._index}-"
                f"{self._thread_generation}"
            ),
            daemon=True,
        )

    def _start_process(self) -> subprocess.Popen:
        command = [
            sys.executable,
            "-m",
            "legged_lab.recovery.certificate_worker",
            "--parameters",
            str(self._parameters_path),
            "--mode",
            self._worker_mode,
        ]
        if self._nominal_parameters_path is not None:
            command.extend(("--nominal-parameters", str(self._nominal_parameters_path)))
        if self._worker_mode == "plane":
            command.extend(("--z-sole", str(self._z_sole)))
        return subprocess.Popen(
            tuple(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._environment,
            bufsize=0,
        )

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def _restart_process(self, failed_process: subprocess.Popen | None = None) -> None:
        with self._process_lock:
            if self._closing:
                return
            if failed_process is not None and self.process is not failed_process:
                return
            self._stop_process(self.process)
            self.process = self._start_process()

    def _ensure_healthy(self) -> None:
        with self._process_lock:
            if self._closing:
                raise RuntimeError("certificate worker is closing")
            if self.process.poll() is not None:
                self._stop_process(self.process)
                self.process = self._start_process()

    def _replace_after_transport_failure(
        self,
        failed_process: subprocess.Popen,
    ) -> bool:
        """Replace the failed process and its I/O thread for this queue slot."""

        with self._process_lock:
            if self._closing:
                return False
            if self.process is failed_process:
                self._stop_process(self.process)
                self.process = self._start_process()
            replacement = self._new_io_thread()
            self.thread = replacement
            replacement.start()
            return True

    def submit(self, query) -> Future:
        future: Future = Future()
        try:
            self._ensure_healthy()
        except Exception as exc:
            future.set_exception(exc)
            return future
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
            process = self.process
            try:
                if process.poll() is not None:
                    self._restart_process(process)
                    process = self.process
                assert process.stdin is not None
                assert process.stdout is not None
                request = pickle.dumps(query, protocol=pickle.HIGHEST_PROTOCOL)
                process.stdin.write(_HEADER.pack(len(request)))
                process.stdin.write(request)
                process.stdin.flush()
                readable, _, _ = select.select(
                    (process.stdout,), (), (), _RESPONSE_TIMEOUT_S
                )
                if not readable:
                    raise TimeoutError(
                        "clean certificate worker did not respond within "
                        f"{_RESPONSE_TIMEOUT_S:.0f}s"
                    )
                response_size = _HEADER.unpack(_read_exact(process.stdout, _HEADER.size))[0]
                success, payload = pickle.loads(_read_exact(process.stdout, response_size))
                if success:
                    future.set_result(payload)
                else:
                    future.set_exception(RuntimeError(payload))
            except BaseException as exc:
                future.set_exception(exc)
                transport_failure = isinstance(
                    exc,
                    (BrokenPipeError, EOFError, OSError, TimeoutError, pickle.PickleError),
                ) or process.poll() is not None
                if transport_failure:
                    if self._replace_after_transport_failure(process):
                        return

    def request_close(self) -> None:
        self._closing = True
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
            except (BrokenPipeError, OSError, ValueError):
                pass

    def stderr(self) -> str:
        if self.process.stderr is None or self.process.poll() is None:
            return ""
        return self.process.stderr.read().decode("utf-8", errors="replace")


class CertificateProcessPool:
    """Small round-robin pool exposing the subset of Executor used at runtime."""

    def __init__(
        self,
        parameters_path: str | Path,
        workers: int,
        *,
        worker_mode: str = "flat",
        nominal_parameters_path: str | Path | None = None,
        z_sole: float = -0.045,
    ) -> None:
        if workers <= 0:
            raise ValueError("certificate process workers must be positive")
        path = Path(parameters_path).expanduser().resolve()
        nominal_path = (
            Path(nominal_parameters_path).expanduser().resolve()
            if nominal_parameters_path is not None
            else None
        )
        if worker_mode not in ("flat", "plane"):
            raise ValueError("certificate worker mode must be 'flat' or 'plane'")
        if worker_mode == "plane" and nominal_path is None:
            raise ValueError("plane certificate workers require nominal parameters")
        self._workers = tuple(
            _Worker(path, index, worker_mode, nominal_path, z_sole)
            for index in range(workers)
        )
        self._next_worker = 0
        self._closed = False

    def submit(self, query) -> Future:
        if self._closed:
            raise RuntimeError("certificate process pool is closed")
        worker = self._workers[self._next_worker]
        self._next_worker = (self._next_worker + 1) % len(self._workers)
        worker._ensure_healthy()
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
        for worker in self._workers:
            worker._stop_process(worker.process)
