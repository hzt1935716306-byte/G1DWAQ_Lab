"""Dependency-light configuration, identity, and atomic I/O helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
REPOSITORY = LAB.parent
EVALUATION_SYSTEM = "paper_eval_final"
TERMINAL_OUTCOMES = {"SUCCESS", "FAILURE", "INVALID"}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write(path: str | Path, data: str | bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = data.encode("utf-8") if isinstance(data, str) else data
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                  allow_nan=False) + "\n")


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_bytes(value) + b"\n"
    with target.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def csv_value(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return int(value)
    return value


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip()


def git_dirty_paths() -> list[str]:
    output = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPOSITORY, text=True)
    return [line[3:] for line in output.splitlines() if line]


def code_hash() -> str:
    paths = [ROOT / "run.py", *sorted((ROOT / "src").glob("*.py"))]
    return digest({path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths})


def within(path: str | Path, parent: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def protocol_bundle() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = load_yaml(ROOT / "protocol/protocol_v1_2.yaml")
    freeze = load_yaml(ROOT / "protocol/implementation_freeze.yaml")
    metrics = protocol["metrics"]
    return protocol, freeze, {
        "protocol_version": str(protocol["protocol_version"]),
        "protocol_hash": digest(protocol),
        "metrics_config_hash": digest(metrics),
        "physics_profile_hash": digest(protocol["physics"]),
    }
