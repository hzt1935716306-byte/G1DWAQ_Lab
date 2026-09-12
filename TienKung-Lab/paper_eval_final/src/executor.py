"""One-process/one-experiment Isaac execution and isolated throughput probes."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from .common import EVALUATION_SYSTEM, ROOT, code_hash, digest, git_head, protocol_bundle, read_json, write_json
from .environment_adapter import DisturbanceRuntime, close_environment, make_environment
from .model_loader import ModelIdentity, resolve_baseline, resolve_model
from .statistics import summarize
from .storage import RunStore
from .trial_generator import manifest_path, prepare_manifest, smoke_subset, validate_manifest


# Keep SimulationApp alive until the worker's explicit process boundary.  In
# Isaac Sim 5.1, SimulationApp.close() can deadlock after this native Legged Lab
# VecEnv has already completed.  run.py flushes results and uses os._exit for
# internal workers, a pattern already used by the repository's batch evaluators.
_APPLICATIONS: list[Any] = []


def _identity(model: ModelIdentity, manifest: dict[str, Any], experiment: int, effective: dict[str, Any],
              *, dataset_role: str) -> dict[str, Any]:
    _, _, protocol_identity = protocol_bundle()
    return {
        "evaluation_system": EVALUATION_SYSTEM,
        "protocol_version": protocol_identity["protocol_version"],
        "protocol_hash": protocol_identity["protocol_hash"],
        "metrics_config_hash": protocol_identity["metrics_config_hash"],
        "physics_profile_hash": protocol_identity["physics_profile_hash"],
        "manifest_hash": manifest["manifest_sha256"],
        "stage": manifest["stage"],
        "experiment_id": int(experiment),
        "dataset_role": dataset_role,
        "model_id": model.model_id,
        "method": model.method,
        "task_name": model.task_name,
        "train_seed": model.train_seed,
        "checkpoint_sha256": model.checkpoint_sha256,
        "training_commit": model.training_commit,
        "evaluation_code_commit": git_head(),
        "evaluation_code_hash": code_hash(),
        "asset_hash": effective["asset_hash"],
        "simulator_version": effective["simulator_version"],
        "termination_body_names": effective["termination_body_names"],
        "actual_physics_hash": effective["actual_physics_hash"],
    }


def _run_path(stage: str, experiment: int, model: ModelIdentity, *, smoke: bool) -> Path:
    category = "technical_smoke" if smoke else stage
    return ROOT / "results" / category / f"experiment_{experiment}" / model.model_id / f"train_seed_{model.train_seed}"


def execute_one(*, stage: str, experiment: int, model_id: str | None, baseline_id: str | None,
                num_envs: int, device: str, smoke: bool, train_seed: int | None = None) -> dict[str, Any]:
    import torch
    from isaaclab.app import AppLauncher

    protocol, _, _ = protocol_bundle()
    manifest = prepare_manifest(stage, experiment)
    validate_manifest(manifest)
    trials = list(manifest["trials"])
    if smoke:
        trials = smoke_subset(trials, experiment)
    if experiment == 1 and not smoke:
        if not baseline_id:
            raise ValueError("Experiment 1 requires --baseline; certificate-context policies are forbidden")
        model = resolve_baseline(baseline_id, train_seed)
    elif experiment == 1 and baseline_id:
        model = resolve_baseline(baseline_id, train_seed)
    else:
        if not model_id:
            raise ValueError("Experiments 2/3 require --model")
        model = resolve_model(model_id, train_seed)
    if experiment == 1 and model.actor_certificate_context and not smoke:
        raise ValueError("Experiment 1 confirmatory baseline cannot receive Nmin/margin")

    application = AppLauncher(headless=True, device=device).app
    _APPLICATIONS.append(application)
    env = runtime = None
    try:
        env, runner, policy, effective = make_environment(
            model, protocol, num_envs=num_envs, device=device, headless=True,
        )
        dataset_role = "TECHNICAL_SMOKE_NONCONFIRMATORY" if smoke else "FIXED_BUDGET"
        identity = _identity(model, manifest, experiment, effective, dataset_role=dataset_role)
        path = _run_path(stage, experiment, model, smoke=smoke)
        store = RunStore(path, identity)
        if (path / "completion.json").exists():
            return {"status": "ALREADY_COMPLETE", "path": str(path), "completion": read_json(path / "completion.json")}
        completed = store.completed_ids()
        remaining = [trial for trial in trials if trial["trial_id"] not in completed]
        write_json(path / "effective_environment.json", effective)
        started = time.monotonic()
        with torch.inference_mode():
            for offset in range(0, len(remaining), num_envs):
                selected = remaining[offset:offset + num_envs]
                padded = selected + [copy.deepcopy(selected[-1]) for _ in range(num_envs - len(selected))]
                observation, extras = env.begin_batch(padded)
                machines: list[Any | None] = [__import__(
                    "paper_eval_final.src.trial_machine", fromlist=["TrialMachine"]
                ).TrialMachine(plan, protocol) for plan in selected] + [None] * (num_envs - len(selected))
                runtime = DisturbanceRuntime(
                    env, model, protocol, machines, certificate_enabled=experiment == 1,
                )
                initial = env.snapshot_rows()
                for index, machine in enumerate(machines[:len(selected)]):
                    machine.feed_physics(0.0, initial[index]["foot_force_norms_N"])
                    initial[index]["timestamp"] = 0.0
                    machine.feed_policy(initial[index])
                max_time = 12.0 if experiment == 2 else max(row["observation_end_s"] for row in selected) + 0.05
                step = 0
                saved: set[str] = set()
                while any(machine is not None and not machine.status for machine in machines):
                    step += 1
                    actions = policy(observation, extras)
                    finite = torch.isfinite(actions).all(dim=1)
                    if not bool(finite.all()):
                        for index, machine in enumerate(machines):
                            if machine is not None and not machine.status and not bool(finite[index]):
                                machine.controller_numerical_failure(step * env.step_dt)
                                env.eval_controller_failure_mask[index] = True
                        actions = torch.where(finite[:, None], actions, torch.zeros_like(actions))
                    env.eval_last_actions = actions.detach().clone()
                    observation, _, dones, extras = env.step(actions)
                    timestamp = step * float(env.step_dt)
                    frames = env.eval_capture if env.eval_capture is not None else env.snapshot_rows()
                    for index, machine in enumerate(machines):
                        if machine is None or machine.plan["trial_id"] in saved:
                            continue
                        frame = dict(frames[index])
                        frame["timestamp"] = timestamp
                        if not machine.status:
                            if bool(dones[index]):
                                reason = "NATIVE_PHYSICAL_RESET" if not bool(frame.get("timeout")) else "UNEXPECTED_NATIVE_TIMEOUT"
                                machine.feed_physics(timestamp, frame["foot_force_norms_N"], reason)
                            machine.feed_policy(frame)
                        if machine.status:
                            if machine.termination_kind == "PHYSICAL_RESET":
                                machine.attach_pre_reset_snapshot({**frame, "timestamp": timestamp})
                            record = machine.result()
                            record.update({
                                "reset_time": timestamp if machine.termination_kind == "PHYSICAL_RESET" else None,
                                "out_of_area": bool(frame.get("out_of_area", False)),
                            })
                            store.save_trial(record, machine.frames, machine.events, machine.pre_reset_snapshot)
                            saved.add(machine.plan["trial_id"])
                    if timestamp > max_time + 0.1:
                        raise ValueError("trial state machine exceeded its bounded physical-time horizon")
                runtime.close()
                runtime = None
                write_json(path / "progress.json", {
                    "completed": len(store.completed_ids()), "scheduled": len(trials),
                    "elapsed_wall_s": time.monotonic() - started, "last_batch_offset": offset,
                })
        records = store.load_records()
        summary = summarize(records, len(trials))
        summary.update({"status": "COMPLETE", "dataset_role": dataset_role,
                        "technical_only_nonbaseline": bool(smoke and experiment == 1 and model.actor_certificate_context)})
        completion = store.seal([trial["trial_id"] for trial in trials], summary)
        return {"status": "COMPLETE", "path": str(path), "summary": summary, "completion": completion}
    except BaseException as exc:
        if env is not None:
            failure_path = _run_path(stage, experiment, model, smoke=smoke) if "model" in locals() else ROOT / "results/failure.json"
            failure_path.mkdir(parents=True, exist_ok=True)
            write_json(failure_path / "failure.json", {"error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if runtime is not None:
            runtime.close()
        close_environment(env)


def benchmark_once(*, model_id: str, num_envs: int, steps: int, warmup_steps: int, device: str) -> dict[str, Any]:
    import torch
    from isaaclab.app import AppLauncher

    protocol, _, _ = protocol_bundle()
    manifest = prepare_manifest("screening", 2)
    model = resolve_model(model_id)
    application = AppLauncher(headless=True, device=device).app
    _APPLICATIONS.append(application)
    env = None
    try:
        env, _, policy, effective = make_environment(model, protocol, num_envs=num_envs, device=device, headless=True)
        source = manifest["trials"]
        plans = [copy.deepcopy(source[index % len(source)]) for index in range(num_envs)]
        observation, extras = env.begin_batch(plans)
        with torch.inference_mode():
            for _ in range(warmup_steps):
                actions = policy(observation, extras)
                env.eval_last_actions = actions
                observation, _, _, extras = env.step(actions)
            torch.cuda.synchronize() if device.startswith("cuda") else None
            torch.cuda.reset_peak_memory_stats(torch.device(device)) if device.startswith("cuda") else None
            started = time.perf_counter()
            for _ in range(steps):
                actions = policy(observation, extras)
                env.eval_last_actions = actions
                observation, _, _, extras = env.step(actions)
            torch.cuda.synchronize() if device.startswith("cuda") else None
            elapsed = time.perf_counter() - started
        return {
            "model_id": model_id, "checkpoint_sha256": model.checkpoint_sha256,
            "num_envs": num_envs, "warmup_policy_steps": warmup_steps, "measured_policy_steps": steps,
            "wall_time_s": elapsed, "policy_steps_per_second": steps / elapsed,
            "environment_steps_per_second": num_envs * steps / elapsed,
            "valid_simulated_seconds_per_wall_second": num_envs * steps * protocol["physics"]["dt_policy_s"] / elapsed,
            "peak_cuda_memory_MiB": (torch.cuda.max_memory_allocated(torch.device(device)) / 2**20
                                      if device.startswith("cuda") else None),
            "actual_physics_hash": effective["actual_physics_hash"],
            "simulator_version": effective["simulator_version"],
        }
    except BaseException as exc:
        # Emit the real simulator/adapter error before Kit shutdown.  Some
        # native shutdown paths can otherwise hide the actionable traceback.
        print(f"BENCHMARK_FAILURE={exc!r}", flush=True)
        traceback.print_exc()
        raise
    finally:
        close_environment(env)
