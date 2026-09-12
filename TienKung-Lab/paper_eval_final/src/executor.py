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

from .common import (EVALUATION_SYSTEM, ROOT, code_hash, digest, git_head, load_yaml,
                     protocol_bundle, read_json, result_namespace, write_json)
from .environment_adapter import DisturbanceRuntime, close_environment, make_environment
from .model_loader import ModelIdentity, resolve_baseline, resolve_model
from .statistics import summarize
from .storage import RunStore
from .trial_generator import (coverage_probe_manifest, generate_experiment1_continuation, prepare_manifest,
                              smoke_subset, validate_manifest)


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


def _run_path(stage: str, experiment: int, model: ModelIdentity, *, smoke: bool,
              coverage_probe: bool = False, probe_manifest_hash: str | None = None) -> Path:
    category = "coverage_probe" if coverage_probe else "technical_smoke" if smoke else stage
    probe_namespace = (f"probe_{probe_manifest_hash[:12]}" if coverage_probe and probe_manifest_hash
                       else None)
    root = ROOT / "results" / category / result_namespace()
    if probe_namespace:
        root = root / probe_namespace
    return (root / f"experiment_{experiment}"
            / model.model_id / f"train_seed_{model.train_seed}")


def _campaign_schedule(path: Path, manifest: dict[str, Any], initial: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schedule_path = path / "candidate_schedule.json"
    if not schedule_path.exists():
        payload = {"initial_manifest_sha256": manifest["manifest_sha256"], "trials": initial}
        payload["schedule_sha256"] = digest(payload)
        write_json(schedule_path, payload)
        return list(initial)
    payload = read_json(schedule_path)
    claimed = payload.pop("schedule_sha256", None)
    if claimed != digest(payload):
        raise ValueError("candidate schedule integrity mismatch")
    if payload.get("initial_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("candidate schedule belongs to a different initial manifest")
    trials = list(payload.get("trials", []))
    if trials[:len(initial)] != initial or len({row["trial_id"] for row in trials}) != len(trials):
        raise ValueError("candidate schedule initial prefix or trial IDs are invalid")
    return trials


def _write_campaign_schedule(path: Path, manifest: dict[str, Any], trials: list[dict[str, Any]]) -> None:
    payload = {"initial_manifest_sha256": manifest["manifest_sha256"], "trials": trials}
    payload["schedule_sha256"] = digest(payload)
    write_json(path / "candidate_schedule.json", payload)


def execute_one(*, stage: str, experiment: int, model_id: str | None, baseline_id: str | None,
                num_envs: int, device: str, smoke: bool, train_seed: int | None = None,
                coverage_probe: bool = False, probe_candidates: int | None = None) -> dict[str, Any]:
    import torch
    from isaaclab.app import AppLauncher

    protocol, _, _ = protocol_bundle()
    if coverage_probe and (stage != "pilot" or experiment != 1 or smoke):
        raise ValueError("coverage probe is only valid for non-smoke pilot Experiment 1")
    manifest = (coverage_probe_manifest(protocol, count=probe_candidates)
                if coverage_probe else prepare_manifest(stage, experiment))
    validate_manifest(manifest)
    initial_trials = list(manifest["trials"])
    if smoke:
        initial_trials = smoke_subset(initial_trials, experiment)
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
    adaptive_exp1 = experiment == 1 and stage in ("pilot", "formal") and not smoke and not coverage_probe
    if adaptive_exp1:
        experiment1_cfg = load_yaml(ROOT / "configs/experiment1.yaml")
        if model.model_id != experiment1_cfg["model_id"] or model.checkpoint_sha256 != experiment1_cfg["checkpoint_sha256"]:
            raise ValueError("Experiment 1 pilot/formal checkpoint differs from the frozen g1_slope_nosys_d_matched checkpoint")

    application = AppLauncher(headless=True, device=device).app
    _APPLICATIONS.append(application)
    env = runtime = None
    try:
        env, runner, policy, effective = make_environment(
            model, protocol, stage=stage, num_envs=num_envs, device=device, headless=True,
        )
        dataset_role = ("COVERAGE_PROBE_NONANALYSIS" if coverage_probe else
                        "TECHNICAL_SMOKE_NONCONFIRMATORY" if smoke else
                        "ADAPTIVE_OUTCOME_BLIND_CANDIDATE_STREAM" if adaptive_exp1 else "FIXED_BUDGET")
        identity = _identity(model, manifest, experiment, effective, dataset_role=dataset_role)
        path = _run_path(
            stage, experiment, model, smoke=smoke, coverage_probe=coverage_probe,
            probe_manifest_hash=manifest["manifest_sha256"],
        )
        store = RunStore(path, identity)
        if (path / "completion.json").exists():
            return {"status": "ALREADY_COMPLETE", "path": str(path), "completion": read_json(path / "completion.json")}
        trials = _campaign_schedule(path, manifest, initial_trials) if adaptive_exp1 else initial_trials
        write_json(path / "effective_environment.json", effective)
        started = time.monotonic()
        with torch.inference_mode():
            while True:
                completed = store.completed_ids()
                remaining = [trial for trial in trials if trial["trial_id"] not in completed]
                if not remaining:
                    records = store.load_records()
                    if not adaptive_exp1:
                        summary = summarize(records, len(trials))
                        summary.update({"status": "COMPLETE", "dataset_role": dataset_role,
                                        "technical_only_nonbaseline": bool(
                                            smoke and experiment == 1 and model.actor_certificate_context
                                        ), "analysis_excluded": bool(coverage_probe)})
                        completion = store.seal([trial["trial_id"] for trial in trials], summary)
                        return {"status": "COMPLETE", "path": str(path), "summary": summary,
                                "completion": completion}

                    from .experiment1_sampling import (
                        experiment1_config, fit_pilot_boundaries, load_boundaries,
                        select_calibration_samples, select_formal_samples,
                        select_pilot_analysis_samples, write_sampling_assignments,
                    )
                    config = experiment1_config()
                    if stage == "pilot":
                        boundaries = load_boundaries()
                        calibration = select_calibration_samples(
                            records,
                            per_nmin=int(config["calibration_sampling"]["certificate_valid_per_Nmin"]),
                        )
                        if calibration["complete"] and boundaries.get("status") != "FROZEN":
                            boundary_result = fit_pilot_boundaries(records, source_identity=identity)
                            write_json(path / "pilot_margin_status.json", boundary_result)
                            if boundary_result["status"] != "FROZEN":
                                return {"status": boundary_result["status"], "path": str(path),
                                        "margin_boundaries": boundary_result}
                            boundaries = load_boundaries(require_frozen=True)
                        # A frozen boundary may come from a separate independent pilot
                        # calibration campaign (including an audited, outcome-blind
                        # carry-over). load_boundaries() already verifies protocol,
                        # target Nmin, checkpoint, hashes, and pilot provenance.
                        selection = (select_pilot_analysis_samples(
                            records, boundaries,
                            per_cell=int(config["pilot_analysis_sampling"]["certificate_valid_per_cell"]),
                        ) if boundaries.get("status") == "FROZEN" else None)
                    else:
                        boundaries = load_boundaries(require_frozen=True)
                        selection = select_formal_samples(
                            records, boundaries,
                            per_cell=int(config["formal_sampling"]["certificate_valid_per_cell"]),
                        )

                    if selection is not None and selection["complete"]:
                        write_sampling_assignments(path, selection)
                        accepted = set(selection["accepted_trial_ids"])
                        analysis_records = [row for row in records if row["trial_id"] in accepted]
                        summary = summarize(records, len(trials))
                        summary.update({
                            "status": "COMPLETE", "dataset_role": dataset_role,
                            "candidate_count": len(trials),
                            "accepted_count": len(analysis_records),
                            "analysis_summary": summarize(analysis_records, len(analysis_records)),
                            "analysis_cells": selection["cells"],
                            "shared_q1": boundaries["shared_q1"],
                            "shared_q2": boundaries["shared_q2"],
                            "calibration_manifest_hash": boundaries["calibration_manifest_hash"],
                            "evaluation_manifest_hash": selection["evaluation_manifest_hash"],
                        })
                        completion = store.seal([trial["trial_id"] for trial in trials], summary)
                        return {"status": "COMPLETE", "path": str(path), "summary": summary,
                                "completion": completion}

                    cfg = config[f"{stage}_sampling"]
                    if stage == "formal":
                        freeze = load_yaml(ROOT / "protocol/implementation_freeze.yaml")
                        cap = freeze["formal_stratified_caps"]["max_candidates_total"]
                        layer_cap = freeze["formal_stratified_caps"]["max_candidates_per_condition_layer"]
                    else:
                        cap = cfg["max_candidates_total"]
                        layer_cap = None
                    if cap is None:
                        raise ValueError(f"{stage} Experiment 1 candidate cap is not frozen")
                    if len(trials) >= int(cap):
                        status = {"status": "TARGET_NOT_REACHED", "candidate_count": len(trials),
                                  "max_candidates_total": int(cap),
                                  "calibration_deficits": calibration["deficits"] if stage == "pilot" else None,
                                  "analysis_cells": selection["cells"] if selection is not None else None}
                        write_json(path / "sampling_status.json", status)
                        return {**status, "path": str(path)}
                    batch_count = min(int(cfg["continuation_batch_candidates"]), int(cap) - len(trials))
                    continuation = generate_experiment1_continuation(
                        stage, max(row["sequence"] for row in trials) + 1,
                        batch_count, records, protocol,
                        max_candidates_per_condition_layer=(int(layer_cap) if layer_cap is not None else None),
                    )
                    if not continuation:
                        raise ValueError("adaptive continuation produced no trials before quotas were complete")
                    trials.extend(continuation)
                    _write_campaign_schedule(path, manifest, trials)
                    remaining = continuation

                selected = remaining[:num_envs]
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
                terminal_payloads: list[tuple[Any, dict[str, Any]]] = []
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
                            terminal_payloads.append((machine, record))
                            saved.add(machine.plan["trial_id"])
                    if timestamp > max_time + 0.1:
                        raise ValueError("trial state machine exceeded its bounded physical-time horizon")
                if adaptive_exp1:
                    from .experiment1_sampling import annotate_sampling_batch, load_boundaries
                    batch_records = [record for _, record in terminal_payloads]
                    prior_records = store.load_records()
                    boundaries = load_boundaries(require_frozen=True) if stage == "formal" else load_boundaries()
                    if boundaries.get("status") != "FROZEN":
                        boundaries = None
                    annotate_sampling_batch(batch_records, prior_records, stage=stage, boundaries=boundaries)
                for machine, record in terminal_payloads:
                    store.save_trial(record, machine.frames, machine.events, machine.pre_reset_snapshot)
                runtime.close()
                runtime = None
                write_json(path / "progress.json", {
                    "completed": len(store.completed_ids()), "scheduled": len(trials),
                    "elapsed_wall_s": time.monotonic() - started,
                    "last_batch_size": len(selected),
                })
    except BaseException as exc:
        if env is not None:
            failure_path = _run_path(stage, experiment, model, smoke=smoke,
                                     coverage_probe=coverage_probe,
                                     probe_manifest_hash=manifest.get("manifest_sha256")) if "model" in locals() else ROOT / "results/failure.json"
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
        env, _, policy, effective = make_environment(
            model, protocol, stage="screening", num_envs=num_envs, device=device, headless=True,
        )
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
