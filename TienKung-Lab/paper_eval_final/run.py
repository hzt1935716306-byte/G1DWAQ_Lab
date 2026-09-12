#!/usr/bin/env python3
"""CLI for the protocol-v1.2 paper evaluation and adaptive Experiment 1 sampling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval_final.src.aggregate import aggregate
from paper_eval_final.src.common import ROOT, load_yaml, read_json, write_json
from paper_eval_final.src.reporting import write_experiment_report, write_screening_report
from paper_eval_final.src.trial_generator import prepare_all


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage", choices=("screening", "pilot", "formal"))
    value.add_argument("--experiment", type=int, choices=(1, 2, 3))
    value.add_argument("--experiments", type=int, nargs="+", choices=(1, 2, 3))
    value.add_argument("--baseline")
    value.add_argument("--model", choices=("M1", "M2", "M3", "M4", "M5"))
    value.add_argument("--train-seed", type=int, help="run one registered training seed (default: all registered seeds)")
    value.add_argument("--aggregate", action="store_true")
    value.add_argument("--prepare", action="store_true")
    value.add_argument("--smoke", action="store_true")
    value.add_argument("--full-audit", action="store_true")
    value.add_argument("--num-envs", type=int)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--benchmark", action="store_true")
    value.add_argument("--env-counts", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    value.add_argument("--benchmark-steps", type=int, default=100)
    value.add_argument("--benchmark-warmup-steps", type=int, default=10)
    value.add_argument("--_execute-one", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--_benchmark-one", type=int, help=argparse.SUPPRESS)
    return value


def experiments(args: argparse.Namespace) -> list[int]:
    result = ([] if args.experiment is None else [args.experiment]) + (args.experiments or [])
    result = list(dict.fromkeys(result))
    if not result and not args.prepare and not args.benchmark and not args.full_audit:
        raise ValueError("select --experiment or --experiments")
    return result


def selected_num_envs(args: argparse.Namespace) -> int:
    if args.num_envs:
        if args.num_envs < 1:
            raise ValueError("--num-envs must be positive")
        return args.num_envs
    freeze = load_yaml(ROOT / "protocol/implementation_freeze.yaml")
    selected = freeze["parallel_environments"]["selected"]
    if selected is None:
        raise ValueError("parallel environment count is not frozen; run --benchmark first or pass --num-envs")
    return int(selected)


def formal_guard(args: argparse.Namespace, chosen: list[int]) -> None:
    if args.stage != "formal":
        return
    freeze = load_yaml(ROOT / "protocol/implementation_freeze.yaml")
    if not freeze["formal_enabled"] or not freeze["config_frozen"]:
        raise ValueError("formal run refused: pilot/config freeze has not been approved")
    if 1 in chosen and args.baseline != freeze["formal_experiment1_baseline"]:
        raise ValueError("formal Experiment 1 baseline is not frozen or does not match")
    if 1 in chosen:
        from paper_eval_final.src.experiment1_sampling import load_boundaries

        boundaries = load_boundaries(require_frozen=True)
        frozen_boundary = freeze.get("experiment1_margin_boundaries", {})
        if (frozen_boundary.get("status") != "FROZEN"
                or frozen_boundary.get("boundary_id") != boundaries["boundary_id"]):
            raise ValueError("formal Experiment 1 refused: pilot margin boundary freeze does not match")
        caps = freeze["formal_stratified_caps"]
        if caps["max_candidates_total"] is None or caps["max_candidates_per_condition_layer"] is None:
            raise ValueError("formal Experiment 1 refused: finite stratified candidate caps are not frozen")


def experiment1_baseline_guard(args: argparse.Namespace, chosen: list[int]) -> None:
    if 1 not in chosen or args.smoke or args.stage not in ("pilot", "formal"):
        return
    config = load_yaml(ROOT / "configs/experiment1.yaml")
    required = str(config["baseline_id"])
    if args.baseline != required:
        raise ValueError(
            f"Experiment 1 {args.stage} is bound to --baseline {required}; "
            "screening-only checkpoints cannot enter the pilot/formal boundary dataset"
        )


def registered_seeds(args: argparse.Namespace, experiment: int) -> list[int]:
    from paper_eval_final.src.model_loader import registry

    data = registry()
    if experiment == 1:
        if args.smoke and args.model:
            model_id = args.model
        elif not args.baseline:
            raise ValueError("Experiment 1 requires --baseline")
        else:
            baseline = data["screening_baselines"].get(args.baseline)
            if not baseline or not baseline.get("model"):
                raise ValueError(f"baseline {args.baseline!r} has no registered checkpoint")
            model_id = baseline["model"]
    else:
        if not args.model:
            raise ValueError("Experiments 2/3 require --model")
        model_id = args.model
    checkpoints = data["models"][model_id].get("checkpoints", [])
    all_seeds = sorted(int(item["train_seed"]) for item in checkpoints)
    if args.stage == "formal" and len(all_seeds) != 3:
        raise ValueError(f"{model_id}: TRAIN_SEED_INCOMPLETE (formal requires exactly 3 independent training seeds)")
    seeds = list(all_seeds)
    if args.train_seed is not None:
        seeds = [seed for seed in seeds if seed == args.train_seed]
    if not seeds:
        raise ValueError(f"{model_id}: requested training seed/checkpoint is not registered")
    return seeds


def worker_command(args: argparse.Namespace, experiment: int, train_seed: int) -> list[str]:
    command = [sys.executable, str(ROOT / "run.py"), "--_execute-one", "--stage", args.stage,
               "--experiment", str(experiment), "--train-seed", str(train_seed),
               "--num-envs", str(selected_num_envs(args)), "--device", args.device]
    if args.model:
        command += ["--model", args.model]
    if args.baseline:
        command += ["--baseline", args.baseline]
    if args.smoke:
        command.append("--smoke")
    return command


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model:
        raise ValueError("--benchmark requires --model")
    rows = []
    for count in args.env_counts:
        command = [sys.executable, str(ROOT / "run.py"), "--_benchmark-one", str(count),
                   "--model", args.model, "--device", args.device,
                   "--benchmark-steps", str(args.benchmark_steps),
                   "--benchmark-warmup-steps", str(args.benchmark_warmup_steps)]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        markers = [line for line in completed.stdout.splitlines() if line.startswith("BENCHMARK_RESULT=")]
        if not markers:
            sys.stderr.write(completed.stdout + completed.stderr)
            raise ValueError(f"benchmark worker {count} did not return a result")
        row = json.loads(markers[-1].split("=", 1)[1])
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    winner = max(rows, key=lambda row: row["valid_simulated_seconds_per_wall_second"])
    report = {"selection_metric": "valid_simulated_seconds_per_wall_second", "candidates": rows,
              "selected_num_envs": winner["num_envs"], "selected_value": winner["valid_simulated_seconds_per_wall_second"]}
    path = ROOT / "reports/performance/parallel_env_benchmark.json"
    write_json(path, report)
    print(f"BENCHMARK_REPORT={path}")
    return report


def full_audit(stage: str) -> dict[str, Any]:
    from paper_eval_final.src.storage import validate_record
    audited = 0
    for completion_path in (ROOT / "results" / stage).rglob("completion.json"):
        shard = completion_path.parent
        for record_path in (shard / "records").glob("*.json"):
            record = read_json(record_path)
            validate_record(record)
            audited += 1
    return {"status": "PASS", "stage": stage, "audited_records": audited}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args._benchmark_one is not None:
        from paper_eval_final.src.executor import benchmark_once
        result = benchmark_once(model_id=args.model, num_envs=args._benchmark_one, steps=args.benchmark_steps,
                                warmup_steps=args.benchmark_warmup_steps, device=args.device)
        print("BENCHMARK_RESULT=" + json.dumps(result, sort_keys=True))
        return 0
    chosen = experiments(args)
    if args.prepare:
        stages = (args.stage,) if args.stage else ("screening", "pilot", "formal")
        print(json.dumps(prepare_all(stages, chosen or (1, 2, 3)), indent=2))
        return 0
    if args.benchmark:
        run_benchmark(args)
        return 0
    if not args.stage:
        raise ValueError("--stage is required")
    if args.full_audit:
        print(json.dumps(full_audit(args.stage), indent=2))
        return 0
    if args.aggregate:
        payload = aggregate(args.stage, chosen)
        for experiment, result in payload.items():
            write_experiment_report(args.stage, int(experiment), result)
            if args.stage == "screening" and int(experiment) == 1:
                write_screening_report(result)
        print(json.dumps(payload, indent=2))
        return 0
    experiment1_baseline_guard(args, chosen)
    formal_guard(args, chosen)
    prepare_all((args.stage,), chosen)
    if args._execute_one:
        from paper_eval_final.src.executor import execute_one
        result = execute_one(stage=args.stage, experiment=chosen[0], model_id=args.model,
                             baseline_id=args.baseline, num_envs=selected_num_envs(args),
                             device=args.device, smoke=args.smoke, train_seed=args.train_seed)
        print("RUN_RESULT=" + json.dumps(result, sort_keys=True))
        return 0
    # Each experiment is a separate process. Completion closes Isaac before the next launch.
    for experiment in chosen:
        for train_seed in registered_seeds(args, experiment):
            subprocess.run(worker_command(args, experiment, train_seed), check=True)
    return 0


if __name__ == "__main__":
    internal_worker = "--_execute-one" in sys.argv or "--_benchmark-one" in sys.argv
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if internal_worker:
            os._exit(1)
        raise
    sys.stdout.flush()
    sys.stderr.flush()
    if internal_worker:
        # Isaac Sim 5.1 may deadlock in SimulationApp.close() for this native
        # VecEnv.  Child certificate workers and physics are already stopped;
        # process teardown is the reliable final ownership boundary.
        os._exit(exit_code)
    raise SystemExit(exit_code)
