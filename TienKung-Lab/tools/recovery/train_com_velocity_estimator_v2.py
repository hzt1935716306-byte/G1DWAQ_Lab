#!/usr/bin/env python3
"""Iteration-based long training for the 5x(96+3) deployable-IMU CoM estimator."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from isaaclab.app import AppLauncher


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER = REPOSITORY_ROOT / "logs/g1_slope_sys_d.pt"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="g1_com_velocity_estimator_v2")
parser.add_argument("--teacher_checkpoint", type=Path, default=DEFAULT_TEACHER)
parser.add_argument("--resume_checkpoint", type=Path, default=None)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--train_envs", type=int, default=3584)
parser.add_argument("--validation_envs", type=int, default=512)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--num_steps_per_iteration", type=int, default=24)
parser.add_argument("--mini_batch_size", type=int, default=16384)
parser.add_argument("--num_learning_epochs", type=int, default=1)
parser.add_argument("--validation_interval_iterations", type=int, default=50)
parser.add_argument("--save_interval_iterations", type=int, default=100)
parser.add_argument(
    "--train_policy_steps",
    type=int,
    default=None,
    help="Deprecated V2 compatibility argument; iteration mode uses --max_iterations.",
)
parser.add_argument("--evaluation_policy_steps", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1.0e-3)
parser.add_argument("--recovery_group_fraction", type=float, default=0.5)
parser.add_argument("--recovery_push_interval_s", type=float, nargs=2, default=(0.6, 0.9))
parser.add_argument("--recovery_window_s", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--output_root", type=Path, default=REPOSITORY_ROOT / "logs/g1_com_velocity_estimator"
)
parser.add_argument("--run_name", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.envs.mdp.events import push_by_setting_velocity  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.estimation import (  # noqa: E402
    ComVelocityEstimator,
    ComVelocityEstimatorV2TrainCfg,
    ErrorMetricAccumulator,
    EstimatorFrameHistory,
    EstimatorRolloutBuffer,
    ManualPushAlignmentDiagnostic,
    ResetWarmupMask,
    TouchdownAfterTransientTracker,
    extract_com_velocity_target,
    fixed_length_rollout_indices,
    iteration_checkpoint_filename,
    latest_actor_frame,
    load_v2_training_checkpoint,
    partitioned_recovery_group_mask,
    velocity_estimator_selection_score,
)
from legged_lab.recovery.state_extractor import G1PrivilegedStateExtractor  # noqa: E402
from legged_lab.utils import task_registry  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _metric_accumulators() -> dict[str, ErrorMetricAccumulator]:
    return {
        name: ErrorMetricAccumulator()
        for name in ("overall", "transient", "touchdown", "td0", "td1")
    }


def _add_metrics(
    accumulators: dict[str, ErrorMetricAccumulator],
    prediction: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
    transient: torch.Tensor,
    touchdown: torch.Tensor,
    td0: torch.Tensor,
    td1: torch.Tensor,
) -> None:
    accumulators["overall"].add(prediction, target, eligible)
    accumulators["transient"].add(prediction, target, eligible & transient)
    accumulators["touchdown"].add(prediction, target, eligible & touchdown)
    accumulators["td0"].add(prediction, target, eligible & td0)
    accumulators["td1"].add(prediction, target, eligible & td1)


def _summarize(accumulators: dict[str, ErrorMetricAccumulator]) -> dict[str, Any]:
    return {name: accumulator.summary() for name, accumulator in accumulators.items()}


class RecoveryHeavyScheduler:
    """Schedule extra velocity jumps before physics for half of each data split."""

    def __init__(self, cfg: ComVelocityEstimatorV2TrainCfg, device: str) -> None:
        self.cfg = cfg
        self.device = device
        self.group = partitioned_recovery_group_mask(
            cfg.num_envs, cfg.train_envs, cfg.recovery_group_fraction, device
        )
        self.remaining = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
        self.window_remaining = torch.zeros_like(self.remaining)
        self._resample(self.group)

    def _resample(self, mask: torch.Tensor) -> None:
        ids = mask.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        lower, upper = self.cfg.recovery_push_interval_steps
        self.remaining[ids] = torch.randint(lower, upper + 1, (ids.numel(),), device=self.device)

    def before_step(self, previous_reset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decide and expose a push before the matching action is stepped."""

        previous_reset = previous_reset.to(device=self.device, dtype=torch.bool)
        self.window_remaining = torch.clamp(self.window_remaining - 1, min=0)
        reset_recovery = self.group & previous_reset
        self.window_remaining[reset_recovery] = 0
        self._resample(reset_recovery)
        active = self.group & ~previous_reset
        self.remaining[active] -= 1
        due = active & (self.remaining <= 0)
        self._resample(due)
        self.window_remaining[due] = self.cfg.recovery_window_steps
        return due, self.window_remaining > 0


def _apply_recovery_velocity_jump(env, due: torch.Tensor) -> int:
    env_ids = due.nonzero(as_tuple=False).flatten()
    if env_ids.numel() == 0:
        return 0
    push_by_setting_velocity(env, env_ids, {"x": (-1.0, 1.0), "y": (-1.0, 1.0)})
    return int(env_ids.numel())


def _build_config() -> ComVelocityEstimatorV2TrainCfg:
    cfg = ComVelocityEstimatorV2TrainCfg(
        estimator_task=args.task,
        teacher_checkpoint=str(args.teacher_checkpoint.resolve()),
        num_envs=args.num_envs,
        train_envs=args.train_envs,
        validation_envs=args.validation_envs,
        max_iterations=args.max_iterations,
        num_steps_per_iteration=args.num_steps_per_iteration,
        mini_batch_size=args.mini_batch_size,
        num_learning_epochs=args.num_learning_epochs,
        validation_interval_iterations=args.validation_interval_iterations,
        save_interval_iterations=args.save_interval_iterations,
        evaluation_policy_steps=args.evaluation_policy_steps,
        learning_rate=args.learning_rate,
        recovery_group_fraction=args.recovery_group_fraction,
        recovery_push_interval_s=tuple(args.recovery_push_interval_s),
        recovery_window_s=args.recovery_window_s,
        seed=args.seed,
    )
    cfg.validate()
    return cfg


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return _cpu_copy(model.state_dict())


def _teacher_snapshot(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in policy.named_parameters()}


def _teacher_is_identical(
    before: dict[str, torch.Tensor], policy: torch.nn.Module
) -> bool:
    current = dict(policy.named_parameters())
    return before.keys() == current.keys() and all(
        torch.equal(before[name], current[name].detach().cpu()) for name in before
    )


def _checkpoint_payload(
    *,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    cfg: ComVelocityEstimatorV2TrainCfg,
    teacher: Path,
    teacher_hash: str,
    current_iteration: int,
    global_policy_steps: int,
    optimizer_updates: int,
    best_selection_score: float,
    best_iteration: int,
    best_model_state: dict[str, torch.Tensor],
    best_optimizer_state: dict[str, Any],
    validation_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "training_format": "iteration_v1",
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "current_iteration": int(current_iteration),
        "global_policy_steps": int(global_policy_steps),
        "optimizer_updates": int(optimizer_updates),
        "best_selection_score": float(best_selection_score),
        "best_iteration": int(best_iteration),
        "best_model_state_dict": best_model_state,
        "best_optimizer_state_dict": best_optimizer_state,
        "best_validation_metrics": validation_metrics,
        "training_configuration": asdict(cfg),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "input_dim": cfg.input_dim,
        "per_frame_obs_dim": cfg.estimator_frame_dim,
        "actor_per_frame_obs_dim": cfg.per_frame_obs_dim,
        "imu_input_dim": cfg.imu_input_dim,
        "history_length": cfg.estimator_history_length,
        "hidden_dims": list(cfg.hidden_dims),
        "output_dim": cfg.output_dim,
        "activation": "ELU",
        "output_frame": "heading",
        "output_quantity": "whole_body_com_velocity_xy",
        "output_unit": "m/s",
        "input_frame_layout": "5 x [actor_noisy_scaled_96, pelvis_IMU_specific_force_body_xyz_scaled]",
        "imu_quantity": "deployable_pelvis_specific_force_body_xyz",
        "imu_unit_before_scaling": "m/s^2",
        "imu_acceleration_scale": cfg.imu_acceleration_scale,
        "teacher_task": cfg.teacher_task,
        "teacher_checkpoint": str(teacher),
        "teacher_checkpoint_hash": teacher_hash,
        "git_commit": _git_commit(),
        "data_generation": {
            "nominal_group_fraction": 1.0 - cfg.recovery_group_fraction,
            "recovery_heavy_group_fraction": cfg.recovery_group_fraction,
            "recovery_disturbance": "manual push_by_setting_velocity before env.step",
            "built_in_disturbance": "unchanged g1_slope_sys_d interval velocity push",
            "recovery_push_interval_s": list(cfg.recovery_push_interval_s),
            "recovery_window_s": cfg.recovery_window_s,
        },
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    torch.save(payload, path)


def train() -> Path:  # noqa: C901,PLR0915
    cfg = _build_config()
    teacher_checkpoint = Path(cfg.teacher_checkpoint)
    teacher_hash = _sha256(teacher_checkpoint)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S_v2_long")
    run_dir = args.output_root.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    iteration_log_path = run_dir / "iteration_metrics.jsonl"
    iteration_log_path.write_text("", encoding="utf-8")
    tensorboard_writer = SummaryWriter(log_dir=str(run_dir))

    env_cfg, agent_cfg = task_registry.get_cfgs(cfg.estimator_task)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.scene.seed = cfg.seed
    env_cfg.device = args.device
    agent_cfg.device = args.device
    env = task_registry.get_task_class(cfg.estimator_task)(env_cfg, headless=args.headless)
    if "imu" not in env.scene.sensors:
        raise RuntimeError("V2 task has no deployable pelvis IMU")
    if abs(env.step_dt - cfg.policy_dt) > 1.0e-9:
        raise RuntimeError(f"policy dt mismatch: {env.step_dt} != {cfg.policy_dt}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(teacher_checkpoint), load_optimizer=False)
    runner.eval_mode()
    runner.alg.policy.requires_grad_(False)
    teacher_policy = runner.get_inference_policy(device=args.device)
    observations, _ = env.get_observations()
    current_actor, _ = env.compute_current_observations()
    actor = runner.alg.policy.actor
    first = next(layer for layer in actor if isinstance(layer, torch.nn.Linear))
    last = next(layer for layer in reversed(actor) if isinstance(layer, torch.nn.Linear))
    teacher_checks = {
        "strict_checkpoint_load": True,
        "optimizer_loaded": False,
        "teacher_eval": not runner.alg.policy.training,
        "teacher_has_trainable_parameter": any(
            value.requires_grad for value in runner.alg.policy.parameters()
        ),
        "actor_input_dim": first.in_features,
        "actor_observation_dim": observations.shape[1],
        "actor_frame_dim": current_actor.shape[1],
        "actor_history_length": env.cfg.robot.actor_obs_history_length,
        "action_dim": last.out_features,
    }
    if (
        teacher_checks["teacher_has_trainable_parameter"]
        or not teacher_checks["teacher_eval"]
        or [
            teacher_checks[key]
            for key in (
                "actor_input_dim",
                "actor_observation_dim",
                "actor_frame_dim",
                "actor_history_length",
                "action_dim",
            )
        ]
        != [960, 960, 96, 10, 29]
    ):
        raise RuntimeError(f"frozen teacher contract failed: {teacher_checks}")
    frozen_teacher = _teacher_snapshot(runner.alg.policy)

    estimator = ComVelocityEstimator(input_dim=cfg.input_dim).to(args.device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=cfg.learning_rate)
    resume_report = {
        "requested_checkpoint": str(args.resume_checkpoint.resolve())
        if args.resume_checkpoint is not None
        else None,
        "mode": "fresh_start",
        "optimizer_restarted": False,
    }
    start_iteration = 0
    global_policy_steps = 0
    optimizer_updates = 0
    best_score = float("inf")
    best_iteration = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_validation_metrics: dict[str, Any] | None = None
    if args.resume_checkpoint is not None:
        resume_info = load_v2_training_checkpoint(
            args.resume_checkpoint.resolve(),
            estimator,
            optimizer,
            teacher_hash=teacher_hash,
        )
        start_iteration = resume_info.start_iteration
        global_policy_steps = resume_info.global_policy_steps
        optimizer_updates = resume_info.optimizer_updates
        best_score = resume_info.best_selection_score
        best_iteration = resume_info.best_iteration
        resume_report.update(
            {
                "mode": resume_info.mode,
                "optimizer_restarted": resume_info.optimizer_restarted,
                "restored_iteration": start_iteration,
                "restored_global_policy_steps": global_policy_steps,
            }
        )
        best_state = _cpu_copy(
            resume_info.payload.get("best_model_state_dict", resume_info.payload["model_state_dict"])
        )
        best_optimizer_state = _cpu_copy(
            resume_info.payload.get(
                "best_optimizer_state_dict", resume_info.payload["optimizer_state_dict"]
            )
        )
        best_validation_metrics = resume_info.payload.get("best_validation_metrics")
        print(
            f"[EstimatorV2] resumed formal iteration checkpoint at iteration {start_iteration}",
            flush=True,
        )
    if start_iteration >= cfg.max_iterations:
        raise ValueError(
            f"max_iterations={cfg.max_iterations} must exceed resumed iteration={start_iteration}"
        )

    extractor = G1PrivilegedStateExtractor(env)
    history = EstimatorFrameHistory(
        cfg.num_envs,
        history_length=cfg.estimator_history_length,
        actor_frame_dim=cfg.per_frame_obs_dim,
        imu_dim=cfg.imu_input_dim,
        imu_acceleration_scale=cfg.imu_acceleration_scale,
        device=args.device,
    )
    rollout = EstimatorRolloutBuffer(
        cfg.num_steps_per_iteration,
        cfg.train_envs,
        cfg.input_dim,
        device=args.device,
    )
    scheduler = RecoveryHeavyScheduler(cfg, args.device)
    warmup = ResetWarmupMask(cfg.num_envs, cfg.reset_warmup_policy_steps, args.device)
    td_tracker = TouchdownAfterTransientTracker(cfg.num_envs, args.device)
    previous_target = torch.zeros(cfg.num_envs, 2, device=args.device)
    previous_valid = torch.zeros(cfg.num_envs, dtype=torch.bool, device=args.device)
    previous_reset = torch.zeros(cfg.num_envs, dtype=torch.bool, device=args.device)
    alignment = ManualPushAlignmentDiagnostic()
    validation_window = _metric_accumulators()
    val_slice = slice(cfg.train_envs, cfg.num_envs)
    total_manual_pushes = 0
    latest_validation: dict[str, Any] | None = None
    long_best_path = run_dir / "com_velocity_estimator_v2_long_best.pt"
    long_last_path = run_dir / "com_velocity_estimator_v2_long_last.pt"

    training_config = asdict(cfg) | {
        "total_expected_policy_steps": cfg.total_expected_policy_steps,
        "expected_train_samples": cfg.expected_train_samples,
        "resume": resume_report,
        "teacher_checkpoint_hash": teacher_hash,
        "git_commit": _git_commit(),
        "device": args.device,
        "rollout_buffer_bytes": rollout.allocated_bytes,
        "deprecated_train_policy_steps_argument": args.train_policy_steps,
    }
    (run_dir / "train_config.yaml").write_text(
        yaml.safe_dump(training_config, sort_keys=False), encoding="utf-8"
    )
    print(
        "[EstimatorV2] iteration training\n"
        f"  max_iterations={cfg.max_iterations}\n"
        f"  num_steps_per_iteration={cfg.num_steps_per_iteration}\n"
        f"  total_expected_policy_steps={cfg.total_expected_policy_steps}\n"
        f"  expected_train_samples={cfg.expected_train_samples}\n"
        f"  mini_batch_size={cfg.mini_batch_size}\n"
        f"  num_learning_epochs={cfg.num_learning_epochs}\n"
        f"  rollout_buffer={rollout.allocated_bytes / (1024**2):.2f} MiB",
        flush=True,
    )
    if args.train_policy_steps is not None:
        print(
            "[EstimatorV2] --train_policy_steps is deprecated and ignored in iteration mode",
            flush=True,
        )

    def rollout_step(frame_step: int):
        nonlocal observations, total_manual_pushes
        with torch.inference_mode():
            actions = teacher_policy(observations)
        due, recovery_window = scheduler.before_step(previous_reset)
        push_sim_step = int(env.sim_step_counter)
        pushed_count = _apply_recovery_velocity_jump(env, due)
        total_manual_pushes += pushed_count
        with torch.inference_mode():
            observations, _, dones, _ = env.step(actions)
        post_step_sim_step = int(env.sim_step_counter)
        dones = dones.to(dtype=torch.bool)
        imu = env.scene.sensors["imu"]
        imu_acceleration = imu.data.lin_acc_b
        state = extractor.extract()
        reset = dones | state.episode_reset
        target = extract_com_velocity_target(state).detach()
        estimator_input = history.append(
            latest_actor_frame(observations), imu_acceleration, reset
        )
        eligible = warmup.eligible_after_step(reset)
        transient = (
            previous_valid
            & ~reset
            & (
                torch.linalg.vector_norm(target - previous_target, dim=1)
                > cfg.transient_delta_v_threshold
            )
        )
        td0, td1 = td_tracker.update(transient, state.touchdown, reset)
        if pushed_count:
            due_ids = due.nonzero(as_tuple=False).flatten()
            imu_timestamp = float(imu._timestamp_last_update[due_ids].mean().item())
            imu_current_timestamp = float(imu._timestamp[due_ids].mean().item())
            alignment.record(
                pushed_env_count=pushed_count,
                global_policy_step=frame_step,
                push_sim_step=push_sim_step,
                post_step_sim_step=post_step_sim_step,
                sim_decimation=env.cfg.sim.decimation,
                observation_frame_sim_step=post_step_sim_step,
                imu_frame_sim_step=post_step_sim_step,
                target_frame_sim_step=post_step_sim_step,
                imu_timestamp_s=imu_timestamp,
                imu_current_timestamp_s=imu_current_timestamp,
            )
        previous_target.copy_(target)
        previous_valid.copy_(~reset)
        previous_reset.copy_(reset)
        return state, target, estimator_input, eligible, transient, td0, td1, recovery_window

    training_start = time.perf_counter()
    for iteration_index in range(start_iteration, cfg.max_iterations):
        iteration = iteration_index + 1
        iteration_start = time.perf_counter()
        rollout.clear()
        for _ in fixed_length_rollout_indices(cfg.num_steps_per_iteration):
            global_policy_steps += 1
            state, target, estimator_input, eligible, transient, td0, td1, recovery_window = (
                rollout_step(global_policy_steps)
            )
            rollout.add(
                estimator_input[: cfg.train_envs],
                target[: cfg.train_envs],
                eligible[: cfg.train_envs],
                transient[: cfg.train_envs],
                recovery_window[: cfg.train_envs],
                state.touchdown[: cfg.train_envs],
                td0[: cfg.train_envs],
                td1[: cfg.train_envs],
            )
            with torch.inference_mode():
                estimator.eval()
                prediction = estimator(estimator_input[val_slice])
                _add_metrics(
                    validation_window,
                    prediction,
                    target[val_slice],
                    eligible[val_slice],
                    transient[val_slice],
                    state.touchdown[val_slice],
                    td0[val_slice],
                    td1[val_slice],
                )

        counts = rollout.counts()
        if counts["eligible_train_samples"] <= 0:
            raise RuntimeError("iteration rollout has no eligible training samples")
        estimator.train()
        loss_numerator = 0.0
        loss_denominator = 0
        iteration_updates = 0
        for _, _, batch_inputs, batch_targets in rollout.iter_minibatches(
            cfg.mini_batch_size, cfg.num_learning_epochs
        ):
            prediction = estimator(batch_inputs)
            loss = torch.mean(torch.square(prediction - batch_targets))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_count = int(batch_inputs.shape[0])
            loss_numerator += float(loss.detach()) * batch_count
            loss_denominator += batch_count
            iteration_updates += 1
        optimizer_updates += iteration_updates
        mean_loss = loss_numerator / max(loss_denominator, 1)

        validation_due = (
            iteration % cfg.validation_interval_iterations == 0
            or iteration == cfg.max_iterations
        )
        validation_score = None
        best_updated = False
        if validation_due:
            latest_validation = _summarize(validation_window)
            validation_score = velocity_estimator_selection_score(latest_validation)
            if validation_score < best_score:
                best_score = validation_score
                best_iteration = iteration
                best_state = _model_state(estimator)
                best_optimizer_state = _cpu_copy(optimizer.state_dict())
                best_validation_metrics = latest_validation
                best_updated = True
            validation_window = _metric_accumulators()

        if best_state is None or best_optimizer_state is None:
            # Before the first scheduled validation, keep a resumable provisional best.
            best_state = _model_state(estimator)
            best_optimizer_state = _cpu_copy(optimizer.state_dict())

        elapsed_iteration = time.perf_counter() - iteration_start
        elapsed_total = time.perf_counter() - training_start
        env_steps_per_second = (
            cfg.num_envs * cfg.num_steps_per_iteration / max(elapsed_iteration, 1.0e-9)
        )
        row = {
            "iteration": iteration,
            "global_policy_steps": global_policy_steps,
            "optimizer_updates": optimizer_updates,
            **counts,
            "mean_supervised_loss": mean_loss,
            "validation_score": validation_score,
            "best_selection_score": best_score,
            "best_iteration": best_iteration,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_time_s": elapsed_total,
            "iteration_time_s": elapsed_iteration,
            "environment_steps_per_second": env_steps_per_second,
            "manual_recovery_pushes_total": total_manual_pushes,
        }
        with iteration_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        tensorboard_writer.add_scalar("Train/mean_supervised_loss", mean_loss, iteration)
        tensorboard_writer.add_scalar(
            "Train/eligible_train_samples", counts["eligible_train_samples"], iteration
        )
        tensorboard_writer.add_scalar(
            "Train/transient_samples", counts["transient_samples"], iteration
        )
        tensorboard_writer.add_scalar(
            "Train/recovery_window_samples", counts["recovery_window_samples"], iteration
        )
        tensorboard_writer.add_scalar("Train/TD0_samples", counts["TD0_samples"], iteration)
        tensorboard_writer.add_scalar("Train/TD1_samples", counts["TD1_samples"], iteration)
        tensorboard_writer.add_scalar("Train/optimizer_updates", optimizer_updates, iteration)
        tensorboard_writer.add_scalar(
            "Train/environment_steps_per_second", env_steps_per_second, iteration
        )
        tensorboard_writer.add_scalar(
            "Train/learning_rate", optimizer.param_groups[0]["lr"], iteration
        )
        tensorboard_writer.add_scalar(
            "Progress/global_policy_steps", global_policy_steps, iteration
        )
        if validation_score is not None and latest_validation is not None:
            tensorboard_writer.add_scalar("Validation/selection_score", validation_score, iteration)
            tensorboard_writer.add_scalar("Validation/best_selection_score", best_score, iteration)
            for group, metrics in latest_validation.items():
                for metric_name in (
                    "vector_rmse",
                    "p95_vector_error",
                    "bias_x",
                    "bias_y",
                ):
                    value = metrics[metric_name]
                    if value is not None:
                        tensorboard_writer.add_scalar(
                            f"Validation/{group}_{metric_name}", value, iteration
                        )
            tensorboard_writer.flush()
        print(
            f"[EstimatorV2] iteration={iteration}/{cfg.max_iterations} "
            f"policy_steps={global_policy_steps} samples={counts['eligible_train_samples']} "
            f"updates={iteration_updates} loss={mean_loss:.6f} "
            f"validation={validation_score} env_steps/s={env_steps_per_second:.0f}",
            flush=True,
        )

        def payload_for_current() -> dict[str, Any]:
            return _checkpoint_payload(
                model_state=_model_state(estimator),
                optimizer_state=_cpu_copy(optimizer.state_dict()),
                cfg=cfg,
                teacher=teacher_checkpoint,
                teacher_hash=teacher_hash,
                current_iteration=iteration,
                global_policy_steps=global_policy_steps,
                optimizer_updates=optimizer_updates,
                best_selection_score=best_score,
                best_iteration=best_iteration,
                best_model_state=best_state,
                best_optimizer_state=best_optimizer_state,
                validation_metrics=best_validation_metrics,
            )

        if best_updated:
            best_payload = _checkpoint_payload(
                model_state=best_state,
                optimizer_state=best_optimizer_state,
                cfg=cfg,
                teacher=teacher_checkpoint,
                teacher_hash=teacher_hash,
                current_iteration=best_iteration,
                global_policy_steps=best_iteration * cfg.num_steps_per_iteration,
                optimizer_updates=optimizer_updates,
                best_selection_score=best_score,
                best_iteration=best_iteration,
                best_model_state=best_state,
                best_optimizer_state=best_optimizer_state,
                validation_metrics=best_validation_metrics,
            )
            _write_checkpoint(long_best_path, best_payload)
        periodic_due = iteration % cfg.save_interval_iterations == 0
        if periodic_due:
            current_payload = payload_for_current()
            _write_checkpoint(run_dir / iteration_checkpoint_filename(iteration), current_payload)
            _write_checkpoint(long_last_path, current_payload)
        elif validation_due or iteration == cfg.max_iterations:
            _write_checkpoint(long_last_path, payload_for_current())

    if best_state is not None and best_optimizer_state is not None and not long_best_path.is_file():
        carried_best_payload = _checkpoint_payload(
            model_state=best_state,
            optimizer_state=best_optimizer_state,
            cfg=cfg,
            teacher=teacher_checkpoint,
            teacher_hash=teacher_hash,
            current_iteration=best_iteration,
            global_policy_steps=best_iteration * cfg.num_steps_per_iteration,
            optimizer_updates=optimizer_updates,
            best_selection_score=best_score,
            best_iteration=best_iteration,
            best_model_state=best_state,
            best_optimizer_state=best_optimizer_state,
            validation_metrics=best_validation_metrics,
        )
        _write_checkpoint(long_best_path, carried_best_payload)
    if best_state is None or not long_best_path.is_file():
        raise RuntimeError("long training did not produce a validated best checkpoint")
    if not long_last_path.is_file():
        raise RuntimeError("long training did not produce a last checkpoint")

    best_model = ComVelocityEstimator(input_dim=cfg.input_dim).to(args.device)
    best_model.load_state_dict(best_state, strict=True)
    best_model.eval().requires_grad_(False)
    last_model = ComVelocityEstimator(input_dim=cfg.input_dim).to(args.device)
    last_model.load_state_dict(estimator.state_dict(), strict=True)
    last_model.eval().requires_grad_(False)
    evaluation = {"best": _metric_accumulators(), "last": _metric_accumulators()}
    print(
        f"[EstimatorV2] shared best/last evaluation steps={cfg.evaluation_policy_steps}",
        flush=True,
    )
    for evaluation_step in fixed_length_rollout_indices(cfg.evaluation_policy_steps):
        state, target, estimator_input, eligible, transient, td0, td1, _ = rollout_step(
            global_policy_steps + evaluation_step + 1
        )
        with torch.inference_mode():
            for name, model in (("best", best_model), ("last", last_model)):
                prediction = model(estimator_input[val_slice])
                _add_metrics(
                    evaluation[name],
                    prediction,
                    target[val_slice],
                    eligible[val_slice],
                    transient[val_slice],
                    state.touchdown[val_slice],
                    td0[val_slice],
                    td1[val_slice],
                )
    final_evaluation = {name: _summarize(value) for name, value in evaluation.items()}

    teacher_unchanged = _teacher_is_identical(frozen_teacher, runner.alg.policy)
    if not teacher_unchanged:
        raise RuntimeError("frozen teacher parameters changed during supervised training")
    reload_model = ComVelocityEstimator(input_dim=cfg.input_dim).to(args.device)
    reload_payload = torch.load(long_last_path, map_location=args.device, weights_only=False)
    reload_model.load_state_dict(reload_payload["model_state_dict"], strict=True)
    probe = estimator_input[val_slice][: min(16, cfg.validation_envs)]
    with torch.inference_mode():
        reload_difference = float(torch.max(torch.abs(last_model(probe) - reload_model(probe))))
    if reload_difference != 0.0:
        raise RuntimeError(f"long checkpoint reload inference mismatch: {reload_difference}")

    report = {
        "teacher_checks": teacher_checks
        | {"parameters_identical_before_after_training": teacher_unchanged},
        "architecture": [495, 256, 128, 64, 2],
        "target": "G1PrivilegedStateExtractor whole-body CoM velocity XY in heading frame",
        "iteration_training": {
            "start_iteration": start_iteration,
            "completed_iteration": cfg.max_iterations,
            "num_steps_per_iteration": cfg.num_steps_per_iteration,
            "global_policy_steps": global_policy_steps,
            "optimizer_updates": optimizer_updates,
            "rollout_buffer_bytes": rollout.allocated_bytes,
            "total_manual_recovery_pushes": total_manual_pushes,
        },
        "resume": resume_report,
        "best": {
            "iteration": best_iteration,
            "selection_score": best_score,
            "validation_metrics": best_validation_metrics,
        },
        "shared_best_last_evaluation": final_evaluation,
        "manual_push_frame_alignment": {
            "record_count": len(alignment.records),
            "all_aligned": all(record["aligned"] for record in alignment.records),
            "records": alignment.records,
        },
        "checkpoint_reload_max_abs_difference": reload_difference,
        "paths": {
            "best": str(long_best_path),
            "last": str(long_last_path),
            "iteration_metrics": str(iteration_log_path),
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tensorboard_writer.flush()
    tensorboard_writer.close()
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return long_best_path


if __name__ == "__main__":
    try:
        train()
    finally:
        simulation_app.close()
