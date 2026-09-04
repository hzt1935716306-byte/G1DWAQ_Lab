#!/usr/bin/env python3
"""Train the 5x(96+3) deployable-IMU whole-body CoM velocity estimator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from isaaclab.app import AppLauncher


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER = REPOSITORY_ROOT / "logs/g1_slope_sys_d.pt"
DEFAULT_V1 = (
    REPOSITORY_ROOT
    / "logs/g1_com_velocity_estimator/2026-09-04_19-51-28/com_velocity_estimator_best.pt"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="g1_com_velocity_estimator_v2")
parser.add_argument("--teacher_checkpoint", type=Path, default=DEFAULT_TEACHER)
parser.add_argument("--v1_checkpoint", type=Path, default=DEFAULT_V1)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--train_envs", type=int, default=3584)
parser.add_argument("--validation_envs", type=int, default=512)
parser.add_argument("--train_policy_steps", type=int, default=5000)
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

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.estimation import (  # noqa: E402
    ComVelocityEstimator,
    ComVelocityEstimatorV2TrainCfg,
    ErrorMetricAccumulator,
    EstimatorFrameHistory,
    ResetWarmupMask,
    TouchdownAfterTransientTracker,
    extract_com_velocity_target,
    extract_recent_actor_history,
    latest_actor_frame,
    partitioned_recovery_group_mask,
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


def _selection_score(metrics: dict[str, Any]) -> float:
    """Favor the deployment-critical TD0/TD1 while retaining nominal accuracy."""

    weighted = (("td0", 0.50), ("td1", 0.25), ("overall", 0.25))
    available = [
        (float(metrics[name]["vector_rmse"]), weight)
        for name, weight in weighted
        if metrics[name]["vector_rmse"] is not None
    ]
    if not available:
        return float("inf")
    return sum(value * weight for value, weight in available) / sum(
        weight for _, weight in available
    )


class RecoveryHeavyScheduler:
    """Add frequent velocity jumps to half of each data partition."""

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
        self.remaining[ids] = torch.randint(
            lower, upper + 1, (ids.numel(),), device=self.device
        )

    def advance(self, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dones = dones.to(device=self.device, dtype=torch.bool)
        self.window_remaining = torch.clamp(self.window_remaining - 1, min=0)
        active = self.group & ~dones
        self.remaining[active] -= 1
        due = active & (self.remaining <= 0)
        self._resample(due)
        self.window_remaining[due] = self.cfg.recovery_window_steps
        reset_recovery = self.group & dones
        self.window_remaining[reset_recovery] = 0
        self._resample(reset_recovery)
        return due, self.window_remaining > 0


def _apply_recovery_velocity_jump(env, due: torch.Tensor) -> int:
    env_ids = due.nonzero(as_tuple=False).flatten()
    if env_ids.numel() == 0:
        return 0
    push_by_setting_velocity(
        env,
        env_ids,
        {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
    )
    return int(env_ids.numel())


def _build_config() -> ComVelocityEstimatorV2TrainCfg:
    cfg = ComVelocityEstimatorV2TrainCfg(
        estimator_task=args.task,
        teacher_checkpoint=str(args.teacher_checkpoint.resolve()),
        num_envs=args.num_envs,
        train_envs=args.train_envs,
        validation_envs=args.validation_envs,
        train_policy_steps=args.train_policy_steps,
        evaluation_policy_steps=args.evaluation_policy_steps,
        learning_rate=args.learning_rate,
        recovery_group_fraction=args.recovery_group_fraction,
        recovery_push_interval_s=tuple(args.recovery_push_interval_s),
        recovery_window_s=args.recovery_window_s,
        seed=args.seed,
    )
    cfg.validate()
    return cfg


def _load_v1(path: Path, device: str, teacher_hash: str) -> ComVelocityEstimator:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "input_dim": 480,
        "per_frame_obs_dim": 96,
        "history_length": 5,
        "hidden_dims": [256, 128, 64],
        "output_dim": 2,
        "output_frame": "heading",
        "output_quantity": "whole_body_com_velocity_xy",
        "output_unit": "m/s",
        "teacher_checkpoint_hash": teacher_hash,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"V1 checkpoint semantic mismatch for {key}")
    model = ComVelocityEstimator(input_dim=480).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def _checkpoint_payload(
    state_dict: dict[str, torch.Tensor],
    cfg: ComVelocityEstimatorV2TrainCfg,
    teacher: Path,
    teacher_hash: str,
    metrics: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "model_state_dict": state_dict,
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
        "training_step": int(step),
        "validation_metrics": metrics,
        "data_generation": {
            "nominal_group_fraction": 1.0 - cfg.recovery_group_fraction,
            "recovery_heavy_group_fraction": cfg.recovery_group_fraction,
            "recovery_disturbance": "push_by_setting_velocity; no added physical force",
            "recovery_push_interval_s": list(cfg.recovery_push_interval_s),
            "recovery_window_s": cfg.recovery_window_s,
        },
    }


def train() -> Path:  # noqa: C901
    cfg = _build_config()
    teacher_checkpoint = Path(cfg.teacher_checkpoint)
    v1_checkpoint = args.v1_checkpoint.resolve()
    teacher_hash = _sha256(teacher_checkpoint)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S_v2")
    run_dir = args.output_root.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    env_cfg, agent_cfg = task_registry.get_cfgs(cfg.estimator_task)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.scene.seed = cfg.seed
    env_cfg.device = args.device
    agent_cfg.device = args.device
    env_class = task_registry.get_task_class(cfg.estimator_task)
    env = env_class(env_cfg, headless=args.headless)
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
        or [teacher_checks[key] for key in ("actor_input_dim", "actor_observation_dim", "actor_frame_dim", "actor_history_length", "action_dim")]
        != [960, 960, 96, 10, 29]
    ):
        raise RuntimeError(f"frozen teacher contract failed: {teacher_checks}")

    extractor = G1PrivilegedStateExtractor(env)
    history = EstimatorFrameHistory(
        cfg.num_envs,
        history_length=cfg.estimator_history_length,
        actor_frame_dim=cfg.per_frame_obs_dim,
        imu_dim=cfg.imu_input_dim,
        imu_acceleration_scale=cfg.imu_acceleration_scale,
        device=args.device,
    )
    scheduler = RecoveryHeavyScheduler(cfg, args.device)
    warmup = ResetWarmupMask(cfg.num_envs, cfg.reset_warmup_policy_steps, args.device)
    td_tracker = TouchdownAfterTransientTracker(cfg.num_envs, args.device)
    previous_target = torch.zeros(cfg.num_envs, 2, device=args.device)
    previous_valid = torch.zeros(cfg.num_envs, dtype=torch.bool, device=args.device)

    estimator = ComVelocityEstimator(input_dim=cfg.input_dim).to(args.device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=cfg.learning_rate)
    best_score = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    window_metrics = _metric_accumulators()
    total_samples = 0
    total_transient = 0
    total_recovery_window = 0
    total_manual_pushes = 0
    loss_log = []
    val_slice = slice(cfg.train_envs, cfg.num_envs)

    def rollout_step():
        nonlocal observations, total_manual_pushes
        with torch.inference_mode():
            actions = teacher_policy(observations)
            observations, _, dones, _ = env.step(actions)
        dones = dones.to(dtype=torch.bool)
        due, recovery_window = scheduler.advance(dones)
        total_manual_pushes += _apply_recovery_velocity_jump(env, due)
        state = extractor.extract()
        reset = dones | state.episode_reset
        target = extract_com_velocity_target(state).detach()
        estimator_input = history.append(
            latest_actor_frame(observations), env.scene.sensors["imu"].data.lin_acc_b, reset
        )
        eligible = warmup.eligible_after_step(reset)
        transient = (
            previous_valid
            & ~reset
            & (torch.linalg.vector_norm(target - previous_target, dim=1) > cfg.transient_delta_v_threshold)
        )
        td0, td1 = td_tracker.update(transient, state.touchdown, reset)
        previous_target.copy_(target)
        previous_valid.copy_(~reset)
        return state, target, estimator_input, eligible, transient, td0, td1, recovery_window

    print(
        f"[EstimatorV2] train envs={cfg.num_envs} train={cfg.train_envs} "
        f"validation={cfg.validation_envs} steps={cfg.train_policy_steps}",
        flush=True,
    )
    for step in range(1, cfg.train_policy_steps + 1):
        state, target, estimator_input, eligible, transient, td0, td1, recovery_window = rollout_step()
        train_ok = eligible[: cfg.train_envs]
        estimator.train()
        if torch.any(train_ok):
            prediction = estimator(estimator_input[: cfg.train_envs][train_ok])
            loss = torch.mean(torch.square(prediction - target[: cfg.train_envs][train_ok]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(train_ok.sum())
            total_samples += count
            total_transient += int((transient[: cfg.train_envs] & train_ok).sum())
            total_recovery_window += int((recovery_window[: cfg.train_envs] & train_ok).sum())
        else:
            loss = torch.zeros((), device=args.device)

        with torch.inference_mode():
            estimator.eval()
            val_prediction = estimator(estimator_input[val_slice])
            _add_metrics(
                window_metrics,
                val_prediction,
                target[val_slice],
                eligible[val_slice],
                transient[val_slice],
                state.touchdown[val_slice],
                td0[val_slice],
                td1[val_slice],
            )
        if step % cfg.validation_interval_steps == 0 or step == cfg.train_policy_steps:
            summary = _summarize(window_metrics)
            score = _selection_score(summary)
            if score < best_score:
                best_score = score
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone() for key, value in estimator.state_dict().items()
                }
            window_metrics = _metric_accumulators()
        if step % cfg.log_interval_steps == 0 or step == cfg.train_policy_steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "samples": total_samples,
                "transient_samples": total_transient,
                "recovery_window_samples": total_recovery_window,
                "recovery_window_fraction": total_recovery_window / max(total_samples, 1),
                "best_validation_selection_score": best_score,
            }
            loss_log.append(row)
            print(
                f"[EstimatorV2] {step}/{cfg.train_policy_steps} loss={row['loss']:.6f} "
                f"transient={total_transient/max(total_samples,1):.3%} "
                f"recovery_window={row['recovery_window_fraction']:.3%} "
                f"best={best_score:.6f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("V2 training produced no validation candidate")
    last_state = {key: value.detach().cpu().clone() for key, value in estimator.state_dict().items()}
    candidates = {"window_best": best_state, "last": last_state}
    candidate_steps = {"window_best": best_step, "last": cfg.train_policy_steps}
    candidate_models = {}
    for name, state_dict in candidates.items():
        model = ComVelocityEstimator(input_dim=cfg.input_dim).to(args.device)
        model.load_state_dict(state_dict, strict=True)
        candidate_models[name] = model.eval().requires_grad_(False)
    v1_model = _load_v1(v1_checkpoint, args.device, teacher_hash)
    shared_accumulators = {
        **{name: _metric_accumulators() for name in candidates},
        "V1_5x96": _metric_accumulators(),
    }

    print(f"[EstimatorV2] shared V1/V2 evaluation steps={cfg.evaluation_policy_steps}", flush=True)
    for _ in range(cfg.evaluation_policy_steps):
        state, target, estimator_input, eligible, transient, td0, td1, _ = rollout_step()
        with torch.inference_mode():
            for name, model in candidate_models.items():
                prediction = model(estimator_input[val_slice])
                _add_metrics(
                    shared_accumulators[name], prediction, target[val_slice], eligible[val_slice],
                    transient[val_slice], state.touchdown[val_slice], td0[val_slice], td1[val_slice]
                )
            v1_input = extract_recent_actor_history(observations)[val_slice]
            _add_metrics(
                shared_accumulators["V1_5x96"], v1_model(v1_input), target[val_slice],
                eligible[val_slice], transient[val_slice], state.touchdown[val_slice],
                td0[val_slice], td1[val_slice]
            )
    shared_metrics = {name: _summarize(value) for name, value in shared_accumulators.items()}
    scores = {name: _selection_score(shared_metrics[name]) for name in candidates}
    selected = min(scores, key=scores.get)
    selected_state = candidates[selected]
    selected_step = candidate_steps[selected]
    selected_metrics = shared_metrics[selected]

    recovery_fraction = total_recovery_window / max(total_samples, 1)
    if recovery_fraction < 0.01:
        raise RuntimeError(f"recovery-window training fraction is too small: {recovery_fraction:.3%}")

    best_payload = _checkpoint_payload(
        selected_state, cfg, teacher_checkpoint, teacher_hash, selected_metrics, selected_step
    )
    best_payload["selection"] = {"method": "0.5*TD0 + 0.25*TD1 + 0.25*overall vector RMSE", "scores": scores, "selected": selected}
    last_payload = _checkpoint_payload(
        last_state, cfg, teacher_checkpoint, teacher_hash, shared_metrics["last"], cfg.train_policy_steps
    )
    best_path = run_dir / "com_velocity_estimator_v2_best.pt"
    last_path = run_dir / "com_velocity_estimator_v2_last.pt"
    torch.save(best_payload, best_path)
    torch.save(last_payload, last_path)

    reload_model = ComVelocityEstimator(input_dim=cfg.input_dim).to(args.device)
    reload_payload = torch.load(best_path, map_location=args.device, weights_only=False)
    reload_model.load_state_dict(reload_payload["model_state_dict"], strict=True)
    reload_model.eval()
    probe = estimator_input[val_slice][: min(16, cfg.validation_envs)]
    with torch.inference_mode():
        reload_difference = float(
            torch.max(torch.abs(candidate_models[selected](probe) - reload_model(probe)))
        )
    if reload_difference != 0.0:
        raise RuntimeError(f"V2 reload inference mismatch: {reload_difference}")

    config = asdict(cfg) | {
        "input_dim": cfg.input_dim,
        "estimator_frame_dim": cfg.estimator_frame_dim,
        "teacher_checkpoint_hash": teacher_hash,
        "v1_checkpoint": str(v1_checkpoint),
        "v1_checkpoint_hash": _sha256(v1_checkpoint),
        "git_commit": _git_commit(),
        "device": args.device,
        "IMU_semantics": {
            "sensor": "G1 pelvis imu_in_pelvis",
            "quantity": "specific force body xyz",
            "unit": "m/s^2 before configured scaling",
            "privileged_acceleration": False,
        },
    }
    (run_dir / "train_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    report = {
        "teacher_checks": teacher_checks,
        "architecture": [495, 256, 128, 64, 2],
        "target": "G1PrivilegedStateExtractor whole-body CoM velocity XY in heading frame",
        "training": {
            "policy_steps": cfg.train_policy_steps,
            "samples": total_samples,
            "transient_samples": total_transient,
            "transient_fraction": total_transient / max(total_samples, 1),
            "recovery_window_samples": total_recovery_window,
            "recovery_window_fraction": recovery_fraction,
            "manual_recovery_pushes": total_manual_pushes,
            "loss_log": loss_log,
        },
        "shared_evaluation": {
            "policy_steps": cfg.evaluation_policy_steps,
            "V1": shared_metrics["V1_5x96"],
            "V2": selected_metrics,
            "V2_candidates": {name: shared_metrics[name] for name in candidates},
        },
        "selection": best_payload["selection"],
        "checkpoint_reload_max_abs_difference": reload_difference,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return best_path


if __name__ == "__main__":
    try:
        train()
    finally:
        simulation_app.close()
