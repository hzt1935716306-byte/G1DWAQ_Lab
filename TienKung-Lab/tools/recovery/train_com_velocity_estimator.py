#!/usr/bin/env python3
"""Train a standalone whole-body CoM horizontal velocity estimator online."""

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
DEFAULT_TEACHER_CHECKPOINT = REPOSITORY_ROOT / "logs/g1_slope_sys_d.pt"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="g1_com_velocity_estimator")
parser.add_argument("--teacher_checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--train_envs", type=int, default=3584)
parser.add_argument("--validation_envs", type=int, default=512)
parser.add_argument("--train_policy_steps", type=int, default=5000)
parser.add_argument("--evaluation_policy_steps", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1.0e-3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_root", type=Path, default=REPOSITORY_ROOT / "logs/g1_com_velocity_estimator")
parser.add_argument("--run_name", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.estimation.com_velocity_estimator import (  # noqa: E402
    ComVelocityEstimator,
    ComVelocityEstimatorTrainCfg,
    ErrorMetricAccumulator,
    ResetWarmupMask,
    extract_com_velocity_target,
    extract_recent_actor_history,
    weighted_velocity_mse,
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
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _metric_accumulators() -> dict[str, ErrorMetricAccumulator]:
    return {
        "overall": ErrorMetricAccumulator(),
        "transient": ErrorMetricAccumulator(),
        "touchdown": ErrorMetricAccumulator(),
        "td0": ErrorMetricAccumulator(),
    }


def _add_metrics(
    accumulators: dict[str, ErrorMetricAccumulator],
    prediction: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
    transient: torch.Tensor,
    touchdown: torch.Tensor,
    td0: torch.Tensor,
) -> None:
    accumulators["overall"].add(prediction, target, eligible)
    accumulators["transient"].add(prediction, target, eligible & transient)
    accumulators["touchdown"].add(prediction, target, eligible & touchdown)
    accumulators["td0"].add(prediction, target, eligible & td0)


def _summarize(accumulators: dict[str, ErrorMetricAccumulator]) -> dict[str, Any]:
    return {name: accumulator.summary() for name, accumulator in accumulators.items()}


def _transient_weighted_selection_score(metrics: dict[str, Any], transient_weight: float) -> float:
    """Use the training weights to compare candidates on one shared validation rollout."""

    overall = metrics["overall"]
    transient = metrics["transient"]
    overall_count = int(overall["count"])
    transient_count = int(transient["count"])
    if overall_count == 0 or overall["vector_rmse"] is None:
        return float("inf")
    numerator = overall_count * float(overall["vector_rmse"]) ** 2
    denominator = float(overall_count)
    if transient_count > 0 and transient["vector_rmse"] is not None:
        additional_weight = transient_weight - 1.0
        numerator += additional_weight * transient_count * float(transient["vector_rmse"]) ** 2
        denominator += additional_weight * transient_count
    return (numerator / denominator) ** 0.5


def _checkpoint_payload(
    *,
    model_state_dict: dict[str, torch.Tensor],
    cfg: ComVelocityEstimatorTrainCfg,
    teacher_checkpoint: Path,
    teacher_hash: str,
    git_commit: str,
    validation_metrics: dict[str, Any],
    training_step: int,
) -> dict[str, Any]:
    return {
        "model_state_dict": model_state_dict,
        "input_dim": cfg.input_dim,
        "per_frame_obs_dim": cfg.per_frame_obs_dim,
        "history_length": cfg.estimator_history_length,
        "hidden_dims": list(cfg.hidden_dims),
        "output_dim": cfg.output_dim,
        "output_frame": "heading",
        "output_quantity": "whole_body_com_velocity_xy",
        "output_unit": "m/s",
        "teacher_task": cfg.teacher_task,
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_hash": teacher_hash,
        "git_commit": git_commit,
        "training_step": int(training_step),
        "validation_metrics": validation_metrics,
    }


def _transient_and_td0_masks(
    target: torch.Tensor,
    previous_target: torch.Tensor,
    previous_valid: torch.Tensor,
    dones: torch.Tensor,
    touchdown: torch.Tensor,
    pending_td0: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    transient = (
        previous_valid
        & ~dones
        & (torch.linalg.vector_norm(target - previous_target, dim=1) > threshold)
    )
    pending_td0[dones] = False
    pending_td0 |= transient
    td0 = touchdown & pending_td0 & ~dones
    pending_td0[td0] = False
    return transient, td0


def _step_teacher_and_extract(
    env,
    teacher_policy,
    actor_observations: torch.Tensor,
    extractor: G1PrivilegedStateExtractor,
    cfg: ComVelocityEstimatorTrainCfg,
):
    with torch.inference_mode():
        actions = teacher_policy(actor_observations)
        next_observations, _, dones, _ = env.step(actions.to(env.device))
        state = extractor.extract()
        target = extract_com_velocity_target(state).detach()
        estimator_input = extract_recent_actor_history(
            next_observations,
            teacher_history_length=cfg.teacher_history_length,
            estimator_history_length=cfg.estimator_history_length,
            per_frame_obs_dim=cfg.per_frame_obs_dim,
        )
    return next_observations, dones.to(dtype=torch.bool), state, target, estimator_input


def _build_config() -> ComVelocityEstimatorTrainCfg:
    cfg = ComVelocityEstimatorTrainCfg(
        estimator_task=args.task,
        teacher_checkpoint=str(args.teacher_checkpoint.resolve()),
        num_envs=args.num_envs,
        train_envs=args.train_envs,
        validation_envs=args.validation_envs,
        train_policy_steps=args.train_policy_steps,
        evaluation_policy_steps=args.evaluation_policy_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    cfg.validate()
    return cfg


def train() -> Path:  # noqa: C901
    cfg = _build_config()
    teacher_checkpoint = Path(cfg.teacher_checkpoint).resolve()
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {teacher_checkpoint}")

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_root.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    env_cfg, teacher_agent_cfg = task_registry.get_cfgs(cfg.estimator_task)
    env_class = task_registry.get_task_class(cfg.estimator_task)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.scene.seed = cfg.seed
    env_cfg.device = args.device
    teacher_agent_cfg.device = args.device
    env = None
    try:
        env = env_class(env_cfg, headless=args.headless)
        teacher_runner = OnPolicyRunner(
            env,
            teacher_agent_cfg.to_dict(),
            log_dir=None,
            device=args.device,
        )
        teacher_runner.load(str(teacher_checkpoint), load_optimizer=False)
        teacher_runner.eval_mode()
        teacher_runner.alg.policy.requires_grad_(False)
        teacher_policy = teacher_runner.get_inference_policy(device=args.device)

        actor = teacher_runner.alg.policy.actor
        first_actor_layer = next(module for module in actor if isinstance(module, torch.nn.Linear))
        last_actor_layer = next(module for module in reversed(actor) if isinstance(module, torch.nn.Linear))
        current_actor_obs, _ = env.compute_current_observations()
        actor_observations, _ = env.get_observations()
        teacher_checks = {
            "strict_checkpoint_load": True,
            "optimizer_loaded": False,
            "optimizer_updated": False,
            "teacher_eval": not teacher_runner.alg.policy.training,
            "teacher_requires_grad": any(
                parameter.requires_grad for parameter in teacher_runner.alg.policy.parameters()
            ),
            "actor_input_dim": int(first_actor_layer.in_features),
            "actor_observation_dim": int(actor_observations.shape[1]),
            "per_frame_actor_observation_dim": int(current_actor_obs.shape[1]),
            "actor_history_length": int(env.cfg.robot.actor_obs_history_length),
            "action_dim": int(last_actor_layer.out_features),
        }
        expected_checks = {
            "actor_input_dim": cfg.teacher_history_length * cfg.per_frame_obs_dim,
            "actor_observation_dim": cfg.teacher_history_length * cfg.per_frame_obs_dim,
            "per_frame_actor_observation_dim": cfg.per_frame_obs_dim,
            "actor_history_length": cfg.teacher_history_length,
            "action_dim": 29,
        }
        for name, expected in expected_checks.items():
            if teacher_checks[name] != expected:
                raise RuntimeError(
                    f"teacher contract mismatch for {name}: {teacher_checks[name]} != {expected}"
                )
        if teacher_checks["teacher_requires_grad"] or not teacher_checks["teacher_eval"]:
            raise RuntimeError("teacher must be frozen in eval mode")

        estimator = ComVelocityEstimator(
            input_dim=cfg.input_dim,
            hidden_dims=cfg.hidden_dims,
            output_dim=cfg.output_dim,
        ).to(args.device)
        optimizer = torch.optim.Adam(estimator.parameters(), lr=cfg.learning_rate)
        extractor = G1PrivilegedStateExtractor(env)
        warmup = ResetWarmupMask(cfg.num_envs, cfg.reset_warmup_policy_steps, args.device)
        previous_target = torch.zeros(cfg.num_envs, 2, device=args.device)
        previous_valid = torch.zeros(cfg.num_envs, dtype=torch.bool, device=args.device)
        pending_td0 = torch.zeros(cfg.num_envs, dtype=torch.bool, device=args.device)

        train_loss_records: list[dict[str, float | int]] = []
        window_metrics = _metric_accumulators()
        best_score = float("inf")
        best_step = 0
        best_state_dict: dict[str, torch.Tensor] | None = None
        best_window_metrics: dict[str, Any] | None = None
        total_training_samples = 0
        total_training_transient_samples = 0

        print(
            f"[Estimator] train: envs={cfg.num_envs} train={cfg.train_envs} "
            f"validation={cfg.validation_envs} steps={cfg.train_policy_steps}"
        )
        for step in range(1, cfg.train_policy_steps + 1):
            actor_observations, dones, state, target, estimator_input = _step_teacher_and_extract(
                env, teacher_policy, actor_observations, extractor, cfg
            )
            eligible = warmup.eligible_after_step(dones)
            transient, td0 = _transient_and_td0_masks(
                target,
                previous_target,
                previous_valid,
                dones,
                state.touchdown,
                pending_td0,
                cfg.transient_delta_v_threshold,
            )

            train_eligible = eligible[: cfg.train_envs]
            estimator.train()
            if torch.any(train_eligible):
                train_input = estimator_input[: cfg.train_envs][train_eligible]
                train_target = target[: cfg.train_envs][train_eligible]
                train_transient = transient[: cfg.train_envs][train_eligible]
                prediction = estimator(train_input)
                loss = weighted_velocity_mse(
                    prediction,
                    train_target,
                    train_transient,
                    cfg.transient_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                sample_count = int(train_eligible.sum())
                transient_count = int(train_transient.sum())
                total_training_samples += sample_count
                total_training_transient_samples += transient_count
            else:
                loss = torch.zeros((), device=args.device)

            val_slice = slice(cfg.train_envs, cfg.num_envs)
            with torch.inference_mode():
                estimator.eval()
                validation_prediction = estimator(estimator_input[val_slice])
                _add_metrics(
                    window_metrics,
                    validation_prediction,
                    target[val_slice],
                    eligible[val_slice],
                    transient[val_slice],
                    state.touchdown[val_slice],
                    td0[val_slice],
                )

            previous_target.copy_(target)
            previous_valid.copy_(~dones)

            if step % cfg.validation_interval_steps == 0 or step == cfg.train_policy_steps:
                summary = _summarize(window_metrics)
                score = summary["overall"]["vector_rmse"]
                if score is not None and score < best_score:
                    best_score = float(score)
                    best_step = step
                    best_state_dict = {
                        name: value.detach().cpu().clone()
                        for name, value in estimator.state_dict().items()
                    }
                    best_window_metrics = summary
                window_metrics = _metric_accumulators()

            if step % cfg.log_interval_steps == 0 or step == cfg.train_policy_steps:
                record = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "training_samples": total_training_samples,
                    "training_transient_samples": total_training_transient_samples,
                    "best_validation_vector_rmse": best_score,
                }
                train_loss_records.append(record)
                print(
                    f"[Estimator] step={step}/{cfg.train_policy_steps} "
                    f"loss={record['loss']:.6f} best_val_vector_rmse={best_score:.6f}"
                )

        if best_state_dict is None or best_window_metrics is None:
            raise RuntimeError("training produced no eligible validation samples")
        last_state_dict = {
            name: value.detach().cpu().clone() for name, value in estimator.state_dict().items()
        }

        candidate_state_dicts = {
            "window_best": best_state_dict,
            "last": last_state_dict,
        }
        candidate_steps = {
            "window_best": best_step,
            "last": cfg.train_policy_steps,
        }
        candidate_models: dict[str, ComVelocityEstimator] = {}
        candidate_metrics_accumulators = {
            name: _metric_accumulators() for name in candidate_state_dicts
        }
        for name, state_dict in candidate_state_dicts.items():
            model = ComVelocityEstimator(
                input_dim=cfg.input_dim,
                hidden_dims=cfg.hidden_dims,
                output_dim=cfg.output_dim,
            ).to(args.device)
            model.load_state_dict(state_dict, strict=True)
            model.eval().requires_grad_(False)
            candidate_models[name] = model
        print(
            f"[Estimator] evaluating window-best step {best_step} and last step "
            f"{cfg.train_policy_steps} on one shared {cfg.evaluation_policy_steps}-step rollout"
        )
        for _ in range(cfg.evaluation_policy_steps):
            actor_observations, dones, state, target, estimator_input = _step_teacher_and_extract(
                env, teacher_policy, actor_observations, extractor, cfg
            )
            eligible = warmup.eligible_after_step(dones)
            transient, td0 = _transient_and_td0_masks(
                target,
                previous_target,
                previous_valid,
                dones,
                state.touchdown,
                pending_td0,
                cfg.transient_delta_v_threshold,
            )
            val_slice = slice(cfg.train_envs, cfg.num_envs)
            with torch.inference_mode():
                for name, model in candidate_models.items():
                    prediction = model(estimator_input[val_slice])
                    _add_metrics(
                        candidate_metrics_accumulators[name],
                        prediction,
                        target[val_slice],
                        eligible[val_slice],
                        transient[val_slice],
                        state.touchdown[val_slice],
                        td0[val_slice],
                    )
            previous_target.copy_(target)
            previous_valid.copy_(~dones)

        candidate_metrics = {
            name: _summarize(accumulators)
            for name, accumulators in candidate_metrics_accumulators.items()
        }
        candidate_scores = {
            name: _transient_weighted_selection_score(metrics, cfg.transient_weight)
            for name, metrics in candidate_metrics.items()
        }
        selected_name = min(candidate_scores, key=candidate_scores.get)
        selected_state_dict = candidate_state_dicts[selected_name]
        selected_step = candidate_steps[selected_name]
        final_metrics = candidate_metrics[selected_name]
        selection = {
            "method": "shared_rollout_transient_weighted_vector_rmse",
            "normal_weight": 1.0,
            "transient_weight": cfg.transient_weight,
            "transient_delta_v_threshold": cfg.transient_delta_v_threshold,
            "policy_steps": cfg.evaluation_policy_steps,
            "candidates": candidate_scores,
            "selected": selected_name,
        }
        teacher_hash = _sha256(teacher_checkpoint)
        git_commit = _git_commit()
        best_payload = _checkpoint_payload(
            model_state_dict=selected_state_dict,
            cfg=cfg,
            teacher_checkpoint=teacher_checkpoint,
            teacher_hash=teacher_hash,
            git_commit=git_commit,
            validation_metrics=final_metrics,
            training_step=selected_step,
        )
        best_payload["selection"] = selection
        last_payload = _checkpoint_payload(
            model_state_dict=last_state_dict,
            cfg=cfg,
            teacher_checkpoint=teacher_checkpoint,
            teacher_hash=teacher_hash,
            git_commit=git_commit,
            validation_metrics=candidate_metrics["last"],
            training_step=cfg.train_policy_steps,
        )
        best_path = run_dir / "com_velocity_estimator_best.pt"
        last_path = run_dir / "com_velocity_estimator_last.pt"
        torch.save(best_payload, best_path)
        torch.save(last_payload, last_path)

        reload_model = ComVelocityEstimator(
            input_dim=cfg.input_dim,
            hidden_dims=cfg.hidden_dims,
            output_dim=cfg.output_dim,
        ).to(args.device)
        reloaded = torch.load(best_path, map_location=args.device, weights_only=False)
        reload_model.load_state_dict(reloaded["model_state_dict"], strict=True)
        reload_model.eval()
        probe = estimator_input[val_slice][: min(16, cfg.validation_envs)]
        with torch.inference_mode():
            reload_max_abs_difference = float(
                torch.max(torch.abs(candidate_models[selected_name](probe) - reload_model(probe)))
            )
        if reload_max_abs_difference != 0.0:
            raise RuntimeError(
                "reloaded estimator inference differs from saved model: "
                f"max_abs={reload_max_abs_difference}"
            )

        serializable_cfg = asdict(cfg)
        serializable_cfg.update(
            {
                "teacher_checkpoint": str(teacher_checkpoint),
                "teacher_checkpoint_hash": teacher_hash,
                "git_commit": git_commit,
                "device": args.device,
                "headless": bool(args.headless),
                "actor_frame_layout": [
                    "root_ang_vel_b[3]",
                    "projected_gravity_b[3]",
                    "command[3]",
                    "joint_pos-default[29]",
                    "joint_vel-default[29]",
                    "previous_action[29]",
                ],
            }
        )
        (run_dir / "train_config.yaml").write_text(
            yaml.safe_dump(serializable_cfg, sort_keys=False), encoding="utf-8"
        )
        report = {
            "teacher_checks": teacher_checks,
            "history_order": "oldest_to_newest; estimator uses final 5 frames",
            "target": {
                "quantity": "mass-weighted whole-body CoM horizontal velocity",
                "frame": "yaw-only heading",
                "source": "G1PrivilegedStateExtractor.state.com_velocity[:, :2]",
            },
            "partitions": {
                "training_env_ids": [0, cfg.train_envs - 1],
                "validation_env_ids": [cfg.train_envs, cfg.num_envs - 1],
                "validation_used_for_gradient": False,
            },
            "training": {
                "policy_steps": cfg.train_policy_steps,
                "samples": total_training_samples,
                "transient_samples": total_training_transient_samples,
                "best_step": selected_step,
                "loss_log": train_loss_records,
            },
            "validation_metrics": final_metrics,
            "candidate_selection": selection,
            "candidate_metrics": candidate_metrics,
            "checkpoint_reload_max_abs_difference": reload_max_abs_difference,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return best_path
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        train()
    finally:
        simulation_app.close()
