#!/usr/bin/env python3
"""Gate 2: paired GT/Estimator Plane V1 certificate sensitivity."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER = PROJECT_DIR / "logs/g1_slope_sys_d.pt"
DEFAULT_ESTIMATOR = (
    PROJECT_DIR
    / "logs/g1_com_velocity_estimator/v2_iteration_long_5000_random_init_fixed"
    / "com_velocity_estimator_v2_long_best.pt"
)
DEFAULT_NOMINAL = (
    PROJECT_DIR
    / "tools/recovery/generated/g1_plane_nominal_params_g1_slope_sys_d_candidate.yaml"
)
DEFAULT_FLAT = PROJECT_DIR / "tools/recovery/generated/g1_recovery_params.yaml"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "tools/recovery/generated/g1_plane_v1_estimator_gate2_long_best.json"
)
SLOPES_DEGREES = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
DIRECTIONS = ("+x", "-x", "+y", "-y")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher_checkpoint", type=Path, default=DEFAULT_TEACHER)
parser.add_argument("--estimator_checkpoint", type=Path, default=DEFAULT_ESTIMATOR)
parser.add_argument("--flat_parameters", type=Path, default=DEFAULT_FLAT)
parser.add_argument("--nominal_parameters", type=Path, default=DEFAULT_NOMINAL)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--policy_steps", type=int, default=1000)
parser.add_argument("--speed_range", type=float, nargs=2, default=(0.2, 0.4))
parser.add_argument("--certificate_workers", type=int, default=16)
parser.add_argument("--seed", type=int, default=20260906)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.sensors import ImuCfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
import legged_lab.mdp as mdp  # noqa: E402
from legged_lab.estimation import (  # noqa: E402
    EstimatorFrameHistory,
    ResetWarmupMask,
    latest_actor_frame,
    load_com_velocity_estimator_for_inference,
)
from legged_lab.recovery.dwaq_estimator_diagnostic import (  # noqa: E402
    certificate_agreement,
    dcm_velocity_error_statistics,
)
from legged_lab.recovery.plane_certificate_runtime import (  # noqa: E402
    PlaneCalibratedG1CertificateEvaluator,
)
from legged_lab.recovery.dwaq_estimator_diagnostic import (  # noqa: E402
    query_with_replaced_com_velocity,
)
from legged_lab.terrains import make_plane_recovery_terrain_cfg  # noqa: E402
from legged_lab.utils import task_registry  # noqa: E402


OLD_V2_TD0 = {
    "velocity_vector_rmse_m_per_s": 0.2490,
    "velocity_p95_m_per_s": 0.5416,
    "N_exact_agreement": 0.6510,
    "N_within_one_agreement": 0.8698,
    "N_MAE": 0.5469,
    "margin_spearman": 0.735,
    "margin_sign_agreement": 0.9063,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_gate2(env_cfg) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.seed = args.seed
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.scene.terrain_type = "generator"
    env_cfg.scene.terrain_generator = make_plane_recovery_terrain_cfg(SLOPES_DEGREES)
    env_cfg.scene.max_init_terrain_level = 0
    env_cfg.scene.imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=ImuCfg.OffsetCfg(pos=(0.04525, 0.0, -0.08339)),
        update_period=0.0,
        gravity_bias=(0.0, 0.0, 9.81),
    )
    env_cfg.noise.add_noise = True
    env_cfg.robot.actor_obs_history_length = 10
    env_cfg.robot.critic_obs_history_length = 10
    env_cfg.recovery_context.enabled = False
    env_cfg.recovery_context.mode = "zero"
    env_cfg.stage2_reward.enabled = False
    env_cfg.stage2_reward.enable_shared_event_reward = False
    env_cfg.stage2_reward.enable_certificate_reward = False
    env_cfg.stage2_reward.enable_soft_reward_scaling = False
    env_cfg.stage2_reward.defer_certificate_reward_to_rollout_end = False
    env_cfg.push_curriculum.enable_push_curriculum = False
    env_cfg.push_curriculum.adaptive_upgrades_enabled = False
    env_cfg.push_curriculum.easy_sample_probability = 0.0
    env_cfg.plane_recovery.nominal_parameters_path = str(
        args.nominal_parameters.expanduser().resolve()
    )
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.debug_vis = False
    env_cfg.commands.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.ranges.lin_vel_x = (-0.4, 0.4)
    env_cfg.commands.ranges.lin_vel_y = (-0.4, 0.4)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = None
    reset_base = env_cfg.domain_rand.events.reset_base
    reset_base.params["pose_range"]["x"] = (0.0, 0.0)
    reset_base.params["pose_range"]["y"] = (0.0, 0.0)
    reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    reset_base.params["velocity_range"]["yaw"] = (0.0, 0.0)
    env_cfg.domain_rand.events.push_robot = EventTerm(
        func=mdp.fixed_full_range_push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={},
    )


def _balanced_commands(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    terrain_types = env.scene.terrain.terrain_types.to(dtype=torch.long)
    direction_indices = torch.empty_like(terrain_types)
    for slope_index in range(len(SLOPES_DEGREES)):
        env_ids = (terrain_types == slope_index).nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            raise RuntimeError(f"no environment for slope index {slope_index}")
        direction_indices[env_ids] = torch.arange(
            env_ids.numel(), dtype=torch.long, device=env.device
        ) % len(DIRECTIONS)
    speed_min, speed_max = (float(value) for value in args.speed_range)
    if not 0.0 < speed_min <= speed_max <= 0.4:
        raise ValueError("speed range must satisfy 0 < min <= max <= 0.4")
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed + 17)
    speed = speed_min + (speed_max - speed_min) * torch.rand(
        env.num_envs, generator=generator, device=env.device
    )
    commands = torch.zeros((env.num_envs, 3), device=env.device)
    commands[direction_indices == 0, 0] = speed[direction_indices == 0]
    commands[direction_indices == 1, 0] = -speed[direction_indices == 1]
    commands[direction_indices == 2, 1] = speed[direction_indices == 2]
    commands[direction_indices == 3, 1] = -speed[direction_indices == 3]
    return terrain_types, direction_indices, commands


def _pin_commands(env, commands: torch.Tensor) -> None:
    env.command_generator.command.copy_(commands)
    env.command_generator.is_standing_env[:] = False
    env.command_generator.is_heading_env[:] = False
    original_reset = env.command_generator.reset

    def reset_with_commands(env_ids):
        result = original_reset(env_ids)
        env.command_generator.command[env_ids] = commands[env_ids]
        env.command_generator.is_standing_env[env_ids] = False
        env.command_generator.is_heading_env[env_ids] = False
        return result

    env.command_generator.reset = reset_with_commands


def _distribution(rows: list[dict], key: str) -> dict[str, int]:
    counts = Counter(int(row[key]) for row in rows)
    return {str(value): int(counts.get(value, 0)) for value in range(7)}


def _velocity_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "vector_rmse": None, "p95_vector_error": None}
    error = np.asarray([row["velocity_error_xy"] for row in rows], dtype=np.float64)
    vector = np.linalg.norm(error, axis=1)
    return {
        "count": len(rows),
        "rmse_x": float(np.sqrt(np.mean(np.square(error[:, 0])))),
        "rmse_y": float(np.sqrt(np.mean(np.square(error[:, 1])))),
        "vector_rmse": float(np.sqrt(np.mean(np.sum(np.square(error), axis=1)))),
        "p95_vector_error": float(np.quantile(vector, 0.95)),
        "bias_x": float(np.mean(error[:, 0])),
        "bias_y": float(np.mean(error[:, 1])),
    }


def _touchdown_summary(rows: list[dict]) -> dict:
    agreement = certificate_agreement(rows, "EST")
    dcm = dcm_velocity_error_statistics(
        [row["estimated_velocity_xy"] for row in rows],
        [row["gt_velocity_xy"] for row in rows],
        [row["omega"] for row in rows],
    ) if rows else {"sample_count": 0}
    return {
        "paired_valid_count": int(agreement.get("sample_count", 0)),
        "certificate": agreement,
        "velocity": _velocity_summary(rows),
        "dcm_velocity_induced_error_cm": dcm,
        "N_GT_distribution": _distribution(rows, "N_GT") if rows else {},
        "N_EST_distribution": _distribution(rows, "N_EST") if rows else {},
    }


def _gate_status(td0: dict) -> tuple[str, list[str]]:
    agreement = td0["certificate"]
    failures = []
    checks = (
        ("N_within_one_agreement", 0.90, "minimum"),
        ("N_absolute_error_mean", 0.40, "maximum"),
        ("margin_spearman", 0.70, "minimum"),
        ("margin_sign_agreement", 0.90, "minimum"),
    )
    for key, limit, sense in checks:
        value = agreement.get(key)
        failed = value is None or (value < limit if sense == "minimum" else value > limit)
        if failed:
            failures.append(f"{key}={value} violates {sense} {limit}")
    if failures:
        return "FAIL", failures
    if agreement.get("N_exact_agreement", 0.0) < 0.70:
        return "CONDITIONAL PASS", ["N_exact_agreement is below preferred 0.70"]
    return "PASS", []


def evaluate() -> dict:  # noqa: C901,PLR0915
    if args.num_envs < 192 or args.policy_steps < 800:
        raise ValueError("Gate 2 requires at least 192 envs and 800 policy steps")
    teacher_path = args.teacher_checkpoint.expanduser().resolve()
    estimator_path = args.estimator_checkpoint.expanduser().resolve()
    nominal_path = args.nominal_parameters.expanduser().resolve()
    flat_path = args.flat_parameters.expanduser().resolve()
    for path in (teacher_path, estimator_path, nominal_path, flat_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    estimator, estimator_payload = load_com_velocity_estimator_for_inference(
        estimator_path, device=args.device
    )
    if float(estimator_payload["imu_acceleration_scale"]) != 0.05:
        raise RuntimeError("Gate 2 estimator IMU scale is not 0.05")
    env_cfg, agent_cfg = task_registry.get_cfgs("g1_plane_symmetric_stage2_baseline")
    _configure_gate2(env_cfg)
    env_cfg.device = args.device
    agent_cfg.device = args.device
    env = task_registry.get_task_class("g1_plane_symmetric_stage2_baseline")(
        env_cfg, headless=args.headless
    )
    terrain_types, direction_indices, commands = _balanced_commands(env)
    _pin_commands(env, commands)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(teacher_path), load_optimizer=False)
    runner.eval_mode()
    runner.alg.policy.requires_grad_(False)
    teacher = runner.get_inference_policy(device=args.device)
    observations, _ = env.get_observations()
    first = next(module for module in runner.alg.policy.actor if isinstance(module, torch.nn.Linear))
    last = next(
        module for module in reversed(runner.alg.policy.actor) if isinstance(module, torch.nn.Linear)
    )
    if (
        first.in_features != 960
        or observations.shape[1] != 960
        or last.out_features != 29
        or runner.alg.policy.training
        or any(parameter.requires_grad for parameter in runner.alg.policy.parameters())
    ):
        raise RuntimeError("frozen teacher contract failed in Gate 2")

    history = EstimatorFrameHistory(
        env.num_envs,
        history_length=5,
        actor_frame_dim=96,
        imu_dim=3,
        imu_acceleration_scale=0.05,
        device=args.device,
    )
    warmup = ResetWarmupMask(env.num_envs, 5, args.device)
    next_touchdown = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    push_delta = torch.zeros((env.num_envs, 2), device=env.device)
    paired_rows: list[dict] = []
    invalid_pair_count = 0
    certificate_batch_count = 0
    pushed_envs = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    evaluator = PlaneCalibratedG1CertificateEvaluator(
        flat_path,
        nominal_path,
        workers=args.certificate_workers,
        executor_type="subprocess",
        z_sole=-0.045,
        use_state_b=False,
    )
    try:
        for step in range(args.policy_steps):
            with torch.inference_mode():
                actions = teacher(observations)
                observations, _, dones, _ = env.step(actions)
            env.command_generator.command.copy_(commands)
            state = env._last_recovery_state
            reset = dones.to(dtype=torch.bool) | state.episode_reset
            estimator_input = history.append(
                latest_actor_frame(observations, history_length=10, per_frame_obs_dim=96),
                env.scene.sensors["imu"].data.lin_acc_b,
                reset,
            )
            eligible = warmup.eligible_after_step(reset)
            with torch.inference_mode():
                estimate = estimator(estimator_input)
            next_touchdown[reset] = -1
            push_mask = env._last_push_started_mask & ~reset
            next_touchdown[push_mask] = 0
            pushed_envs |= push_mask
            push_delta[push_mask] = env.last_push_delta_v_xy[push_mask]
            collect = (
                state.touchdown
                & eligible
                & state.terrain_plane_valid
                & (next_touchdown >= 0)
                & (next_touchdown <= 2)
                & ~reset
            )
            env_ids = collect.nonzero(as_tuple=False).flatten()
            if env_ids.numel():
                certificate_batch_count += 1
                trace_batch = certificate_batch_count <= 5 or certificate_batch_count % 20 == 0
                if trace_batch:
                    print(
                        f"[PlaneV1EstimatorGate2] certificate_batch={certificate_batch_count} "
                        f"size={env_ids.numel()} phase=submit_GT",
                        flush=True,
                    )
                gt_pending = evaluator.submit(state, env_ids)
                estimate_queries = []
                omega_values = []
                for local_index, (env_id, query) in enumerate(zip(env_ids.tolist(), gt_pending.queries)):
                    lookup = evaluator.lookup_nominal(query.command, query.alpha)
                    if not lookup.valid or lookup.value is None:
                        # Preserve the production evaluator's strict no-fallback
                        # semantics.  Both sides remain the same invalid query
                        # and are excluded from paired accuracy below.
                        omega_values.append(float("nan"))
                        estimate_queries.append(query)
                        continue
                    omega = float(lookup.value.omega)
                    omega_values.append(omega)
                    estimate_queries.append(
                        query_with_replaced_com_velocity(
                            query,
                            estimate[env_id].detach().cpu().numpy(),
                            state.com_velocity[env_id, :2].detach().cpu().numpy(),
                            omega,
                        )
                    )
                est_pending = evaluator.submit_queries(
                    tuple(estimate_queries), env.device, env_ids=tuple(env_ids.tolist())
                )
                if trace_batch:
                    print(
                        f"[PlaneV1EstimatorGate2] certificate_batch={certificate_batch_count} "
                        "phase=resolve_GT",
                        flush=True,
                    )
                n_gt, margin_gt, valid_gt = evaluator.resolve_with_validity(gt_pending)
                if trace_batch:
                    print(
                        f"[PlaneV1EstimatorGate2] certificate_batch={certificate_batch_count} "
                        "phase=resolve_EST",
                        flush=True,
                    )
                n_est, margin_est, valid_est = evaluator.resolve_with_validity(est_pending)
                if trace_batch:
                    print(
                        f"[PlaneV1EstimatorGate2] certificate_batch={certificate_batch_count} "
                        "phase=resolved",
                        flush=True,
                    )
                for local_index, env_id in enumerate(env_ids.tolist()):
                    if not bool(valid_gt[local_index] and valid_est[local_index]):
                        invalid_pair_count += 1
                        continue
                    td = int(next_touchdown[env_id].item())
                    gt_velocity = state.com_velocity[env_id, :2].detach().cpu().numpy()
                    est_velocity = estimate[env_id].detach().cpu().numpy()
                    paired_rows.append(
                        {
                            "touchdown": td,
                            "env_id": env_id,
                            "slope_degrees": SLOPES_DEGREES[int(terrain_types[env_id])],
                            "direction": DIRECTIONS[int(direction_indices[env_id])],
                            "speed": float(torch.linalg.vector_norm(commands[env_id, :2]).item()),
                            "push_delta_v_xy": push_delta[env_id].detach().cpu().tolist(),
                            "omega": omega_values[local_index],
                            "gt_velocity_xy": gt_velocity.tolist(),
                            "estimated_velocity_xy": est_velocity.tolist(),
                            "velocity_error_xy": (est_velocity - gt_velocity).tolist(),
                            "N_GT": int(n_gt[local_index].item()),
                            "margin_GT": float(margin_gt[local_index].item()),
                            "certificate_valid_GT": True,
                            "N_EST": int(n_est[local_index].item()),
                            "margin_EST": float(margin_est[local_index].item()),
                            "certificate_valid_EST": True,
                        }
                    )
                next_touchdown[env_ids] += 1
            if (step + 1) % 100 == 0 or step + 1 == args.policy_steps:
                counts = Counter(row["touchdown"] for row in paired_rows)
                print(
                    f"[PlaneV1EstimatorGate2] step={step + 1}/{args.policy_steps} "
                    f"pushed={int(pushed_envs.sum())} TD0={counts[0]} TD1={counts[1]} "
                    f"TD2={counts[2]} invalid_pairs={invalid_pair_count}",
                    flush=True,
                )
    finally:
        evaluator.close()

    by_touchdown = {
        f"TD{td}": _touchdown_summary(
            [row for row in paired_rows if row["touchdown"] == td]
        )
        for td in range(3)
    }
    status, failures = _gate_status(by_touchdown["TD0"])
    td0_long = by_touchdown["TD0"]
    comparison = {
        "old_V2": OLD_V2_TD0,
        "long_run_best": {
            "velocity_vector_rmse_m_per_s": td0_long["velocity"]["vector_rmse"],
            "velocity_p95_m_per_s": td0_long["velocity"]["p95_vector_error"],
            "N_exact_agreement": td0_long["certificate"].get("N_exact_agreement"),
            "N_within_one_agreement": td0_long["certificate"].get("N_within_one_agreement"),
            "N_MAE": td0_long["certificate"].get("N_absolute_error_mean"),
            "margin_spearman": td0_long["certificate"].get("margin_spearman"),
            "margin_sign_agreement": td0_long["certificate"].get("margin_sign_agreement"),
        },
    }
    combinations = Counter(
        (row["slope_degrees"], row["direction"])
        for row in paired_rows
        if row["touchdown"] == 0
    )
    return {
        "schema_version": 1,
        "gate": "Plane_V1_estimator_certificate_sensitivity_Gate_2",
        "status": status,
        "failures_or_conditions": failures,
        "sample_semantics": {
            "paired_state": "same physical touchdown state; only whole-body CoM velocity XY and derived b are replaced",
            "slopes_degrees": list(SLOPES_DEGREES),
            "directions": list(DIRECTIONS),
            "speed_range_m_per_s": list(args.speed_range),
            "push": "final fixed-full-range component-wise velocity jump at random event time/gait phase",
            "touchdowns": ["TD0", "TD1", "TD2"],
        },
        "num_envs": args.num_envs,
        "policy_steps": args.policy_steps,
        "seed": args.seed,
        "pushed_environment_count": int(pushed_envs.sum().item()),
        "invalid_pair_count": invalid_pair_count,
        "TD0_slope_direction_coverage": {
            f"slope={slope:+g},direction={direction}": combinations[(slope, direction)]
            for slope in SLOPES_DEGREES
            for direction in DIRECTIONS
        },
        "teacher_checkpoint": {"path": str(teacher_path), "sha256": _sha256(teacher_path)},
        "estimator_checkpoint": {
            "path": str(estimator_path),
            "sha256": _sha256(estimator_path),
            "best_iteration": int(estimator_payload.get("best_iteration", -1)),
        },
        "by_touchdown": by_touchdown,
        "old_V2_vs_long_run_best_TD0": comparison,
        "acceptance": {
            "N_within_one_min": 0.90,
            "N_MAE_max": 0.40,
            "margin_spearman_min": 0.70,
            "margin_sign_agreement_min": 0.90,
            "N_exact_preferred_min": 0.70,
        },
    }


def main() -> None:
    try:
        report = evaluate()
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        compact = {
            "status": report["status"],
            "failures_or_conditions": report["failures_or_conditions"],
            "by_touchdown": report["by_touchdown"],
        }
        print(f"[PlaneV1EstimatorGate2] report={output}", flush=True)
        print(json.dumps(compact, indent=2), flush=True)
        if report["status"] == "FAIL":
            raise SystemExit(2)
    except BaseException:
        # Isaac shutdown can block for minutes.  Emit the actionable failure
        # before entering cleanup so a Gate failure is never hidden as a hang.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
