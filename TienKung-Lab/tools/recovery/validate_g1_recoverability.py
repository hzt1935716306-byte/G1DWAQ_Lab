#!/usr/bin/env python3
"""Run nominal and disturbed recoverability Gates for the fixed Stage1B G1 policy."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS = PROJECT_DIR / "tools/recovery/generated/g1_recovery_params.yaml"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="g1_flat_symmetric_recovery")
parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
parser.add_argument("--checkpoint_path", type=Path, default=None)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--warmup_steps", type=int, default=250)
parser.add_argument("--nominal_touchdowns_per_speed", type=int, default=30)
parser.add_argument("--max_gate1_steps", type=int, default=12000)
parser.add_argument("--max_gate1_over_horizon_fraction", type=float, default=0.10)
parser.add_argument(
    "--reuse_gate1_report",
    type=Path,
    default=None,
    help="Reuse Gate 1 thresholds from a prior report for large disturbed-test batches.",
)
parser.add_argument("--validation_speed", type=float, default=0.6)
parser.add_argument("--push_levels", type=float, nargs="+", default=(0.5, 1.0, 1.25, 1.5))
parser.add_argument("--direction_count", type=int, default=4)
parser.add_argument("--push_phases", type=float, nargs="+", default=(0.0, 0.5))
parser.add_argument("--trials_per_condition", type=int, default=1)
parser.add_argument("--trial_timeout_s", type=float, default=8.0)
parser.add_argument(
    "--random_trials",
    type=int,
    default=0,
    help="Run this many randomized trials instead of the fixed push grid.",
)
parser.add_argument(
    "--in_range_fraction",
    type=float,
    default=0.90,
    help="Fraction of randomized trials inside the trained per-axis push range.",
)
parser.add_argument(
    "--trained_abs_delta_v_xy",
    type=float,
    default=1.0,
    help="Absolute per-axis velocity-jump limit used during training.",
)
parser.add_argument(
    "--outside_abs_delta_v_xy",
    type=float,
    default=1.5,
    help="Absolute per-axis sampling limit for out-of-training-range trials.",
)
parser.add_argument(
    "--random_command_speed_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Random forward command range; defaults to the calibrated command range.",
)
parser.add_argument(
    "--certificate_workers",
    type=int,
    default=min(16, os.cpu_count() or 1),
    help="CPU workers used to solve saved touchdown certificate queries after simulation.",
)
parser.add_argument(
    "--recovery_manager_validation",
    action="store_true",
    help=(
        "Keep each disturbed rollout through five new touchdowns (or a real fall), "
        "then replay its event/touchdown certificates through RecoveryManager."
    ),
)
parser.add_argument(
    "--q_memory_diagnostic",
    action="store_true",
    help="Continue diagnostic logging through TD8 without changing the TD5 state-machine timeout.",
)
parser.add_argument(
    "--q_memory_sample_count",
    type=int,
    default=30,
    help="Maximum TD5 N=1 TIMEOUT trajectories and nominal touchdowns used by the diagnostic.",
)
parser.add_argument(
    "--passive_log_touchdowns",
    type=int,
    default=5,
    help="Keep passive labels after the unchanged TD5 state-machine exit; allowed range is 5--8.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_DIR / "tools/recovery/generated/g1_recoverability_report.yaml",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.utils.math import quat_apply  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.recovery.certificate import (  # noqa: E402
    CertificateState,
    HalfspaceRegion2D,
    RecoverabilityConfig,
    certify_recoverability,
)
from legged_lab.recovery.recovery_manager import (  # noqa: E402
    RecoveryExitReason,
    RecoveryManager,
)
from legged_lab.recovery.state_extractor import (  # noqa: E402
    G1PrivilegedStateExtractor,
    G1StateExtractorCfg,
    theoretical_periodic_state,
)
from legged_lab.utils import task_registry  # noqa: E402


def _native(value):
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _load_yaml(path: Path) -> dict:
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _disable_evaluation_randomization(env_cfg):
    events = env_cfg.domain_rand.events
    events.push_robot = None
    events.physics_material = None
    events.add_base_mass = None
    for name in ("randomize_dome_light", "randomize_distant_light"):
        if hasattr(events, name):
            setattr(events, name, None)
    for key in events.reset_base.params["pose_range"]:
        events.reset_base.params["pose_range"][key] = (0.0, 0.0)
    for key in events.reset_base.params["velocity_range"]:
        events.reset_base.params["velocity_range"][key] = (0.0, 0.0)
    events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def _make_env_policy(parameters: dict):
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_cfg.scene.num_envs = max(args.num_envs, len(parameters["provenance"]["commands_vx"]))
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.scene.terrain_type = "plane"
    env_cfg.scene.terrain_generator = None
    env_cfg.noise.add_noise = False
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.debug_vis = False
    env_cfg.commands.resampling_time_range = (1.0e9, 1.0e9)
    command_speeds = parameters["provenance"]["commands_vx"]
    env_cfg.commands.ranges.lin_vel_x = (min(command_speeds), max(command_speeds))
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = None
    _disable_evaluation_randomization(env_cfg)

    env = task_registry.get_task_class(args.task)(env_cfg, args.headless)
    checkpoint = (
        args.checkpoint_path.expanduser().resolve()
        if args.checkpoint_path is not None
        else Path(parameters["provenance"]["checkpoint"]).expanduser().resolve()
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    return env, runner.get_inference_policy(device=env.device), checkpoint


def _set_commands(env, speeds):
    command = env.command_generator.command
    command[:, 0] = torch.as_tensor(speeds, device=env.device)
    command[:, 1:] = 0.0
    env.command_generator.is_standing_env[:] = False


def _period_at(parameters: dict, speed: float) -> float:
    model = parameters["step_period"]
    if model["type"] == "linear":
        return max(0.10, float(model["intercept"] + model["slope"] * speed))
    return float(model["value"])


def _bounds(region: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    return tuple(region["x"]), tuple(region["y"])


def _certificate_config(parameters: dict, speed: float) -> RecoverabilityConfig:
    period = _period_at(parameters, speed)
    omega = float(parameters["omega"])
    theory = theoretical_periodic_state(speed, 0.0, period, omega, float(parameters["w"]))
    c_left_x, c_left_y = _bounds(parameters["C_left"])
    c_right_x, c_right_y = _bounds(parameters["C_right"])
    l_left_x, l_left_y = _bounds(parameters["L_left"])
    l_right_x, l_right_y = _bounds(parameters["L_right"])
    return RecoverabilityConfig(
        gravity=9.81,
        h_eff=float(parameters["h_eff"]),
        step_period=period,
        max_steps=5,
        cop_left=HalfspaceRegion2D.box(c_left_x, c_left_y),
        cop_right=HalfspaceRegion2D.box(c_right_x, c_right_y),
        landing_left=HalfspaceRegion2D.box(l_left_x, l_left_y),
        landing_right=HalfspaceRegion2D.box(l_right_x, l_right_y),
        swing_velocity_limits=(float(parameters["v_max"]["x"]), float(parameters["v_max"]["y"])),
        nominal_cop_left=(0.0, 0.0),
        nominal_cop_right=(0.0, 0.0),
        nominal_step_left=theory["landing_left"],
        nominal_step_right=theory["landing_right"],
        nominal_b_left=theory["b_left"],
        nominal_b_right=theory["b_right"],
        nominal_q_left=theory["q_left"],
        nominal_q_right=theory["q_right"],
        epsilon_b=(float(parameters["epsilon_b"]["x"]), float(parameters["epsilon_b"]["y"])),
        epsilon_q=(float(parameters["epsilon_q"]["x"]), float(parameters["epsilon_q"]["y"])),
    )


def _support_position(state, env_id: int) -> torch.Tensor:
    return (
        state.left_foot_position[env_id]
        if bool(state.support_is_left[env_id].item())
        else state.right_foot_position[env_id]
    )


def _certificate_query(state, env_id: int, parameters: dict, delta_v_heading: np.ndarray | None = None):
    speed = float(state.command_velocity[env_id, 0].item())
    omega = float(parameters["omega"])
    support = _support_position(state, env_id)
    b = (
        state.com_position[env_id, :2]
        + state.com_velocity[env_id, :2] / omega
        - support[:2]
    ).detach().cpu().numpy()
    if delta_v_heading is not None:
        b = b + np.asarray(delta_v_heading) / omega
    q = state.q[env_id].detach().cpu().numpy()
    side = "left" if bool(state.support_is_left[env_id].item()) else "right"
    return {
        "command_vx": speed,
        "b": b,
        "q": q,
        "support_side": side,
        "phase": float(state.phase[env_id].item()),
    }


def _solve_certificate_query(query: dict, parameters: dict):
    config = _certificate_config(parameters, float(query["command_vx"]))
    return certify_recoverability(
        CertificateState(
            b=np.asarray(query["b"]),
            q=np.asarray(query["q"]),
            support_side=query["support_side"],
            phase=float(query["phase"]),
            step_period=config.step_period,
            omega=config.omega,
        ),
        config,
    )


def _annotate_q_memory_sample(sample: dict, query: dict, parameters: dict) -> None:
    """Attach model errors without changing the certificate or its inputs."""
    config = _certificate_config(parameters, float(query["command_vx"]))
    nominal_b, nominal_q = config.terminal_nominal(query["support_side"])
    b_real = np.asarray(query["b"], dtype=np.float64)
    q_real = np.asarray(query["q"], dtype=np.float64)
    e_b = b_real - nominal_b
    e_q = q_real - nominal_q
    # At touchdown q_k is the old support foot relative to the new support,
    # hence q_k = -l_{k-1} in the certificate sign convention.
    previous_landing_actual = -q_real
    previous_landing_nominal = -nominal_q
    previous_landing_error = previous_landing_actual - previous_landing_nominal
    sample.update(
        {
            "e_b_xy": e_b,
            "e_q_xy": e_q,
            "previous_landing_actual_xy": previous_landing_actual,
            "previous_landing_nominal_xy": previous_landing_nominal,
            "previous_landing_error_xy": previous_landing_error,
            "q_plus_previous_landing_error_xy": e_q + previous_landing_error,
        }
    )


def _certify(state, env_id: int, parameters: dict, delta_v_heading: np.ndarray | None = None):
    return _solve_certificate_query(
        _certificate_query(state, env_id, parameters, delta_v_heading),
        parameters,
    )


def _per_step_metric(state, env_id: int) -> tuple[float, np.ndarray]:
    velocity_error = torch.linalg.vector_norm(
        state.com_velocity[env_id, :2] - state.command_velocity[env_id, :2]
    )
    return float(velocity_error.item()), np.abs(state.root_roll_pitch[env_id].detach().cpu().numpy())


def _touchdown_diagnostic_snapshot(state, env_id: int) -> dict:
    velocity_error = (
        state.com_velocity[env_id, :2] - state.command_velocity[env_id, :2]
    ).detach().cpu().numpy()
    return {
        "velocity_tracking_error_xy": velocity_error,
        "root_roll_pitch": state.root_roll_pitch[env_id].detach().cpu().numpy(),
        "good_cycle": None,
        "practical_entered": False,
        "practical_confirmed": False,
        "practical_enter_step": None,
        "practical_confirmed_step": None,
    }


def _gate1(env, policy, extractor, parameters):
    commands = [float(value) for value in parameters["provenance"]["commands_vx"]]
    assigned = np.asarray([commands[index % len(commands)] for index in range(env.num_envs)])
    targets = {speed: args.nominal_touchdowns_per_speed for speed in commands}
    counts = {speed: 0 for speed in commands}
    n_counts = {label: 0 for label in (0, 1, 2, 3, 4, 5, 6)}
    margins = []
    solver_failures = 0
    cycle_velocity_error = []
    cycle_abs_roll_pitch = []
    last_touchdown_side: list[int | None] = [None] * env.num_envs
    interval_velocity: list[list[float]] = [[] for _ in range(env.num_envs)]
    interval_tilt: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]

    _set_commands(env, assigned)
    obs, _ = env.get_observations()
    for step in range(args.max_gate1_steps):
        _set_commands(env, assigned)
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            state = extractor.extract()

        for env_id in range(env.num_envs):
            velocity_error, tilt = _per_step_metric(state, env_id)
            interval_velocity[env_id].append(velocity_error)
            interval_tilt[env_id].append(tilt)

        for env_id in (dones | state.episode_reset).nonzero(as_tuple=False).flatten().tolist():
            last_touchdown_side[env_id] = None
            interval_velocity[env_id].clear()
            interval_tilt[env_id].clear()

        for env_id in state.touchdown.nonzero(as_tuple=False).flatten().tolist():
            side = int(state.touchdown_foot[env_id].item())
            alternating = last_touchdown_side[env_id] is not None and side != last_touchdown_side[env_id]
            if step >= args.warmup_steps and alternating and interval_velocity[env_id]:
                cycle_velocity_error.append(float(np.mean(interval_velocity[env_id])))
                cycle_abs_roll_pitch.append(np.mean(np.asarray(interval_tilt[env_id]), axis=0))
            interval_velocity[env_id].clear()
            interval_tilt[env_id].clear()
            last_touchdown_side[env_id] = side

            speed = float(assigned[env_id])
            if step >= args.warmup_steps and counts[speed] < targets[speed]:
                result = _certify(state, env_id, parameters)
                if result.n_min is None or result.margin is None:
                    solver_failures += 1
                else:
                    n_counts[int(result.n_min)] += 1
                    margins.append(float(result.margin))
                counts[speed] += 1
        if step > 0 and step % 100 == 0:
            print(f"[INFO] Gate 1 step={step}, touchdown_counts={counts}", flush=True)
        if all(counts[speed] >= targets[speed] for speed in commands):
            break
    else:
        print(f"[WARN] Gate 1 reached {args.max_gate1_steps} steps: counts={counts}")

    if not cycle_velocity_error or not cycle_abs_roll_pitch:
        raise RuntimeError("No complete alternating nominal cycles were collected for recovery thresholds")
    cycle_velocity_error = np.asarray(cycle_velocity_error)
    cycle_abs_roll_pitch = np.asarray(cycle_abs_roll_pitch)
    thresholds = {
        "mean_velocity_error": float(np.quantile(cycle_velocity_error, 0.95) * 1.25 + 0.05),
        "mean_abs_roll": float(np.quantile(cycle_abs_roll_pitch[:, 0], 0.95) * 1.25 + 0.01),
        "mean_abs_pitch": float(np.quantile(cycle_abs_roll_pitch[:, 1], 0.95) * 1.25 + 0.01),
        "derivation": "1.25 * nominal complete-cycle p95 + small numerical allowance",
        "baseline_cycle_count": int(cycle_velocity_error.size),
    }
    total = sum(n_counts.values())
    fractions = {str(key) if key < 6 else ">5": value / max(total, 1) for key, value in n_counts.items()}
    over_fraction = fractions[">5"]
    passed = total > 0 and solver_failures == 0 and over_fraction <= args.max_gate1_over_horizon_fraction
    report = {
        "passed": passed,
        "criterion": f"P(N>5) <= {args.max_gate1_over_horizon_fraction} and no solver failures",
        "touchdowns_by_command": {str(key): value for key, value in counts.items()},
        "N_min_count": {str(key) if key < 6 else ">5": value for key, value in n_counts.items()},
        "N_min_probability": fractions,
        "margin": {
            "count": len(margins),
            "mean": float(np.mean(margins)) if margins else None,
            "median": float(np.median(margins)) if margins else None,
            "p05": float(np.quantile(margins, 0.05)) if margins else None,
            "p95": float(np.quantile(margins, 0.95)) if margins else None,
            "min": float(np.min(margins)) if margins else None,
            "max": float(np.max(margins)) if margins else None,
        },
        "solver_failures": solver_failures,
        "actual_recovery_thresholds": thresholds,
        "baseline_cycle_statistics": {
            "mean_velocity_error_p50": float(np.median(cycle_velocity_error)),
            "mean_velocity_error_p95": float(np.quantile(cycle_velocity_error, 0.95)),
            "mean_abs_roll_p95": float(np.quantile(cycle_abs_roll_pitch[:, 0], 0.95)),
            "mean_abs_pitch_p95": float(np.quantile(cycle_abs_roll_pitch[:, 1], 0.95)),
        },
    }
    return report, thresholds, obs


def _random_trial_plans(parameters: dict) -> list[dict]:
    if args.random_trials <= 0:
        return []
    if not 0.0 <= args.in_range_fraction <= 1.0:
        raise ValueError("in_range_fraction must be in [0, 1]")
    if args.trained_abs_delta_v_xy <= 0.0:
        raise ValueError("trained_abs_delta_v_xy must be positive")
    if args.outside_abs_delta_v_xy <= args.trained_abs_delta_v_xy:
        raise ValueError("outside_abs_delta_v_xy must exceed trained_abs_delta_v_xy")

    calibrated_speeds = [float(value) for value in parameters["provenance"]["commands_vx"]]
    speed_min, speed_max = (
        tuple(args.random_command_speed_range)
        if args.random_command_speed_range is not None
        else (min(calibrated_speeds), max(calibrated_speeds))
    )
    if not 0.0 < speed_min <= speed_max:
        raise ValueError("random command speeds must satisfy 0 < MIN <= MAX")

    rng = np.random.default_rng(args.seed)
    in_range_count = int(round(args.random_trials * args.in_range_fraction))
    scopes = np.asarray(
        ["in_training_range"] * in_range_count
        + ["outside_training_range"] * (args.random_trials - in_range_count),
        dtype=object,
    )
    rng.shuffle(scopes)
    plans = []
    for trial_index, scope in enumerate(scopes.tolist()):
        if scope == "in_training_range":
            delta = rng.uniform(
                -args.trained_abs_delta_v_xy,
                args.trained_abs_delta_v_xy,
                size=2,
            )
        else:
            # Sample from the larger square conditioned on at least one axis
            # exceeding the trained component-wise limit.
            while True:
                delta = rng.uniform(
                    -args.outside_abs_delta_v_xy,
                    args.outside_abs_delta_v_xy,
                    size=2,
                )
                if np.any(np.abs(delta) > args.trained_abs_delta_v_xy):
                    break
        level = math.hypot(float(delta[0]), float(delta[1]))
        plans.append(
            {
                "trial_index": trial_index,
                "push_scope": scope,
                "level": level,
                "direction_index": None,
                "angle_rad": float(math.atan2(delta[1], delta[0])),
                "delta_v_heading_xy": delta,
                "target_phase": float(rng.uniform(0.0, 1.0)),
                "command_vx": float(rng.uniform(speed_min, speed_max)),
                "repeat": 0,
            }
        )
    return plans


def _trial_plans(parameters: dict) -> list[dict]:
    if args.random_trials > 0:
        return _random_trial_plans(parameters)
    plans = []
    for level in args.push_levels:
        for direction_index in range(args.direction_count):
            angle = 2.0 * math.pi * direction_index / args.direction_count
            for phase in args.push_phases:
                if not 0.0 <= phase < 1.0:
                    raise ValueError(f"push phase must be in [0, 1): {phase}")
                for repeat in range(args.trials_per_condition):
                    plans.append(
                        {
                            "level": float(level),
                            "direction_index": direction_index,
                            "angle_rad": angle,
                            "target_phase": float(phase),
                            "repeat": repeat,
                        }
                    )
    rng = np.random.default_rng(args.seed)
    rng.shuffle(plans)
    return plans


def _spearman_summary(first: np.ndarray, second: np.ndarray) -> dict:
    if np.unique(first).size < 2 or np.unique(second).size < 2:
        return {
            "rho": None,
            "p_value": None,
            "reason": "undefined because at least one variable is constant",
        }
    rho, p_value = spearmanr(first, second)
    return {"rho": float(rho), "p_value": float(p_value)}


def _start_trial(env, state, env_id: int, plan: dict, parameters: dict, step: int) -> dict:
    delta_heading = np.asarray(
        plan.get(
            "delta_v_heading_xy",
            (plan["level"] * math.cos(plan["angle_rad"]), plan["level"] * math.sin(plan["angle_rad"])),
        ),
        dtype=np.float64,
    )
    initial_query = _certificate_query(state, env_id, parameters, delta_heading)
    delta_heading_tensor = torch.tensor(
        [[delta_heading[0], delta_heading[1], 0.0]], dtype=torch.float32, device=env.device
    )
    delta_world = quat_apply(state.heading_quat_w[env_id].unsqueeze(0), delta_heading_tensor)[0]
    ids = torch.tensor([env_id], dtype=torch.long, device=env.device)
    root_velocity = env.robot.data.root_vel_w[ids].clone()
    root_velocity[:, :2] += delta_world[:2]
    env.robot.write_root_velocity_to_sim(root_velocity, env_ids=ids)
    starts_at_touchdown = bool(plan["target_phase"] == 0.0 and state.touchdown[env_id].item())
    support_foot = 0 if bool(state.support_is_left[env_id].item()) else 1
    start_time = float(state.time[env_id].item())
    return {
        **plan,
        "env_id": env_id,
        "start_step": step,
        "start_time": start_time,
        "applied_phase": float(state.phase[env_id].item()),
        "delta_v_heading_xy": delta_heading,
        "delta_v_world_xy": delta_world[:2].detach().cpu().numpy(),
        "N_theory": None,
        "margin": None,
        "certificate_status": "pending",
        "success": False,
        "practical_entered": False,
        "practical_enter_step": None,
        "N_actual": None,
        "N_confirmation": None,
        "t_recovery": None,
        "t_confirmation": None,
        "failure_reason": None,
        "touchdowns": 0,
        "last_touchdown_foot": support_foot if starts_at_touchdown else None,
        "interval_started_after_touchdown": starts_at_touchdown,
        "interval_start_touchdown": 0 if starts_at_touchdown else None,
        "interval_start_time": start_time if starts_at_touchdown else None,
        "interval_velocity": [],
        "interval_tilt": [],
        "consecutive_good_cycles": 0,
        "recovery_onset_touchdown": None,
        "recovery_onset_time": None,
        "cycle_metrics": [],
        "certificate_trace": [
            {
                "touchdown": 0,
                "time_after_push": 0.0,
                "N_theory": None,
                "margin": None,
                "good_cycle": None,
                "practical_entered": False,
                "practical_confirmed": False,
                "practical_enter_step": None,
                "practical_confirmed_step": None,
                "status": "pending",
                "_query": initial_query,
            }
        ],
    }


def _finalize_trial(trial: dict) -> dict:
    internal_fields = {
        "interval_velocity",
        "interval_tilt",
        "consecutive_good_cycles",
        "interval_started_after_touchdown",
        "interval_start_touchdown",
        "interval_start_time",
        "recovery_onset_touchdown",
        "recovery_onset_time",
    }
    finalized = {
        key: value
        for key, value in trial.items()
        if key not in internal_fields
    }
    if finalized["success"]:
        recovery_touchdown = int(finalized["N_actual"])
        finalized["certificate_trace"] = [
            {
                **sample,
                "N_actual_remaining": max(recovery_touchdown - int(sample["touchdown"]), 0),
            }
            for sample in finalized["certificate_trace"]
        ]
    return finalized


def _resolve_saved_certificates(trials: list[dict], parameters: dict) -> None:
    jobs = []
    for trial_index, trial in enumerate(trials):
        if args.recovery_manager_validation:
            trace_limit = 8 if args.q_memory_diagnostic else args.passive_log_touchdowns
            trial["certificate_trace"] = [
                sample for sample in trial["certificate_trace"] if sample["touchdown"] <= trace_limit
            ]
        elif trial["success"]:
            trial["certificate_trace"] = [
                sample
                for sample in trial["certificate_trace"]
                if sample["touchdown"] <= trial["N_actual"]
            ]
        else:
            trial["certificate_trace"] = trial["certificate_trace"][:1]
        for sample_index, sample in enumerate(trial["certificate_trace"]):
            jobs.append((trial_index, sample_index, sample["_query"]))

    print(
        f"[INFO] Solving {len(jobs)} saved certificate states with "
        f"{args.certificate_workers} CPU workers",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.certificate_workers) as executor:
        results = list(
            executor.map(
                lambda job: _solve_certificate_query(job[2], parameters),
                jobs,
            )
        )
    for (trial_index, sample_index, _), result in zip(jobs, results):
        sample = trials[trial_index]["certificate_trace"][sample_index]
        query = sample.pop("_query")
        sample.setdefault("support_side", query["support_side"])
        sample["N_theory"] = result.n_min
        sample["margin"] = result.margin
        sample["status"] = result.status.value
        sample["orbit_recovered"] = result.n_min == 0
        if args.q_memory_diagnostic:
            _annotate_q_memory_sample(sample, query, parameters)

    for trial in trials:
        initial = trial["certificate_trace"][0]
        trial["N_theory"] = initial["N_theory"]
        trial["margin"] = initial["margin"]
        trial["certificate_status"] = initial["status"]


def _resolve_nominal_diagnostic_samples(samples: list[dict], parameters: dict) -> None:
    for sample in samples:
        query = sample.pop("_query")
        result = _solve_certificate_query(query, parameters)
        sample["N_theory"] = result.n_min
        sample["margin"] = result.margin
        sample["status"] = result.status.value
        sample["orbit_recovered"] = result.n_min == 0
        _annotate_q_memory_sample(sample, query, parameters)


def _q_memory_diagnostic_summary(trials: list[dict], nominal_samples: list[dict]) -> dict:
    candidates = []
    for trial in sorted(trials, key=lambda item: int(item.get("trial_index", 0))):
        episode = trial.get("recovery_state_machine")
        if episode is None or episode["exit_reason"] != RecoveryExitReason.TIMEOUT.value:
            continue
        by_touchdown = {int(sample["touchdown"]): sample for sample in trial["certificate_trace"]}
        if by_touchdown.get(5, {}).get("N_theory") != 1:
            continue
        if not all(touchdown in by_touchdown for touchdown in range(5, 9)):
            continue
        candidates.append((trial, by_touchdown))
    selected = candidates[: args.q_memory_sample_count]
    nominal = nominal_samples[: args.q_memory_sample_count]
    if not selected:
        raise RuntimeError("q-memory diagnostic found no complete TD5 N=1 TIMEOUT trajectories")
    if not nominal:
        raise RuntimeError("q-memory diagnostic collected no nominal touchdown samples")

    def percentile_summary(values) -> dict:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "median": float(np.median(array)),
            "p95": float(np.quantile(array, 0.95)),
        }

    touchdown_summary = {}
    relation_residuals = []
    for touchdown in range(5, 9):
        samples = [by_touchdown[touchdown] for _, by_touchdown in selected]
        for sample in samples:
            relation_residuals.extend(np.abs(sample["q_plus_previous_landing_error_xy"]))
        velocity = np.abs(np.asarray([sample["velocity_tracking_error_xy"] for sample in samples]))
        tilt = np.abs(np.asarray([sample["root_roll_pitch"] for sample in samples]))
        valid_good = [sample["good_cycle"] for sample in samples if sample["good_cycle"] is not None]
        touchdown_summary[f"TD{touchdown}"] = {
            "abs_e_q_x": percentile_summary(
                [abs(float(sample["e_q_xy"][0])) for sample in samples]
            ),
            "P_N_equals_0": float(np.mean([sample["N_theory"] == 0 for sample in samples])),
            "N_distribution": {
                (str(value) if value < 6 else ">5"): sum(sample["N_theory"] == value for sample in samples)
                for value in sorted(set(int(sample["N_theory"]) for sample in samples))
            },
            "abs_velocity_tracking_error": {
                "x_median": float(np.median(velocity[:, 0])),
                "x_p95": float(np.quantile(velocity[:, 0], 0.95)),
                "y_median": float(np.median(velocity[:, 1])),
                "y_p95": float(np.quantile(velocity[:, 1], 0.95)),
            },
            "abs_root_tilt": {
                "roll_median": float(np.median(tilt[:, 0])),
                "roll_p95": float(np.quantile(tilt[:, 0], 0.95)),
                "pitch_median": float(np.median(tilt[:, 1])),
                "pitch_p95": float(np.quantile(tilt[:, 1], 0.95)),
            },
            "good_cycle_fraction": float(np.mean(valid_good)) if valid_good else None,
            "practical_entered_fraction": float(
                np.mean([sample["practical_entered"] for sample in samples])
            ),
            "practical_confirmed_fraction": float(
                np.mean([sample["practical_confirmed"] for sample in samples])
            ),
        }

    nominal_abs_e_q_x = [abs(float(sample["e_q_xy"][0])) for sample in nominal]
    enter_steps = [
        trial["practical_enter_step"]
        for trial, _ in selected
        if trial["practical_enter_step"] is not None
    ]
    confirmed_steps = [
        trial["N_confirmation"] for trial, _ in selected if trial["N_confirmation"] is not None
    ]
    trajectories = []
    for trial, by_touchdown in selected:
        trajectories.append(
            {
                "trial_index": trial.get("trial_index"),
                "push_scope": trial.get("push_scope"),
                "delta_v_heading_xy": trial["delta_v_heading_xy"],
                "applied_phase": trial["applied_phase"],
                "practical_enter_step": trial["practical_enter_step"],
                "practical_confirmed_step": trial["N_confirmation"],
                "samples": [by_touchdown[touchdown] for touchdown in range(5, 9)],
            }
        )
    return {
        "definition": (
            "Offline logging continuation after the unchanged formal TD5 TIMEOUT; "
            "RecoveryManager is not reactivated."
        ),
        "eligible_complete_TD5_N1_timeout_count": len(candidates),
        "selected_trial_count": len(selected),
        "nominal_sample_count": len(nominal),
        "nominal_abs_e_q_x": percentile_summary(nominal_abs_e_q_x),
        "touchdowns": touchdown_summary,
        "q_equals_negative_previous_landing_error_check": {
            "definition": "e_q_k + e_l_(k-1) should be zero",
            "median_abs_residual": float(np.median(relation_residuals)),
            "max_abs_residual": float(np.max(relation_residuals)),
        },
        "practical_enter_step_distribution": {
            str(value): enter_steps.count(value) for value in sorted(set(enter_steps))
        },
        "practical_confirmed_step_distribution": {
            str(value): confirmed_steps.count(value) for value in sorted(set(confirmed_steps))
        },
        "not_practical_confirmed_by_TD8_count": sum(
            trial["N_confirmation"] is None for trial, _ in selected
        ),
        "trajectories": trajectories,
    }


def _attach_recovery_state_machine(trials: list[dict]) -> dict:
    """Replay saved event/touchdown certificates through the minimal manager."""
    exit_counts = {reason.value: 0 for reason in RecoveryExitReason}
    confirmed_exit_counts = {reason.value: 0 for reason in RecoveryExitReason}
    orbit_exit_counts = {reason.value: 0 for reason in RecoveryExitReason}
    incomplete_count = 0
    alternating_checks = 0
    alternation_violations = 0
    duplicate_touchdowns = 0
    all_rewards_zero = True
    reset_failures = 0
    old_confirmed_timeout_new_entered_success = 0
    practical_success_without_orbit = 0
    practical_enter_step_distribution = {str(step): 0 for step in range(1, 6)}
    success_n_distribution = {"0": 0, "1": 0, ">=2": 0}
    timeout_n_distribution = {str(value): 0 for value in range(6)} | {">5": 0}

    for trial in trials:
        trace = trial["certificate_trace"]
        if not trace or trace[0]["N_theory"] is None or trace[0]["margin"] is None:
            trial["recovery_state_machine"] = None
            incomplete_count += 1
            continue

        manager = RecoveryManager(max_touchdowns=5)
        initial = trace[0]
        manager.on_push(
            n=int(initial["N_theory"]),
            margin=float(initial["margin"]),
            delta_v_xy=tuple(float(value) for value in trial["delta_v_heading_xy"]),
            phase_at_push=float(trial["applied_phase"]),
            policy_step=int(trial["start_step"]),
        )
        completed_episode = None
        confirmed_exit_reason = None
        orbit_exit_reason = None
        for sample in trace[1:]:
            if sample["N_theory"] is None or sample["margin"] is None:
                continue
            update = manager.on_touchdown(
                n=int(sample["N_theory"]),
                margin=float(sample["margin"]),
                practical_entered=bool(sample.get("practical_entered", False)),
                practical_confirmed=bool(sample.get("practical_confirmed", False)),
                touchdown_token=int(sample["touchdown"]),
                support_side=sample.get("support_side"),
                time_after_push=float(sample["time_after_push"]),
            )
            duplicate_touchdowns += int(update.duplicate_touchdown)
            if update.transition is not None and update.transition.support_alternating is not None:
                alternating_checks += 1
                alternation_violations += int(not update.transition.support_alternating)
            if update.completed_episode is not None:
                completed_episode = update.completed_episode
            if confirmed_exit_reason is None:
                if bool(sample.get("practical_confirmed", False)):
                    confirmed_exit_reason = RecoveryExitReason.SUCCESS
                elif int(sample["touchdown"]) >= 5:
                    confirmed_exit_reason = RecoveryExitReason.TIMEOUT
            if orbit_exit_reason is None:
                if int(sample["N_theory"]) == 0:
                    orbit_exit_reason = RecoveryExitReason.SUCCESS
                elif int(sample["touchdown"]) >= 5:
                    orbit_exit_reason = RecoveryExitReason.TIMEOUT

        if manager.recovery_active and trial.get("failure_reason") == "fall_or_illegal_contact":
            completed_episode = manager.on_fall()
        if confirmed_exit_reason is None and trial.get("failure_reason") == "fall_or_illegal_contact":
            confirmed_exit_reason = RecoveryExitReason.FALL
        if orbit_exit_reason is None and trial.get("failure_reason") == "fall_or_illegal_contact":
            orbit_exit_reason = RecoveryExitReason.FALL
        if completed_episode is None and manager.completed_episodes:
            completed_episode = manager.completed_episodes[-1]

        if completed_episode is None:
            trial["recovery_state_machine"] = None
            incomplete_count += 1
        else:
            manager.record_actual_recovery(
                enter_step=(int(trial["N_actual"]) if trial["N_actual"] is not None else None),
                confirmed_step=(
                    int(trial["N_confirmation"])
                    if trial["N_confirmation"] is not None
                    else None
                ),
                recovery_time=(
                    float(trial["t_recovery"])
                    if trial["t_recovery"] is not None
                    else None
                ),
                practical_enter_step=(
                    int(trial["practical_enter_step"])
                    if trial["practical_enter_step"] is not None
                    else None
                ),
                episode=completed_episode,
            )
            trial["recovery_state_machine"] = completed_episode.to_dict()
            exit_counts[completed_episode.exit_reason.value] += 1
            all_rewards_zero = all_rewards_zero and completed_episode.recovery_reward == 0.0
            practical_success_without_orbit += int(
                completed_episode.exit_reason is RecoveryExitReason.SUCCESS
                and not completed_episode.orbit_recovered
            )
            if confirmed_exit_reason is not None:
                confirmed_exit_counts[confirmed_exit_reason.value] += 1
                old_confirmed_timeout_new_entered_success += int(
                    confirmed_exit_reason is RecoveryExitReason.TIMEOUT
                    and completed_episode.exit_reason is RecoveryExitReason.SUCCESS
                )
            if orbit_exit_reason is not None:
                orbit_exit_counts[orbit_exit_reason.value] += 1
            trial["recovery_state_machine_comparison"] = {
                "old_practical_confirmed_exit_reason": (
                    confirmed_exit_reason.value if confirmed_exit_reason is not None else None
                ),
                "new_practical_entered_exit_reason": completed_episode.exit_reason.value,
            }
            if completed_episode.exit_reason is RecoveryExitReason.SUCCESS:
                assert completed_episode.practical_enter_step is not None
                practical_enter_step_distribution[str(completed_episode.practical_enter_step)] += 1
                exit_n = completed_episode.transitions[-1].n_current
                success_n_distribution["0" if exit_n == 0 else "1" if exit_n == 1 else ">=2"] += 1
            elif completed_episode.exit_reason is RecoveryExitReason.TIMEOUT:
                exit_n = completed_episode.transitions[-1].n_current
                timeout_n_distribution[str(exit_n) if exit_n <= 5 else ">5"] += 1

        manager.reset()
        reset_failures += int(manager.recovery_active or manager.current_episode is not None)

    return {
        "definition": (
            "Known velocity perturbation enters RECOVERY; certificates are evaluated at the "
            "push and once per new touchdown only."
        ),
        "success_condition": "first completed practical good gait cycle (practical_entered)",
        "timeout_condition": "five new touchdowns without practical_entered",
        "orbit_recovered_definition": "N_min == 0; logged independently and does not control exit",
        "exit_counts": exit_counts,
        "old_practical_confirmed_exit_counts_on_same_trajectories": confirmed_exit_counts,
        "orbit_exit_counts_on_same_trajectories": orbit_exit_counts,
        "old_confirmed_timeout_new_entered_success_count": (
            old_confirmed_timeout_new_entered_success
        ),
        "practical_success_without_orbit_count": practical_success_without_orbit,
        "practical_enter_step_distribution": practical_enter_step_distribution,
        "success_N_at_exit_distribution": success_n_distribution,
        "timeout_N_at_TD5_distribution": timeout_n_distribution,
        "practical_confirmed_final_count": sum(bool(trial["success"]) for trial in trials),
        "practical_confirmed_final_fraction": (
            sum(bool(trial["success"]) for trial in trials) / len(trials) if trials else None
        ),
        "incomplete_count": incomplete_count,
        "max_new_touchdowns": 5,
        "passive_logging_through_touchdown": (
            8 if args.q_memory_diagnostic else args.passive_log_touchdowns
        ),
        "touchdown_duplicate_count": duplicate_touchdowns,
        "support_alternation_check_count": alternating_checks,
        "support_alternation_violation_count": alternation_violations,
        "reset_state_failure_count": reset_failures,
        "all_recovery_rewards_zero": all_rewards_zero,
        "timeout_terminates_environment": False,
    }


def _save_raw_disturbed_trials(
    trials: list[dict], parameters: dict, thresholds: dict, nominal_samples: list[dict]
) -> Path:
    output = args.output.expanduser().resolve()
    raw_output = output.with_name(f"{output.stem}_raw{output.suffix}")
    raw_report = {
        "schema_version": 1,
        "parameters": str(args.params.expanduser().resolve()),
        "checkpoint": parameters["provenance"]["checkpoint"],
        "actual_recovery_thresholds": thresholds,
        "trial_count": len(trials),
        "trials": trials,
        "q_memory_nominal_samples": nominal_samples if args.q_memory_diagnostic else None,
    }
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with raw_output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(_native(raw_report), stream, sort_keys=False, allow_unicode=True)
    print(f"[INFO] Saved raw disturbed trials to {raw_output}", flush=True)
    return raw_output


def _gate2(env, policy, extractor, parameters, thresholds, obs):
    if args.q_memory_diagnostic and not args.recovery_manager_validation:
        raise ValueError("--q_memory_diagnostic requires --recovery_manager_validation")
    if args.q_memory_sample_count <= 0:
        raise ValueError("--q_memory_sample_count must be positive")
    if not 5 <= args.passive_log_touchdowns <= 8:
        raise ValueError("--passive_log_touchdowns must be between 5 and 8")
    plans = _trial_plans(parameters)
    pending = list(plans)
    worker_env_count = min(env.num_envs, len(plans))
    active: list[dict | None] = [None] * env.num_envs
    prepared: list[dict | None] = [None] * env.num_envs
    prepared_since = np.full(env.num_envs, -1, dtype=np.int64)
    cooldown_until = np.full(env.num_envs, args.warmup_steps, dtype=np.int64)
    stable_touchdowns = np.zeros(env.num_envs, dtype=np.int64)
    last_phase = np.zeros(env.num_envs)
    completed = []
    nominal_diagnostic_samples = []
    nominal_sampled_envs = set()
    timeout_steps = int(math.ceil(args.trial_timeout_s / env.step_dt))
    assigned_speeds = np.full(env.num_envs, args.validation_speed)

    for step in range(args.max_gate1_steps + max(1000, len(plans) * timeout_steps)):
        _set_commands(env, assigned_speeds)
        step_start = time.perf_counter()
        with torch.inference_mode():
            if step == 0:
                print("[INFO] Gate 2 step 0: starting policy inference", flush=True)
            actions = policy(obs)
            policy_done = time.perf_counter()
            if step == 0:
                print(
                    f"[INFO] Gate 2 step 0: policy finished in {policy_done - step_start:.3f}s; "
                    "starting env.step",
                    flush=True,
                )
            obs, _, dones, _ = env.step(actions)
            env_done = time.perf_counter()
            if step == 0:
                print(
                    f"[INFO] Gate 2 step 0: env.step finished in {env_done - policy_done:.3f}s; "
                    "starting state extraction",
                    flush=True,
                )
            state = extractor.extract()
            extractor_done = time.perf_counter()
        # One batched transfer avoids thousands of synchronizing CUDA .item()
        # calls when validation is run with large environment counts.
        phase_values = state.phase.detach().cpu().numpy()
        if step < 3:
            print(
                f"[INFO] Gate 2 timing step={step}: policy={policy_done - step_start:.3f}s, "
                f"env_step={env_done - policy_done:.3f}s, "
                f"extractor={extractor_done - env_done:.3f}s",
                flush=True,
            )

        reset_mask = dones | state.episode_reset
        for env_id in reset_mask.nonzero(as_tuple=False).flatten().tolist():
            if active[env_id] is not None:
                active[env_id]["failure_reason"] = "fall_or_illegal_contact"
                completed.append(_finalize_trial(active[env_id]))
                active[env_id] = None
            if prepared[env_id] is not None:
                pending.append(prepared[env_id])
                prepared[env_id] = None
                prepared_since[env_id] = -1
            stable_touchdowns[env_id] = 0
            cooldown_until[env_id] = step + args.warmup_steps

        for env_id in state.touchdown.nonzero(as_tuple=False).flatten().tolist():
            stable_touchdowns[env_id] += 1
            trial = active[env_id]
            if (
                args.q_memory_diagnostic
                and trial is None
                and prepared[env_id] is not None
                and stable_touchdowns[env_id] >= 3
                and env_id not in nominal_sampled_envs
                and len(nominal_diagnostic_samples) < args.q_memory_sample_count
            ):
                nominal_diagnostic_samples.append(
                    {
                        "sample_index": len(nominal_diagnostic_samples),
                        "env_id": env_id,
                        "support_side": (
                            "left" if int(state.touchdown_foot[env_id].item()) == 0 else "right"
                        ),
                        "_query": _certificate_query(state, env_id, parameters),
                        **_touchdown_diagnostic_snapshot(state, env_id),
                    }
                )
                nominal_sampled_envs.add(env_id)
            if trial is None:
                continue
            foot = int(state.touchdown_foot[env_id].item())
            touchdown_time = float(state.time[env_id].item())
            trial["touchdowns"] += 1
            trial["certificate_trace"].append(
                {
                    "touchdown": trial["touchdowns"],
                    "time_after_push": touchdown_time - trial["start_time"],
                    "support_side": "left" if foot == 0 else "right",
                    "N_theory": None,
                    "margin": None,
                    "good_cycle": None,
                    "practical_entered": bool(trial["practical_entered"]),
                    "practical_confirmed": bool(trial["success"]),
                    "practical_enter_step": trial["practical_enter_step"],
                    "practical_confirmed_step": trial["N_confirmation"],
                    "status": "pending",
                    "_query": _certificate_query(state, env_id, parameters),
                    **(
                        _touchdown_diagnostic_snapshot(state, env_id)
                        if args.q_memory_diagnostic
                        else {}
                    ),
                }
            )
            alternating = trial["last_touchdown_foot"] is None or foot != trial["last_touchdown_foot"]
            good = None
            if trial["interval_started_after_touchdown"] and trial["interval_velocity"]:
                mean_velocity = float(np.mean(trial["interval_velocity"]))
                mean_tilt = np.mean(np.asarray(trial["interval_tilt"]), axis=0)
                good = bool(
                    alternating
                    and mean_velocity <= thresholds["mean_velocity_error"]
                    and mean_tilt[0] <= thresholds["mean_abs_roll"]
                    and mean_tilt[1] <= thresholds["mean_abs_pitch"]
                )
                trial["cycle_metrics"].append(
                    {
                        "mean_velocity_error": mean_velocity,
                        "mean_abs_roll": float(mean_tilt[0]),
                        "mean_abs_pitch": float(mean_tilt[1]),
                        "alternating": alternating,
                        "good": good,
                        "start_touchdown": trial["interval_start_touchdown"],
                        "end_touchdown": trial["touchdowns"],
                    }
                )
                if good:
                    if trial["consecutive_good_cycles"] == 0:
                        # The first good complete cycle shows that normal gait
                        # had resumed at its starting touchdown.  Keep waiting
                        # for a second good cycle only to confirm stability.
                        trial["recovery_onset_touchdown"] = trial["interval_start_touchdown"]
                        trial["recovery_onset_time"] = trial["interval_start_time"] - trial["start_time"]
                        if not trial["practical_entered"]:
                            trial["practical_entered"] = True
                            trial["practical_enter_step"] = trial["touchdowns"]
                    trial["consecutive_good_cycles"] += 1
                else:
                    trial["consecutive_good_cycles"] = 0
                    trial["recovery_onset_touchdown"] = None
                    trial["recovery_onset_time"] = None
            trial["certificate_trace"][-1]["good_cycle"] = good
            trial["interval_started_after_touchdown"] = True
            trial["interval_start_touchdown"] = trial["touchdowns"]
            trial["interval_start_time"] = touchdown_time
            trial["last_touchdown_foot"] = foot
            trial["interval_velocity"].clear()
            trial["interval_tilt"].clear()
            if trial["consecutive_good_cycles"] >= 2 and not trial["success"]:
                trial["success"] = True
                trial["N_actual"] = trial["recovery_onset_touchdown"]
                trial["N_confirmation"] = trial["touchdowns"]
                trial["t_recovery"] = trial["recovery_onset_time"]
                trial["t_confirmation"] = touchdown_time - trial["start_time"]
                if not args.recovery_manager_validation:
                    completed.append(_finalize_trial(trial))
                    active[env_id] = None
                    cooldown_until[env_id] = step + 100
                    stable_touchdowns[env_id] = 0
            trial["certificate_trace"][-1].update(
                {
                    "practical_entered": bool(trial["practical_entered"]),
                    "practical_confirmed": bool(trial["success"]),
                    "practical_enter_step": trial["practical_enter_step"],
                    "practical_confirmed_step": trial["N_confirmation"],
                }
            )
            if (
                args.recovery_manager_validation
                and active[env_id] is trial
                and trial["touchdowns"]
                >= (8 if args.q_memory_diagnostic else args.passive_log_touchdowns)
            ):
                completed.append(_finalize_trial(trial))
                active[env_id] = None
                cooldown_until[env_id] = step + 100
                stable_touchdowns[env_id] = 0

        for env_id in range(worker_env_count):
            trial = active[env_id]
            if trial is not None:
                velocity_error, tilt = _per_step_metric(state, env_id)
                trial["interval_velocity"].append(velocity_error)
                trial["interval_tilt"].append(tilt)
                if step - trial["start_step"] >= timeout_steps:
                    trial["failure_reason"] = "recovery_timeout"
                    completed.append(_finalize_trial(trial))
                    active[env_id] = None
                    cooldown_until[env_id] = step + 100
                    stable_touchdowns[env_id] = 0

            if (
                active[env_id] is None
                and prepared[env_id] is None
                and pending
                and step >= cooldown_until[env_id]
            ):
                prepared[env_id] = pending.pop()
                prepared_since[env_id] = step
                assigned_speeds[env_id] = prepared[env_id].get("command_vx", args.validation_speed)
                stable_touchdowns[env_id] = 0

            if (
                active[env_id] is None
                and prepared[env_id] is not None
                and step - prepared_since[env_id] >= max(500, 2 * args.warmup_steps)
            ):
                pending.append(prepared[env_id])
                prepared[env_id] = None
                prepared_since[env_id] = -1
                stable_touchdowns[env_id] = 0

            if active[env_id] is None and prepared[env_id] is not None and stable_touchdowns[env_id] >= 3:
                target_phase = prepared[env_id]["target_phase"]
                phase_now = float(phase_values[env_id])
                crossed = (
                    (target_phase == 0.0 and phase_now < last_phase[env_id])
                    or (last_phase[env_id] < target_phase <= phase_now)
                )
                if crossed:
                    active[env_id] = _start_trial(env, state, env_id, prepared[env_id], parameters, step)
                    prepared[env_id] = None
                    prepared_since[env_id] = -1
                    stable_touchdowns[env_id] = 0
            last_phase[env_id] = phase_values[env_id]

        if (
            not pending
            and all(plan is None for plan in prepared[:worker_env_count])
            and all(trial is None for trial in active[:worker_env_count])
        ):
            break
        if step > 0 and step % 100 == 0:
            print(
                f"[INFO] Gate 2 step={step}, completed={len(completed)}, pending={len(pending)}, "
                f"prepared={sum(plan is not None for plan in prepared)}, "
                f"active={sum(trial is not None for trial in active)}",
                flush=True,
            )
    else:
        print(f"[WARN] Gate 2 stopped with {len(pending)} pending plans")

    if args.random_trials > 0:
        _save_raw_disturbed_trials(
            completed, parameters, thresholds, nominal_diagnostic_samples
        )
    _resolve_saved_certificates(completed, parameters)
    if args.q_memory_diagnostic:
        _resolve_nominal_diagnostic_samples(nominal_diagnostic_samples, parameters)
    state_machine_summary = (
        _attach_recovery_state_machine(completed)
        if args.recovery_manager_validation
        else None
    )
    q_memory_summary = (
        _q_memory_diagnostic_summary(completed, nominal_diagnostic_samples)
        if args.q_memory_diagnostic
        else None
    )
    valid = [trial for trial in completed if trial["N_theory"] is not None and trial["margin"] is not None]
    if not valid:
        raise RuntimeError("Gate 2 produced no trials with a valid certificate result")
    max_success_steps = max((trial["N_actual"] for trial in valid if trial["success"]), default=5)
    theory = np.asarray([trial["N_theory"] for trial in valid], dtype=float)
    actual_difficulty = np.asarray(
        [trial["N_actual"] if trial["success"] else max_success_steps + 1 for trial in valid], dtype=float
    )
    margin = np.asarray([trial["margin"] for trial in valid], dtype=float)
    success = np.asarray([trial["success"] for trial in valid], dtype=float)

    grouped = {}
    for n_value in sorted(set(int(value) for value in theory)):
        group = [trial for trial in valid if int(trial["N_theory"]) == n_value]
        successful = [trial for trial in group if trial["success"]]
        label = str(n_value) if n_value < 6 else ">5"
        grouped[label] = {
            "count": len(group),
            "success_rate": len(successful) / len(group),
            "mean_N_actual_success_only": float(np.mean([item["N_actual"] for item in successful])) if successful else None,
            "median_N_actual_success_only": float(np.median([item["N_actual"] for item in successful])) if successful else None,
            "mean_t_recovery_success_only": float(np.mean([item["t_recovery"] for item in successful])) if successful else None,
            "median_t_recovery_success_only": float(np.median([item["t_recovery"] for item in successful])) if successful else None,
            "mean_margin": float(np.mean([item["margin"] for item in group])),
        }

    def summarize(group):
        successful = [trial for trial in group if trial["success"]]
        return {
            "count": len(group),
            "success_rate": len(successful) / len(group) if group else None,
            "mean_N_actual_success_only": (
                float(np.mean([trial["N_actual"] for trial in successful])) if successful else None
            ),
            "median_N_actual_success_only": (
                float(np.median([trial["N_actual"] for trial in successful])) if successful else None
            ),
            "mean_t_recovery_success_only": (
                float(np.mean([trial["t_recovery"] for trial in successful])) if successful else None
            ),
        }

    scope_groups = {}
    for scope in sorted(set(trial.get("push_scope", "grid") for trial in valid)):
        scope_groups[scope] = summarize(
            [trial for trial in valid if trial.get("push_scope", "grid") == scope]
        )

    magnitude_groups = {}
    magnitude_edges = (0.0, 0.5, 1.0, 1.5, 2.0, math.inf)
    for lower, upper in zip(magnitude_edges[:-1], magnitude_edges[1:]):
        group = [trial for trial in valid if lower <= float(trial["level"]) < upper]
        if group:
            label = f"[{lower:.1f}, {'inf' if math.isinf(upper) else f'{upper:.1f}'})"
            magnitude_groups[label] = summarize(group)

    command_speeds = np.asarray([float(trial.get("command_vx", args.validation_speed)) for trial in valid])

    theory_trace_pairs = []
    nonincreasing_count = 0
    transition_count = 0
    monotonic_trials = 0
    trace_trials = 0
    theory_at_recovery = []
    for trial in valid:
        if not trial["success"]:
            continue
        samples = [
            sample
            for sample in trial["certificate_trace"]
            if sample["N_theory"] is not None and sample["touchdown"] <= trial["N_actual"]
        ]
        if not samples:
            continue
        trace_trials += 1
        encoded = [min(int(sample["N_theory"]), 6) for sample in samples]
        remaining = [int(sample["N_actual_remaining"]) for sample in samples]
        theory_trace_pairs.extend(zip(encoded, remaining))
        transitions = list(zip(encoded[:-1], encoded[1:]))
        transition_count += len(transitions)
        trial_nonincreasing = sum(next_value <= value for value, next_value in transitions)
        nonincreasing_count += trial_nonincreasing
        if trial_nonincreasing == len(transitions):
            monotonic_trials += 1
        recovery_samples = [
            sample for sample in trial["certificate_trace"] if sample["touchdown"] == trial["N_actual"]
        ]
        if recovery_samples and recovery_samples[0]["N_theory"] is not None:
            theory_at_recovery.append(min(int(recovery_samples[0]["N_theory"]), 6))

    if theory_trace_pairs:
        trace_theory = np.asarray([pair[0] for pair in theory_trace_pairs], dtype=float)
        trace_remaining = np.asarray([pair[1] for pair in theory_trace_pairs], dtype=float)
        trace_correlation = _spearman_summary(trace_theory, trace_remaining)
        trace_mae = float(np.mean(np.abs(trace_theory - trace_remaining)))
        trace_exact = float(np.mean(trace_theory == trace_remaining))
    else:
        trace_correlation = {"rho": None, "p_value": None, "reason": "no valid trajectory pairs"}
        trace_mae = None
        trace_exact = None

    n_actual_distribution = {
        str(value): sum(trial["success"] and int(trial["N_actual"]) == value for trial in valid)
        for value in sorted(set(int(trial["N_actual"]) for trial in valid if trial["success"]))
    }
    n_theory_distribution = {
        (str(value) if value < 6 else ">5"): sum(int(trial["N_theory"]) == value for trial in valid)
        for value in sorted(set(int(trial["N_theory"]) for trial in valid))
    }
    if args.random_trials > 0:
        conditions = {
            "mode": "randomized",
            "requested_trial_count": args.random_trials,
            "in_range_fraction": args.in_range_fraction,
            "trained_delta_v_component_range": [
                -args.trained_abs_delta_v_xy,
                args.trained_abs_delta_v_xy,
            ],
            "outside_delta_v_component_limit": args.outside_abs_delta_v_xy,
            "command_vx_range": [float(command_speeds.min()), float(command_speeds.max())],
            "push_phase_range": [0.0, 1.0],
            "frame": "heading",
            "seed": args.seed,
            "note": "Forward command only because the current theory parameters were calibrated on straight walking.",
        }
    else:
        conditions = {
            "mode": "fixed_grid",
            "velocity_jump_magnitudes": list(args.push_levels),
            "direction_count": args.direction_count,
            "target_phases": list(args.push_phases),
            "trials_per_condition": args.trials_per_condition,
            "frame": "heading",
            "training_delta_v_component_range": [-1.0, 1.0],
            "note": "Levels above 1.0 m/s deliberately test beyond the trained per-axis range.",
        }
    return {
        "trial_count": len(completed),
        "valid_certificate_trial_count": len(valid),
        "conditions": conditions,
        "metric_definitions": {
            "N_actual": "touchdowns before the start of the first of two consecutive good complete cycles",
            "N_confirmation": "touchdowns observed through the end of the second consecutive good cycle",
            "t_recovery": "time to the start of the first confirmed-good cycle",
            "t_confirmation": "time through the end of the second confirmed-good cycle",
        },
        "recovery_state_machine": state_machine_summary,
        "q_memory_diagnostic": q_memory_summary,
        "spearman": {
            "N_theory_vs_actual_difficulty": _spearman_summary(theory, actual_difficulty),
            "margin_vs_actual_difficulty": _spearman_summary(margin, actual_difficulty),
            "margin_vs_success": _spearman_summary(margin, success),
            "failed_trial_actual_difficulty_encoding": int(max_success_steps + 1),
        },
        "step_distributions": {
            "N_theory_initial": n_theory_distribution,
            "N_actual_success_only": n_actual_distribution,
        },
        "trajectory_consistency": {
            "definition": "Initial state and each touchdown through the measured recovery onset.",
            "successful_trials_with_trace": trace_trials,
            "transition_count": transition_count,
            "nonincreasing_transition_fraction": (
                nonincreasing_count / transition_count if transition_count else None
            ),
            "fully_nonincreasing_trial_fraction": monotonic_trials / trace_trials if trace_trials else None,
            "N_theory_vs_N_actual_remaining_spearman": trace_correlation,
            "encoded_over_horizon_value": 6,
            "mean_absolute_step_error": trace_mae,
            "exact_step_match_fraction": trace_exact,
            "N_theory_at_actual_recovery_distribution": {
                (str(value) if value < 6 else ">5"): theory_at_recovery.count(value)
                for value in sorted(set(theory_at_recovery))
            },
        },
        "by_N_theory": grouped,
        "by_push_scope": scope_groups,
        "by_push_magnitude": magnitude_groups,
        "command_vx_statistics": {
            "min": float(command_speeds.min()),
            "mean": float(command_speeds.mean()),
            "median": float(np.median(command_speeds)),
            "max": float(command_speeds.max()),
        },
        "trials": completed,
    }


def _save_report(report):
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(_native(report), stream, sort_keys=False, allow_unicode=True)
    print(f"[INFO] Saved G1 recoverability report to {output}")
    trial_count = (report.get("gate2_disturbed") or {}).get("trial_count", 0)
    if trial_count <= 100:
        print(yaml.safe_dump(_native(report), sort_keys=False, allow_unicode=True))
    else:
        summary = dict(report)
        summary["gate2_disturbed"] = dict(report["gate2_disturbed"])
        summary["gate2_disturbed"].pop("trials", None)
        print(yaml.safe_dump(_native(summary), sort_keys=False, allow_unicode=True))


def main():
    try:
        parameters = _load_yaml(args.params)
        env, policy, checkpoint = _make_env_policy(parameters)
        extractor = G1PrivilegedStateExtractor(
            env, G1StateExtractorCfg(h_eff=float(parameters["h_eff"]), fallback_step_period=float(parameters["T"]))
        )
        if args.reuse_gate1_report is not None:
            baseline_report_path = args.reuse_gate1_report.expanduser().resolve()
            baseline_report = _load_yaml(baseline_report_path)
            gate1 = baseline_report["gate1_nominal"]
            thresholds = gate1["actual_recovery_thresholds"]
            obs, _ = env.get_observations()
            print(f"[INFO] Reusing Gate 1 results from {baseline_report_path}", flush=True)
        else:
            gate1, thresholds, obs = _gate1(env, policy, extractor, parameters)
        report = {
            "schema_version": 3,
            "parameters": str(args.params.expanduser().resolve()),
            "checkpoint": str(checkpoint),
            "gate1_nominal": gate1,
            "gate2_disturbed": None,
        }
        if not gate1["passed"]:
            report["stopped_after_gate1"] = True
            report["stop_reason"] = "Nominal gait failed the configured certificate sanity criterion."
            _save_report(report)
            return
        report["gate2_disturbed"] = _gate2(env, policy, extractor, parameters, thresholds, obs)
        report["stopped_after_gate1"] = False
        _save_report(report)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
