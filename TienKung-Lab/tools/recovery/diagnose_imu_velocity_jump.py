#!/usr/bin/env python3
"""Measure whether deployable pelvis-IMU acceleration observes velocity-setting pushes."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER = REPOSITORY_ROOT / "logs/g1_slope_sys_d.pt"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "tools/recovery/generated/g1_velocity_jump_imu_observability.yaml"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="g1_com_velocity_estimator_v2")
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_TEACHER)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--push_interval_s", type=float, default=6.0)
parser.add_argument("--max_policy_steps", type=int, default=400)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.envs.mdp.events import push_by_setting_velocity  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.recovery.state_extractor import G1PrivilegedStateExtractor  # noqa: E402
from legged_lab.utils import task_registry  # noqa: E402


def _observable_velocity_push(
    env,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Apply the production velocity-setting push while exposing its exact delta."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene[asset_cfg.name]
    before = robot.data.root_vel_w[env_ids, :3].clone()
    push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
    after = robot.data.root_vel_w[env_ids, :3].clone()
    env._imu_jump_event_mask[env_ids] = True
    env._imu_jump_delta_v_w[env_ids] = after - before


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "P50": None, "P90": None, "P95": None, "max": None}
    return {
        "count": int(values.size),
        "P50": float(np.quantile(values, 0.50)),
        "P90": float(np.quantile(values, 0.90)),
        "P95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _sample(env, extractor: G1PrivilegedStateExtractor, dones: torch.Tensor) -> dict[str, np.ndarray]:
    state = extractor.extract()
    imu = env.scene.sensors["imu"]
    specific_force_w = quat_apply(imu.data.quat_w, imu.data.lin_acc_b)
    dynamic_acc_w = specific_force_w.clone()
    dynamic_acc_w[:, 2] -= 9.81
    heading = yaw_quat(env.robot.data.root_quat_w)
    acceleration_heading = quat_apply_inverse(heading, dynamic_acc_w)
    root_velocity_heading = quat_apply_inverse(heading, env.robot.data.root_lin_vel_w)
    return {
        "com_velocity_heading": state.com_velocity.detach().cpu().numpy(),
        "root_velocity_heading": root_velocity_heading.detach().cpu().numpy(),
        "imu_specific_force_body": imu.data.lin_acc_b.detach().cpu().numpy(),
        "imu_dynamic_acceleration_heading": acceleration_heading.detach().cpu().numpy(),
        "done": dones.detach().cpu().numpy().astype(bool),
    }


def _row(sample: dict[str, np.ndarray], env_id: int, relative_frame: int) -> dict[str, Any]:
    return {
        "relative_policy_frame": int(relative_frame),
        "GT_CoM_velocity_heading_m_per_s": sample["com_velocity_heading"][env_id].tolist(),
        "root_velocity_heading_m_per_s": sample["root_velocity_heading"][env_id].tolist(),
        "IMU_specific_force_body_m_per_s2": sample["imu_specific_force_body"][env_id].tolist(),
        "IMU_dynamic_acceleration_heading_m_per_s2": sample[
            "imu_dynamic_acceleration_heading"
        ][env_id].tolist(),
        "reset_or_fall": bool(sample["done"][env_id]),
    }


def _summarize_windows(windows: list[dict[str, Any]], acceleration_sample_dt: float) -> dict[str, Any]:
    complete = [window for window in windows if len(window["frames"]) == 9]
    push = np.asarray([window["push_delta_v_heading_m_per_s"] for window in complete])
    integrals = []
    post_peaks = []
    pre_peaks = []
    for window in complete:
        frames = window["frames"]
        acc = np.asarray(
            [frame["IMU_dynamic_acceleration_heading_m_per_s2"] for frame in frames],
            dtype=np.float64,
        )
        # t=0 is the event boundary; t=1..5 are the sensor updates after it.
        # Isaac's IMU finite-differences the link velocity at the physics update
        # that observes the jump.  Preserve that impulse sample duration instead
        # of incorrectly treating its peak as constant for a full policy step.
        integrals.append(np.sum(acc[3:, :2], axis=0) * acceleration_sample_dt)
        post_peaks.append(np.max(np.linalg.norm(acc[3:, :2], axis=1)))
        pre_peaks.append(np.max(np.linalg.norm(acc[:3, :2], axis=1)))
    integral = np.asarray(integrals)
    post_peaks_array = np.asarray(post_peaks)
    pre_peaks_array = np.asarray(pre_peaks)
    component_rho = _pearson(push[:, :2], integral)
    x_rho = _pearson(push[:, 0], integral[:, 0])
    y_rho = _pearson(push[:, 1], integral[:, 1])
    sign_agreement = float(np.mean(np.sign(push[:, :2]) == np.sign(integral)))
    observable = bool(
        component_rho is not None
        and component_rho >= 0.5
        and sign_agreement >= 0.65
    )
    return {
        "complete_window_count": len(complete),
        "window_policy_frames": [-3, -2, -1, 0, 1, 2, 3, 4, 5],
        "acceleration_impulse_sample_dt_s": acceleration_sample_dt,
        "push_delta_v_norm_m_per_s": _distribution(np.linalg.norm(push[:, :2], axis=1)),
        "post_push_horizontal_accel_peak_m_per_s2": _distribution(post_peaks_array),
        "pre_push_horizontal_accel_peak_m_per_s2": _distribution(pre_peaks_array),
        "short_window_accel_integral_norm_m_per_s": _distribution(
            np.linalg.norm(integral, axis=1)
        ),
        "delta_v_vs_integral_acceleration_pearson": {
            "flattened_xy_components": component_rho,
            "x": x_rho,
            "y": y_rho,
        },
        "delta_v_vs_integral_component_sign_agreement": sign_agreement,
        "velocity_jump_observable_at_policy_rate": observable,
        "observability_rule": "flattened component Pearson >=0.5 and component sign agreement >=0.65",
    }


def main() -> Path:
    checkpoint = args.checkpoint.resolve()
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.seed = args.seed
    env_cfg.device = args.device
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (0.4, 0.4)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    push_cfg = env_cfg.domain_rand.events.push_robot
    push_cfg.func = _observable_velocity_push
    push_cfg.interval_range_s = (args.push_interval_s, args.push_interval_s)
    agent_cfg.device = args.device

    env_class = task_registry.get_task_class(args.task)
    env = None
    windows: list[dict[str, Any]] = []
    try:
        env = env_class(env_cfg, headless=args.headless)
        if "imu" not in env.scene.sensors:
            raise RuntimeError("V2 diagnostic task did not instantiate the deployable IMU")
        env._imu_jump_event_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        env._imu_jump_delta_v_w = torch.zeros(args.num_envs, 3, device=args.device)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(checkpoint), load_optimizer=False)
        runner.eval_mode()
        runner.alg.policy.requires_grad_(False)
        policy = runner.get_inference_policy(device=args.device)
        observations, _ = env.get_observations()
        extractor = G1PrivilegedStateExtractor(env)
        pre_history: deque[dict[str, np.ndarray]] = deque(maxlen=3)
        active: dict[int, dict[str, Any]] = {}

        for step in range(args.max_policy_steps):
            env._imu_jump_event_mask.zero_()
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, dones, _ = env.step(actions)
            sample = _sample(env, extractor, dones.to(dtype=torch.bool))

            for env_id, window in list(active.items()):
                relative = int(window["next_relative_frame"])
                window["frames"].append(_row(sample, env_id, relative))
                window["next_relative_frame"] = relative + 1
                if relative >= 5:
                    windows.append(window)
                    del active[env_id]

            triggered = env._imu_jump_event_mask.nonzero(as_tuple=False).flatten().tolist()
            if triggered and len(pre_history) == 3:
                heading = yaw_quat(env.robot.data.root_quat_w)
                delta_heading = quat_apply_inverse(heading, env._imu_jump_delta_v_w)
                for env_id in triggered:
                    if env_id in active:
                        continue
                    frame_rows = [
                        _row(old, env_id, relative)
                        for old, relative in zip(pre_history, (-3, -2, -1))
                    ]
                    frame_rows.append(_row(sample, env_id, 0))
                    active[env_id] = {
                        "env_id": env_id,
                        "event_policy_step": step + 1,
                        "push_delta_v_heading_m_per_s": delta_heading[env_id]
                        .detach()
                        .cpu()
                        .tolist(),
                        "frames": frame_rows,
                        "next_relative_frame": 1,
                    }
            pre_history.append(sample)
            if len(windows) >= args.num_envs:
                break

        summary = _summarize_windows(windows, env.physics_dt)
        report = {
            "schema_version": 1,
            "diagnostic": "deployable_IMU_observability_of_velocity_setting_push",
            "task": args.task,
            "teacher_checkpoint": str(checkpoint),
            "teacher_frozen_eval": True,
            "disturbance": {
                "implementation": "Isaac Lab push_by_setting_velocity",
                "timing": "after policy-step physics integration, matching g1_slope_sys_d",
                "velocity_range_world_m_per_s": push_cfg.params["velocity_range"],
                "interval_s": args.push_interval_s,
            },
            "IMU": {
                "mount": "G1 pelvis at onboard imu_in_pelvis offset",
                "prim_path": env_cfg.scene.imu.prim_path,
                "offset_m": list(env_cfg.scene.imu.offset.pos),
                "raw_quantity": "specific force in IMU/body frame",
                "gravity_bias_m_per_s2": list(env_cfg.scene.imu.gravity_bias),
                "update_period_s": env_cfg.scene.imu.update_period,
                "deployable": True,
                "privileged_CoM_acceleration_used": False,
            },
            "summary": summary,
            "events": windows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
        print(yaml.safe_dump({"summary": summary}, sort_keys=False))
        print(f"[INFO] Saved IMU observability report to {args.output.resolve()}")
        return args.output.resolve()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
