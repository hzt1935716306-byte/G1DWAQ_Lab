#!/usr/bin/env python3
"""Collect nominal plane gait from a DWAQ policy without fitting C/L/v_max."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from types import MethodType
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument(
    "--task",
    default="g1_dwaq",
    help="Registered task whose environment and runner architecture match the checkpoint.",
)
parser.add_argument("--slopes_degrees", type=float, nargs="+", required=True)
parser.add_argument(
    "--directions",
    nargs="+",
    choices=("+x", "-x", "+y", "-y", "px", "nx", "py", "ny"),
    default=("+x", "-x", "+y", "-y"),
    help="Cardinal commands; px/nx/py/ny aliases avoid argparse treating -x/-y as options.",
)
parser.add_argument("--speeds", type=float, nargs="+", default=(0.2, 0.4))
parser.add_argument("--samples_per_node", type=int, default=40)
parser.add_argument("--envs_per_node", type=int, default=2)
parser.add_argument("--warmup_touchdowns", type=int, default=4)
parser.add_argument(
    "--segment_reset_on_invalid_plane",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "End only the current short collection segment when the 5-degree plane "
        "applicability gate becomes invalid, then reset and continue accumulating."
    ),
)
parser.add_argument("--max_steps", type=int, default=12000)
parser.add_argument("--slip_velocity_threshold", type=float, default=0.30)
parser.add_argument(
    "--slip_grace_time_s",
    type=float,
    default=0.04,
    help="Ignore contact-foot velocity for this duration after each contact rising edge.",
)
parser.add_argument(
    "--slip_consecutive_frames",
    type=int,
    default=2,
    help="Settled-contact threshold exceedances required consecutively for severe slip.",
)
parser.add_argument("--illegal_contact_force_threshold", type=float, default=1.0)
parser.add_argument(
    "--max_resets_per_node",
    type=int,
    default=0,
    help="Deprecated compatibility option; reset count is now diagnostic, not a node veto.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--expected_actor_history_length", type=int, default=None)
parser.add_argument("--expected_actor_frame_dim", type=int, default=None)
parser.add_argument(
    "--flat_parameters",
    type=Path,
    default=Path("tools/recovery/generated/g1_recovery_params.yaml"),
)
parser.add_argument(
    "--anchor_nominal",
    type=Path,
    default=Path("tools/recovery/generated/g1_plane_nominal_params.yaml"),
)
parser.add_argument(
    "--preserve_official_flat_anchor",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Keep official flat +x nodes immutable (default). Disable only for a "
        "separate frozen-policy validation table."
    ),
)
parser.add_argument(
    "--terminal_epsilon_semantics",
    choices=("joint_normalized_max_p95", "per_axis_absolute_p95"),
    default="joint_normalized_max_p95",
    help=(
        "Terminal tolerance calibration. Use per_axis_absolute_p95 for a "
        "validation-only table; the default preserves production semantics."
    ),
)
parser.add_argument(
    "--heading_alignment_diagnostic",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Record robot yaw, world slope direction, independently reconstructed "
        "alignment, terrain-plane validity, and frame rejection reasons."
    ),
)
parser.add_argument(
    "--heading_diagnostic_stride",
    type=int,
    default=1,
    help="Record one heading diagnostic frame every N policy steps.",
)
parser.add_argument(
    "--heading_diagnostic_max_records",
    type=int,
    default=20000,
    help="Maximum number of per-frame heading diagnostic records.",
)
parser.add_argument(
    "--slip_diagnostic",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Write a diagnostic-only report of the existing contact-foot horizontal "
        "speed slip metric; nominal nodes are not calibrated or updated."
    ),
)
parser.add_argument(
    "--slip_diagnostic_cycles_per_node",
    type=int,
    default=150,
    help="Complete post-warmup touchdown intervals to record per slope node.",
)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from rsl_rl.runners import DWAQOnPolicyRunner, OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.recovery.plane_nominal_calibration import (  # noqa: E402
    calibrate_nominal_node,
    mark_collection_done,
)
from legged_lab.recovery.plane_nominal_params import upsert_nominal_nodes  # noqa: E402
from legged_lab.recovery.practical_metrics import practical_frame_errors  # noqa: E402
from legged_lab.recovery.state_extractor import (  # noqa: E402
    G1PrivilegedStateExtractor,
    G1StateExtractorCfg,
)
from legged_lab.terrains import make_plane_recovery_terrain_cfg  # noqa: E402
from legged_lab.utils import task_registry  # noqa: E402


def _disable_randomization(cfg) -> None:
    cfg.noise.add_noise = False
    cfg.domain_rand.action_delay.enable = False
    events = cfg.domain_rand.events
    for name in (
        "push_robot",
        "physics_material",
        "add_base_mass",
        "randomize_actuator_gains",
        "randomize_com",
        "randomize_dome_light",
        "randomize_distant_light",
    ):
        if hasattr(events, name):
            setattr(events, name, None)
    reset_base = events.reset_base
    for key in reset_base.params["pose_range"]:
        reset_base.params["pose_range"][key] = (0.0, 0.0)
    for key in reset_base.params["velocity_range"]:
        reset_base.params["velocity_range"][key] = (0.0, 0.0)
    reset_joints = events.reset_robot_joints
    reset_joints.params["position_range"] = (1.0, 1.0)
    reset_joints.params["velocity_range"] = (0.0, 0.0)


def _command(direction: str, speed: float) -> tuple[float, float, float]:
    return {
        "+x": (speed, 0.0, 0.0),
        "-x": (-speed, 0.0, 0.0),
        "+y": (0.0, speed, 0.0),
        "-y": (0.0, -speed, 0.0),
    }[direction]


def _normalize_direction(direction: str) -> str:
    return {
        "+x": "+x",
        "-x": "-x",
        "+y": "+y",
        "-y": "-y",
        "px": "+x",
        "nx": "-x",
        "py": "+y",
        "ny": "-y",
    }[direction]


def _checkpoint_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_id(path: Path, task: str) -> str:
    return f"{task}:{path.name}:{_checkpoint_hash(path)[:12]}"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _build_node_assignments(slopes, directions, speeds, envs_per_node):
    assignments = []
    for slope in slopes:
        for direction in directions:
            for speed in speeds:
                for _ in range(envs_per_node):
                    assignments.append(
                        {
                            "slope_degrees": float(slope),
                            "direction": direction,
                            "speed": float(speed),
                            "command": _command(direction, float(speed)),
                        }
                    )
    return assignments


def _attach_plane_provider(env, slopes) -> None:
    def provider(self):
        terrain_types = self.scene.terrain.terrain_types
        slope_table = torch.tensor(slopes, dtype=torch.float32, device=self.device)
        valid = (terrain_types >= 0) & (terrain_types < len(slopes))
        safe_types = torch.clamp(terrain_types, 0, len(slopes) - 1)
        alpha = torch.deg2rad(slope_table[safe_types])
        normal = torch.stack(
            (-torch.sin(alpha), torch.zeros_like(alpha), torch.cos(alpha)), dim=-1
        )
        return normal, self.scene.env_origins.clone(), valid

    env.get_recovery_plane_geometry = MethodType(provider, env)


def _set_commands(env, assignments) -> None:
    commands = torch.tensor(
        [item["command"] for item in assignments],
        dtype=env.command_generator.command.dtype,
        device=env.device,
    )
    env.command_generator.command.copy_(commands)
    env.command_generator.is_standing_env[:] = False
    env.command_generator.is_heading_env[:] = False


def _node_key(item) -> tuple[float, str, float]:
    return (float(item["slope_degrees"]), str(item["direction"]), float(item["speed"]))


def _load_yaml(path: Path):
    return yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))


def _acceptance_ratio(accepted: int, candidates: int) -> float:
    return float(accepted) / max(int(candidates), 1)


def _node_collection_complete(samples, candidates, required: int) -> bool:
    del candidates
    return len(samples) >= required


def _request_segment_reset(env, env_id: int) -> None:
    # Let the normal step/reset path produce fresh post-reset observations and
    # clear DWAQ history on the following policy step.
    env.episode_length_buf[env_id] = env.max_episode_length


def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Return world yaw for Isaac's scalar-first quaternion convention."""

    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _slope_uphill_yaw_world(normal_world: torch.Tensor) -> torch.Tensor:
    """Return the world azimuth of uphill, i.e. minus the horizontal normal."""

    return torch.atan2(-normal_world[:, 1], -normal_world[:, 0])


def _axis_alignment_angle(yaw: torch.Tensor, slope_yaw: torch.Tensor) -> torch.Tensor:
    """Smallest unsigned angle between heading x and the undirected slope axis."""

    delta = yaw - slope_yaw
    return torch.acos(torch.clamp(torch.abs(torch.cos(delta)), 0.0, 1.0))


def _heading_frame_reject_reason(
    normal_heading: torch.Tensor,
    plane_valid: bool,
    tolerance: float,
) -> str:
    if plane_valid:
        return "accepted"
    if not bool(torch.all(torch.isfinite(normal_heading)).item()):
        return "nonfinite_terrain_normal"
    norm = float(torch.linalg.vector_norm(normal_heading).item())
    if norm <= 0.0:
        return "zero_terrain_normal"
    if float(normal_heading[2].item()) <= 0.0:
        return "terrain_normal_not_upward"
    horizontal = float(torch.linalg.vector_norm(normal_heading[:2]).item())
    if horizontal < math.sin(math.radians(0.5)):
        return "provider_or_geometry_invalid"
    alignment = math.atan2(
        abs(float(normal_heading[1].item())),
        abs(float(normal_heading[0].item())),
    )
    if alignment > tolerance:
        return "heading_misalignment"
    return "provider_or_geometry_invalid"


def _heading_diagnostic_summary(records: list[dict]) -> dict:
    if not records:
        return {
            "record_count": 0,
            "valid_frame_count": 0,
            "invalid_frame_count": 0,
            "reject_reason_counts": {},
        }
    alignment = np.asarray(
        [record["alignment_angle_degrees"] for record in records], dtype=np.float64
    )
    expected = np.asarray(
        [record["expected_alignment_degrees"] for record in records], dtype=np.float64
    )
    valid = np.asarray(
        [record["terrain_plane_valid"] for record in records], dtype=np.bool_
    )
    reasons: dict[str, int] = {}
    for record in records:
        reason = str(record["reject_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1

    def percentiles(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"median": None, "p95": None, "maximum": None}
        return {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95.0)),
            "maximum": float(np.max(values)),
        }

    return {
        "record_count": len(records),
        "valid_frame_count": int(np.count_nonzero(valid)),
        "invalid_frame_count": int(np.count_nonzero(~valid)),
        "reject_reason_counts": reasons,
        "alignment_degrees": percentiles(alignment),
        "invalid_alignment_degrees": percentiles(alignment[~valid]),
        "alignment_consistency_abs_error_degrees": percentiles(
            np.abs(alignment - expected)
        ),
    }


def _collector_cycle_reject_reason(
    *,
    interval_plane_valid: bool,
    interval_heading_valid: bool,
    previous: dict,
    current_foot: int,
    current_plane_valid: bool,
    current_height_valid: bool,
    interval_slip: bool,
    interval_illegal_contact: bool,
    has_interval_frames: bool,
) -> str:
    """Mirror the collector's existing rejection priority without changing it."""

    if not interval_plane_valid:
        return "heading_misalignment" if not interval_heading_valid else "invalid_plane"
    if not previous["valid_transition_start"]:
        if not previous["plane_valid"]:
            return "heading_misalignment" if not interval_heading_valid else "invalid_plane"
        if not previous["height_valid"]:
            return "invalid_height"
        if previous["slip"]:
            return "slip"
        if previous["illegal_contact"]:
            return "illegal_contact"
        return "invalid_transition_start"
    if previous["foot"] == current_foot:
        return "non_alternating"
    if not current_plane_valid:
        return "invalid_plane"
    if not current_height_valid:
        return "invalid_height"
    if interval_slip:
        return "slip"
    if interval_illegal_contact:
        return "illegal_contact"
    if not has_interval_frames:
        return "invalid_transition_start"
    return "accepted"


def _slip_frame_location_counts(frames: list[dict], radius_frames: int = 2) -> dict:
    touchdown_indices = [index for index, frame in enumerate(frames) if frame["touchdown"]]
    liftoff_indices = [index for index, frame in enumerate(frames) if frame["liftoff"]]
    counts = {"touchdown_near": 0, "support_mid_phase": 0, "liftoff_near": 0}
    for index, frame in enumerate(frames):
        if not frame["severe_slip"]:
            continue
        if any(abs(index - event) <= radius_frames for event in touchdown_indices):
            counts["touchdown_near"] += 1
        elif any(abs(index - event) <= radius_frames for event in liftoff_indices):
            counts["liftoff_near"] += 1
        else:
            counts["support_mid_phase"] += 1
    return counts


def _slip_cycle_record(
    *,
    cycle_index: int,
    key: tuple[float, str, float],
    env_id: int,
    previous: dict,
    current_foot: int,
    current_support: str,
    frames: list[dict],
    threshold: float,
    step_dt: float,
    collector_reject_reason: str,
    interval_plane_valid: bool,
    maximum_heading_error_degrees: float,
    step_period_s: float,
    measured_landing_xy: list[float],
    command_vx: float,
) -> dict:
    values = np.asarray([frame["metric"] for frame in frames], dtype=np.float64)
    transient_values = np.asarray(
        [frame["transient_metric"] for frame in frames], dtype=np.float64
    )
    settled_values = np.asarray(
        [frame["settled_metric"] for frame in frames], dtype=np.float64
    )
    settled_exceedance_count = sum(
        bool(frame["settled_threshold_exceeded"]) for frame in frames
    )
    severe_slip_frame_count = sum(bool(frame["severe_slip"]) for frame in frames)
    actual_mean_vx = float(np.mean([frame["actual_vx"] for frame in frames]))
    landing_x = float(measured_landing_xy[0])
    return {
        "slope_degrees": key[0],
        "direction": key[1],
        "speed": key[2],
        "cycle_index": cycle_index,
        "env_id": env_id,
        "transition_support_side": previous["support_side"],
        "touchdown_support_side": current_support,
        "ending_touchdown_foot": current_foot,
        "slip_threshold_m_per_s": threshold,
        "frame_count": len(frames),
        "max_slip_metric_m_per_s": float(np.max(values)),
        "mean_slip_metric_m_per_s": float(np.mean(values)),
        "p95_slip_metric_m_per_s": float(np.percentile(values, 95.0)),
        "touchdown_transient_max_speed_m_per_s": float(np.max(transient_values)),
        "settled_stance_max_speed_m_per_s": float(np.max(settled_values)),
        "settled_stance_mean_speed_m_per_s": float(np.mean(settled_values)),
        "settled_stance_p95_speed_m_per_s": float(
            np.percentile(settled_values, 95.0)
        ),
        "settled_exceedance_frame_count": settled_exceedance_count,
        "maximum_consecutive_exceedance_frames": max(
            int(frame["consecutive_exceedance_frames"]) for frame in frames
        ),
        "severe_slip_frame_count": severe_slip_frame_count,
        "severe_slip_duration_s": severe_slip_frame_count * step_dt,
        "slip_exceedance_location_frame_counts": _slip_frame_location_counts(frames),
        "tangential_foot_speed_diagnostic_max_m_per_s": max(
            float(frame["tangential_metric"]) for frame in frames
        ),
        "collector_reject_reason": collector_reject_reason,
        "slip_gate_rejected": collector_reject_reason == "slip",
        "terrain_plane_valid_throughout": interval_plane_valid,
        "maximum_heading_error_degrees": maximum_heading_error_degrees,
        "actual_mean_vx_m_per_s": actual_mean_vx,
        "step_period_s": step_period_s,
        "landing_foot": current_support,
        "measured_landing_xy_m": measured_landing_xy,
        "command_vx_times_T_m": command_vx * step_period_s,
        "delta_l_x_m": landing_x - command_vx * step_period_s,
        "reset_or_fall_during_complete_cycle": False,
        "next_alternating_touchdown_completed": None,
        "segment_reset_before_next_touchdown": None,
        "natural_reset_or_fall_before_next_touchdown": None,
    }


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50.0)),
        "p75": float(np.percentile(array, 75.0)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "maximum": float(np.max(array)),
    }


def _slip_node_summary(records: list[dict], natural_reset_count: int) -> dict:
    slip_rejected = [record for record in records if record["slip_gate_rejected"]]
    resolved = [
        record
        for record in slip_rejected
        if record["next_alternating_touchdown_completed"] is not None
    ]
    location_counts = {"touchdown_near": 0, "support_mid_phase": 0, "liftoff_near": 0}
    for record in records:
        for name, count in record["slip_exceedance_location_frame_counts"].items():
            location_counts[name] += int(count)
    return {
        "complete_candidate_cycle_count": len(records),
        "cycle_max_slip_metric_m_per_s": _distribution(
            [record["max_slip_metric_m_per_s"] for record in records]
        ),
        "cycle_mean_slip_metric_m_per_s": _distribution(
            [record["mean_slip_metric_m_per_s"] for record in records]
        ),
        "cycle_p95_slip_metric_m_per_s": _distribution(
            [record["p95_slip_metric_m_per_s"] for record in records]
        ),
        "touchdown_transient_max_speed_m_per_s": _distribution(
            [record["touchdown_transient_max_speed_m_per_s"] for record in records]
        ),
        "settled_stance_max_speed_m_per_s": _distribution(
            [record["settled_stance_max_speed_m_per_s"] for record in records]
        ),
        "tangential_foot_speed_diagnostic_max_m_per_s": _distribution(
            [
                record["tangential_foot_speed_diagnostic_max_m_per_s"]
                for record in records
            ]
        ),
        "maximum_consecutive_exceedance_frames": _distribution(
            [record["maximum_consecutive_exceedance_frames"] for record in records]
        ),
        "cycles_with_any_settled_threshold_exceedance": sum(
            int(record["settled_exceedance_frame_count"] > 0) for record in records
        ),
        "cycles_with_sustained_settled_threshold_exceedance": sum(
            int(record["maximum_consecutive_exceedance_frames"] >= 2)
            for record in records
        ),
        "slip_rejected_cycle_count": len(slip_rejected),
        "slip_rejection_rate": len(slip_rejected) / max(len(records), 1),
        "slip_exceedance_location_frame_counts": location_counts,
        "rejected_outcome_resolved_count": len(resolved),
        "rejected_then_next_alternating_touchdown_count": sum(
            bool(record["next_alternating_touchdown_completed"]) for record in resolved
        ),
        "rejected_then_segment_reset_count": sum(
            bool(record["segment_reset_before_next_touchdown"]) for record in resolved
        ),
        "rejected_then_natural_reset_or_fall_count": sum(
            bool(record["natural_reset_or_fall_before_next_touchdown"])
            for record in resolved
        ),
        "natural_reset_or_fall_count_during_diagnostic": natural_reset_count,
        "gait_speed_diagnostic": {
            "actual_mean_vx_m_per_s": float(
                np.mean([record["actual_mean_vx_m_per_s"] for record in records])
            ),
            "median_step_period_s": float(
                np.median([record["step_period_s"] for record in records])
            ),
            "median_step_period_left_s": float(
                np.median(
                    [
                        record["step_period_s"]
                        for record in records
                        if record["landing_foot"] == "left"
                    ]
                )
            ),
            "median_step_period_right_s": float(
                np.median(
                    [
                        record["step_period_s"]
                        for record in records
                        if record["landing_foot"] == "right"
                    ]
                )
            ),
            "measured_median_landing_left_xy_m": [
                float(
                    np.median(
                        [
                            record["measured_landing_xy_m"][axis]
                            for record in records
                            if record["landing_foot"] == "left"
                        ]
                    )
                )
                for axis in (0, 1)
            ],
            "measured_median_landing_right_xy_m": [
                float(
                    np.median(
                        [
                            record["measured_landing_xy_m"][axis]
                            for record in records
                            if record["landing_foot"] == "right"
                        ]
                    )
                )
                for axis in (0, 1)
            ],
            "median_command_vx_times_T_m": float(
                np.median([record["command_vx_times_T_m"] for record in records])
            ),
            "median_command_vx_times_T_left_m": float(
                np.median(
                    [
                        record["command_vx_times_T_m"]
                        for record in records
                        if record["landing_foot"] == "left"
                    ]
                )
            ),
            "median_command_vx_times_T_right_m": float(
                np.median(
                    [
                        record["command_vx_times_T_m"]
                        for record in records
                        if record["landing_foot"] == "right"
                    ]
                )
            ),
            "median_delta_l_x_m": float(
                np.median([record["delta_l_x_m"] for record in records])
            ),
            "median_delta_l_x_left_m": float(
                np.median(
                    [
                        record["delta_l_x_m"]
                        for record in records
                        if record["landing_foot"] == "left"
                    ]
                )
            ),
            "median_delta_l_x_right_m": float(
                np.median(
                    [
                        record["delta_l_x_m"]
                        for record in records
                        if record["landing_foot"] == "right"
                    ]
                )
            ),
        },
    }


def main() -> None:
    if args.samples_per_node < 10 or args.envs_per_node <= 0 or args.max_steps <= 0:
        raise ValueError("sample/env/step counts are invalid")
    if args.max_resets_per_node < 0:
        raise ValueError("max_resets_per_node must be non-negative")
    if args.slip_velocity_threshold <= 0.0 or args.illegal_contact_force_threshold <= 0.0:
        raise ValueError("contact and slip thresholds must be positive")
    if args.slip_grace_time_s < 0.0:
        raise ValueError("slip_grace_time_s must be non-negative")
    if args.slip_consecutive_frames <= 0:
        raise ValueError("slip_consecutive_frames must be positive")
    if any(abs(slope) >= 45.0 for slope in args.slopes_degrees):
        raise ValueError("collector slopes must satisfy abs(alpha) < 45 degrees")
    if any(speed <= 0.0 for speed in args.speeds):
        raise ValueError("collector speeds must be positive")
    if len(set(args.slopes_degrees)) != len(args.slopes_degrees):
        raise ValueError("collector slopes must be unique")
    if len(set(args.speeds)) != len(args.speeds):
        raise ValueError("collector speeds must be unique")
    if args.heading_diagnostic_stride <= 0:
        raise ValueError("heading_diagnostic_stride must be positive")
    if args.heading_diagnostic_max_records <= 0:
        raise ValueError("heading_diagnostic_max_records must be positive")
    if args.slip_diagnostic_cycles_per_node <= 0:
        raise ValueError("slip_diagnostic_cycles_per_node must be positive")

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    flat_path = args.flat_parameters.expanduser().resolve()
    anchor_path = args.anchor_nominal.expanduser().resolve()
    output = args.output.expanduser().resolve()
    flat_parameters = _load_yaml(flat_path)
    anchor = _load_yaml(anchor_path)
    existing_documents = [anchor]
    if output.is_file() and output != anchor_path:
        existing_documents.append(_load_yaml(output))
    existing_nodes = [
        node
        for document in existing_documents
        for node in document.get("nominal_plane_gait", {}).get("nodes", ())
        if bool(node.get("valid", True))
    ]
    previous_node_reports = {}
    if output.is_file():
        previous_node_reports.update(
            _load_yaml(output).get("collection", {}).get("node_reports", {})
        )
    anchor_nodes = [
        node
        for node in anchor["nominal_plane_gait"]["nodes"]
        if abs(float(node["slope_degrees"])) <= 1.0e-9
        and str(node["direction"]) == "+x"
        and str(node["calibration_policy_id"]).startswith("flat_official_")
    ]
    if len(anchor_nodes) < 2:
        raise ValueError(
            "anchor_nominal must contain the immutable official flat +x speed endpoints"
        )
    slopes = tuple(float(value) for value in args.slopes_degrees)
    directions = tuple(_normalize_direction(value) for value in args.directions)
    if len(set(directions)) != len(directions):
        raise ValueError("collector directions must be unique")
    assignments = _build_node_assignments(
        slopes, directions, tuple(args.speeds), args.envs_per_node
    )
    node_keys = sorted(set(_node_key(item) for item in assignments))
    env_ids_by_node = {key: [] for key in node_keys}
    for env_id, assignment in enumerate(assignments):
        env_ids_by_node[_node_key(assignment)].append(env_id)

    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_cfg.scene.num_envs = len(assignments)
    env_cfg.scene.seed = args.seed
    env_cfg.scene.max_episode_length_s = 1000.0
    env_cfg.scene.terrain_type = "generator"
    env_cfg.scene.terrain_generator = make_plane_recovery_terrain_cfg(slopes)
    env_cfg.scene.max_init_terrain_level = 0
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    _disable_randomization(env_cfg)
    if hasattr(args, "device"):
        env_cfg.device = args.device
        agent_cfg.device = args.device

    env = task_registry.get_task_class(args.task)(env_cfg, args.headless)
    _attach_plane_provider(env, slopes)
    # Terrain types are contiguous blocks; assignments were constructed in the
    # same slope-major order.  Refuse to collect if that invariant is broken.
    actual_slopes = torch.tensor(slopes, device=env.device)[env.scene.terrain.terrain_types]
    expected_slopes = torch.tensor(
        [item["slope_degrees"] for item in assignments], device=env.device
    )
    if not torch.equal(actual_slopes, expected_slopes):
        raise RuntimeError("terrain type ordering does not match nominal node assignments")
    assignment_alpha = torch.deg2rad(expected_slopes)
    terrain_normal_world_by_env = torch.stack(
        (
            -torch.sin(assignment_alpha),
            torch.zeros_like(assignment_alpha),
            torch.cos(assignment_alpha),
        ),
        dim=-1,
    )

    runner_class_name = str(agent_cfg.runner_class_name)
    if runner_class_name == "DWAQOnPolicyRunner":
        runner_class = DWAQOnPolicyRunner
        runner_kind = "dwaq"
    elif runner_class_name == "OnPolicyRunner":
        runner_class = OnPolicyRunner
        runner_kind = "on_policy"
    else:
        raise ValueError(
            f"plane nominal collector does not support runner {runner_class_name!r}"
        )
    runner = runner_class(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(str(checkpoint), load_optimizer=False)
    runner.eval_mode()
    policy = runner.alg.policy
    policy.requires_grad_(False)
    if policy.training or any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("calibration policy must be frozen in eval mode")
    inference_policy = (
        policy.act_inference
        if runner_kind == "dwaq"
        else runner.get_inference_policy(device=agent_cfg.device)
    )
    extractor = G1PrivilegedStateExtractor(
        env,
        G1StateExtractorCfg(
            h_eff=None,
            use_terrain_plane_geometry=True,
            slope_alignment_tolerance=math.radians(5.0),
        ),
    )
    foot_ids, _ = env.robot.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )
    illegal_contact_ids, _ = env.contact_sensor.find_bodies(
        "(?!.*ankle_roll.*).*"
    )

    samples = {key: [] for key in node_keys}
    candidate_cycles = {key: 0 for key in node_keys}
    resets = {key: 0 for key in node_keys}
    segment_resets = {key: 0 for key in node_keys}
    segment_counts = {key: len(env_ids_by_node[key]) for key in node_keys}
    node_completed_policy_step = {key: None for key in node_keys}
    rejected = {
        key: {
            "warmup": 0,
            "invalid_transition_start": 0,
            "non_alternating": 0,
            "slip": 0,
            "illegal_contact": 0,
            "invalid_height": 0,
            "invalid_plane": 0,
        }
        for key in node_keys
    }
    touchdown_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    slip_since_touchdown = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    illegal_since_touchdown = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    previous_touchdown = [None] * env.num_envs
    interval_velocity_errors = [[] for _ in range(env.num_envs)]
    interval_actual_vx = [[] for _ in range(env.num_envs)]
    interval_actual_vy = [[] for _ in range(env.num_envs)]
    interval_roll = [[] for _ in range(env.num_envs)]
    interval_pitch = [[] for _ in range(env.num_envs)]
    collection_done = [False] * env.num_envs
    segment_reset_pending = [False] * env.num_envs
    plane_valid_since_touchdown = torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    heading_valid_since_touchdown = torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    heading_diagnostic_records: list[dict] = []
    slip_diagnostic_records = {key: [] for key in node_keys}
    slip_interval_frames = [[] for _ in range(env.num_envs)]
    pending_slip_outcome = [None] * env.num_envs
    slip_collection_done = [False] * env.num_envs
    natural_resets = {key: 0 for key in node_keys}
    heading_applicability_exits = {key: 0 for key in node_keys}
    previous_contacts_for_slip = torch.zeros(
        (env.num_envs, 2), dtype=torch.bool, device=env.device
    )
    contact_age_frames = torch.zeros(
        (env.num_envs, 2), dtype=torch.long, device=env.device
    )
    consecutive_slip_frames = torch.zeros(
        (env.num_envs, 2), dtype=torch.long, device=env.device
    )
    slip_grace_frames = int(math.ceil(args.slip_grace_time_s / float(env.step_dt)))

    _set_commands(env, assignments)
    obs, observation_aux = env.get_observations()
    actor_history_length = int(env.cfg.robot.actor_obs_history_length)
    actor_observation_dim = int(obs.shape[1])
    if args.expected_actor_history_length is not None and (
        actor_history_length != args.expected_actor_history_length
    ):
        raise RuntimeError(
            f"actor history mismatch: expected {args.expected_actor_history_length}, "
            f"got {actor_history_length}"
        )
    if args.expected_actor_frame_dim is not None:
        expected_total = actor_history_length * args.expected_actor_frame_dim
        if actor_observation_dim != expected_total:
            raise RuntimeError(
                f"actor observation mismatch: expected {actor_history_length}x"
                f"{args.expected_actor_frame_dim}={expected_total}, got {actor_observation_dim}"
            )
    for policy_step in range(args.max_steps):
        _set_commands(env, assignments)
        with torch.inference_mode():
            if runner_kind == "dwaq":
                actions = inference_policy(obs, observation_aux)
            else:
                actions = inference_policy(obs)
            obs, _, dones, extras = env.step(actions)
            if runner_kind == "dwaq":
                observation_aux = extras["observations"]["obs_hist"]
            state = extractor.extract()

        foot_velocity_w = env.robot.data.body_lin_vel_w[:, foot_ids, :3]
        feet_speed = torch.linalg.vector_norm(foot_velocity_w[:, :, :2], dim=-1)
        reset_now = dones | state.episode_reset
        previous_contacts_for_slip[reset_now] = False
        contact_age_frames[reset_now] = 0
        consecutive_slip_frames[reset_now] = 0
        contact_rising = state.contacts & ~previous_contacts_for_slip
        liftoff_now = previous_contacts_for_slip & ~state.contacts
        contact_age_frames = torch.where(
            state.contacts,
            contact_age_frames + 1,
            torch.zeros_like(contact_age_frames),
        )
        contact_age_frames[contact_rising] = 1
        grace_contact = state.contacts & (contact_age_frames <= slip_grace_frames)
        settled_contact = state.contacts & (contact_age_frames > slip_grace_frames)
        settled_threshold_exceeded = settled_contact & (
            feet_speed > args.slip_velocity_threshold
        )
        consecutive_slip_frames = torch.where(
            settled_threshold_exceeded,
            consecutive_slip_frames + 1,
            torch.zeros_like(consecutive_slip_frames),
        )
        slip_now = torch.any(
            consecutive_slip_frames >= args.slip_consecutive_frames, dim=1
        )
        previous_contacts_for_slip.copy_(state.contacts)
        contact_foot_speed = torch.where(
            state.contacts, feet_speed, torch.zeros_like(feet_speed)
        )
        slip_metric = torch.max(contact_foot_speed, dim=1).values
        transient_metric = torch.max(
            torch.where(grace_contact, feet_speed, torch.zeros_like(feet_speed)), dim=1
        ).values
        settled_metric = torch.max(
            torch.where(settled_contact, feet_speed, torch.zeros_like(feet_speed)), dim=1
        ).values
        normal_for_feet = terrain_normal_world_by_env.unsqueeze(1)
        tangent_velocity = foot_velocity_w - torch.sum(
            foot_velocity_w * normal_for_feet, dim=-1, keepdim=True
        ) * normal_for_feet
        tangential_speed = torch.linalg.vector_norm(tangent_velocity, dim=-1)
        tangential_metric = torch.max(
            torch.where(
                state.contacts, tangential_speed, torch.zeros_like(tangential_speed)
            ),
            dim=1,
        )
        tangential_metric = tangential_metric.values
        illegal_force = env.contact_sensor.data.net_forces_w[:, illegal_contact_ids]
        illegal_now = torch.any(
            torch.linalg.vector_norm(illegal_force, dim=-1)
            > args.illegal_contact_force_threshold,
            dim=1,
        )
        frame_velocity_error, _ = practical_frame_errors(
            state.com_velocity[:, :2],
            state.command_velocity[:, :2],
            state.root_roll_pitch,
            torch.zeros_like(state.root_roll_pitch),
        )
        slip_since_touchdown |= slip_now
        illegal_since_touchdown |= illegal_now
        plane_valid_since_touchdown &= state.terrain_plane_valid
        normal_h = state.terrain_normal_heading
        near_flat = torch.linalg.vector_norm(normal_h[:, :2], dim=1) < math.sin(
            math.radians(0.5)
        )
        alignment = torch.atan2(torch.abs(normal_h[:, 1]), torch.abs(normal_h[:, 0]))
        heading_valid_now = near_flat | (
            alignment <= extractor.cfg.slope_alignment_tolerance
        )
        heading_valid_since_touchdown &= heading_valid_now

        if (
            args.heading_alignment_diagnostic
            and policy_step % args.heading_diagnostic_stride == 0
            and len(heading_diagnostic_records) < args.heading_diagnostic_max_records
        ):
            robot_yaw = _yaw_from_wxyz(state.heading_quat_w)
            slope_table = torch.tensor(slopes, dtype=normal_h.dtype, device=env.device)
            terrain_types = env.scene.terrain.terrain_types
            alpha = torch.deg2rad(slope_table[terrain_types])
            normal_world = torch.stack(
                (-torch.sin(alpha), torch.zeros_like(alpha), torch.cos(alpha)), dim=-1
            )
            slope_yaw_world = _slope_uphill_yaw_world(normal_world)
            expected_alignment = _axis_alignment_angle(robot_yaw, slope_yaw_world)
            remaining = args.heading_diagnostic_max_records - len(
                heading_diagnostic_records
            )
            for env_id in range(min(env.num_envs, remaining)):
                heading_diagnostic_records.append(
                    {
                        "policy_step": policy_step + 1,
                        "env_id": env_id,
                        "slope_degrees": float(assignments[env_id]["slope_degrees"]),
                        "direction": str(assignments[env_id]["direction"]),
                        "speed": float(assignments[env_id]["speed"]),
                        "robot_yaw_degrees": math.degrees(
                            float(robot_yaw[env_id].item())
                        ),
                        "slope_uphill_yaw_world_degrees": math.degrees(
                            float(slope_yaw_world[env_id].item())
                        ),
                        "alignment_angle_degrees": math.degrees(
                            float(alignment[env_id].item())
                        ),
                        "expected_alignment_degrees": math.degrees(
                            float(expected_alignment[env_id].item())
                        ),
                        "terrain_normal_heading": normal_h[env_id]
                        .detach()
                        .cpu()
                        .tolist(),
                        "terrain_plane_valid": bool(
                            state.terrain_plane_valid[env_id].item()
                        ),
                        "reject_reason": _heading_frame_reject_reason(
                            normal_h[env_id],
                            bool(state.terrain_plane_valid[env_id].item()),
                            extractor.cfg.slope_alignment_tolerance,
                        ),
                    }
                )

        if args.segment_reset_on_invalid_plane:
            invalid_ids = (~state.terrain_plane_valid).nonzero(
                as_tuple=False
            ).flatten().tolist()
            for env_id in invalid_ids:
                if (
                    (collection_done[env_id] and not args.slip_diagnostic)
                    or slip_collection_done[env_id]
                    or segment_reset_pending[env_id]
                ):
                    continue
                key = _node_key(assignments[env_id])
                heading_applicability_exits[key] += int(
                    not bool(heading_valid_now[env_id].item())
                )
                if (
                    previous_touchdown[env_id] is not None
                    and touchdown_count[env_id] > args.warmup_touchdowns
                ):
                    candidate_cycles[key] += 1
                    rejected[key]["invalid_plane"] += 1
                    rejected[key].setdefault("heading_misalignment", 0)
                    rejected[key]["heading_misalignment"] += int(
                        not bool(heading_valid_now[env_id].item())
                    )
                previous_touchdown[env_id] = None
                interval_velocity_errors[env_id] = []
                interval_actual_vx[env_id] = []
                interval_actual_vy[env_id] = []
                interval_roll[env_id] = []
                interval_pitch[env_id] = []
                slip_interval_frames[env_id] = []
                segment_reset_pending[env_id] = True
                _request_segment_reset(env, env_id)

        reset_ids = (dones | state.episode_reset).nonzero(as_tuple=False).flatten().tolist()
        for env_id in reset_ids:
            if (collection_done[env_id] and not args.slip_diagnostic) or slip_collection_done[env_id]:
                continue
            key = _node_key(assignments[env_id])
            resets[key] += 1
            was_segment_reset = segment_reset_pending[env_id]
            if was_segment_reset:
                segment_resets[key] += 1
                segment_reset_pending[env_id] = False
            else:
                natural_resets[key] += 1
            pending = pending_slip_outcome[env_id]
            if pending is not None:
                pending["next_alternating_touchdown_completed"] = False
                pending["segment_reset_before_next_touchdown"] = bool(
                    was_segment_reset
                )
                pending["natural_reset_or_fall_before_next_touchdown"] = bool(
                    not was_segment_reset
                )
                pending_slip_outcome[env_id] = None
            segment_counts[key] += 1
            touchdown_count[env_id] = 0
            slip_since_touchdown[env_id] = False
            illegal_since_touchdown[env_id] = False
            previous_touchdown[env_id] = None
            interval_velocity_errors[env_id] = []
            interval_actual_vx[env_id] = []
            interval_actual_vy[env_id] = []
            interval_roll[env_id] = []
            interval_pitch[env_id] = []
            slip_interval_frames[env_id] = []
            plane_valid_since_touchdown[env_id] = True
            heading_valid_since_touchdown[env_id] = True

        touchdown_ids = state.touchdown.nonzero(as_tuple=False).flatten().tolist()
        for env_id in touchdown_ids:
            key = _node_key(assignments[env_id])
            if (
                (collection_done[env_id] and not args.slip_diagnostic)
                or slip_collection_done[env_id]
                or segment_reset_pending[env_id]
            ):
                continue
            touchdown_count[env_id] += 1
            current_foot = int(state.touchdown_foot[env_id].item())
            current_support = "left" if bool(state.support_is_left[env_id].item()) else "right"
            support_position = (
                state.left_foot_position[env_id, :2]
                if current_support == "left"
                else state.right_foot_position[env_id, :2]
            )
            current_plane_valid = bool(state.terrain_plane_valid[env_id].item())
            current_height_valid = bool(
                torch.isfinite(state.com_height[env_id]).item()
                and state.com_height[env_id].item() >= extractor.cfg.minimum_com_height
            )
            interval_slip = bool(slip_since_touchdown[env_id].item())
            interval_illegal_contact = bool(illegal_since_touchdown[env_id].item())
            interval_plane_valid = bool(plane_valid_since_touchdown[env_id].item())
            interval_heading_valid = bool(
                heading_valid_since_touchdown[env_id].item()
            )
            current_slip = bool(slip_now[env_id].item())
            current_illegal_contact = bool(illegal_now[env_id].item())
            slip_frame = {
                "metric": float(slip_metric[env_id].item()),
                "transient_metric": float(transient_metric[env_id].item()),
                "settled_metric": float(settled_metric[env_id].item()),
                "settled_threshold_exceeded": bool(
                    torch.any(settled_threshold_exceeded[env_id]).item()
                ),
                "consecutive_exceedance_frames": int(
                    torch.max(consecutive_slip_frames[env_id]).item()
                ),
                "severe_slip": bool(slip_now[env_id].item()),
                "tangential_metric": float(tangential_metric[env_id].item()),
                "actual_vx": float(state.com_velocity[env_id, 0].item()),
                "touchdown": True,
                "liftoff": bool(torch.any(liftoff_now[env_id]).item()),
                "phase": float(state.phase[env_id].item()),
                "terrain_plane_valid": current_plane_valid,
                "heading_error_degrees": math.degrees(
                    float(alignment[env_id].item())
                ),
            }
            completed_slip_frames = slip_interval_frames[env_id] + [slip_frame]
            slip_interval_frames[env_id] = [slip_frame]

            pending = pending_slip_outcome[env_id]
            if pending is not None:
                pending["next_alternating_touchdown_completed"] = bool(
                    current_foot != pending["ending_touchdown_foot"]
                )
                pending["segment_reset_before_next_touchdown"] = False
                pending["natural_reset_or_fall_before_next_touchdown"] = False
                pending_slip_outcome[env_id] = None
            slip_since_touchdown[env_id] = current_slip
            illegal_since_touchdown[env_id] = current_illegal_contact
            plane_valid_since_touchdown[env_id] = current_plane_valid
            heading_valid_since_touchdown[env_id] = bool(
                heading_valid_now[env_id].item()
            )
            current = {
                "time": float(state.time[env_id].item()),
                "foot": current_foot,
                "support_side": current_support,
                "support_position_H": support_position.detach().cpu().numpy(),
                "q_start_H": state.q[env_id].detach().cpu().numpy(),
                # A rejected touchdown must not become the hidden start of the
                # next accepted swing interval.
                "valid_transition_start": (
                    current_plane_valid
                    and current_height_valid
                    and not current_slip
                    and not current_illegal_contact
                ),
                "plane_valid": current_plane_valid,
                "height_valid": current_height_valid,
                "slip": current_slip,
                "illegal_contact": current_illegal_contact,
            }
            previous = previous_touchdown[env_id]
            previous_touchdown[env_id] = current
            completed_velocity_errors = interval_velocity_errors[env_id]
            completed_actual_vx = interval_actual_vx[env_id]
            completed_actual_vy = interval_actual_vy[env_id]
            completed_roll = interval_roll[env_id]
            completed_pitch = interval_pitch[env_id]
            interval_velocity_errors[env_id] = []
            interval_actual_vx[env_id] = []
            interval_actual_vy[env_id] = []
            interval_roll[env_id] = []
            interval_pitch[env_id] = []
            if previous is None or touchdown_count[env_id] <= args.warmup_touchdowns:
                rejected[key]["warmup"] += 1
                continue
            candidate_cycles[key] += 1
            collector_reject_reason = _collector_cycle_reject_reason(
                interval_plane_valid=interval_plane_valid,
                interval_heading_valid=interval_heading_valid,
                previous=previous,
                current_foot=current_foot,
                current_plane_valid=current_plane_valid,
                current_height_valid=current_height_valid,
                interval_slip=interval_slip,
                interval_illegal_contact=interval_illegal_contact,
                has_interval_frames=bool(completed_velocity_errors),
            )
            if (
                args.slip_diagnostic
                and len(slip_diagnostic_records[key])
                < args.slip_diagnostic_cycles_per_node
            ):
                record = _slip_cycle_record(
                    cycle_index=len(slip_diagnostic_records[key]) + 1,
                    key=key,
                    env_id=env_id,
                    previous=previous,
                    current_foot=current_foot,
                    current_support=current_support,
                    frames=completed_slip_frames,
                    threshold=args.slip_velocity_threshold,
                    step_dt=float(env.step_dt),
                    collector_reject_reason=collector_reject_reason,
                    interval_plane_valid=interval_plane_valid,
                    maximum_heading_error_degrees=max(
                        frame["heading_error_degrees"]
                        for frame in completed_slip_frames
                    ),
                    step_period_s=float(state.step_period[env_id].item()),
                    measured_landing_xy=(-state.q[env_id])
                    .detach()
                    .cpu()
                    .tolist(),
                    command_vx=float(assignments[env_id]["command"][0]),
                )
                slip_diagnostic_records[key].append(record)
                if record["slip_gate_rejected"]:
                    pending_slip_outcome[env_id] = record
            if not interval_plane_valid:
                rejected[key]["invalid_plane"] += 1
                rejected[key].setdefault("heading_misalignment", 0)
                rejected[key]["heading_misalignment"] += int(
                    not interval_heading_valid
                )
                continue
            if not previous["valid_transition_start"]:
                if not previous["plane_valid"]:
                    rejected[key]["invalid_plane"] += 1
                    rejected[key].setdefault("heading_misalignment", 0)
                    rejected[key]["heading_misalignment"] += int(
                        not interval_heading_valid
                    )
                elif not previous["height_valid"]:
                    rejected[key]["invalid_height"] += 1
                elif previous["slip"]:
                    rejected[key]["slip"] += 1
                elif previous["illegal_contact"]:
                    rejected[key]["illegal_contact"] += 1
                else:
                    rejected[key]["invalid_transition_start"] += 1
                continue
            if previous["foot"] == current_foot:
                rejected[key]["non_alternating"] += 1
                continue
            if not current_plane_valid:
                rejected[key]["invalid_plane"] += 1
                continue
            if not current_height_valid:
                rejected[key]["invalid_height"] += 1
                continue
            if interval_slip:
                rejected[key]["slip"] += 1
                continue
            if interval_illegal_contact:
                rejected[key]["illegal_contact"] += 1
                continue
            if not completed_velocity_errors:
                rejected[key]["invalid_transition_start"] += 1
                continue
            l_h = -state.q[env_id].detach().cpu().numpy()
            command = np.asarray(assignments[env_id]["command"], dtype=np.float64)
            samples[key].append(
                {
                    "T": float(state.step_period[env_id].item()),
                    "h_geom": float(state.com_height[env_id].item()),
                    "command": command[:2].tolist(),
                    "support_side": current_support,
                    "transition_support": previous["support_side"],
                    "com_position_H": state.com_position[env_id, :2].detach().cpu().tolist(),
                    "com_velocity_H": state.com_velocity[env_id, :2].detach().cpu().tolist(),
                    "support_position_H": support_position.detach().cpu().tolist(),
                    "q_H": state.q[env_id].detach().cpu().tolist(),
                    "q_start_H": np.asarray(previous["q_start_H"]).tolist(),
                    "l_H": l_h.tolist(),
                    "interval_velocity_error": completed_velocity_errors,
                    "interval_actual_vx": completed_actual_vx,
                    "interval_actual_vy": completed_actual_vy,
                    "interval_roll": completed_roll,
                    "interval_pitch": completed_pitch,
                }
            )
            if not args.slip_diagnostic and _node_collection_complete(
                samples[key], candidate_cycles[key], args.samples_per_node
            ):
                node_completed_policy_step[key] = policy_step + 1
                mark_collection_done(
                    env_ids_by_node[key],
                    collection_done,
                    previous_touchdown,
                    interval_velocity_errors,
                    interval_actual_vx,
                    interval_actual_vy,
                    interval_roll,
                    interval_pitch,
                )

        # Match runtime ordering: a touchdown closes the previous interval,
        # resets its sums, then the current policy frame starts the next one.
        for env_id, touchdown in enumerate(previous_touchdown):
            if (
                (collection_done[env_id] and not args.slip_diagnostic)
                or slip_collection_done[env_id]
                or touchdown is None
            ):
                continue
            interval_velocity_errors[env_id].append(
                float(frame_velocity_error[env_id].item())
            )
            interval_actual_vx[env_id].append(
                float(state.com_velocity[env_id, 0].item())
            )
            interval_actual_vy[env_id].append(
                float(state.com_velocity[env_id, 1].item())
            )
            interval_roll[env_id].append(
                float(state.root_roll_pitch[env_id, 0].item())
            )
            interval_pitch[env_id].append(
                float(state.root_roll_pitch[env_id, 1].item())
            )
            if env_id not in touchdown_ids:
                slip_interval_frames[env_id].append(
                    {
                        "metric": float(slip_metric[env_id].item()),
                        "transient_metric": float(
                            transient_metric[env_id].item()
                        ),
                        "settled_metric": float(settled_metric[env_id].item()),
                        "settled_threshold_exceeded": bool(
                            torch.any(settled_threshold_exceeded[env_id]).item()
                        ),
                        "consecutive_exceedance_frames": int(
                            torch.max(consecutive_slip_frames[env_id]).item()
                        ),
                        "severe_slip": bool(slip_now[env_id].item()),
                        "tangential_metric": float(
                            tangential_metric[env_id].item()
                        ),
                        "actual_vx": float(
                            state.com_velocity[env_id, 0].item()
                        ),
                        "touchdown": False,
                        "liftoff": bool(torch.any(liftoff_now[env_id]).item()),
                        "phase": float(state.phase[env_id].item()),
                        "terrain_plane_valid": bool(
                            state.terrain_plane_valid[env_id].item()
                        ),
                        "heading_error_degrees": math.degrees(
                            float(alignment[env_id].item())
                        ),
                    }
                )

        if args.slip_diagnostic:
            for key in node_keys:
                if (
                    len(slip_diagnostic_records[key])
                    >= args.slip_diagnostic_cycles_per_node
                    and all(
                        pending_slip_outcome[env_id] is None
                        for env_id in env_ids_by_node[key]
                    )
                ):
                    for env_id in env_ids_by_node[key]:
                        slip_collection_done[env_id] = True
                        previous_touchdown[env_id] = None
                        interval_velocity_errors[env_id] = []
                        interval_actual_vx[env_id] = []
                        interval_actual_vy[env_id] = []
                        interval_roll[env_id] = []
                        interval_pitch[env_id] = []
                        slip_interval_frames[env_id] = []

        if (policy_step + 1) % 250 == 0:
            active_records = slip_diagnostic_records if args.slip_diagnostic else samples
            minimum = min(len(rows) for rows in active_records.values())
            done_flags = slip_collection_done if args.slip_diagnostic else collection_done
            completed = sum(
                all(done_flags[index] for index in env_ids_by_node[key])
                for key in node_keys
            )
            print(
                f"[plane-nominal] step={policy_step + 1}/{args.max_steps} "
                f"nodes={completed}/{len(node_keys)} min_samples={minimum}",
                flush=True,
            )
        if all(slip_collection_done if args.slip_diagnostic else collection_done):
            break

    if args.slip_diagnostic:
        slip_nodes = {}
        for key in node_keys:
            label = f"alpha={key[0]:+g},direction={key[1]},speed={key[2]:g}"
            slip_nodes[label] = {
                "summary": _slip_node_summary(
                    slip_diagnostic_records[key], natural_resets[key]
                ),
                "cycles": slip_diagnostic_records[key],
            }
        output_document = {
            "schema_version": 1,
            "diagnostic_type": "existing_nominal_slip_gate",
            "diagnostic_only": True,
            "checkpoint": str(checkpoint),
            "metric_semantics": (
                "Production slip uses horizontal world-frame linear speed of each "
                "ankle-roll body while its contact force exceeds 5 N. Contact-rise "
                "grace frames are ignored, then the threshold must be exceeded for "
                "the configured number of consecutive settled-contact frames."
            ),
            "slip_threshold_m_per_s": args.slip_velocity_threshold,
            "contact_force_threshold_n": extractor.cfg.contact_force_threshold,
            "policy_step_dt_s": float(env.step_dt),
            "slip_grace_time_s": args.slip_grace_time_s,
            "slip_grace_frames": slip_grace_frames,
            "slip_consecutive_frames": args.slip_consecutive_frames,
            "tangential_speed_semantics": (
                "Diagnostic only: norm of v-(v dot n)n for contacted ankle-roll "
                "bodies; it does not participate in production rejection."
            ),
            "event_location_semantics": (
                "An exceedance is touchdown_near or liftoff_near when within two "
                "policy frames of the corresponding measured contact edge; all "
                "other exceedances are support_mid_phase."
            ),
            "target_complete_candidate_cycles_per_node": args.slip_diagnostic_cycles_per_node,
            "policy_steps": policy_step + 1,
            "nodes": slip_nodes,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            yaml.safe_dump(output_document, sort_keys=False), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "policy_steps": policy_step + 1,
                    "complete_candidate_cycles": {
                        label: node["summary"]["complete_candidate_cycle_count"]
                        for label, node in slip_nodes.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        simulation_app.close()
        return

    rng = np.random.default_rng(args.seed)
    collected_nodes = []
    node_reports = {}
    policy_id = _checkpoint_id(checkpoint, args.task)
    for key in node_keys:
        rows = samples[key]
        rng.shuffle(rows)
        label = f"alpha={key[0]:+g},direction={key[1]},speed={key[2]:g}"
        # The official flat +x calibration is immutable and remains the
        # production anchor over its whole speed interval.  DWAQ flat +x data
        # are comparison-only, including speeds not explicitly stored in the
        # anchor table, so interpolation can never mix two policy identities.
        anchor_preserved = bool(
            args.preserve_official_flat_anchor
            and abs(key[0]) <= 1.0e-9
            and key[1] == "+x"
        )
        acceptance_ratio = _acceptance_ratio(len(rows), candidate_cycles[key])
        common_report = {
            "valid_cycle_count": len(rows),
            "total_candidate_cycles": candidate_cycles[key],
            "candidate_cycle_count": candidate_cycles[key],
            "accepted_cycle_count": len(rows),
            "rejected_cycle_count": max(candidate_cycles[key] - len(rows), 0),
            "acceptance_ratio": acceptance_ratio,
            "policy_steps_to_complete": node_completed_policy_step[key],
            "segment_count": segment_counts[key],
            "reset_count": resets[key],
            "natural_reset_count": natural_resets[key],
            "reset_rate": resets[key] / max(candidate_cycles[key], 1),
            "segment_reset_count": segment_resets[key],
            "heading_applicability_exit_count": heading_applicability_exits[key],
            "invalid_plane_cycle_count": rejected[key]["invalid_plane"],
            "heading_misalignment_cycle_count": rejected[key].get(
                "heading_misalignment", 0
            ),
            "slip_rejection_count": rejected[key]["slip"],
            "severe_slip_rejection_count": rejected[key]["slip"],
            "illegal_contact_rejection_count": rejected[key]["illegal_contact"],
        }
        if len(rows) < args.samples_per_node:
            node_reports[label] = {
                "valid": False,
                "reason": "insufficient_valid_cycles",
                "reason_detail": "did not reach the requested strict valid cycle count",
                "sample_count": len(rows),
                "required_sample_count": args.samples_per_node,
                "reset_count": resets[key],
                "rejected": rejected[key],
                **common_report,
            }
            continue
        try:
            node, diagnostic = calibrate_nominal_node(
                rows,
                flat_parameters,
                slope_degrees=key[0],
                direction=key[1],
                speed=key[2],
                calibration_policy_id=policy_id,
                terminal_epsilon_semantics=args.terminal_epsilon_semantics,
            )
        except ValueError as exc:
            node_reports[label] = {
                "valid": False,
                "reason": str(exc),
                "sample_count": len(rows),
                "reset_count": resets[key],
                "rejected": rejected[key],
                **common_report,
            }
            continue
        node_reports[label] = {
            "valid": True,
            "anchor_preserved": anchor_preserved,
            "node": node,
            "diagnostic": diagnostic,
            "reset_count": resets[key],
            "rejected": rejected[key],
            **common_report,
        }
        if not anchor_preserved:
            collected_nodes.append(node)

    if args.preserve_official_flat_anchor:
        merged_nodes = upsert_nominal_nodes(existing_nodes, collected_nodes)
        merged_node_reports = dict(previous_node_reports)
        merged_node_reports.update(node_reports)
    else:
        # A validation-only table must be closed over this exact collection
        # grid. Retaining unrelated legacy slope/speed nodes can make a small
        # heading-induced alpha change interpolate through incompatible data.
        merged_nodes = list(collected_nodes)
        merged_node_reports = dict(node_reports)
    output_document = {
        "schema_version": 1,
        "description": "Frozen-policy continuous-plane nominal calibration candidate; C/L/v_max remain in the separate flat capability file.",
        "calibration_semantics_version": "plane_nominal_frozen_policy_per_axis_p95_v2",
        "teacher": {
            "task": args.task,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _checkpoint_hash(checkpoint),
            "checkpoint_iteration": int(
                torch.load(checkpoint, map_location="cpu", weights_only=False).get(
                    "iter", -1
                )
            ),
            "runner_class": runner_class_name,
            "policy_class": type(policy).__name__,
            "eval_mode": not policy.training,
            "requires_grad": any(
                parameter.requires_grad for parameter in policy.parameters()
            ),
            "actor_history_length": actor_history_length,
            "actor_frame_dimension": (
                actor_observation_dim // actor_history_length
                if actor_observation_dim % actor_history_length == 0
                else None
            ),
            "actor_observation_dimension": actor_observation_dim,
        },
        "git_commit": _git_commit(),
        "flat_capability_path": os.path.relpath(
            flat_path, args.output.expanduser().resolve().parent
        ),
        "nominal_plane_gait": {
            "nodes": merged_nodes,
        },
        "collection": {
            "checkpoint": str(checkpoint),
            "calibration_policy_id": policy_id,
            "slopes_degrees": list(slopes),
            "directions": list(directions),
            "speeds": list(args.speeds),
            "samples_per_node": args.samples_per_node,
            "node_pass_semantics": "strict_valid_cycle_count_at_least_required",
            "acceptance_ratio_semantics": "diagnostic_only",
            "terminal_epsilon_semantics": args.terminal_epsilon_semantics,
            "segment_reset_on_invalid_plane": args.segment_reset_on_invalid_plane,
            "preserve_official_flat_anchor": args.preserve_official_flat_anchor,
            "max_resets_per_node": args.max_resets_per_node,
            "slip_velocity_threshold": args.slip_velocity_threshold,
            "slip_grace_time_s": args.slip_grace_time_s,
            "slip_grace_frames": slip_grace_frames,
            "slip_consecutive_frames": args.slip_consecutive_frames,
            "illegal_contact_force_threshold": args.illegal_contact_force_threshold,
            "policy_steps": policy_step + 1,
            "node_reports": merged_node_reports,
            "invalid_nodes": [
                {
                    "node": label,
                    "reason": report.get("reason", "unspecified"),
                    "reason_detail": report.get("reason_detail"),
                    "valid_cycle_count": report.get("valid_cycle_count", 0),
                }
                for label, report in node_reports.items()
                if not report["valid"]
            ],
        },
    }
    if args.heading_alignment_diagnostic:
        output_document["heading_alignment_diagnostic"] = {
            "semantics": (
                "Independent expected alignment is the smallest axis angle "
                "between robot world yaw and the world uphill/downhill slope axis."
            ),
            "tolerance_degrees": math.degrees(
                extractor.cfg.slope_alignment_tolerance
            ),
            "stride_policy_steps": args.heading_diagnostic_stride,
            "max_records": args.heading_diagnostic_max_records,
            "summary": _heading_diagnostic_summary(heading_diagnostic_records),
            "records": heading_diagnostic_records,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(output_document, sort_keys=False), encoding="utf-8")
    summary = {
        "output": str(output),
        "policy_steps": policy_step + 1,
        "valid_nodes": sum(report["valid"] for report in node_reports.values()),
        "written_dwaq_nodes": len(collected_nodes),
        "missing_nodes": [name for name, report in node_reports.items() if not report["valid"]],
        "preserved_flat_anchor_nodes": [
            f"alpha={float(node['slope_degrees']):+g},direction={node['direction']},speed={float(node['speed']):g}"
            for node in anchor_nodes
            if args.preserve_official_flat_anchor
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        try:
            simulation_app.close()
        finally:
            os._exit(1)
