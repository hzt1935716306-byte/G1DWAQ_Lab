#!/usr/bin/env python3
"""Collect nominal plane gait from a DWAQ policy without fitting C/L/v_max."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import MethodType
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
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
parser.add_argument("--max_steps", type=int, default=12000)
parser.add_argument("--slip_velocity_threshold", type=float, default=0.30)
parser.add_argument("--illegal_contact_force_threshold", type=float, default=1.0)
parser.add_argument(
    "--max_resets_per_node",
    type=int,
    default=0,
    help="Mark a nominal node invalid if its assigned environments fall/reset more often.",
)
parser.add_argument("--seed", type=int, default=42)
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
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from rsl_rl.runners import DWAQOnPolicyRunner  # noqa: E402

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


def _checkpoint_id(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"dwaq:{path.name}:{digest}"


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


def main() -> None:
    if args.samples_per_node < 10 or args.envs_per_node <= 0 or args.max_steps <= 0:
        raise ValueError("sample/env/step counts are invalid")
    if args.max_resets_per_node < 0:
        raise ValueError("max_resets_per_node must be non-negative")
    if args.slip_velocity_threshold <= 0.0 or args.illegal_contact_force_threshold <= 0.0:
        raise ValueError("contact and slip thresholds must be positive")
    if any(abs(slope) >= 45.0 for slope in args.slopes_degrees):
        raise ValueError("collector slopes must satisfy abs(alpha) < 45 degrees")
    if any(speed <= 0.0 for speed in args.speeds):
        raise ValueError("collector speeds must be positive")
    if len(set(args.slopes_degrees)) != len(args.slopes_degrees):
        raise ValueError("collector slopes must be unique")
    if len(set(args.speeds)) != len(args.speeds):
        raise ValueError("collector speeds must be unique")

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

    env_cfg, agent_cfg = task_registry.get_cfgs("g1_dwaq")
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

    env = task_registry.get_task_class("g1_dwaq")(env_cfg, args.headless)
    _attach_plane_provider(env, slopes)
    # Terrain types are contiguous blocks; assignments were constructed in the
    # same slope-major order.  Refuse to collect if that invariant is broken.
    actual_slopes = torch.tensor(slopes, device=env.device)[env.scene.terrain.terrain_types]
    expected_slopes = torch.tensor(
        [item["slope_degrees"] for item in assignments], device=env.device
    )
    if not torch.equal(actual_slopes, expected_slopes):
        raise RuntimeError("terrain type ordering does not match nominal node assignments")

    runner = DWAQOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    runner.eval_mode()
    inference_policy = runner.alg.policy.act_inference
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
    resets = {key: 0 for key in node_keys}
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
    interval_roll = [[] for _ in range(env.num_envs)]
    interval_pitch = [[] for _ in range(env.num_envs)]
    collection_done = [False] * env.num_envs

    _set_commands(env, assignments)
    obs, obs_hist = env.get_observations()
    for policy_step in range(args.max_steps):
        _set_commands(env, assignments)
        with torch.inference_mode():
            actions = inference_policy(obs, obs_hist)
            obs, _, dones, extras = env.step(actions)
            obs_hist = extras["observations"]["obs_hist"]
            state = extractor.extract()

        feet_speed = torch.linalg.vector_norm(
            env.robot.data.body_lin_vel_w[:, foot_ids, :2], dim=-1
        )
        slip_now = torch.any(
            state.contacts & (feet_speed > args.slip_velocity_threshold), dim=1
        )
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

        reset_ids = (dones | state.episode_reset).nonzero(as_tuple=False).flatten().tolist()
        for env_id in reset_ids:
            if collection_done[env_id]:
                continue
            key = _node_key(assignments[env_id])
            resets[key] += 1
            touchdown_count[env_id] = 0
            slip_since_touchdown[env_id] = False
            illegal_since_touchdown[env_id] = False
            previous_touchdown[env_id] = None
            interval_velocity_errors[env_id] = []
            interval_roll[env_id] = []
            interval_pitch[env_id] = []

        touchdown_ids = state.touchdown.nonzero(as_tuple=False).flatten().tolist()
        for env_id in touchdown_ids:
            key = _node_key(assignments[env_id])
            if collection_done[env_id] or len(samples[key]) >= args.samples_per_node:
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
            current_slip = bool(slip_now[env_id].item())
            current_illegal_contact = bool(illegal_now[env_id].item())
            slip_since_touchdown[env_id] = current_slip
            illegal_since_touchdown[env_id] = current_illegal_contact
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
            }
            previous = previous_touchdown[env_id]
            previous_touchdown[env_id] = current
            completed_velocity_errors = interval_velocity_errors[env_id]
            completed_roll = interval_roll[env_id]
            completed_pitch = interval_pitch[env_id]
            interval_velocity_errors[env_id] = []
            interval_roll[env_id] = []
            interval_pitch[env_id] = []
            if previous is None or touchdown_count[env_id] <= args.warmup_touchdowns:
                rejected[key]["warmup"] += 1
                continue
            if not previous["valid_transition_start"]:
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
                    "interval_roll": completed_roll,
                    "interval_pitch": completed_pitch,
                }
            )
            if len(samples[key]) >= args.samples_per_node:
                mark_collection_done(
                    env_ids_by_node[key],
                    collection_done,
                    previous_touchdown,
                    interval_velocity_errors,
                    interval_roll,
                    interval_pitch,
                )

        # Match runtime ordering: a touchdown closes the previous interval,
        # resets its sums, then the current policy frame starts the next one.
        for env_id, touchdown in enumerate(previous_touchdown):
            if collection_done[env_id] or touchdown is None:
                continue
            interval_velocity_errors[env_id].append(
                float(frame_velocity_error[env_id].item())
            )
            interval_roll[env_id].append(
                float(state.root_roll_pitch[env_id, 0].item())
            )
            interval_pitch[env_id].append(
                float(state.root_roll_pitch[env_id, 1].item())
            )

        if (policy_step + 1) % 250 == 0:
            minimum = min(len(rows) for rows in samples.values())
            completed = sum(len(rows) >= args.samples_per_node for rows in samples.values())
            print(
                f"[plane-nominal] step={policy_step + 1}/{args.max_steps} "
                f"nodes={completed}/{len(node_keys)} min_samples={minimum}",
                flush=True,
            )
        if all(len(rows) >= args.samples_per_node for rows in samples.values()):
            break

    rng = np.random.default_rng(args.seed)
    collected_nodes = []
    node_reports = {}
    policy_id = _checkpoint_id(checkpoint)
    for key in node_keys:
        rows = samples[key]
        rng.shuffle(rows)
        label = f"alpha={key[0]:+g},direction={key[1]},speed={key[2]:g}"
        # The official flat +x calibration is immutable and remains the
        # production anchor over its whole speed interval.  DWAQ flat +x data
        # are comparison-only, including speeds not explicitly stored in the
        # anchor table, so interpolation can never mix two policy identities.
        anchor_preserved = abs(key[0]) <= 1.0e-9 and key[1] == "+x"
        if len(rows) < args.samples_per_node:
            node_reports[label] = {
                "valid": False,
                "reason": "did not reach the requested stable touchdown sample count",
                "sample_count": len(rows),
                "required_sample_count": args.samples_per_node,
                "reset_count": resets[key],
                "rejected": rejected[key],
            }
            continue
        if resets[key] > args.max_resets_per_node:
            node_reports[label] = {
                "valid": False,
                "reason": "policy was not continuously stable at this nominal node",
                "sample_count": len(rows),
                "reset_count": resets[key],
                "max_resets_per_node": args.max_resets_per_node,
                "rejected": rejected[key],
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
            )
        except ValueError as exc:
            node_reports[label] = {
                "valid": False,
                "reason": str(exc),
                "sample_count": len(rows),
                "reset_count": resets[key],
                "rejected": rejected[key],
            }
            continue
        node_reports[label] = {
            "valid": True,
            "anchor_preserved": anchor_preserved,
            "node": node,
            "diagnostic": diagnostic,
            "reset_count": resets[key],
            "rejected": rejected[key],
        }
        if not anchor_preserved:
            collected_nodes.append(node)

    merged_nodes = upsert_nominal_nodes(existing_nodes, collected_nodes)
    merged_node_reports = dict(previous_node_reports)
    merged_node_reports.update(node_reports)
    output_document = {
        "schema_version": 1,
        "description": "Flat anchors plus validated DWAQ continuous-plane nominal nodes; C/L/v_max remain in the separate flat capability file.",
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
            "max_resets_per_node": args.max_resets_per_node,
            "slip_velocity_threshold": args.slip_velocity_threshold,
            "illegal_contact_force_threshold": args.illegal_contact_force_threshold,
            "policy_steps": policy_step + 1,
            "node_reports": merged_node_reports,
        },
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
