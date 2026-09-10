"""Strict config/provenance audit for Plane V3 and DWAQ V4."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from reward_shaping_contract import LAB, canonical


PLANE_V2 = "g1_plane_v1_estimator_context_no_reward_matched_v2"
PLANE_V3 = "g1_plane_v1_estimator_context_no_reward_matched_v3"
DWAQ = "g1_dwaq_slope_nosys_d_matched"
DWAQ_V2 = f"{DWAQ}_v2"
DWAQ_V3 = f"{DWAQ}_v3"
DWAQ_V4 = f"{DWAQ}_v4"

ESTIMATOR = (
    LAB
    / "logs/g1_com_velocity_estimator/v2_iteration_long_5000_random_init_fixed"
    / "com_velocity_estimator_v2_long_best.pt"
)
ESTIMATOR_SHA256 = "8574d845f28cbc7908e437250de46ba865898056483741e8fb72a81281f9e319"
PARENT_CHECKPOINT_SHA256 = "9bde392d2a91ef7e20bfd0b6e84625e2663caf205577f02f0b720f1830a879c1"
PARENT_CHECKPOINT_ITERATION = 9999


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _active_rewards(reward_cfg) -> dict:
    reward = canonical(reward_cfg)
    return {
        name: term
        for name, term in reward.items()
        if isinstance(term, dict) and term.get("weight", 0.0) != 0.0
    }


def _only_reward_terms_differ(parent_env, child_env, expected: set[str]) -> dict:
    parent = canonical(parent_env)
    child = canonical(child_env)
    parent_rewards = _active_rewards(parent["reward"])
    child_rewards = _active_rewards(child["reward"])
    changed = {
        name
        for name in parent_rewards.keys() | child_rewards.keys()
        if parent_rewards.get(name) != child_rewards.get(name)
    }
    if changed != expected:
        raise ValueError(f"Unexpected reward diff: expected={expected}, actual={changed}")
    if {key: value for key, value in parent["reward"].items() if key not in expected} != {
        key: value for key, value in child["reward"].items() if key not in expected
    }:
        raise ValueError("Inactive or unrelated reward configuration changed")
    if {key: value for key, value in parent.items() if key != "reward"} != {
        key: value for key, value in child.items() if key != "reward"
    }:
        raise ValueError("Non-reward environment configuration changed")
    return {
        name: {"before": parent["reward"].get(name), "after": child["reward"].get(name)}
        for name in sorted(changed)
    }


def _agent_diff(parent_agent, child_agent, allowed: set[str]) -> dict:
    parent = canonical(parent_agent)
    child = canonical(child_agent)
    changed = {
        key for key in parent.keys() | child.keys() if parent.get(key) != child.get(key)
    }
    unexpected = changed - allowed
    if unexpected:
        raise ValueError(f"Unexpected agent/network/training diff: {unexpected}")
    return {key: {"before": parent.get(key), "after": child.get(key)} for key in sorted(changed)}


def locate_formal_plane_v2_checkpoint() -> dict:
    """Select by embedded provenance/budget; fail instead of guessing on ambiguity."""

    root = LAB / "logs" / PLANE_V2
    candidates = []
    for path in sorted(root.glob("*/model_*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        provenance = checkpoint.get("training_provenance") or {}
        if provenance.get("task_name") != PLANE_V2:
            continue
        candidates.append(
            {
                "path": path.resolve(),
                "iteration": int(checkpoint["iter"]),
                "training_transitions": int(provenance.get("training_transitions", -1)),
                "training_run_id": provenance.get("training_run_id"),
                "optimizer_state_count": len(checkpoint["optimizer_state_dict"]["state"]),
            }
        )
    if not candidates:
        raise FileNotFoundError(f"No formal {PLANE_V2} checkpoint found below {root}")
    maximum = max(item["training_transitions"] for item in candidates)
    finalists = [item for item in candidates if item["training_transitions"] == maximum]
    if len(finalists) != 1:
        paths = [str(item["path"]) for item in finalists]
        raise RuntimeError(f"Ambiguous final Plane V2 checkpoints: {paths}")
    selected = finalists[0]
    selected["sha256"] = file_sha256(selected["path"])
    selected["source_run"] = selected["path"].parent.name
    return selected


def audit_registry(registry, *, verify_files: bool = True) -> dict:
    plane_v2_env, plane_v2_agent = registry.get_cfgs(PLANE_V2)
    plane_v3_env, plane_v3_agent = registry.get_cfgs(PLANE_V3)
    dwaq_env, dwaq_agent = registry.get_cfgs(DWAQ)
    dwaq_v2_env, dwaq_v2_agent = registry.get_cfgs(DWAQ_V2)
    dwaq_v3_env, dwaq_v3_agent = registry.get_cfgs(DWAQ_V3)
    dwaq_v4_env, dwaq_v4_agent = registry.get_cfgs(DWAQ_V4)

    plane_reward_diff = _only_reward_terms_differ(plane_v2_env, plane_v3_env, {"idle_penalty"})
    plane_agent_diff = _agent_diff(
        plane_v2_agent,
        plane_v3_agent,
        {
            "experiment_name",
            "wandb_project",
            "run_name",
            "resume",
            "max_iterations",
            "parent_task",
            "parent_checkpoint_sha256",
            "parent_checkpoint_iteration",
            "fine_tune_iterations",
            "resume_iteration_is_last_completed",
        },
    )
    assert plane_v2_env.reward.idle_penalty.weight == -0.1
    assert plane_v3_env.reward.idle_penalty.weight == -2.0
    for env in (plane_v2_env, plane_v3_env):
        assert env.reward.idle_penalty.params == {"cmd_threshold": 0.2, "vel_threshold": 0.1}
        assert env.reward.feet_swing_height.weight == -0.2
        assert env.reward.feet_swing_height.params["target_height"] == 0.08
        assert env.com_velocity_source == "estimator"
        assert env.recovery_context.enabled and env.recovery_context.mode == "certificate"
        assert not env.plane_v1_reward.enabled
    assert plane_v3_agent.resume and plane_v3_agent.max_iterations == 2000
    assert plane_v3_agent.parent_checkpoint_sha256 == PARENT_CHECKPOINT_SHA256
    assert plane_v3_agent.parent_checkpoint_iteration == PARENT_CHECKPOINT_ITERATION

    original_v4 = _only_reward_terms_differ(dwaq_env, dwaq_v4_env, {"idle_penalty", "feet_swing_height"})
    v2_v4 = _only_reward_terms_differ(dwaq_v2_env, dwaq_v4_env, {"feet_swing_height"})
    v3_v4 = _only_reward_terms_differ(dwaq_v3_env, dwaq_v4_env, {"idle_penalty"})
    _agent_diff(
        dwaq_agent,
        dwaq_v4_agent,
        {"experiment_name", "wandb_project", "run_name"},
    )
    assert dwaq_v4_env.reward.idle_penalty is None
    assert dwaq_v4_env.reward.feet_swing_height is None
    assert dwaq_v4_env.reward.gait_phase_contact is None
    assert dwaq_v4_agent.resume is False
    assert dwaq_v4_agent.seed == 42
    assert dwaq_v4_agent.num_steps_per_env == 24
    assert dwaq_v4_agent.max_iterations == 10000
    assert dwaq_v4_env.scene.num_envs == 4096
    assert registry.get_task_class(DWAQ_V4) is registry.get_task_class(DWAQ)
    assert registry.get_task_class(PLANE_V3) is registry.get_task_class(PLANE_V2)

    parent = None
    if verify_files:
        if file_sha256(ESTIMATOR) != ESTIMATOR_SHA256:
            raise ValueError("Estimator SHA256 mismatch")
        parent = locate_formal_plane_v2_checkpoint()
        if parent["sha256"] != PARENT_CHECKPOINT_SHA256:
            raise ValueError(f"Plane V2 parent SHA256 mismatch: {parent['sha256']}")
        if parent["iteration"] != PARENT_CHECKPOINT_ITERATION:
            raise ValueError(f"Plane V2 parent iteration mismatch: {parent['iteration']}")
        if parent["optimizer_state_count"] == 0:
            raise ValueError("Plane V2 parent has no optimizer state")

    return {
        "status": "PASS",
        "plane_v2_to_v3": {
            "reward_diff": plane_reward_diff,
            "agent_diff": plane_agent_diff,
        },
        "dwaq_original_to_v4": original_v4,
        "dwaq_v2_to_v4": v2_v4,
        "dwaq_v3_to_v4": v3_v4,
        "parent": parent,
        "estimator_sha256": ESTIMATOR_SHA256,
    }
