"""Strict Stage1A 960-D actor to Stage2 context-actor warm start."""

from __future__ import annotations

from pathlib import Path

import torch


STAGE1A_ACTOR_OBS_DIM = 960
CONTEXT_ACTOR_OBS_DIM = 963
ACTOR_FIRST_LAYER_KEY = "actor.0.weight"


def migrate_stage1a_model_state_dict(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Expand only the actor's first input layer and reject every other mismatch."""

    source_keys = set(source_state)
    target_keys = set(target_state)
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise RuntimeError(
            f"Stage1A checkpoint key mismatch: missing={missing}, unexpected={unexpected}"
        )
    if ACTOR_FIRST_LAYER_KEY not in source_state:
        raise RuntimeError(f"Stage1A checkpoint is missing {ACTOR_FIRST_LAYER_KEY!r}")

    migrated: dict[str, torch.Tensor] = {}
    for key, target_value in target_state.items():
        source_value = source_state[key]
        if key == ACTOR_FIRST_LAYER_KEY:
            expected_source = (target_value.shape[0], STAGE1A_ACTOR_OBS_DIM)
            expected_target = (target_value.shape[0], CONTEXT_ACTOR_OBS_DIM)
            if tuple(source_value.shape) != expected_source or tuple(target_value.shape) != expected_target:
                raise RuntimeError(
                    "Unexpected actor first-layer migration shapes: "
                    f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}, "
                    f"expected_source={expected_source}, expected_target={expected_target}"
                )
            expanded = torch.zeros_like(target_value)
            expanded[:, :STAGE1A_ACTOR_OBS_DIM].copy_(
                source_value.to(device=expanded.device, dtype=expanded.dtype)
            )
            migrated[key] = expanded
            continue
        if source_value.shape != target_value.shape:
            raise RuntimeError(
                f"Unexpected Stage1A checkpoint shape mismatch for {key}: "
                f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
            )
        migrated[key] = source_value.to(device=target_value.device, dtype=target_value.dtype)
    return migrated


def warm_start_context_policy(policy, checkpoint_path: str | Path) -> dict[str, object]:
    """Load Stage1A model weights strictly while leaving the Stage2 optimizer fresh."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
    migrated = migrate_stage1a_model_state_dict(
        checkpoint["model_state_dict"],
        policy.state_dict(),
    )
    policy.load_state_dict(migrated, strict=True)
    first_layer = policy.state_dict()[ACTOR_FIRST_LAYER_KEY]
    if torch.count_nonzero(first_layer[:, STAGE1A_ACTOR_OBS_DIM:]).item() != 0:
        raise RuntimeError("The three new actor context columns were not initialized to zero")
    return {
        "checkpoint": str(checkpoint_path),
        "source_iteration": int(checkpoint.get("iter", -1)),
        "actor_input_before": STAGE1A_ACTOR_OBS_DIM,
        "actor_input_after": CONTEXT_ACTOR_OBS_DIM,
        "optimizer_loaded": False,
    }
