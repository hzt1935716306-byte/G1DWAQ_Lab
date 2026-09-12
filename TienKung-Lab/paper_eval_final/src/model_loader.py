"""Explicit model registry, checkpoint identity, and native inference loading."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .common import LAB, ROOT, load_yaml, sha256_file


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    method: str
    task_name: str
    actor_certificate_context: tuple[str, ...]
    train_seed: int
    checkpoint_path: Path
    checkpoint_sha256: str
    training_commit: str
    estimator_path: Path | None
    estimator_sha256: str | None
    formal_status: str


def registry() -> dict[str, Any]:
    return load_yaml(ROOT / "configs/models.yaml")


def resolve_model(model_id: str, train_seed: int | None = None) -> ModelIdentity:
    models = registry()["models"]
    if model_id not in models:
        raise ValueError(f"Unknown model ID: {model_id}")
    model = models[model_id]
    checkpoints = model.get("checkpoints", [])
    if not checkpoints:
        raise ValueError(f"{model_id} has no registered checkpoint ({model['formal_status']})")
    selected = [item for item in checkpoints if train_seed is None or item["train_seed"] == train_seed]
    if len(selected) != 1:
        raise ValueError(f"{model_id}: expected exactly one checkpoint for requested train seed")
    item = selected[0]
    checkpoint = (LAB / item["checkpoint_path"]).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != item["checkpoint_sha256"]:
        raise ValueError(f"{model_id}: checkpoint missing or SHA-256 mismatch")
    estimator = (LAB / model["estimator_path"]).resolve() if model.get("estimator_path") else None
    if estimator is not None and (not estimator.is_file() or sha256_file(estimator) != model["estimator_sha256"]):
        raise ValueError(f"{model_id}: estimator missing or SHA-256 mismatch")
    return ModelIdentity(
        model_id=model_id, method=model["method"], task_name=model["task_name"],
        actor_certificate_context=tuple(model["actor_certificate_context"]),
        train_seed=int(item["train_seed"]), checkpoint_path=checkpoint,
        checkpoint_sha256=item["checkpoint_sha256"], training_commit=item["training_commit"],
        estimator_path=estimator, estimator_sha256=model.get("estimator_sha256"),
        formal_status=model["formal_status"],
    )


def resolve_baseline(baseline_id: str, train_seed: int | None = None) -> ModelIdentity:
    baselines = registry()["screening_baselines"]
    if baseline_id not in baselines:
        raise ValueError(f"Unknown baseline ID: {baseline_id}")
    row = baselines[baseline_id]
    if not row.get("model"):
        raise ValueError(f"Baseline {baseline_id}: {row.get('status', 'not registered')}")
    identity = resolve_model(row["model"], train_seed)
    if identity.actor_certificate_context:
        raise ValueError("Experiment 1 baseline must not receive certificate context")
    return identity


def validate_context_ablation_configs(task_registry: Any) -> None:
    from legged_lab.recovery.context_ablation_contract import unexpected_config_differences

    ids = ("M3", "M4", "M5")
    entries = registry()["models"]
    configs = {model_id: copy.deepcopy(task_registry.get_cfgs(entries[model_id]["task_name"])[0]) for model_id in ids}
    expected = {
        "M3": ("margin_norm", "valid"), "M4": ("n_norm", "valid"),
        "M5": ("n_norm", "margin_norm", "valid"),
    }
    reference = configs["M5"].to_dict()
    for model_id, cfg in configs.items():
        if tuple(getattr(cfg, "actor_recovery_context_fields", ("n_norm", "margin_norm", "valid"))) != expected[model_id]:
            raise ValueError(f"{model_id}: actor-visible context is not the registered ablation")
        if cfg.reward.idle_penalty.weight != -0.1 or cfg.reward.feet_swing_height.weight != -0.2:
            raise ValueError(f"{model_id}: idle/swing reward contract mismatch")
        if cfg.plane_v1_reward.enabled is not False:
            raise ValueError(f"{model_id}: plane_v1_reward must be disabled")
        differences = unexpected_config_differences(
            reference, cfg.to_dict(), allowed_paths=("actor_recovery_context_fields",)
        )
        if differences:
            raise ValueError(f"{model_id}: unexpected configuration differences: {differences[:3]}")


def dwaq_history_from_step_payload(payload: Any) -> Any:
    """Normalize both registered DWAQ environment return contracts.

    ``G1DWAQEnv.get_observations()`` returns ``(actor_obs, obs_hist)`` during
    initial reset, while ``step()`` returns the usual extras mapping containing
    ``observations.obs_hist``. Recovery-context DWAQ tasks use the mapping in
    both places. Refuse any third shape instead of guessing.
    """
    history = payload
    if isinstance(payload, dict):
        observations = payload.get("observations")
        if isinstance(observations, dict):
            history = observations.get("obs_hist")
    if history is None or getattr(history, "ndim", None) != 2:
        raise ValueError("unsupported DWAQ observation-history contract")
    return history


def load_native_policy(env: Any, agent_cfg: Any, identity: ModelIdentity, device: str) -> tuple[Any, Callable]:
    import torch
    from rsl_rl.runners import DWAQOnPolicyRunner, OnPolicyRunner

    runner_cls = DWAQOnPolicyRunner if agent_cfg.runner_class_name == "DWAQOnPolicyRunner" else OnPolicyRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=device)
    payload = torch.load(identity.checkpoint_path, map_location=device, weights_only=False)
    runner.alg.policy.load_state_dict(payload["model_state_dict"], strict=True)
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(payload["obs_norm_state_dict"], strict=True)
        runner.privileged_obs_normalizer.load_state_dict(payload["privileged_obs_norm_state_dict"], strict=True)
    runner.eval_mode()
    modules = (runner.alg.policy, runner.obs_normalizer, runner.privileged_obs_normalizer)
    for module in modules:
        module.eval()
        module.requires_grad_(False)
    if agent_cfg.runner_class_name == "DWAQOnPolicyRunner":
        # DWAQ's nominally deterministic inference path still samples its VAE
        # latent.  Generate that Gaussian noise from one CPU stream per trial,
        # in blocks, so the result is invariant to GPU batch size/resume while
        # keeping encoder/actor inference vectorized.
        stream_trial_ids: tuple[str, ...] = ()
        streams: list[Any] = []
        noise_block: Any | None = None
        noise_cursor = 0
        block_size = 256

        def policy(observation: Any, extras: dict[str, Any]) -> Any:
            nonlocal stream_trial_ids, streams, noise_block, noise_cursor
            trial_ids = tuple(str(plan["trial_id"]) for plan in env.eval_plans)
            if trial_ids != stream_trial_ids:
                stream_trial_ids = trial_ids
                streams = [torch.Generator(device="cpu").manual_seed(int(plan["eval_seed"])) for plan in env.eval_plans]
                noise_block = None
                noise_cursor = 0
            module = runner.alg.policy
            history = dwaq_history_from_step_payload(extras)
            encoded = module.encoder(history)
            mean_latent = module.encode_mean_latent(encoded)
            logvar_latent = torch.clamp(module.encode_logvar_latent(encoded), min=-10.0, max=10.0)
            mean_vel = module.encode_mean_vel(encoded)
            logvar_vel = torch.clamp(module.encode_logvar_vel(encoded), min=-10.0, max=10.0)
            latent_dim = int(mean_latent.shape[1])
            total_dim = latent_dim + int(mean_vel.shape[1])
            if noise_block is None or noise_cursor >= block_size:
                cpu_rows = [torch.randn((block_size, total_dim), generator=stream) for stream in streams]
                noise_block = torch.stack(cpu_rows, dim=1).to(device=device)
                noise_cursor = 0
            noise = noise_block[noise_cursor]
            noise_cursor += 1
            code_latent = mean_latent + torch.exp(logvar_latent * 0.5) * noise[:, :latent_dim]
            code_vel = mean_vel + torch.exp(logvar_vel * 0.5) * noise[:, latent_dim:]
            return module.actor(torch.cat((code_vel, code_latent, observation), dim=-1))
    else:
        native = runner.get_inference_policy(device=device)
        policy = lambda observation, extras: native(observation)
    return runner, policy
