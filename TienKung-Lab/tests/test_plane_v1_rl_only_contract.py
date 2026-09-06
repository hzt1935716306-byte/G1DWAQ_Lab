"""Simulator-independent checks for the Plane V1 RL-only ablation contract."""

from __future__ import annotations

from pathlib import Path

from legged_lab.recovery.rl_only_matched_contract import unexpected_config_differences


def test_config_diff_rejects_unexpected_training_protocol_change() -> None:
    reference = {"reward": {"tracking": 2.0}, "actor": {"history": 5, "context": 3}}
    rl_only = {"reward": {"tracking": 1.0}, "actor": {"history": 5}}
    differences = unexpected_config_differences(
        reference,
        rl_only,
        allowed_prefixes=("actor.context",),
    )
    assert [(item.path, item.reference, item.rl_only) for item in differences] == [
        ("reward.tracking", 2.0, 1.0)
    ]


def test_rl_only_task_uses_baseenv_matched_path_and_is_registered() -> None:
    root = Path(__file__).resolve().parents[1]
    config = (root / "legged_lab/envs/g1/g1_plane_v1_rl_only_config.py").read_text()
    registry = (root / "legged_lab/envs/__init__.py").read_text()
    assert "class G1PlaneV1RLOnlyMatchedEnvCfg(G1SlopeSysDMatchedEnvCfg)" in config
    assert "self.robot.actor_obs_history_length = 5" in config
    assert "self.robot.critic_obs_history_length = 10" in config
    assert '"g1_plane_v1_rl_only_matched"' in registry
    assert "G1SlopeBaselineMatchedEnv" in registry
