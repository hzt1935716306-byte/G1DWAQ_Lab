"""Unit checks for strict V3/V4 diff and resume contracts."""

import copy
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/recovery"))
from reward_shaping_v3_v4_contract import _agent_diff, _only_reward_terms_differ


class Config:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return copy.deepcopy(self.value)


def env(idle=-0.1, swing=-0.2):
    return Config(
        {
            "terrain": {"curriculum": True},
            "reward": {
                "idle_penalty": None if idle is None else {"weight": idle, "params": {"threshold": 0.2}},
                "feet_swing_height": None if swing is None else {"weight": swing, "params": {"height": 0.08}},
                "alive": {"weight": 0.15},
            },
        }
    )


def test_plane_v3_accepts_only_idle_weight_change():
    diff = _only_reward_terms_differ(env(), env(idle=-2.0), {"idle_penalty"})
    assert set(diff) == {"idle_penalty"}


@pytest.mark.parametrize("confound", ["terrain", "swing", "alive"])
def test_plane_v3_rejects_other_environment_changes(confound):
    parent = env()
    child = env(idle=-2.0)
    if confound == "terrain":
        child.value["terrain"]["curriculum"] = False
    elif confound == "swing":
        child.value["reward"]["feet_swing_height"]["weight"] = -0.3
    else:
        child.value["reward"]["alive"]["weight"] = 0.2
    with pytest.raises(ValueError):
        _only_reward_terms_differ(parent, child, {"idle_penalty"})


def test_dwaq_v4_two_by_two_diffs():
    original = env(idle=-2.0, swing=-0.2)
    v2 = env(idle=None, swing=-0.2)
    v3 = env(idle=-2.0, swing=None)
    v4 = env(idle=None, swing=None)
    assert set(_only_reward_terms_differ(original, v4, {"idle_penalty", "feet_swing_height"})) == {
        "idle_penalty",
        "feet_swing_height",
    }
    assert set(_only_reward_terms_differ(v2, v4, {"feet_swing_height"})) == {"feet_swing_height"}
    assert set(_only_reward_terms_differ(v3, v4, {"idle_penalty"})) == {"idle_penalty"}


def test_agent_diff_rejects_network_or_ppo_changes():
    parent = Config({"run_name": "v2", "algorithm": {"lr": 1e-3}, "actor_hidden_dims": [512, 256, 128]})
    child = Config({"run_name": "v3", "algorithm": {"lr": 2e-3}, "actor_hidden_dims": [512, 256, 128]})
    with pytest.raises(ValueError):
        _agent_diff(parent, child, {"run_name"})


def test_agent_diff_accepts_only_declared_metadata():
    parent = Config({"run_name": "v2", "resume": False, "algorithm": {"lr": 1e-3}})
    child = Config({"run_name": "v3", "resume": True, "algorithm": {"lr": 1e-3}})
    assert set(_agent_diff(parent, child, {"run_name", "resume"})) == {"run_name", "resume"}

