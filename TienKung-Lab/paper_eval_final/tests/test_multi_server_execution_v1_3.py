from pathlib import Path

import yaml

from paper_eval_final.src.common import ROOT, protocol_bundle
from paper_eval_final.src.trial_generator import generate_trials


def _addendum() -> dict:
    path = Path(ROOT) / "protocol/multi_server_execution_v1_3.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_multiserver_seed_contract_matches_trial_generator():
    addendum = _addendum()
    protocol, _, identity = protocol_bundle()
    assert addendum["protocol_hash"] == identity["protocol_hash"]
    assert addendum["seed_contract"]["namespaces"] == {
        "screening": 1_301_000,
        "pilot": 1_302_000,
        "formal": 1_303_000,
    }
    for experiment in (1, 2, 3):
        first = generate_trials("formal", experiment, protocol)[0]
        expected = 1_303_000 + experiment * 1_000_000
        assert first["eval_seed"] == expected
        assert addendum["seed_contract"]["formal_first_eval_seed"][f"experiment_{experiment}"] == expected


def test_each_available_model_has_exactly_one_server_owner():
    assignments = _addendum()["ownership"]["assignments"]
    owners = {
        model: server
        for server, assignment in assignments.items()
        for model in assignment["models"]
    }
    assert owners == {"M1": "server_a", "M2": "server_b", "M5": "server_c"}
    assert len(owners) == sum(len(value["models"]) for value in assignments.values())


def test_machine_identity_never_changes_eval_seed():
    contract = _addendum()["seed_contract"]
    assert contract["worker_id_participates"] is False
    assert contract["hostname_participates"] is False
    assert contract["gpu_id_participates"] is False
    assert contract["server_specific_offset_prohibited"] is True
    assert contract["paired_across_models"] is True
