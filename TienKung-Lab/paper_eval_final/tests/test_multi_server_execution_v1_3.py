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


def test_protocol_leaves_model_to_server_assignment_to_user():
    ownership = _addendum()["ownership"]
    assert ownership["assignment_policy"] == "user_selected_before_execution"
    assert ownership["protocol_assigns_models_to_servers"] is False
    assert ownership["server_count"] == 3
    assert ownership["servers_interchangeable"] is True
    assert ownership["assignments"] == {}
    assert ownership["single_writer_required"] is True
    assert _addendum()["current_resume_point"]["owner_selected_by_user"] is True


def test_user_may_dispatch_any_protocol_task_in_natural_language():
    dispatch = _addendum()["task_dispatch"]
    assert dispatch["mode"] == "user_defined_natural_language"
    assert dispatch["protocol_prescribes_task_menu"] is False
    assert dispatch["server_identity_required"] is False
    assert dispatch["user_may_select"] == [
        "any_registered_frozen_model",
        "any_protocol_stage",
        "any_protocol_experiment_or_combination",
    ]
    assert dispatch["fail_closed_rule"].startswith("reject_requests_that_conflict")


def test_machine_identity_never_changes_eval_seed():
    contract = _addendum()["seed_contract"]
    assert contract["worker_id_participates"] is False
    assert contract["hostname_participates"] is False
    assert contract["gpu_id_participates"] is False
    assert contract["server_specific_offset_prohibited"] is True
    assert contract["paired_across_models"] is True
