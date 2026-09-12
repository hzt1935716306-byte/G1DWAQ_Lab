import copy

import pytest

from paper_eval_final.src.common import digest
from paper_eval_final.src.statistics import summarize
from paper_eval_final.src.storage import compatibility_gate
from paper_eval_final.src.trial_generator import equal_effective_deficits, generate_trials, select_stratified_relation_set


def _identity(stage="pilot", system="paper_eval_final"):
    return {
        "evaluation_system": system, "protocol_version": "1.2", "stage": stage,
        "experiment_id": 2, "manifest_hash": "a", "metrics_config_hash": "b",
        "physics_profile_hash": "c", "evaluation_code_commit": "d",
        "asset_hash": "e", "simulator_version": "f",
    }


@pytest.mark.parametrize("field", ["protocol_version", "manifest_hash", "metrics_config_hash", "physics_profile_hash",
                                   "evaluation_code_commit", "asset_hash", "simulator_version"])
def test_mismatch_refuses_merge(field):
    first, second = _identity(), _identity()
    second[field] += "-different"
    with pytest.raises(ValueError, match="REFUSE TO MERGE"):
        compatibility_gate([first, second])


def test_pilot_formal_and_legacy_refuse_merge():
    with pytest.raises(ValueError, match="stage"):
        compatibility_gate([_identity("pilot"), _identity("formal")])
    with pytest.raises(ValueError, match="legacy"):
        compatibility_gate([_identity(), _identity(system="legacy")])


def test_only_eval_invalid_creates_equal_effective_deficit():
    trials = generate_trials("screening", 2)
    records = [{**row, "invalid_kind": None} for row in trials]
    records[0]["failure_reason"] = "PHYSICAL_FAILURE"
    assert sum(equal_effective_deficits(records, trials).values()) == 0
    records[0]["invalid_kind"] = "EVAL_INVALID"
    assert sum(equal_effective_deficits(records, trials).values()) == 1


def test_fixed_budget_and_stratified_relation_set_remain_distinct():
    trials = generate_trials("screening", 1)
    records = []
    for index, row in enumerate(trials):
        records.append({**row, "certificate_valid": True, "Nmin": 3 + index % 3,
                        "stage": "pilot", "margin_raw": index / 100.0,
                        "margin_storage_dtype": "float64", "margin_was_rounded": False,
                        "task_outcome": "SUCCESS" if index % 2 else "FAILURE"})
    selection = select_stratified_relation_set(records, stage="pilot", manifest_seed=7)
    assert selection["dataset_role"] == "PILOT_BOUNDARY_SET"
    assert all("task_outcome" not in value for value in selection["cells"].values())


def test_constant_cell_spearman_is_null_not_nan():
    rows = [{
        "valid": 1, "termination_kind": "HORIZON_REACHED", "certificate_valid": True,
        "Nmin": 3, "nTD0": value, "CERT_AFTER_RECOVERY": False,
        "eval_invalid": 0, "cert_invalid": 0, "failure_reason": None,
        "push_applied": True, "recovered_sustained": True,
        "survived_post_observation": True, "recovery_time": 1.0,
        "Krec": 2, "RMSE_v": 0.1, "RMSE_vx": 0.1, "RMSE_vy": 0.0,
    } for value in (1, 2, 3)]
    relation = summarize(rows, len(rows))["spearman_Nmin_nTD0"]
    assert relation == {"n": 3, "rho": None, "pvalue": None}


def test_two_point_spearman_undefined_pvalue_is_null_not_nan():
    rows = [{
        "valid": 1, "termination_kind": "HORIZON_REACHED", "certificate_valid": True,
        "Nmin": n_min, "nTD0": ntd0, "CERT_AFTER_RECOVERY": False,
        "eval_invalid": 0, "cert_invalid": 0, "failure_reason": None,
        "push_applied": True, "recovered_sustained": True,
        "survived_post_observation": True, "recovery_time": 1.0,
        "Krec": 2, "RMSE_v": 0.1, "RMSE_vx": 0.1, "RMSE_vy": 0.0,
    } for n_min, ntd0 in ((3, 4), (4, 5))]
    relation = summarize(rows, len(rows))["spearman_Nmin_nTD0"]
    assert relation == {"n": 2, "rho": pytest.approx(1.0), "pvalue": None}
