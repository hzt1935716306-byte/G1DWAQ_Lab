"""Reward ablations retain the native DWAQ inference family and distinct tasks."""
import pytest
from g1_recovery_protocol import TASKS, method_for, evaluation_key

@pytest.mark.parametrize('key', ['dwaq', 'dwaq_no_idle', 'dwaq_no_swing'])
def test_native_family(key):
    assert method_for(TASKS[key]) == 'dwaq'

def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match='Unsupported task'):
        method_for('g1_dwaq_slope_nosys_d_matched_v4')

def test_task_identities_remain_distinct():
    assert len({TASKS[k] for k in ('dwaq','dwaq_no_idle','dwaq_no_swing')}) == 3

@pytest.mark.parametrize('rows', [
    [{'trial_id':'a','status':'FELL'}],
    [{'trial_id':'a','status':'FELL'},{'trial_id':'a','status':'FELL'}],
    [{'trial_id':'a','status':'FELL'},{'trial_id':'b','status':'EVALUATION_ERROR'}],
])
def test_report_rejects_missing_duplicate_or_error_trials(rows):
    from g1_dwaq_ablation_report import paired_records
    with pytest.raises(ValueError):paired_records(rows,[{'trial_id':'a'},{'trial_id':'b'}])

def test_report_keeps_legal_failures():
    from g1_dwaq_ablation_report import paired_records
    paired_records([{'trial_id':'a','status':'PRECONDITION_FAILED'},{'trial_id':'b','status':'FELL'}],
                   [{'trial_id':'a'},{'trial_id':'b'}])
