"""Reward ablations retain the native DWAQ inference family and distinct tasks."""
import pytest
from g1_recovery_protocol import TASKS, method_for, evaluation_key

@pytest.mark.parametrize('key', ['dwaq', 'dwaq_no_idle', 'dwaq_no_swing', 'dwaq_no_idle_no_swing'])
def test_native_family(key):
    assert method_for(TASKS[key]) == 'dwaq'

def test_task_identities_remain_distinct():
    assert len({TASKS[k] for k in ('dwaq','dwaq_no_idle','dwaq_no_swing','dwaq_no_idle_no_swing')}) == 4

def test_context_v2_preserves_native_family_and_distinct_task():
    assert method_for(TASKS['context_only_v2']) == 'context_only'
    assert TASKS['context_only_v2'] != TASKS['context_only']

def test_context_v3_preserves_native_family_and_distinct_task():
    assert method_for(TASKS['context_only_v3']) == 'context_only'
    assert TASKS['context_only_v3'] != TASKS['context_only_v2']

def test_context_reward_v2_preserves_native_family_and_distinct_task():
    assert method_for(TASKS['context_reward_v2']) == 'context_reward'
    assert TASKS['context_reward_v2'] != TASKS['context_reward']

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

def test_new_reduced_evaluation_contains_only_requested_variants():
    from g1_new_variants_reduced_eval import MODELS
    assert set(MODELS) == {'dwaq_v4', 'context_only_v3', 'context_reward_v2'}
    assert MODELS['dwaq_v4']['native_model'] == 'dwaq'
    assert MODELS['context_only_v3']['checkpoint'].endswith('/model_11999.pt')
    assert MODELS['context_reward_v2']['native_model'] == 'context_reward'
