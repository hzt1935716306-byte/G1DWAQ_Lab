"""Budget selection and scheduling contracts; no simulator needed."""
import ast
from collections import Counter
import copy
import random
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_reduced_budget import select, score, execution_rows, TARGET_COUNTS, BASE_COMMIT
from g1_robustness_protocol import protocol, generate_plan
from g1_reduced_compatibility import expected_queue_source
from g1_audit_compatibility import source_at
from g1_recovery_protocol import LAB


@pytest.fixture(scope='module')
def full():
    p,_=protocol();return generate_plan(p)


def test_exact_reduced_quotas_and_unchanged_plans(full):
    rows,coverage=select(full);old={r['trial_id']:r for r in full}
    assert len(rows)==1960 and dict(Counter(r['suite'] for r in rows))==TARGET_COUNTS
    assert all(r==old[r['trial_id']] for r in rows)
    assert sum(r['suite']=='extreme_main' and r['slope_deg']==0 for r in rows)==520
    assert sum(r['suite']=='extreme_main' and r['slope_deg']!=0 for r in rows)==672
    assert all(x['n']==20 for x in coverage if not x['suite'].startswith('extreme'))


def test_selection_independent_of_order_and_fake_outcomes(full):
    selected,_=select(full)
    changed=copy.deepcopy(full);random.Random(123).shuffle(changed)
    for k,r in enumerate(changed):r.update(outcome='FELL' if k%2 else 'SUCCESS',model='arbitrary',completed=bool(k%3))
    again,_=select(changed)
    assert [r['trial_id'] for r in again]==[r['trial_id'] for r in selected]


def test_direction_and_foot_phase_quotas(full):
    rows,audit=select(full)
    for c in audit:
        if c['suite'] in ('force_pulse','constant_force','wrench_pulse'):
            assert sorted(c['directions'].values())==[2]*4+[3]*4
    for c in audit:
        if c['suite']=='extreme_ood':
            selected=[r for r in rows if r['condition_id']==c['condition_id']]
            assert len({r['reference_touchdown_foot'] for r in selected})==2
            assert len({r['target_phase'] for r in selected})==2


def test_random_conditions_use_exact_hash_without_waveform_reselection(full):
    rows,_=select(full)
    for family in ('velocity_ood','random_force','repeated_impulse'):
        for cid in {r['condition_id'] for r in full if r['suite']==family}:
            expected=sorted([r for r in full if r['condition_id']==cid],key=score)[:20]
            assert {r['trial_id'] for r in rows if r['condition_id']==cid}=={r['trial_id'] for r in expected}


def test_final_batch_only_real_targets_are_active(full):
    plans=full[:65]
    first,n=execution_rows(plans,0,64);last,k=execution_rows(plans,64,64)
    assert len(first)==len(last)==64 and n==64 and k==1
    assert first[:n]+last[:k]==plans
    assert len({r['trial_id'] for r in first[:n]+last[:k]})==65
    with pytest.raises(ValueError):execution_rows(plans,128,64)


def test_physical_step_loop_exactly_unchanged():
    name='tools/evaluation/g1_complete_robustness.py'
    old=ast.parse(source_at(BASE_COMMIT,name));new=ast.parse((LAB/name).read_text())
    find=lambda t:next(n for n in ast.walk(t) if isinstance(n,ast.While))
    assert ast.dump(find(old),include_attributes=False)==ast.dump(find(new),include_attributes=False)


def test_queue_only_proof_rejects_a_physics_change():
    name='tools/evaluation/g1_complete_robustness.py';old=source_at(BASE_COMMIT,name)
    expected=expected_queue_source(name,old);current=(LAB/name).read_text()
    assert current==expected
    assert current.replace('env.step(actions)','env.step(actions * 0.9)')!=expected


def test_stop_request_is_after_committed_progress_and_before_completion():
    source=(LAB/'tools/evaluation/g1_complete_robustness.py').read_text()
    assert source.index("write_json(run/'progress.json'")<source.index("root/'STOP_AFTER_BATCH'")<source.index('store.complete(workers=8)')


def test_pending_queue_cannot_include_a_reused_trial(tmp_path,monkeypatch):
    import g1_reduced_budget as budget
    from g1_recovery_protocol import write_json,atomic_write,jsonl,digest,sha256
    plans=[{'trial_id':'a'},{'trial_id':'b'}];source=tmp_path/'source';master=tmp_path/'master'
    atomic_write(source/'manifest.jsonl',jsonl(plans))
    write_json(master/'reduced_budget_manifest_v1.json',dict(protocol_hash=digest({}),source_root=str(source),
        source_manifest_sha256=sha256(source/'manifest.jsonl'),trials=plans))
    write_json(master/'pending_execution_queues.json',{'model':['a','b']})
    write_json(master/'reduced_budget_reuse_index.json',dict(entries=[
        {'model':'model','target_trial_id':'a','reuse_decision':'PENDING_EXECUTION'},
        {'model':'model','target_trial_id':'b','reuse_decision':'REUSE_EXISTING'}]))
    monkeypatch.setattr(budget,'select',lambda _: (plans,[]))
    info=dict(reduced_budget_root=str(master),reduced_model='model',execution_batch_size=64,
        reduced_manifest_sha256=sha256(master/'reduced_budget_manifest_v1.json'),
        execution_queue_sha256=sha256(master/'pending_execution_queues.json'),reuse_index_sha256=sha256(master/'reduced_budget_reuse_index.json'))
    with pytest.raises(ValueError,match='reused or unassigned'):
        budget.validate_pending(tmp_path,{},plans,info)
    write_json(master/'pending_execution_queues.json',{'model':['a']})
    with pytest.raises(ValueError,match='Frozen budget queue'):
        budget.validate_pending(tmp_path,{},plans[:1],info)
    info['execution_queue_sha256']=sha256(master/'pending_execution_queues.json')
    budget.validate_pending(tmp_path,{},plans[:1],info)
