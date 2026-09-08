"""Pure CPU checks; never starts simulator or performance process."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_batch_profile_control import compare
from g1_robustness_protocol import protocol,generate_plan

def test_fixed_2048_full_batches_no_padding():
    p,_=protocol();full=generate_plan(p)
    blocks=np.linspace(0,len(full)//64-1,32,dtype=int)
    plans=[r for b in blocks for r in full[b*64:(b+1)*64]]
    assert len(plans)==len({r['trial_id'] for r in plans})==2048
    assert all(len(plans)%n==0 for n in [64,128,256,512,1024,2048])
    assert {r['suite'] for r in plans}=={r['suite'] for r in full}

def fixture(root,margin=1.,touch=False):
    for sub in ['trial_records','traces']:(root/sub).mkdir(parents=True)
    fields=['status','push_applied','precondition_passed','push_start_time','push_end_time','disturbance_start_time','recovery_clock_start_time','survived','recovered_sustained_and_survived','recovery_time','recovery_steps','first_recovery_entry','first_confirmation','sustained_recovery_entry','relapse_count','failure_category','common_gate_failure_counts','common_complete_windows','common_task_gate_pass_windows','readiness_hold_confirmed','first_readiness_confirmation_time','first_post_push_sample_time','sustained_confirmation','physical_touchdown_count','post_push_event_counts']
    record={k:None for k in fields};record.update(trial_id='x',status='RECOVERED_AND_SURVIVED',recovery_time=1.,recovery_steps=2)
    (root/'trial_records/x.json').write_text(json.dumps(record))
    d=dict(time=np.array([0.,.02]),physical_touchdown_flags=np.array([[False,False],[touch,False]]),alternating_touchdown_flag=np.zeros(2,dtype=bool),physical_contact=np.ones((2,2),dtype=bool),post_push_sample=np.ones(2,dtype=bool),plane_json=np.array([json.dumps(dict(N=1,margin=margin,context_valid=True,query_failed=False,query_category='success'))]*2))
    np.savez(root/'traces/x.npz',**d)

def test_same_trace_passes(tmp_path):
    fixture(tmp_path/'a');fixture(tmp_path/'b')
    assert compare(tmp_path/'a',tmp_path/'b',[{'trial_id':'x'}],'context_only')['passed']

def test_touchdown_difference_fails(tmp_path):
    fixture(tmp_path/'a');fixture(tmp_path/'b',touch=True)
    r=compare(tmp_path/'a',tmp_path/'b',[{'trial_id':'x'}],'ppo_plain')
    assert not r['passed'] and r['field_counts']['physical_touchdown_flags']==1

def test_native_margin_difference_fails(tmp_path):
    fixture(tmp_path/'a');fixture(tmp_path/'b',margin=.8)
    r=compare(tmp_path/'a',tmp_path/'b',[{'trial_id':'x'}],'context_only')
    assert not r['passed'] and r['field_counts']['certificate_margin']==1

def test_missing_required_outcome_fails(tmp_path):
    fixture(tmp_path/'a');fixture(tmp_path/'b')
    for name in ['a','b']:
        p=tmp_path/name/'trial_records/x.json';r=json.loads(p.read_text());del r['recovery_steps'];p.write_text(json.dumps(r))
    assert not compare(tmp_path/'a',tmp_path/'b',[{'trial_id':'x'}],'ppo_plain')['passed']
