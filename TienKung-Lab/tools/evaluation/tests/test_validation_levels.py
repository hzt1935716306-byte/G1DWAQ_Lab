"""Audit strategy tests: deterministic coverage, explicit replay and immutable evidence."""
import copy,json
from pathlib import Path
import pytest
import g1_run_audit as audit
import g1_recovery_protocol as protocol

def records(n=200):
    return [dict(trial_id=f't{i:04d}',status=audit.MANDATORY[i%5],relapse_count=int(i%19==0),family=f'f{i%6}',push_magnitude=(i%13)/4,slope_deg=[-15,-10,0,10,15][i%5],direction_deg=45*(i%8)) for i in range(n)]

@pytest.mark.parametrize('n,k',[(0,0),(4,4),(100,16),(2048,21),(16128,64)])
def test_sample_size_and_order_independence(n,k):
    rs=records(n);a=audit.select_sample('run',rs)
    assert len(a)==k and a==audit.select_sample('run',list(reversed(rs)))

def test_mandatory_terminal_and_relapse_coverage():
    rs=records();ids=set(audit.select_sample('run',rs));chosen=[r for r in rs if r['trial_id'] in ids]
    assert set(audit.MANDATORY)<={r['status'] for r in chosen}
    assert any(r['relapse_count']>0 for r in chosen)
    assert {r['family'] for r in chosen}=={r['family'] for r in rs}

def common_store(tmp_path):
    from test_common_task_v2 import complete_synthetic
    root=tmp_path/'prepared';info=protocol.prepare(root,protocol.COMMON_PROTOCOL)
    m=complete_synthetic();dest=tmp_path/'run';dest.mkdir()
    identity={**info,'synthetic':True,'manifest_hash':protocol.digest([m.plan]),'evaluation_runtime_sha256':'synthetic_runtime'}
    protocol.write_json(dest/'identity.json',identity)
    for name in ['protocol_snapshot.yaml','common_detector_config.yaml']:
        source='protocol.yaml' if name.startswith('protocol') else name
        protocol.atomic_write(dest/name,(root/source).read_bytes())
    protocol.atomic_write(dest/'manifest_snapshot.jsonl',protocol.jsonl([m.plan]))
    store=protocol.RunStore(dest);store.save_trial(m.result(),m.trace(),m.all_events())
    return store,m

def test_light_never_calls_common_replay(tmp_path,monkeypatch):
    import g1_common_task_trial
    store,_=common_store(tmp_path)
    monkeypatch.setattr(g1_common_task_trial,'replay_common_trial',lambda *a:pytest.fail('LIGHT invoked replay'))
    store.validate(validation_level='light',allow_synthetic=True)
    store.validate(require_complete=False,allow_synthetic=True) # resume default

@pytest.mark.parametrize('mode,expected',[('sampled',16),('full',100)])
def test_levels_replay_only_requested_ids(tmp_path,monkeypatch,mode,expected):
    rs=records(100);run=tmp_path/'run';run.mkdir();calls=[]
    monkeypatch.setattr(audit,'replay_job',lambda job:(calls.append(job[1]['trial_id']) or dict(trial_id=job[1]['trial_id'],replayed=True,error=None)))
    store=protocol.RunStore(run)
    a,_=audit.run_replay_audit(store,{'evaluation_runtime_sha256':'test'},rs,rs,{},mode)
    assert len(calls)==a['replayed_trials']==expected
    assert calls==(audit.select_sample('run',rs) if mode=='sampled' else sorted(r['trial_id'] for r in rs))

def test_sampled_mismatch_blocks_completion(tmp_path,monkeypatch):
    store,_=common_store(tmp_path)
    original=protocol.RunStore.validate
    monkeypatch.setattr(protocol.RunStore,'validate',lambda self,*a,**kw:original(self,*a,allow_synthetic=True,**kw))
    monkeypatch.setattr(audit,'replay_job',lambda j:dict(trial_id=j[1]['trial_id'],replayed=False,error=dict(field='recovery_steps',stored_value=2,replayed_value=3,trace_sha='test')))
    with pytest.raises(ValueError,match='replay mismatch'):store.complete({'status':'COMPLETE'})
    assert not (store.path/'completion.json').exists()
    assert protocol.read_json(store.path/'sampled_replay_audit.json')['mismatch_count']==1

def test_full_failure_audit_does_not_mutate_sealed_raw(tmp_path,monkeypatch):
    store,_=common_store(tmp_path);protocol.write_json(store.path/'completion.json',{'synthetic_fixture':True})
    before={str(p):protocol.sha256(p) for p in store.path.rglob('*') if p.is_file()}
    monkeypatch.setattr(audit,'replay_job',lambda j:dict(trial_id=j[1]['trial_id'],replayed=False,error=dict(field='recovery_time',stored_value=1.,replayed_value=2.,trace_sha='test')))
    i=protocol.read_json(store.path/'identity.json');manifest=protocol.read_jsonl(store.path/'manifest_snapshot.jsonl')
    result,path=audit.run_replay_audit(store,i,store.records(),manifest,{},'full')
    assert result['status']=='AUDIT_FAILED' and result['mismatch_count']==1
    assert not path.is_relative_to(store.path)
    assert before=={str(p):protocol.sha256(p) for p in store.path.rglob('*') if p.is_file()}

def test_light_summary_tamper_rejected(tmp_path):
    from g1_recovery_metrics import summarize
    store,m=common_store(tmp_path);s=summarize(store.records(),[m.plan]);s['executed']+=1
    protocol.write_json(store.path/'summary.json',s)
    with pytest.raises(ValueError,match='summary'):store.validate(allow_synthetic=True)

def test_report_register_resume_do_not_explicitly_request_full():
    import ast
    for name in ['g1_recovery_eval.py','g1_recovery_report.py','g1_complete_robustness.py']:
        tree=ast.parse((protocol.LAB/'tools/evaluation'/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='validate':
                assert all(not (k.arg=='validation_level' and isinstance(k.value,ast.Constant) and k.value.value=='full') for k in node.keywords)
    import inspect
    assert inspect.signature(protocol.RunStore.validate).parameters['validation_level'].default=='light'

def test_full_common_still_detects_changed_field(tmp_path):
    store,_=common_store(tmp_path);r=store.records()[0];r['recovery_steps']+=1
    protocol.write_json(store.path/'trial_records'/f"{r['trial_id']}.json",r)
    with pytest.raises(ValueError,match='replay mismatch'):store.validate(validation_level='full',allow_synthetic=True)
    audits=list(tmp_path.glob('audits/run/*/full_replay_audit.json'))
    assert audits and protocol.read_json(audits[0])['mismatches'][0]['field']=='recovery_steps'


def test_audit_transition_projection_rejects_physical_loop_change():
    from g1_audit_compatibility import projection
    path='tools/evaluation/g1_complete_robustness.py'
    a='def immutable_json(p,v): pass\ndef execute(args):\n    actions=policy(obs,extra)\n'
    b=a.replace('actions=policy(obs,extra)','actions=0*policy(obs,extra)')
    assert projection(a,path)!=projection(b,path)
    c=a.replace('def immutable_json(p,v): pass','def immutable_json(p,v):\n    validate_audit_transition(p,v)')
    assert projection(a,path)==projection(c,path)


def test_missing_historical_sample_is_reported_without_replay(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();protocol.write_json(run/'completion.json',{'status':'COMPLETE'})
    monkeypatch.setattr(audit,'run_replay_audit',lambda *a,**k:pytest.fail('Report silently started full replay'))
    assert audit.report_audit_status(run)['sampled']=='SAMPLED_REPLAY_NOT_AVAILABLE'


def test_register_uses_light_not_repeated_full(tmp_path,monkeypatch):
    run=tmp_path/'runs'/'r';run.mkdir(parents=True)
    protocol.write_json(run/'completion.json',{'status':'COMPLETE'})
    protocol.write_json(tmp_path/'registry.json',{'evaluations':{}})
    calls=[]
    def validated(self,*a,**kw):
        calls.append(kw.get('validation_level','light'))
        return {'evaluation_runtime_sha256':'fixture'},[]
    monkeypatch.setattr(protocol.RunStore,'validate',validated)
    protocol.register_result(tmp_path,run)
    assert calls==['light']
    assert 'r' in protocol.read_json(tmp_path/'registry.json')['evaluations']
