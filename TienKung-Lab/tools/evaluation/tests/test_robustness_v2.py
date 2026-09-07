"""Independent robustness planning, causal release and replay contracts."""
import copy
from collections import Counter
import numpy as np
import pytest
from g1_recovery_protocol import canonical
from g1_robustness_protocol import protocol, generate_plan, COUNTS, waveform, waveform_sha
from g1_robustness_trial import RobustnessTrial, replay
from g1_robustness_physics import force_and_arm, vector_world
from test_common_task_v2 import physical_frame, plan

@pytest.fixture(scope='module')
def p():return protocol()[0]

@pytest.fixture(scope='module')
def rows(p):return generate_plan(p)

def test_exact_paired_plan(rows):
    assert Counter(r['suite'] for r in rows)==COUNTS
    assert len(rows)==16128
    assert len({r['trial_id'] for r in rows})==len(rows)
    assert len({r['reset_seed'] for r in rows})==len(rows)
    for slope in [-20,-15,-10,0,10,15,20]:
        for mag in np.arange(0,3.01,.25):
            cell=[r for r in rows if r['family']=='velocity_jump' and r['slope_deg']==slope and r['push_magnitude']==mag]
            assert len(cell)==128
            assert set(Counter(r['direction_deg'] for r in cell).values())=={16}
            assert sum(r.get('sham',False) for r in cell)==(128 if mag==0 else 0)

def test_waveform_and_sequence(rows):
    for r in rows:
        if r['family']=='random_force':
            w=waveform(r)
            assert w.shape==(2000,2)
            assert waveform_sha(w)==r['waveform_sha256']
            assert np.array_equal(w,waveform(r))
        if r['family']=='repeated_impulse':
            assert r['impacts'][0]['offset_s']==0
            assert r['impacts'][-1]['offset_s']==8
            assert len(r['impacts'])==(17 if r['period_s']==.5 else 9)

@pytest.mark.parametrize('duration',[.05,.2,.5])
def test_force_exact_physics_duration(duration):
    r=dict(family='force_pulse',duration_s=duration,push_direction_heading=[1.,0.],equivalent_delta_v_mps=1.)
    f=np.array([force_and_arm(r,30,k*.005)[0] for k in range(102)])
    assert np.count_nonzero(f[:,0])==round(duration/.005)
    assert f.sum(axis=0)*.005==pytest.approx([30,0,0])

@pytest.mark.parametrize('mode,arm,torque',[('com',0,[0,0,0]),('upper_pitch',.2,[0,30,0]),('lateral_yaw',.15,[0,0,22.5])])
def test_wrench_moment(mode,arm,torque):
    r=dict(family='wrench_pulse',duration_s=.2,push_direction_heading=[1,0],equivalent_delta_v_mps=1,wrench_mode=mode,moment_arm_m=arm,torque_sign=1)
    f,a=force_and_arm(r,30,0)
    assert np.cross(a,f)==pytest.approx(torque)

def test_heading_and_world_distinct():
    r=dict(family='velocity_jump',push_direction_heading=[1,0],push_magnitude=1,vector_frame='heading_at_onset')
    assert vector_world(r,np.pi/2)==pytest.approx([0,1])
    r['vector_frame']='world'
    assert vector_world(r,np.pi/2)==pytest.approx([1,0])

def synthetic(p,duration,mode='normal'):
    r=plan();r.update(evaluation_role='disturbance_family_suite',family='force_pulse',duration_s=duration,vector_frame='world')
    m=RobustnessTrial(r,p)
    for k in range(2000):
        t=k*.02;speed=.4
        if m.release_time is not None and t>m.release_time+10 and m.t<m.release_time+10:
            t=m.release_time+10
        if m.onset is not None and t<=m.release_time:speed=0
        if mode=='relapse' and m.onset is not None and m.release_time+3<t<m.release_time+4:speed=0
        fall=mode=='fall' and m.onset is not None and t>=m.release_time+7
        if mode=='early_fall':fall=m.onset is not None and t>=m.onset+.04
        due=m.feed(physical_frame(t,speed=speed,fallen=fall))
        if due:m.start_intervention(dict(time=t,duration_s=duration,heading_yaw=0.,mass_kg=30.))
        if m.status:return m
    raise AssertionError('Unbounded trial')

@pytest.mark.parametrize('duration',[0,.05,.2,5,8,10])
@pytest.mark.parametrize('mode',['normal','relapse','fall','early_fall'])
def test_release_recovery_replay(p,duration,mode):
    m=synthetic(p,duration,mode);r=m.result()
    restored=replay(m.plan,p,m.trace(),r,m.all_events())
    assert canonical(restored.result())==canonical(r)
    if r['recovered_sustained_and_survived']:
        assert r['recovery_time']>=.6-1e-9
        assert r['first_confirmation']-r['first_recovery_entry']>=.5-1e-9
        assert min(f['time'] for f in m.frames if f['post_push_sample'])>m.release_time
    if mode=='relapse':
        assert r['relapse_count']>=1
        assert r['sustained_recovery_entry']>r['first_recovery_entry']
    if mode in ('fall','early_fall'):
        assert r['recovery_time'] is None
        assert not r['recovered_sustained_and_survived']
    if mode=='early_fall' and duration>=.05:
        assert not r['disturbance_phase_survived']
        assert r['recovery_clock_start_time'] is None

def test_statistical_denominators_and_censored_boundary():
    from g1_robustness_analysis import curve,boundary,aggregate
    rows=[dict(push_magnitude=.25,sham=False,push_applied=True,survived=True),
          dict(push_magnitude=.25,sham=False,push_applied=False,survived=False),
          dict(push_magnitude=0,sham=True,push_applied=False,survived=True)]
    a=curve(rows,'survived');b=curve(rows,'survived',True)
    assert len(a)==1 and a[0]['rate']==.5 and b[0]['rate']==1
    points=[{'delta_v_mps':x,'rate':y} for x,y in [(.25,1),(.5,.8),(.75,1)]]
    assert boundary(points,.9)['value_mps']==pytest.approx(.375)
    assert boundary(points,.9)['censor']=='none'
    assert boundary(points,.5)['censor']=='right'
    assert boundary([{'delta_v_mps':.25,'rate':.1}],.9)['censor']=='left'

def test_paired_statistics_and_holm():
    from g1_robustness_analysis import paired_tests,holm,BINARY
    from g1_robustness_protocol import MODEL_ORDER
    rows=[dict(trial_id=str(i),push_applied=True,sham=False,recovery_time=1.,recovery_steps=4,**{k:True for k in BINARY}) for i in range(5)]
    models={m:copy.deepcopy(rows) for m in MODEL_ORDER}
    models['context_only'][0]['recovery_time']=2.
    result=holm(paired_tests(models,'test'))
    assert len(result)==35
    r=next(r for r in result if r['model_a']=='context_only' and r['endpoint']=='recovery_time')
    assert r['n']==5 and r['effect']==pytest.approx(.2)
    assert r['holm_p']>=r['raw_p']
    models['rl_only'].pop()
    with pytest.raises(ValueError,match='paired'):paired_tests(models,'bad')

def test_first_refresh_invalid_is_not_replaced():
    import json
    from g1_robustness_analysis import native_diagnostics
    valid=dict(context_valid=True,N=2,margin=.1,query_event=True,query_failed=False,query_category='ok')
    invalid=dict(context_valid=False,N=None,margin=None,query_event=True,query_failed=True,query_category='geometry')
    planes=[dict(valid,query_id=1),dict(invalid,query_id=2),dict(valid,query_id=3)]
    trace={'time':np.array([0.,.02,.04]),'plane_json':np.array([json.dumps(p) for p in planes]),'physical_touchdown_flags':np.zeros((3,2),bool)}
    r=dict(trial_id='x',disturbance_start_time=0.,recovered_sustained_and_survived=False,recovery_steps=None,recovery_time=None)
    out=native_diagnostics(trace,r)
    assert out['N_pre']==2 and out['N_post'] is None and out['first_post_refresh_valid'] is False
    assert out['query_failures']==1

@pytest.mark.parametrize('duration',[0,.05,.2,5,10])
def test_archived_substep_physics_replay(p,duration,tmp_path):
    from g1_robustness_store import RobustnessStore,validate_trial
    m=synthetic(p,duration)
    m.plan.update(equivalent_delta_v_mps=1.)
    for j in range(round(duration/.005)+1):
        start=m.onset+j*.005;f,arm=force_and_arm(m.plan,30,start-m.onset)
        m.add_physics(dict(time=start+.005,start_time=start,dt_s=.005,mass_kg=30.,terminal=False,
            force_requested_world_n=f.tolist(),force_world_n=f.tolist(),composed_force_world_n=f.tolist(),
            force_active=bool(np.any(f)),application_point_world_m=[0,0,.8],whole_robot_com_world_m=[0,0,.8],
            application_link_position_world_m=[0,0,.8],torque_about_com_world_nm=[0,0,0],composed_torque_about_link_world_nm=[0,0,0]))
    store=RobustnessStore(tmp_path);store.save_trial(m.result(),m.trace(),m.all_events())
    assert validate_trial((str(tmp_path),m.plan,p))==m.plan['trial_id']

def test_report_template_five_models_and_figures(tmp_path,monkeypatch,p,rows):
    import json
    import zipfile
    import g1_robustness_report as report
    from g1_recovery_protocol import write_json,sha256
    from g1_robustness_protocol import MODEL_ORDER,COUNTS
    monkeypatch.setattr(report,'LAB',tmp_path)
    monkeypatch.setattr(report,'COUNTS',{s:1 for s in COUNTS})
    root=tmp_path/'new';index={'models':[]}
    base=synthetic(p,0);trace=base.trace();record=base.result()
    for model in MODEL_ORDER:
        run=root/'runs'/model
        identity={k:'same' for k in ('evaluation_code_commit','evaluation_runtime_sha256','actual_physics_hash','realized_environment_hash','candidate_parameters_sha256','protocol_hash','manifest_hash','metrics_config_hash')}
        identity['checkpoint_sha256']=model
        write_json(run/'identity.json',identity)
        records=[]
        for suite in COUNTS:
            assignment=next(r for r in rows if r['suite']==suite and not r.get('sham',False))
            r={**record,**assignment};records.append(r)
            write_json(run/'trial_records'/f'{r["trial_id"]}.json',r)
            path=run/'traces'/f'{r["trial_id"]}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,**trace)
        write_json(run/'completion.json',{'status':'COMPLETE','files':{str(x.relative_to(run)):sha256(x) for x in run.rglob('*') if x.is_file()}})
        write_json(root/'completed_models'/f'{model}.json',{'evaluation_id':model,'completion_sha256':sha256(run/'completion.json')})
        old=tmp_path/'experiments/g1_recovery_eval_v2/runs'/model
        write_json(old/'trial_records'/'e1.json',record);write_json(old/'completion.json',{'status':'COMPLETE'})
        (old/'traces').mkdir(exist_ok=True);np.savez_compressed(old/'traces'/f'{record["trial_id"]}.npz',**trace)
        index['models'].append({'model':model,'evaluation_id':model,'completion_sha256':sha256(old/'completion.json')})
    write_json(root/'standard_benchmark_five_model_index.json',index)
    result=report.build_report(root)
    md=__import__('pathlib').Path(result['markdown']).read_text()
    assert 'Context-only' in md and 'OUT-OF-DOMAIN' in md and '16. Experimental Integrity' in md
    assert len(list((root/'report/figures').glob('*.png')))>=15
    with zipfile.ZipFile(result['docx']) as z:
        assert 'word/document.xml' in z.namelist()
        assert len([n for n in z.namelist() if n.startswith('word/media/')])>=15


def test_nonempty_native_bucket_json_serialization():
    import json
    from g1_robustness_analysis import association
    rows=[dict(N_post=n,sustained_success=True,recovery_steps=n+3,recovery_time=.6+n*.1) for n in (1,2,2,3)]
    result=association(rows,'N_post')
    assert len(json.loads(json.dumps(result,allow_nan=False))['calibration'])==3


def test_report_event_latencies_use_actual_release(p):
    from g1_robustness_analysis import aggregate
    m=synthetic(p,5.);r=m.result();a=aggregate([r])
    assert a['first_recovery_entry_latency_s']['median']==pytest.approx(r['first_recovery_entry']-m.release_time)
    assert a['first_confirmation_latency_s']['median']==pytest.approx(r['first_confirmation']-m.release_time)
