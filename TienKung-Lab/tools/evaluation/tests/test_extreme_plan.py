import ast
from pathlib import Path
import pytest
from g1_extreme_plan import scenario,conditions,SLOPES,formal_levels,scan_level
from g1_extreme_d50 import RecoveryPoint as P
from g1_extreme_report import summary


def test_seed_allocation_disjoint_and_no_collisions():
    seen=set()
    for phase in ['explore','formal','baseline']:
        for c in conditions(SLOPES):
            for repeat in range(20):
                r=scenario(c,1,repeat,phase)
                assert r['reset_seed'] not in seen
                seen.add(r['reset_seed'])
                assert 0<=r['reset_seed']<2**32
    with pytest.raises(ValueError):scenario((0,.5,0,'B1'),1,1000,'formal')


def test_same_reset_and_timing_across_magnitude_no_future_data():
    a=scenario((10,1.,90,'B1'),.9,4,'explore')
    b=scenario((10,1.,90,'B1'),1.125,4,'explore')
    for field in ['initial_pose_parameters','initial_velocity_parameters','initial_joint_parameters','reset_seed','sampled_onset','onset','release']:
        assert a[field]==b[field]
    assert 3<=a['sampled_onset']<=a['onset']<=5
    assert a['onset']-a['sampled_onset']<.02
    assert a['release']-a['onset']==pytest.approx(.2)
    v=scenario((10,1.,90,'B2'),1,4,'explore');assert v['release']==v['onset']
    assert scenario((0,.5,0,'A'),0,0,'baseline')['onset'] is None


def test_small_formal_selection_never_exceeds_ppo_range():
    points=[P(.3,20,20,20),P(.9,20,20,18),P(1.125,20,20,12),P(1.40625,20,20,4)]
    a=formal_levels(points,'B1')
    assert len(a)==3 and a[0]==.3 and a[-1]==1.40625
    assert 1.125<a[1]<1.40625
    assert len(formal_levels([P(.3,20,0,0),P(.9,20,0,0)],'B1'))==3
    assert scan_level(1,'B2')==3.125


def test_controller_reduced_budget_and_ppo_only_exploration():
    import inspect,g1_extreme_eval as e
    src=inspect.getsource(e.run_group)
    assert 'range(100)' not in src and 'supplement' not in src
    tree=ast.parse(src)
    exploration=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='wave' and any(k.arg=='models' for k in n.keywords)]
    assert len(exploration)==2
    assert all(ast.literal_eval(next(k.value for k in n.keywords if k.arg=='models'))==('ppo',) for n in exploration)


def test_scan_resume_ignores_future_rounds(tmp_path,monkeypatch):
    import g1_extreme_eval as e
    from g1_recovery_protocol import write_json
    monkeypatch.setattr(e,'ROOT',tmp_path)
    for n in range(3):
        run=tmp_path/'waves'/f'original_explore_{n:02d}'/'formal/ppo/all/attempt-001'
        (run/'records').mkdir(parents=True)
        write_json(run/'records/x.json',{'round':n})
        write_json(run/'completion.json',{'records_sha256':{'x.json':'dummy'}})
    assert e.wave_rows('explore','original','ppo',max_round=1)==[{'round':0}]
    assert len(e.wave_rows('explore','original','ppo'))==3


def test_empty_push_denominator_na_not_zero():
    r=dict(actually_perturbed=False,recovered=False,pre_push_failure=True,fell=True,episode_success=False,status='FELL',survived_after_push=None,within_5_touchdowns=False,velocity_rmse_mps=None,tracking_duration_s=0)
    s=summary([r])
    assert s['recovery_rate'] is None and s['survival_rate'] is None
    assert s['pre_onset_fall']==1 and s['recovery_steps_mean'] is None


def test_saved_plan_and_old_judge_unchanged():
    import g1_paper_recovery as r
    assert r.VERSION=='paper_fixed_time_confirmation_v1_candidate'
    assert len(r.plans())==6420


def test_report_admits_only_sealed_paired_waves_and_rejects_mutation(tmp_path):
    from g1_recovery_protocol import write_json,sha256,digest
    from g1_extreme_report import load,report
    from g1_extreme_plan import MODELS
    runtime=dict(evaluation_code_commit='frozen',evaluation_runtime_sha256='native',paper_sources={'sim':'same'})
    identity=dict(runtime=runtime,checkpoint_sha256={m:m+'-sha' for m in MODELS},protocol_sha256='protocol')
    write_json(tmp_path/'identity.json',identity)
    wave=tmp_path/'waves/original_formal_00';wave.mkdir(parents=True)
    plan=[scenario((0,.5,0,'B1'),.9,0,'formal')]
    write_json(wave/'formal_frozen_manifest.json',plan);write_json(wave/'wave_models.json',list(MODELS))
    for m in MODELS:
        run=wave/'formal'/m/'all/attempt-001';(run/'records').mkdir(parents=True)
        r=dict(plan[0],checkpoint_sha256=m+'-sha',actually_perturbed=True,recovered=False,pre_push_failure=False,fell=True,episode_success=False,status='FELL',survived_after_push=False,within_5_touchdowns=False,velocity_rmse_mps=.4,tracking_duration_s=2.,recovery_time=None,recovery_steps=None)
        write_json(run/'records/r.json',r)
        binding=dict(runtime=runtime,checkpoint_sha256=m+'-sha',manifest_hash=digest(plan),protocol_sha256='protocol',num_envs=64)
        write_json(run/'binding.json',binding)
        write_json(run/'completion.json',dict(status='COMPLETE',episodes=1,binding_sha256=sha256(run/'binding.json'),records_sha256={'r.json':sha256(run/'records/r.json')}))
    assert load(tmp_path)==[]
    write_json(wave/'WAVE_COMPLETE.json',dict(models=list(MODELS),episodes_per_model=1))
    assert len(load(tmp_path))==4
    assert report(tmp_path)['formal_episodes']==4
    assert (tmp_path/'report/curve_B1_s+0.png').exists()
    last=wave/'formal/ours/all/attempt-001/records/r.json';write_json(last,{'modified':True})
    with pytest.raises(ValueError,match='Sealed record changed'):load(tmp_path)
