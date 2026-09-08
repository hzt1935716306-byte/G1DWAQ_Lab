"""Evaluation-only wrench regression; no Isaac application in CPU tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from g1_world_wrench import (rotation_world_from_link,rotate,world_wrench_at_point_to_link_wrench,
    application_point_world,set_world_wrenches,clear_wrench,verify_wrench_sample)
from g1_robustness_physics import PhysicalInterventions,force_and_arm


def quat(roll=0,pitch=0,yaw=0):
    from scipy.spatial.transform import Rotation
    q=Rotation.from_euler('xyz',[roll,pitch,yaw],degrees=True).as_quat()
    return q[[3,0,1,2]]


class Composer:
    def __init__(self,n=1):
        self.composed_force_as_torch=torch.zeros((n,1,3))
        self.composed_torque_as_torch=torch.zeros((n,1,3));self.calls=[];self.corrupt=False
    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.composed_force_as_torch[ids]=0;self.composed_torque_as_torch[ids]=0
    def set_forces_and_torques(self,**kw):
        self.calls.append(kw)
        self.composed_force_as_torch[:]=kw['forces'];self.composed_torque_as_torch[:]=kw['torques']
        if self.corrupt:self.composed_torque_as_torch+=1


def sample(q=None):
    q=quat(pitch=30) if q is None else q
    c=Composer();s=set_world_wrenches(c,[[300,0,0]],[[0,0,.2]],[[0,0,0]],[[0,0,0]],[q],0,'cpu')[0]
    s.update(application_point_semantics='whole_robot_com_plus_world_offset',
        whole_robot_com_world_m=[0,0,0],declared_world_offset_m=[0,0,.2])
    return c,s


def test_known_installed_kernel_mismatch_and_correct_local_path():
    from g1_wrench_validation import installed_kernel_case
    r=installed_kernel_case()
    assert r['classification']=='KNOWN_UPSTREAM_FRAME_MISMATCH'
    assert r['old_global_position']['torque_world_nm']==pytest.approx([0,51.961517333984375,0],abs=1e-5)
    assert r['new_explicit_local_wrench']['torque_world_nm']==pytest.approx([0,60,0],abs=1e-5)


def test_pitch30_exact_adapter():
    q=quat(pitch=30);f,t=world_wrench_at_point_to_link_wrench([300,0,0],[0,0,.2],[0,0,0],[0,0,0],q)
    assert f==pytest.approx([300*np.sqrt(3)/2,0,150])
    assert t==pytest.approx([0,60,0])


@pytest.mark.parametrize('angles',[(0,0,0),(30,0,0),(0,30,0),(0,0,45),(20,30,45)])
def test_rotations_and_free_plus_arm(angles):
    q=quat(*angles);r=np.array([.2,.3,.4]);f=np.array([300,10,-20]);free=np.array([2,3,4])
    fl,tl=world_wrench_at_point_to_link_wrench(f,r,free,[0,0,0],q);rot=rotation_world_from_link(q)
    np.testing.assert_allclose(rotate(rot,fl),f,atol=1e-12)
    np.testing.assert_allclose(rotate(rot,tl),free+np.cross(r,f),atol=1e-12)


def test_100_random_quaternion_covariance_and_roundtrip():
    rng=np.random.default_rng(20260908);q=rng.normal(size=(100,4));q/=np.linalg.norm(q,axis=1)[:,None]
    r=rng.normal(size=(100,3));f=rng.normal(size=(100,3))*300;free=rng.normal(size=(100,3))*20
    rot=rotation_world_from_link(q);rt=rot.swapaxes(-1,-2)
    np.testing.assert_allclose(rotate(rt,np.cross(r,f)),np.cross(rotate(rt,r),rotate(rt,f)),atol=1e-11)
    fl,tl=world_wrench_at_point_to_link_wrench(f,r,free,np.zeros_like(r),q)
    np.testing.assert_allclose(rotate(rot,fl),f,atol=1e-11)
    np.testing.assert_allclose(rotate(rot,tl),free+np.cross(r,f),atol=1e-11)
    c=Composer(100);samples=set_world_wrenches(c,f,r,free,np.zeros_like(r),q,0,'cpu')
    for i,s in enumerate(samples):
        np.testing.assert_allclose(s['composed_torque_about_link_world_nm'],free[i]+np.cross(r[i],f[i]),atol=.001)


def test_zero_arm():
    _,t=world_wrench_at_point_to_link_wrench([1,2,3],[2,3,4],[0,0,0],[2,3,4],quat(30,20,50))
    assert t==pytest.approx([0,0,0])


def test_pure_torque_zero_force():
    q=quat(30,20,50);f,t=world_wrench_at_point_to_link_wrench([0,0,0],[2,3,4],[1,2,3],[0,0,0],q)
    assert f==pytest.approx([0,0,0]);assert rotate(rotation_world_from_link(q),t)==pytest.approx([1,2,3])


def test_link_fixed_offset_rotates():
    p=application_point_world('link_fixed_offset',[1,2,3],quat(yaw=90),declared_offset=[1,0,0])
    assert p==pytest.approx([1,3,3])


def test_world_fixed_point_does_not_follow_body():
    for l,q in [([1,2,3],quat()),([3,5,8],quat(20,40,60))]:
        assert application_point_world('world_fixed_point',l,q,declared_world_point=[4,5,6])==pytest.approx([4,5,6])


def test_existing_com_following_world_offset_semantics():
    assert application_point_world('whole_robot_com_plus_world_offset',[7,8,9],quat(pitch=30),
        declared_offset=[0,0,.2],whole_robot_com=[1,2,3])==pytest.approx([1,2,3.2])


def test_composer_positions_none():
    c,s=sample();assert c.calls[-1]['positions'] is None


def test_composer_local_frame():
    c,s=sample();assert c.calls[-1]['is_global'] is False


def test_reconstructed_world_force():
    _,s=sample();assert s['composed_force_world_n']==pytest.approx([300,0,0],abs=2e-5)


def test_reconstructed_world_torque():
    _,s=sample();assert s['composed_torque_about_link_world_nm']==pytest.approx([0,60,0],abs=1e-5)
    verify_wrench_sample(s)


@pytest.mark.parametrize('field',['force_link_n','equivalent_torque_link_nm','composed_force_link_n','composed_torque_link_nm'])
def test_corrupt_layer_fails(field):
    _,s=sample();s[field][0]+=1
    with pytest.raises(ValueError):verify_wrench_sample(s)


@pytest.mark.parametrize('duration,count',[(.05,10),(.2,40),(.5,100),(5.,1000)])
def test_exact_substep_counts_and_release(duration,count):
    plan=dict(family='force_pulse',duration_s=duration,equivalent_delta_v_mps=.5,push_direction_heading=[1,0])
    values=[force_and_arm(plan,30,j*.005)[0] for j in range(count+1)]
    assert sum(bool(np.any(f)) for f in values)==count
    assert not np.any(values[-1])
    assert np.sum(values,axis=0)*.005==pytest.approx([15,0,0])


def test_explicit_release_zero_and_next_substep():
    c,s=sample();clear_wrench(c)
    assert not torch.any(c.composed_force_as_torch);assert not torch.any(c.composed_torque_as_torch)
    s=set_world_wrenches(c,[[0,0,0]],[[0,0,0]],[[0,0,0]],[[0,0,0]],[quat()],0,'cpu')[0]
    assert s['composed_force_world_n']==[0,0,0] and s['composed_torque_about_link_world_nm']==[0,0,0]


def test_bad_composer_cleanup():
    c=Composer();c.corrupt=True
    with pytest.raises(ValueError):set_world_wrenches(c,[[1,2,3]],[[0,0,0]],[[0,0,0]],[[0,0,0]],[quat()],0,'cpu')
    assert not torch.any(c.composed_force_as_torch);assert not torch.any(c.composed_torque_as_torch)


def controller_stub():
    c=Composer();x=PhysicalInterventions.__new__(PhysicalInterventions)
    x.env=SimpleNamespace(robot=SimpleNamespace(permanent_wrench_composer=c))
    x.enabled=True;x.pending=[1];x.resetting=set()
    return x,c


def test_reset_suppresses_reapplication_and_clears_even_exception():
    x,c=controller_stub();c.composed_force_as_torch[:]=5
    def reset(ids):
        assert 0 in x.resetting and not torch.any(c.composed_force_as_torch)
        raise RuntimeError('reset failure')
    x.original_reset=reset
    with pytest.raises(RuntimeError):x.reset([0])
    assert not x.resetting and not torch.any(c.composed_force_as_torch)


def test_fall_cleanup_is_per_env():
    c=Composer(2);c.composed_force_as_torch[:]=5;c.composed_torque_as_torch[:]=3
    clear_wrench(c,[0])
    assert not torch.any(c.composed_force_as_torch[0]) and not torch.any(c.composed_torque_as_torch[0])
    assert torch.all(c.composed_force_as_torch[1]==5)


@pytest.mark.parametrize('method',['write','update'])
def test_physics_exception_cleanup(method):
    x,c=controller_stub();c.composed_force_as_torch[:]=5
    def fail(*a,**kw):raise KeyboardInterrupt()
    setattr(x,'_'+method,fail)
    with pytest.raises(KeyboardInterrupt):getattr(x,method)()
    assert not x.enabled and not x.pending and not torch.any(c.composed_force_as_torch)


def test_precondition_failure_clear():
    x,c=controller_stub();c.composed_torque_as_torch[:]=3;x.clear()
    assert not x.enabled and not torch.any(c.composed_torque_as_torch)


def test_fixed_smoke_plan_integrity(tmp_path):
    from g1_robustness_protocol import prepare,load_prepared
    from g1_complete_robustness import INDEX
    from g1_recovery_protocol import read_jsonl,atomic_write,jsonl,read_json,write_json,digest
    prepare(tmp_path,INDEX,wrench_smoke=True);_,rows,info=load_prepared(tmp_path)
    assert [r['repeat_id'] for r in rows]==[0,33,66,99]
    assert [r['direction_deg'] for r in rows]==[0,90,180,270]
    rows[0]['reset_seed']+=1
    atomic_write(tmp_path/'manifest.jsonl',jsonl(rows));info['manifest_hash']=digest(rows);write_json(tmp_path/'prepared.json',info)
    with pytest.raises(ValueError,match='fixed four'):load_prepared(tmp_path)


def test_old_invalid_writer_refuses_append(tmp_path,monkeypatch):
    import g1_complete_robustness as runner
    import g1_recovery_eval as evaluator
    from g1_recovery_protocol import write_json
    code=dict(evaluation_runtime_dirty=False,evaluation_code_commit='commit',
        evaluation_runtime_sha256='hash',evaluation_runtime_sources={})
    monkeypatch.setattr(runner,'code_identity',lambda:code)
    monkeypatch.setattr(runner,'load_prepared',lambda root:({'robustness':{'num_envs':64}},[],{}))
    monkeypatch.setattr(runner,'model_identity',lambda *a:{'task_name':'fake_ppo'})
    monkeypatch.setattr(evaluator,'snapshot_checkpoint',lambda *a:None)
    monkeypatch.setattr(runner,'evaluation_key',lambda i:'known')
    write_json(tmp_path/'standard_benchmark_five_model_index.json',{})
    old=tmp_path/'runs/known-attempt-0001';old.mkdir(parents=True)
    write_json(old/'failure.json',{'status':'INVALID','error':'original frame mismatch'})
    before=(old/'failure.json').read_bytes()
    with pytest.raises(ValueError,match='INVALID attempt preserved'):
        runner.execute(SimpleNamespace(output_root=tmp_path,model='ppo_plain'))
    assert (old/'failure.json').read_bytes()==before
    assert not (old/'trial_records').exists()


def test_old_invalid_run_unchanged_if_local_evidence_present():
    from g1_recovery_protocol import LAB,sha256
    root=LAB/'experiments/g1_wrench_frame_fix_validation/development'
    snapshot=root/'old_invalid_before_sha256.json'
    if not snapshot.exists():pytest.skip('Local historical evidence not distributed with source checkout')
    old=LAB/'experiments/g1_complete_robustness_v2/runs/f754ef075b7393c0ec093ca8-attempt-0001'
    before=json.loads(snapshot.read_text())
    after={str(p.relative_to(old)):sha256(p) for p in old.rglob('*') if p.is_file()}
    assert after==before
    assert not (old/'completion.json').exists()


def test_each_substep_uses_current_pose_without_observations():
    c=Composer();robot=SimpleNamespace(permanent_wrench_composer=c,data=SimpleNamespace(
        body_com_pos_w=torch.tensor([[[0.,0.,.1]]]),body_link_pos_w=torch.zeros((1,1,3)),
        body_link_quat_w=torch.tensor(np.array([[quat()]]),dtype=torch.float32)))
    env=SimpleNamespace(robot=robot,eval_masses=torch.tensor([[30.]]),num_envs=1,
        device='cpu',eval_history_calls=7,sim_step_counter=1)
    x=PhysicalInterventions.__new__(PhysicalInterventions);x.env=env;x.enabled=True;x.dt=.005;x.body_id=0
    x.base_counter=0;x.resetting=set();x.endpoint_frames={};x.waves={};x.original_write=lambda:None
    m=SimpleNamespace(status=None,onset=0.,first_terminal_physics_time=None,
        plan=dict(family='force_pulse',duration_s=.05,equivalent_delta_v_mps=.5,push_direction_heading=[1,0]))
    x.machines=[m];x.write();first=x.pending[0][1]
    env.sim_step_counter=2;robot.data.body_link_quat_w[:]=torch.tensor(np.array([[quat(pitch=30)]]),dtype=torch.float32)
    x.write();second=x.pending[0][1]
    assert first['force_link_n']!=second['force_link_n']
    assert second['composed_force_world_n']==pytest.approx(first['force_world_n'],abs=1e-4)
    assert env.eval_history_calls==7
