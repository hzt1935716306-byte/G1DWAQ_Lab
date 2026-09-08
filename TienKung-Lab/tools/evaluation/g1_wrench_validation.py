#!/usr/bin/env python3
"""Bounded wrench fix gates. Never launches the full robustness suite."""
from __future__ import annotations
import argparse
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np
from g1_recovery_protocol import LAB, code_identity, read_json, sha256, write_json
from g1_world_wrench import (rotation_world_from_link, rotate,
    world_wrench_at_point_to_link_wrench, set_world_wrenches, clear_wrench)

KERNEL=LAB.parents[2]/'IsaacLab/source/isaaclab/isaaclab/utils/warp/kernels.py'


def installed_kernel_case():
    """Execute the unchanged installed Warp kernel on CPU, including safe path."""
    import warp as wp
    spec=importlib.util.spec_from_file_location('wrench_validation_installed_kernel',KERNEL)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    wp.init();device='cpu';a=math.radians(30)
    q=np.array([math.cos(a/2),0.,math.sin(a/2),0.]);rot=rotation_world_from_link(q)
    f=np.array([300.,0,0]);r=np.array([0.,0,.2]);tau=np.cross(r,f)
    ids=wp.array([0],dtype=wp.int32,device=device)
    vec=lambda x:wp.array(np.asarray(x).reshape(1,1,3),dtype=wp.vec3f,device=device)
    link=vec([0,0,0]);quat=wp.array(q[[1,2,3,0]].reshape(1,1,4),dtype=wp.quatf,device=device)
    outputs=[]
    fl,tl=world_wrench_at_point_to_link_wrench(f,r,[0,0,0],[0,0,0],q)
    for global_position in (True,False):
        outf=wp.zeros((1,1),dtype=wp.vec3f,device=device);outt=wp.zeros((1,1),dtype=wp.vec3f,device=device)
        point=vec(r) if global_position else wp.empty((0,0),dtype=wp.vec3f,device=device)
        wp.launch(module.set_forces_and_torques_at_position,dim=(1,1),inputs=[ids,ids,
            vec(f if global_position else fl),vec([0,0,0] if global_position else tl),
            point,link,quat,outf,outt,global_position],device=device)
        wp.synchronize()
        outputs.append(dict(force_world_n=rotate(rot,outf.numpy()[0,0]).tolist(),
                            torque_world_nm=rotate(rot,outt.numpy()[0,0]).tolist()))
    if np.linalg.norm(np.asarray(outputs[0]['torque_world_nm'])-tau)<1:
        raise ValueError('Installed upstream behavior changed; review diagnostic classification')
    np.testing.assert_allclose(outputs[1]['force_world_n'],f,atol=5e-5,rtol=0)
    np.testing.assert_allclose(outputs[1]['torque_world_nm'],tau,atol=1e-5,rtol=0)
    return dict(classification='KNOWN_UPSTREAM_FRAME_MISMATCH',kernel_path=str(KERNEL),kernel_sha256=sha256(KERNEL),
        expected_force_world_n=f.tolist(),expected_torque_world_nm=tau.tolist(),
        old_global_position=outputs[0],new_explicit_local_wrench=outputs[1])


def require_gate(path,code):
    r=read_json(path)
    if r.get('status')!='PASS' or any(r.get(k)!=code[k] for k in ('evaluation_runtime_sha256','evaluation_code_commit')):
        raise ValueError('Required API gate missing, failed or bound to another runtime')
    return r


def check_model_smoke(root,model):
    from g1_robustness_store import RobustnessStore
    root=Path(root);ref=read_json(root/'completed_models'/f'{model}.json');run=root/'runs'/ref['evaluation_id']
    if sha256(run/'completion.json')!=ref['completion_sha256']:raise ValueError('Smoke completion changed')
    identity,rows=RobustnessStore(run).validate(workers=1)
    code=code_identity()
    if any(identity[k]!=code[k] for k in ('evaluation_runtime_sha256','evaluation_code_commit')):
        raise ValueError('Smoke gate belongs to another runtime')
    if identity['evaluation_role']!='wrench_frame_fix_physical_smoke' or len(rows)!=4:
        raise ValueError('Wrong physical smoke role or count')
    pushed=[r for r in rows if r['push_applied']]
    # Readiness/fall remain legitimate results; the gate nevertheless needs
    # at least one observed full pulse + release + post-force observation.
    complete=[r for r in pushed if r['disturbance_phase_survived']]
    if not complete:raise ValueError('Physical smoke has no complete pulse/release evidence')
    for r in complete:
        if r['physics_substeps_applied']!=10 or abs(r['actual_duration']-.05)>1e-10 or not r['release_zero_wrench_verified']:
            raise ValueError('Pulse/release timing gate failed')
        if not r['post_disturbance_observation_started']:raise ValueError('Missing post-force recovery chain')
    return dict(status='PASS',model=model,evaluation_id=run.name,designated=4,actually_pushed=len(pushed),
        complete_pulses=len(complete),evaluation_errors=0,
        outcomes=[dict(trial_id=r['trial_id'],status=r['status'],physics_substeps_applied=r['physics_substeps_applied'],
            actual_duration=r['actual_duration'],release_zero_wrench_verified=r['release_zero_wrench_verified']) for r in rows],
        **{k:identity[k] for k in ('evaluation_runtime_sha256','evaluation_code_commit','actual_physics_hash','realized_environment_hash','manifest_hash')})


def api_smoke(root,device):
    """One real environment, two free-space poses, one short known wrench each.

    No policy inference. Advance only physics, and restore the pose before each
    case. Composer buffers and zero-force next substep are checked explicitly.
    """
    import os
    import torch
    from types import SimpleNamespace
    from g1_complete_robustness import INDEX, model_identity
    from g1_robustness_protocol import protocol,generate_plan,select_wrench_smoke
    from g1_recovery_eval import snapshot_checkpoint,make_evaluation_environment
    root=Path(root);receipt=root/'api_smoke.json'
    if receipt.exists():raise ValueError('Preserve existing API receipt; use a new diagnostic root')
    code=code_identity()
    if code['evaluation_runtime_dirty']:raise ValueError('Commit runtime before API smoke')
    evidence=dict(status='FAIL',**code,kernel_diagnostic=installed_kernel_case(),cases=[])
    p,_=protocol();identity=model_identity(root/'api_inputs','ppo_plain',read_json(INDEX))
    snapshot=snapshot_checkpoint(root/'api_inputs',identity)
    os.chdir(LAB);sys.path.insert(0,str(LAB));sys.path.insert(0,str(LAB/'rsl_rl'))
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device=device).app;env=None
    try:
        args=SimpleNamespace(task=identity['task_name'],num_envs=1,headless=True,device=device)
        env,runner,policy,weights,effective=make_evaluation_environment(args,p,identity,snapshot)
        plan=select_wrench_smoke(generate_plan(p))[0]
        ids,names=env.robot.find_bodies('torso_link',preserve_order=True)
        if names!=['torso_link']:raise ValueError('Unique torso required')
        body=int(ids[0]);composer=env.robot.permanent_wrench_composer
        with torch.inference_mode():
            for pitch_deg in (0.,30.):
                clear_wrench(composer);env.begin_batch([plan])
                state=env.robot.data.root_state_w.clone()
                state[:,2]=env.scene.env_origins[:,2]+3. # Free space: no contact impulses.
                angle=math.radians(pitch_deg)/2
                state[:,3:7]=torch.tensor([math.cos(angle),0,math.sin(angle),0],device=device)
                state[:,7:]=0
                env.robot.write_root_state_to_sim(state)
                env.robot.write_joint_state_to_sim(env.robot.data.default_joint_pos.clone(),torch.zeros_like(env.robot.data.joint_vel))
                env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
                history=env.eval_history_calls
                l=env.robot.data.body_link_pos_w[:,body].cpu().numpy().astype(float)
                q=env.robot.data.body_link_quat_w[:,body].cpu().numpy().astype(float)
                rot=rotation_world_from_link(q)
                expected_rot=rotation_world_from_link(state[:,3:7].cpu().numpy())
                np.testing.assert_allclose(rot,expected_rot,atol=2e-6,rtol=0)
                sample=set_world_wrenches(composer,[[300.,0,0]],l+[[0,0,.2]],[[0,0,0]],l,q,body,device)[0]
                before=env.robot.data.body_com_vel_w.clone()
                env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
                after=env.robot.data.body_com_vel_w.clone()
                clear_wrench(composer);env.scene.write_data_to_sim()
                env.sim.step(render=False);env.scene.update(env.physics_dt)
                for value in (composer.composed_force_as_torch,composer.composed_torque_as_torch):
                    if bool(torch.any(value!=0)):raise ValueError('Residual force in next physical substep')
                if env.eval_history_calls!=history:raise ValueError('API smoke advanced actor history')
                evidence['cases'].append(dict(pitch_deg=pitch_deg,physics_steps_applied=1,
                    duration_s=float(env.physics_dt),next_substep_zero=True,actor_history_advances=0,
                    body_com_velocity_before=before.cpu().tolist(),body_com_velocity_after=after.cpu().tolist(),wrench=sample))
            if any(not torch.equal(v.cpu(),weights[k]) for k,v in runner.alg.policy.state_dict().items()):raise ValueError('Weights changed')
        evidence.update(status='PASS',num_envs=1,policy_inference_calls=0)
    except BaseException as exc:
        evidence['error']=repr(exc)
        raise
    finally:
        if env is not None:clear_wrench(env.robot.permanent_wrench_composer)
        write_json(receipt,evidence)
        print({'api_smoke':evidence['status'],'receipt':str(receipt)},flush=True)
        app.close()
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('kernel','prepare-smoke','api-smoke','check-smoke'))
    parser.add_argument('--output_root',required=True);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--model',choices=('ppo_plain','dwaq'));args=parser.parse_args();root=Path(args.output_root)
    if args.command=='kernel':
        result=installed_kernel_case();write_json(root/'installed_kernel_diagnostic.json',result)
    elif args.command=='api-smoke':result=api_smoke(root,args.device)
    elif args.command=='prepare-smoke':
        from g1_robustness_protocol import prepare
        from g1_complete_robustness import INDEX
        result=prepare(root,INDEX,wrench_smoke=True)
    else:
        result=check_model_smoke(root,args.model);write_json(root.parent/f'{args.model}_smoke_gate.json',result)
    print(result,flush=True)
if __name__=='__main__':main()
