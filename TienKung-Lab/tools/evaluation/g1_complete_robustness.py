#!/usr/bin/env python3
"""Strict paired robustness execution, independent of the sealed standard suite."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time
import yaml
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parent))
from g1_recovery_protocol import (LAB, atomic_write, code_identity, digest, evaluation_key,
    inspect_checkpoint, jsonl, load_yaml, lock, read_json, sha256, write_json)
from g1_robustness_protocol import prepare, load_prepared, MODEL_ORDER
from g1_robustness_store import RobustnessStore

ROOT=LAB/'experiments/g1_complete_robustness_v2'
INDEX=LAB/'experiments/g1_recovery_eval_v2/standard_benchmark_five_model_index.json'
ESTIMATOR=LAB/'logs/g1_com_velocity_estimator/v2_iteration_long_5000_random_init_fixed/com_velocity_estimator_v2_long_best.pt'


def immutable_json(path,value):
    if path.exists():
        if read_json(path)!=value:
            from g1_audit_compatibility import admit_immutable
            if not admit_immutable(path,value):raise ValueError('Immutable paired identity differs: '+str(path))
    else:write_json(path,value)


def model_identity(root,model,index):
    row=next(r for r in index['models'] if r['model']==model)
    old=LAB/'experiments/g1_recovery_eval_v2/runs'/row['evaluation_id']
    if sha256(old/'completion.json')!=row['completion_sha256']:raise ValueError('Standard seal changed')
    identity=read_json(old/'identity.json')
    mapping={k:identity[k] for k in ('checkpoint_sha256','agent_config_sha256','env_config_sha256')}
    mapping['task_name']=row['task'];mapping['resources']={}
    for role,r in identity['resources'].items():
        if role in ('native_nominal','native_capability','estimator'):
            path=ESTIMATOR if role=='estimator' else old/r['snapshot']
            if sha256(path)!=r['sha256']:raise ValueError('Native snapshot changed')
            mapping['resources'][role]={'path':str(path),'sha256':r['sha256']}
    path=root/'model_inputs'/f'{model}.json';immutable_json(path,mapping)
    result=inspect_checkpoint(row['task'],LAB/row['checkpoint_path'],model+'_complete_robustness_v2',
        'final',ESTIMATOR if 'estimator' in mapping['resources'] else None,path)
    for k in ('checkpoint_sha256','agent_config_sha256','env_config_sha256','native_configuration_sha256'):
        if result[k]!=identity[k]:raise ValueError('Native model identity differs from standard: '+k)
    if {k:v['sha256'] for k,v in result['resources'].items()}!={k:v['sha256'] for k,v in identity['resources'].items()}:
        raise ValueError('Native input resources differ from standard')
    return result


def execute(args):
    import numpy as np
    import torch
    from g1_recovery_eval import (snapshot_checkpoint, make_evaluation_environment, runtime_environment_versions,
        measure_realized_slopes, realized_environment_payload, TERRAIN_SOURCES, apply_trial_intervention)
    from g1_robustness_trial import RobustnessTrial
    from g1_robustness_physics import PhysicalInterventions
    root=Path(args.output_root).resolve();p,manifest,prepared=load_prepared(root)
    code=code_identity()
    if code['evaluation_runtime_dirty']:raise ValueError('Commit all runtime changes before physical execution')
    immutable_json(root/'runtime_commit.json',{k:code[k] for k in ('evaluation_code_commit','evaluation_runtime_sha256','evaluation_runtime_sources')})
    identity=model_identity(root,args.model,read_json(root/'standard_benchmark_five_model_index.json'))
    snapshot=snapshot_checkpoint(root,identity)
    smoke=prepared.get('validation_mode')=='wrench_frame_fix_physical_smoke'
    if smoke:
        from g1_wrench_validation import require_gate
        require_gate(root.parent/'api_smoke.json',code)
        if args.model not in ('ppo_plain','dwaq'):raise ValueError('Physical smoke is PPO/DWAQ only')
        if args.model=='dwaq':
            from g1_wrench_validation import check_model_smoke
            check_model_smoke(root,'ppo_plain')
    identity.update(prepared,evaluation_role='wrench_frame_fix_physical_smoke' if smoke else 'complete_robustness_suite',subset='fixed_four_force_pulse' if smoke else 'complete_paired_16128',
        trials=len(manifest),synthetic=False,realized_environment_schema_version=1)
    args.task=identity['task_name'];args.num_envs=prepared['num_envs'] if smoke else p['robustness']['num_envs'];args.headless=True
    key=evaluation_key(identity);run=root/'runs'/(key+'-attempt-0001');run.mkdir(parents=True,exist_ok=True)
    store=RobustnessStore(run)
    with lock(root/'.physical-execution.lock'),lock(run/'.run.lock'):
        if (run/'completion.json').exists():
            store.validate(workers=8);return {'status':'ALREADY_COMPLETE','evaluation_id':run.name}
        if (run/'failure.json').exists():raise ValueError('INVALID attempt preserved; physical-contract failure requires review')
        if not (run/'identity.json').exists():
            write_json(run/'identity.json',identity)
            for name,target in [('protocol.yaml','protocol_snapshot.yaml'),('manifest.jsonl','manifest_snapshot.jsonl'),('common_detector_config.yaml','common_detector_config.yaml')]:
                atomic_write(run/target,(root/name).read_bytes())
            for role,res in identity['resources'].items():
                if role!='estimator':atomic_write(run/res['snapshot'],(snapshot.parent/res['snapshot']).read_bytes())
            atomic_write(run/'native_configuration.json',(snapshot.parent/'native_configuration.json').read_bytes())
        else:
            previous=read_json(run/'identity.json')
            for field in ('evaluation_code_commit','evaluation_runtime_sha256','checkpoint_sha256','manifest_hash'):
                if previous[field]!=identity[field]:raise ValueError('Resume identity mismatch: '+field)
            if (run/'effective_env_config.yaml').exists():store.validate(require_complete=False,workers=8)
        done={r['trial_id'] for r in store.records()};app=None;machines=[];controller=None
        try:
            os.chdir(LAB);sys.path.insert(0,str(LAB));sys.path.insert(0,str(LAB/'rsl_rl'))
            from isaaclab.app import AppLauncher
            app=AppLauncher(headless=True,device=args.device).app
            env,runner,policy,weights,effective=make_evaluation_environment(args,p,identity,snapshot)
            effective['environment_versions']=runtime_environment_versions()
            effective['realized_slopes_deg']=measure_realized_slopes(env.eval_mesh,effective['terrain_origins'],effective['slopes_deg'])
            effective['terrain_source_sha256']={k:v for k,v in identity['evaluation_runtime_sources'].items() if k in TERRAIN_SOURCES}
            effective['realized_environment_hash']=digest(realized_environment_payload(effective))
            target=run/'effective_env_config.yaml'
            if target.exists() and load_yaml(target)!=effective:raise ValueError('Physical environment changed on resume')
            atomic_write(target,yaml.safe_dump(effective,sort_keys=False))
            identity.update(actual_physics_hash=effective['actual_physics_hash'],realized_environment_hash=effective['realized_environment_hash'])
            identity['software'].update({k:effective['environment_versions'][k] for k in ('IsaacLab','IsaacSim')})
            identity['hardware']['GPU']=torch.cuda.get_device_name(torch.device(args.device))
            write_json(run/'identity.json',identity)
            common={k:identity[k] for k in ('evaluation_code_commit','evaluation_runtime_sha256','actual_physics_hash',
                'realized_environment_hash','candidate_parameters_sha256','protocol_hash','metrics_config_hash','manifest_hash')}
            common['robot_masses_kg']=env.eval_masses.detach().cpu().tolist()
            immutable_json(root/'paired_physics_identity.json',common)
            controller=PhysicalInterventions(env,identity,p);start=time.monotonic()
            with torch.inference_mode():
                # Keep canonical batch slots even after resume: physical terrain origins
                # and stochastic native DWAQ inference remain paired by original trial.
                for offset in range(0,len(manifest),args.num_envs):
                    selected=manifest[offset:offset+args.num_envs]
                    if all(r['trial_id'] in done for r in selected):continue
                    controller.clear();obs,extra=env.begin_batch(selected)
                    machines=[RobustnessTrial(r,p) for r in selected]
                    controller.begin(machines)
                    frames=env.physical_snapshot()
                    for i,m in enumerate(machines):
                        state={'root_state_world':env.robot.data.root_state_w[i].cpu().tolist(),
                            'joint_position':env.robot.data.joint_pos[i].cpu().tolist(),
                            'joint_velocity':env.robot.data.joint_vel[i].cpu().tolist()}
                        immutable_json(root/'paired_initial_states'/f'{m.plan["trial_id"]}.json',state)
                        m.feed({**frames[i],'time':0.,'plane':env.plane_diagnostics(i)})
                        if m.plan['trial_id'] in done:m.status='ALREADY_COMMITTED'
                    step=0
                    while any(not m.status for m in machines):
                        history=env.eval_history_calls;actions=policy(obs,extra)
                        if not torch.isfinite(actions).all():raise ValueError('Nonfinite actor action')
                        obs,_,dones,extra=env.step(actions);step+=1
                        if env.eval_history_calls!=history+1:raise ValueError('Native actor history must advance exactly once')
                        if not torch.equal(env.command_generator.command,env.eval_commands):raise ValueError('Paired command changed')
                        if env.eval_capture is None:raise ValueError('Missing pre-reset GT frame')
                        for i,m in enumerate(machines):
                            if m.status:continue
                            frame=controller.endpoint_frames.pop(i,None)
                            if frame is None:frame={**env.eval_capture[i],'time':step*env.step_dt,
                                'plane':env.plane_diagnostics(i) if not bool(dones[i]) else None}
                            if m.feed(frame):
                                if m.sham:apply_trial_intervention(env,m,i,frame,identity)
                                else:controller.start(i,m,frame)
                            if m.status:
                                store.save_trial(m.result(),m.trace(),m.all_events());done.add(m.plan['trial_id'])
                        if step*env.step_dt>30:raise ValueError('Trial exceeded bounded readiness + disturbance + observation')
                    progress={'model':args.model,'evaluation_id':run.name,'completed':len(done),'designated':len(manifest),
                        'elapsed_s':time.monotonic()-start,'last_batch_offset':offset}
                    write_json(run/'progress.json',progress);print(progress,flush=True)
                controller.clear()
                if any(not torch.equal(v.cpu(),weights[k]) for k,v in runner.alg.policy.state_dict().items()):raise ValueError('Policy weights changed')
            store.complete(workers=8)
            immutable_json(root/'completed_models'/f'{args.model}.json',dict(model=args.model,evaluation_id=run.name,
                completion_sha256=sha256(run/'completion.json')))
            print({'status':'COMPLETE','evaluation_id':run.name},flush=True)
        except BaseException as exc:
            if controller is not None:controller.clear()
            for m in machines:
                if m.plan['trial_id'] not in done and m.frames:
                    m.finish('EVALUATION_ERROR');m.events.append({'event':'execution_error','time':m.t,'error':str(exc)})
                    store.save_trial(m.result(include_metrics=False),m.trace(),m.all_events())
            write_json(run/'failure.json',{'status':'INVALID','error':repr(exc),'completed_trials':len(done)})
            raise
        finally:
            if controller is not None:controller.clear()
            if app is not None:app.close()
    return {'status':'COMPLETE','evaluation_id':run.name}


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=('prepare','run','report'))
    a.add_argument('--output_root',default=str(ROOT));a.add_argument('--standard_index',default=str(INDEX))
    a.add_argument('--model',choices=MODEL_ORDER);a.add_argument('--device',default='cuda:0');args=a.parse_args()
    if args.command=='prepare':result=prepare(args.output_root,args.standard_index)
    elif args.command=='run':
        if args.model is None:a.error('--model required')
        result=execute(args)
    else:
        from g1_robustness_report import build_report
        result=build_report(Path(args.output_root))
    print(result,flush=True)
if __name__=='__main__':main()
