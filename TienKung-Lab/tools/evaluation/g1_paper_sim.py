"""Fixed-time G1 paper simulator, reusing native inference and verified world wrench adapter."""
import argparse,copy,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import yaml
from g1_paper_recovery import ROOT,Trial,direction,VERSION,PhysicsImpulseLedger
from g1_recovery_protocol import LAB,read_json,write_json,sha256,load_yaml,digest,code_identity


def execute(args):
    import torch
    from g1_recovery_eval import snapshot_checkpoint,make_evaluation_environment,measure_realized_slopes,runtime_environment_versions
    from g1_world_wrench import set_world_wrenches,clear_wrench
    identities=read_json(ROOT/'checkpoint_identities.json');identity=identities[args.model]
    from g1_recovery_protocol import inspect_checkpoint,CheckpointIdentity
    fresh=inspect_checkpoint(identity['task_name'],LAB/identity['checkpoint_path'],args.model,'final',estimator=LAB/identity['estimator_path'] if identity.get('estimator_path') else None)
    for field in ['checkpoint_sha256','env_config_sha256','agent_config_sha256','native_configuration_sha256','resources']:
        if fresh[field]!=identity[field]:raise ValueError('Pinned native identity changed: '+field)
    identity=CheckpointIdentity(identity,inputs=fresh.inputs)
    rows=read_json(ROOT/(('formal_frozen' if args.stage in ('formal','validation') else args.stage)+'_manifest.json'))
    if args.suite:rows=[r for r in rows if r['suite']==args.suite]
    rows=rows[args.offset:]
    if args.limit:rows=rows[:args.limit]
    out=ROOT/args.stage/args.model/(args.suite or 'all')/args.attempt
    out.mkdir(parents=True,exist_ok=True)
    for folder in ['records','traces','events']:(out/folder).mkdir(exist_ok=True)
    runtime={**code_identity(),'paper_sources':{p:sha256(LAB/p) for p in ['tools/evaluation/g1_paper_recovery.py','tools/evaluation/g1_paper_sim.py']}}
    binding={'runtime':runtime,'checkpoint_sha256':identity['checkpoint_sha256'],'manifest_hash':digest(rows),'protocol_sha256':sha256(ROOT/('protocol.json' if args.stage=='pilot' else 'protocol_frozen.json')),'num_envs':args.num_envs,'stage':args.stage}
    if (out/'binding.json').exists() and read_json(out/'binding.json')!=binding:raise ValueError('Existing run runtime/input mismatch; preserve and use new attempt')
    write_json(out/'binding.json',binding)
    if (out/'failure.json').exists():raise ValueError('Prior evaluation error preserved; requires inspected new attempt')
    snapshot=snapshot_checkpoint(ROOT/args.model,identity)
    os.chdir(LAB);sys.path.insert(0,str(LAB));sys.path.insert(0,str(LAB/'rsl_rl'))
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device=args.device).app
    env=None;machines=[];active_count=0
    try:
        p=load_yaml(LAB/'tools/evaluation/configs/g1_recovery_eval_lite_v2.yaml')
        # Reuse only physical GT adapter schema; old common judge is NEVER called.
        p['slopes_deg']=[-10,0,10];p['manifest_seed']=26101000
        envargs=argparse.Namespace(task=identity['task_name'],num_envs=args.num_envs,device=args.device,headless=True)
        env,runner,policy,weights,effective=make_evaluation_environment(envargs,p,identity,snapshot)
        effective['realized_slopes_deg']=measure_realized_slopes(env.eval_mesh,effective['terrain_origins'],p['slopes_deg'])
        effective['versions']=runtime_environment_versions();(out/'effective_environment.yaml').write_text(yaml.safe_dump(effective,sort_keys=False))
        # Paired physical baseline independent of native observation/network structures.
        contract={k:effective[k] for k in ['actual_physics_hash','realized_slopes_deg']};contract['base_masses']=env.robot.root_physx_view.get_masses().cpu().tolist()
        cpath=ROOT/('paired_physics_'+str(args.num_envs)+'.json')
        if cpath.exists() and read_json(cpath)!=contract:raise ValueError('Methods physical baseline differs')
        write_json(cpath,contract)
        native_ranges=env.cfg.commands.ranges
        if native_ranges.lin_vel_x[1]<1.:raise ValueError('Native task command range does not support1m/s')
        ids,body_names=env.robot.find_bodies('torso_link',preserve_order=True)
        if body_names!=['torso_link']:raise ValueError('Unique torso required')
        body=int(ids[0]);base_mass=env.robot.root_physx_view.get_masses().clone();base_inertia=env.robot.root_physx_view.get_inertias().clone();total=base_mass.sum(1)
        original_write=env.scene.write_data_to_sim;original_update=env.scene.update;enabled=False;base_counter=0;impulses=np.zeros((args.num_envs,3));force_steps=np.zeros(args.num_envs,int);substep_falls={};ledger=None
        def write():
            if not enabled:return original_write()
            t=(int(env.sim_step_counter)-base_counter-1)*env.physics_dt
            forces=np.zeros((env.num_envs,3));active=[]
            for j,m in enumerate(machines[:active_count]):
                if m.status or m.onset is None or j in substep_falls:continue
                plan=m.plan;suite=plan['suite']
                if suite not in ('B1','C_force'):continue
                stop=m.release if suite=='B1' else 12.
                if m.onset-1e-9<=t<stop-1e-9:
                    magnitude=float(total[j])*(plan['strength']/.2 if suite=='B1' else plan['strength']*9.81)
                    forces[j]=direction(plan,True)*magnitude;active.append(j)
            link=env.robot.data.body_link_pos_w[:,body].detach().cpu().numpy();q=env.robot.data.body_link_quat_w[:,body].detach().cpu().numpy();point=env.robot.data.body_com_pos_w[:,body].detach().cpu().numpy()
            set_world_wrenches(env.robot.permanent_wrench_composer,forces,point,np.zeros_like(forces),link,q,body,env.device)
            ledger.write(forces)
            return original_write()
        def update(*a,**kw):
            result=original_update(*a,**kw)
            if enabled:
                ledger.advance(int(env.sim_step_counter))
                contact=env.contact_sensor.data.net_forces_w_history[:,:,env.termination_contact_cfg.body_ids]
                fall=torch.any(torch.max(torch.linalg.vector_norm(contact,dim=-1),dim=1)[0]>1.,dim=1).cpu().tolist()
                for j in range(active_count):
                    if fall[j] and j not in substep_falls:
                        substep_falls[j]=(int(env.sim_step_counter)-base_counter)*env.physics_dt
                        clear_wrench(env.robot.permanent_wrench_composer,[j])
            return result
        env.scene.write_data_to_sim=write;env.scene.update=update
        completed={p.stem for p in (out/'records').glob('*.json')};start=time.monotonic()
        with torch.inference_mode():
            for offset in range(0,len(rows),args.num_envs):
                selected=rows[offset:offset+args.num_envs];active_count=len(selected)
                if all(r['trial_id'] in completed for r in selected):continue
                padded=selected+[copy.deepcopy(selected[-1]) for _ in range(args.num_envs-active_count)]
                enabled=False;clear_wrench(env.robot.permanent_wrench_composer)
                mass=base_mass.clone();inertia=base_inertia.clone()
                for j,plan in enumerate(padded):
                    if plan['suite']=='C_load':
                        added=total[j]*plan['strength'];mass[j,body]+=added
                        # Spherical ballast at existing body CoM: no CoM shift/parallel-axis term.
                        for diag in [0,4,8]:inertia[j,body,diag]+=.4*added*.1**2
                indices=torch.arange(env.num_envs,dtype=torch.int32,device='cpu')
                env.robot.root_physx_view.set_masses(mass,indices);env.robot.root_physx_view.set_inertias(inertia,indices)
                if not torch.allclose(env.robot.root_physx_view.get_masses(),mass) or not torch.allclose(env.robot.root_physx_view.get_inertias(),inertia):raise ValueError('Payload mass/inertia write mismatch')
                env.eval_masses=mass.to(env.device)
                obs,extra=env.begin_batch(padded);machines=[Trial(r) for r in padded]
                for j in range(active_count,env.num_envs):machines[j].status='NOT_SCHEDULED'
                substep_falls.clear();base_counter=int(env.sim_step_counter)
                ledger=PhysicsImpulseLedger(args.num_envs,env.physics_dt,base_counter);impulses=ledger.impulse;force_steps=ledger.steps;enabled=True
                frames=env.physical_snapshot()
                for j,m in enumerate(machines[:active_count]):m.feed({**frames[j],'time':0.})
                for step in range(1,601):
                    if all(m.status for m in machines):break
                    history=env.eval_history_calls;actions=policy(obs,extra)
                    if not torch.isfinite(actions).all():raise ValueError('Nonfinite action')
                    obs,_,dones,extra=env.step(actions)
                    if env.eval_history_calls!=history+1:raise ValueError('Actor history advanced incorrectly')
                    if not torch.equal(env.command_generator.command,env.eval_commands):raise ValueError('Fixed command drift')
                    if env.eval_capture is None:raise ValueError('Missing pre-reset capture')
                    for j,m in enumerate(machines[:active_count]):
                        if m.status:continue
                        frame={**env.eval_capture[j],'time':step*env.step_dt}
                        if j in substep_falls:frame['fell']=True
                        due=m.feed(frame)
                        if due:
                            metadata={'mass_kg':float(total[j]),'application':'torso body CoM','force_world_N':(direction(m.plan,True)*float(total[j])*(m.plan['strength']/.2 if m.plan['suite']=='B1' else m.plan['strength']*9.81)).tolist() if m.plan['suite']!='B2' else None}
                            if m.plan['suite']=='B2':
                                before=env.robot.data.root_vel_w[j].clone();after=before.clone();after[:3]+=torch.tensor(direction(m.plan)*m.plan['strength'],device=env.device)
                                env.robot.write_root_velocity_to_sim(after[None],env_ids=torch.tensor([j],device=env.device));actual=env.robot.data.root_vel_w[j]
                                if not torch.allclose(actual,after,atol=5e-6,rtol=0) or not torch.equal(actual[3:],before[3:]):raise ValueError('Velocity delta/angular preservation violated')
                                metadata.update(velocity_before=before.cpu().tolist(),velocity_after=actual.cpu().tolist(),delta_velocity_world=(direction(m.plan)*m.plan['strength']).tolist())
                                # Preserve the existing evaluator's native Plane push-notification semantics.
                                if identity['method'].startswith('context') and not bool(env.recovery_active[j]):
                                    env._on_curriculum_push(torch.tensor([j],device=env.device),torch.tensor((direction(m.plan)[:2]*m.plan['strength'])[None],dtype=torch.float32,device=env.device),torch.zeros(1,dtype=torch.long,device=env.device))
                                    if not torch.equal(env.robot.data.root_vel_w[j],after):raise ValueError('Native notification altered velocity')
                            m.applied(frame,metadata)
                        if m.status:
                            tid=m.plan['trial_id'];record=m.result();record.update(method=args.model,task=identity['task_name'],checkpoint=identity['checkpoint_path'],checkpoint_sha256=identity['checkpoint_sha256'],training_seed=identity.get('training_seed'),mass_kg=float(mass[j].sum()),original_mass_kg=float(total[j]),payload_mass_kg=float(mass[j].sum()-total[j]),applied_impulse_Ns=impulses[j].tolist(),force_substeps=int(force_steps[j]),first_fall_time=substep_falls.get(j))
                            if m.plan['suite']=='B1' and m.onset is not None and record['observation_end']>=m.release-1e-9 and m.status!='FELL':
                                if force_steps[j]!=40:raise ValueError(f'Force pulse not exactly0.2s: trial={tid}, steps={force_steps[j]}, onset={m.onset}, release={m.release}')
                                expected=direction(m.plan,True)*float(total[j])*m.plan['strength']
                                if not np.allclose(impulses[j],expected,atol=1e-5):raise ValueError('Impulse contract mismatch')
                            if tid not in completed:
                                fields=['time','com_velocity','root_velocity','root_position','root_quaternion_wxyz','roll_pitch','yaw','angular_velocity_world_z','forces','physical_contact','touchdown_flags','command','fell','out_of_test_area','root_clearance_m']
                                target=out/'traces'/(tid+'.npz');np.savez_compressed(target,**{k:np.array([f[k] for f in m.frames]) for k in fields});record['trace_sha256']=sha256(target)
                                write_json(out/'events'/(tid+'.json'),m.events);write_json(out/'records'/(tid+'.json'),record);completed.add(tid)
                    if step%100==0:print(json.dumps({'stage':args.stage,'model':args.model,'suite':args.suite,'batch':offset,'sim_time':step*.02,'completed':len(completed),'designated':len(rows)}),flush=True)
                if any(not m.status for m in machines[:active_count]):raise ValueError('Episode exceeded12s')
                write_json(out/'progress.json',{'completed':len(completed),'designated':len(rows),'elapsed_s':time.monotonic()-start,'last_batch':offset});print(json.dumps(read_json(out/'progress.json')),flush=True)
        enabled=False;clear_wrench(env.robot.permanent_wrench_composer)
        if any(not torch.equal(v.cpu(),weights[k]) for k,v in runner.alg.policy.state_dict().items()):raise ValueError('Checkpoint weights modified')
        write_json(out/'completion.json',{'status':'COMPLETE','episodes':len(completed),'binding_sha256':sha256(out/'binding.json'),'records_sha256':{p.name:sha256(p) for p in sorted((out/'records').glob('*.json'))}})
    except BaseException as e:
        write_json(out/'failure.json',{'error':str(e),'traceback':traceback.format_exc()});raise
    finally:
        if env is not None:
            clear_wrench(env.robot.permanent_wrench_composer)
        app.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['ppo','dwaq','ours'],required=True);p.add_argument('--stage',choices=['pilot','calibration','formal','validation'],required=True);p.add_argument('--suite',choices=['A','B1','B2','C_force','C_load']);p.add_argument('--num_envs',type=int,default=64);p.add_argument('--device',default='cuda:0');p.add_argument('--offset',type=int,default=0);p.add_argument('--limit',type=int);p.add_argument('--attempt',default='attempt-001');execute(p.parse_args())
