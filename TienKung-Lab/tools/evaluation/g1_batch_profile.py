#!/usr/bin/env python3
"""Isolated PERFORMANCE_ONLY runner using unchanged native/evaluation primitives.

Never creates completed_models entries or a formal robustness completion seal.
"""
from __future__ import annotations
import argparse,json,os,sys,time,threading,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from g1_recovery_protocol import LAB,read_json,read_jsonl,write_json,atomic_write,digest,code_identity,sha256
from g1_robustness_protocol import load_prepared
COHORT=LAB/'experiments/g1_complete_robustness_v2/cohorts/wrench_fix_96e85ee1'
ROOT=LAB/'experiments/g1_batch_size_profiling_v2/PERFORMANCE_ONLY'

def prepare():
    p,full,prepared=load_prepared(COHORT)
    blocks=np.linspace(0,len(full)//64-1,64,dtype=int).tolist()
    selected=[r for b in blocks for r in full[b*64:(b+1)*64]]
    # Predeclare 128 validation IDs, spanning all eight suites; same order at every size.
    suites=sorted({r['suite'] for r in selected})
    validation=[r for suite in suites for r in [x for x in selected if x['suite']==suite][:16]]
    ids={r['trial_id'] for r in validation}
    selected=validation+[r for r in selected if r['trial_id'] not in ids]
    spec=dict(evaluation_role='PERFORMANCE_ONLY',formal_statistics_eligible=False,
        candidates=[64,128,256,512,1024,2046,4096],screen_seconds=90,validation_count=128,validation_trial_ids=[r['trial_id'] for r in validation],models=['ppo_plain','context_only'],
        trial_count=len(selected),source_manifest_sha256=sha256(COHORT/'manifest.jsonl'),
        canonical_64_batch_indices=blocks,manifest_hash=digest(selected),
        slowdown_ratio=1.2,minimum_slowdown_completed=256,hang_no_step_seconds=180,
        selection='90s warmed screening by environment physics steps/s; confirm fastest shared candidate on128 fixed IDs; final choice requires faster completed-workload throughput and exact invariance, otherwise64',
        invariance='Exact discrete outcomes/events; continuous time tolerance1e-7s; native margin atol1e-6; any discrete or timing change => retain64',
        native_profile_telemetry=True,code=code_identity(),harness_sha256=sha256(Path(__file__)))
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'plan.json').exists():
        if read_json(ROOT/'plan.json')!=spec:raise ValueError('Immutable performance plan differs')
    else:
        write_json(ROOT/'plan.json',spec)
        atomic_write(ROOT/'manifest.jsonl',''.join(json.dumps(r)+'\n' for r in selected))
    return spec

class Telemetry:
    def __init__(self,path):
        self.path=path;self.env=None;self.stop=threading.Event();self.thread=threading.Thread(target=self.sample,daemon=True)
    def sample(self):
        import psutil
        parent=psutil.Process();known={};previous=None;last=time.monotonic()
        while not self.stop.is_set():
            now=time.monotonic();procs=[parent]+parent.children(recursive=True);ram=0;cpu=0
            for p in procs:
                try:
                    ram+=p.memory_info().rss;c=p.cpu_times();v=c.user+c.system
                    if p.pid in known:cpu+=max(0,v-known[p.pid])
                    known[p.pid]=v
                except psutil.Error:pass
            item=dict(time=time.time(),process_tree_ram_bytes=ram,cpu_percent=100*cpu/max(now-last,1e-6),cpu_percent_definition='process tree summed core utilization;100%=one core',system_cpu_percent=psutil.cpu_percent())
            last=now
            try:
                s=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits','-i','0'],text=True,timeout=3)
                util,mem=map(float,s.strip().split(','));item.update(gpu_utilization_percent=util,gpu_vram_mib=mem)
            except Exception as exc:item['gpu_sample_error']=str(exc)
            evaluator=getattr(self.env,'_certificate_evaluator',None)
            executor=getattr(evaluator,'_executor',None)
            workers=getattr(executor,'_workers',())
            if evaluator is not None:
                item['certificate_queued_chunks_sample']=sum(w.queue.qsize() for w in workers)
                item['queue_definition']='queued worker chunks, sampled1Hz; excludes currently executing chunks'
            with self.path.open('a') as f:f.write(json.dumps(item)+'\n')
            self.stop.wait(1)
    def start(self):self.thread.start()
    def finish(self):self.stop.set();self.thread.join(timeout=5)

def run(args):
    import torch,yaml
    from g1_complete_robustness import model_identity
    from g1_recovery_eval import snapshot_checkpoint,make_evaluation_environment,runtime_environment_versions,measure_realized_slopes,realized_environment_payload,TERRAIN_SOURCES,apply_trial_intervention
    from g1_robustness_trial import RobustnessTrial
    from g1_robustness_physics import PhysicalInterventions
    from g1_robustness_store import RobustnessStore,validate_trial
    spec=read_json(ROOT/'plan.json');p,_,prepared=load_prepared(COHORT)
    manifest=read_jsonl(ROOT/'manifest.jsonl');assert digest(manifest)==spec['manifest_hash']
    assert args.num_envs in spec['candidates']
    code=code_identity();assert code['evaluation_runtime_sha256']==spec['code']['evaluation_runtime_sha256'] and not code['evaluation_runtime_dirty']
    target=ROOT/('screen' if args.screen_seconds else 'invariance')/args.model/f'n{args.num_envs}';target.mkdir(parents=True,exist_ok=False)
    telemetry=Telemetry(target/'telemetry.jsonl');telemetry.start();wall=time.monotonic();app=None;controller=None;machines=[];done=set();steps=0
    progress=dict(status='INITIALIZING',evaluation_role='PERFORMANCE_ONLY',pid=os.getpid(),completed=0,physics_vector_steps=0)
    write_json(target/'progress.json',progress)
    identity=model_identity(ROOT,args.model,read_json(COHORT/'standard_benchmark_five_model_index.json'))
    checkpoint=snapshot_checkpoint(ROOT,identity)
    identity.update(prepared,performance_stage='screen' if args.screen_seconds else 'invariance',evaluation_role='PERFORMANCE_ONLY',formal_statistics_eligible=False,execution_batch_size=args.num_envs,manifest_hash=digest(manifest),trials=len(manifest),harness_sha256=sha256(Path(__file__)))
    write_json(target/'identity.json',identity);atomic_write(target/'protocol_snapshot.yaml',yaml.safe_dump(p,sort_keys=False))
    atomic_write(target/'manifest_snapshot.jsonl',(ROOT/'manifest.jsonl').read_bytes());store=RobustnessStore(target)
    try:
        os.chdir(LAB);sys.path.insert(0,str(LAB));sys.path.insert(0,str(LAB/'rsl_rl'))
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,device=args.device).app
        args.task=identity['task_name'];args.headless=True
        env,runner,policy,weights,effective=make_evaluation_environment(args,p,identity,checkpoint)
        telemetry.env=env
        evaluator=getattr(env,'_certificate_evaluator',None)
        initial_query_sequence=getattr(evaluator,'_diagnostic_query_sequence',0)
        if args.model.startswith('context'):env.enable_plane_v1_profiling()
        effective['environment_versions']=runtime_environment_versions()
        effective['realized_slopes_deg']=measure_realized_slopes(env.eval_mesh,effective['terrain_origins'],effective['slopes_deg'])
        effective['terrain_source_sha256']={k:v for k,v in identity['evaluation_runtime_sources'].items() if k in TERRAIN_SOURCES}
        effective['realized_environment_hash']=digest(realized_environment_payload(effective))
        atomic_write(target/'effective_env_config.yaml',yaml.safe_dump(effective,sort_keys=False))
        identity.update(actual_physics_hash=effective['actual_physics_hash'],realized_environment_hash=effective['realized_environment_hash']);write_json(target/'identity.json',identity)
        controller=PhysicalInterventions(env,identity,p);execution_start=time.monotonic();query_ids=set();query_failure_ids=set();query_counts=0;failures=0;query_profiles={};budget_exhausted=False;targets=set(spec['validation_trial_ids'])
        with torch.inference_mode():
            for offset in range(0,len(manifest),args.num_envs):
                selected=manifest[offset:offset+args.num_envs]
                if len(selected)<args.num_envs:break  # Never pad or duplicate work.
                controller.clear();obs,extra=env.begin_batch(selected)
                machines=[RobustnessTrial(r,p) for r in selected];controller.begin(machines);frames=env.physical_snapshot()
                for i,m in enumerate(machines):
                    state=dict(root_state_world=env.robot.data.root_state_w[i].cpu().tolist(),joint_position=env.robot.data.joint_pos[i].cpu().tolist(),joint_velocity=env.robot.data.joint_vel[i].cpu().tolist())
                    write_json(target/'initial_states'/f'{m.plan["trial_id"]}.json',state)
                    m.feed({**frames[i],'time':0.,'plane':env.plane_diagnostics(i)})
                step=0
                while any(not m.status for m in machines):
                    if args.screen_seconds and time.monotonic()-execution_start>=args.screen_seconds:
                        budget_exhausted=True;break
                    if not args.screen_seconds and targets.issubset(done):break
                    history=env.eval_history_calls;actions=policy(obs,extra)
                    if not torch.isfinite(actions).all():raise ValueError('Nonfinite policy action')
                    obs,_,dones,extra=env.step(actions);step+=1;steps+=1
                    if env.eval_history_calls!=history+1:raise ValueError('Actor history contract failed')
                    if not torch.equal(env.command_generator.command,env.eval_commands):raise ValueError('Command changed')
                    if env.eval_capture is None:raise ValueError('Missing GT frame')
                    evaluator=getattr(env,'_certificate_evaluator',None)
                    if evaluator is not None:
                        for d in evaluator.last_query_diagnostics.values():
                            q=d['query_id']
                            if q not in query_ids:
                                query_ids.add(q);query_counts+=1
                                if d['failed']:query_failure_ids.add(q);failures+=1
                        query_counts=int(evaluator._diagnostic_query_sequence)-initial_query_sequence
                        if step%100==0:query_profiles=evaluator.profile_statistics
                    for i,m in enumerate(machines):
                        if m.status:continue
                        frame=controller.endpoint_frames.pop(i,None)
                        if frame is None:frame={**env.eval_capture[i],'time':step*env.step_dt,'plane':env.plane_diagnostics(i) if not bool(dones[i]) else None}
                        if m.feed(frame):
                            if m.sham:apply_trial_intervention(env,m,i,frame,identity)
                            else:controller.start(i,m,frame)
                        if m.status:store.save_trial({**m.result(),"performance_only":True,"formal_statistics_eligible":False},m.trace(),m.all_events());done.add(m.plan['trial_id'])
                    if step*env.step_dt>30:raise ValueError('Trial exceeded bounded duration')
                    if step%5==0:
                        progress.update(status='RUNNING',completed=len(done),physics_vector_steps=steps*4,execution_wall_s=time.monotonic()-execution_start,updated_unix=time.time(),certificate_query_count=query_counts,query_failure_count=failures,certificate_profile=query_profiles)
                        write_json(target/'progress.json',progress)
                write_json(target/'batch_metrics'/f'{offset:05d}.json',dict(completed=len(done),elapsed_s=time.monotonic()-execution_start,physics_vector_steps=steps*4))
                if budget_exhausted or (not args.screen_seconds and targets.issubset(done)):break
            controller.clear()
            if any(not torch.equal(v.cpu(),weights[k]) for k,v in runner.alg.policy.state_dict().items()):raise ValueError('Policy weights changed')
        elapsed=time.monotonic()-execution_start
        if evaluator is not None:
            query_profiles=evaluator.profile_statistics
            write_json(target/'native_profile_summary.json',env.plane_v1_profile_summary())
        # Unfinished workload instances are censored PERFORMANCE_ONLY artifacts, not task failures.
        for m in machines:
            if m.plan['trial_id'] not in done and m.frames:
                import io
                stream=io.BytesIO();np.savez_compressed(stream,**m.trace())
                atomic_write(target/'partial_traces'/f"{m.plan['trial_id']}.npz",stream.getvalue())
        progress.update(status='VALIDATING',completed=len(done),updated_unix=time.time());write_json(target/'progress.json',progress)
        from g1_run_audit import light_trace,select_sample
        profile_records=store.records()
        for j,r in enumerate(profile_records):
            light_trace(target/'traces'/f"{r['trial_id']}.npz",r,common=True,robustness=True)
            if j%32==0:progress.update(light_validated=j,updated_unix=time.time());write_json(target/'progress.json',progress)
        selected_ids=set(select_sample(target.name+'_'+args.model,profile_records))
        for row in manifest:
            if row['trial_id'] in selected_ids:validate_trial((str(target),row,p))
        write_json(target/'sampled_replay_audit.json',dict(evaluation_role='PERFORMANCE_ONLY',selected_trial_ids=sorted(selected_ids),selected_count=len(selected_ids),total_count=len(profile_records),mismatch_count=0))
        telemetry.finish();samples=read_jsonl(target/'telemetry.jsonl')
        values=lambda k:[x[k] for x in samples if k in x]
        summary=dict(status='PERFORMANCE_ONLY_COMPLETE',formal_statistics_eligible=False,execution_batch_size=args.num_envs,completed=len(done),screen_seconds=args.screen_seconds,budget_exhausted=budget_exhausted,validation_targets_complete=targets.issubset(done),warmed_policy_steps=steps,wall_time_s=time.monotonic()-wall,trial_execution_wall_time_s=elapsed,
            completed_trials_per_min=len(done)*60/elapsed,validation_trials_per_min=len(targets & done)*60/elapsed,physics_vector_steps_per_s=steps*4/elapsed,physics_env_steps_per_s=steps*4*args.num_envs/elapsed,
            gpu_utilization_mean_percent=float(np.mean(values('gpu_utilization_percent'))),peak_vram_mib=max(values('gpu_vram_mib')),
            cpu_utilization_mean_percent=float(np.mean(values('cpu_percent'))),peak_ram_bytes=max(values('process_tree_ram_bytes')),EVALUATION_ERROR=0,
            certificate_query_count=query_counts if args.model.startswith('context') else None,query_failure_count=failures if args.model.startswith('context') else None,
            certificate_profile=query_profiles,peak_sampled_queued_chunks=max(values('certificate_queued_chunks_sample'),default=None),
            query_latency_definition='Native worker_wait_ms/worker_solve_ms and environment synchronous solve timing; chunk-level distributions, not fabricated per-query latency',
            timing_definition='Throughput excludes initialization/replay; wall_time includes both; GPU/RAM/CPU sampled1Hz incl initialization and replay',
            physics_steps_definition='physics dt=.005s; vector steps and env transitions separately reported; no duplicate trials/padding',
            validation='LIGHT completed records plus deterministic SAMPLED replay; unfinished performance workload censored, no automatic full replay',sampled_replay_count=len(selected_ids))
        write_json(target/'performance_summary.json',summary);progress.update(status='PERFORMANCE_ONLY_COMPLETE',updated_unix=time.time());write_json(target/'progress.json',progress)
    except BaseException as exc:
        if controller is not None:controller.clear()
        for m in machines:
            if m.plan['trial_id'] not in done and m.frames:
                m.finish('EVALUATION_ERROR');store.save_trial({**m.result(include_metrics=False),"performance_only":True,"formal_statistics_eligible":False},m.trace(),m.all_events())
        write_json(target/'failure.json',dict(status='PERFORMANCE_ONLY_FAILED',error=repr(exc),completed=len(done)));raise
    finally:
        telemetry.finish()
        if controller is not None:controller.clear()
        if app is not None:app.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run']);a.add_argument('--model',choices=['ppo_plain','context_only']);a.add_argument('--num_envs',type=int);a.add_argument('--device',default='cuda:0');a.add_argument('--screen_seconds',type=float,default=0);args=a.parse_args()
    if args.command=='prepare':print(prepare())
    else:run(args)
