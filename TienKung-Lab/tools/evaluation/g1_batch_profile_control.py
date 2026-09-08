"""PERFORMANCE_ONLY scheduling and read-only batch invariance comparisons."""
from __future__ import annotations
import json,math,os,signal,subprocess,time
from pathlib import Path
import numpy as np
from g1_batch_profile import ROOT,COHORT,prepare
from g1_recovery_protocol import LAB,read_json,read_jsonl,write_json
PYTHON='/home/zt/miniconda3/envs/g1/bin/python'

def stop_owned(proc):
    proc.terminate()
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()

def compare(left,right,plans,model):
    diffs=[];fields=['status','push_applied','precondition_passed','push_start_time','push_end_time','disturbance_start_time','recovery_clock_start_time','survived','recovered_sustained_and_survived','recovery_time','recovery_steps','first_recovery_entry','first_confirmation','sustained_recovery_entry','relapse_count','failure_category','common_gate_failure_counts','common_complete_windows','common_task_gate_pass_windows','readiness_hold_confirmed','first_readiness_confirmation_time','first_post_push_sample_time','sustained_confirmation','physical_touchdown_count','post_push_event_counts']
    event_keys=['physical_touchdown_flags','alternating_touchdown_flag','physical_contact','post_push_sample']
    counts={}
    for plan in plans:
        tid=plan['trial_id'];a=read_json(left/'trial_records'/f'{tid}.json');b=read_json(right/'trial_records'/f'{tid}.json');changed=[]
        if a['trial_id']!=b['trial_id']:changed.append('trial_id')
        if a['status']=='EVALUATION_ERROR' or b['status']=='EVALUATION_ERROR':changed.append('EVALUATION_ERROR')
        for k in fields:
            
            if k not in a or k not in b:
                changed.append('missing_'+k);continue
            x,y=a[k],b[k]
            eq=abs(x-y)<=1e-7 if isinstance(x,float) and isinstance(y,(float,int)) else x==y
            if not eq:changed.append(k)
        with np.load(left/'traces'/f'{tid}.npz',allow_pickle=False) as x,np.load(right/'traces'/f'{tid}.npz',allow_pickle=False) as y:
            same_time=x['time'].shape==y['time'].shape and np.allclose(x['time'],y['time'],atol=1e-7,rtol=0)
            if not same_time:changed.append('trace_timestamps')
            for k in event_keys:
                if not same_time or not np.array_equal(x[k],y[k]):changed.append(k)
            if model.startswith('context'):
                xp=[json.loads(str(v)) for v in x['plane_json']];yp=[json.loads(str(v)) for v in y['plane_json']]
                if not same_time:changed.append('certificate_timeline')
                else:
                    for u,v in zip(xp,yp):
                        if (u is None)!=(v is None):changed.append('certificate_available');break
                        if u is None:continue
                        for k in ('N','context_valid','query_failed','query_category'):
                            if u.get(k)!=v.get(k):changed.append('certificate_'+k)
                        m,n=u.get('margin'),v.get('margin')
                        if (m is None)!=(n is None) or (m is not None and not math.isclose(m,n,abs_tol=1e-6,rel_tol=0)):changed.append('certificate_margin')
        for k in set(changed):counts[k]=counts.get(k,0)+1
        if changed:diffs.append(dict(trial_id=tid,fields=sorted(set(changed)),left={k:a.get(k) for k in fields},right={k:b.get(k) for k in fields}))
    return dict(passed=not diffs,trials=len(plans),different_trials=len(diffs),field_counts=counts,differences=diffs,left=str(left),right=str(right),tolerances={'time_s':1e-7,'margin':1e-6},criterion='Conservative exact discrete outcomes and event timeline; any difference retains64')

def main():
    gate=read_json(COHORT/'development/profiling_gate_status.json')
    if gate['status']!='READY_FOR_PERFORMANCE_ONLY':raise ValueError('DWAQ must finish and seal before profiling')
    spec=prepare();write_json(ROOT/'execution_status.json',dict(status='PERFORMANCE_ONLY_RUNNING',pid=os.getpid()))
    summaries={};stops={}
    for model in spec['models']:
        summaries[model]={};baseline_batches={}
        for n in spec['candidates']:
            target=ROOT/model/f'n{n}';log=ROOT/f'{model}_n{n}.log'
            if target.exists():raise ValueError('Do not overwrite performance attempts')
            command=[PYTHON,'-u','-B',str(LAB/'tools/evaluation/g1_batch_profile.py'),'run','--model',model,'--num_envs',str(n),'--device','cuda:0']
            start=time.time();reason=None;completed_seen=0
            with log.open('xb',buffering=0) as stream:
                proc=subprocess.Popen(command,cwd=LAB,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                write_json(ROOT/'execution_status.json',dict(status='PERFORMANCE_ONLY_RUNNING',pid=os.getpid(),model=model,n=n,child_pid=proc.pid))
                while proc.poll() is None:
                    summary=target/'performance_summary.json';failure=target/'failure.json';progress=target/'progress.json'
                    if summary.exists():
                        for _ in range(6):
                            if proc.poll() is not None:break
                            time.sleep(10)
                        if proc.poll() is None:stop_owned(proc)
                        break
                    if failure.exists():reason='EVALUATION_ERROR: '+read_json(failure)['error'];stop_owned(proc);break
                    if progress.exists():
                        s=read_json(progress);age=time.time()-s.get('updated_unix',progress.stat().st_mtime)
                        if age>spec['hang_no_step_seconds'] and s['status']!='INITIALIZING':reason='HANG_NO_PROGRESS';stop_owned(proc);break
                        if s['status']=='INITIALIZING' and time.time()-start>900:reason='INITIALIZATION_TIMEOUT';stop_owned(proc);break
                        # Compare completed physical batches at the same fixed prefix; no comparison during replay.
                        if n>64 and s['status']=='RUNNING':
                            metrics=sorted((target/'batch_metrics').glob('*.json'))
                            if metrics:
                                v=read_json(metrics[-1]);count=v['completed'];base=baseline_batches.get(count)
                                if count>=spec['minimum_slowdown_completed'] and base and v['elapsed_s']>spec['slowdown_ratio']*base:
                                    reason='CLEARLY_SLOWER_THAN64_AT_SAME_PREFIX';stop_owned(proc);break
                    time.sleep(5)
            if not (target/'performance_summary.json').exists():
                reason=reason or 'PROCESS_EXIT_WITHOUT_PERFORMANCE_COMPLETION'
                stopped=dict(reason=reason,returncode=proc.returncode,wall_time_s=time.time()-start,formal_statistics_eligible=False,next_larger_candidates_skipped=True)
                if (target/'progress.json').exists():stopped['last_progress']=read_json(target/'progress.json')
                samples=read_jsonl(target/'telemetry.jsonl') if (target/'telemetry.jsonl').exists() else []
                values=lambda k:[x[k] for x in samples if k in x]
                peak=lambda k:max(values(k),default=None)
                mean=lambda k:float(np.mean(values(k))) if values(k) else None
                records=[read_json(x) for x in (target/'trial_records').glob('*.json')]
                last=stopped.get('last_progress',{});elapsed=last.get('execution_wall_s')
                text=log.read_text(errors='replace')[-10000:]
                stopped.update(EVALUATION_ERROR=sum(r['status']=='EVALUATION_ERROR' for r in records),
                    OOM=any(x in text.lower() for x in ('out of memory','cuda_error_out_of_memory')),
                    completed_trials_per_min=len(records)*60/elapsed if elapsed else None,
                    physics_vector_steps_per_s=last.get('physics_vector_steps',0)/elapsed if elapsed else None,
                    gpu_utilization_mean_percent=mean('gpu_utilization_percent'),peak_vram_mib=peak('gpu_vram_mib'),
                    cpu_utilization_mean_percent=mean('cpu_percent'),peak_ram_bytes=peak('process_tree_ram_bytes'),
                    certificate_query_count=last.get('certificate_query_count'),query_failure_count=last.get('query_failure_count'),
                    certificate_profile=last.get('certificate_profile'),peak_sampled_queued_chunks=peak('certificate_queued_chunks_sample'),
                    aborted_incomplete_trials='Not classified as task outcomes; PERFORMANCE_ONLY aborted grade')
                write_json(ROOT/f'{model}_n{n}_stopped.json',stopped);stops[model]=stopped;break
            s=read_json(target/'performance_summary.json');summaries[model][n]=s
            if n==64:
                baseline_batches={v['completed']:v['elapsed_s'] for v in [read_json(x) for x in (target/'batch_metrics').glob('*.json')]}
            elif s['trial_execution_wall_time_s']>spec['slowdown_ratio']*summaries[model][64]['trial_execution_wall_time_s']:
                stops[model]=dict(reason='CLEARLY_SLOWER_COMPLETE',n=n,next_larger_candidates_skipped=True);break
    shared=set.intersection(*(set(x) for x in summaries.values()))
    gains={n:math.prod(summaries[m][n]['completed_trials_per_min']/summaries[m][64]['completed_trials_per_min'] for m in spec['models'])**.5 for n in shared} if 64 in shared else {}
    candidate=max(gains,key=gains.get) if gains else 64
    if gains.get(candidate,0)<1.1:candidate=64
    plans=read_jsonl(ROOT/'manifest.jsonl');comparisons={}
    ppo_ref=read_json(COHORT/'completed_models/ppo_plain.json');old=COHORT/'runs'/ppo_ref['evaluation_id']
    if 64 in summaries['ppo_plain']:
        comparisons['ppo_reproduces_sealed64']=compare(old,ROOT/'ppo_plain/n64',plans,'ppo_plain')
    if candidate!=64:
        comparisons['ppo_64_vs_candidate']=compare(ROOT/'ppo_plain/n64',ROOT/f'ppo_plain/n{candidate}',plans,'ppo_plain')
        comparisons['context_64_vs_candidate']=compare(ROOT/'context_only/n64',ROOT/f'context_only/n{candidate}',plans,'context_only')
        comparisons['ppo_sealed_vs_candidate']=compare(old,ROOT/f'ppo_plain/n{candidate}',plans,'ppo_plain')
    decision=dict(status='PERFORMANCE_ONLY_ANALYSIS_COMPLETE',formal_statistics_eligible=False,best_performance_candidate=candidate,
        geometric_throughput_gains=gains,all_invariance_passed=bool(comparisons) and all(v['passed'] for v in comparisons.values()),
        selected_execution_batch_size=candidate if candidate!=64 and comparisons and all(v['passed'] for v in comparisons.values()) else 64,
        summaries=summaries,stops=stops,comparison_summaries={k:{a:b for a,b in v.items() if a!='differences'} for k,v in comparisons.items()},
        next_formal_model='BLOCKED_PENDING_AGENT_REVIEW',threshold_changed=False)
    for name,data in comparisons.items():write_json(ROOT/'invariance'/f'{name}.json',data)
    write_json(ROOT/'decision.json',decision);write_json(ROOT/'execution_status.json',dict(status='PERFORMANCE_ONLY_ANALYSIS_COMPLETE',pid=os.getpid()))

if __name__=='__main__':main()
