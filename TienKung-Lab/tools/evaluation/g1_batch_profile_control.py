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

def execute(spec,model,n,stage):
    target=ROOT/stage/model/f'n{n}';log=ROOT/f'{stage}_{model}_n{n}.log'
    if target.exists() or log.exists():raise ValueError('Do not overwrite performance attempts')
    command=[PYTHON,'-u','-B',str(LAB/'tools/evaluation/g1_batch_profile.py'),'run','--model',model,'--num_envs',str(n),'--device','cuda:0']
    if stage=='screen':command+=['--screen_seconds',str(spec['screen_seconds'])]
    start=time.time();reason=None
    with log.open('xb',buffering=0) as stream:
        proc=subprocess.Popen(command,cwd=LAB,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(ROOT/'execution_status.json',dict(status='PERFORMANCE_ONLY_RUNNING',pid=os.getpid(),stage=stage,model=model,n=n,child_pid=proc.pid))
        while proc.poll() is None:
            if (target/'performance_summary.json').exists():
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:stop_owned(proc)
                break
            if (target/'failure.json').exists():
                reason=read_json(target/'failure.json')['error'];stop_owned(proc);break
            progress=target/'progress.json'
            if progress.exists():
                state=read_json(progress);age=time.time()-state.get('updated_unix',progress.stat().st_mtime)
                if state['status']=='INITIALIZING' and time.time()-start>900:reason='INITIALIZATION_TIMEOUT'
                elif state['status']=='RUNNING' and age>spec['hang_no_step_seconds']:reason='HANG_NO_PROGRESS'
                elif state['status']=='VALIDATING' and age>900:reason='AUDIT_OR_SAVE_TIMEOUT'
                if reason:stop_owned(proc);break
            elif time.time()-start>900:reason='STARTUP_TIMEOUT';stop_owned(proc);break
            time.sleep(3)
    if (target/'performance_summary.json').exists():return read_json(target/'performance_summary.json')
    samples=read_jsonl(target/'telemetry.jsonl') if (target/'telemetry.jsonl').exists() else []
    values=lambda k:[x[k] for x in samples if k in x]
    progress=read_json(target/'progress.json') if (target/'progress.json').exists() else {}
    tail=log.read_text(errors='replace')[-20000:]
    result=dict(status='PERFORMANCE_ONLY_FAILED',reason=reason or 'PROCESS_EXIT_WITHOUT_COMPLETION',returncode=proc.returncode,wall_time_s=time.time()-start,
        OOM=any(x in tail.lower() for x in ('out of memory','cuda_error_out_of_memory')),
        formal_statistics_eligible=False,last_progress=progress,
        gpu_utilization_mean_percent=float(np.mean(values('gpu_utilization_percent'))) if values('gpu_utilization_percent') else None,
        peak_vram_mib=max(values('gpu_vram_mib'),default=None),peak_ram_bytes=max(values('process_tree_ram_bytes'),default=None),
        cpu_utilization_mean_percent=float(np.mean(values('cpu_percent'))) if values('cpu_percent') else None,
        certificate_query_count=progress.get('certificate_query_count'),certificate_profile=progress.get('certificate_profile'),
        peak_sampled_queued_chunks=max(values('certificate_queued_chunks_sample'),default=None),
        EVALUATION_ERROR=sum(read_json(x)['status']=='EVALUATION_ERROR' for x in (target/'trial_records').glob('*.json')))
    write_json(ROOT/f'{stage}_{model}_n{n}_stopped.json',result)
    return result

def choose_candidate(spec,summaries):
    shared=set.intersection(*(set(summaries[m]) for m in spec['models']))
    eligible=[n for n in shared if all(summaries[m][n]['warmed_policy_steps']>=100 and
        (not m.startswith('context') or summaries[m][n]['certificate_query_count']>0) for m in spec['models'])]
    gains={n:math.prod(summaries[m][n]['physics_env_steps_per_s']/summaries[m][64]['physics_env_steps_per_s'] for m in spec['models'])**.5 for n in eligible} if 64 in eligible else {}
    candidate=max(gains,key=gains.get) if gains else 64
    return (candidate if gains.get(candidate,0)>=1.1 else 64),gains

def main():
    gate=read_json(COHORT/'development/profiling_gate_status.json')
    if gate['status']!='READY_FOR_PERFORMANCE_ONLY':raise ValueError('DWAQ must finish and seal before profiling')
    spec=prepare();summaries={};stops={}
    for model in spec['models']:
        summaries[model]={}
        for n in spec['candidates']:
            result=execute(spec,model,n,'screen')
            if result['status']!='PERFORMANCE_ONLY_COMPLETE':stops[model]=result;break
            summaries[model][n]=result
            if n!=64 and result['warmed_policy_steps']>=100 and result['physics_env_steps_per_s']<summaries[model][64]['physics_env_steps_per_s']/spec['slowdown_ratio']:
                stops[model]=dict(reason='CLEARLY_SLOWER_SCREEN',n=n,next_larger_candidates_skipped=True);break
    candidate,gains=choose_candidate(spec,summaries);validation={};comparisons={}
    plans=[p for p in read_jsonl(ROOT/'manifest.jsonl') if p['trial_id'] in set(spec['validation_trial_ids'])]
    if candidate!=64:
        for model in spec['models']:
            validation[model]={}
            for n in [64,candidate]:
                result=execute(spec,model,n,'invariance');validation[model][n]=result
                if result['status']!='PERFORMANCE_ONLY_COMPLETE' or not result['validation_targets_complete']:break
        complete=all(all(validation.get(m,{}).get(n,{}).get('validation_targets_complete',False) for n in [64,candidate]) for m in spec['models'])
        if complete:
            old=COHORT/'runs'/read_json(COHORT/'completed_models/ppo_plain.json')['evaluation_id']
            comparisons['ppo_reproduces_sealed64']=compare(old,ROOT/'invariance/ppo_plain/n64',plans,'ppo_plain')
            for model in spec['models']:
                comparisons[model+'_64_vs_candidate']=compare(ROOT/'invariance'/model/'n64',ROOT/'invariance'/model/f'n{candidate}',plans,model)
            comparisons['ppo_sealed_vs_candidate']=compare(old,ROOT/'invariance/ppo_plain'/f'n{candidate}',plans,'ppo_plain')
    passed=bool(comparisons) and all(x['passed'] for x in comparisons.values())
    faster=bool(validation) and all(validation[m].get(candidate,{}).get('validation_trials_per_min',0)>1.1*validation[m][64].get('validation_trials_per_min',float('inf')) for m in spec['models'])
    decision=dict(status='PERFORMANCE_ONLY_ANALYSIS_COMPLETE',formal_statistics_eligible=False,best_performance_candidate=candidate,
        geometric_screen_environment_step_gains=gains,all_invariance_passed=passed,confirmed_fixed_subset_speedup=faster,
        selected_execution_batch_size=candidate if passed and faster else 64,summaries=summaries,validation=validation,stops=stops,
        comparison_summaries={k:{a:b for a,b in v.items() if a!='differences'} for k,v in comparisons.items()},
        next_formal_model='BLOCKED_PENDING_AGENT_REVIEW',threshold_changed=False)
    for name,data in comparisons.items():write_json(ROOT/'invariance'/f'{name}.json',data)
    write_json(ROOT/'decision.json',decision);write_json(ROOT/'execution_status.json',dict(status='PERFORMANCE_ONLY_ANALYSIS_COMPLETE',pid=os.getpid()))

if __name__=='__main__':main()
