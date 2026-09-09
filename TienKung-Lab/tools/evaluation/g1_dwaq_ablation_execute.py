"""Run the frozen standard390 + reduced1960 for two new DWAQ checkpoints.

Never starts/stops training. Every evaluation subprocess is owned by this queue.
Old records, trial plans and physical initial states are read-only references.
"""
from pathlib import Path
from collections import Counter
import argparse
import copy
import os
import subprocess
import sys
import time
import yaml
from g1_recovery_protocol import (LAB, COMMON_PROTOCOL, code_identity, inspect_checkpoint,
    prepare, read_json, read_jsonl, write_json, sha256, digest, jsonl, atomic_write, lock)
from g1_recovery_protocol import load_prepared as load_standard
from g1_reduced_budget import ROOT as OLD, SOURCE, immutable
from g1_robustness_protocol import load_prepared

ROOT=LAB/'experiments/g1_dwaq_reward_ablation_eval_v1'
RUNS={
 'dwaq_v2':('g1_dwaq_slope_nosys_d_matched_v2','2026-09-09_01-12-29_dwaq_nosys_matched_no_idle'),
 'dwaq_v3':('g1_dwaq_slope_nosys_d_matched_v3','2026-09-09_01-13-06_dwaq_nosys_matched_no_swing_height')}
BASE_COMMIT='8af6cdd18a784b032aa373b0c5d3321f948e381a'
FIELDS=('actual_physics_hash','realized_environment_hash','candidate_parameters_sha256','protocol_hash','metrics_config_hash')


def expected_source(name,old):
    if name=='tools/evaluation/g1_recovery_protocol.py':
        old=old.replace("    'dwaq': 'g1_dwaq_slope_nosys_d_matched',\n", "    'dwaq': 'g1_dwaq_slope_nosys_d_matched',\n    'dwaq_no_idle': 'g1_dwaq_slope_nosys_d_matched_v2',\n    'dwaq_no_swing': 'g1_dwaq_slope_nosys_d_matched_v3',\n")
        old=old.replace('EVALUATION_RUNTIME_SOURCES = (\n', "EVALUATION_RUNTIME_SOURCES = (\n    'legged_lab/envs/g1/g1_reward_shaping_ablation_config.py',\n")
        old=old.replace('        if task == name:\n            return method', "        if task == name:\n            if method in ('dwaq_no_idle', 'dwaq_no_swing'):\n                return 'dwaq'\n            return method")
    if name=='tools/evaluation/g1_complete_robustness.py':
        old=old.replace("    old=LAB/'experiments/g1_recovery_eval_v2/runs'/row['evaluation_id']", "    old=LAB/row.get('run_path', 'experiments/g1_recovery_eval_v2/runs/'+row['evaluation_id'])")
    return old


def verify_runtime():
    code=code_identity()
    if code['evaluation_runtime_dirty']:raise ValueError('Commit runtime first')
    # Exact source transition, not a blanket waiver of differing runtime hashes.
    for name in code['evaluation_runtime_sources']:
        old=subprocess.check_output(['git','show',BASE_COMMIT+':TienKung-Lab/'+name],cwd=LAB,text=True)
        if (LAB/name).read_text()!=expected_source(name,old):raise ValueError('Unexpected runtime change: '+name)
    return code


def prepare_standard():
    code=verify_runtime()
    immutable(ROOT/'runtime_compatibility.json',dict(base_commit=BASE_COMMIT,
        scope='task registration, ablation config hashing and explicit sealed standard source path only',
        limit='Same physical and judge semantics; no claim of bitwise GPU determinism',
        execution_commit=code['evaluation_code_commit'],runtime_sha256=code['evaluation_runtime_sha256'],
        shared_gpu_with_training=True,source_sha256=code['evaluation_runtime_sources']))
    baseline=read_json(OLD/'standard_benchmark_five_model_index.json')
    parent=next(x for x in baseline['models'] if x['model']=='dwaq')
    old=LAB/'experiments/g1_recovery_eval_v2/runs'/parent['evaluation_id']
    for model,(task,run_name) in RUNS.items():
        cp=LAB/'logs'/task/run_name/'model_9999.pt'
        identity=inspect_checkpoint(task,cp,model,'final')
        assert identity['training_iteration']==9999 and identity['training_transitions']==983040000
        immutable(ROOT/model/'checkpoint_identity.json',dict(identity))
        out=ROOT/model/'standard';prepare(out,COMMON_PROTOCOL)
        if load_standard(out)[1]!=read_jsonl(old/'manifest_snapshot.jsonl'):raise ValueError('Standard plan differs')
    return code


def sealed(root):
    roots=list((root/'runs').glob('*')) if (root/'runs').exists() else []
    if any((r/'failure.json').exists() for r in roots):raise ValueError('Invalid evaluation preserved: '+str(root))
    finished=[r for r in roots if (r/'completion.json').exists() and read_json(r/'completion.json')['status']=='COMPLETE']
    if len(finished)>1:raise ValueError('Ambiguous completed evaluation')
    return finished[0] if finished else None


def check_physics(run,baseline):
    a,b=read_json(run/'identity.json'),read_json(baseline/'identity.json')
    for k in FIELDS:
        if a[k]!=b[k]:raise ValueError('Incompatible physical/detector identity: '+k)


def prepare_robustness(model,standard):
    master=ROOT/model
    target=master/'execution/dwaq'
    selection=read_json(OLD/'reduced_budget_manifest_v1.json')
    immutable(master/'reduced_budget_manifest_v1.json',selection)
    # Match the sorted full reduced queue used by Context-only; retain every trial field.
    rows=read_jsonl(OLD/'execution/context_only/manifest.jsonl')
    assert len(rows)==1960 and {x['trial_id'] for x in rows}=={x['trial_id'] for x in selection['trials']}
    ids=[r['trial_id'] for r in rows]
    immutable(master/'pending_execution_queues.json',{'dwaq':ids})
    immutable(master/'reduced_budget_reuse_index.json',{'entries':[dict(model='dwaq',target_trial_id=t,reuse_decision='PENDING_EXECUTION') for t in ids]})
    identity=read_json(standard/'identity.json')
    row=dict(model='dwaq',task=identity['task_name'],evaluation_id=standard.name,
             run_path=str(standard.relative_to(LAB)),checkpoint_path=identity['checkpoint_path'],
             checkpoint_sha256=identity['checkpoint_sha256'],completion_sha256=sha256(standard/'completion.json'))
    index={'models':[row]};immutable(master/'standard_benchmark_five_model_index.json',index)
    p,_,original=load_prepared(SOURCE)
    info={**original,'budget_mode':'reduced_budget_v1','reduced_model':'dwaq','reduced_budget_root':str(master),
          'standard_index_sha256':sha256(master/'standard_benchmark_five_model_index.json'),
          'reduced_manifest_sha256':sha256(master/'reduced_budget_manifest_v1.json'),'execution_batch_size':64,
          'execution_queue_sha256':sha256(master/'pending_execution_queues.json'),
          'reuse_index_sha256':sha256(master/'reduced_budget_reuse_index.json'),
          'manifest_hash':digest(rows),'counts_per_model':dict(Counter(r['suite'] for r in rows)),
          'trials_per_model':1960,'total_new_robustness_trials':1960}
    for name,data in {'protocol.yaml':yaml.safe_dump(p,sort_keys=False).encode(),'manifest.jsonl':jsonl(rows).encode(),
        'standard_benchmark_five_model_index.json':(master/'standard_benchmark_five_model_index.json').read_bytes(),
        'common_detector_config.yaml':(SOURCE/'common_detector_config.yaml').read_bytes()}.items():
        path=target/name
        if path.exists() and path.read_bytes()!=data:raise ValueError('Prepared input changed')
        if not path.exists():atomic_write(path,data)
    immutable(target/'prepared.json',info)
    for tid in ids:
        immutable(target/'paired_initial_states'/f'{tid}.json',read_json(SOURCE/'paired_initial_states'/f'{tid}.json'))
    load_prepared(target)
    return target


def run_one(command,out,label,state):
    if sealed(out):return sealed(out)
    log=ROOT/(label+'.log')
    if log.exists():raise ValueError('Previous incomplete attempt requires inspection: '+str(log))
    with log.open('xb',buffering=0) as stream:
        env=os.environ.copy();env['PYTHONPATH']=str(LAB/'rsl_rl')+':'+str(LAB)+':'+env.get('PYTHONPATH','')
        proc=subprocess.Popen(command,cwd=LAB,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(status='RUNNING',current=label,pid=proc.pid);write_json(ROOT/'status.json',state)
        since=None
        try:
            while True:
                time.sleep(10)
                run=sealed(out)
                if run:
                    since=since or time.time()
                    if proc.poll() is not None or time.time()-since>30:return run
                elif proc.poll() is not None:raise ValueError('Evaluator exited before COMPLETE: '+label)
                state['completed_current']=sum(1 for _ in out.glob('runs/*/trial_records/*.json'))
                write_json(ROOT/'status.json',state)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with lock(ROOT/'.queue.lock'):
        code=prepare_standard();state=dict(status='PREPARED',driver_pid=os.getpid(),commit=code['evaluation_code_commit'],completed=[])
        write_json(ROOT/'status.json',state)
        try:
            standards={}
            parent=next(x for x in read_json(OLD/'standard_benchmark_five_model_index.json')['models'] if x['model']=='dwaq')
            old=LAB/'experiments/g1_recovery_eval_v2/runs'/parent['evaluation_id']
            for model,(task,run_name) in RUNS.items():
                assert verify_runtime()['evaluation_code_commit']==state['commit']
                cp=LAB/'logs'/task/run_name/'model_9999.pt';out=ROOT/model/'standard'
                command=[sys.executable,'-u','-B','tools/evaluation/g1_recovery_eval.py','run','--protocol',str(COMMON_PROTOCOL),
                         '--task',task,'--checkpoint',str(cp),'--model_alias',model,'--checkpoint_stage','final',
                         '--suite','lite','--num_envs','8','--headless','--device','cuda:0','--output_root',str(out)]
                standards[model]=run_one(command,out,model+'_standard',state)
                check_physics(standards[model],old);state['completed'].append(model+'_standard')
            for model in RUNS:
                assert verify_runtime()['evaluation_code_commit']==state['commit']
                out=prepare_robustness(model,standards[model])
                command=[sys.executable,'-u','-B','tools/evaluation/g1_complete_robustness.py','run','--output_root',str(out),'--model','dwaq','--device','cuda:0']
                run=run_one(command,out,model+'_robustness',state)
                oldref=read_json(SOURCE/'completed_models/dwaq.json')
                check_physics(run,SOURCE/'runs'/oldref['evaluation_id']);state['completed'].append(model+'_robustness')
            state.update(status='PHYSICS_COMPLETE');write_json(ROOT/'status.json',state)
            from g1_dwaq_ablation_report import build_report
            build_report(ROOT)
            state.update(status='COMPLETE');write_json(ROOT/'status.json',state)
        except BaseException as exc:
            state.update(status='STOPPED_ERROR',error=repr(exc));write_json(ROOT/'status.json',state);raise

if __name__=='__main__':main()
