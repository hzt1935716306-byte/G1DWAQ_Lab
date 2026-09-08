"""Read-only provenance admission for the fixed reduced primary cohort."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import time
import yaml

from g1_recovery_protocol import LAB, read_json, read_jsonl, sha256, digest, write_json, atomic_write, jsonl
from g1_reduced_budget import ROOT, SOURCE, BASE_COMMIT, TARGET_COUNTS, immutable
from g1_robustness_protocol import MODEL_ORDER, load_prepared
from g1_audit_compatibility import AUDIT_BOUNDARIES, source_at, projection

COMPATIBLE_FIELDS=('actual_physics_hash','realized_environment_hash','candidate_parameters_sha256',
                   'protocol_hash','metrics_config_hash')


def runtime_to_baseline(identity):
    """Only the previously audited audit-policy changes may differ in old sources."""
    changed=[]
    for name,old_hash in identity['evaluation_runtime_sources'].items():
        baseline=source_at(BASE_COMMIT,name)
        expected=hashlib.sha256(baseline.encode()).hexdigest()
        if old_hash==expected:continue
        old=source_at(identity['evaluation_code_commit'],name)
        if hashlib.sha256(old.encode()).hexdigest()!=old_hash:raise ValueError('Source commit/hash mismatch')
        if name not in AUDIT_BOUNDARIES or projection(old,name)!=projection(baseline,name):
            raise ValueError('Cannot reuse physical/semantic runtime difference: '+name)
        changed.append(name)
    if digest(identity['evaluation_runtime_sources'])!=identity['evaluation_runtime_sha256']:raise ValueError('Bad runtime digest')
    return dict(baseline_commit=BASE_COMMIT,audit_only_different_files=changed)


def check_source(run, root=ROOT):
    from g1_robustness_store import RobustnessStore
    run=Path(run);root=Path(root)
    if not run.resolve().is_relative_to((SOURCE/'runs').resolve()):raise ValueError('Only the repaired cohort is admitted as an old robustness source')
    if (run/'failure.json').exists():raise ValueError('An INVALID source requires a separate unaffected-trial review')
    sealed=(run/'completion.json').exists()
    identity,records=RobustnessStore(run).validate(require_complete=sealed,validation_level='light')
    if any(r['status']=='EVALUATION_ERROR' for r in records):raise ValueError('Unreviewed execution error in source')
    if identity['evaluation_role']!='complete_robustness_suite':raise ValueError('No performance/development source')
    proof=runtime_to_baseline(identity)
    result=dict(source_run_id=run.name,source_run_path=str(run),source_status='COMPLETE' if sealed else 'PARTIAL',
        identity_sha256=sha256(run/'identity.json'),completion_sha256=sha256(run/'completion.json') if sealed else None,
        source_runtime_sha256=identity['evaluation_runtime_sha256'],record_count=len(records),
        records_sha256=digest(records),validation='LIGHT source files/identities/records + independent target references; final FULL primary audit required',
        physical_runtime_compatibility=proof,checked_at_unix=time.time())
    immutable(root/'source_checks'/f'{run.name}.json',result)
    return result


def build_reuse(root=ROOT):
    root=Path(root);master=read_json(root/'reduced_budget_manifest_v1.json');plans=master['trials']
    index=read_json(root/'standard_benchmark_five_model_index.json');models={r['model']:r for r in index['models']}
    expected=read_json(SOURCE/'paired_physics_identity.json')
    sources={m:[] for m in MODEL_ORDER};extended=[]
    for checkpath in sorted((root/'source_checks').glob('*.json')):
        check=read_json(checkpath);run=Path(check['source_run_path']);i=read_json(run/'identity.json')
        if sha256(run/'identity.json')!=check['identity_sha256']:raise ValueError('Source identity changed after LIGHT check')
        if check['completion_sha256'] and sha256(run/'completion.json')!=check['completion_sha256']:raise ValueError('Source seal changed')
        if any(i[k]!=expected[k] for k in COMPATIBLE_FIELDS):raise ValueError('Source physical identity mismatch')
        model=next((m for m in MODEL_ORDER if models[m]['checkpoint_sha256']==i['checkpoint_sha256']),None)
        if model is None:raise ValueError('Unknown checkpoint in source')
        standard_run=LAB/'experiments/g1_recovery_eval_v2/runs'/models[model]['evaluation_id']
        si=read_json(standard_run/'identity.json')
        for key in ('native_configuration_sha256','checkpoint_sha256','agent_config_sha256','env_config_sha256'):
            if si[key]!=i[key]:raise ValueError('Native model input mismatch: '+key)
        if {k:v['sha256'] for k,v in si['resources'].items()}!={k:v['sha256'] for k,v in i['resources'].items()}:raise ValueError('Native resource mismatch')
        rows=[read_json(p) for p in sorted((run/'trial_records').glob('*.json'))]
        if digest(rows)!=check['records_sha256']:raise ValueError('Source records changed after LIGHT check')
        lookup={r['trial_id']:r for r in rows};sources[model].append((run,i,check,lookup))
        wanted={r['trial_id'] for r in plans};extras=[r for r in rows if r['trial_id'] not in wanted]
        extended.append(dict(model=model,source_run_id=run.name,source_status=check['source_status'],
            extra_counts=dict(Counter(r['suite'] for r in extras)),extra_trial_ids=[r['trial_id'] for r in extras],
            use='Extended available data only; excluded from Primary matched cohort'))
    entries=[];queue={m:[] for m in MODEL_ORDER}
    for model in MODEL_ORDER:
        for plan in plans:
            tid=plan['trial_id'];entry=dict(model=model,checkpoint_sha256=models[model]['checkpoint_sha256'],
                target_trial_id=tid,suite=plan['suite'],source_run_id=None,source_trial_id=None,source_trace_sha256=None,
                source_record_sha256=None,source_events_sha256=None,source_runtime_sha256=None,
                reuse_decision='PENDING_EXECUTION',reuse_reason='No eligible saved source for the predetermined trial')
            for run,i,check,lookup in sorted(sources[model],key=lambda s:(s[2]['source_status']!='COMPLETE',s[0].name)):
                if tid not in lookup:continue
                record=lookup[tid]
                if any(record.get(k)!=v for k,v in plan.items()):raise ValueError('Saved trial input differs from fixed target')
                if record['status']=='EVALUATION_ERROR':continue
                trace=run/'traces'/f'{tid}.npz';events=run/'trial_events'/f'{tid}.jsonl';rp=run/'trial_records'/f'{tid}.json'
                if not trace.is_file() or not events.is_file() or sha256(trace)!=record['trace_sha256']:raise ValueError('Missing/corrupt source evidence')
                read_jsonl(events)
                entry.update(source_run_id=run.name,source_run_path=str(run),source_trial_id=tid,source_status=check['source_status'],
                    source_trace_sha256=record['trace_sha256'],source_record_sha256=sha256(rp),source_events_sha256=sha256(events),
                    source_runtime_sha256=i['evaluation_runtime_sha256'],source_identity_sha256=check['identity_sha256'],
                    source_completion_sha256=check['completion_sha256'],source_check_sha256=sha256(root/'source_checks'/f'{run.name}.json'),
                    **{k:i[k] for k in COMPATIBLE_FIELDS},reuse_decision='REUSE_EXISTING',
                    reuse_reason='Exact assigned inputs, admitted repaired runtime, native resources and LIGHT-verified independent record/trace/events; valid failures retained')
                break
            if entry['reuse_decision']=='PENDING_EXECUTION':queue[model].append(tid)
            entries.append(entry)
    immutable(root/'reduced_budget_reuse_index.json',dict(manifest_sha256=sha256(root/'reduced_budget_manifest_v1.json'),entries=entries))
    immutable(root/'pending_execution_queues.json',queue)
    immutable(root/'extended_available_index.json',dict(sources=extended))
    return {m:dict(target=1960,reused=1960-len(queue[m]),pending=len(queue[m])) for m in MODEL_ORDER}


def prepare_pending(root=ROOT):
    root=Path(root);master=read_json(root/'reduced_budget_manifest_v1.json');lookup={r['trial_id']:r for r in master['trials']}
    queues=read_json(root/'pending_execution_queues.json');p,_,info=load_prepared(SOURCE)
    for model,ids in queues.items():
        if not ids:continue
        rows=[lookup[tid] for tid in ids];target=root/'execution'/model
        prepared={**info,'budget_mode':'reduced_budget_v1','reduced_model':model,'reduced_budget_root':str(root),
            'reduced_manifest_sha256':sha256(root/'reduced_budget_manifest_v1.json'),'execution_batch_size':64,
            'execution_queue_sha256':sha256(root/'pending_execution_queues.json'),
            'reuse_index_sha256':sha256(root/'reduced_budget_reuse_index.json'),
            'manifest_hash':digest(rows),'counts_per_model':dict(Counter(r['suite'] for r in rows)),
            'trials_per_model':len(rows),'total_new_robustness_trials':len(rows)}
        for name,data in {'protocol.yaml':yaml.safe_dump(p,sort_keys=False),'manifest.jsonl':jsonl(rows),
            'standard_benchmark_five_model_index.json':(root/'standard_benchmark_five_model_index.json').read_bytes(),
            'common_detector_config.yaml':(SOURCE/'common_detector_config.yaml').read_bytes()}.items():
            path=target/name;content=data.encode() if isinstance(data,str) else data
            if path.exists() and path.read_bytes()!=content:raise ValueError('Pending input changed')
            if not path.exists():atomic_write(path,content)
        immutable(target/'prepared.json',prepared)
        # The existing executor checks these at the real reset before acting.
        # Bind every new reset to the original cross-model physical initial state.
        for row in rows:
            source=SOURCE/'paired_initial_states'/f"{row['trial_id']}.json"
            if not source.is_file():raise ValueError('Missing original paired physical reset')
            immutable(target/'paired_initial_states'/source.name,read_json(source))
    return queues


def verify_reference(entry,header_cache=None):
    run=Path(entry['source_run_path']);tid=entry['source_trial_id']
    for path,key in [(run/'trial_records'/f'{tid}.json','source_record_sha256'),
                     (run/'traces'/f'{tid}.npz','source_trace_sha256'),
                     (run/'trial_events'/f'{tid}.jsonl','source_events_sha256')]:
        if sha256(path)!=entry[key]:raise ValueError('Referenced source changed: '+str(path))
    header=(str(run),entry['source_identity_sha256'],entry.get('source_completion_sha256'))
    if header_cache is None or header not in header_cache:
        if sha256(run/'identity.json')!=entry['source_identity_sha256']:raise ValueError('Referenced identity changed')
        if entry.get('source_completion_sha256') and sha256(run/'completion.json')!=entry['source_completion_sha256']:
            raise ValueError('Referenced completion changed')
        if header_cache is not None:header_cache.add(header)
    return read_json(run/'trial_records'/f'{tid}.json')


def finalize_sources(root=ROOT):
    from g1_robustness_store import RobustnessStore
    from g1_reduced_compatibility import verify_proof
    root=Path(root);proof=verify_proof(root)
    original=read_json(root/'reduced_budget_reuse_index.json');entries=[];new={};headers=set()
    plans={r['trial_id']:r for r in read_json(root/'reduced_budget_manifest_v1.json')['trials']}
    expected=read_json(SOURCE/'paired_physics_identity.json')
    for model,ids in read_json(root/'pending_execution_queues.json').items():
        if not ids:continue
        execution=root/'execution'/model;ref=read_json(execution/'completed_models'/f'{model}.json')
        run=execution/'runs'/ref['evaluation_id']
        if sha256(run/'completion.json')!=ref['completion_sha256']:raise ValueError('New seal changed')
        identity,records=RobustnessStore(run).validate(validation_level='light')
        if identity['evaluation_runtime_sha256']!=proof['evaluation_runtime_sha256']:raise ValueError('Unapproved new runtime')
        if any(identity[k]!=expected[k] for k in COMPATIBLE_FIELDS):raise ValueError('New physics/metrics identity mismatch')
        if {r['trial_id'] for r in records}!=set(ids):raise ValueError('New run is not the exact pending set')
        new[model]=(run,identity,{r['trial_id']:r for r in records},ref['completion_sha256'])
    for old in original['entries']:
        e=dict(old);tid=e['target_trial_id'];model=e['model']
        if e['reuse_decision']=='REUSE_EXISTING':
            record=verify_reference(e,headers);e['execution_origin']='REUSED'
        else:
            run,i,records,completion=new[model];record=records[tid]
            if i['checkpoint_sha256']!=e['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            e.update(source_run_id=run.name,source_run_path=str(run),source_trial_id=tid,source_status='COMPLETE',
                source_trace_sha256=record['trace_sha256'],source_record_sha256=sha256(run/'trial_records'/f'{tid}.json'),
                source_events_sha256=sha256(run/'trial_events'/f'{tid}.jsonl'),source_identity_sha256=sha256(run/'identity.json'),
                source_completion_sha256=completion,source_runtime_sha256=i['evaluation_runtime_sha256'],
                **{k:i[k] for k in COMPATIBLE_FIELDS},execution_origin='NEW_EXECUTION')
            verify_reference(e,headers)
        if record['status']=='EVALUATION_ERROR' or any(record.get(k)!=v for k,v in plans[tid].items()):
            raise ValueError('Invalid or mismatched primary record')
        entries.append(e)
    if len(entries)!=9800:raise ValueError('Primary cohort incomplete')
    out=dict(manifest_sha256=sha256(root/'reduced_budget_manifest_v1.json'),
        reuse_index_sha256=sha256(root/'reduced_budget_reuse_index.json'),
        runtime_compatibility_sha256=sha256(root/'reduced_budget_runtime_compatibility.json'),entries=entries)
    immutable(root/'primary_matched_source_index.json',out)
    return out


def full_primary_audit(root=ROOT,workers=16):
    """One final CPU-only replay of primary references, excluding historical extras."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from g1_run_audit import replay_job
    from g1_recovery_protocol import load_yaml
    root=Path(root);index=read_json(root/'primary_matched_source_index.json');jobs=[];references=[];headers=set()
    p,_,_=load_prepared(SOURCE)
    plans={r['trial_id']:r for r in read_json(root/'reduced_budget_manifest_v1.json')['trials']}
    for e in index['entries']:
        verify_reference(e,headers);jobs.append((e['source_run_path'],plans[e['target_trial_id']],p,True))
        references.append(dict(model=e['model'],source_run_id=e['source_run_id'],trial_id=e['target_trial_id']))
    for model in read_json(root/'standard_benchmark_five_model_index.json')['models']:
        run=LAB/'experiments/g1_recovery_eval_v2/runs'/model['evaluation_id']
        if sha256(run/'completion.json')!=model['completion_sha256']:raise ValueError('Standard seal changed')
        protocol=load_yaml(run/'protocol_snapshot.yaml')
        for plan in read_jsonl(run/'manifest_snapshot.jsonl'):
            jobs.append((str(run),plan,protocol,False));references.append(dict(model=model['model'],source_run_id=run.name,trial_id=plan['trial_id']))
    start=time.time();results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for number,(reference,result) in enumerate(zip(references,pool.map(replay_job,jobs,chunksize=4)),1):
            results.append({**reference,**result})
            if number%128==0:print(f'Final primary FULL audit {number}/{len(jobs)}',flush=True)
    mismatches=[r for r in results if r.get('error') or r.get('execution_error_record')]
    out=dict(status='AUDIT_FAILED' if mismatches else 'AUDIT_PASSED',mode='full_primary_only',
        designated=len(jobs),replayed=sum(r['replayed'] for r in results),mismatch_count=len(mismatches),mismatches=mismatches,
        primary_source_index_sha256=sha256(root/'primary_matched_source_index.json'),wall_time_s=time.time()-start,
        source_runs_modified=False,automatic_physical_reruns=False)
    immutable(root/'audits/final_primary_full_audit.json',out)
    if mismatches:raise ValueError('Final primary audit failed; evidence retained')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check-source','reuse','prepare-pending']);p.add_argument('--run')
    a=p.parse_args()
    result=check_source(a.run) if a.command=='check-source' else build_reuse() if a.command=='reuse' else prepare_pending()
    print(json.dumps(result,indent=2))
