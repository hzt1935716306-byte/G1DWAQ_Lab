"""CPU-only structural, sampled and explicit full audits; never steps a simulator."""
from __future__ import annotations
import hashlib,json,math,os,time,zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
from g1_recovery_protocol import read_json,read_jsonl,write_json,load_yaml,digest,sha256,canonical,COMMON_VERSION

LEVELS=('light','sampled','full')
SAMPLING_ALGORITHM='sampled_replay_v1'
MANDATORY=('RECOVERED_AND_SURVIVED','ALIVE_NOT_RECOVERED','FELL','OUT_OF_TEST_AREA','PRECONDITION_FAILED')

class ReplayMismatch(ValueError):
    def __init__(self,field,stored,replayed):
        self.field=field;self.stored=stored;self.replayed=replayed
        super().__init__('Stored/replayed mismatch: '+field)

def select_sample(evaluation_id,records):
    """Stable score tie-breaks; mandatory classes then greedy condition coverage."""
    n=len(records);k=min(n,64,max(16,math.ceil(.01*n)))
    score=lambda r:hashlib.sha256((evaluation_id+r['trial_id']+SAMPLING_ALGORITHM).encode()).hexdigest()
    ordered=sorted(records,key=lambda r:(score(r),r['trial_id']));chosen=[];ids=set()
    def add(r):
        if r is not None and r['trial_id'] not in ids:chosen.append(r);ids.add(r['trial_id'])
    for status in MANDATORY:add(next((r for r in ordered if r['status']==status),None))
    add(next((r for r in ordered if r.get('relapse_count',0)>0),None))
    def features(r):
        v={(key,str(r.get(key))) for key in ('family','suite','push_magnitude','slope_deg','direction_deg') if r.get(key) is not None}
        d=r.get('plane_diagnostics')
        # Use saved record diagnostics only; never run a detector to select audit samples.
        if isinstance(d,dict):
            for key,value in d.items():
                if any(x in key for x in ('valid','failure','failed')) and isinstance(value,(bool,int,float)):
                    v.add(('context_'+key,str(bool(value))))
                    if 'valid' in key and any(x in key for x in ('fraction','rate','ratio')):
                        v.add(('context_some_invalid',str(value<1)))
        return v
    seen=set().union(*(features(r) for r in chosen)) if chosen else set()
    while len(chosen)<k:
        remaining=[r for r in ordered if r['trial_id'] not in ids]
        r=max(remaining,key=lambda r:len(features(r)-seen));add(r);seen|=features(r)
    return sorted((r['trial_id'] for r in chosen))

def light_trace(path,record,*,common,robustness=False):
    """Read NPZ headers and timestamps, without replaying state or decompressing all arrays."""
    path=Path(path)
    if not isinstance(record.get('trace_sha256'),str) or len(record['trace_sha256'])!=64:raise ValueError('Missing trace_sha256 metadata')
    if not path.is_file() or sha256(path)!=record['trace_sha256']:raise ValueError('Missing/corrupt trace')
    required={'time','root_velocity','root_velocity_world','com_velocity','roll_pitch','yaw','forces','root_position','fell','timeout','out_of_test_area','plane_json'}
    if common and record['status']!='EVALUATION_ERROR':
        required|={'command','root_quaternion_wxyz','root_clearance_m','local_plane_normal','local_plane_point','angular_velocity_world_z','data_valid','gravity_tilt_rad','physical_contact','physical_touchdown_flags','alternating_touchdown_flag','post_push_sample'}
    if robustness:required|={'post_sham_sample','physics_json'}
    shapes={}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith('.npy'):raise ValueError('Unexpected NPZ member')
            with archive.open(member) as f:
                version=np.lib.format.read_magic(f)
                shape,_,dtype=np.lib.format._read_array_header(f,version)
                if dtype.hasobject:raise ValueError('Object/pickle trace arrays forbidden')
                shapes[member[:-4]]=shape
    if not required.issubset(shapes):raise ValueError('Missing required trace arrays')
    with np.load(path,allow_pickle=False) as trace:t=np.asarray(trace['time'])
    if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or t[0]<-1e-7:raise ValueError('Invalid trace timestamps')
    if any(not shapes[k] or shapes[k][0]!=len(t) for k in required-{'physics_json'}):raise ValueError('Trace frame counts differ')
    d=np.diff(t)
    if len(d):
        valid=np.allclose(d,.02,atol=1e-7,rtol=0)
        if robustness:
            valid=np.allclose(d[:-1],.02,atol=1e-7,rtol=0) and 0<d[-1]<=.0200001
            if abs(d[-1]-.02)>1e-7:
                end=record.get('planned_disturbance_end_time')
                valid=valid and end is not None and abs(t[-1]-end-10)<=1e-7
        if not valid:raise ValueError('Trace must contain legal consecutive 50 Hz timestamps')
    return t

def extra_light_checks(store,identity,records,manifest,*,robustness):
    if len({r['trial_id'] for r in manifest})!=len(manifest):raise ValueError('Duplicate manifest trial')
    if len({r['trial_id'] for r in records})!=len(records):raise ValueError('Duplicate record trial')
    if identity.get('evaluation_runtime_sources'):
        if digest(identity['evaluation_runtime_sources'])!=identity.get('evaluation_runtime_sha256'):raise ValueError('Runtime source hash identity mismatch')
    elif not identity.get('synthetic'):raise ValueError('Missing runtime source hash identity')
    required={'trial_id','status','push_applied','survived','recovery_time','recovery_steps','trace_sha256'}
    if identity.get('metrics_version')==COMMON_VERSION:
        required|={'recovered_sustained_and_survived','recovered_once_and_survived','relapse_count','precondition_passed'}
    for r in records:
        if not required.issubset(r):raise ValueError('Missing required record fields: '+r.get('trial_id','UNKNOWN'))
        if Path(r['trial_id']).name!=r['trial_id']:raise ValueError('Unsafe trial ID')
        if not (store.path/'trial_records'/f'{r["trial_id"]}.json').is_file():raise ValueError('Record filename/trial ID mismatch')
    completion_path=store.path/'completion.json'
    if robustness and completion_path.exists():
        completion=read_json(completion_path)
        required_files={'identity.json','protocol_snapshot.yaml','manifest_snapshot.jsonl','common_detector_config.yaml','effective_env_config.yaml','native_configuration.json','trials.csv','summary.json'}
        required_files.update(f'{folder}/{r["trial_id"]}{suffix}' for r in records for folder,suffix in [('traces','.npz'),('trial_records','.json'),('trial_events','.jsonl')])
        if not required_files.issubset(completion['files']):raise ValueError('Incomplete completion file index')
        if any(Path(name).is_absolute() or '..' in Path(name).parts for name in completion['files']):raise ValueError('Unsafe completion file path')
    summary=store.path/'summary.json'
    if summary.exists():
        if robustness:
            from collections import Counter
            expected=dict(status='INVALID' if any(r['status']=='EVALUATION_ERROR' for r in records) else 'COMPLETE' if len(records)==len(manifest) else 'PARTIAL',designated=len(manifest),executed=len(records),statuses=dict(Counter(r['status'] for r in records)))
        else:
            from g1_recovery_metrics import summarize
            expected=summarize(records,manifest)
        if canonical(expected)!=canonical(read_json(summary)):raise ValueError('Saved summary disagrees with trial records')
    audit=store.path/'sampled_replay_audit.json'
    if completion_path.exists():
        c=read_json(completion_path)
        if c.get('validation_level')=='sampled':
            if not audit.exists() or c.get('sampled_replay_audit_sha256')!=sha256(audit):raise ValueError('Completion sampled audit binding mismatch')
            if 'sampled_replay_audit.json' not in c['files']:raise ValueError('Sample audit missing from completion file index')
    if audit.exists():
        a=read_json(audit)
        if a['evaluation_id']!=store.path.name or a['runtime_sha']!=identity.get('evaluation_runtime_sha256','SYNTHETIC_UNBOUND') or a['mismatch_count']!=0:raise ValueError('Invalid sampled replay audit')
        if a['selected_trial_ids']!=select_sample(store.path.name,records):raise ValueError('Sampled replay assignment mismatch')
        if a['total_count']!=len(records) or a.get('records_sha256')!=digest(records):raise ValueError('Stale sampled replay audit')


def common_full_replay(plan,protocol,data,r,run):
    """Original common replay checks, retained without changing detector mathematics."""
    from g1_common_task_trial import replay_common_trial
    from g1_development_validation import ACCEPTANCE_FIELDS
    required={'time','root_velocity','root_velocity_world','com_velocity','roll_pitch','yaw','forces','root_position','fell','timeout','out_of_test_area','command','root_quaternion_wxyz','root_clearance_m','local_plane_normal','local_plane_point','angular_velocity_world_z','data_valid','gravity_tilt_rad','physical_contact','physical_touchdown_flags','alternating_touchdown_flag','post_push_sample'}
    if any(not np.isfinite(data[k]).all() for k in required):raise ValueError('Non-finite physical trace without EVALUATION_ERROR')
    replayed=replay_common_trial(plan,protocol,data,r);actual=replayed.result()
    for key in ('status','push_applied','survived','first_post_push_sample_time','first_recovery_entry','first_confirmation','sustained_recovery_entry','recovered_once_and_survived','recovered_sustained_and_survived','relapse_count','out_of_domain_duration_after_confirmation','recovery_time','recovery_steps','confirmation_steps','intervention_marker_applied','sham_applied','sham_marker_time','sham_task_entry','sham_task_confirmation','sham_continuity','sham_detection_latency','sham_complete_windows','sham_task_gate_failed_windows',*ACCEPTANCE_FIELDS):
        if r.get(key)!=actual.get(key):raise ReplayMismatch(key,r.get(key),actual.get(key))
    rebuilt=replayed.trace()
    for key in ('post_push_sample','physical_contact','physical_touchdown_flags','alternating_touchdown_flag'):
        if not np.array_equal(data[key],rebuilt[key]):raise ReplayMismatch(key,data[key].tolist(),rebuilt[key].tolist())
    if r.get('sham'):
        if 'post_sham_sample' not in data or not np.array_equal(data['post_sham_sample'],rebuilt['post_sham_sample']):raise ReplayMismatch('post_sham_sample',data['post_sham_sample'].tolist() if 'post_sham_sample' in data else None,rebuilt['post_sham_sample'].tolist())
        events=read_jsonl(Path(run)/'trial_events'/f'{r["trial_id"]}.jsonl')
        if canonical(events)!=canonical(replayed.all_events()):raise ReplayMismatch('events',events,replayed.all_events())


def replay_job(job):
    run,plan,p,robustness=job;run=Path(run);tid=plan['trial_id'];r=read_json(run/'trial_records'/f'{tid}.json')
    try:
        if r['status']=='EVALUATION_ERROR':return dict(trial_id=tid,replayed=False,error=None,execution_error_record=True)
        if robustness:
            from g1_robustness_store import validate_trial
            validate_trial((str(run),plan,p))
        elif r.get('metrics_version')==COMMON_VERSION:
            with np.load(run/'traces'/f'{tid}.npz',allow_pickle=False) as data:common_full_replay(plan,p,data,r,run)
        else:return dict(trial_id=tid,replayed=False,error=None,legacy_no_common_detector=True)
        return dict(trial_id=tid,replayed=True,error=None)
    except Exception as exc:
        return dict(trial_id=tid,replayed=False,error=dict(field=getattr(exc,'field','validation'),stored_value=getattr(exc,'stored',None),replayed_value=getattr(exc,'replayed',None),message=str(exc),trace_sha=r.get('trace_sha256')))


def audit_path(run,mode):
    run=Path(run)
    if mode=='sampled' and not (run/'completion.json').exists():return run/'sampled_replay_audit.json'
    stamp=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'_'+str(time.time_ns())
    base=run.parent.parent if run.parent.name=='runs' else run.parent
    return base/'audits'/run.name/stamp/(mode+'_replay_audit.json')


def run_replay_audit(store,identity,records,manifest,p,mode,*,workers=1,output=None,robustness=False):
    start=time.time();selected=select_sample(store.path.name,records) if mode=='sampled' else sorted(r['trial_id'] for r in records)
    plans={r['trial_id']:r for r in manifest};jobs=[(str(store.path),plans[tid],p,robustness) for tid in selected]
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            results=[]
            for count,result in enumerate(pool.map(replay_job,jobs,chunksize=4),1):
                results.append(result)
                if count%128==0:print(f'Explicit {mode} replay validated {count}/{len(jobs)}',flush=True)
    else:results=[replay_job(j) for j in jobs]
    mismatches=[dict(trial_id=r['trial_id'],**r['error']) for r in results if r['error']]
    from g1_recovery_protocol import code_identity
    code=code_identity();completion=store.path/'completion.json'
    out=dict(evaluation_id=store.path.name,status='AUDIT_FAILED' if mismatches else 'AUDIT_PASSED',mode=mode,
        completion_sha=sha256(completion) if completion.exists() else None,evaluation_runtime_sha=identity.get('evaluation_runtime_sha256','SYNTHETIC_UNBOUND'),runtime_sha=identity.get('evaluation_runtime_sha256','SYNTHETIC_UNBOUND'),
        audit_code_sha=code.get('audit_code_sha256',sha256(Path(__file__))),sampling_algorithm=SAMPLING_ALGORITHM if mode=='sampled' else 'all_trials_sorted_v1',
        records_sha256=digest(records),selected_trial_ids=selected,selected_count=len(selected),total_count=len(records),total_trials=len(records),replayed_trials=sum(r['replayed'] for r in results),
        mismatch_count=len(mismatches),mismatches=mismatches,mismatch_details=mismatches,
        execution_error_records=sum(r.get('execution_error_record',False) for r in results),
        started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(start)),completed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),wall_time_s=time.time()-start)
    target=Path(output) if output else audit_path(store.path,mode)
    if (store.path/'completion.json').exists() and target.resolve().is_relative_to(store.path.resolve()):raise ValueError('Sealed run audit must be outside raw run')
    if target.exists():raise ValueError('Audit output exists; do not overwrite')
    write_json(target,out);return out,target


def validate_store(store,require_complete=True,*,allow_synthetic=False,validation_level='light',workers=1):
    if validation_level not in LEVELS:raise ValueError('Unknown validation level')
    identity,records=store._validate_light(require_complete,allow_synthetic=allow_synthetic)
    robust=identity.get('evaluation_role') in ('complete_robustness_suite','wrench_frame_fix_physical_smoke')
    manifest=read_jsonl(store.path/'manifest_snapshot.jsonl');p=load_yaml(store.path/'protocol_snapshot.yaml')
    extra_light_checks(store,identity,records,manifest,robustness=robust)
    if validation_level=='sampled' and (store.path/'sampled_replay_audit.json').exists():
        return identity,records # LIGHT just verified the immutable assignment and record digest.
    if validation_level!='light':
        audit,path=run_replay_audit(store,identity,records,manifest,p,validation_level,workers=workers,robustness=robust)
        if audit['mismatch_count']:raise ValueError(f'{validation_level} replay mismatch (record or physical event/sample-role); audit: {path}')
    return identity,records


def explicit_audit(root,mode='full',workers=None):
    root=Path(root);workers=workers or min(16,max(1,len(os.sched_getaffinity(0))//2))
    runs=[root] if (root/'completion.json').exists() else sorted(p.parent for p in root.rglob('completion.json') if 'runs' in p.parts)
    if not runs:raise ValueError('No sealed runs found for explicit audit')
    results=[]
    for run in runs:
        i=read_json(run/'identity.json');robust=i.get('evaluation_role') in ('complete_robustness_suite','wrench_frame_fix_physical_smoke')
        if robust:
            from g1_robustness_store import RobustnessStore
            store=RobustnessStore(run)
        else:
            from g1_recovery_protocol import RunStore
            store=RunStore(run)
        start=time.time()
        try:
            identity,records=store.validate(validation_level='light')
            audit,path=run_replay_audit(store,identity,records,read_jsonl(run/'manifest_snapshot.jsonl'),load_yaml(run/'protocol_snapshot.yaml'),mode,workers=workers,robustness=robust)
        except Exception as exc:
            path=audit_path(run,mode);audit=dict(status='AUDIT_FAILED',evaluation_id=run.name,completion_sha=sha256(run/'completion.json'),evaluation_runtime_sha=i.get('evaluation_runtime_sha256'),mismatch_count=1,mismatch_details=[dict(trial_id=None,field='light_validation',stored_value=None,replayed_value=None,error=str(exc))],total_trials=i.get('trials'),replayed_trials=0,started_at=start,completed_at=time.time());write_json(path,audit)
        results.append(dict(evaluation_id=run.name,status=audit['status'],audit=str(path),mismatch_count=audit['mismatch_count']))
    return dict(mode=mode,status='AUDIT_FAILED' if any(r['mismatch_count'] for r in results) else 'AUDIT_PASSED',runs=results)


def report_audit_status(run):
    """Read audit evidence only. Missing historical samples never trigger replay."""
    run=Path(run);sample=run/'sampled_replay_audit.json'
    sampled='SAMPLED_REPLAY_NOT_AVAILABLE'
    if sample.exists():
        a=read_json(sample);sampled='SAMPLED_REPLAY_PASSED' if a['mismatch_count']==0 else 'AUDIT_FAILED'
    base=run.parent.parent if run.parent.name=='runs' else run.parent
    completion=sha256(run/'completion.json') if (run/'completion.json').exists() else None
    full=[]
    for p in sorted((base/'audits'/run.name).glob('*/full_replay_audit.json')):
        a=read_json(p)
        if completion is not None and a.get('completion_sha')==completion:full.append((p,a))
    status='FULL_EXPLICIT_AUDIT_NOT_AVAILABLE'
    if full:status='AUDIT_FAILED' if any(a.get('mismatch_count',0) for _,a in full) else 'FULL_REPLAY_PASSED'
    if sampled=='AUDIT_FAILED' or status=='AUDIT_FAILED':raise ValueError('AUDIT_FAILED: '+run.name)
    return dict(sampled=sampled,full=status,full_audit_files=[str(p) for p,_ in full])
