"""Transactional robustness records with independent physical/replay validation."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import json
import math
import numpy as np

from g1_recovery_protocol import (RunStore, STATUSES, atomic_write, csv_bytes, digest,
    jsonl, load_yaml, read_json, read_jsonl, sha256, validate_resource_identity, write_json)
from g1_development_validation import realized_environment_payload
from g1_robustness_protocol import validate_protocol, waveform, waveform_sha
from g1_robustness_physics import vector_world, force_and_arm
from g1_robustness_trial import replay
from g1_common_task_detector import EPS
from g1_world_wrench import verify_wrench_sample


def verify_physics(plan,record,events,trace,p):
    onsets=[e for e in events if e['event']=='disturbance_onset']
    impacts=[e for e in events if e['event']=='velocity_jump']
    physics=[json.loads(str(x)) for x in trace['physics_json']]
    if not record['push_applied']:
        if onsets or impacts or physics:raise ValueError('Non-pushed trial has physical intervention evidence')
        return
    if len(onsets)!=1:raise ValueError('Exactly one real disturbance onset required')
    start=onsets[0]['time'];duration=plan['duration_s'];mass=onsets[0]['mass_kg'];family=plan['family']
    if not math.isfinite(mass) or mass<=0:raise ValueError('Measured positive robot mass required')
    expected=[]
    if family in ('velocity_jump','velocity_ood'):
        expected=[(start,vector_world(plan,onsets[0]['heading_yaw']))]
    elif family=='repeated_impulse':
        expected=[(start+x['offset_s'],np.asarray(x['delta_v_world_xy'])) for x in plan['impacts']
                  if start+x['offset_s']<min(record['observed_terminal_time'],record['first_terminal_physics_time'] if record['first_terminal_physics_time'] is not None else float('inf'))-EPS or
                  (record['survived'] and start+x['offset_s']<=record['observed_terminal_time']+EPS)]
    if len(impacts)!=len(expected):raise ValueError('Missing/additional physical velocity jumps')
    for index,(event,(t,delta)) in enumerate(zip(impacts,expected)):
        if abs(event['time']-t)>EPS or event['impact_index']!=index:raise ValueError('Impact timing/index differs from plan')
        before=np.asarray(event['velocity_before']);after=np.asarray(event['velocity_after'])
        target=before.copy();target[:2]+=delta
        if not np.isfinite(before).all() or not np.allclose(after,target,atol=5e-6,rtol=0):
            raise ValueError('Physical velocity increment differs from manifest')
        if not np.allclose(event['delta_v_world_xy'],delta,atol=1e-12,rtol=0):raise ValueError('Impact vector differs')
        if index==0:
            frame=np.flatnonzero(np.isclose(trace['time'],t,atol=EPS,rtol=0))
            if len(frame)!=1 or not np.allclose(before,trace['root_velocity_world'][frame[0]],atol=5e-6,rtol=0):
                raise ValueError('Initial impact before-state disagrees with GT sample')
    dt=p['robustness']['physics_dt_s']
    if not physics:raise ValueError('Missing physical substep evidence')
    wave=waveform(plan,dt) if family=='random_force' else None
    if wave is not None and waveform_sha(wave)!=plan['waveform_sha256']:raise ValueError('Waveform hash mismatch')
    previous=None;terminal_time=None;during=[]
    for s in physics:
        t=s['start_time']
        if previous is not None and t<=previous+EPS:raise ValueError('Physical substeps out of order')
        previous=t
        if abs(s['dt_s']-dt)>EPS or abs(s['time']-t-dt)>EPS:raise ValueError('Wrong actual force integration interval')
        if abs(s['mass_kg']-mass)>1e-6:raise ValueError('Mass changed during disturbance')
        force,arm=force_and_arm(plan,mass,t-start,wave,dt)
        if terminal_time is not None:force[:]=0
        if s['terminal'] and terminal_time is None:terminal_time=s['time']
        if t<=start+duration+EPS:during.append(t)
        if not np.allclose(s['force_requested_world_n'],force,atol=1e-9,rtol=1e-12):raise ValueError('Requested force differs from pinned waveform')
        actual=np.asarray(s['force_world_n']);composed=np.asarray(s['composed_force_world_n'])
        point=np.asarray(s['application_point_world_m']);com=np.asarray(s['whole_robot_com_world_m']);link=np.asarray(s['application_link_position_world_m'])
        if not all(np.isfinite(x).all() for x in (actual,composed,point,com,link)):raise ValueError('Nonfinite wrench evidence')
        verify_wrench_sample(s)
        tolerance=64*np.finfo(np.float32).eps*max(1,float(np.linalg.norm(force)))
        if not np.allclose(actual,force,atol=tolerance,rtol=0) or not np.allclose(composed,force,atol=tolerance,rtol=0):raise ValueError('Actual force differs from requested input')
        if not np.allclose(point-com,arm,atol=1e-4,rtol=0):raise ValueError('Application moment arm differs from protocol')
        if not np.allclose(s['torque_about_com_world_nm'],np.cross(point-com,actual),atol=tolerance,rtol=0):raise ValueError('Wrong torque about robot CoM')
        if not np.allclose(s['composed_torque_about_link_world_nm'],np.cross(point-link,actual),atol=tolerance,rtol=0):raise ValueError('Wrong actual link torque')
        if s['force_active']!=bool(np.any(force)):raise ValueError('Wrong force-active marker')
    # Every force-phase substep is required; a terminal first substep beyond the
    # force interval is also retained, but no post-release zero waveform is fabricated.
    last=min(start+duration,record['observed_terminal_time']-dt)
    expected_n=max(0,int(math.floor((last-start)/dt+1e-6))+1)
    if len(during)!=expected_n or (len(during)>1 and not np.allclose(np.diff(during),dt,atol=EPS,rtol=0)):
        raise ValueError('Missing disturbance substeps / incorrect release boundary')


def validate_trial(args):
    run_string,plan,p=args;run=Path(run_string);tid=plan['trial_id'];r=read_json(run/'trial_records'/f'{tid}.json')
    if any(r.get(k)!=v for k,v in plan.items()):raise ValueError('Record differs from assigned paired trial')
    if r['status'] not in STATUSES|{'SHAM_COMPLETED'}:raise ValueError('Unknown status')
    path=run/'traces'/f'{tid}.npz'
    if sha256(path)!=r['trace_sha256']:raise ValueError('Corrupted physical trace')
    events=read_jsonl(run/'trial_events'/f'{tid}.jsonl')
    with np.load(path,allow_pickle=False) as data:
        required={'time','root_velocity','root_velocity_world','com_velocity','roll_pitch','yaw','forces',
            'root_position','fell','timeout','out_of_test_area','plane_json','physics_json','command',
            'root_quaternion_wxyz','root_clearance_m','local_plane_normal','local_plane_point',
            'angular_velocity_world_z','data_valid','gravity_tilt_rad','physical_contact',
            'physical_touchdown_flags','alternating_touchdown_flag','post_push_sample','post_sham_sample'}
        if not required.issubset(data.files) or len(data['time'])==0:raise ValueError('Missing physical measurements')
        if any(len(data[k])!=len(data['time']) for k in required-{'physics_json'}):raise ValueError('Trace frame lengths differ')
        intervals=np.diff(data['time'])
        if len(intervals) and (not np.allclose(intervals[:-1],.02,atol=1e-7,rtol=0) or not 0<intervals[-1]<=.0200001):
            raise ValueError('Missing 50 Hz GT sample / exact physical endpoint')
        if len(intervals) and abs(intervals[-1]-.02)>1e-7:
            if r['planned_disturbance_end_time'] is None or abs(data['time'][-1]-r['planned_disturbance_end_time']-10)>EPS:
                raise ValueError('Only exact observation endpoint may shorten the final interval')
        if r['status']=='EVALUATION_ERROR':return tid
        if any(not np.isfinite(data[k]).all() for k in required-{'plane_json','physics_json'}):raise ValueError('Nonfinite measurement')
        verify_physics(plan,r,events,data,p)
        reconstructed=replay(plan,p,data,r,events)
        expected=reconstructed.result()
        # Initial state comes from the physical reset, not the recovery adapter.
        for k,v in expected.items():
            if r.get(k)!=v:
                from g1_run_audit import ReplayMismatch
                raise ReplayMismatch(k,r.get(k),v)
        rebuilt=reconstructed.trace()
        for k in ('physical_contact','physical_touchdown_flags','alternating_touchdown_flag','post_push_sample','post_sham_sample'):
            if not np.array_equal(rebuilt[k],data[k]):raise ValueError('Physical event or sample phase differs on replay')
    return tid


class RobustnessStore(RunStore):
    def rebuild_tables(self):
        # Per-trial JSON is the commit record. Rewriting all prior events after
        # each trial is quadratic for 16,128 trials; tables are derived at seal.
        pass

    def validate(self,require_complete=True,workers=1,*,validation_level='light',allow_synthetic=False):
        from g1_run_audit import validate_store
        return validate_store(self,require_complete,workers=workers,
                              validation_level=validation_level,allow_synthetic=allow_synthetic)

    def _validate_light(self,require_complete=True,*,allow_synthetic=False):
        i=read_json(self.path/'identity.json');p=load_yaml(self.path/'protocol_snapshot.yaml')
        contract=validate_protocol(p);manifest=read_jsonl(self.path/'manifest_snapshot.jsonl')
        if i.get('evaluation_role') not in ('complete_robustness_suite','wrench_frame_fix_physical_smoke') or i.get('synthetic'):
            raise ValueError('Explicit real robustness role required')
        if i['evaluation_role']=='wrench_frame_fix_physical_smoke':
            from g1_robustness_protocol import generate_plan,select_wrench_smoke
            if i.get('validation_mode')!='wrench_frame_fix_physical_smoke' or manifest!=select_wrench_smoke(generate_plan(p)):
                raise ValueError('Physical smoke manifest/role mismatch')
        if digest(p)!=i['protocol_hash'] or digest(manifest)!=i['manifest_hash']:raise ValueError('Protocol/plan identity mismatch')
        if any(i[k]!=v for k,v in contract.items()) or digest(p['metrics'])!=i['metrics_config_hash']:
            raise ValueError('Common detector configuration changed')
        if sha256(self.path/'common_detector_config.yaml')!=i['common_detector_config_sha256']:
            raise ValueError('Detector snapshot differs')
        validate_resource_identity(i)
        if digest(read_json(self.path/'native_configuration.json'))!=i['native_configuration_sha256']:
            raise ValueError('Native configuration snapshot mismatch')
        for role,res in i['resources'].items():
            if role=='estimator':continue
            if sha256(self.path/res['snapshot'])!=res['sha256']:raise ValueError('Native input SHA mismatch')
        e=load_yaml(self.path/'effective_env_config.yaml')
        if digest(e['actual_physics'])!=i['actual_physics_hash']:raise ValueError('Actual physics mismatch')
        if digest(realized_environment_payload(e))!=i['realized_environment_hash']:raise ValueError('Realized environment mismatch')
        if e['native_configuration_sha256']!=i['native_configuration_sha256']:raise ValueError('Effective native configuration differs')
        if e['native_inputs']!={k:v['sha256'] for k,v in i['resources'].items()}:raise ValueError('Native runtime inputs mismatch')
        plans={r['trial_id']:r for r in manifest};records=self.records()
        if len(plans)!=len(manifest) or any(r['trial_id'] not in plans for r in records):raise ValueError('Unassigned/duplicate trial')
        from g1_run_audit import light_trace
        for r in records:
            if any(r.get(k)!=v for k,v in plans[r['trial_id']].items()):raise ValueError('Record differs from assigned trial')
            if r.get('status') not in STATUSES|{'SHAM_COMPLETED'}:raise ValueError('Unknown status')
            light_trace(self.path/'traces'/f"{r['trial_id']}.npz",r,common=True,robustness=True)
            read_jsonl(self.path/'trial_events'/f"{r['trial_id']}.jsonl")
        if require_complete:
            if len(records)!=len(manifest) or any(r['status']=='EVALUATION_ERROR' for r in records):raise ValueError('Incomplete/invalid run')
            completion=read_json(self.path/'completion.json')
            if completion['status']!='COMPLETE':raise ValueError('Missing COMPLETE seal')
            for name,h in completion['files'].items():
                if sha256(self.path/name)!=h:raise ValueError('Sealed file changed: '+name)
        return i,records

    def complete(self,workers=8):
        identity,records=self.validate(require_complete=False,workers=workers,validation_level='sampled')
        if len(records)!=identity['trials'] or any(r['status']=='EVALUATION_ERROR' for r in records):
            raise ValueError('Cannot seal missing/invalid trials')
        atomic_write(self.path/'trials.csv',csv_bytes(records))
        from collections import Counter
        write_json(self.path/'summary.json',dict(status='COMPLETE',designated=identity['trials'],executed=len(records),
            statuses=dict(Counter(r['status'] for r in records))))
        files={str(p.relative_to(self.path)):sha256(p) for p in sorted(self.path.rglob('*'))
            if p.is_file() and p.name not in ('.run.lock','run.log','completion.json','progress.json')}
        if (self.path/'completion.json').exists():raise ValueError('Run already sealed')
        write_json(self.path/'completion.json',dict(status='COMPLETE',files=files,trial_count=len(records),
            protocol_hash=identity['protocol_hash'],manifest_hash=identity['manifest_hash'],
            validation='light structure and deterministic sampled replay; final full audit is explicit',
            validation_level='sampled',sampled_replay_audit_sha256=sha256(self.path/'sampled_replay_audit.json')))
        return identity,records
