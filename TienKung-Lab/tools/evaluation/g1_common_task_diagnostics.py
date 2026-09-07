"""Read-only archived no-push acceptance diagnostics. No simulated recovery claims."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from g1_common_task_detector import TimestampWindow, PhysicalTouchdowns, classify_window, EPS
from g1_recovery_protocol import (RunStore, LAB, common_contract, load_yaml, read_jsonl,
                                  digest, sha256, write_json, atomic_write)


def offline_acceptance(source, protocol_path, output_root):
    source, output = Path(source).resolve(), Path(output_root).resolve()
    historical = (LAB / 'experiments/g1_recovery_eval').resolve()
    if output.is_relative_to(source) or output.is_relative_to(historical):
        raise ValueError('Offline diagnostics must not modify source sealed/historical directories')
    identity, records = RunStore(source).validate()
    if any(r['push_applied'] for r in records):
        raise ValueError('This diagnostic supports only existing unpushed traces, not post-push evaluation')
    p = load_yaml(protocol_path)
    _, contract = common_contract(p)
    before = {str(f.relative_to(source)): sha256(f) for f in source.rglob('*') if f.is_file()}
    analysis_id = digest({'source_completion': sha256(source / 'completion.json'), 'contract': contract,
                          'diagnostics_code': sha256(__file__),
                          'detector_code': sha256(Path(__file__).with_name('g1_common_task_detector.py'))})[:24]
    dest = output / 'development/offline_acceptance' / analysis_id
    if dest.exists():
        return {'status': 'ALREADY_ANALYZED', 'path': str(dest)}
    plans = {r['trial_id']: r for r in read_jsonl(source / 'manifest_snapshot.jsonl')}
    report = {'analysis_type': 'UNPUSHED_ACCEPTANCE_DIAGNOSTIC_ONLY', 'parameter_status': 'candidate_unvalidated',
              'source_evaluation_id': source.name, 'source_completion_sha256': sha256(source / 'completion.json'),
              'candidate_parameters_sha256': contract['candidate_parameters_sha256'],
              'config': p['metrics']['detector'], 'real_pushes': 0, 'real_recovery_conclusions': None,
              'full_gate_acceptance': None,
              'missing_measurements': ['archived root vertical clearance and independently recorded actual local plane'],
              'provenance': {'com_velocity': 'sealed GT CoM heading XY', 'root_velocity': 'sealed root heading velocity',
                             'command': 'sealed immutable manifest, fixed command checked by original runtime',
                             'tilt': 'acos(cos(root roll)*cos(root pitch)); archived XYZ Euler, not teacher-relative',
                             'angular_error': 'sealed root_velocity_world[5] minus manifest world-z command',
                             'clearance': 'NOT reconstructed from commanded slope; no zero substitution'},
              'semantics': 'W-complete windows with left boundary >=2s; frame endpoint rates are descriptive correlated samples. Recovery envelope here has NO push or recovery state machine.',
              'trials': {}}
    md = ['Common Task v2 — existing no-push trace diagnostic', '',
          'candidate_unvalidated；无真实push，无恢复/抗扰结论。完整gate接受率 N/A：旧trace未记录root竖直间隙及可独立核对的实际局部平面。',
          '仅诊断已记录速度、可由root Euler验证的重力倾角、world-Z角速度及近期公共触地；不以零代替高度。', '',
          '| Trial | 阶段包络（无推扰） | 完整窗口数 | 已有5项任务约束通过 | 加近期4次交替触地 | 完整gate |',
          '|---|---|---:|---:|---:|---|']
    for record in records:
        tid = record['trial_id']; plan = plans[tid]
        with np.load(source / 'traces' / (tid+'.npz'), allow_pickle=False) as z:
            times = z['time'].astype(float)
            if len(times) < 2 or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > .02+EPS):
                raise ValueError('Damaged/missing timestamps in archived trajectory')
            for key in ('com_velocity', 'roll_pitch', 'root_velocity_world', 'forces'):
                if not np.isfinite(z[key]).all():
                    raise ValueError('Non-finite archived measurements')
            contacts = PhysicalTouchdowns(p['metrics']['contact_force_threshold_n'], p['metrics']['contact_release_threshold_n'],
                                          p['metrics']['contact_stable_frames'], p['metrics']['contact_debounce_s'])
            windows = {phase: TimestampWindow(p['metrics']['detector'][phase]['window_s'], .02) for phase in ('readiness', 'recovery')}
            outcomes = {phase: [] for phase in windows}
            for i, t in enumerate(times):
                contacts.update(t, z['forces'][i])
                recent = contacts.recent_count(t, 1.2)
                roll, pitch = z['roll_pitch'][i].astype(float)
                signals = {'velocity_error_mps': float(np.linalg.norm(z['com_velocity'][i, :2].astype(float)-[plan['command_vx'], plan['command_vy']])),
                           'tilt_rad': float(np.arccos(np.clip(np.cos(roll)*np.cos(pitch), -1, 1))),
                           'yaw_rate_error_radps': float(z['root_velocity_world'][i, 5])-plan['command_yaw']}
                for phase, window in windows.items():
                    window.append(t, signals)
                    s = window.statistics()
                    if s is None or s['start'] < 2-EPS:
                        continue
                    gate = classify_window(s, p['metrics']['detector'][phase], time=float(t),
                                           terminal=bool(z['fell'][i]), out_of_area=bool(z['out_of_test_area'][i]))
                    gate.update(partial_available_checks_passed=all(v for v in gate['checks'].values() if v is not None),
                                recent_alternating_touchdowns=recent)
                    assert gate['passed'] is None
                    outcomes[phase].append(gate)
        trial = {'source_status': record['status'], 'push_applied': False, 'full_gate_acceptance': None, 'phases': {}}
        for phase, data in outcomes.items():
            n = len(data); passed = sum(x['partial_available_checks_passed'] for x in data)
            recent = sum(x['partial_available_checks_passed'] and x['recent_alternating_touchdowns'] >= 4 for x in data) if phase == 'readiness' else None
            trial['phases'][phase] = {'complete_windows': n, 'partial_passed': passed, 'partial_rate': passed/n if n else None,
                                     'partial_with_recent_touchdowns': recent, 'windows': data}
            md.append(f'| {tid} | {phase} | {n} | {passed}/{n} ({100*passed/n:.2f}%) | {recent if recent is not None else "不要求"} | N/A |')
        report['trials'][tid] = trial
    assert before == {str(f.relative_to(source)): sha256(f) for f in source.rglob('*') if f.is_file()}
    report['source_files_unchanged'] = True
    dest.mkdir(parents=True)
    write_json(dest / 'acceptance.json', report)
    atomic_write(dest / 'acceptance.md', '\n'.join(md)+'\n')
    atomic_write(dest / 'protocol.yaml', Path(protocol_path).read_bytes())
    return {'status': 'DIAGNOSTIC_ONLY', 'path': str(dest), 'full_gate_acceptance': None, 'source_files_unchanged': True}
