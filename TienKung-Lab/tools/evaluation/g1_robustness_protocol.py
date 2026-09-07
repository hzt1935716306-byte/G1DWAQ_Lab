"""Independent, preregistered five-model robustness plan; no simulator imports.

The standard 390 protocol, manifests and common detector remain untouched.
Repeated impacts include both t=0 and t=8 (17/9 impacts); release is the last
actual impact. Family vectors are world XY; envelope vectors use onset heading.
"""
from __future__ import annotations
import copy
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import yaml

from g1_recovery_protocol import (LAB, COMMON_PROTOCOL, COMMON_VERSION, atomic_write,
    code_identity, common_contract, digest, jsonl, load_yaml, read_json, read_jsonl,
    sha256, write_json)

CONFIG = LAB / 'tools/evaluation/configs/g1_complete_robustness_v2.yaml'
BASE_SHA = '70e168e6c11642915d902a4a591520d2241d9d12e402b0e697fce9f845315ba2'
MODEL_ORDER = ('ppo_plain', 'dwaq', 'rl_only', 'context_only', 'context_reward')
LABELS = dict(zip(MODEL_ORDER, ('PPO', 'DWAQ', 'RL-only', 'Context-only', 'Context+Reward')))
FAMILIES = ('velocity_ood', 'force_pulse', 'constant_force', 'repeated_impulse', 'random_force', 'wrench_pulse')
ROLES = ('extreme_velocity_envelope', 'disturbance_family_suite')
COUNTS = {'extreme_main': 8320, 'extreme_ood': 3328, 'velocity_ood': 768,
          'force_pulse': 1152, 'constant_force': 640, 'repeated_impulse': 768,
          'random_force': 768, 'wrench_pulse': 384}
PAIR_COMPARISONS = (('context_only', 'rl_only'), ('context_reward', 'context_only'),
    ('context_reward', 'rl_only'), ('context_reward', 'ppo_plain'), ('context_reward', 'dwaq'))


def protocol(config=CONFIG):
    if sha256(COMMON_PROTOCOL) != BASE_SHA:
        raise ValueError('Protected standard protocol changed')
    spec = load_yaml(config)
    if spec != load_yaml(CONFIG):
        raise ValueError('Robustness requires the exact committed preregistration')
    if spec['base_protocol_sha256'] != BASE_SHA:
        raise ValueError('Wrong base protocol binding')
    base = load_yaml(COMMON_PROTOCOL)
    _, contract = common_contract(base)
    p = copy.deepcopy(base)
    p['protocol_id'] = spec['protocol_id']
    p['manifest_seed'] = spec['manifest_seed']
    p['slopes_deg'] = sorted(spec['main_slopes_deg'] + spec['ood_slopes_deg'])
    p['robustness'] = spec
    assert p['metrics'] == base['metrics']
    return p, contract


def validate_protocol(p):
    expected, contract = protocol()
    if p != expected:
        raise ValueError('Robustness protocol differs from committed specification')
    return contract


def initial_state(seed, p):
    rng = random.Random(seed)
    init = p['initial_state']
    sample = lambda ranges: {k: rng.uniform(*v) for k, v in ranges.items()}
    return dict(initial_pose_parameters=sample(init['pose']),
                initial_velocity_parameters=sample(init['velocity']),
                initial_joint_parameters={
                    'position_scale': [rng.uniform(*init['joint_position_scale']) for _ in range(init['joint_count'])],
                    'velocity': [rng.uniform(*init['joint_velocity']) for _ in range(init['joint_count'])]})


def waveform(plan, dt=.005):
    """Pre-generated stationary OU samples; RMS parameter means vector RMS.

    Exact discrete OU covariance exp(-dt/tau); do not renormalize a finite
    realization or a truncated (fallen) prefix. Save requested and actual RMS.
    """
    n = round(plan['duration_s'] / dt)
    rng = np.random.default_rng(plan['waveform_seed'])
    sigma = plan['rms_acceleration_mps2'] / math.sqrt(2.)
    alpha = math.exp(-dt / plan['correlation_time_s'])
    noise = sigma * math.sqrt(1 - alpha * alpha)
    result = np.empty((n, 2), dtype=np.float64)
    state = rng.normal(scale=sigma, size=2)
    for i in range(n):
        result[i] = state
        state = alpha * state + noise * rng.normal(size=2)
    return result


def waveform_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def conditions(s):
    for bound in s['velocity_component_bounds_mps']:
        yield 'velocity_ood', f'square_{bound:g}', {'component_bound_mps': bound, 'duration_s': 0.}
    for dv in s['force_pulse_equivalent_delta_v_mps']:
        for duration in s['force_pulse_durations_s']:
            yield 'force_pulse', f'dv{dv:g}_t{duration:g}', {'equivalent_delta_v_mps': dv, 'duration_s': duration}
    for a in s['constant_accelerations_mps2']:
        yield 'constant_force', f'a{a:g}', {'acceleration_mps2': a, 'duration_s': s['constant_duration_s']}
    for dv in s['repeated_magnitudes_mps']:
        for period in s['repeated_periods_s']:
            yield 'repeated_impulse', f'dv{dv:g}_p{period:g}', {'single_delta_v_mps': dv, 'period_s': period, 'duration_s': s['repeated_duration_s']}
    for rms in s['random_rms_accelerations_mps2']:
        for tau in s['random_correlation_times_s']:
            yield 'random_force', f'rms{rms:g}_tau{tau:g}', {'rms_acceleration_mps2': rms, 'correlation_time_s': tau, 'duration_s': s['random_duration_s']}
    for mode in s['wrench_modes']:
        arm = s['upper_pitch_arm_m'] if mode == 'upper_pitch' else s['lateral_yaw_arm_m'] if mode == 'lateral_yaw' else 0.
        yield 'wrench_pulse', mode, {'wrench_mode': mode, 'moment_arm_m': arm,
            'equivalent_delta_v_mps': s['wrench_equivalent_delta_v_mps'], 'duration_s': s['wrench_duration_s']}


def generate_plan(p):
    validate_protocol(p)
    s = p['robustness']; rows = []

    def add(suite, condition, sample, slope, angle, magnitude, extra):
        seed = s['manifest_seed'] + len(rows)
        direction = [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
        row = dict(trial_id=f'{suite}_{condition}_r{sample:03d}', condition_id=f'{suite}_{condition}',
            experiment='E1', suite=suite, repeat_id=sample, reset_seed=seed,
            evaluation_role=ROLES[0] if suite.startswith('extreme_') else ROLES[1],
            domain='OOD_SLOPE_EXTRAPOLATION' if suite == 'extreme_ood' else 'MAIN',
            slope_deg=slope, command_name='+x', command_vx=.4, command_vy=0., command_yaw=0.,
            direction_deg=angle, push_magnitude=magnitude, push_direction=f'{angle:g}deg',
            push_direction_heading=direction, push_type='robustness_intervention',
            target_phase=s['reference_phases'][(sample // 2) % 2],
            reference_touchdown_foot='left' if sample % 2 == 0 else 'right',
            vector_frame=s['envelope_vector_frame'] if suite.startswith('extreme_') else s['family_vector_frame'],
            **initial_state(seed, p), **extra)
        if suite.startswith('extreme_') and magnitude == 0:
            row.update(sham=True, push_type='sham_marker_no_physical_perturbation')
        rows.append(row)
        return row

    for suite, slopes in [('extreme_main', s['main_slopes_deg']), ('extreme_ood', s['ood_slopes_deg'])]:
        for slope in slopes:
            for dv in s['magnitudes_mps']:
                for angle in s['directions_deg']:
                    for sample in range(s['envelope_trials_per_cell'] // len(s['directions_deg'])):
                        add(suite, f's{slope:+g}_dv{dv:g}_d{angle}', sample, slope, angle, dv,
                            {'duration_s': 0., 'family': 'velocity_jump'})
    for family, condition, values in conditions(s):
        n = s['velocity_ood_trials_per_condition'] if family == 'velocity_ood' else s['family_trials_per_condition']
        for sample in range(n):
            angle = s['directions_deg'][(sample // 16) % 8]
            row = add(family, condition, sample, 0, angle,
                values.get('equivalent_delta_v_mps', values.get('single_delta_v_mps', 0.)), {'family': family, **values})
            rng = np.random.default_rng(row['reset_seed'] ^ 0x47515242)
            if family == 'velocity_ood':
                delta = rng.uniform(-values['component_bound_mps'], values['component_bound_mps'], 2)
                row.update(delta_v_world_xy=delta.tolist(), push_magnitude=float(np.linalg.norm(delta)),
                    direction_deg=float(np.degrees(np.arctan2(delta[1], delta[0])) % 360))
                row['push_direction_heading']=(delta/np.linalg.norm(delta)).tolist()
                row['push_direction']=f"{row['direction_deg']:g}deg_world"
            if family == 'repeated_impulse':
                count = round(values['duration_s'] / values['period_s']) + 1
                angles = rng.uniform(0., 2 * math.pi, count)
                row['impacts'] = [{'offset_s': j * values['period_s'],
                    'delta_v_world_xy': [values['single_delta_v_mps'] * math.cos(a), values['single_delta_v_mps'] * math.sin(a)]}
                    for j, a in enumerate(angles)]
                row.update(direction_deg=None,push_direction='time_varying_world_sequence')
            if family == 'random_force':
                row['waveform_seed'] = int(rng.integers(0, 2**31 - 1))
                wave = waveform(row, s['physics_dt_s'])
                row['waveform_sha256'] = waveform_sha(wave)
                row['planned_actual_rms_acceleration_mps2'] = float(np.sqrt(np.mean(np.sum(wave * wave, axis=1))))
                row.update(direction_deg=None,push_direction='time_varying_world_ou')
            if family == 'wrench_pulse':
                row['torque_sign'] = -1. if (sample // 4) % 2 else 1.
    from collections import Counter
    if dict(Counter(r['suite'] for r in rows)) != COUNTS or len({r['trial_id'] for r in rows}) != len(rows):
        raise ValueError('Preregistered plan counts/IDs differ')
    return rows


def prepare(root, standard_index, config=CONFIG):
    root = Path(root).resolve()
    protected = (LAB / 'experiments/g1_recovery_eval_v2').resolve()
    if root.is_relative_to(protected) or root.is_relative_to((LAB / 'experiments/g1_recovery_eval').resolve()):
        raise ValueError('Robustness must use its independent output root')
    p, contract = protocol(config)
    index = read_json(standard_index)
    if {r['model'] for r in index['models']} != set(MODEL_ORDER) or len(index['models']) != 5:
        raise ValueError('Five-model standard index required')
    rows = generate_plan(p)
    info = {**contract, 'schema_version': 1, 'protocol_hash': digest(p), 'manifest_hash': digest(rows),
        'standard_index_sha256': sha256(standard_index), 'metrics_version': COMMON_VERSION,
        'metrics_config_hash': digest(p['metrics']), 'metrics_reference_sha256': None,
        'physics_profile_hash': digest(p['physics']), 'inference_mode': p['physics']['inference_mode'],
        'counts_per_model': COUNTS, 'trials_per_model': len(rows), 'total_new_robustness_trials': 5 * len(rows)}
    files = {'protocol.yaml': yaml.safe_dump(p, sort_keys=False), 'manifest.jsonl': jsonl(rows),
             'standard_benchmark_five_model_index.json': Path(standard_index).read_bytes(),
             'common_detector_config.yaml': yaml.safe_dump(p['metrics']['detector'], sort_keys=True)}
    for name, data in files.items():
        data = data.encode() if isinstance(data, str) else data
        path = root / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError('Prepared input is immutable: ' + name)
        if not path.exists(): atomic_write(path, data)
    if (root / 'prepared.json').exists() and read_json(root / 'prepared.json') != info:
        raise ValueError('Prepared identity changed')
    write_json(root / 'prepared.json', info)
    return info


def load_prepared(root):
    root = Path(root); p = load_yaml(root / 'protocol.yaml'); contract = validate_protocol(p)
    info = read_json(root / 'prepared.json'); rows = read_jsonl(root / 'manifest.jsonl')
    if digest(p) != info['protocol_hash'] or digest(rows) != info['manifest_hash']:
        raise ValueError('Prepared protocol/plan hash mismatch')
    if any(info[k] != v for k, v in contract.items()):
        raise ValueError('Candidate detector identity mismatch')
    if sha256(root / 'standard_benchmark_five_model_index.json') != info['standard_index_sha256']:
        raise ValueError('Standard index changed')
    return p, rows, info
