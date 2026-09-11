"""Offline protocol, identity and transactional result storage. No simulator imports."""
from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tempfile
import time

import yaml

LAB = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = LAB / 'tools/evaluation/configs/g1_recovery_eval_lite_v1.yaml'
COMMON_PROTOCOL = LAB / 'tools/evaluation/configs/g1_recovery_eval_lite_v2.yaml'
COMMON_VERSION = 'common_task_window_v1'
TASKS = {
    'ppo_plain': 'g1_slope_nosys_d_matched',
    'ppo_symmetric': 'g1_slope_sys_d_matched',
    'dwaq': 'g1_dwaq_slope_nosys_d_matched',
    'dwaq_no_idle': 'g1_dwaq_slope_nosys_d_matched_v2',
    'dwaq_no_swing': 'g1_dwaq_slope_nosys_d_matched_v3',
    'dwaq_no_idle_no_swing': 'g1_dwaq_slope_nosys_d_matched_v4',
    'rl_only': 'g1_plane_v1_rl_only_matched',
    'context_only': 'g1_plane_v1_estimator_context_no_reward_matched',
    'context_only_v2': 'g1_plane_v1_estimator_context_no_reward_matched_v2',
    'context_only_v3': 'g1_plane_v1_estimator_context_no_reward_matched_v3',
    'context_reward': 'g1_plane_v1_estimator_context_reward_matched',
    'context_reward_v2': 'g1_plane_v1_estimator_context_reward_matched_v2',
}
COMPATIBILITY = ('protocol_hash', 'manifest_hash', 'metrics_version', 'metrics_config_hash',
                 'metrics_reference_sha256', 'physics_profile_hash', 'inference_mode',
                 'evaluation_runtime_sha256')
STATUSES = {'PRECONDITION_FAILED', 'RECOVERED_AND_SURVIVED', 'ALIVE_NOT_RECOVERED',
            'FELL', 'OUT_OF_TEST_AREA', 'EVALUATION_ERROR'}

# This allowlist is intentionally explicit. Only sources that can alter native
# inference, trial execution, physical judging, or recovery metrics belong here.
EVALUATION_RUNTIME_SOURCES = (
    'legged_lab/envs/g1/g1_reward_shaping_ablation_config.py',
    'legged_lab/envs/g1/g1_plane_reward_matched_v2_config.py',
    'legged_lab/envs/__init__.py',
    'tools/evaluation/g1_reduced_budget.py',
    'tools/evaluation/g1_world_wrench.py',
    'tools/evaluation/g1_wrench_validation.py',
    'tools/evaluation/g1_complete_robustness.py',
    'tools/evaluation/g1_robustness_protocol.py',
    'tools/evaluation/g1_robustness_physics.py',
    'tools/evaluation/g1_robustness_trial.py',
    'tools/evaluation/g1_robustness_store.py',
    'tools/evaluation/configs/g1_complete_robustness_v2.yaml',
    'tools/evaluation/g1_recovery_eval.py',
    'tools/evaluation/g1_recovery_protocol.py',
    'tools/evaluation/g1_recovery_metrics.py',
    'tools/evaluation/g1_common_task_detector.py',
    'tools/evaluation/g1_common_task_trial.py',
    'tools/evaluation/g1_development_validation.py',
    'legged_lab/terrains/__init__.py',
    'legged_lab/terrains/plane_terrain_cfg.py',
    'legged_lab/recovery/plane_terrain_math.py',
    'legged_lab/envs/base/base_env.py',
    'legged_lab/envs/g1/g1_dwaq_env.py',
    'legged_lab/envs/g1/g1_plane_v1_env.py',
    'legged_lab/envs/g1/g1_slope_matched_config.py',
    'legged_lab/envs/g1/g1_plane_v1_matched_config.py',
    'legged_lab/envs/g1/g1_plane_v1_rl_only_config.py',
    'legged_lab/recovery/state_extractor.py',
    'legged_lab/recovery/plane_certificate_runtime.py',
    'legged_lab/recovery/g1_certificate_runtime.py',
    'legged_lab/recovery/certificate.py',
    'legged_lab/recovery/certificate_process_pool.py',
    'legged_lab/recovery/certificate_ipc.py',
    'legged_lab/recovery/plane_adapter.py',
    'legged_lab/recovery/plane_nominal_params.py',
    'legged_lab/recovery/recovery_context.py',
    'legged_lab/recovery/plane_v1.py',
    'legged_lab/recovery/stage2_reward.py',
    'rsl_rl/rsl_rl/modules/actor_critic.py',
    'rsl_rl/rsl_rl/modules/actor_critic_DWAQ.py',
    'rsl_rl/rsl_rl/modules/normalizer.py',
    'rsl_rl/rsl_rl/runners/on_policy_runner.py',
    'rsl_rl/rsl_rl/runners/dwaq_on_policy_runner.py',
)
REPORT_SOURCE_PATHS = ('tools/evaluation/g1_robustness_analysis.py', 'tools/evaluation/g1_robustness_report.py', 'tools/evaluation/g1_recovery_report.py', 'tools/evaluation/README.md')
REPORT_SOURCE_DIRS = ('tools/evaluation/tests', 'tools/evaluation/docs')


def resolve_input_path(path):
    """Resolve a CLI input relative to the caller's current directory."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve(strict=True)


def project_path(path):
    """Resolve a user path and require it to stay inside the project checkout."""
    resolved = resolve_input_path(path)
    try:
        resolved.relative_to(LAB)
    except ValueError as exc:
        raise ValueError(f'Path must be inside the project directory: {path}') from exc
    return resolved


def portable_path(path):
    """Return a repository-relative path suitable for persisted metadata."""
    return project_path(path).relative_to(LAB).as_posix()


class CheckpointIdentity(dict):
    """Serializable content identity plus transient, explicitly resolved input handles."""
    def __init__(self, *args, inputs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = inputs or {}


RESOURCE_FIELDS = {'native_nominal': ('plane_recovery', 'nominal_parameters_path'),
                   'native_capability': ('stage2_reward', 'certificate_parameters_path')}
NATIVE_CONFIG_FIELDS = ('plane_recovery', 'stage2_reward', 'recovery_context', 'plane_v1_reward',
                        'com_velocity_source', 'estimator_imu_acceleration_scale')


def native_contract(env):
    """Path-free solver/context settings; file contents have separate identities."""
    import copy
    result = {k: copy.deepcopy(env[k]) for k in NATIVE_CONFIG_FIELDS if k in env}
    for section, field in RESOURCE_FIELDS.values():
        if section in result:
            result[section].pop(field, None)
    # A diagnostic output path is not an input to the certificate.
    if 'stage2_reward' in result:
        result['stage2_reward'].pop('certificate_query_record_path', None)
    return result


def resolve_resource(role, saved_path, declaration, manifest_dir, expected_sha=None):
    """No suffix/basename rebasing: relocation requires an explicit SHA-bound map."""
    if declaration:
        if not declaration.get('path') or not declaration.get('sha256'):
            raise ValueError(f'{role}: resource mapping requires path and sha256')
        candidate = Path(declaration['path'])
        source = resolve_input_path(candidate if candidate.is_absolute() else manifest_dir / candidate)
        if expected_sha and declaration['sha256'] != expected_sha:
            raise ValueError(f'{role}: declared SHA conflicts with training provenance')
        expected_sha = declaration['sha256']
    else:
        candidate = Path(saved_path)
        try:
            source = project_path(candidate if candidate.is_absolute() else LAB / candidate)
        except (ValueError, OSError) as exc:
            raise ValueError(f'{role}: saved resource unavailable in this checkout; provide '
                             '--identity_manifest with explicit path and sha256') from exc
    actual = sha256(source)
    if expected_sha and actual != expected_sha:
        raise ValueError(f'{role}: resource SHA mismatch')
    suffix = '.yaml' if role in RESOURCE_FIELDS else '.pt'
    return source, {'sha256': actual, 'snapshot': f'resources/{actual}/{role}{suffix}',
                    'evidence': 'declared_sha256' if expected_sha else 'observed_at_registration'}


def validate_resource_identity(identity):
    for role in (*RESOURCE_FIELDS, 'estimator'):
        if identity.get(role + '_sha256') != identity.get('resources', {}).get(role, {}).get('sha256'):
            raise ValueError(f'{role}: identity SHA conflicts with resource binding')
    if digest(identity['native_configuration']) != identity['native_configuration_sha256']:
        raise ValueError('Native configuration identity mismatch')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data.encode('utf-8') if isinstance(data, str) else data
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        d = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, data):
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def jsonl(rows):
    return ''.join(canonical(row) + '\n' for row in rows)


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class ConfigLoader(yaml.SafeLoader):
    """Read Isaac config snapshots as data; never instantiate Python objects."""


def _tagged(loader, tag, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


ConfigLoader.add_multi_constructor('tag:yaml.org,2002:python/', _tagged)


def load_yaml(path):
    return yaml.load(Path(path).read_text(), Loader=ConfigLoader)


def method_for(task):
    for method, name in TASKS.items():
        if task == name:
            if method in ('dwaq_no_idle', 'dwaq_no_swing', 'dwaq_no_idle_no_swing'):
                return 'dwaq'
            if method in ('context_only_v2', 'context_only_v3'):
                return 'context_only'
            if method == 'context_reward_v2':
                return 'context_reward'
            return method
    raise ValueError(f'Unsupported task: {task}; privileged tasks are excluded')


def nominal_reference(protocol):
    path = Path(protocol['metrics']['reference'])
    if not path.is_absolute():
        path = LAB / path
    document = load_yaml(path)
    nodes = document['nominal_plane_gait']['nodes']
    fields = ('roll_star', 'pitch_star', 'mean_velocity_error_threshold',
              'mean_abs_roll_error_threshold', 'mean_abs_pitch_error_threshold')
    table = {}
    for slope in protocol['slopes_deg']:
        for direction, command in protocol['commands'].items():
            if direction == 'standing':
                continue  # E0 standing uses residuals, never a gait recovery detector.
            speed = math.hypot(*command[:2])
            found = [n for n in nodes if n['slope_degrees'] == slope
                     and n['direction'] == direction and abs(n['speed'] - speed) < 1e-9]
            if len(found) != 1:
                raise ValueError(f'Common nominal reference requires one exact node: {slope}/{direction}/{speed}')
            n = found[0]
            if n.get('practical_metric_version') != 'interval_mean_v1':
                raise ValueError('Reference must use interval_mean_v1 frame-error semantics')
            values = {k: float(n[k]) for k in fields}
            if not all(math.isfinite(v) for v in values.values()) or any(values[k] <= 0 for k in fields[2:]):
                raise ValueError('Invalid nominal values/thresholds')
            table[f'{slope}:{direction}'] = values
    return path, table


def generate_manifest(p):
    rows = []
    push_index = 0

    def add(experiment, slope, direction, repeat, push=None):
        nonlocal push_index
        idx = len(rows)
        seed = p['manifest_seed'] + idx
        rng = random.Random(seed)
        init = p['initial_state']
        sample = lambda ranges: {k: rng.uniform(*v) for k, v in ranges.items()}
        command = p['commands'][direction]
        condition = f'{experiment}_s{slope:+g}_{direction}'
        if push:
            name, magnitude, phase = push
            condition += f'_{name}_v{magnitude:g}_p{phase:g}'
        row = dict(trial_id=f'{condition}_r{repeat:02d}', experiment=experiment,
                   condition_id=condition, repeat_id=repeat, slope_deg=slope,
                   command_name=direction, command_vx=command[0], command_vy=command[1], command_yaw=command[2],
                   push_type='additive_root_velocity_jump' if push else 'none',
                   push_magnitude=push[1] if push else 0,
                   push_direction=push[0] if push else None,
                   push_direction_heading=p['e1']['directions'][push[0]] if push else [0, 0],
                   target_phase=push[2] if push else None,
                   reference_touchdown_foot=('left' if push_index % 2 == 0 else 'right') if push else None,
                   reset_seed=seed, initial_pose_parameters=sample(init['pose']),
                   initial_velocity_parameters=sample(init['velocity']),
                   initial_joint_parameters={
                       'position_scale': [rng.uniform(*init['joint_position_scale']) for _ in range(init['joint_count'])],
                       'velocity': [rng.uniform(*init['joint_velocity']) for _ in range(init['joint_count'])]})
        rows.append(row)
        if push:
            push_index += 1

    for slope in p['slopes_deg']:
        for direction in p['commands']:
            for r in range(p['e0']['repeats']):
                add('E0', slope, direction, r)
    for slope in p['slopes_deg']:
        for direction in p['e1']['directions']:
            for magnitude in p['e1']['magnitudes']:
                for phase in p['e1']['phases']:
                    for r in range(p['e1']['repeats']):
                        add('E1', slope, p['e1']['command'], r, (direction, magnitude, phase))
    return rows


def is_common(p):
    return p['metrics']['version'] == COMMON_VERSION


def manifest_names(p):
    version = 'v2' if is_common(p) else 'v1'
    return [f'manifests/nominal_lite_{version}.jsonl', f'manifests/push_lite_{version}.jsonl']


def common_contract(p):
    from g1_common_task_detector import validate_detector_config, DETECTOR_CODE_VERSION
    if p['protocol_version'] != '2.0-dev' or 'reference' in p['metrics']:
        raise ValueError('Common detector requires independent 2.0-dev configuration, no teacher reference')
    config = validate_detector_config(p['metrics']['detector'])
    if p['metrics'].get('validation') != {'status': 'unvalidated', 'evidence': []}:
        raise ValueError('This candidate version cannot claim completed validation/freeze')
    if p['slopes_deg'] != [-10, 0, 10]:
        raise ValueError('Common v1 scope is Lite slopes -10/0/+10 only')
    expected = {'+x': [.4, 0, 0], '-x': [-.4, 0, 0], '+y': [0, .4, 0], '-y': [0, -.4, 0], 'standing': [0, 0, 0]}
    if p['commands'] != expected or p['physics']['policy_hz'] != 50:
        raise ValueError('Common v1 requires cardinal 0.4 m/s, standing E0 and 50 Hz')
    for k, value in {'warmup_s': 2., 'readiness_deadline_s': 6., 'reference_wait_s': 3.,
                     'observation_s': 10., 'recovery_deadline_s': 8., 'required_alternating_touchdowns': 4}.items():
        if p['e1'][k] != value:
            raise ValueError('Unsupported common temporal contract: ' + k)
    if p['e0']['warmup_s'] != 2 or p['e0']['observation_s'] != 10 or p['e1']['command'] == 'standing':
        raise ValueError('Unsupported common E0/E1 task timing')
    raw = yaml.safe_dump(config, sort_keys=True).encode()
    return raw, {'protocol_version': p['protocol_version'], 'detector_code_version': DETECTOR_CODE_VERSION,
                 'candidate_parameters_sha256': digest(config), 'common_detector_config_sha256': hashlib.sha256(raw).hexdigest(),
                 'validation_evidence_sha256': digest(p['metrics']['validation']), 'parameter_status': config['parameter_status']}


def validate_common_snapshot(path, p, info):
    raw, fields = common_contract(p)
    if (info.get('metrics_reference_sha256') is not None or any(info.get(k) != v for k, v in fields.items())
            or info.get('metrics_version') != COMMON_VERSION or info.get('metrics_config_hash') != digest(p['metrics'])):
        raise ValueError('Common detector identity/configuration mismatch')
    if (Path(path) / 'common_detector_config.yaml').read_bytes() != raw:
        raise ValueError('Common detector snapshot integrity mismatch')
    if (Path(path) / 'metrics_reference.yaml').exists():
        raise ValueError('Common judge cannot contain a teacher metrics reference snapshot')


def prepare(root, protocol_path=DEFAULT_PROTOCOL):
    root = Path(root)
    p = load_yaml(protocol_path)
    from g1_recovery_metrics import METRICS_VERSION
    if p['metrics']['version'] not in (METRICS_VERSION, COMMON_VERSION):
        raise ValueError('Unsupported metrics definition; implement and test the requested detector first')
    if p['physics']['observation_noise'] or p['physics']['randomization'] or p['physics']['action_delay_steps'] != 0:
        raise ValueError('This controlled suite requires noise/randomization/delay disabled')
    if not is_common(p) and p['protocol_version'] not in ('1.0-dev', '1.0'):
        raise ValueError('Unsupported protocol version')
    extra = {}
    if is_common(p):
        if root.resolve().is_relative_to((LAB / 'experiments/g1_recovery_eval').resolve()):
            raise ValueError('v2 requires a new result directory; historical root is protected')
        common_raw, extra = common_contract(p)
        reference, table = None, None
    else:
        reference, table = nominal_reference(p)
    rows = generate_manifest(p)
    info = dict(protocol_hash=digest(p), manifest_hash=digest(rows),
                metrics_version=p['metrics']['version'], metrics_config_hash=digest(p['metrics']),
                metrics_reference_sha256=sha256(reference) if reference else None, physics_profile_hash=digest(p['physics']),
                inference_mode=p['physics']['inference_mode'], trials=len(rows), **extra)
    with lock(root / '.prepare.lock'):
        if (root / 'prepared.json').exists():
            if read_json(root / 'prepared.json') != info:
                raise ValueError('Prepared protocol changed; use a new output root/version, never overwrite manifests')
            load_prepared(root)
            return info
        root.mkdir(parents=True, exist_ok=True)
        atomic_write(root / 'protocol.yaml', yaml.safe_dump(p, allow_unicode=True, sort_keys=False))
        if is_common(p):
            atomic_write(root / 'common_detector_config.yaml', common_raw)
        else:
            atomic_write(root / 'metrics_reference.yaml', reference.read_bytes())
            write_json(root / 'metrics_nodes.json', table)
        for e, filename in zip(('E0', 'E1'), manifest_names(p)):
            atomic_write(root / filename, jsonl([r for r in rows if r['experiment'] == e]))
        balances = {}
        for r in rows:
            if r['experiment'] == 'E1':
                cell = balances.setdefault(r['condition_id'], {'left': 0, 'right': 0})
                cell[r['reference_touchdown_foot']] += 1
        write_json(root / 'manifests/reference_foot_counts.json', balances)
        if not (root / 'models.yaml').exists():
            atomic_write(root / 'models.yaml', yaml.safe_dump({'models': [dict(method=m, task_name=t,
                         model_alias=m, checkpoint=None, status='PENDING_CHECKPOINT') for m, t in TASKS.items()]}, sort_keys=False))
        if not (root / 'registry.json').exists():
            write_json(root / 'registry.json', {'evaluations': {}})
        if not (root / 'report/notes.yaml').exists():
            atomic_write(root / 'report/notes.yaml', 'notes: {}\n')
        write_json(root / 'prepared.json', info)
    return info


def load_prepared(root):
    root = Path(root)
    p = load_yaml(root / 'protocol.yaml')
    info = read_json(root / 'prepared.json')
    rows = sum((read_jsonl(root / name) for name in manifest_names(p)), [])
    if digest(p) != info['protocol_hash'] or digest(rows) != info['manifest_hash']:
        raise ValueError('Protocol/manifest integrity mismatch')
    if is_common(p):
        validate_common_snapshot(root, p, info)
    elif sha256(root / 'metrics_reference.yaml') != info['metrics_reference_sha256']:
        raise ValueError('Metrics reference integrity mismatch')
    if rows != generate_manifest(p):
        raise ValueError('Manifest does not match the fixed protocol')
    # Re-derive nodes from the frozen snapshot, never trust an edited cache.
    if not is_common(p):
        snap_p = json.loads(canonical(p))
        snap_p['metrics']['reference'] = str((root / 'metrics_reference.yaml').resolve())
        _, nodes = nominal_reference(snap_p)
        if read_json(root / 'metrics_nodes.json') != nodes:
            raise ValueError('Metrics nodes integrity mismatch')
    return p, rows, info


def code_identity():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(LAB), *args], text=True).strip()
    runtime = {name: sha256(LAB / name) for name in EVALUATION_RUNTIME_SOURCES}
    report_names = list(REPORT_SOURCE_PATHS)
    for directory in REPORT_SOURCE_DIRS:
        report_names.extend(p.relative_to(LAB).as_posix() for p in sorted((LAB / directory).rglob('*'))
                            if p.is_file() and p.suffix in ('.py', '.md', '.json'))
    report = {name: sha256(LAB / name) for name in sorted(set(report_names))}
    audit_names = ('tools/evaluation/g1_run_audit.py', 'tools/evaluation/g1_audit_compatibility.py',
                   'tools/evaluation/g1_recovery_protocol.py', 'tools/evaluation/g1_robustness_store.py')
    audit = {name: sha256(LAB / name) for name in audit_names}
    return {'audit_code_sha256': digest(audit), 'audit_code_sources': audit,
            'evaluation_code_commit': git('rev-parse', 'HEAD'),
            'evaluation_runtime_sha256': digest(runtime),
            'evaluation_runtime_sources': runtime,
            'evaluation_runtime_dirty': bool(git('status', '--porcelain', '--', *EVALUATION_RUNTIME_SOURCES)),
            'report_code_sha256': digest(report),
            'report_code_sources': report,
            'report_code_dirty': bool(git('status', '--porcelain', '--', *REPORT_SOURCE_PATHS, *REPORT_SOURCE_DIRS))}


def inspect_checkpoint(task, checkpoint, alias, stage='unknown', estimator=None, identity_manifest=None):
    """CPU-only strict shape/config audit; authoritative native strict load runs again on run."""
    import torch
    method = method_for(task)
    checkpoint = resolve_input_path(checkpoint)
    agent_path, env_path = checkpoint.parent / 'params/agent.yaml', checkpoint.parent / 'params/env.yaml'
    a, e = load_yaml(agent_path), load_yaml(env_path)
    before = sha256(checkpoint)
    config_hashes = {'agent_config_sha256': sha256(agent_path), 'env_config_sha256': sha256(env_path)}
    c = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if before != sha256(checkpoint):
        raise ValueError('Checkpoint changed while being inspected')
    infos = c.get('infos') if isinstance(c.get('infos'), dict) else {}
    provenance = c.get('training_provenance') or infos.get('training_provenance') or infos
    declaration, manifest_dir = {}, LAB
    if identity_manifest:
        manifest_file = resolve_input_path(identity_manifest)
        declaration, manifest_dir = load_yaml(manifest_file), manifest_file.parent
        for key, value in {'checkpoint_sha256': before, **config_hashes}.items():
            if declaration.get(key) != value:
                raise ValueError(f'Identity manifest {key} mismatch')
        for key in ('task_name', 'training_run_id', 'training_transitions'):
            if declaration.get(key) is not None and provenance.get(key) is not None and declaration[key] != provenance[key]:
                raise ValueError(f'Identity manifest conflicts with training {key}')
    for key, value in config_hashes.items():
        if key in provenance and provenance[key] != value:
            raise ValueError(f'Training provenance {key} mismatch')
    runner = 'DWAQOnPolicyRunner' if method == 'dwaq' else 'OnPolicyRunner'
    if a['runner_class_name'] != runner:
        raise ValueError('runner type mismatch')
    explicit_tasks = [x for x in (provenance.get('task_name'), c.get('task_name'), infos.get('task_name'),
                                  a.get('task_name'), e.get('task_name'),
                                  declaration.get('task_name')) if x]
    if any(x != task for x in explicit_tasks):
        raise ValueError('Explicit task mismatch')
    experiment = a.get('experiment_name')
    if experiment in TASKS.values() and experiment != task:
        raise ValueError('task mismatch in params/agent.yaml')
    established = bool(provenance.get('task_name') or c.get('task_name') or infos.get('task_name')
                       or a.get('task_name') or e.get('task_name'))
    established |= experiment == (task if method in ('ppo_plain', 'ppo_symmetric', 'dwaq') else 'g1_plane_v1_matched')
    if not established and not (declaration.get('task_name') == task and declaration.get('task_confirmed') is True):
        raise ValueError('Insufficient legacy task evidence: require SHA-bound --identity_manifest '
                         'with task_name and task_confirmed: true; strict weights remain required')
    if method.startswith('context'):
        if (e.get('com_velocity_source') != 'estimator' or
                e.get('plane_v1_reward', {}).get('enabled') is not (method == 'context_reward')):
            raise ValueError('Plane source/reward task mismatch')
        if e.get('recovery_context', {}).get('enabled') is not True or e['recovery_context'].get('mode') != 'certificate':
            raise ValueError('Plane context configuration mismatch')
    elif ('com_velocity_source' in e or e.get('recovery_context', {}).get('enabled') or
          e.get('plane_v1_reward', {}).get('enabled')):
        raise ValueError('Baseline/RL-only must not contain estimator/context/recovery reward configuration')
    ah, ch = int(e['robot']['actor_obs_history_length']), int(e['robot']['critic_obs_history_length'])
    expected = {'ppo_plain': (960, 1010, 10, 10), 'ppo_symmetric': (960, 1010, 10, 10),
                'dwaq': (115, 307, 1, 1), 'rl_only': (480, 1010, 5, 10),
                'context_only': (483, 1010, 5, 10), 'context_reward': (483, 1010, 5, 10)}[method]
    sd = c['model_state_dict']
    actor, critic = int(sd['actor.0.weight'].shape[1]), int(sd['critic.0.weight'].shape[1])
    final = sorted((k for k in sd if k.startswith('actor.') and k.endswith('.weight')), key=lambda k: int(k.split('.')[1]))[-1]
    actions = int(sd[final].shape[0])
    if (actor, critic, ah, ch) != expected or actions != 29:
        raise ValueError(f'Native dimensions/history mismatch: {(actor, critic, ah, ch, actions)}')
    if method == 'dwaq' and (int(e['robot']['dwaq_obs_history_length']) != 5 or tuple(sd['encoder.0.weight'].shape) != (128, 480)):
        raise ValueError('DWAQ encoder/history mismatch')
    if a['empirical_normalization'] and not all(k in c for k in ('obs_norm_state_dict', 'privileged_obs_norm_state_dict')):
        raise ValueError('Missing normalizer state')
    if method.startswith('context') and not estimator:
        raise ValueError('Estimator task requires --estimator_checkpoint')
    if not method.startswith('context') and estimator:
        raise ValueError('Baseline/RL-only must not instantiate an estimator')
    # Strict CPU load of every native tensor catches missing decoder/critic keys
    # before AppLauncher. No algorithm, optimizer, simulator or GT inputs needed.
    import sys
    native_root = str(LAB / 'rsl_rl')
    if native_root not in sys.path:
        sys.path.insert(0, native_root)
    from rsl_rl.modules.actor_critic import ActorCritic
    from rsl_rl.modules.actor_critic_DWAQ import ActorCritic_DWAQ
    if method == 'dwaq':
        model = ActorCritic_DWAQ(actor, critic, actions, 480, 19, 96,
                                activation=a['policy']['activation'], init_noise_std=a['policy']['init_noise_std'])
    else:
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            model = ActorCritic(actor, critic, actions, **{k: v for k, v in a['policy'].items()
                                if k in ('actor_hidden_dims', 'critic_hidden_dims', 'activation', 'init_noise_std', 'noise_std_type')})
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.requires_grad_(False)
    if a['empirical_normalization']:
        from rsl_rl.modules.normalizer import EmpiricalNormalization
        for key, dimension in [('obs_norm_state_dict', 96 if method == 'dwaq' else actor),
                               ('privileged_obs_norm_state_dict', critic)]:
            normalizer = EmpiricalNormalization(shape=[dimension])
            normalizer.load_state_dict(c[key], strict=True)
            normalizer.eval()
    estimator_hash = None
    if estimator:
        estimator = resolve_input_path(estimator)
        estimator_hash = sha256(estimator)
        # Pure native loader; it verifies schema, dimensions, units and strict weights.
        import sys
        if str(LAB) not in sys.path:
            sys.path.insert(0, str(LAB))
        from legged_lab.estimation import load_com_velocity_estimator_for_inference
        load_com_velocity_estimator_for_inference(str(estimator), device='cpu')
    inputs = {'checkpoint': checkpoint, 'agent_config': agent_path, 'env_config': env_path}
    resources = {}
    for role, (section, field) in RESOURCE_FIELDS.items():
        saved_path = e.get(section, {}).get(field)
        if method.startswith('context') and not saved_path:
            raise ValueError(f'Missing native input: {role}')
        if saved_path:
            inputs[role], resources[role] = resolve_resource(role, saved_path,
                declaration.get('resources', {}).get(role), manifest_dir,
                provenance.get('resources', {}).get(role, {}).get('sha256'))
    if estimator:
        inputs['estimator'] = estimator
        expected_estimator = provenance.get('resources', {}).get('estimator', {}).get('sha256')
        if expected_estimator and expected_estimator != estimator_hash:
            raise ValueError('Estimator SHA conflicts with training provenance')
        estimator_declaration = declaration.get('resources', {}).get('estimator')
        if estimator_declaration:
            _, declared_resource = resolve_resource('estimator', estimator, estimator_declaration, manifest_dir, expected_estimator)
            if declared_resource['sha256'] != estimator_hash:
                raise ValueError('Estimator SHA conflicts with identity manifest')
        resources['estimator'] = {'sha256': estimator_hash, 'snapshot': f'resources/{estimator_hash}/estimator.pt'}
    run_id = declaration.get('training_run_id') or provenance.get('training_run_id')
    transitions = declaration.get('training_transitions')
    if transitions is None:
        transitions = provenance.get('training_transitions')
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError('training_run_id must be a nonempty unique training lineage ID')
    if transitions is not None and (type(transitions) is not int or transitions < 0):
        raise ValueError('training_transitions must be a nonnegative integer')
    if any(sha256(inputs[k.removesuffix('_sha256')]) != v for k, v in config_hashes.items()):
        raise ValueError('Training config changed during inspection')
    def input_label(path, fallback):
        return portable_path(path) if path.is_relative_to(LAB) else fallback
    return CheckpointIdentity(task_name=task, method=method, model_alias=alias, identity_schema_version=2,
                inputs=inputs, resources=resources, native_configuration=native_contract(e),
                native_configuration_sha256=digest(native_contract(e)),
                checkpoint_path=input_label(checkpoint, f'checkpoints/{before}/model.pt'),
                checkpoint_sha256=before, training_commit=infos.get('training_commit', 'unknown'),
                training_iteration=c.get('iter', 'unknown'), training_seed=a.get('seed', 'unknown'),
                training_run_id=run_id, training_transitions=transitions,
                training_metadata_source='identity_manifest' if declaration else 'checkpoint' if provenance else 'unknown',
                checkpoint_stage=stage, estimator_sha256=estimator_hash,
                estimator_path=input_label(estimator, resources['estimator']['snapshot']) if estimator else None,
                native_nominal_sha256=resources.get('native_nominal', {}).get('sha256'),
                native_nominal_path=resources.get('native_nominal', {}).get('snapshot'),
                native_capability_sha256=resources.get('native_capability', {}).get('sha256'),
                actor_input_dimension=actor, critic_input_dimension=critic, action_dimension=actions,
                actor_history_length=ah, critic_history_length=ch,
                encoder_history_length=5 if method == 'dwaq' else None,
                runner_type=runner, empirical_normalization=a['empirical_normalization'],
                **config_hashes,
                identity_evidence=['params/agent.yaml', 'params/env.yaml', 'checkpoint tensor shapes and iter'],
                task_confirmation=bool(declaration.get('task_confirmed')), run_name=a.get('run_name'),
                software={'Python': platform.python_version(), 'Torch': str(torch.__version__), 'IsaacLab': 'unknown', 'IsaacSim': 'unknown'},
                hardware={'CPU': platform.processor() or platform.machine(), 'GPU': 'unknown'}, **code_identity())


def register_model(root, identity):
    root = Path(root)
    with lock(root / '.registry.lock'):
        document = load_yaml(root / 'models.yaml')
        models = document['models']
        models[:] = [m for m in models if m.get('model_alias') != identity['model_alias']
                     and not (m['method'] == identity['method'] and m['status'] == 'PENDING_CHECKPOINT')]
        models.append({**identity, 'checkpoint': identity['checkpoint_path'], 'status': 'READY_NOT_EVALUATED'})
        atomic_write(root / 'models.yaml', yaml.safe_dump(document, allow_unicode=True, sort_keys=False))


def evaluation_key(identity):
    keys = COMPATIBILITY + ('task_name', 'checkpoint_sha256', 'estimator_sha256', 'native_nominal_sha256',
                           'agent_config_sha256', 'env_config_sha256')
    fields = {k: identity[k] for k in keys}
    if identity.get('identity_schema_version', 1) >= 2:
        fields.update({k: identity.get(k) for k in ('native_capability_sha256', 'native_configuration_sha256',
                      'training_run_id', 'training_transitions', 'checkpoint_stage')})
    if identity.get('evaluation_role') == 'development_validation':
        from g1_development_validation import DEVELOPMENT_ID_FIELDS
        fields.update({k: identity[k] for k in DEVELOPMENT_ID_FIELDS})
    return digest(fields)[:24]


def compatible(a, b):
    if a.get('evaluation_role') != b.get('evaluation_role'):
        return False
    if a.get('evaluation_role') == 'development_validation':
        if any(not a.get(k) or a.get(k) != b.get(k) for k in
               ('development_manifest_sha256', 'development_manifest_schema_version', 'realized_environment_hash')):
            return False
    return all(k in a and k in b and a[k] == b[k] for k in COMPATIBILITY)


def csv_bytes(rows):
    if not rows:
        return ''
    fields = sorted(set().union(*(r.keys() for r in rows)))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: canonical(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    return stream.getvalue()


class RunStore:
    """Per-trial JSON is the commit record; trace first, then trial, derived CSV last."""
    def __init__(self, path):
        self.path = Path(path)

    def records(self):
        return [read_json(p) for p in sorted((self.path / 'trial_records').glob('*.json'))]

    def save_trial(self, result, trace, events):
        import numpy as np
        tid = result['trial_id']
        if Path(tid).name != tid:
            raise ValueError('Unsafe trial ID')
        dest = self.path / 'trial_records' / (tid + '.json')
        if dest.exists():
            raise ValueError(f'Trial already committed: {tid}; use a new attempt')
        allowed = STATUSES | ({'E0_COMPLETED', 'SHAM_COMPLETED'} if result.get('metrics_version') == COMMON_VERSION else set())
        if result['status'] not in allowed:
            raise ValueError('Unknown terminal status')
        buf = io.BytesIO()
        np.savez_compressed(buf, **trace)
        atomic_write(self.path / 'traces' / (tid + '.npz'), buf.getvalue())
        atomic_write(self.path / 'trial_events' / (tid + '.jsonl'), jsonl(events))
        result = {**result, 'trace_sha256': sha256(self.path / 'traces' / (tid + '.npz'))}
        write_json(dest, result)
        self.rebuild_tables()

    def rebuild_tables(self):
        records = self.records()
        atomic_write(self.path / 'trials.csv', csv_bytes(records))
        events = [e for r in records for e in read_jsonl(self.path / 'trial_events' / (r['trial_id'] + '.jsonl'))]
        atomic_write(self.path / 'events.csv', csv_bytes(events))

    def validate(self, require_complete=True, *, allow_synthetic=False, validation_level='light', workers=1):
        from g1_run_audit import validate_store
        return validate_store(self, require_complete, allow_synthetic=allow_synthetic,
                              validation_level=validation_level, workers=workers)

    def _validate_light(self, require_complete=True, *, allow_synthetic=False):
        identity = read_json(self.path / 'identity.json')
        if identity.get('synthetic') and not allow_synthetic:
            raise ValueError('Synthetic results cannot enter the real registry')
        manifest = read_jsonl(self.path / 'manifest_snapshot.jsonl')
        if digest(manifest) != identity['manifest_hash']:
            raise ValueError('Run manifest mismatch')
        protocol = load_yaml(self.path / 'protocol_snapshot.yaml')
        if digest(protocol) != identity['protocol_hash']:
            raise ValueError('Run protocol mismatch')
        common = is_common(protocol)
        if common:
            validate_common_snapshot(self.path, protocol, identity)
        elif sha256(self.path / 'metrics_reference.yaml') != identity['metrics_reference_sha256']:
            raise ValueError('Run reference mismatch')
        if identity.get('evaluation_role') == 'development_validation':
            from g1_development_validation import validate_development_snapshot
            validate_development_snapshot(self.path, protocol, identity, manifest)
        elif any(r.get('sham') for r in manifest):
            raise ValueError('Sham requires the pinned development manifest and role')
        effective_path = self.path / 'effective_env_config.yaml'
        if effective_path.exists():
            effective = load_yaml(effective_path)
            if digest(effective['actual_physics']) != identity.get('actual_physics_hash'):
                raise ValueError('Effective physics does not match identity')
            if common or identity.get('realized_environment_schema_version'):
                from g1_development_validation import realized_environment_payload
                realized = digest(realized_environment_payload(effective))
                if realized != identity.get('realized_environment_hash') or realized != effective.get('realized_environment_hash'):
                    raise ValueError('Realized environment hash mismatch')
                if any(identity.get('software', {}).get(k) != effective['environment_versions'][k] for k in ('IsaacLab', 'IsaacSim')):
                    raise ValueError('Realized environment software mismatch')
                terrain_sources = effective['terrain_source_sha256']
                if any(identity.get('evaluation_runtime_sources', {}).get(k) != v for k, v in terrain_sources.items()):
                    raise ValueError('Realized terrain source/runtime provenance mismatch')
            if identity.get('identity_schema_version', 1) >= 2:
                expected_inputs = {k: v['sha256'] for k, v in identity.get('resources', {}).items()}
                if effective.get('native_inputs') != expected_inputs or effective.get('native_configuration_sha256') != identity['native_configuration_sha256']:
                    raise ValueError('Effective native inputs do not match identity')
        if identity.get('identity_schema_version', 1) >= 2:
            validate_resource_identity(identity)
            if digest(read_json(self.path / 'native_configuration.json')) != identity['native_configuration_sha256']:
                raise ValueError('Run native configuration mismatch')
            for role, resource in identity.get('resources', {}).items():
                if role == 'estimator':
                    continue
                path = (self.path / resource['snapshot']).resolve(strict=True)
                if not path.is_relative_to(self.path.resolve()) or sha256(path) != resource['sha256']:
                    raise ValueError(f'Run {role} snapshot mismatch')
        expected = {r['trial_id']: r for r in manifest}
        records = self.records()
        if len({r['trial_id'] for r in records}) != len(records):
            raise ValueError('Duplicate trial ID')
        for r in records:
            if r['trial_id'] not in expected or any(r.get(k) != v for k, v in expected[r['trial_id']].items()):
                raise ValueError('Trial is not the assigned manifest trial')
            if r['status'] not in (STATUSES | ({'E0_COMPLETED', 'SHAM_COMPLETED'} if common else set())):
                raise ValueError('Invalid status')
            if common and r.get('metrics_version') != COMMON_VERSION:
                raise ValueError('Common run cannot contain legacy metric records')
            trace = self.path / 'traces' / (r['trial_id'] + '.npz')
            if sha256(trace) != r['trace_sha256']:
                raise ValueError('Corrupt trace')
            read_jsonl(self.path / 'trial_events' / (r['trial_id'] + '.jsonl'))
            from g1_run_audit import light_trace
            times = light_trace(trace, r, common=common)
            if r.get('survived'):
                intervention_time = r.get('sham_marker_time') if r.get('sham_applied') else r.get('actual_push_time')
                deadline = protocol['e0']['warmup_s'] + protocol['e0']['observation_s'] if r['experiment'] == 'E0' else intervention_time + protocol['e1']['observation_s']
                if times[-1] < deadline - 1e-7:
                    raise ValueError('Survival claimed before fixed observation deadline')
            if r['status'] == 'RECOVERED_AND_SURVIVED' and not (r['push_applied'] and r['survived']
                    and r['recovery_time'] is not None and r['recovery_steps'] is not None):
                raise ValueError('Inconsistent recovered-and-survived result')
        if require_complete and (len(records) != len(expected) or any(r['status'] == 'EVALUATION_ERROR' for r in records)):
            raise ValueError('PARTIAL/INVALID: missing trials or execution errors')
        if (self.path / 'completion.json').exists():
            completion = read_json(self.path / 'completion.json')
            required_files = {'identity.json', 'protocol_snapshot.yaml', 'manifest_snapshot.jsonl',
                              'common_detector_config.yaml' if common else 'metrics_reference.yaml', 'effective_env_config.yaml', 'trials.csv', 'events.csv',
                              'summary.json', 'run.log'}
            required_files.update(f'{folder}/{r["trial_id"]}{suffix}' for r in records
                                  for folder, suffix in [('traces', '.npz'), ('trial_records', '.json'), ('trial_events', '.jsonl')])
            if identity.get('evaluation_role') == 'development_validation':
                required_files.add('development_manifest_snapshot.json')
            if completion.get('status') != 'COMPLETE' or not required_files.issubset(completion['files']):
                raise ValueError('Incomplete completion file index')
            for rel, sha in completion['files'].items():
                if Path(rel).is_absolute() or '..' in Path(rel).parts or sha256(self.path / rel) != sha:
                    raise ValueError(f'Completion integrity mismatch: {rel}')
        return identity, records

    def complete(self, summary):
        self.validate(validation_level='sampled')
        if summary.get('status') != 'COMPLETE':
            raise ValueError('Summary is not complete')
        if not (self.path / 'effective_env_config.yaml').exists() or not (self.path / 'run.log').exists():
            raise ValueError('Missing effective environment or run log')
        self.rebuild_tables()
        write_json(self.path / 'summary.json', summary)
        files = {str(p.relative_to(self.path)): sha256(p) for p in self.path.rglob('*')
                 if p.is_file() and p.name not in ('completion.json', '.run.lock')}
        write_json(self.path / 'completion.json', {'status': 'COMPLETE', 'validation_level': 'sampled',
            'sampled_replay_audit_sha256': sha256(self.path / 'sampled_replay_audit.json'), 'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'files': files})


def register_result(root, run):
    identity, _ = RunStore(run).validate()
    if not (Path(run) / 'completion.json').exists():
        raise ValueError('Cannot register incomplete result')
    with lock(Path(root) / '.registry.lock'):
        registry = read_json(Path(root) / 'registry.json')
        registry['evaluations'][Path(run).name] = {'identity': identity, 'completion_sha256': sha256(Path(run) / 'completion.json')}
        write_json(Path(root) / 'registry.json', registry)


def import_results(source, root):
    source, root = Path(source).resolve(), Path(root).resolve()
    candidates = [source] if (source / 'completion.json').exists() else sorted(source.rglob('completion.json'))
    imported = []
    for candidate in candidates:
        src = candidate.parent if candidate.is_file() else candidate
        if any(p.is_symlink() for p in src.rglob('*')):
            raise ValueError('Imported run contains symlinks')
        identity, _ = RunStore(src).validate()
        content = sha256(src / 'completion.json')
        with lock(root / '.import.lock'):
            existing = [p.parent for p in (root / 'runs').glob('*/completion.json') if sha256(p) == content]
            if existing:
                imported.append(existing[0].name)
                continue
            name = evaluation_key(identity) + '-import-' + content[:12]
            dest = root / 'runs' / name
            if dest.exists():
                raise ValueError('Import collision; refusing overwrite')
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix='.import-', dir=dest.parent))
            try:
                shutil.copytree(src, tmp, dirs_exist_ok=True)
                RunStore(tmp).validate()
                os.replace(tmp, dest)
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp)
            register_result(root, dest)
            imported.append(name)
    if not candidates:
        raise ValueError('No complete runs found')
    return imported
