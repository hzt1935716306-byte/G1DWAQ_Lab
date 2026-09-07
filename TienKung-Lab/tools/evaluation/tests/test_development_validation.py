"""Registered development execution contracts using CPU synthetic traces only."""
import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import yaml

import g1_recovery_protocol as protocol
from g1_development_validation import (DEVELOPMENT_MANIFEST, DEVELOPMENT_MANIFEST_SHA256,
    select_development_assignment, realized_environment_payload, measure_realized_slopes, ACCEPTANCE_FIELDS)
from g1_common_task_trial import CommonTaskTrialMachine, replay_common_trial
from g1_recovery_eval import apply_trial_intervention
from g1_recovery_metrics import additive_velocity, summarize
from g1_recovery_report import (compatible_run, development_report_blocks, is_full_lite_run, markdown, curve_plots)
from test_common_task_v2 import physical_frame, P, plan


def assignment(method='ppo_plain'):
    return select_development_assignment(P, protocol.TASKS[method], method)


def synthetic_trial(row, *, stationary_after_warmup=False, fail_after_marker=False):
    machine = CommonTaskTrialMachine(copy.deepcopy(row), P)
    for k in range(1100):
        t = k*.02
        speed = 0. if stationary_after_warmup and t >= 2 else .4
        fall = fail_after_marker and machine.sham_marker_time is not None and t > machine.sham_marker_time+3
        f = physical_frame(t, speed=speed, fallen=fall)
        if machine.feed(f):
            env = NS(device='cpu', robot=NS(data=NS(root_vel_w=torch.tensor([f['root_velocity_world']], dtype=torch.float64))))
            def write(velocity, env_ids):
                env.robot.data.root_vel_w[env_ids] = velocity.to(torch.float64)
            env.robot.write_root_velocity_to_sim = write
            apply_trial_intervention(env, machine, 0, f, {'method': 'ppo_plain'})
        if machine.status:
            return machine
    raise AssertionError('Synthetic trial did not finish')


@pytest.fixture(scope='module')
def machines():
    return [synthetic_trial(r) for r in assignment()[0]]


@pytest.mark.parametrize('method', ['ppo_plain', 'dwaq'])
def test_pinned_exact_assignments_select_eight_without_resampling(method, monkeypatch):
    monkeypatch.setattr(protocol, 'generate_manifest', lambda *a: pytest.fail('resampled'))
    rows, fields = assignment(method)
    original = next(a for a in protocol.read_json(DEVELOPMENT_MANIFEST)['assignments'] if a['method'] == method)
    assert len(rows) == 8 and rows == original['trials']
    assert fields['development_manifest_sha256'] == protocol.sha256(DEVELOPMENT_MANIFEST)
    assert fields['development_assignment_sha256'] == protocol.digest(original)


@pytest.mark.parametrize('change', ['metadata', 'pose', 'seed', 'command', 'joint', 'whitespace', 'hash'])
def test_any_manifest_byte_or_hash_change_rejected(tmp_path, change):
    raw = DEVELOPMENT_MANIFEST.read_bytes()
    d = json.loads(raw)
    row = d['assignments'][0]['trials'][0]
    if change == 'metadata': d['status'] = 'EXECUTED'
    if change == 'pose': row['initial_pose_parameters']['x'] += .001
    if change == 'seed': row['reset_seed'] += 1
    if change == 'command': row['command_vx'] = .5
    if change == 'joint': row['initial_joint_parameters']['position_scale'][0] += .001
    path = tmp_path/'design.json'
    path.write_bytes(raw if change == 'hash' else raw+b' ' if change == 'whitespace' else json.dumps(d).encode())
    with pytest.raises(ValueError, match='SHA256'):
        select_development_assignment(P, protocol.TASKS['ppo_plain'], 'ppo_plain', path,
                                     '0'*64 if change == 'hash' else DEVELOPMENT_MANIFEST_SHA256)


def test_development_rejects_other_version_or_method():
    bad = copy.deepcopy(P); bad['protocol_version'] = '2.0'
    with pytest.raises(ValueError, match='2.0-dev'): select_development_assignment(bad, protocol.TASKS['ppo_plain'], 'ppo_plain')
    with pytest.raises(ValueError, match='no registered'): select_development_assignment(P, protocol.TASKS['rl_only'], 'rl_only')


def test_cli_has_explicit_development_entry_without_launching(monkeypatch):
    import g1_recovery_eval as evaluator
    captured = []
    monkeypatch.setattr(evaluator, 'execute_run', lambda args: captured.append(args) or {'offline_dispatch_test': True})
    evaluator.main(['run-development', '--task', protocol.TASKS['dwaq'], '--checkpoint', 'unused.pt', '--model_alias', 'dwaq'])
    args = captured[0]
    assert args.development_manifest_sha256 == DEVELOPMENT_MANIFEST_SHA256 and args.num_envs == 1
    assert args.protocol == str(protocol.COMMON_PROTOCOL)


def test_development_identity_fields_change_evaluation_key():
    from test_final_polish import _identity
    i = {**_identity(), **assignment()[1]}
    original = protocol.evaluation_key(i)
    for key in assignment()[1]:
        changed = {**i, key: 'changed'}
        assert protocol.evaluation_key(changed) != original


def test_sham_uses_identical_phase_but_never_writes_velocity_or_creates_recovery(machines):
    sham = next(m for m in machines if m.sham)
    row = copy.deepcopy(sham.plan)
    row.update(sham=False, push_type='instant_velocity_jump', push_magnitude=.5)
    real = synthetic_trial(row)
    assert sham.sham_marker_time == real.push_time
    assert sham.trigger['scheduled_push_time'] == real.trigger['scheduled_push_time']
    r = sham.result()
    assert r['sham_applied'] and r['intervention_marker_applied'] and not r['push_applied']
    assert r['sham_continuity'] and r['sham_detection_latency'] == pytest.approx(1.12)
    assert r['sham_task_entry']-r['sham_marker_time'] == pytest.approx(.62)
    assert r['recovery_time'] is None and r['recovery_steps'] is None and not r['recovery_applicable']
    assert sham.recovery is None and not any(e['event'] == 'velocity_jump' for e in sham.all_events())
    m = CommonTaskTrialMachine(sham.plan, P)
    for k in range(401):
        f = physical_frame(k*.02)
        if m.feed(f): break
    robot = NS(data=NS(root_vel_w=torch.tensor([f['root_velocity_world']], dtype=torch.float64)),
               write_root_velocity_to_sim=lambda *a, **kw: pytest.fail('sham physical velocity write'))
    env = NS(robot=robot, _on_curriculum_push=lambda *a: pytest.fail('sham native hook'))
    apply_trial_intervention(env, m, 0, f, {'method': 'context_only'})
    assert m.sham_marker_time == sham.sham_marker_time
    with pytest.raises(ValueError, match='Sham'): m.push_applied([0]*6, [0]*6, 0)


def test_sham_replay_and_denominators_are_separate(machines):
    records = [m.result() for m in machines]
    s = summarize(records, assignment()[0])
    assert s['designated'] == s['executed'] == 8
    assert s['E1']['designated'] == s['E1']['pushed'] == 4
    for key in ('recovery_within_3_touchdowns', 'recovery_within_5_touchdowns', 'sustained_designated'):
        assert s['E1']['rates'][key]['denominator'] == 4
    assert s['sham']['designated'] == s['sham']['marker_reached'] == 2
    for m in machines:
        if not m.sham: continue
        restored = replay_common_trial(m.plan, P, m.trace(), m.result())
        assert protocol.canonical(restored.result()) == protocol.canonical(m.result())
        assert protocol.canonical(restored.all_events()) == protocol.canonical(m.all_events())
    shams = [m for m in machines if m.sham]
    only = summarize([m.result() for m in shams], [m.plan for m in shams])
    assert only['E1']['designated'] == only['E1']['pushed'] == 0
    assert only['E1']['pushed_10s_survival_rate'] is None
    assert only['E1']['task_recovery_success_rate'] is None
    assert only['E1']['recovery_within_3_touchdowns'] is None
    pending = summarize([], assignment()[0])
    assert pending['E1']['designated'] == 4 and pending['sham']['designated'] == 2
    assert pending['E1']['pending'] == 4 and pending['pending'] == 8
    from g1_recovery_metrics import paired_comparison
    with pytest.raises(ValueError, match='Sham'): paired_comparison(records, records, 6)


def test_sham_unreached_or_later_fall_is_not_real_recovery(machines):
    row = next(m.plan for m in machines if m.sham)
    unready = synthetic_trial(row, stationary_after_warmup=True).result()
    assert not unready['sham_applied'] and unready['sham_continuity'] is None
    fallen = synthetic_trial(row, fail_after_marker=True).result()
    assert fallen['status'] == 'FELL' and fallen['sham_continuity'] is False
    assert not fallen['push_applied'] and fallen['recovery_time'] is None


def test_warmup_excluded_earliest_full_window_and_hold(machines):
    m = machines[0]
    windows = [e for e in m.all_events() if e['event'] == 'common_readiness_window']
    assert min(e['time'] for e in windows) == 2.
    complete = [e for e in windows if e['window_complete']]
    assert complete[0]['time'] == pytest.approx(3.2)
    assert complete[0]['window_start'] == pytest.approx(2.)
    assert m.result()['first_readiness_confirmation_time'] == pytest.approx(3.4)
    assert any(e['event'] == 'readiness_window_started' and e['warmup_samples_excluded'] for e in m.all_events())
    short = CommonTaskTrialMachine(plan(), P)
    for k in range(160):
        assert not short.feed(physical_frame(k*.02))
        assert not short.precondition_passed
    assert len(short.readiness_gate.window.samples) == 60


def test_moving_e0_statistics_and_standing_na(machines):
    e0 = machines[0].result()
    assert e0['common_complete_windows'] == e0['common_task_gate_pass_windows'] == 441
    assert e0['common_task_plus_recent_touchdown_rate'] == 1.
    assert e0['readiness_hold_confirmed']
    assert not any(e0['common_gate_failure_counts'].values())
    bad = synthetic_trial(machines[0].plan, stationary_after_warmup=True).result()
    assert bad['common_task_gate_pass_rate'] == 0 and bad['common_gate_failure_counts']['mean_velocity'] == 441
    standing = CommonTaskTrialMachine(plan('E0', standing=True), P)
    for k in range(601):
        f = physical_frame(k*.02, speed=0, forces=[20,20]); f['command'] = [0,0,0]
        standing.feed(f)
    assert all(standing.result()[k] is None for k in ACCEPTANCE_FIELDS)
    summary = summarize([standing.result()], [standing.plan])
    assert all(summary['E0']['standing'][k] is None for k in ACCEPTANCE_FIELDS)
    assert e0['torso_orientation_status'] == 'NOT_IMPLEMENTED' and e0['torso_gravity_tilt_rad'] is None


def effective_fixture():
    physics = {'fixture': 'synthetic CPU physics'}
    sources = protocol.code_identity()['evaluation_runtime_sources']
    return dict(actual_physics=physics, actual_physics_hash=protocol.digest(physics), terrain_mesh_sha256=protocol.digest([[0,0,0]]),
                terrain_origins=[[[0,0,0]]], realized_slopes_deg=[0],
                environment_versions={k: 'synthetic-1' for k in ('IsaacLab','IsaacSim','USD','PhysX','Kit','trimesh','numpy')},
                terrain_source_sha256={k: sources[k] for k in ('legged_lab/terrains/__init__.py',
                    'legged_lab/terrains/plane_terrain_cfg.py', 'legged_lab/recovery/plane_terrain_math.py')})


@pytest.mark.parametrize('field', ['terrain_mesh_sha256','terrain_origins','realized_slopes_deg','environment_versions'])
def test_realized_hash_binds_mesh_origins_slopes_versions(field):
    e = effective_fixture(); original = protocol.digest(realized_environment_payload(e))
    changed = copy.deepcopy(e)
    if field == 'environment_versions': changed[field]['IsaacSim'] = 'synthetic-2'
    elif field == 'terrain_mesh_sha256': changed[field] = protocol.digest([[0,0,1]])
    else: changed[field] = [1]
    assert protocol.digest(realized_environment_payload(changed)) != original


def test_actual_plane_fitting_and_runtime_sources():
    points = [[x,y,np.tan(np.deg2rad(-10))*x] for x,y in [(-32,-32),(-32,32),(32,-32),(32,32)]]
    assert measure_realized_slopes(points, [[[0,0,0]]], [-10]) == pytest.approx([-10])
    points[0][2] += 1
    with pytest.raises(ValueError, match='corners'): measure_realized_slopes(points, [[[0,0,0]]], [-10])
    assert 'legged_lab/terrains/plane_terrain_cfg.py' in protocol.EVALUATION_RUNTIME_SOURCES


def test_strict_development_pairing_and_formal_exclusion(tmp_path, machines):
    from test_final_polish import _identity
    records = [m.result() for m in machines]
    i = {**_identity(), **assignment()[1], 'method':'ppo_plain', 'model_alias':'PPO',
         'metrics_version':protocol.COMMON_VERSION, 'realized_environment_hash':'environment-A', 'subset':'lite_full'}
    a = dict(identity=i, records=records, manifest=assignment()[0], summary=summarize(records, assignment()[0]), id='ppo-fixture')
    b = copy.deepcopy(a); b['id'] = 'dwaq-fixture'; b['identity'].update(method='dwaq', **assignment('dwaq')[1])
    assert compatible_run(a,b)
    assert not is_full_lite_run(a, a['manifest'])
    with pytest.raises(ValueError, match='development'): curve_plots(tmp_path, [a], 'fixture')
    md = markdown(development_report_blocks([a,b]), tmp_path)
    assert 'Common v2 Development Validation' in md and 'CANDIDATE_UNVALIDATED' in md and 'Strict matched environment' in md
    b['identity']['realized_environment_hash'] = 'environment-B'
    assert not compatible_run(a,b)
    assert '不能严格配对' in markdown(development_report_blocks([a,b]), tmp_path)
    del b['identity']['realized_environment_hash']
    assert not compatible_run(a,b)


def store_fixture(tmp_path, machines):
    root = tmp_path/'prepared'; info = protocol.prepare(root, protocol.COMMON_PROTOCOL)
    rows, fields = assignment()
    dest = tmp_path/'synthetic_only'; dest.mkdir()
    e = effective_fixture(); e['realized_environment_hash'] = protocol.digest(realized_environment_payload(e))
    i = {**info, **fields, **protocol.code_identity(), 'synthetic':True,
         'task_name':protocol.TASKS['ppo_plain'], 'method':'ppo_plain', 'manifest_hash':protocol.digest(rows),
         'actual_physics_hash':e['actual_physics_hash'], 'realized_environment_hash':e['realized_environment_hash'],
         'software':e['environment_versions']}
    protocol.write_json(dest/'identity.json', i)
    for src, dst in [('protocol.yaml','protocol_snapshot.yaml'), ('common_detector_config.yaml','common_detector_config.yaml')]:
        protocol.atomic_write(dest/dst, (root/src).read_bytes())
    protocol.atomic_write(dest/'development_manifest_snapshot.json', DEVELOPMENT_MANIFEST.read_bytes())
    protocol.atomic_write(dest/'manifest_snapshot.jsonl', protocol.jsonl(rows))
    protocol.atomic_write(dest/'effective_env_config.yaml', yaml.safe_dump(e))
    store = protocol.RunStore(dest)
    for m in machines: store.save_trial(m.result(), m.trace(), m.all_events())
    return store


def test_store_recomputes_sham_e0_and_realized_environment(tmp_path, machines):
    store = store_fixture(tmp_path, machines)
    assert len(store.validate(allow_synthetic=True)[1]) == 8
    sham = next(m for m in machines if m.sham)
    path = store.path/'trial_records'/f"{sham.plan['trial_id']}.json"
    original = protocol.read_json(path)
    for field, value in [('sham_continuity',False), ('sham_detection_latency',99), ('sham_marker_time',99), ('push_applied',True)]:
        protocol.write_json(path, {**original, field:value})
        with pytest.raises((ValueError, KeyError)): store.validate(allow_synthetic=True)
    protocol.write_json(path, original)
    events_path = store.path/'trial_events'/f"{sham.plan['trial_id']}.jsonl"
    events = protocol.read_jsonl(events_path)
    protocol.atomic_write(events_path, protocol.jsonl(events+[dict(event='velocity_jump', time=sham.sham_marker_time)]))
    with pytest.raises(ValueError, match='Sham event replay'): store.validate(allow_synthetic=True)
    protocol.atomic_write(events_path, protocol.jsonl(events))
    e0_path = store.path/'trial_records'/f"{machines[0].plan['trial_id']}.json"
    e0 = protocol.read_json(e0_path)
    protocol.write_json(e0_path, {**e0, 'common_task_gate_pass_windows': e0['common_task_gate_pass_windows']+1})
    with pytest.raises(ValueError, match='physical replay'): store.validate(allow_synthetic=True)
    protocol.write_json(e0_path, e0)
    effective = protocol.load_yaml(store.path/'effective_env_config.yaml'); effective['terrain_mesh_sha256'] = 'changed'
    protocol.atomic_write(store.path/'effective_env_config.yaml', yaml.safe_dump(effective))
    with pytest.raises(ValueError, match='Realized environment hash'): store.validate(allow_synthetic=True)


def test_runner_rejects_noncommon_before_checkpoint_or_simulator(tmp_path, monkeypatch):
    import g1_recovery_eval as evaluator
    monkeypatch.setattr(evaluator, 'inspect_checkpoint', lambda *a: pytest.fail('checkpoint accessed before protocol rejection'))
    with pytest.raises(ValueError, match='2.0-dev'):
        evaluator.execute_run(NS(command='run-development', output_root=str(tmp_path), protocol=protocol.DEFAULT_PROTOCOL))
    assert list(tmp_path.iterdir()) == []
