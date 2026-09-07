"""End-to-end common candidate tests: synthetic physics, no simulator/policy."""
import copy
import io
import json
import math
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
import pytest

import g1_recovery_protocol as protocol
from g1_common_task_detector import (CommonTaskGate, CommonTaskMeasurements, TimestampWindow, classify_window,
                                     METRICS_VERSION, GATE_FIELDS)
from g1_common_task_trial import CommonTaskTrialMachine, replay_common_trial
from g1_recovery_metrics import make_trial_machine, summarize, paired_comparison
from g1_recovery_report import build_report, common_report_blocks, markdown, word_document, is_full_lite_run

P = protocol.load_yaml(protocol.COMMON_PROTOCOL)
C = P['metrics']['detector']


def physical_frame(t, *, speed=.4, pitch_deg=2., omega=0., height=.8, fallen=False, forces=None):
    # Regular .26s alternating touchdown, independent foot airborne confirmation.
    k = int(round(t/.02))
    phase = k % 26
    force = [20. if 0 <= phase < 5 else 0., 20. if 13 <= phase < 18 else 0.]
    pitch = math.radians(pitch_deg)
    return dict(time=float(t), com_velocity=[speed, 0, 0], root_velocity=[speed, 0, 0],
                root_velocity_world=[speed, 0, 0, 0, 0, omega], root_quaternion_wxyz=[math.cos(pitch/2), 0, math.sin(pitch/2), 0],
                roll_pitch=[0, pitch], yaw=0., command=[.4, 0, 0], angular_velocity_world_z=omega,
                root_position=[0, 0, height], root_clearance_m=height, local_plane_normal=[0, 0, 1],
                local_plane_point=[0, 0, 0], forces=force if forces is None else forces,
                fell=fallen, timeout=False, out_of_test_area=False, data_valid=True, plane=None)


def plan(experiment='E1', standing=False):
    return copy.deepcopy(next(r for r in protocol.generate_manifest(P) if r['experiment'] == experiment and
                             (r['command_name'] == 'standing' if standing else r['command_name'] == '+x')))


def complete_synthetic(*, no_motion=False, relapse=False, fallen=False, experiment='E1'):
    m = make_trial_machine(plan(experiment), P)
    for k in range(1100):
        t = k*.02
        speed = 0 if no_motion else .4
        if relapse and m.push_time is not None and t >= m.push_time+9.:
            speed = 0
        fall = fallen and m.push_time is not None and t >= m.push_time+7.
        due = m.feed(physical_frame(t, speed=speed, fallen=fall))
        if due:
            before = [.4, 0, 0, 0, 0, 0]
            after = [.9, 0, 0, 0, 0, 0]
            m.push_applied(before, after, 0.)
        if m.status:
            return m
    raise AssertionError('Synthetic trial did not terminate')


def test_exact_candidate_table():
    for phase, values in [('readiness', [.16, .50, 10., 20., .40, .50]), ('recovery', [.10, .30, 5., 15., .20, .50])]:
        assert [C[phase][k] for k in GATE_FIELDS] == values
    assert C['parameter_status'] == 'candidate_unvalidated'


@pytest.mark.parametrize('pitch', [-3., -1., 0., 2., 3.])
def test_different_reasonable_pitch_centers_pass_same_common_gate(pitch):
    gate = CommonTaskGate(C, 'recovery', push_end=-.02)
    for k in range(31):
        value = gate.update(CommonTaskMeasurements.from_frame(physical_frame(k*.02, pitch_deg=pitch)))
    assert value['passed']


@pytest.mark.parametrize('mode', ['stationary', 'oscillating', 'rotation', 'low_height', 'tilted'])
def test_task_negative_examples(mode):
    gate = CommonTaskGate(C, 'recovery', push_end=-.02)
    for k in range(31):
        kw = {'speed': 0} if mode == 'stationary' else {'speed': .4+(.3 if k%2 else -.3)} if mode == 'oscillating' else {'omega': .3} if mode == 'rotation' else {'height': .49} if mode == 'low_height' else {'pitch_deg': 10}
        value = gate.update(CommonTaskMeasurements.from_frame(physical_frame(k*.02, **kw)))
    assert not value['passed']


def test_boundary_interpolates_angle_before_squaring_not_squared_signal():
    w = TimestampWindow(.6, .25)
    for t in [0, .2, .4, .6, .7]:
        w.append(t, {'tilt_rad': t, 'yaw_rate_error_radps': -t})
    s = w.statistics()
    # Clipped knots .1,.2,.4,.6,.7. Squared signal interpolated at .1 would
    # wrongly use .02 instead of (.1)^2=.01.
    knots = np.array([.1, .2, .4, .6, .7])
    expected = math.sqrt(np.sum((knots[:-1]**2+knots[1:]**2)*.5*np.diff(knots))/.6)
    assert s['tilt_rad']['rms'] == pytest.approx(expected)
    assert s['yaw_rate_error_radps']['rms'] == pytest.approx(expected)


def test_threshold_equality_has_no_physical_slack():
    b = C['recovery']
    s = {'start': 0, 'end': .6, 'time_intervals': 30,
         'velocity_error_mps': {'mean': .1, 'max': .3},
         'tilt_rad': {'rms': math.radians(5), 'max': math.radians(15)},
         'yaw_rate_error_radps': {'rms': .2}, 'clearance_m': {'min': .5}}
    assert classify_window(s, b, time=.6)['passed']
    for section, key, direction in [('velocity_error_mps', 'mean', 1), ('velocity_error_mps', 'max', 1),
                                     ('tilt_rad', 'rms', 1), ('tilt_rad', 'max', 1),
                                     ('yaw_rate_error_radps', 'rms', 1), ('clearance_m', 'min', -1)]:
        modified = copy.deepcopy(s)
        modified[section][key] = np.nextafter(modified[section][key], math.inf if direction > 0 else -math.inf)
        assert not classify_window(modified, b, time=.6)['passed']


def test_scope_standing_missing_and_window_integrity():
    gate = CommonTaskGate(C, 'recovery', push_end=1.)
    with pytest.raises(ValueError, match='post-push'):
        gate.update(CommonTaskMeasurements.from_frame(physical_frame(1.)))
    gate.update(CommonTaskMeasurements.from_frame(physical_frame(1.02)))
    with pytest.raises(ValueError, match='gap'):
        gate.update(CommonTaskMeasurements.from_frame(physical_frame(1.06)))
    with pytest.raises(ValueError, match='actual push'):
        CommonTaskGate(C, 'recovery')
    f = physical_frame(0); f['command'] = [0, 0, 0]
    with pytest.raises(ValueError, match='0.4'):
        CommonTaskGate(C, 'readiness').update(CommonTaskMeasurements.from_frame(f))


def test_online_and_offline_use_identical_real_trigger_and_results():
    m = complete_synthetic()
    r = m.result()
    assert r['status'] == 'RECOVERED_AND_SURVIVED'
    assert r['first_post_push_sample_time'] == pytest.approx(r['push_end_time']+.02)
    assert r['recovery_time'] == pytest.approx(.62)
    assert r['first_confirmation']-r['first_recovery_entry'] == pytest.approx(.5)
    restored = replay_common_trial(m.plan, P, m.trace(), r)
    assert protocol.canonical(restored.result()) == protocol.canonical(r)
    assert protocol.canonical(restored.all_events()) == protocol.canonical(m.all_events())
    expected = sum(e['event'] == 'physical_touchdown' and m.push_time < e['time'] <= r['sustained_recovery_entry'] for e in m.all_events())
    assert r['recovery_steps'] == expected
    assert not bool(m.trace()['post_push_sample'][int(round(m.push_time/.02))])


def test_no_push_e0_and_failed_readiness_have_no_recovery_conclusion():
    for m in [complete_synthetic(no_motion=True), complete_synthetic(experiment='E0')]:
        r = m.result()
        assert not r['push_applied'] and not r['recovery_applicable']
        assert r['recovery_time'] is None and not r['recovered_once_and_survived']
        s = summarize([r], [m.plan])
        assert s['E1']['pushed_10s_survival_rate'] is None
        assert s['E1']['conditional_recovery_success_rate'] is None
    assert m.result()['e0_outcome'] == 'NORMAL_OBSERVATION_COMPLETED'
    assert m.status == 'E0_COMPLETED'


def test_standing_e0_does_not_need_touchdown():
    m = CommonTaskTrialMachine(plan('E0', standing=True), P)
    for k in range(601):
        f = physical_frame(k*.02, speed=0, forces=[20, 20]); f['command'] = [0, 0, 0]
        m.feed(f)
    assert m.status == 'E0_COMPLETED' and m.result()['touchdown_count'] == 0


def test_later_relapse_and_fall_are_valid_results_and_main_time_null():
    for m in [complete_synthetic(relapse=True), complete_synthetic(fallen=True)]:
        r = m.result()
        assert not r['recovered_sustained_and_survived']
        assert r['recovery_time'] is None and r['recovery_steps'] is None
        assert r['first_confirmation'] is not None
    assert m.status == 'FELL'


def test_prepare_common_never_reads_teacher_and_snapshots_are_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, 'nominal_reference', lambda p: pytest.fail('teacher read'))
    root = tmp_path / 'v2'
    info = protocol.prepare(root, protocol.COMMON_PROTOCOL)
    p, rows, loaded = protocol.load_prepared(root)
    assert len(rows) == 390 and loaded == info
    assert not (root/'metrics_reference.yaml').exists() and not (root/'metrics_nodes.json').exists()
    assert protocol.prepare(root, protocol.COMMON_PROTOCOL) == info
    (root/'common_detector_config.yaml').write_text('damaged: true\n')
    with pytest.raises(ValueError, match='integrity'):
        protocol.load_prepared(root)


def test_common_cannot_overwrite_old_root_or_freeze(tmp_path):
    with pytest.raises(ValueError, match='protected'):
        protocol.prepare(protocol.LAB/'experiments/g1_recovery_eval', protocol.COMMON_PROTOCOL)
    modified = copy.deepcopy(P); modified['protocol_version'] = '2.0'
    config = tmp_path/'bad.yaml'
    import yaml
    config.write_text(yaml.safe_dump(modified))
    with pytest.raises(ValueError, match='2.0-dev'):
        protocol.prepare(tmp_path/'bad', config)


def test_teacher_input_forbidden_and_native_bindings_remain_separate():
    with pytest.raises(ValueError, match='teacher'):
        CommonTaskTrialMachine(plan(), P, {'roll_star': 0})
    modified = copy.deepcopy(C); modified['policy'] = 'ppo'
    with pytest.raises(ValueError, match='only'):
        CommonTaskGate(modified, 'readiness')
    from g1_recovery_eval import bind_native_inputs, assert_native_inputs
    assert callable(bind_native_inputs) and callable(assert_native_inputs)
    assert protocol.RESOURCE_FIELDS['native_nominal'] == ('plane_recovery', 'nominal_parameters_path')


def test_new_report_candidate_na_and_no_cross_version_rank(tmp_path):
    info = protocol.prepare(tmp_path, protocol.COMMON_PROTOCOL)
    m = complete_synthetic(no_motion=True); r = m.result()
    identity = {'metrics_version': METRICS_VERSION, 'model_alias': 'synthetic', 'checkpoint_stage': 'final', 'subset': 'lite_full'}
    run = {'id': 'synthetic_fixture', 'identity': identity, 'records': [r], 'summary': summarize([r], [m.plan]), 'manifest': [m.plan]}
    assert not is_full_lite_run(run, [m.plan])
    blocks = common_report_blocks(tmp_path, P, info, [run], [], plots=False)
    md = markdown(blocks, tmp_path)
    assert 'CANDIDATE_UNVALIDATED' in md and 'N/A' in md and '不直接比较' in md
    with zipfile.ZipFile(io.BytesIO(word_document(blocks))) as archive:
        document = ET.fromstring(archive.read('word/document.xml'))
        assert document.tag == '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}document'
        assert 'CANDIDATE_UNVALIDATED' in ''.join(document.itertext())
    old = {**r, 'metrics_version': 'practical_interval_confirm2_v1'}
    with pytest.raises(ValueError, match='not directly comparable'):
        summarize([r, old], [m.plan, m.plan])
    with pytest.raises(ValueError, match='not directly comparable'):
        paired_comparison([r], [old], 1)
    assert build_report(tmp_path, ('md',))['outputs']


def test_common_storage_revalidates_gate_timing_events_and_snapshot(tmp_path):
    root = tmp_path/'prepared'; info = protocol.prepare(root, protocol.COMMON_PROTOCOL)
    m = complete_synthetic(); record = m.result(); trace = m.trace()
    dest = tmp_path/'synthetic_only'; dest.mkdir()
    identity = {**info, 'synthetic': True, 'manifest_hash': protocol.digest([m.plan])}
    protocol.write_json(dest/'identity.json', identity)
    protocol.atomic_write(dest/'protocol_snapshot.yaml', (root/'protocol.yaml').read_bytes())
    protocol.atomic_write(dest/'common_detector_config.yaml', (root/'common_detector_config.yaml').read_bytes())
    protocol.atomic_write(dest/'manifest_snapshot.jsonl', protocol.jsonl([m.plan]))
    store = protocol.RunStore(dest)
    store.save_trial(record, trace, m.all_events())
    assert store.validate(allow_synthetic=True)[1][0]['recovered_sustained_and_survived']
    with pytest.raises(ValueError, match='Synthetic'):
        store.validate()
    stored = store.records()[0]; stored['recovery_steps'] += 1
    protocol.write_json(dest/'trial_records'/f"{record['trial_id']}.json", stored)
    with pytest.raises(ValueError, match='replay'):
        store.validate(allow_synthetic=True)
    stored['recovery_steps'] -= 1
    protocol.write_json(dest/'trial_records'/f"{record['trial_id']}.json", stored)
    trace['post_push_sample'][:] = False
    stream = io.BytesIO(); np.savez_compressed(stream, **trace)
    path = dest/'traces'/f"{record['trial_id']}.npz"
    path.write_bytes(stream.getvalue()); stored['trace_sha256'] = protocol.sha256(path)
    protocol.write_json(dest/'trial_records'/f"{record['trial_id']}.json", stored)
    with pytest.raises(ValueError, match='sample-role'):
        store.validate(allow_synthetic=True)


def test_old_sealed_run_validates_without_mutation():
    root = protocol.LAB/'experiments/g1_recovery_eval/runs/50471a7a5a61afd8d7f6cf93-attempt-0001'
    if not root.exists():
        pytest.skip('Local sealed development trace not distributed with portable tests')
    before = {str(p): protocol.sha256(p) for p in root.rglob('*') if p.is_file()}
    identity, records = protocol.RunStore(root).validate()
    assert identity['metrics_version'] == 'practical_interval_confirm2_v1' and len(records) == 4
    assert before == {str(p): protocol.sha256(p) for p in root.rglob('*') if p.is_file()}


def test_common_validation_inventory_cannot_use_legacy_acceptance(tmp_path, monkeypatch):
    import g1_recovery_report as report
    protocol.prepare(tmp_path, protocol.COMMON_PROTOCOL)
    monkeypatch.setattr(report, 'detector_protocol_hashes', lambda *a: pytest.fail('legacy gate used'))
    evidence = report.detector_validation_report(tmp_path)
    assert evidence['status'] == 'CANDIDATE_UNVALIDATED'
    assert evidence['full_gate_acceptance'] is None and evidence['pushed_trajectories'] == []
    assert not evidence['automatic_freeze'] and evidence['protocol_version'] == '2.0-dev'


def test_fixed_development_design_counts_pairing_and_holdout_separation():
    design = protocol.read_json(protocol.COMMON_PROTOCOL.parent/'g1_recovery_eval_v2_development_16.json')
    assert design['status'] == 'PLANNED_NOT_EXECUTED'
    assert design['candidate_parameters_sha256'] == protocol.common_contract(P)[1]['candidate_parameters_sha256']
    assignments = design['assignments']
    assert {a['method'] for a in assignments} == {'ppo_plain', 'dwaq'}
    assert assignments[0]['trials'] == assignments[1]['trials']
    trials = [t for a in assignments for t in a['trials']]
    e0 = [t for t in trials if t['experiment'] == 'E0']
    held = [t for t in trials if t['experiment'] == 'E1']
    assert len(trials) == 16 and len(e0) == 4 and len(held) == 12
    assert {t['reset_seed'] for t in e0}.isdisjoint(t['reset_seed'] for t in held)
    assert sum(t['push_magnitude'] > 0 for t in held) == 8
    assert sum(t['push_magnitude'] == 0 for t in held) == 4
    assert {t['push_magnitude'] for t in held} == {0, .5, 1.}


def test_adapter_filters_policy_and_certificate_diagnostics_before_measurements(monkeypatch):
    original = CommonTaskMeasurements.from_frame
    def strict_measurements(frame):
        assert not ({'plane', 'method', 'N', 'margin', 'teacher', 'certificate_valid'} & frame.keys())
        return original(frame)
    monkeypatch.setattr(CommonTaskMeasurements, 'from_frame', strict_measurements)
    m = CommonTaskTrialMachine(plan(), P)
    f = physical_frame(0)
    f.update(plane={'N': 100, 'margin': -100}, method='arbitrary', teacher='arbitrary', certificate_valid=False)
    assert not m.feed(f)
