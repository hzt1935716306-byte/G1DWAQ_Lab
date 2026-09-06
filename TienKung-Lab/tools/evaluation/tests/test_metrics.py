"""Synthetic detector tests; no simulator, no formal-result writes."""
import importlib.util
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from g1_recovery_metrics import (RecoveryDetector, TouchdownDetector, TrialMachine, additive_velocity,
                                 frame_errors, summarize, paired_comparison)
from g1_recovery_protocol import DEFAULT_PROTOCOL, LAB, generate_manifest, load_yaml, nominal_reference

P = load_yaml(DEFAULT_PROTOCOL)
PLANS = generate_manifest(P)
NODES = nominal_reference(P)[1]


def frame(t, forces=(0, 0), **kw):
    return dict(time=t, root_velocity=[.4, 0, 0], root_velocity_world=[.4, 0, 0, 0, 0, 0],
                com_velocity=[.4, 0, 0], roll_pitch=[0, 0], yaw=0., forces=list(forces),
                root_position=[0, 0, .8], fell=False, timeout=False, out_of_test_area=False, **kw)


def test_frame_error_matches_existing_practical_module():
    spec = importlib.util.spec_from_file_location('existing_practical', LAB / 'legged_lab/recovery/practical_metrics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v, rp = torch.tensor([[1., 0.], [-1., 0.]]), torch.tensor([[.1, -.2], [-.1, .2]])
    existing = module.practical_frame_errors(v, torch.zeros_like(v), rp, torch.zeros_like(rp))
    ours = [frame_errors(x, [0, 0], y, [0, 0]) for x, y in zip(v.numpy(), rp.numpy())]
    assert np.mean([x[0] for x in ours]) == 1
    assert float(torch.linalg.vector_norm(v.mean(0))) == 0  # cancellation must not imply recovery
    np.testing.assert_allclose([x[0] for x in ours], existing[0].numpy())
    np.testing.assert_allclose([x[1] for x in ours], existing[1].numpy())


@pytest.mark.parametrize('yaw,direction,expected', [(0, [1, 0], [1, 0]), (math.pi / 2, [1, 0], [0, 1]),
                                                    (math.pi, [0, 1], [0, -1]), (0, [-1, 0], [-1, 0])])
def test_additive_push_preserves_other_components(yaw, direction, expected):
    before = np.array([.3, -.2, .8, .4, .5, .6])
    after = additive_velocity(before, yaw, direction, 1)
    np.testing.assert_allclose(after[:2] - before[:2], expected, atol=1e-12)
    np.testing.assert_array_equal(after[2:], before[2:])
    assert before[0] == .3


def test_single_interval_not_confirmed_and_failed_next_restarts():
    d = RecoveryDetector(1)
    d.interval(.9, 1.2, True, 1)
    assert d.candidate is None
    d.interval(1.2, 1.5, True, 2)
    assert d.confirmed is None
    d.interval(1.5, 1.8, False, 3)
    d.interval(1.8, 2.1, True, 4)
    d.interval(2.1, 2.4, True, 5)
    assert d.result(True)['recovery_time'] == pytest.approx(1.1)
    assert d.result(True)['recovery_steps'] == 4
    assert not d.result(True)['recovery_within_3_touchdowns']
    assert d.result(True)['recovery_within_5_touchdowns']
    assert d.confirmed['confirmation_delay'] == pytest.approx(.3)


def test_confirmation_after_eight_allowed_entry_before_eight():
    d = RecoveryDetector(1)
    d.interval(8.5, 9, True, 12)
    d.interval(9, 9.5, True, 13)
    assert d.result(True)['recovery_steps'] == 12
    assert d.result(True)['recovery_time'] == 8
    fallen = d.result(False)
    assert fallen['recovery_steps'] is None and fallen['recovery_time'] is None
    assert fallen['confirmed_recovery_event'] is not None


def test_confirmation_outside_observation_rejected():
    d = RecoveryDetector(0)
    d.interval(7, 8, True, 20)
    d.interval(8, 10.02, True, 21)
    assert d.confirmed is None
    assert d.result(True)['recovery_steps'] is None


def test_nonadjacent_passing_intervals_not_confirmed():
    d = RecoveryDetector(0)
    d.interval(.1, .4, True, 2)
    d.interval(.6, .9, True, 4)
    assert d.confirmed is None


def test_contact_chatter_same_foot_and_alternation():
    d = TouchdownDetector(P['metrics'])
    d.update(0, [0, 0])
    assert d.update(.02, [10, 0]) is None
    assert d.update(.04, [0, 0]) is None
    assert d.chatter == 1
    d.update(.1, [10, 0])
    assert d.update(.12, [10, 0])['foot'] == 'left'
    d.update(.2, [0, 0]); d.update(.22, [0, 0])
    d.update(.3, [10, 0]); assert d.update(.32, [10, 0]) is None
    assert d.repeated == 1 and d.count == 1
    d.update(.4, [0, 10]); td = d.update(.42, [0, 10])
    assert td['foot'] == 'right' and d.count == 2
    assert td['interval_s'] == pytest.approx(.3)


def test_standing_e0_needs_no_touchdowns():
    plan = next(r for r in PLANS if r['command_name'] == 'standing')
    m = TrialMachine(plan, P, None)
    m.feed(frame(0)); m.feed(frame(2)); m.feed(frame(12))
    assert m.status == 'ALIVE_NOT_RECOVERED'
    assert m.result()['survived']
    assert m.result()['touchdown_count'] == 0


def test_warmup_fall_retained_and_reset_pose_never_recovers():
    m = TrialMachine(PLANS[0], P, NODES['-10:+x'])
    f = frame(.2); f['fell'] = True
    m.feed(f)
    m.feed(frame(.22))  # a would-be reset pose must not belong to this trial
    assert len(m.frames) == 1
    assert m.result()['status'] == 'FELL'
    assert m.result()['com_xy_velocity_rmse'] is None


def test_precondition_timeout_not_replaced():
    plan = next(r for r in PLANS if r['experiment'] == 'E1')
    m = TrialMachine(plan, P, NODES['-10:+x'])
    m.feed(frame(0)); m.feed(frame(2)); m.feed(frame(6))
    assert m.status == 'PRECONDITION_FAILED'
    assert m.result()['recovery_steps'] is None
    assert not m.result()['push_applied']


def test_trigger_uses_past_interval_once_and_domain_exit_not_terminal():
    plan = next(r for r in PLANS if r['experiment'] == 'E1')
    m = TrialMachine(plan, P, NODES['-10:+x'])
    m.state = 'WAIT_REFERENCE_TOUCHDOWN'; m.ready_time = 2
    m.contact.contact = np.array([False, True]); m.contact.last_time = 1.8
    m.contact.last_foot = 1
    m.feed(frame(2, [20, 0], plane={'invalid_reason': 'heading_geometry'}))
    assert not m.feed(frame(2.02, [20, 0]))
    assert m.trigger['T_used_for_trigger'] == pytest.approx(.22)
    assert m.feed(frame(2.08, [20, 0]))
    before = [0] * 6
    after = additive_velocity(before, 0, plan['push_direction_heading'], plan['push_magnitude'])
    m.push_applied(before, after.tolist(), 0)
    assert m.state == 'OBSERVE' and m.status is None
    with pytest.raises(ValueError, match='exactly once'):
        m.push_applied(before, after.tolist(), 0)
    assert m.trigger['trigger_timing_error'] == pytest.approx(.005)


def test_execution_error_not_fall():
    m = TrialMachine(PLANS[0], P, NODES['-10:+x'])
    f = frame(.1); f['com_velocity'] = [float('nan'), 0, 0]
    m.feed(f)
    assert m.status == 'EVALUATION_ERROR'


def test_summary_denominators_and_pair_intersection():
    plans = [r for r in PLANS if r['experiment'] == 'E1'][:3]
    base = dict(push_applied=True, precondition_passed=True, survived=True,
                recovery_within_3_touchdowns=True, recovery_within_5_touchdowns=True, recovery_time=1., recovery_steps=3)
    good = {**plans[0], **base, 'status': 'RECOVERED_AND_SURVIVED'}
    bad = {**plans[1], **base, 'status': 'FELL', 'survived': False,
           'recovery_within_3_touchdowns': False, 'recovery_within_5_touchdowns': False, 'recovery_time': None, 'recovery_steps': None}
    pre = {**bad, **plans[2], 'status': 'PRECONDITION_FAILED', 'push_applied': False, 'precondition_passed': False}
    s = summarize([good, bad, pre], plans)
    assert s['E1']['task_recovery_success_rate'] == 1 / 3
    assert s['E1']['conditional_recovery_success_rate'] == .5
    assert s['curves']['time'][-1]['designated'] == 1 / 3
    comp = paired_comparison([good, bad, pre], [{**good, 'recovery_time': 2}, bad, pre], 3)
    assert comp['paired_success_count'] == 1
    assert comp['paired_mean_time_delta_s'] == -1
    assert comp['paired_relative_time_delta'] == -.5
