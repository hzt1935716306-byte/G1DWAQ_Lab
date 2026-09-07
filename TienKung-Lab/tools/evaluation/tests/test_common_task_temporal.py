"""Synthetic contract tests; test values are NOT candidate task thresholds."""
from collections import deque
import math

import numpy as np
import pytest

from g1_common_task_detector import (
    CommonTaskMeasurements, CommonReadinessDetector, CommonRecoveryDetector,
    PhysicalTouchdowns, TimestampWindow, gravity_tilt_wxyz, vertical_plane_clearance, wrap_angle,
)


def frame(t=0., **changes):
    return dict(time=t, com_velocity=[.4, 0, 0], root_velocity=[.4, 0, 0],
                command=[.4, 0, 0], root_quaternion_wxyz=[1, 0, 0, 0],
                angular_velocity_world_z=0., root_clearance_m=.8, forces=[0., 0.],
                fell=False, timeout=False, out_of_test_area=False, data_valid=True) | changes


def counts(physical=0, alternating=0):
    return dict(physical=physical, alternating=alternating, repeated_same_foot=0, simultaneous_events=0)


def contacts():
    return PhysicalTouchdowns(5., 3., 2, .08)


@pytest.mark.parametrize('duration,intervals', [(.60, 30), (1.20, 60)])
def test_complete_window_needs_intervals_plus_one_boundary_samples(duration, intervals):
    w = TimestampWindow(duration, .04)
    for t in np.arange(intervals) * .02:
        w.append(t, {'error': 1.})
        assert w.statistics() is None
    w.append(duration, {'error': 1.})
    s = w.statistics()
    assert s['time_intervals'] == intervals
    assert s['error']['integral'] == pytest.approx(duration)
    assert s['error']['mean'] == pytest.approx(1.)


def test_irregular_timestamps_clipped_boundary_use_time_not_sample_count():
    w = TimestampWindow(.60, .25)
    for t in [0, .07, .23, .41, .52, .73]:
        w.append(t, {'linear': 2*t+1})
    s = w.statistics()
    assert s['start'] == pytest.approx(.13)
    assert s['linear']['mean'] == pytest.approx(1 + .13 + .73)
    assert s['linear']['integral'] == pytest.approx(.60 * (1 + .13 + .73))


def test_gap_and_reset_cannot_supply_unobserved_hold():
    w = TimestampWindow(.60, .04)
    w.append(1., {'x': 0.})
    with pytest.raises(ValueError, match='gap'):
        w.append(1.1, {'x': 0.})
    with pytest.raises(ValueError, match='increase'):
        w.append(0., {'x': 0.})


def test_recovery_buffer_rejects_pre_push_and_boundary():
    w = TimestampWindow(.60, .04, after=2.)
    for t in [1.98, 2.]:
        with pytest.raises(ValueError, match='post-push'):
            w.append(t, {'x': 0.})
    for t in 2.02 + np.arange(30)*.02:
        w.append(t, {'x': 0.})
    assert w.statistics() is None
    w.append(2.62, {'x': 0.})
    assert w.statistics()['start'] == pytest.approx(2.02)


def test_error_formed_before_integration_prevents_cancellation():
    w = TimestampWindow(.6, .04)
    for i in range(31):
        m = CommonTaskMeasurements.from_frame(frame(i*.02, com_velocity=[.4+(-1)**i, 0, 0]))
        w.append(m.time, m.signals())
    assert w.statistics()['velocity_error_squared']['mean'] == pytest.approx(1.)
    still = CommonTaskMeasurements.from_frame(frame(com_velocity=[0, 0, 0]))
    assert still.signals()['velocity_error_squared'] == pytest.approx(.16)


@pytest.mark.parametrize('pitch_deg', [-10, -2, 0, 3, 10])
def test_gravity_tilt_has_no_teacher_center(pitch_deg):
    angle = math.radians(pitch_deg)
    q = [math.cos(angle/2), 0, math.sin(angle/2), 0]
    m = CommonTaskMeasurements.from_frame(frame(root_quaternion_wxyz=q))
    assert m.gravity_tilt_rad == pytest.approx(abs(angle), abs=1e-8)
    assert gravity_tilt_wxyz(-np.asarray(q)) == pytest.approx(abs(angle), abs=1e-8)


def test_quaternion_convention_and_wrap():
    assert gravity_tilt_wxyz([0, 0, 0, 1]) == pytest.approx(0.)  # yaw 180deg
    assert gravity_tilt_wxyz([0, 1, 0, 0]) == pytest.approx(math.pi)  # roll 180deg
    assert wrap_angle(math.radians(361)) == pytest.approx(math.radians(1))
    with pytest.raises(ValueError, match='normalized'):
        gravity_tilt_wxyz([2, 0, 0, 0])
    with pytest.raises(ValueError, match='mismatch'):
        CommonTaskMeasurements.from_frame(frame(gravity_tilt_rad=.2))


def test_local_plane_height_changes_with_horizontal_position():
    a = math.radians(-10)
    n, p = [-math.sin(a), 0, math.cos(a)], [10, 20, 3]
    for x in [-4, 0, 5]:
        root = [10+x, 20, 3+math.tan(a)*x+.8]
        assert vertical_plane_clearance(root, n, p) == pytest.approx(.8)
    assert vertical_plane_clearance([11, 20, 3.8], n, p) == pytest.approx(.8-math.tan(a))


@pytest.mark.parametrize('field', ['command', 'root_clearance_m', 'angular_velocity_world_z', 'data_valid'])
def test_missing_measurements_never_default_to_zero(field):
    f = frame(); del f[field]
    with pytest.raises(ValueError, match='Missing'):
        CommonTaskMeasurements.from_frame(f)


def test_tilt_only_requires_provenance_and_nonfinite_is_error():
    f = frame(); del f['root_quaternion_wxyz']; f['gravity_tilt_rad'] = .02
    with pytest.raises(ValueError, match='provenance'):
        CommonTaskMeasurements.from_frame(f)
    f['gravity_tilt_provenance'] = 'acos(cos(roll)*cos(pitch)) from archived XYZ Euler'
    assert CommonTaskMeasurements.from_frame(f).gravity_tilt_rad == .02
    with pytest.raises(ValueError, match='Non-finite'):
        CommonTaskMeasurements.from_frame(frame(root_clearance_m=np.nan))


def test_initial_contact_is_not_landing_then_true_same_foot_relanding_counts():
    d = contacts()
    for t in [0, .02, .04]:
        assert d.update(t, [20, 20])[0] == []
    for t in [.1, .12]: d.update(t, [0, 20])
    d.update(.2, [20, 20]); physical, _ = d.update(.22, [20, 20])
    assert len(physical) == 1 and d.count == 1
    for t in [.3, .32]: d.update(t, [0, 20])
    d.update(.4, [20, 20]); physical, alternate = d.update(.42, [20, 20])
    assert len(physical) == 1 and d.count == 2 and d.repeated == 1
    assert alternate is None


def test_contact_chatter_and_independent_foot_debounce():
    d = contacts()
    d.update(0, [20, 20]); d.update(.02, [0, 20]); d.update(.04, [20, 20])
    assert d.count == 0
    for t in [.1, .12]: d.update(t, [0, 0])
    d.update(.2, [20, 0]); d.update(.22, [20, 0])
    d.update(.24, [20, 20]); d.update(.26, [20, 20])
    assert d.count == 2  # right landing is not suppressed by left's .08 s dead-time


def test_simultaneous_has_two_physical_events_without_fabricated_order():
    d = contacts()
    d.update(0, [0, 0]); d.update(.02, [0, 0])
    d.update(.1, [20, 20]); physical, alternate = d.update(.12, [20, 20])
    assert len(physical) == 2 and all(e['simultaneous'] for e in physical)
    assert d.count == 2 and d.simultaneous == 1
    assert alternate is None and d.alternating_count == 0
    d.reset()
    assert d.update(0, [20, 20])[0] == [] and d.count == 0


def ready_contacts():
    d = contacts()
    d.recent_alternating = deque({'time': t, 'foot': foot} for t, foot in
                                 [(1.2, 'right'), (1.5, 'left'), (1.7, 'right'), (2., 'left')])
    d.completed_intervals.extend([{'start': 1.5, 'end': 1.72, 'start_foot': 'left', 'interval_s': .22},
                                  {'start': 1.72, 'end': 2.06, 'start_foot': 'right', 'interval_s': .34}])
    return d


def test_readiness_warmup_hold_and_same_start_foot_prediction():
    d = ready_contacts(); ready = CommonReadinessDetector('left', .25)
    assert not ready.update(1.98, True, d, None)
    assert not ready.update(2., True, d, None)
    assert not ready.update(2.18, True, d, None)
    event = {'foot': 'left', 'time': 2.2, 'interval_s': .34}
    assert not ready.update(2.2, True, d, event)
    assert ready.ready_time == 2.2
    assert ready.scheduled == pytest.approx(2.255)
    assert ready.trigger['interval_prediction']['source'] == 'same_start_foot'
    assert ready.update(2.26, True, d, None)
    ready.push_applied()
    assert not ready.update(2.28, True, d, None)


def test_readiness_loss_does_not_retry_or_push():
    d = ready_contacts(); ready = CommonReadinessDetector('left', .25)
    ready.update(2., True, d, None)
    ready.update(2.2, True, d, {'foot': 'left', 'time': 2.2, 'interval_s': .34})
    assert not ready.update(2.24, False, d, None)
    assert ready.reason == 'readiness_lost_before_push'
    assert not ready.update(2.28, True, d, None)
    with pytest.raises(ValueError, match='readiness'):
        ready.push_applied()


def test_old_reset_touchdowns_do_not_satisfy_recent_readiness():
    d = ready_contacts(); ready = CommonReadinessDetector('left', .25)
    d.alternating_count = 100
    ready.update(5.8, True, d, None)
    ready.update(6., True, d, None)
    assert ready.reason == 'insufficient_recent_alternating_touchdowns'


def test_recovery_entry_confirmation_and_physical_steps_boundary():
    d = CommonRecoveryDetector(2.)
    d.update(2.02, False, counts())
    d.update(2.62, True, counts(2, 1))
    d.update(3.10, True, counts(3, 2))
    assert d.first_entry is None
    d.update(3.12, True, counts(4, 3))
    assert d.first_entry == pytest.approx(2.62)
    assert d.first_confirmation == pytest.approx(3.12)
    d.update(12., True, counts(30, 25))
    r = d.result(True)
    assert r['recovery_time'] == pytest.approx(.62)  # never subtract W
    assert r['recovery_steps'] == 2 and r['confirmation_steps'] == 4
    assert r['recovery_within_3_touchdowns'] and r['recovered_sustained_and_survived']


def test_relapse_preserves_once_and_restarts_sustained_entry():
    d = CommonRecoveryDetector(0)
    for t, good, steps in [(.02, False, 0), (.62, True, 1), (1.12, True, 3),
                           (2., False, 4), (2.3, True, 5), (2.8, True, 7), (10., True, 20)]:
        d.update(t, good, counts(steps))
    r = d.result(True)
    assert r['first_recovery_entry'] == .62 and r['first_confirmation'] == 1.12
    assert r['relapse_count'] == 1
    assert r['out_of_domain_duration_after_confirmation'] == pytest.approx(.3)
    assert r['sustained_recovery_entry'] == 2.3 and r['recovery_steps'] == 5


@pytest.mark.parametrize('last_entry,expected', [(8., True), (8.02, False)])
def test_entry_deadline_and_confirmation_after_eight(last_entry, expected):
    d = CommonRecoveryDetector(0)
    d.update(.02, False, counts())
    d.update(last_entry, True, counts(10))
    d.update(last_entry+.5, True, counts(12))
    d.update(10., True, counts(20))
    r = d.result(True)
    assert r['recovered_sustained_and_survived'] is expected
    assert (r['recovery_time'] is not None) is expected


def test_final_relapse_or_later_fall_does_not_preserve_main_success():
    for terminal in [False, True]:
        d = CommonRecoveryDetector(0)
        d.update(.62, True, counts(2)); d.update(1.12, True, counts(4))
        d.update(10., False, counts(30), terminal=terminal)
        r = d.result(not terminal)
        assert r['recovered_once_and_survived'] is (not terminal)
        assert not r['recovered_sustained_and_survived']
        assert r['recovery_time'] is None and r['recovery_steps'] is None
        assert r['first_confirmation'] == 1.12


def test_incomplete_observation_or_no_confirmation_is_not_success():
    d = CommonRecoveryDetector(0)
    d.update(.62, True, counts()); d.update(1.1, True, counts())
    assert not d.result(True)['recovered_once_and_survived']
    d.update(1.12, True, counts())
    assert not d.result(True)['recovered_sustained_and_survived']
    with pytest.raises(ValueError, match='post-push'):
        CommonRecoveryDetector(1).update(1., True, counts())
