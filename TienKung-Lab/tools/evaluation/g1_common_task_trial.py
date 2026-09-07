"""v2 trial adapter: reuse the evaluator, native inference and legacy storage API."""
from __future__ import annotations
import math
import numpy as np

from g1_common_task_detector import (METRICS_VERSION, EPS, MEASUREMENT_FIELDS, CommonTaskMeasurements, CommonTaskGate,
    PhysicalTouchdowns, CommonReadinessDetector, CommonRecoveryDetector, vertical_plane_clearance)
from g1_recovery_metrics import TrialMachine, additive_velocity, trajectory_metrics


class CommonTaskTrialMachine(TrialMachine):
    def __init__(self, plan, protocol, reference=None):
        if reference is not None:
            raise ValueError('Common task judge must not receive teacher nominal inputs')
        super().__init__(plan, protocol, None)
        m = protocol['metrics']
        self.contact = PhysicalTouchdowns(m['contact_force_threshold_n'], m['contact_release_threshold_n'],
                                         m['contact_stable_frames'], m['contact_debounce_s'])
        self.config = m['detector']
        self.standing = plan['command_name'] == 'standing'
        self.readiness_gate = None if self.standing else CommonTaskGate(self.config, 'readiness')
        self.recovery_gate = None
        self.readiness = None
        if plan['experiment'] == 'E1':
            if self.standing:
                raise ValueError('Standing recovery is not implemented')
            self.readiness = CommonReadinessDetector(plan['reference_touchdown_foot'], plan['target_phase'])
        self.push_counts = None
        self.failure_reason = None
        self.last_measurement_time = None

    def feed(self, frame):
        if self.status:
            return False
        self.t = float(frame['time'])
        f = dict(frame)
        self.frames.append(f)
        # Native Plane diagnostics remain in f for archival; they are not even
        # passed into the typed common measurement constructor.
        m = CommonTaskMeasurements.from_frame({k: f[k] for k in MEASUREMENT_FIELDS if k in f})
        if self.last_measurement_time is not None:
            dt = m.time - self.last_measurement_time
            if dt <= EPS or dt > self.config['sampling']['nominal_period_s'] + EPS:
                raise ValueError('Missing/out-of-order physical sample; cannot interpolate a missing trajectory')
        self.last_measurement_time = m.time
        if not np.allclose(m.command, [self.plan[k] for k in ('command_vx', 'command_vy', 'command_yaw')], atol=1e-7, rtol=0):
            raise ValueError('Measured command differs from manifest')
        expected_clearance = vertical_plane_clearance(f['root_position'], f['local_plane_normal'], f['local_plane_point'])
        if not math.isclose(expected_clearance, m.root_clearance_m, abs_tol=1e-5):
            raise ValueError('Root clearance differs from local physical plane')
        physical, alternating = self.contact.update(m.time, m.forces)
        f.update(gravity_tilt_rad=m.gravity_tilt_rad, physical_contact=self.contact.contact.tolist(),
                 physical_touchdown_flags=[any(e['foot'] == foot for e in physical) for foot in ('left', 'right')],
                 alternating_touchdown_flag=alternating is not None,
                 post_push_sample=self.push_time is not None and m.time > self.push_time + EPS)
        if m.terminal or m.out_of_area:
            if self.recovery is not None:
                self.recovery.update(m.time, False, self.post_push_counts(), terminal=True)
            self.finish('FELL' if m.terminal else 'OUT_OF_TEST_AREA')
            return False
        if self.push_time is not None:
            gate = self.recovery_gate.update(m)
            self.events.append({'event': 'common_recovery_window', **gate})
            self.recovery.update(m.time, gate['passed'], self.post_push_counts())
            if m.time >= self.push_time + self.p['e1']['observation_s'] - EPS:
                good = self.recovery.result(True)['recovered_sustained_and_survived']
                self.finish('RECOVERED_AND_SURVIVED' if good else 'ALIVE_NOT_RECOVERED')
            return False
        gate = self.readiness_gate.update(m) if self.readiness_gate else None
        if gate:
            self.events.append({'event': 'common_readiness_window', **gate})
        if self.plan['experiment'] == 'E0':
            if m.time >= self.p['e0']['warmup_s'] and self.state == 'WARMUP':
                self.transition('OBSERVE', m.time)
            if m.time >= self.p['e0']['warmup_s'] + self.p['e0']['observation_s'] - EPS:
                self.finish('E0_COMPLETED')
            return False
        due = self.readiness.update(m.time, gate['passed'], self.contact, alternating)
        self.precondition_passed = self.readiness.ready_time is not None
        self.trigger.update(self.readiness.trigger)
        if self.readiness.state == 'PRECONDITION_FAILED':
            self.failure_reason = self.readiness.reason.upper()
            self.finish('PRECONDITION_FAILED')
        elif self.state != self.readiness.state:
            self.transition(self.readiness.state, m.time)
        return due

    def post_push_counts(self):
        # Snapshot includes physical events exactly at t_e; subtracting implements
        # strict t_e < event_time <= current sample, including simultaneous events.
        now = self.contact.counts()
        return {key: now[key] - self.push_counts[key] for key in now}

    def push_applied(self, before, after, yaw):
        if self.push_time is not None or self.state != 'APPLY_PUSH_ONCE':
            raise ValueError('Physical jump may be applied exactly once')
        expected = additive_velocity(before, yaw, self.plan['push_direction_heading'], self.plan['push_magnitude'])
        if not np.allclose(expected, after, atol=2e-6, rtol=0):
            raise ValueError('Actual velocity increment mismatch')
        self.readiness.push_applied()
        self.push_time = self.t
        self.push_counts = self.contact.counts()
        self.trigger.update(actual_push_time=self.t, push_start_time=self.t, push_end_time=self.t,
                            trigger_timing_error=self.t - self.trigger['scheduled_push_time'],
                            actual_contact_state=self.contact.contact.tolist(), velocity_before=list(before),
                            velocity_after=list(after), push_heading_yaw=yaw)
        self.events.append({'event': 'velocity_jump', **self.trigger})
        self.recovery_gate = CommonTaskGate(self.config, 'recovery', push_end=self.t)
        self.recovery = CommonRecoveryDetector(self.t)
        self.transition('OBSERVE', self.t)

    def result(self, include_metrics=True):
        if not self.status:
            raise ValueError('Cannot finalize active trial')
        survived = self.status in ('E0_COMPLETED', 'RECOVERED_AND_SURVIVED', 'ALIVE_NOT_RECOVERED')
        r = self.recovery.result(survived) if self.recovery else CommonRecoveryDetector(0).result(False)
        record = {**self.plan, 'metrics_version': METRICS_VERSION, 'status': self.status,
                  'parameter_status': 'candidate_unvalidated', 'precondition_passed': self.precondition_passed,
                  'precondition_failure_reason': self.failure_reason, 'push_applied': self.push_time is not None,
                  'survived': survived, 'recovery_applicable': self.push_time is not None,
                  'e0_outcome': 'NORMAL_OBSERVATION_COMPLETED' if self.status == 'E0_COMPLETED' else None,
                  'observed_duration': self.t - (self.push_time or 0),
                  'observed_touchdowns': self.post_push_counts()['physical'] if self.push_time is not None else 0,
                  'touchdown_count': self.contact.count, 'physical_touchdown_count': self.contact.count,
                  'alternating_touchdown_count': self.contact.alternating_count,
                  'repeated_same_foot_count': self.contact.repeated, 'simultaneous_touchdown_events': self.contact.simultaneous,
                  'post_push_event_counts': self.post_push_counts() if self.push_time is not None else None,
                  'contact_chatter_count': self.contact.chatter,
                  'step_intervals_s': [e['interval_s'] for e in self.contact.completed_intervals],
                  'readiness_confirmation': self.readiness.ready_time if self.readiness else None,
                  **self.trigger, **r}
        start = self.p['e0']['warmup_s'] if self.plan['experiment'] == 'E0' else self.push_time
        frames = [f for f in self.frames if start is not None and f['time'] >= start]
        if include_metrics:
            record.update(trajectory_metrics(frames, self.plan))
            record.update(continuous_diagnostics(self.frames, frames))
        else:
            record['metrics_unavailable'] = 'execution_error'
        return record

    def all_events(self):
        events = super().all_events()
        if self.readiness:
            events.extend({**e, 'trial_id': self.plan['trial_id']} for e in self.readiness.events)
        return sorted(events, key=lambda e: e.get('time', 0))

    def trace(self):
        trace = super().trace()
        extra = ('root_quaternion_wxyz', 'angular_velocity_world_z', 'command', 'root_clearance_m',
                 'local_plane_normal', 'local_plane_point', 'data_valid', 'gravity_tilt_rad',
                 'physical_contact', 'physical_touchdown_flags', 'alternating_touchdown_flag', 'post_push_sample')
        for key in extra:
            if all(key in f for f in self.frames):
                trace[key] = np.asarray([f[key] for f in self.frames])
        return trace


def continuous_diagnostics(all_frames, selected):
    """Heading/path are descriptive and never enter CommonTaskMeasurements/gates."""
    if not all_frames:
        return {'heading_initial_rad': None, 'path_displacement_xy_m': None}
    initial = all_frames[0]
    yaw = np.unwrap([f['yaw'] for f in all_frames])
    position = np.array([f['root_position'] for f in all_frames])
    command = np.array([f['command'] for f in all_frames])[:, :2]
    # Path intent uses initial H; current-H tracking remains the primary task.
    c, s = np.cos(yaw[0]), np.sin(yaw[0])
    expected_velocity = command @ np.array([[c, s], [-s, c]])
    dt = np.diff([f['time'] for f in all_frames])
    desired = np.vstack([np.zeros(2), np.cumsum((expected_velocity[:-1]+expected_velocity[1:])*.5*dt[:, None], axis=0)])
    deviation = position[:, :2]-position[0, :2]-desired
    return {'heading_initial_rad': float(yaw[0]), 'heading_final_rad': float(yaw[-1]),
            'path_initial_xy_m': position[0, :2].tolist(),
            'heading_offset_peak_rad': float(np.max(np.abs(yaw-yaw[0]))),
            'path_displacement_xy_m': (position[-1, :2]-position[0, :2]).tolist(),
            'path_deviation_final_xy_m': deviation[-1].tolist(),
            'path_deviation_peak_m': float(np.linalg.norm(deviation, axis=1).max()),
            'gravity_tilt_peak_rad': max((f['gravity_tilt_rad'] for f in selected), default=None),
            'root_vertical_clearance_min_m': min((f['root_clearance_m'] for f in selected), default=None),
            'world_z_angular_velocity_peak_abs_radps': max((abs(f['angular_velocity_world_z']) for f in selected), default=None)}


def replay_common_trial(plan, protocol, trace, source_record):
    """Same online machine and real recorded trigger; no counterfactual pushes."""
    import json
    machine = CommonTaskTrialMachine(plan, protocol)
    for index, _ in enumerate(trace['time']):
        frame = {key: trace[key][index].tolist() for key in trace if key != 'plane_json'}
        frame['plane'] = json.loads(str(trace['plane_json'][index])) if 'plane_json' in trace else None
        due = machine.feed(frame)
        recorded = (source_record.get('push_applied') and abs(frame['time'] - source_record['actual_push_time']) < EPS)
        if bool(due) != bool(recorded):
            raise ValueError('Recorded trigger disagrees with causal online detector; cannot invent a push')
        if recorded:
            machine.push_applied(source_record['velocity_before'], source_record['velocity_after'], source_record['push_heading_yaw'])
    if not machine.status:
        raise ValueError('Recorded trial ends before its declared observation/precondition deadline')
    return machine
