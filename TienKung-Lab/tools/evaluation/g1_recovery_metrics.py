"""Simulator-independent physical judge and trial state machine.

Frame errors follow legged_lab/recovery/practical_metrics.py exactly: mean of
per-frame norms/absolute errors, never norm of mean velocity. GT is judge-only.
"""
from __future__ import annotations
from collections import Counter
import math
import numpy as np

METRICS_VERSION = 'practical_interval_confirm2_v1'


def frame_errors(com_xy, command_xy, roll_pitch, nominal):
    return float(np.linalg.norm(np.asarray(com_xy) - command_xy)), np.abs(np.asarray(roll_pitch) - nominal)


def additive_velocity(before, yaw, direction, magnitude):
    before = np.asarray(before, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if before.shape != (6,) or direction.shape != (2,) or not np.all(np.isfinite(before)):
        raise ValueError('Velocity jump requires finite six-component root velocity and XY direction')
    if not np.isfinite(yaw) or not np.isfinite(magnitude) or magnitude < 0 or not np.isclose(np.linalg.norm(direction), 1):
        raise ValueError('Invalid heading/direction/magnitude')
    c, s = math.cos(yaw), math.sin(yaw)
    delta = np.array([[c, -s], [s, c]]) @ (direction * magnitude)
    after = before.copy()
    after[:2] += delta
    return after


class TouchdownDetector:
    """Hysteresis plus stable frames and global dead-time; repeated feet logged, not counted."""
    def __init__(self, config):
        self.cfg = config
        self.contact = None
        self.pending = [0, 0]
        self.last_time = -math.inf
        self.last_foot = None
        self.count = 0
        self.chatter = 0
        self.repeated = 0
        self.last_interval = None
        self.events = []

    def update(self, t, forces):
        f = np.asarray(forces)
        if self.contact is None:
            self.contact = f > self.cfg['contact_force_threshold_n']
            return None
        candidates = []
        for i in range(2):
            target = f[i] > (self.cfg['contact_release_threshold_n'] if self.contact[i] else self.cfg['contact_force_threshold_n'])
            if target == self.contact[i]:
                if self.pending[i]:
                    self.chatter += 1
                    self.events.append({'time': t, 'event': 'contact_chatter', 'foot': i})
                self.pending[i] = 0
                continue
            self.pending[i] += 1
            if self.pending[i] >= self.cfg['contact_stable_frames']:
                self.contact[i] = target
                self.pending[i] = 0
                if target:
                    candidates.append(i)
        if not candidates:
            return None
        foot = max(candidates, key=lambda i: f[i])
        if t - self.last_time < self.cfg['contact_debounce_s'] - 1e-9:
            self.chatter += 1
            self.events.append({'time': t, 'event': 'touchdown_debounce', 'foot': foot})
            return None
        if foot == self.last_foot:
            self.repeated += 1
            self.events.append({'time': t, 'event': 'repeated_same_foot', 'foot': foot})
            return None
        interval = t - self.last_time if math.isfinite(self.last_time) else None
        self.last_interval = interval
        self.last_time, self.last_foot = t, foot
        self.count += 1
        event = {'time': t, 'foot': 'left' if foot == 0 else 'right', 'interval_s': interval, 'event': 'touchdown'}
        self.events.append(event)
        return event


class RecoveryDetector:
    def __init__(self, push_time, deadline_s=8, observation_s=10):
        self.push_time = push_time
        self.deadline_s = deadline_s
        self.observation_s = observation_s
        self.candidate = None
        self.confirmed = None
        self.events = []

    def interval(self, start, end, passed, steps):
        if self.confirmed or start < self.push_time - 1e-9 or end > self.push_time + self.observation_s + 1e-9:
            return
        if not passed:
            self.candidate = None
            return
        if self.candidate and abs(start - self.candidate['entry_time']) < 1e-7:
            self.confirmed = {**self.candidate, 'confirmation_time': end,
                              'confirmation_delay': end - self.candidate['entry_time']}
            self.events.append({'event': 'recovery_confirmed', **self.confirmed})
        elif end - self.push_time <= self.deadline_s + 1e-9:
            self.candidate = {'entry_time': end, 'recovery_time': end - self.push_time, 'recovery_steps': steps}
            self.events.append({'event': 'recovery_entry_candidate', **self.candidate})
        else:
            self.candidate = None

    def result(self, survived):
        c = self.confirmed
        success = bool(c and survived)
        return {'recovery_time': c['recovery_time'] if success else None,
                'recovery_steps': c['recovery_steps'] if success else None,
                'confirmed_recovery_event': c,
                'recovery_within_3_touchdowns': bool(success and c['recovery_steps'] <= 3),
                'recovery_within_5_touchdowns': bool(success and c['recovery_steps'] <= 5)}


class TrialMachine:
    """RESET → WARMUP → READY_CHECK → WAIT_REFERENCE_TOUCHDOWN → APPLY_PUSH_ONCE → OBSERVE → FINALIZE."""
    def __init__(self, plan, protocol, reference):
        self.plan, self.p, self.reference = plan, protocol, reference
        self.state = 'RESET'
        self.contact = TouchdownDetector(protocol['metrics'])
        self.frames, self.events, self.interval_errors = [], [], []
        self.interval_start = None
        self.last_interval_passed = False
        self.scheduled = None
        self.trigger = {}
        self.push_time = None
        self.push_touchdowns = 0
        self.precondition_passed = False
        self.recovery = None
        self.status = None
        self.t = 0.0
        self.transition('WARMUP', 0)

    def transition(self, state, t):
        self.state = state
        self.events.append({'event': 'state', 'state': state, 'time': t})

    def feed(self, frame):
        """Consume a physical frame captured before any reset. Returns whether a push is due."""
        if self.status:
            return False
        self.t = t = float(frame['time'])
        self.frames.append(frame)
        if not all(np.all(np.isfinite(frame[k])) for k in ('root_velocity', 'com_velocity', 'roll_pitch', 'yaw', 'forces')):
            self.finish('EVALUATION_ERROR')
            return False
        # Physical terminal flags take precedence over apparent recovered/reset poses.
        if frame.get('evaluation_error') or frame.get('timeout'):
            self.finish('EVALUATION_ERROR')
            return False
        if frame.get('fell'):
            self.finish('FELL')
            return False
        if frame.get('out_of_test_area'):
            self.finish('OUT_OF_TEST_AREA')
            return False
        td = self.contact.update(t, frame['forces'])
        errors = None
        if self.reference:
            ref = self.reference
            v, rp = frame_errors(frame['com_velocity'][:2], [self.plan['command_vx'], self.plan['command_vy']],
                                 frame['roll_pitch'], [ref['roll_star'], ref['pitch_star']])
            errors = [v, *rp]
        if td:
            if self.push_time is not None and t > self.push_time:
                self.push_touchdowns += 1
            if self.interval_start is not None and self.interval_errors:
                means = np.mean(self.interval_errors, axis=0)
                ref = self.reference
                passed = bool(np.all(means <= [ref['mean_velocity_error_threshold'], ref['mean_abs_roll_error_threshold'], ref['mean_abs_pitch_error_threshold']]))
                self.last_interval_passed = passed
                self.events.append({'event': 'practical_interval', 'start': self.interval_start,
                                    'time': t, 'means': means.tolist(), 'passed': passed})
                if self.recovery:
                    self.recovery.interval(self.interval_start, t, passed, self.push_touchdowns)
            self.interval_start = t
            self.interval_errors = []
        if errors is not None and self.interval_start is not None:
            self.interval_errors.append(errors)
        if self.plan['experiment'] == 'E0':
            if t >= self.p['e0']['warmup_s'] and self.state == 'WARMUP':
                self.transition('OBSERVE', t)
            if t >= self.p['e0']['warmup_s'] + self.p['e0']['observation_s'] - 1e-8:
                self.finish('ALIVE_NOT_RECOVERED')
            return False
        cfg = self.p['e1']
        if self.state in ('WARMUP', 'READY_CHECK'):
            if t >= cfg['warmup_s']:
                if self.state == 'WARMUP':
                    self.transition('READY_CHECK', t)
                if self.contact.count >= cfg['required_alternating_touchdowns'] and self.last_interval_passed:
                    self.precondition_passed = True
                    self.ready_time = t
                    self.transition('WAIT_REFERENCE_TOUCHDOWN', t)
                elif t >= cfg['readiness_deadline_s'] - 1e-8:
                    self.finish('PRECONDITION_FAILED')
        if self.state == 'WAIT_REFERENCE_TOUCHDOWN':
            if td and td['foot'] == self.plan['reference_touchdown_foot'] and td['interval_s'] is not None:
                T = td['interval_s']
                self.scheduled = t + self.plan['target_phase'] * T
                self.trigger = {'reference_touchdown_time': t, 'T_used_for_trigger': T,
                                'scheduled_push_time': self.scheduled}
                self.transition('APPLY_PUSH_ONCE', t)
            elif t >= self.ready_time + cfg['reference_wait_s']:
                self.finish('PRECONDITION_FAILED')
        if self.state == 'APPLY_PUSH_ONCE' and t >= self.scheduled - 1e-9:
            return True
        if self.state == 'OBSERVE' and t >= self.push_time + cfg['observation_s'] - 1e-8:
            self.finish('RECOVERED_AND_SURVIVED' if self.recovery.confirmed else 'ALIVE_NOT_RECOVERED')
        return False

    def push_applied(self, before, after, yaw):
        if self.state != 'APPLY_PUSH_ONCE' or self.push_time is not None:
            raise ValueError('Physical jump may be applied exactly once')
        expected = additive_velocity(before, yaw, self.plan['push_direction_heading'], self.plan['push_magnitude'])
        if not np.allclose(expected, after, atol=2e-6, rtol=0):
            raise ValueError('Actual velocity increment mismatch')
        self.push_time = self.t
        self.trigger.update(actual_push_time=self.t, trigger_timing_error=self.t - self.scheduled,
                            actual_contact_state=self.contact.contact.tolist(), velocity_before=list(before), velocity_after=list(after),
                            push_heading_yaw=yaw)
        self.events.append({'event': 'velocity_jump', **self.trigger})
        self.recovery = RecoveryDetector(self.t, self.p['e1']['recovery_deadline_s'], self.p['e1']['observation_s'])
        self.transition('OBSERVE', self.t)

    def finish(self, status):
        self.status = status
        self.transition('FINALIZE', self.t)

    def result(self):
        if not self.status:
            raise ValueError('Cannot finalize active trial')
        survived = self.status in ('RECOVERED_AND_SURVIVED', 'ALIVE_NOT_RECOVERED')
        recovery = self.recovery.result(survived) if self.recovery else RecoveryDetector(0).result(False)
        record = {**self.plan, 'status': self.status, 'precondition_passed': self.precondition_passed,
                  'push_applied': self.push_time is not None, 'survived': survived,
                  'observed_duration': self.t - (self.push_time or 0), 'observed_touchdowns': self.push_touchdowns,
                  'touchdown_count': self.contact.count, 'contact_chatter_count': self.contact.chatter,
                  'repeated_same_foot_count': self.contact.repeated,
                  'step_intervals_s': [e['interval_s'] for e in self.contact.events if e['event'] == 'touchdown' and e['interval_s'] is not None],
                  **self.trigger, **recovery}
        start = self.p['e0']['warmup_s'] if self.plan['experiment'] == 'E0' else self.push_time
        frames = [f for f in self.frames if start is not None and f['time'] >= start]
        record.update(trajectory_metrics(frames, self.plan))
        return record

    def all_events(self):
        return [{**e, 'trial_id': self.plan['trial_id']} for e in
                self.events + self.contact.events + (self.recovery.events if self.recovery else [])]

    def trace(self):
        fields = ('time', 'root_velocity', 'root_velocity_world', 'com_velocity', 'roll_pitch', 'yaw', 'forces',
                  'root_position', 'fell', 'timeout', 'out_of_test_area')
        trace = {key: np.asarray([f[key] for f in self.frames]) for key in fields}
        trace['plane_json'] = np.asarray([__import__('json').dumps(f.get('plane')) for f in self.frames])
        return trace


def trajectory_metrics(frames, plan):
    names = ['root_xy_velocity_rmse', 'com_xy_velocity_rmse', 'heading_drift', 'roll_std', 'pitch_std',
             'peak_tilt', 'peak_velocity_error', 'integrated_velocity_error']
    if not frames:
        return {k: None for k in names} | {'plane_diagnostics': None}
    command = np.array([plan['command_vx'], plan['command_vy']])
    root = np.array([f['root_velocity'][:2] for f in frames])
    com = np.array([f['com_velocity'][:2] for f in frames])
    rp = np.array([f['roll_pitch'] for f in frames])
    yaw = np.unwrap([f['yaw'] for f in frames])
    error = np.linalg.norm(com - command, axis=1)
    time = np.array([f['time'] for f in frames])
    values = [np.sqrt(np.mean(np.sum((root - command)**2, axis=1))),
              np.sqrt(np.mean(error**2)), yaw[-1] - yaw[0], *np.std(rp, axis=0),
              np.max(np.linalg.norm(rp, axis=1)), np.max(error), np.trapz(error, time)]
    result = {k: float(v) if np.isfinite(v) else None for k, v in zip(names, values)}
    plane = [f['plane'] for f in frames if f.get('plane') is not None]
    result['plane_diagnostics'] = None
    if plane:
        result['plane_diagnostics'] = {
            'frames': len(plane),
            'context_valid_rate': float(np.mean([d['context_valid'] for d in plane])),
            'standing_na_rate': float(np.mean([d['standing_na'] for d in plane])),
            'invalid_reasons': dict(Counter(d['invalid_reason'] for d in plane if d['invalid_reason'])),
            'N_distribution': dict(Counter(str(d['N']) for d in plane if d['N'] is not None)),
            'margin_distribution': [d['margin'] for d in plane if d['margin'] is not None],
            'solver_failure_frames': sum(d['solver_failure'] for d in plane),
            'reward_components': {k: {'sum': sum(d[k] for d in plane), 'nonzero_frames': sum(d[k] != 0 for d in plane)}
                                  for k in ('reward_progress', 'reward_step_cost', 'reward_td5', 'reward_total')}}
    return result


def quantiles(values):
    values = [v for v in values if v is not None]
    return {'count': len(values), 'median': float(np.median(values)) if values else None,
            'p90': float(np.percentile(values, 90)) if values else None}


def summarize(records, manifest):
    def rate(n, d):
        return n / d if d else None
    e0 = [r for r in records if r['experiment'] == 'E0']
    e1 = [r for r in records if r['experiment'] == 'E1']
    designated = sum(r['experiment'] == 'E1' for r in manifest)
    pushed = [r for r in e1 if r['push_applied']]
    success = [r for r in e1 if r['status'] == 'RECOVERED_AND_SURVIVED']
    summary = {'status': 'INVALID' if any(r['status'] == 'EVALUATION_ERROR' for r in records) else
               ('COMPLETE' if len(records) == len(manifest) else 'PARTIAL'),
               'designated': len(manifest), 'executed': len(records), 'E0': {}, 'E1': {}}
    for label, group in [('moving', [r for r in e0 if r['command_name'] != 'standing']),
                          ('standing', [r for r in e0 if r['command_name'] == 'standing'])]:
        n = sum(r['experiment'] == 'E0' and (r['command_name'] == 'standing') == (label == 'standing') for r in manifest)
        summary['E0'][label] = {'designated': n, 'executed': len(group), 'survival_rate': rate(sum(r['survived'] for r in group), n),
                               **{k: quantiles([r.get(k) for r in group]) for k in
                                  ('root_xy_velocity_rmse', 'com_xy_velocity_rmse', 'heading_drift', 'roll_std', 'pitch_std', 'touchdown_count')},
                               'step_interval_s': quantiles([v for r in group for v in r.get('step_intervals_s', [])])}
    summary['E1'] = {'designated': designated, 'executed': len(e1), 'pushed': len(pushed), 'successes': len(success),
        'precondition_pass_rate': rate(sum(r['precondition_passed'] for r in e1), designated),
        'task_recovery_success_rate': rate(len(success), designated),
        'conditional_recovery_success_rate': rate(len(success), len(pushed)),
        'pushed_10s_survival_rate': rate(sum(r['survived'] for r in pushed), len(pushed)),
        'failure_rate': rate(designated - len(success), designated),
        'recovery_within_3_touchdowns': rate(sum(r['recovery_within_3_touchdowns'] for r in success), designated),
        'recovery_within_5_touchdowns': rate(sum(r['recovery_within_5_touchdowns'] for r in success), designated),
        **{k: quantiles([r.get(k) for r in (success if k.startswith('recovery_') else pushed)]) for k in
           ('recovery_time', 'recovery_steps', 'peak_tilt', 'peak_velocity_error', 'integrated_velocity_error', 'heading_drift')},
        'terminal_counts': dict(Counter(r['status'] for r in e1))}
    summary['curves'] = {}
    for name, key, grid in [('time', 'recovery_time', np.linspace(0, 10, 101)),
                            ('steps', 'recovery_steps', range(0, max([10] + [r['recovery_steps'] for r in success]) + 1))]:
        summary['curves'][name] = [{'x': float(x), 'designated': rate(sum(r[key] <= x for r in success), designated),
                                   'pushed': rate(sum(r[key] <= x for r in success), len(pushed))} for x in grid]
    return summary


def paired_comparison(a, b, designated):
    aa = {r['trial_id']: r for r in a if r['experiment'] == 'E1'}
    bb = {r['trial_id']: r for r in b if r['experiment'] == 'E1'}
    common = [k for k in aa.keys() & bb.keys() if aa[k]['status'] == bb[k]['status'] == 'RECOVERED_AND_SURVIVED']
    delta = [aa[k]['recovery_time'] - bb[k]['recovery_time'] for k in common]
    baseline_time = float(np.mean([bb[k]['recovery_time'] for k in common])) if common else None
    mean_delta = float(np.mean(delta)) if delta else None
    return {'paired_success_count': len(common), 'paired_mean_time_delta_s': mean_delta,
            'paired_relative_time_delta': mean_delta / baseline_time if baseline_time else None,
            'five_touchdown_delta_pp': 100 * (sum(r['recovery_within_5_touchdowns'] for r in aa.values()) -
                                             sum(r['recovery_within_5_touchdowns'] for r in bb.values())) / designated if designated else None}
