"""Causal CPU/NumPy measurements and temporal contracts for common task recovery.

No policy, teacher, estimator or certificate inputs are accepted here. Thresholds
are supplied explicitly; this module supplies no inferred/default task envelope.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

METRICS_VERSION = 'common_task_window_v1'
DETECTOR_CODE_VERSION = 'common_task_window_v1.0'
EPS = 1e-9
MEASUREMENT_FIELDS = ('time', 'com_velocity', 'root_velocity', 'command', 'root_quaternion_wxyz',
                      'gravity_tilt_rad', 'gravity_tilt_provenance', 'angular_velocity_world_z',
                      'root_clearance_m', 'forces', 'fell', 'timeout', 'out_of_test_area', 'data_valid')


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def gravity_tilt_wxyz(quaternion):
    """Angle between root +Z and world +Z; Isaac scalar-first quaternion."""
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('root_quaternion_wxyz requires four finite components')
    norm = np.linalg.norm(q)
    if not math.isclose(float(norm), 1., abs_tol=1e-4):
        raise ValueError('root_quaternion_wxyz must be normalized')
    q = q / norm
    return float(np.arccos(np.clip(1 - 2 * (q[1]**2 + q[2]**2), -1, 1)))


def vertical_plane_clearance(position, normal, point):
    """Vertical distance, NOT signed normal distance, above the local plane."""
    r, n, p = (np.asarray(x, dtype=float) for x in (position, normal, point))
    if (any(x.shape != (3,) or not np.isfinite(x).all() for x in (r, n, p)) or n[2] <= EPS
            or not math.isclose(float(np.linalg.norm(n)), 1., abs_tol=1e-5)):
        raise ValueError('A finite upward local plane and root position are required')
    plane_z = p[2] - np.dot(n[:2], r[:2] - p[:2]) / n[2]
    return float(r[2] - plane_z)


@dataclass(frozen=True)
class CommonTaskMeasurements:
    """Only physical measurements enter the common judge; diagnostics stay outside."""
    time: float
    com_velocity_xy: np.ndarray
    root_velocity_xy: np.ndarray
    command: np.ndarray
    gravity_tilt_rad: float
    angular_velocity_world_z: float
    root_clearance_m: float
    forces: np.ndarray
    terminal: bool
    out_of_area: bool
    data_valid: bool

    @classmethod
    def from_frame(cls, frame):
        required = ('time', 'com_velocity', 'root_velocity', 'command',
                    'angular_velocity_world_z', 'root_clearance_m', 'forces',
                    'fell', 'timeout', 'out_of_test_area', 'data_valid')
        missing = [k for k in required if k not in frame]
        if missing:
            raise ValueError('Missing common measurement fields: ' + ', '.join(missing))
        if any(not isinstance(frame[k], (bool, np.bool_)) for k in ('fell', 'timeout', 'out_of_test_area', 'data_valid')):
            raise ValueError('Physical validity/terminal flags must be booleans')
        if 'root_quaternion_wxyz' in frame:
            tilt = gravity_tilt_wxyz(frame['root_quaternion_wxyz'])
            if 'gravity_tilt_rad' in frame and not math.isclose(tilt, frame['gravity_tilt_rad'], abs_tol=1e-5):
                raise ValueError('Quaternion/gravity tilt mismatch')
        elif 'gravity_tilt_rad' in frame:
            tilt = float(frame['gravity_tilt_rad'])
            if not frame.get('gravity_tilt_provenance'):
                raise ValueError('Gravity tilt without quaternion requires verifiable provenance')
        else:
            raise ValueError('Missing root quaternion or verifiable gravity tilt')
        com, root = (np.asarray(frame[k], dtype=float) for k in ('com_velocity', 'root_velocity'))
        cmd, forces = (np.asarray(frame[k], dtype=float) for k in ('command', 'forces'))
        if com.shape not in ((2,), (3,)) or root.shape not in ((2,), (3,)) or cmd.shape != (3,) or forces.shape != (2,):
            raise ValueError('Invalid common measurement vector dimensions')
        scalar = [frame['time'], tilt, frame['angular_velocity_world_z'], frame['root_clearance_m']]
        if not all(np.isfinite(x).all() for x in (com, root, cmd, forces, scalar)):
            raise ValueError('Non-finite common measurement')
        if not frame['data_valid'] or frame['timeout']:
            raise ValueError('Invalid common measurement/timeout; cannot substitute zero')
        if not 0 <= tilt <= math.pi or np.any(forces < 0):
            raise ValueError('Invalid gravity tilt/force norm')
        return cls(float(frame['time']), com[:2].copy(), root[:2].copy(), cmd.copy(),
                   tilt, float(frame['angular_velocity_world_z']), float(frame['root_clearance_m']),
                   forces.copy(), bool(frame['fell']), bool(frame['out_of_test_area']), True)

    def signals(self):
        """Nonnegative errors are formed BEFORE integration; no sign cancellation."""
        error = self.com_velocity_xy - self.command[:2]
        root_error = self.root_velocity_xy - self.command[:2]
        return {'velocity_error_mps': float(np.linalg.norm(error)),
                'velocity_error_squared': float(np.dot(error, error)),
                'root_velocity_error_squared': float(np.dot(root_error, root_error)),
                'tilt_rad': self.gravity_tilt_rad,
                'tilt_squared': self.gravity_tilt_rad**2,
                'yaw_rate_error_radps': self.angular_velocity_world_z - self.command[2],
                'yaw_rate_error_squared': (self.angular_velocity_world_z - self.command[2])**2,
                'clearance_m': self.root_clearance_m}


class TimestampWindow:
    """Closed [t-W,t] support; trapezoids of sample error signals over W seconds.

    At 50 Hz, W=.60 needs 31 boundary samples / 30 intervals. A clipped left
    boundary interpolates primitive errors/angles, never signed velocities before
    the norm. RMS then squares the interpolated angle/angular-error endpoints.
    Maxima/minima use endpoints and recorded interior samples. Nothing is
    extrapolated; an unobserved gap exceeding the explicit contract is an error.
    """
    def __init__(self, duration_s, max_sample_gap_s, *, after=None):
        if duration_s <= 0 or max_sample_gap_s <= 0:
            raise ValueError('Positive timestamp window and sample gap required')
        self.duration = float(duration_s)
        self.max_gap = float(max_sample_gap_s)
        self.after = after
        self.samples = deque()

    def append(self, time, signals):
        t = float(time)
        if not math.isfinite(t) or not signals or not all(math.isfinite(float(v)) for v in signals.values()):
            raise ValueError('Finite timestamp/signals required')
        if self.after is not None and t <= self.after + EPS:
            raise ValueError('Recovery window rejects pre-push/boundary samples; real post-push only')
        if self.samples:
            dt = t - self.samples[-1][0]
            if dt <= EPS:
                raise ValueError('Timestamps must strictly increase; reset needs a new detector')
            if dt > self.max_gap + EPS:
                raise ValueError('Measurement gap exceeds max_sample_gap_s')
            if signals.keys() != self.samples[-1][1].keys():
                raise ValueError('Window measurement schema changed')
        self.samples.append((t, dict(signals)))
        while len(self.samples) > 2 and self.samples[1][0] < t - self.duration - EPS:
            self.samples.popleft()

    def statistics(self):
        if len(self.samples) < 2:
            return None
        end = self.samples[-1][0]
        start = end - self.duration
        if self.samples[0][0] > start + EPS:
            return None
        times = np.array([t for t, _ in self.samples])
        inner = times[(times > start + EPS) & (times < end - EPS)]
        knots = np.r_[start, inner, end]
        result = {'start': float(start), 'end': end, 'duration_s': self.duration,
                  'time_intervals': len(knots) - 1}
        for key in self.samples[0][1]:
            values = np.array([s[key] for _, s in self.samples])
            clipped = np.interp(knots, times, values)
            integral = float(np.sum((clipped[:-1] + clipped[1:]) * .5 * np.diff(knots)))
            result[key] = {'mean': integral / self.duration, 'min': float(clipped.min()),
                           'max': float(clipped.max()), 'integral': integral,
                           'rms': float(np.sqrt(np.sum((clipped[:-1]**2 + clipped[1:]**2) * .5 * np.diff(knots)) / self.duration))}
        return result


GATE_FIELDS = ('mean_com_velocity_error_max_mps', 'peak_com_velocity_error_max_mps',
               'tilt_rms_max_deg', 'tilt_peak_max_deg', 'world_z_angular_error_rms_max_radps',
               'root_vertical_clearance_min_m')


def validate_detector_config(config):
    """Explicit schema: no teacher/resource fields, hidden defaults or final status."""
    if set(config) != {'parameter_status', 'sampling', 'readiness', 'recovery'}:
        raise ValueError('Common detector config requires only parameter_status/sampling/readiness/recovery')
    if config['parameter_status'] != 'candidate_unvalidated':
        raise ValueError('common_task_window_v1 remains candidate_unvalidated')
    sampling = config['sampling']
    if sampling != {'nominal_period_s': .02, 'time_tolerance_s': EPS}:
        raise ValueError('Current sampling contract is 50 Hz with explicit 1e-9 s time tolerance')
    for phase, w, h in [('readiness', 1.20, .20), ('recovery', .60, .50)]:
        expected = set(GATE_FIELDS) | {'window_s', 'hold_s'}
        if phase == 'readiness':
            expected |= {'alternating_touchdowns_window_s', 'alternating_touchdowns_min'}
        values = config[phase]
        if set(values) != expected or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError('Invalid/missing common gate parameters: ' + phase)
        if values['window_s'] != w or values['hold_s'] != h:
            raise ValueError('Changing window/hold requires a new detector contract')
        if not values['tilt_rms_max_deg'] <= values['tilt_peak_max_deg'] <= 180:
            raise ValueError('Invalid degree tilt bounds')
    if config['readiness']['alternating_touchdowns_window_s'] != 1.2 or config['readiness']['alternating_touchdowns_min'] != 4:
        raise ValueError('Readiness requires four recent alternating touchdowns in 1.20 s')
    return config


class CommonTaskGate:
    """Six independently configured task conditions over a causal timestamp window."""
    def __init__(self, config, phase, *, push_end=None):
        validate_detector_config(config)
        if phase not in ('readiness', 'recovery'):
            raise ValueError('Unknown common task phase')
        if phase == 'recovery' and push_end is None:
            raise ValueError('Real recovery requires an actual push end')
        self.bounds = config[phase]
        self.window = TimestampWindow(self.bounds['window_s'], config['sampling']['nominal_period_s'], after=push_end)
        self.command = None

    def update(self, measurement):
        m = measurement
        cmd = m.command
        # Command representation tolerance only; task thresholds have NO slack.
        if (abs(cmd[2]) > 1e-7 or np.count_nonzero(np.abs(cmd[:2]) > 1e-7) != 1
                or not math.isclose(float(np.linalg.norm(cmd[:2])), .4, abs_tol=1e-7)):
            raise ValueError('Moving common detector supports only cardinal 0.4 m/s, world-z command zero')
        if self.command is not None and not np.array_equal(cmd, self.command):
            raise ValueError('Fixed command changed during trial')
        self.command = cmd.copy()
        self.window.append(m.time, m.signals())
        s = self.window.statistics()
        if s is None:
            return {'time': m.time, 'passed': False, 'window_complete': False, 'reason': 'WINDOW_INCOMPLETE', 'statistics': None}
        return classify_window(s, self.bounds, time=m.time, terminal=m.terminal, out_of_area=m.out_of_area)


def classify_window(s, b, *, time, terminal=False, out_of_area=False):
    """Shared window classification; missing clearance is explicitly unavailable.

    Partial classification is for archived measurement diagnostics only; it
    cannot confirm readiness or recovery. Runtime supplies every signal.
    """
    values = {'mean_com_velocity_error_mps': s['velocity_error_mps']['mean'],
              'peak_com_velocity_error_mps': s['velocity_error_mps']['max'],
              'tilt_rms_rad': s['tilt_rad']['rms'], 'tilt_peak_rad': s['tilt_rad']['max'],
              'world_z_angular_error_rms_radps': s['yaw_rate_error_radps']['rms'],
              'root_vertical_clearance_min_m': s['clearance_m']['min'] if 'clearance_m' in s else None}
    checks = {'mean_velocity': values['mean_com_velocity_error_mps'] <= b[GATE_FIELDS[0]],
              'peak_velocity': values['peak_com_velocity_error_mps'] <= b[GATE_FIELDS[1]],
              'tilt_rms': values['tilt_rms_rad'] <= math.radians(b[GATE_FIELDS[2]]),
              'tilt_peak': values['tilt_peak_rad'] <= math.radians(b[GATE_FIELDS[3]]),
              'world_z_angular_error': values['world_z_angular_error_rms_radps'] <= b[GATE_FIELDS[4]],
              'root_clearance': values['root_vertical_clearance_min_m'] >= b[GATE_FIELDS[5]] if 'clearance_m' in s else None,
              'no_terminal': not terminal, 'inside_test_area': not out_of_area, 'data_valid': True}
    passed = all(checks.values()) if all(v is not None for v in checks.values()) else None
    return {'time': time, 'passed': passed, 'window_complete': True,
            'reason': 'MISSING_CLEARANCE' if passed is None else None if passed else 'TASK_DOMAIN', 'checks': checks,
            'statistics': values, 'window_start': s['start'], 'window_end': s['end'],
            'time_intervals': s['time_intervals']}

class PhysicalTouchdowns:
    """Independent per-foot contact transitions; alternation is a derived stream.

    Simultaneous landings count twice physically. They have no observable order,
    so cannot manufacture alternating steps or a phase reference. They break
    the recent alternating chain. A repeated single foot also breaks that chain.
    """
    def __init__(self, on_n, off_n, stable_frames, debounce_s):
        if not 0 <= off_n < on_n or type(stable_frames) is not int or stable_frames < 1 or debounce_s < 0:
            raise ValueError('Invalid contact hysteresis/debounce contract')
        self.on, self.off, self.stable_frames, self.debounce = on_n, off_n, stable_frames, debounce_s
        self.reset()

    def reset(self):
        self.contact = None
        self.pending = [0, 0]
        self.airborne_frames = [0, 0]
        self.armed = [False, False]
        self.last_landing = [-math.inf, -math.inf]
        self.last_time = None
        self.anchor = None
        self.events = []
        self.recent_alternating = deque()
        self.completed_intervals = deque(maxlen=64)
        self.count = self.alternating_count = self.repeated = self.simultaneous = self.chatter = 0

    def update(self, time, forces):
        t, f = float(time), np.asarray(forces, dtype=float)
        if f.shape != (2,) or not np.isfinite(f).all() or np.any(f < 0):
            raise ValueError('Two finite nonnegative foot force norms required')
        if self.last_time is not None and t <= self.last_time + EPS:
            raise ValueError('Contact timestamps must increase; call reset at boundary')
        self.last_time = t
        if self.contact is None:
            self.contact = f > self.on
        landed = []
        for foot in range(2):
            if not self.contact[foot] and f[foot] <= self.off:
                self.airborne_frames[foot] += 1
                if self.airborne_frames[foot] >= self.stable_frames:
                    self.armed[foot] = True
            else:
                self.airborne_frames[foot] = 0
            target = bool(f[foot] > (self.off if self.contact[foot] else self.on))
            if target == self.contact[foot]:
                self.chatter += int(self.pending[foot] > 0)
                self.pending[foot] = 0
                continue
            self.pending[foot] += 1
            if self.pending[foot] < self.stable_frames:
                continue
            self.contact[foot] = target
            self.pending[foot] = 0
            if not target:
                self.armed[foot] = True  # stable off transition just confirmed
            elif self.armed[foot]:
                self.armed[foot] = False
                if t - self.last_landing[foot] >= self.debounce - EPS:
                    self.last_landing[foot] = t
                    landed.append(foot)
                else:
                    self.chatter += 1
        physical, alternate = [], None
        simultaneous = len(landed) == 2
        if simultaneous:
            self.simultaneous += 1
            self.anchor = None
            self.recent_alternating.clear()
        for foot in landed:
            self.count += 1
            event = {'event': 'physical_touchdown', 'time': t, 'foot': ('left', 'right')[foot],
                     'physical_count': self.count, 'simultaneous': simultaneous,
                     'simultaneous_group': self.simultaneous if simultaneous else None}
            physical.append(event)
            self.events.append(event)
        if len(landed) == 1:
            foot = ('left', 'right')[landed[0]]
            repeat = self.anchor is not None and self.anchor['foot'] == foot
            if repeat:
                self.repeated += 1
                self.recent_alternating.clear()
                self.events.append({'event': 'repeated_same_foot', 'time': t, 'foot': foot})
            else:
                interval = None
                if self.anchor is not None:
                    interval = {'start': self.anchor['time'], 'end': t, 'start_foot': self.anchor['foot'],
                                'interval_s': t - self.anchor['time']}
                    self.completed_intervals.append(interval)
                self.alternating_count += 1
                alternate = {'event': 'alternating_touchdown', 'time': t, 'foot': foot,
                             'interval_s': interval['interval_s'] if interval else None}
                self.events.append(alternate)
                self.recent_alternating.append(alternate)
            self.anchor = {'time': t, 'foot': foot}
        return physical, alternate

    def recent_count(self, time, window_s):
        # (t-W,t]: a contact exactly at the old left boundary is no longer recent.
        while self.recent_alternating and self.recent_alternating[0]['time'] <= time - window_s + EPS:
            self.recent_alternating.popleft()
        return len(self.recent_alternating)

    def predict_interval(self, reference_foot, time, window_s):
        intervals = [x for x in self.completed_intervals if x['end'] <= time + EPS and x['start'] >= time - window_s - EPS]
        matching = [x for x in intervals if x['start_foot'] == reference_foot]
        if not intervals:
            raise ValueError('No recent completed interval for causal phase prediction')
        selected = (matching or intervals)[-1]
        return {**selected, 'source': 'same_start_foot' if matching else 'latest_completed_fallback'}

    def counts(self):
        return {'physical': self.count, 'alternating': self.alternating_count,
                'repeated_same_foot': self.repeated, 'simultaneous_events': self.simultaneous}


class HoldConfirmation:
    def __init__(self, hold_s):
        if hold_s <= 0:
            raise ValueError('Positive hold time required')
        self.hold = hold_s
        self.entry = None
        self.confirmation = None
        self.entry_counts = None
        self.confirmation_counts = None

    def update(self, time, passed, counts):
        if not passed:
            self.entry = self.confirmation = self.entry_counts = self.confirmation_counts = None
        else:
            if self.entry is None:
                self.entry, self.entry_counts = time, dict(counts)
            if self.confirmation is None and time - self.entry >= self.hold - EPS:
                self.confirmation, self.confirmation_counts = time, dict(counts)
        return self.confirmation is not None


class CommonReadinessDetector:
    """One bounded attempt: readiness, first legal reference, causal phase, push.

    The caller supplies the independently measured task-window predicate. Recent
    alternating events and the task predicate are rechecked through the actual
    push sample. Losing readiness is terminal for this attempt, not a retry.
    """
    def __init__(self, reference_foot, phase, *, window_s=1.20, hold_s=.20,
                 warmup_s=2., deadline_s=6., reference_wait_s=3., required_touchdowns=4):
        if reference_foot not in ('left', 'right') or not 0 <= phase < 1:
            raise ValueError('Legal reference foot and phase required')
        self.foot, self.phase = reference_foot, phase
        self.window, self.warmup, self.deadline = window_s, warmup_s, deadline_s
        self.wait, self.required = reference_wait_s, required_touchdowns
        self.hold = HoldConfirmation(hold_s)
        self.state = 'WARMUP'
        self.ready_time = self.scheduled = None
        self.reason = None
        self.trigger = {}
        self.events = []
        self.last_time = None

    def fail(self, time, reason):
        self.state, self.reason = 'PRECONDITION_FAILED', reason
        self.events.append({'event': 'precondition_failed', 'time': time, 'reason': reason})
        return False

    def update(self, time, task_passed, contacts, alternating_event):
        t = float(time)
        if self.last_time is not None and t <= self.last_time + EPS:
            raise ValueError('Readiness timestamps must increase')
        self.last_time = t
        if self.state in ('PRECONDITION_FAILED', 'OBSERVE'):
            return False
        recent = contacts.recent_count(t, self.window)
        good = bool(task_passed and recent >= self.required)
        if self.state in ('WARMUP', 'READY_CHECK'):
            if t < self.warmup - EPS:
                return False
            self.state = 'READY_CHECK'
            if t > self.deadline + EPS:
                return self.fail(t, 'readiness_deadline_exceeded')
            if self.hold.update(t, good, contacts.counts()):
                self.ready_time = t
                self.state = 'WAIT_REFERENCE_TOUCHDOWN'
                self.events.append({'event': 'readiness_confirmed', 'time': t,
                                    'entry_time': self.hold.entry, 'recent_alternating_touchdowns': recent})
            elif t >= self.deadline - EPS:
                reason = 'insufficient_recent_alternating_touchdowns' if recent < self.required else 'task_window_or_readiness_hold_not_satisfied'
                return self.fail(t, reason)
        if self.state in ('WAIT_REFERENCE_TOUCHDOWN', 'APPLY_PUSH_ONCE'):
            if not good:
                return self.fail(t, 'readiness_lost_before_push')
            if t > self.ready_time + self.wait + EPS:
                return self.fail(t, 'reference_wait_deadline_exceeded')
        if self.state == 'WAIT_REFERENCE_TOUCHDOWN':
            if alternating_event and alternating_event['foot'] == self.foot and alternating_event['interval_s'] is not None:
                prediction = contacts.predict_interval(self.foot, t, self.window)
                self.scheduled = t + self.phase * prediction['interval_s']
                self.trigger = {'reference_touchdown_time': t, 'T_used_for_trigger': prediction['interval_s'],
                                'interval_prediction': prediction, 'scheduled_push_time': self.scheduled}
                self.state = 'APPLY_PUSH_ONCE'
                if self.scheduled > self.ready_time + self.wait + EPS:
                    return self.fail(t, 'predicted_push_exceeds_reference_wait_deadline')
            elif t >= self.ready_time + self.wait - EPS:
                return self.fail(t, 'reference_touchdown_not_found')
        return self.state == 'APPLY_PUSH_ONCE' and t >= self.scheduled - EPS

    def push_applied(self):
        if self.state != 'APPLY_PUSH_ONCE':
            raise ValueError('Push requires currently valid, scheduled readiness')
        self.state = 'OBSERVE'


class CommonRecoveryDetector:
    """Temporal recovery state, fed only real post-push window classifications.

    A failed state persists causally on [sample_time,next_sample_time) for the
    out-of-domain duration. Entry is a window END, never corrected by subtracting W.
    """
    def __init__(self, push_end_time, *, hold_s=.50, deadline_s=8., observation_s=10.):
        self.push_end = float(push_end_time)
        self.deadline, self.observation = deadline_s, observation_s
        self.hold = HoldConfirmation(hold_s)
        self.first_entry = self.first_confirmation = None
        self.first_entry_counts = self.first_confirmation_counts = None
        self.first_post_push_sample_time = None
        self.previous_time = None
        self.previous_passed = False
        self.relapse_count = 0
        self.out_of_domain_duration = 0.
        self.events = []
        self.terminal = False

    def update(self, time, passed, counts, *, terminal=False):
        t = float(time)
        if t <= self.push_end + EPS:
            raise ValueError('Recovery accepts only real post-push timestamps')
        if self.previous_time is not None and t <= self.previous_time + EPS:
            raise ValueError('Recovery timestamps must strictly increase')
        if t > self.push_end + self.observation + EPS:
            raise ValueError('Recovery sample exceeds fixed observation horizon')
        if self.terminal:
            raise ValueError('Recovery cannot resume after terminal/reset')
        if self.first_post_push_sample_time is None:
            self.first_post_push_sample_time = t
        if self.first_confirmation is not None and not self.previous_passed:
            self.out_of_domain_duration += t - self.previous_time
        passed = bool(passed and not terminal)
        if self.first_confirmation is not None and self.previous_passed and not passed:
            self.relapse_count += 1
            self.events.append({'event': 'recovery_relapse', 'time': t})
        confirmed_before = self.hold.confirmation
        confirmed = self.hold.update(t, passed, counts)
        if confirmed and confirmed_before is None:
            self.events.append({'event': 'common_recovery_confirmed', 'time': t,
                                'entry_time': self.hold.entry, 'entry_counts': self.hold.entry_counts,
                                'confirmation_counts': self.hold.confirmation_counts})
            if self.first_confirmation is None:
                self.first_entry, self.first_confirmation = self.hold.entry, t
                self.first_entry_counts = dict(self.hold.entry_counts)
                self.first_confirmation_counts = dict(counts)
        self.previous_time, self.previous_passed, self.terminal = t, passed, bool(terminal)

    def result(self, survived):
        observed = self.previous_time is not None and self.previous_time >= self.push_end + self.observation - EPS
        survived = bool(survived and observed and not self.terminal)
        once = bool(survived and self.first_entry is not None and self.first_entry <= self.push_end + self.deadline + EPS)
        sustained_entry = self.hold.entry if self.hold.confirmation is not None and survived else None
        sustained = bool(sustained_entry is not None and sustained_entry <= self.push_end + self.deadline + EPS)
        counts = self.hold.entry_counts if sustained else None
        return {'first_post_push_sample_time': self.first_post_push_sample_time,
                'first_recovery_entry': self.first_entry, 'first_confirmation': self.first_confirmation,
                'first_entry_counts': self.first_entry_counts, 'first_confirmation_counts': self.first_confirmation_counts,
                'recovered_once_and_survived': once, 'relapse_count': self.relapse_count,
                'out_of_domain_duration_after_confirmation': self.out_of_domain_duration,
                'sustained_recovery_entry': sustained_entry,
                'sustained_confirmation': self.hold.confirmation if sustained_entry is not None else None,
                'sustained_entry_counts': self.hold.entry_counts if sustained_entry is not None else None,
                'sustained_confirmation_counts': self.hold.confirmation_counts if sustained_entry is not None else None,
                'recovered_sustained_and_survived': sustained,
                'recovery_time': sustained_entry - self.push_end if sustained else None,
                'recovery_steps': counts['physical'] if sustained else None,
                'confirmation_steps': self.hold.confirmation_counts['physical'] if sustained else None,
                'recovery_within_3_touchdowns': bool(sustained and counts['physical'] <= 3),
                'recovery_within_5_touchdowns': bool(sustained and counts['physical'] <= 5)}
