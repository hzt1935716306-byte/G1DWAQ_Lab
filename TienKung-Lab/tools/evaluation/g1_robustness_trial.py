"""Causal disturbance/release adapter around the unchanged common task judge."""
from __future__ import annotations
import json
import math
import numpy as np

from g1_common_task_trial import CommonTaskTrialMachine
from g1_common_task_detector import (EPS, MEASUREMENT_FIELDS, CommonTaskMeasurements,
    CommonTaskGate, CommonRecoveryDetector, vertical_plane_clearance)


class RobustnessTrial(CommonTaskTrialMachine):
    def __init__(self, plan, protocol):
        super().__init__(plan, protocol)
        self.onset = None
        self.release_time = None
        self.onset_counts = None
        self.disturbance_gate = None
        self.disturbance_windows = []
        self.intervention = None
        self.physics_samples = []
        self.impact_events = []
        self.first_terminal_physics_time = None

    def start_intervention(self, event):
        if self.onset is not None or self.sham or self.state != 'APPLY_PUSH_ONCE':
            raise ValueError('Intervention requires one current causal readiness trigger')
        if abs(event['time'] - self.t) > EPS or event['duration_s'] != self.plan['duration_s']:
            raise ValueError('Intervention onset/duration differs from plan')
        self.readiness.push_applied()
        self.onset = self.t
        self.onset_counts = self.contact.counts()
        self.release_time = self.onset + self.plan['duration_s']
        self.intervention = dict(event)
        self.trigger.update(actual_push_time=self.onset, push_start_time=self.onset,
            scheduled_disturbance_end_time=self.release_time,
            trigger_timing_error=self.onset-self.trigger['scheduled_push_time'],
            actual_contact_state=self.contact.contact.tolist(), push_heading_yaw=event['heading_yaw'])
        self.events.append({'event': 'disturbance_onset', **event})
        self.disturbance_gate = CommonTaskGate(self.config, 'recovery', push_end=self.onset)
        self.transition('DISTURBANCE', self.t)
        if self.plan['duration_s'] == 0:
            self._release(self.contact.counts())

    def _release(self, counts):
        if self.push_time is not None:
            raise ValueError('Recovery release may occur only once')
        self.push_time = self.release_time
        self.push_counts = dict(counts)
        self.trigger['push_end_time'] = self.release_time
        self.recovery_gate = CommonTaskGate(self.config, 'recovery', push_end=self.release_time)
        self.recovery = CommonRecoveryDetector(self.release_time)
        self.events.append({'event': 'disturbance_release', 'time': self.release_time,
                            'physical_touchdown_counts_at_release': dict(counts)})
        self.transition('OBSERVE', self.release_time)

    def feed(self, frame):
        if self.status:
            return False
        if self.onset is None:
            return super().feed(frame)
        self.t = float(frame['time'])
        f = dict(frame)
        self.frames.append(f)
        m = CommonTaskMeasurements.from_frame({k: f[k] for k in MEASUREMENT_FIELDS if k in f})
        if self.last_measurement_time is not None:
            dt = m.time-self.last_measurement_time
            if dt <= EPS or dt > self.config['sampling']['nominal_period_s'] + EPS:
                raise ValueError('Missing/out-of-order robustness physical sample')
        self.last_measurement_time = m.time
        if not np.allclose(m.command, [self.plan[k] for k in ('command_vx','command_vy','command_yaw')], atol=1e-7, rtol=0):
            raise ValueError('Measured command differs from paired plan')
        clearance = vertical_plane_clearance(f['root_position'], f['local_plane_normal'], f['local_plane_point'])
        if not math.isclose(clearance, m.root_clearance_m, abs_tol=1e-5):
            raise ValueError('Clearance does not match actual local plane')
        before_counts = self.contact.counts()
        physical, alternating = self.contact.update(m.time, m.forces)
        if self.push_time is None and m.time >= self.release_time - EPS:
            # A release between samples excludes events at earlier samples, but
            # includes the first event strictly after release in recovery counts.
            counts = self.contact.counts() if abs(m.time-self.release_time) <= EPS else before_counts
            self._release(counts)
        post = self.push_time is not None and m.time > self.push_time + EPS
        f.update(gravity_tilt_rad=m.gravity_tilt_rad, physical_contact=self.contact.contact.tolist(),
            physical_touchdown_flags=[any(e['foot']==foot for e in physical) for foot in ('left','right')],
            alternating_touchdown_flag=alternating is not None, post_push_sample=post, post_sham_sample=False)
        if self.onset + EPS < m.time <= self.release_time + EPS:
            gate = self.disturbance_gate.update(m)
            self.disturbance_windows.append(gate)
            self.events.append({'event': 'disturbance_task_window', **gate})
        if m.terminal or m.out_of_area:
            if self.recovery is not None:
                self.recovery.update(m.time, False, self.post_push_counts(), terminal=True)
            self.finish('FELL' if m.terminal else 'OUT_OF_TEST_AREA')
            return False
        if post:
            gate = self.recovery_gate.update(m)
            self.events.append({'event': 'common_recovery_window', **gate})
            self.recovery.update(m.time, gate['passed'], self.post_push_counts())
            if m.time >= self.release_time + self.p['e1']['observation_s'] - EPS:
                good = self.recovery.result(True)['recovered_sustained_and_survived']
                self.finish('RECOVERED_AND_SURVIVED' if good else 'ALIVE_NOT_RECOVERED')
        return False

    def add_impact(self, event):
        if self.onset is None:
            raise ValueError('Impact precedes intervention onset')
        self.impact_events.append(dict(event))
        self.events.append({'event': 'velocity_jump', **event})

    def add_physics(self, sample):
        self.physics_samples.append(dict(sample))
        if sample.get('terminal') and self.first_terminal_physics_time is None:
            self.first_terminal_physics_time = sample['time']

    def result(self, include_metrics=True):
        r = super().result(include_metrics=include_metrics)
        started = self.onset is not None
        reached_release = started and self.t >= self.release_time-EPS
        phase_survived = bool(reached_release and
            (self.first_terminal_physics_time is None or self.first_terminal_physics_time > self.release_time+EPS) and
            not (self.status in ('FELL','OUT_OF_TEST_AREA') and self.t <= self.release_time+EPS))
        after = [f for f in self.frames if started and self.onset+EPS < f['time'] <= min(self.t,self.release_time)+EPS]
        def rms(field):
            if len(after)<2:return None
            times=np.array([f['time'] for f in after]);v=np.array([field(f) for f in after]);dt=np.diff(times)
            return float(np.sqrt(np.sum((v[:-1]**2+v[1:]**2)*.5*dt)/(times[-1]-times[0])))
        complete=[g for g in self.disturbance_windows if g['window_complete']]
        times=[g['time'] for g in complete]
        weights=np.diff(times) if len(times)>1 else np.array([])
        occupancy=float(np.dot(weights,[g['passed'] for g in complete[:-1]])/weights.sum()) if weights.size and weights.sum()>0 else None
        physical=self.physics_samples
        active=[s for s in physical if s.get('force_active')]
        impulse=np.sum([np.asarray(s['force_world_n'])*s['dt_s'] for s in active],axis=0).tolist() if active else [0.,0.,0.]
        force_rms=float(np.sqrt(np.mean([np.dot(s['force_world_n'],s['force_world_n']) for s in active]))) if active else None
        mass=self.intervention.get('mass_kg') if started else None
        impulses_survived=sum(any(f['time']>e['time']+EPS and not f['fell'] and not f['out_of_test_area'] for f in self.frames if f['time']<=e['time']+.02+EPS) for e in self.impact_events)
        r.update(evaluation_role=self.plan['evaluation_role'], disturbance_start_time=self.onset,
            observed_terminal_time=self.t,
            planned_disturbance_end_time=self.release_time,
            actual_disturbance_stop_time=min(self.t,self.release_time,self.first_terminal_physics_time if self.first_terminal_physics_time is not None else float("inf")) if started else None,
            recovery_clock_start_time=self.push_time, push_applied=started,
            recovery_applicable=started, post_disturbance_observation_started=bool(self.recovery is not None),
            intervention_marker_applied=started or r['sham_applied'],
            disturbance_phase_survived=phase_survived if started else None,
            disturbance_task_domain_occupancy=occupancy,
            disturbance_complete_window_duration_s=float(weights.sum()) if weights.size else 0.,
            disturbance_com_velocity_rmse_mps=rms(lambda f: np.linalg.norm(np.asarray(f['com_velocity'])[:2]-np.asarray(f['command'])[:2])),
            disturbance_tilt_rms_rad=rms(lambda f:f['gravity_tilt_rad']),
            disturbance_physical_touchdowns=(self.push_counts or self.contact.counts())['physical']-self.onset_counts['physical'] if started else None,
            first_terminal_physics_time=self.first_terminal_physics_time,
            mass_kg=mass, force_impulse_world_ns=impulse, force_rms_n=force_rms,
            actual_rms_acceleration_mps2=force_rms/mass if force_rms is not None else None,
            peak_force_n=max((float(np.linalg.norm(s['force_world_n'])) for s in active),default=None),
            force_applied_duration_s=sum(s['dt_s'] for s in active),
            actual_start_time=active[0]['start_time'] if active else None,
            actual_end_time=active[-1]['time'] if active else None,
            actual_duration=sum(s['dt_s'] for s in active),
            physics_substeps_applied=len(active),
            release_zero_wrench_verified=any(abs(s['start_time']-(self.release_time or 0))<=EPS
                and not np.any(s['composed_force_world_n']) and not np.any(s['composed_torque_about_link_world_nm'])
                for s in physical),
            actual_force_lag_one_correlation=(float(np.corrcoef(np.asarray([s['force_world_n'][:2] for s in active])[:-1].ravel(),
                np.asarray([s['force_world_n'][:2] for s in active])[1:].ravel())[0,1])
                if self.plan['family']=='random_force' and len(active)>2 and np.std([s['force_world_n'][:2] for s in active])>0 else None),
            impulses_applied=len(self.impact_events), impulses_survived_first_post_sample=impulses_survived,
            intervention=self.intervention,
            failure_category=('RECOVERED_SUSTAINED' if r['recovered_sustained_and_survived'] else
                'RECOVERED_ONCE_BUT_RELAPSED' if r['recovered_once_and_survived'] else self.status))
        return r

    def trace(self):
        t=super().trace()
        t['physics_json']=np.asarray([json.dumps(s,sort_keys=True,separators=(',',':')) for s in self.physics_samples],dtype=str)
        return t


def replay(plan, protocol, trace, source_record, events):
    """Same detector and state machine; recorded physical application is verified separately."""
    machine=RobustnessTrial(plan,protocol)
    onset=next((e for e in events if e['event']=='disturbance_onset'),None)
    impacts=[e for e in events if e['event']=='velocity_jump']
    physics=[json.loads(str(s)) for s in trace['physics_json']]
    pi=0
    for index,t in enumerate(trace['time']):
        while pi<len(physics) and physics[pi]['time']<=t+EPS:
            machine.add_physics(physics[pi]);pi+=1
        frame={k:trace[k][index].tolist() for k in trace if k not in ('plane_json','physics_json')}
        frame['plane']=json.loads(str(trace['plane_json'][index]))
        due=machine.feed(frame)
        real=onset is not None and abs(t-onset['time'])<=EPS
        sham=source_record.get('sham_applied') and abs(t-source_record['sham_marker_time'])<=EPS
        if bool(due)!=bool(real or sham):raise ValueError('Recorded intervention disagrees with causal readiness')
        if real:machine.start_intervention({k:v for k,v in onset.items() if k not in ('event','trial_id')})
        if sham:machine.apply_sham_marker(source_record['sham_velocity_before'],source_record['sham_velocity_after'])
    machine.impact_events=[dict(e) for e in impacts]
    machine.events.extend(impacts)
    if not machine.status:raise ValueError('Trace ends before a legitimate trial terminal state')
    return machine
