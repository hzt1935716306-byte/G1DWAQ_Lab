"""Pinned development assignments and diagnostics; no simulator or policy imports."""
from pathlib import Path

from g1_recovery_protocol import LAB, COMMON_VERSION, TASKS, common_contract, digest, read_json, sha256
from g1_common_task_detector import TimestampWindow, HoldConfirmation, classify_window, EPS

DEVELOPMENT_MANIFEST = LAB / 'tools/evaluation/configs/g1_recovery_eval_v2_development_16.json'
DEVELOPMENT_MANIFEST_SHA256 = '1a302bd6a36e865a18dc878ba8720910f13b81adb5b8d89c8989d941f6525515'
DEVELOPMENT_ID_FIELDS = ('evaluation_role', 'development_manifest_sha256',
                         'development_manifest_schema_version', 'development_assignment_sha256')
TERRAIN_SOURCES = ('legged_lab/terrains/__init__.py', 'legged_lab/terrains/plane_terrain_cfg.py',
                   'legged_lab/recovery/plane_terrain_math.py')
GATE_FAILURES = ('mean_velocity', 'peak_velocity', 'tilt_rms', 'tilt_peak', 'world_z_angular_error', 'root_clearance')
ACCEPTANCE_FIELDS = ('common_complete_windows', 'common_task_gate_pass_windows', 'common_task_gate_pass_rate',
                     'common_task_plus_recent_touchdown_windows', 'common_task_plus_recent_touchdown_rate',
                     'readiness_hold_confirmed', 'first_readiness_confirmation_time', 'common_gate_failure_counts')


def select_development_assignment(protocol, task, method, path=DEVELOPMENT_MANIFEST,
                                  expected_sha=DEVELOPMENT_MANIFEST_SHA256):
    """Use exact pre-registered bytes. A new list requires an explicitly reviewed code change."""
    if protocol['metrics']['version'] != COMMON_VERSION or protocol['protocol_version'] != '2.0-dev':
        raise ValueError('Development validation requires common_task_window_v1 / 2.0-dev')
    if expected_sha != DEVELOPMENT_MANIFEST_SHA256 or sha256(path) != expected_sha:
        raise ValueError('Development manifest SHA256 mismatch with pinned preregistration')
    design = read_json(path)
    _, contract = common_contract(protocol)
    expected = dict(schema_version=1, status='PLANNED_NOT_EXECUTED', protocol_version='2.0-dev',
                    metrics_version=COMMON_VERSION, total_trials=16, normal_development_E0=4,
                    heldout_trials=12, planned_real_pushes=8, planned_sham_trials=4,
                    candidate_parameters_sha256=contract['candidate_parameters_sha256'])
    if any(design.get(k) != v for k, v in expected.items()):
        raise ValueError('Development manifest schema/version/count/candidate mismatch')
    groups = design['assignments']
    if {a['method'] for a in groups} != {'ppo_plain', 'dwaq'} or len(groups) != 2:
        raise ValueError('Development manifest requires exactly PPO and DWAQ')
    for a in groups:
        trials = a['trials']
        if a['task'] != TASKS[a['method']] or len(trials) != 8 or len({t['trial_id'] for t in trials}) != 8:
            raise ValueError('Development task/assignment count mismatch')
        if (sum(t['experiment'] == 'E0' for t in trials) != 2 or
                sum(is_sham(t) for t in trials) != 2 or sum(t['push_magnitude'] > 0 for t in trials) != 4):
            raise ValueError('Development E0/sham/push allocation mismatch')
    selected = next((a for a in groups if a['task'] == task and a['method'] == method), None)
    if selected is None:
        raise ValueError('Checkpoint method/task has no registered development assignment')
    fields = dict(evaluation_role='development_validation', development_manifest_sha256=expected_sha,
                  development_manifest_schema_version=design['schema_version'],
                  development_assignment_sha256=digest(selected))
    return selected['trials'], fields


def validate_development_snapshot(path, protocol, identity, manifest):
    trials, fields = select_development_assignment(protocol, identity['task_name'], identity['method'],
        Path(path) / 'development_manifest_snapshot.json', identity['development_manifest_sha256'])
    if manifest != trials or any(identity.get(k) != v for k, v in fields.items()):
        raise ValueError('Development assignment identity/snapshot mismatch')


def is_sham(plan):
    tagged = plan.get('sham') is True or plan.get('push_type') == 'sham_marker_no_physical_perturbation'
    if tagged and not (plan.get('sham') is True and plan['push_type'] == 'sham_marker_no_physical_perturbation'
                       and plan['experiment'] == 'E1' and plan['push_magnitude'] == 0):
        raise ValueError('Inconsistent sham intervention assignment')
    return tagged


def realized_environment_payload(effective):
    """Hash actual terrain/physics and collision software, separate from requested profile."""
    required = ('actual_physics_hash', 'terrain_mesh_sha256', 'terrain_origins', 'realized_slopes_deg',
                'environment_versions', 'terrain_source_sha256')
    if any(k not in effective for k in required):
        raise ValueError('Missing realized environment provenance')
    versions = effective['environment_versions']
    if any(not isinstance(versions.get(k), str) or not versions[k] or versions[k] == 'UNKNOWN'
           for k in ('IsaacLab', 'IsaacSim', 'USD', 'PhysX', 'Kit', 'trimesh', 'numpy')):
        raise ValueError('Missing realized collision/runtime version')
    if digest(effective['actual_physics']) != effective['actual_physics_hash']:
        raise ValueError('Realized environment actual physics mismatch')
    if set(effective['terrain_source_sha256']) != set(TERRAIN_SOURCES):
        raise ValueError('Missing realized terrain generator source provenance')
    return {k: effective[k] for k in required}


def measure_realized_slopes(mesh, origins, expected_slopes):
    """Fit each actual 64 m tile's four corners, including unoccupied columns."""
    import numpy as np
    points = np.asarray(mesh, dtype=float)
    table = np.asarray(origins, dtype=float)
    if table.shape != (1, len(expected_slopes), 3) or not np.isfinite(points).all():
        raise ValueError('Unexpected realized plane mesh/origin layout')
    measured = []
    for origin, expected in zip(table[0], expected_slopes):
        relative = points-origin
        corners = relative[(np.abs(np.abs(relative[:, 0])-32) < 1e-3) &
                           (np.abs(np.abs(relative[:, 1])-32) < 1e-3) &
                           (np.abs(relative[:, 2]-np.tan(np.deg2rad(expected))*relative[:, 0]) < 2e-3)]
        if len(np.unique(np.round(corners[:, :2], 3), axis=0)) != 4:
            raise ValueError('Actual mesh lacks four verified local plane corners')
        coefficients = np.linalg.lstsq(np.c_[corners[:, :2], np.ones(len(corners))], corners[:, 2], rcond=None)[0]
        angle = float(np.rad2deg(np.arctan(coefficients[0])))
        if abs(angle-expected) > 1e-3 or abs(coefficients[1]) > 1e-5:
            raise ValueError('Realized signed slope mismatch')
        measured.append(angle)
    return measured


class CommonAcceptance:
    """E0 moving readiness envelope statistics, restricted to post-warmup windows."""
    def __init__(self, hold_s):
        self.hold = HoldConfirmation(hold_s)
        self.complete = self.passed = self.with_contacts = 0
        self.failures = dict.fromkeys(GATE_FAILURES, 0)
        self.first_confirmation = None

    def update(self, gate, recent, counts):
        if gate['window_complete']:
            self.complete += 1
            self.passed += int(gate['passed'])
            self.with_contacts += int(gate['passed'] and recent >= 4)
            for key in self.failures:
                self.failures[key] += int(not gate['checks'][key])
        confirmed = self.hold.update(gate['time'], gate['passed'] and recent >= 4, counts)
        if confirmed and self.first_confirmation is None:
            self.first_confirmation = self.hold.confirmation

    def result(self):
        return dict(common_complete_windows=self.complete, common_task_gate_pass_windows=self.passed,
                    common_task_gate_pass_rate=self.passed/self.complete if self.complete else None,
                    common_task_plus_recent_touchdown_windows=self.with_contacts,
                    common_task_plus_recent_touchdown_rate=self.with_contacts/self.complete if self.complete else None,
                    readiness_hold_confirmed=self.first_confirmation is not None,
                    first_readiness_confirmation_time=self.first_confirmation,
                    common_gate_failure_counts=dict(self.failures))


class ShamTaskObservation:
    """Marker-relative task envelope; never creates a recovery detector/event."""
    def __init__(self, time, config):
        self.marker = time
        self.bounds = config['recovery']
        self.window = TimestampWindow(self.bounds['window_s'], config['sampling']['nominal_period_s'], after=time)
        self.hold = HoldConfirmation(self.bounds['hold_s'])
        self.entry = self.confirmation = None
        self.complete = self.failed = 0
        self.last_time = None
        self.terminal = False

    def update(self, measurement):
        self.last_time = measurement.time
        self.terminal = measurement.terminal or measurement.out_of_area
        self.window.append(measurement.time, measurement.signals())
        stats = self.window.statistics()
        gate = (classify_window(stats, self.bounds, time=measurement.time,
                terminal=measurement.terminal, out_of_area=measurement.out_of_area) if stats else
                dict(time=measurement.time, passed=False, window_complete=False, reason='WINDOW_INCOMPLETE'))
        if gate['window_complete']:
            self.complete += 1
            self.failed += int(not gate['passed'])
        if self.hold.update(measurement.time, gate['passed'], {}) and self.confirmation is None:
            self.entry, self.confirmation = self.hold.entry, self.hold.confirmation
        return gate

    def result(self, survived):
        observed = self.last_time is not None and self.last_time >= self.marker+10-EPS
        return dict(sham_task_entry=self.entry, sham_task_confirmation=self.confirmation,
                    sham_continuity=False if self.terminal else bool(survived and self.complete and not self.failed) if observed else None,
                    sham_detection_latency=self.confirmation-self.marker if self.confirmation is not None else None,
                    sham_complete_windows=self.complete, sham_task_gate_failed_windows=self.failed)
