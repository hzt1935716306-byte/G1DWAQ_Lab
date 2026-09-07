# Common v2 Development Validation — execution contract

Status: **CANDIDATE_UNVALIDATED**, protocol **2.0-dev**. No thresholds, W/H,
detector mathematics, policy, native history, training or certificate changes.
This delivery implements the harness and CPU regression tests; it does **not**
execute the 16 observations, a new Isaac simulation, or the 390-trial suite.

## Exact preregistration and CLI

The unchanged `configs/g1_recovery_eval_v2_development_16.json` is pinned to:

```
1a302bd6a36e865a18dc878ba8720910f13b81adb5b8d89c8989d941f6525515
```

The loader rejects any changed byte, including whitespace, even if the caller
supplies a newly calculated SHA. It also verifies schema_version=1,
status=PLANNED_NOT_EXECUTED, candidate parameter hash, protocol/metrics versions,
16 total observations, 4 E0, 12 heldout, 4 sham and 8 real pushes. Changing this
pre-registration requires separate review, not a command-line override.
The original design-only prose is retained to preserve those exact bytes.

The **future, separately authorized** invocation is:

```bash
python -B tools/evaluation/g1_recovery_eval.py run-development \
  --task g1_slope_nosys_d_matched --checkpoint <PPO-checkpoint> \
  --model_alias PPO-dev --checkpoint_stage final --headless \
  --output_root experiments/g1_recovery_eval_v2
python -B tools/evaluation/g1_recovery_eval.py run-development \
  --task g1_dwaq_slope_nosys_d_matched --checkpoint <DWAQ-checkpoint> \
  --model_alias DWAQ-dev --checkpoint_stage final --headless \
  --output_root experiments/g1_recovery_eval_v2
```

These are documentation examples, **not commands executed in this delivery**.
Native checkpoint/task identity checks and optional SHA-bound identity_manifest
handling remain unchanged. Each checkpoint method/task must match its exact
assignment. PPO selects eight and DWAQ selects eight; no state is regenerated.
The loader returns the literal reset seed, initial pose/velocity/joint scales,
command, slope, phase, reference foot and disturbance fields. The existing
evaluation reset adapter consumes them with its original joint-limit contract.
Default num_envs=1; trial_limit is rejected. Actual physics/software errors stop
the run. Valid readiness failure, non-recovery or fall is saved and the next
predeclared assignment proceeds. Resume and explicit new attempts retain the
existing sealed-run rules.

The development manifest is snapshotted independently of the formal manifest.
The following fields participate in evaluation_key:

- evaluation_role=development_validation
- development_manifest_sha256
- development_manifest_schema_version
- development_assignment_sha256 (hash of the selected method/task/trials object)

The run's manifest_hash is the selected eight literal trials. Both baselines
share the same trial assignments, while assignment object hashes differ by
method/task. RunStore reopens the pinned design snapshot and reselects the exact
assignment. The existing prepared 390-trial files are unchanged.

## Sham is a marker, not a real push

Sham uses the same strict readiness, first eligible reference foot and causal
phase schedule as a real push. At eligibility, apply_trial_intervention branches
to apply_sham_marker before any velocity write or native push bookkeeping hook.
Actual before/after root velocities are recorded and must be exactly equal.
It emits a `sham_marker` event and never a `velocity_jump` event.

- intervention_marker_applied=true
- sham_applied=true; sham_marker_time is the actual marker timestamp
- push_applied=false; recovery_applicable=false
- main recovery_time/recovery_steps=null; real recovery successes=false

The first true post-marker sample starts a separate timestamp window using the
unchanged recovery envelope W=.60/H=.50. There is no CommonRecoveryDetector in
this branch. Outputs are sham_task_entry, sham_task_confirmation,
sham_detection_latency (first confirmation minus marker), sham_continuity,
sham_complete_windows and sham_task_gate_failed_windows. At 50 Hz the earliest
entry is marker+.62 s and earliest confirmation/latency marker+1.12 s / 1.12 s.
Continuity means every complete post-marker task window passed and the robot
survived the entire 10 s observation. An incomplete startup window is not a
domain failure. A fall/area exit is a valid continuity failure; an unreached
marker has continuity=N/A. Missing/corrupt measurements remain execution errors.

Sham designated and marker reached have their own eligibility denominator.
Continuity is conditional on marker reached, and latency quantiles use observed
confirmations. Shams are excluded from **all** real-push designated, pushed,
survival, recovery, 3/5-landing and recovery-curve denominators, even in an empty
or partial run summary. Total trial execution counts still include shams.
RunStore replays marker scheduling, before/after velocities, post-marker sample
roles, outcomes, diagnostic fields and the full sham event stream. Forged marker
times, continuity/latency, or physical jump events are rejected. This proves the
software write/event contract; it does not infer a simulator write from ordinary
between-step acceleration in a trace.

## Realized environment identity

actual_physics_hash is retained. realized_environment_hash is the canonical
SHA256 of these effective environment fields:

| Field | Provenance |
|---|---|
| actual_physics_hash | Actual asset, masses/inertias, material, drives and simulation/control dt |
| terrain_mesh_sha256 | Actual transformed USD terrain vertices read by the existing adapter |
| terrain_origins | Actual TerrainImporter tile-origin table |
| realized_slopes_deg | Least-squares slope of each actual tile's four verified corners, including unoccupied columns |
| environment_versions | IsaacLab, IsaacSim, USD, enabled PhysX extension, Kit build, trimesh and NumPy |
| terrain_source_sha256 | Explicit repository terrain generator source hashes |

The effective YAML and identity.json both store realized_environment_hash.
RunStore recomputes it, verifies actual_physics, software identity and terrain
source hashes against the archived evaluation_runtime_sources. Required engine
versions or source provenance cannot be omitted. Resume compares the entire
effective configuration before continuing.

The explicit runtime allowlist adds `legged_lab/terrains/__init__.py`,
`legged_lab/terrains/plane_terrain_cfg.py`, and
`legged_lab/recovery/plane_terrain_math.py`. This covers
make_plane_recovery_terrain_cfg, x_sloped_plane_terrain and
x_curriculum_sloped_plane_terrain without recursively hashing the project.
Those source files are hashed, **not edited**.

Strict comparison requires nonmissing, identical realized_environment_hash as
well as existing protocol/manifest/runtime/actual-physics compatibility. Retests
with different realized environments are retained separately. Historical V1
sealed records remain readable; missing realized provenance cannot support a
new strict matched comparison or formal main table. Detector evidence inventories
record the realized hash; presence of a hash is not acceptance validation.

## Strict warmup and first-class E0 acceptance

At t<2.0 s the policy and common contact detector run, but no data enters the
readiness CommonTaskGate window. Accumulation begins at t=2.0 s. Earliest full
window is approximately t=3.20 s and H confirmation approximately t=3.40 s.
Recent contacts use (t−1.20,t], so warmup contacts cannot satisfy the first full
window. Readiness deadline remains 6 s, subsequent bounded reference wait 3 s.
`readiness_window_started` records time, warmup_end_s and warmup_samples_excluded.

Moving E0 records and development summaries expose:

- common_complete_windows
- common_task_gate_pass_windows / common_task_gate_pass_rate
- common_task_plus_recent_touchdown_windows / common_task_plus_recent_touchdown_rate
- readiness_hold_confirmed / first_readiness_confirmation_time
- common_gate_failure_counts: mean_velocity, peak_velocity, tilt_rms, tilt_peak,
  world_z_angular_error, root_clearance

Only complete post-warmup windows enter acceptance denominators. The two rates
are descriptive window fractions, not independent-trial probabilities. Trial
hold_confirmed is boolean; summary hold_confirmed counts E0 trials, and first
confirmation times are summarized as quantiles. Standing E0 sets all walking
acceptance fields to null/N/A and never requires touchdowns.

Torso orientation is **NOT_IMPLEMENTED** in this delivery: no independently
verified torso rigid-body frame binding was established without the prohibited
simulator run. torso_gravity_tilt_rad=null and torso_orientation_status explicitly
records that limit; no body index is guessed and nothing enters the task gate.

## Independent development report

`Common v2 Development Validation` shows each method's E0 normal acceptance,
sham marker eligibility/continuity/latency, and .5/1.0 m/s readiness, actual push,
survival, once/sustained recovery, relapse, time and physical landing counts.
Heldout seed/condition pairs are shown only with strict compatible environments;
mismatches are explicitly marked as not strictly pairable. The chapter always
states CANDIDATE_UNVALIDATED. No winner, formal ranking or formal cumulative
curve is generated from this role. The same separate chapter is available when
historical protocols are displayed together, labeled not directly comparable.

The predeclared 16 observations remain unexecuted. They contain only eight real
pushes and cannot establish broad protocol validation or justify a 2.0 freeze.

## CPU verification for this delivery

```
python -B -m pytest -q -p no:cacheprovider tools/evaluation/tests
170 passed in 95.20s
```

The previous 145 tests remain; 25 development regression cases cover exact
assignment selection/byte tampering, CLI dispatch, identity participation, sham
no-write and matched scheduling, online/offline marker replay, separate empty
and populated denominators, strict warmup endpoints, E0 statistics/N/A,
environment hash/strict pairing and RunStore tampering. Existing report fixtures
now explicitly supply synthetic realized-environment identity for strict matching.

Relative to audited commit `140681c0f66f3016afda46be47b44bc96b8d28b5`, the core
detector, candidate YAML and pre-registered JSON are byte-identical. All 68 legacy
history files, 23 pre-existing v2 files and 133 native Python files were checked
byte-identical. Native resource binding, physical_snapshot, begin_batch, reset
and compute_observations have unchanged ASTs. No simulator run was created.
