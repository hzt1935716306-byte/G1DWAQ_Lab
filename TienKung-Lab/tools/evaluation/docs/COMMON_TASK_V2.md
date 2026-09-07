# Common Task Recovery Detector — 2.0-dev

Status: **candidate_unvalidated**. These are predeclared development task bounds,
not teacher P95, theoretical stability limits, validated hardware safety limits,
or values selected to maximize PPO acceptance. Software tests are not physical
positive/negative validation. `2.0` is deliberately rejected by `prepare`.

## Scope and resource separation

Same G1 root/pelvis, current-yaw heading frame, cardinal ±X/±Y commands of 0.4 m/s,
world-Z angular command zero, Lite planes −10/0/+10 degrees, nominal sampling
50 Hz. Standing supports E0 normal observation only. No running, other speeds,
stairs, mixed planes, or standing recovery is claimed.

| Dependency | Legacy `practical_interval_confirm2_v1` | `common_task_window_v1` |
|---|---|---|
| GT whole-body CoM | Mass-weighted body CoM velocity, current H | Same GT computation |
| Common pose target | Teacher roll_star/pitch_star | Gravity angle of root/pelvis +Z |
| Common thresholds | Teacher nominal interval P95 | Independent explicit detector config |
| Contact stream | Hysteresis + global dead-time, repeats excluded | Per-foot physical events; separate derived alternating stream |
| Recovery confirmation | Two good complete touchdown intervals | Complete timestamp W plus H hold; once and sustained separate |
| Teacher metrics snapshot | Required and SHA checked | Absent; metrics_reference_sha256=null |
| Common config snapshot | N/A | common_detector_config.yaml, exact bytes and parameter hash checked |
| Native Plane nominal/capability/Estimator/solver | SHA-bound native snapshots | Unchanged, still required by native policy |
| Policy/method/reward/N/margin/certificate valid | Some available as diagnostics | Never inputs to common measurements or gate |

`g1_common_task_detector.py` is CPU/NumPy only. `CommonTaskTrialMachine` adapts
the existing evaluator and inherits its storage/event interfaces; it does not
copy the simulator, actor inference, history, reset or push implementation.
The adapter filters physical fields before constructing typed measurements.
Native Plane diagnostics stay in `plane_json` and continuous diagnostic records.

The original native resource binding functions remain unchanged:
`bind_native_inputs`, `assert_native_inputs`, `verify_native_snapshot`, and the
SHA checks in `RunStore`. In particular, a new common judge does NOT disable
Estimator/certificate/context or substitute the common configuration for native
nominal/capability files.

## Formula / field mapping

W is world +Z-up; B is the common root/pelvis rigid-body frame; H has current yaw
only, and shares world vertical. Quaternion is scalar-first **wxyz**, B→W.

| Definition | Runtime data / recorded field | Reduction / gate |
|---|---|---|
| v_C,W = Σm_i v_Ci,W / Σm_i; v_C,H = R_WHᵀ v_C,W | Actual PhysX masses and `body_com_vel_w[..., :3]`; `com_velocity` | XY only; no estimator or uncorrected link-origin velocity |
| e_v = ‖v_C,H,xy − v_cmd,xy‖₂ | `com_velocity`, measured `command` | Time mean E_v and sampled V_peak; norm before averaging |
| Root H velocity | `root_lin_vel_w` rotated by same yaw; `root_velocity` | Continuous diagnostic, not replacement CoM |
| θ = acos(clip(e_zᵀ R_WB e_z,−1,1)) | `root_quaternion_wxyz`, derived `gravity_tilt_rad` | RMS and peak of unsigned gravity tilt; no teacher/slope subtraction |
| e_ω = ω_root,W,z − ω_cmd,W,z | `root_vel_w[:,5]`, `angular_velocity_world_z`, `command[2]` | RMS, rad/s; not body-Z rate or Euler yaw derivative |
| z_plane=z0−(nx(x−x0)+ny(y−y0))/nz | `local_plane_normal`, `local_plane_point`, `root_position` in same env-relative frame | Unit upward normal, nz>0 |
| h_root=z_root−z_plane | `root_clearance_m` | Window minimum ≥0.50 m; vertical distance, not normal distance/CoM h_eff |
| Physical foot contact | Foot net-force norms, `forces` | `physical_contact`, physical/alternating event flags and JSONL events |
| Terminal/integrity | `fell`, `timeout`, `out_of_test_area`, `data_valid` | Existing public terminal contract; missing/nonfinite data errors |
| Disturbance timing | `velocity_before/after`, `push_start_time`, `push_end_time`, `actual_push_time` | Increment validated; jump start=end |
| First coherent post sample | `first_post_push_sample_time`, trace `post_push_sample` | Next true physical sample, never the push-before frame |
| Heading/path | Initial yaw, yaw drift/offset, initial XY, displacement and path deviation | Diagnostic only; not a main gate |

Root/pelvis provenance: G1_CFG points to `assets/unitree/g1/g1.usd`; the source
URDF root is pelvis, with the upright G1 convention and default root height
0.80 m. Runtime asserts pelvis is the root body and USD world up-axis is +Z;
the actual asset/config hash is part of the existing controlled physics contract.
The local plane is checked against actual USD tile vertices, not a teacher plane
validity flag. Additional reads access data tensors only and do not call
`get_observations`, `compute_observations`, scene stepping, or actor history.
CPU tests exercise the actual snapshot method and verify history count unchanged.
The binary USD and physics binding still require the next authorized simulator
contract check; this delivery did not launch Isaac to claim that check completed.

Quaternion norm must be within 1e-4 of one (then normalized), finite, and shape
four. Optional stored tilt must agree within 1e-5 rad. Plane normals have 1e-5
unit-norm tolerance and recorded clearance agrees within 1e-5 m. These are
measurement consistency checks, NOT additions to task threshold bounds.

## Complete candidate parameter table

| Field / unit | Readiness | Recovery |
|---|---:|---:|
| window_s | 1.20 | 0.60 |
| hold_s | 0.20 | 0.50 |
| mean_com_velocity_error_max_mps | 0.16 | 0.10 |
| peak_com_velocity_error_max_mps | 0.50 | 0.30 |
| tilt_rms_max_deg | 10.0 | 5.0 |
| tilt_peak_max_deg | 20.0 | 15.0 |
| world_z_angular_error_rms_max_radps | 0.40 | 0.20 |
| root_vertical_clearance_min_m | 0.50 | 0.50 |
| alternating_touchdowns_window_s | 1.20 | Not used |
| alternating_touchdowns_min | 4 | Not used |

Angles are configured in degrees and converted once for comparisons to radians:
5°=0.08726646, 10°=0.17453293, 15°=0.26179939, 20°=0.34906585 rad.
World-Z angular rates are explicitly rad/s throughout (not angle fields).

Readiness warmup=2 s, deadline=6 s, maximum reference wait=3 s.
Recovery entry deadline=8 s after actual push end; observation horizon=10 s.
Contact on/off=5/3 N; stable confirmation=2 samples; per-foot debounce=0.08 s.
Existing torso-contact termination threshold=1 N and test area half extent=24 m
remain unchanged. Violating posture/height gate is not automatically a fall.

All six inequalities are inclusive (`<=` upper, `>=` lower), with **no physical
threshold epsilon**. Timestamp tolerance is separately declared as 1e-9 s.
Command representation uses 1e-7 numerical tolerance for float32 0.4 m/s; that
is not a velocity-error gate tolerance. Thresholds are never adapted per method.

## Timestamp, contact and state contracts

Window support is closed `[t-W,t]`. W=.60 has 30 intervals/31 endpoints at 50 Hz;
W=1.20 has 60 intervals/61 endpoints. Incomplete W returns a normal false gate.
At a clipped boundary interpolate e_v, θ, e_ω and h_root between existing valid
neighbors. Integrate e_v with trapezoids; square the interpolated θ/e_ω values
and then integrate/open square-root for RMS. Max/min include clipped endpoints.
They are sampled extrema, not proof of absent sub-sample spikes.

Runtime timestamps strictly increase. This 50 Hz contract rejects an interval
greater than .02 s plus the time tolerance; it cannot bridge a missing physical
sample or skip NaN/Inf. The reusable timestamp window supports an explicitly
chosen gap contract for isolated CPU tests; production always uses .02 s.
No extrapolation or future input is used. Holds apply at continuous valid sample
points and require actual timestamp span H; one false sample resets the hold.

Warmup physics and public contact detection run normally, but **t<2.0 s never
enters the readiness task window**. The first task sample is at t=2.0 s;
earliest complete 1.20 s window is approximately 3.20 s and earliest 0.20 s hold
confirmation is approximately 3.40 s. The deadline remains 6 s. The
`readiness_window_started` event records the start and warmup exclusion.
Readiness uses the recent alternating stream in `(t-1.20,t]`, not total events
since reset; the first complete window therefore cannot use warmup contacts.
After H confirmation, choose the first legal target-foot event.
Predict phase with the newest recent completed interval **starting at that same
foot**; only if unavailable, use the latest completed interval and record the
fallback. The estimated interval and its start/end/source are recorded. Neither
the ongoing interval's future end nor any future recovery data is used.
Waiting and the actual push sample continuously recheck ReadyCandidate. A loss
records `READINESS_LOST_BEFORE_PUSH`, ends that trial, and does not retry or
extend the single wait deadline. The wait deadline also bounds scheduled push.

Each physical foot has separate hysteresis, stable-frame confirmation and
debounce. It must be confirmed airborne before a new landing; initial stance
is not a touchdown. A genuine repeat of the same foot counts physically.
Simultaneous new contacts count two physical events with a shared simultaneous
group. There is no observed ordering, so simultaneous events do not fabricate
alternation or a phase reference. Repeats/simultaneous contacts break the recent
alternating chain, but never reject recovery by themselves. These are solely
common-judge events; Plane native touchdown code is unchanged.

The recovery gate starts with an empty buffer after the physical jump. At 50 Hz,
the normal first post sample is t_e+.02, earliest full-window entry t_e+.62,
earliest confirmation t_e+1.12. Times are not corrected by subtracting W.
Physical recovery step count is exactly `count(t_e < event_time <= entry_time)`;
confirmation counts and alternating/repeated/simultaneous counts are separate.

First entry becomes a first-recovery event only when that segment achieves H.
Once-success requires that entry ≤t_e+8, confirmation before the observation end,
and survival through t_e+10. Sustained-success additionally requires the last
continuous good segment to extend through the final sample, have entry ≤8 and
H confirmation. Relapse preserves the first event and starts a new candidate.
Domain failure duration after first confirmation holds each sampled gate state
causally on `[sample_time,next_sample_time)`. A fall or area exit prevents success.
Main recovery_time/steps are null if sustained-success fails; never impute 6/8/10.

## Version, storage and report integrity

V1 snapshot/node derivation, trace schema and historical metrics remain supported.
V2 `prepare` explicitly validates common schema/scope/timing and refuses the old
`experiments/g1_recovery_eval` tree. It writes an independent common config,
v2 manifest filenames, and no teacher metrics reference or metrics_nodes cache.
`load_prepared` re-derives and verifies the v2 config; checks were not disabled.
`RunStore` checks mandatory v2 fields and replays the SAME online machine against
stored physics and actual trigger to verify outcomes/times/counts/sample roles.
Synthetic runs remain prohibited from the real registry.

Identity distinguishes detector code version, runtime allowlist hash (including
common detector and trial adapter), candidate parameter hash, common snapshot
byte hash, validation evidence hash, protocol version, and native policy input
resource/config hashes. Metrics config/protocol hashes already participate in
compatibility/evaluation identity. V1/V2 cannot enter the same summary, paired
ranking, or common main table. A combined report labels separate sections
“不直接比较”. All v2 results remain candidate development results, including any
future full-sized manifest; 390 planned trials in prepare are NOT executed trials.

Reports retain designated, readiness-confirmed, actually pushed, survival,
once/sustained recovery, ≤3/≤5 physical landing sustained recovery, entries,
confirmation, relapse, and continuous velocity/pose/heading/path metrics.
They include both designated and pushed cumulative sustained-success curves.
With zero pushes, conditional rates are N/A. E0 ends as `E0_COMPLETED` / normal
observation, not a recovery failure. No-push trials never get real recovery events.

## Offline use and evidence limits

```bash
python -B tools/evaluation/g1_recovery_eval.py prepare \
  --protocol tools/evaluation/configs/g1_recovery_eval_lite_v2.yaml \
  --output_root experiments/g1_recovery_eval_v2
python -B tools/evaluation/g1_recovery_eval.py diagnose-common \
  --source experiments/g1_recovery_eval/runs/50471a7a5a61afd8d7f6cf93-attempt-0001 \
  --output_root experiments/g1_recovery_eval_v2
python -B tools/evaluation/g1_recovery_eval.py report \
  --output_root experiments/g1_recovery_eval_v2 --format md docx
python -B -m pytest -q -p no:cacheprovider tools/evaluation/tests
```

The four sealed no-push traces have no independently archived actual local
plane/vertical clearance. Do not reconstruct it from the commanded slope as if
measured. Root gravity angle is verifiable from archived XYZ Euler as
acos(cos(roll)cos(pitch)); angular velocity is from recorded world-Z component;
fixed command is sourced explicitly from the sealed manifest. No missing value
is filled with zero. Full gate acceptance is **N/A**. Five available task
constraints can be reported separately; no recovery state machine/push is invented.
Diagnostics go to an immutable content-addressed directory under the NEW root,
and source file hashes are checked unchanged.

Initial offline result, with complete window start≥2 s: E0 available-constraint
acceptance is 882/882 readiness-envelope windows and 942/942 recovery-envelope
windows; unpushed E1 is 282/282 and 342/342. Recent four alternating touchdown
condition also holds for all those readiness windows. These correlated endpoints
are not independent trials, full gate validation, or observed recovery outcomes.

The fixed **design manifest** `configs/g1_recovery_eval_v2_development_16.json`
predeclares 4 normal-development E0 and 12 held-out sham/.5/1.0 m/s observations
across PPO and DWAQ at −10°/+X/.4, phase .25, balanced reference feet. Paired
baselines and amplitudes share reset assignments; E0-development and holdout
reset seeds are disjoint. The file is planned, not executed. The dedicated
`run-development` entry now selects each baseline's exact eight assignments;
the ordinary full/prefix CLI continues to use its independent formal manifest.
The original design file bytes (including its historical design-only prose)
are preserved under the pinned SHA. See [development execution contract](DEVELOPMENT_VALIDATION_V2.md)
for the executable entry, sham semantics, environment identity, E0 fields and reports.
Software/physics contract failure stops execution. Valid precondition failures,
non-recovery and falls are saved, then the fixed list continues without tuning.
There are only eight planned real pushes: insufficient for ≥20 reviewed pushes,
negative-class/relapse coverage or any full-protocol freeze. No such simulations
or training were run in this implementation delivery.

## Initial implementation verification (commit 140681c, 2026-09-07)

The full CPU suite passed **145 tests** in 88.85 s with the command above,
retaining the previous 117 tests. Coverage includes timestamp endpoint/interpolation
and missing-sample behavior, primitive error reductions and exact bounds,
post-push separation, readiness expiry, physical contact semantics, recovery
entry/confirmation/counts/relapse/fall, identical synthetic online/offline replay,
common-only prepare and snapshot tampering, unchanged native snapshot binding,
sealed-v1 read compatibility, absent measurements, no-push results, report
denominators/Word XML, and fixed development-design assignments.

Compared with the pre-edit inventory: all **68 historical files** and all
**133 native Python source files** have unchanged SHA256 and unchanged file sets.
`bind_native_inputs` and `assert_native_inputs` have identical ASTs to audited
commit `d72f03fbb58924384f86131827319d41c792cd0a`. Git changes are confined to
`tools/evaluation/`. No native policy/certificate/teacher source was edited.

New-root prepare, Markdown/Word generation and candidate validation inventory
completed with **zero simulation runs**. The 390 assignments were prepared only.
The latest offline diagnostic is
`experiments/g1_recovery_eval_v2/development/offline_acceptance/d444177c37ee74b264a58697/`.
The inventory command `detector-validation --output_root experiments/g1_recovery_eval_v2`
reports `CANDIDATE_UNVALIDATED` and full gate acceptance N/A; it does not invoke
the legacy practical-interval validation logic.
