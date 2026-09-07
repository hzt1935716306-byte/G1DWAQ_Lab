# Five-model complete robustness candidate preregistration

Base: `79f811874fad47a40d5b87f3e07ad73aa7922840`, explicitly selected by the user.
The immutable standard index contains five existing 390-trial sealed runs; only
Context-only was supplemented, before any runtime source modification.

## Separate execution identity

`g1_complete_robustness.py` reuses the native environment/inference factory and
unchanged `CommonTaskTrialMachine` / detector. It does not implement a second
actor or estimator. The new adapter handles external physical substeps and the
release clock. Each policy step advances native actor history exactly once.
Configuration, physical initial states, waveforms and runtime commit are bound
before outcomes. Standard and robustness cohorts must never be pooled.

All five models use the same 64 environment slots, seven signed terrain columns,
manifest, native physics profile and actual body masses. Before every trial,
actual root and joint initial states must exactly match the common immutable
reset record. Across models, the realized-environment hash must match exactly.
The seven-column physical scene is identical even for flat-only families;
OOD ±20 deg rows remain a separate analysis appendix.

## Counts and fixed choices

Per model: 8,320 in-domain envelope + 3,328 OOD envelope + 4,480 family trials
= 16,128. Five models: 80,640 new robustness trials. The 390-trial standard
Context-only supplement makes 81,030 new physical trials for this task.

Envelope: slopes −15/−10/0/+10/+15 deg and separate OOD −20/+20 deg,
Δv 0–3 m/s at 0.25 m/s increments, eight onset-heading directions,
16 trials/direction/cell. Reference foot alternates left/right and target phase
is balanced 0.25/0.75 within each direction. Zero Δv is a sham marker,
never a real push or a recovery success.

Family parameters are exactly in `configs/g1_complete_robustness_v2.yaml`:

| Family | Parameters | Trials/model |
|---|---|---:|
| velocity_ood | world XY square bound 1.25, 1.50, 2.00 m/s | 768 |
| force_pulse | equivalent Δv .5/1/1.5 m/s × .05/.2/.5 s | 1152 |
| constant_force | .25/.5/.75/1/1.5 m/s² × 5 s | 640 |
| repeated_impulse | .25/.5/.75 m/s × period .5/1 s; duration 8 s | 768 |
| random_force | vector RMS .25/.5/1 m/s² × tau .1/1 s; duration 10 s | 768 |
| wrench_pulse | CoM / upper_pitch .20 m / lateral_yaw .15 m; Δv 1 m/s, .20 s | 384 |

Family vectors are world XY; force/wrench conditions use eight balanced
azimuths. Repeated impulses include both t=0 and t=8 s (17 or 9 impacts), so
recovery starts at the actual last impact. No overlapping native episode is
cleared or restarted: native notification occurs only while inactive; every
physical impact is still applied. Force families use true external forces,
not fake velocity jumps or fake native push notifications.

OU force uses a stationary 2-D process with per-axis standard deviation
RMS/√2 and exact discrete coefficient exp(−dt/tau), dt=.005 s. Finite waveforms
are not renormalized. Seeds, SHA, requested RMS, full planned realized RMS,
actual applied-prefix RMS, lag-one correlation, force impulse and peak force
are retained. Falling truncates the applied prefix, not the planned waveform.

## Physical wrench and timing

F=M Δv/T or M a, using actual whole-robot mass. Application point is the actual
mass-weighted whole-robot CoM plus the declared arm, on `torso_link`.
`upper_pitch` uses a +world-Z arm. `lateral_yaw` uses a horizontal arm
perpendicular to F with balanced torque sign. Torque about CoM is r×F.
The installed IsaacLab composer transforms global inputs into local link
coordinates; its cached pose is reset every physical substep. Requested force,
composed force and composed torque about the link transform are checked and
archived. No training/native policy/certificate source is changed.

Force is active on [onset,release) at .005 s physical intervals. The .05 s pulse
is exactly ten intervals, not rounded to policy frames. Recovery uses only
strictly post-release measurements. W=.60 s and H=.50 s are unchanged.
The observation endpoint is exactly release+10 s. If that endpoint falls
between 50 Hz frames, a real .005 s physics-substep measurement is captured
at the endpoint without computing observations or advancing actor history.
Only that final interval can be shorter than .02 s; no future sample is
interpolated or renamed. All other detector samples remain native 50 Hz.
Physical-contact confirmation retains its existing frame rule, including this
explicit final measured endpoint; no unobserved touchdown is inferred.

Readiness failures, falls, out-of-area and non-recovery are valid outcomes.
Disturbance-phase survival and task quality are distinct from release-clock
once/sustained recovery. A software/physical-contract error seals an INVALID
attempt and stops the physical execution queue. It is not automatically rerun.

## Statistics and reporting

Primary probability curves use all designated non-sham trials, with readiness
failures retained as non-success. Conditional-on-actual-push curves are also
saved. Boundaries use the first downward crossing from the smallest tested
real magnitude, linearly interpolated between adjacent rates. A later success
island cannot increase the boundary. `<` / `≥` denote left/right censoring;
missing conditional cells cannot be silently bridged. A single successful
trial never defines a limit.

Five prespecified paired comparisons: Context-only vs RL-only; Context+Reward
vs Context-only, RL-only, PPO, and DWAQ. Binary endpoints use exact McNemar.
Time/steps use paired Wilcoxon only for identical trial IDs successful in both
models; paired mean differences and bootstrap 95% CIs plus rank-biserial
correlation are reported. Bootstrap seed 20260908, 2000 replicates. Holm
correction spans all prespecified endpoint comparisons.

Native N/margin are diagnostic only. First post-onset refresh is retained even
if invalid, rather than replaced by a favorable later valid query. Missing
native diagnostics are N/A. Every touchdown retains held diagnostic values
and query age; full refresh trajectories are archived for mechanism analysis.

`g1_complete_robustness.py prepare` creates immutable independent inputs;
`run --model MODEL` executes one complete assigned model; `report` requires
all five complete seals and generates Markdown, DOCX, figures and detailed
machine-readable statistics. All raw trials/traces/events remain archived.
Protocol stays **2.0-dev / candidate_unvalidated**. No threshold tuning,
retraining, removal of failures, or overwriting historical attempts is allowed.

## Execution stop on the first real force-pulse batch

Runtime commit: `c91d3edc3368c45ac13ac8ab4d0f56a264cab797`.
PPO run `f754ef075b7393c0ec093ca8-attempt-0001` stopped with
`Applied world wrench differs from declared force/application point`.
It contains 12,416 non-error terminal records (11,648 envelope and 768
velocity_ood) and 64 EVALUATION_ERROR records for the interrupted force-pulse
batch. The whole attempt is INVALID, without a COMPLETE seal. All raw files
remain present. No subsequent model was started and no trial was rerun.

A direct CPU call to the installed Warp kernel reproduces a coordinate-frame
inconsistency: the global position offset is translated but not rotated to
link coordinates before being crossed with the link-coordinate force. For
link pitch=30 deg, F_world=(300,0,0) N and r_world=(0,0,.2) m, correct world
moment is (0,60,0) N·m; the kernel produces (0,51.9615173,0) N·m. The installed
kernel file SHA256 is
`b38780baabb26b035595e6b6710e26381a18e99ed3bfbbc82887321aee7f82bf`.

The initial test suite covered the independent force formulas, release timing
and replay but did not exercise this installed global-position API with a
rotated link. That integration-test coverage gap is material. Passing CPU
unit tests is not proof of the physical wrench contract.

No runtime repair or physical restart was performed after this stop. A future
repair should express both force and resultant link torque in the same local
frame inside the evaluation adapter, avoiding this global-position path,
and verify against the installed kernel before launching new physical trials.
It must receive a new runtime identity; the preserved INVALID attempt must
not be relabeled or merged with another runtime cohort.

Evidence and human-readable stop reports are in
`experiments/g1_complete_robustness_v2/development/20260907T170643Z/` and
`experiments/g1_complete_robustness_v2/report/EXECUTION_STOP_REPORT.{md,docx}`.
The full five-model robustness comparison has **not** been completed.
