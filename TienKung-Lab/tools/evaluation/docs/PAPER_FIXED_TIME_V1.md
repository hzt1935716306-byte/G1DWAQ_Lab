# G1 three-method fixed-time recovery experiment

Independent candidate protocol `paper_fixed_time_confirmation_v1_candidate`.
This does not revise or pool with the archived 0.4m/s Common Task benchmark.

## Models

PPO is `g1_slope_nosys_d_matched`, traced to the PPO row in the sealed comparison.
DWAQ is `g1_dwaq_slope_nosys_d_matched_v3` (feet_swing_height disabled).
Ours is `g1_plane_v1_estimator_context_no_reward_matched_v2` (Context-only V2).
Use each previously evaluated model_9999.pt and verify its SHA, task, native
inputs and required estimator through inspect_checkpoint. Each has seed42 only.
No RL-only baseline is added. No native algorithm, locomotion reward, certificate
or existing checkpoint is edited. No matching local reward-on V2 checkpoint was
found; D is pending, not substituted with a reward-on V1 model.

## Plan and coordinate conventions

Three slopes (-10,0,+10 degrees), +x commands0.5/1.0m/s, zero lateral/yaw.
The supported native command upper bound is checked before simulation. Slopes
are fitted against actual USD mesh vertices. Fixed terrain, no curriculum or
training pushes, no action delay sampling, native observation and inference
structures retained. Exactly one native actor history update per policy step.

Each12s trial has the same generated reset parameters, direction and scheduled
onset across methods. Repeats share reset seeds across slopes too; states after
walking naturally differ. Reset xy in[-.2,.2]m, upright root, zero velocity,
joint position scales[.9,1.1], joint velocities zero. Root height follows actual
plane height. Collision isolation and GT extraction reuse the existing adapter.

B onset is uniform[3,5]s rounded upward to the50Hz control grid; retain both
sampled and applied time. No readiness gate. All pre-onset failures are retained.
A600, B1 2160, B2 2160, C_force1200, C_load300 episodes per method, total6420;
three methods total19260. Pilot180 per method is separate from formal results.

B1 is a0.2s true force at the torso rigid-body CoM. Direction labels refer to
initial heading: forward/backward vectors tangent to the slope, left/right along
the contour. Magnitude m*equivalent_delta_v/0.2, levels0.3/0.6/0.9m/s. The
verified world-wrench adapter transforms force and point consistently every
physics substep. This impulse scale is NOT an assertion of actual delta-v.
B2 uses corresponding initial-heading **horizontal XY** velocity jumps; vertical
and angular velocities are unchanged. Native Plane push notification follows
the already-used inference adapter semantics and must not alter the applied jump.

C_force: torso CoM force0.05/0.10*m*g,3s to12s, four directions, each slope0.5m/s.
C_load: add0.05/0.10 of original robot mass at torso body CoM from reset. Payload
inertia is an isotropic solid-sphere equivalent with radius0.1m, increment
(2/5)*added_mass*r²*I. CoM remains fixed; no parallel-axis shift. PhysX mass and
inertia are read back; no extra collision geometry is introduced; mass-weighted GT CoM uses the updated masses.

## Judge

Old Common Task judge is restricted to0.4m/s and is NOT reused as the new judge.
The old adapter's measurement schema is reused only to obtain GT and native
inference. New judge uses CoM heading-XY norm velocity error<=0.2m/s, absolute
world-reference roll and pitch each<=15deg continuously for0.3s. Evaluate every
0.02s. No teacher, certificate, estimator or policy name enters the judge.

T_rec is confirmation time minus force release or velocity-jump onset, within5s.
Later fall or abnormal termination invalidates primary success/time/steps; first
confirmation remains recorded for diagnosis. Do not require the old0.6s window,
old0.5s hold, or final sustained task-gate interval. Trial still runs to12s.

K_rec counts independent physical touchdown events strictly after release,
through confirmation. Existing contact hysteresis5N/3N,2 stable samples,0.08s
debounce; airborne before a new touchdown; initial stance excluded. Both feet
landing simultaneously count2, same-foot landings count, zero is allowed. There
is no5-step termination. Unrecovered main time/steps are null, never0 or timeout.

Tracking uses GT CoM heading XY and world-z angular velocity, timestamp trapezoid
integration from2s through actual end. Save valid duration for early failure.
C has no recovery metric because continuous force does not release during trial.

## Execution and reproduction

Use the same environment as previous Isaac evaluations. From TienKung-Lab:

```bash
export PYTHONPATH="$PWD/rsl_rl:$PWD"
PY=/home/zt/miniconda3/envs/g1/bin/python
$PY -B tools/evaluation/g1_paper_recovery.py
$PY -B -m pytest -q -p no:cacheprovider tools/evaluation/tests
$PY -B tools/evaluation/g1_paper_sim.py --model ppo --stage pilot --attempt attempt-002
$PY -B tools/evaluation/g1_paper_sim.py --model dwaq --stage pilot
$PY -B tools/evaluation/g1_paper_sim.py --model ours --stage pilot --attempt attempt-002
# Initial B2 pilots:72/72 recovered for each method; one common adjustment:
$PY -B tools/evaluation/g1_paper_calibrate.py freeze
for MODEL in ppo dwaq ours; do
  $PY -B tools/evaluation/g1_paper_sim.py --model "$MODEL" --stage calibration
done
$PY -B tools/evaluation/g1_paper_calibrate.py admit
$PY -B tools/evaluation/g1_paper_queue.py
$PY -B tools/evaluation/g1_paper_report.py --stage formal
```

All raw outputs under experiments/g1_paper_fixed_time_v1. Each attempt keeps its
own runtime, protocol and manifest hashes. Failed attempts are preserved and
excluded; model falls are valid completed episodes. Do not pool duplicate trials
or different runtimes. Comparison figures retain failures in binary denominators;
T/K distributions only include recovered successes. Bootstrap across evaluation
repeat blocks is not uncertainty across independent training seeds. Equal-cell
conditional rates are unidentified if a cell has no pushed trials; do not silently
renormalize away that cell. Also retain total designated and pre-push failure counts.
Representative Fig4 is fixed to B1_s+0_v0.5_d0_a0.6_r000 for all three methods,
including early failures; no model-specific favorable replacement.

## Frozen intensity decision

All216 initial B2 pilot episodes recovered. The single allowed common adjustment
sets B2 to0.5/1.5/2.5m/s for all methods. B1 remains0.3/0.6/0.9m/s.
Adjusted B2 calibration has72 episodes/method; it and the initial540 pilot
episodes are excluded from the19260 formal episodes. No further tuning.
The formal queue uses formal_frozen_manifest.json and protocol_frozen.json,
verified against pilot_admission.json; original plans and failed startup attempts
remain preserved. App shutdown occasionally hangs after a complete seal; the
queue verifies record hashes and terminates only its own process group after20s.
This cleanup is distinct from an incomplete or failed experiment.
