# G1 Stage2 experiment comparison record

Last updated: 2026-09-01 (Asia/Shanghai)

The completed Phase-1 out-of-distribution disturbance suite (11 models,
147,840 paired episodes covering velocity jumps, force pulses, constant force,
repeated impacts, OU random force, and force-plus-torque pulses) is recorded in
[`ALL_MODELS_DISTURBANCE_PHASE1.md`](ALL_MODELS_DISTURBANCE_PHASE1.md).

The complete unified comparison of all 11 formal final models (50,688 paired
fixed-protocol episodes) is recorded in
[`ALL_TRAINED_MODEL_COMPREHENSIVE_COMPARISON.md`](ALL_TRAINED_MODEL_COMPREHENSIVE_COMPARISON.md).
This is the primary cross-version report.

The completed certificate-only weight and no-curriculum comparison (five
models, 23,040 paired fixed-protocol episodes) is recorded in
[`CERTIFICATE_WEIGHT_NO_CURRICULUM_COMPARISON.md`](CERTIFICATE_WEIGHT_NO_CURRICULUM_COMPARISON.md).

The completed Shared Baseline 0.2 fixed-protocol comparison, including
per-level, per-command, and paired-trial statistics, is recorded in
[`SHARED_BASELINE_COMPREHENSIVE_COMPARISON.md`](SHARED_BASELINE_COMPREHENSIVE_COMPARISON.md).

The native-system comparison between the latest Unitree model and the Ours
0.20/0.25 curriculum models under component-wise +/-0.5 m/s velocity jumps is
recorded in
[`CROSS_PROJECT_FIXED_PUSH_B050_COMPARISON.md`](CROSS_PROJECT_FIXED_PUSH_B050_COMPARISON.md).
Its primary range-matched Ours checkpoint is the last L3 checkpoint before the
curriculum moved from +/-0.55 to +/-0.70 m/s.

This document records the comparisons completed before and during the current
certificate-only reward sweep.  It separates fixed-protocol evaluation from
online training statistics because the two answer different questions and must
not be mixed.

## Metric definitions

- `P5` is the fraction of recovery episodes that reach the practical success
  condition within at most five touchdowns.
- `mean enter step` and `median enter step` are computed only over successful
  episodes.  TIMEOUT and FALL episodes are excluded, so enter-step statistics
  must always be read together with P5.
- Fixed-protocol results below are inference-only evaluation results.  Online
  training results are cumulative, non-stationary statistics collected while
  the policy is changing.

## Experiment identities

| Short name | Checkpoint | Reward used for Stage2 training | Status |
|---|---|---|---|
| Baseline-original | `logs/g1_flat_symmetric/2026-08-30_02-35-28_stage2_baseline_original_from4999/model_14998.pt` | Original locomotion reward only; no shared events and no certificate | Valid primary baseline |
| Ours cert-only 0.5 | `logs/g1_flat_symmetric/2026-08-30_02-39-40_stage2_ours_cert050_async_from4999/model_14998.pt` | Certificate only, `event_scale=0.5` | Valid primary Ours run |
| Ours old shared+cert 0.2 | `logs/g1_flat_symmetric/2026-08-29_13-12-16_stage2_ours_scale02_solverfix_resume/model_14999.pt` | Touchdown cost, practical success bonus, timeout penalty, and certificate, all with `event_scale=0.2` | Historical ablation only; not certificate-only |
| DWAQ old | `logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt` | Original DWAQ training | Historical reference |
| DWAQ flat new | `logs/model_9999.pt` | Newly trained flat-ground DWAQ | Reference policy |

The historical shared+certificate run resumed from `model_7400.pt`, and its
curriculum controller restarted at L1.  It is therefore not a clean causal
comparison against runs that started continuously from `model_4999.pt`.

## Fixed-protocol comprehensive evaluation

Protocol for every policy and every seed:

- seeds: 42, 123, 2026;
- levels: L1--L6 with ratios 0.25, 0.40, 0.55, 0.70, 0.85, 1.00;
- 256 episodes per level and 1536 episodes per seed;
- 4608 episodes per policy across the three seeds;
- identical saved trial plan for policies within each seed;
- eight command conditions;
- flat plane, observation noise off, physics randomization off;
- maximum five recovery touchdowns.

Values are the mean across the three seeds, followed by population standard
deviation across seeds.

### Overall results

| Policy | P5 | Mean enter step | Mean successful recovery time | Fall rate |
|---|---:|---:|---:|---:|
| Baseline-original | 92.23% +/- 0.59 pp | 3.628 +/- 0.040 | 0.744 +/- 0.009 s | 0% |
| Ours cert-only 0.5 | 86.15% +/- 0.54 pp | 3.780 +/- 0.045 | 0.802 +/- 0.013 s | 0.022% |
| Ours old shared+cert 0.2 | 93.45% +/- 0.65 pp | 3.554 +/- 0.039 | 0.756 +/- 0.012 s | 0% |
| DWAQ old | 34.83% +/- 0.54 pp | 3.426 +/- 0.004 | 1.183 +/- 0.011 s | 0% |
| DWAQ flat new | 94.25% +/- 0.57 pp | 3.509 +/- 0.016 | 1.188 +/- 0.005 s | 0% |

The low successful-episode enter-step mean of DWAQ old is selection bias: only
34.83% of its episodes succeed.  It must not be interpreted as the strongest
recovery policy.

### Per-level P5 and mean enter step

Each cell is `P5 / mean successful enter step`.

| Level | Baseline-original | Ours cert-only 0.5 | Ours old shared+cert 0.2 | DWAQ old | DWAQ flat new |
|---|---:|---:|---:|---:|---:|
| L1 | 88.02% / 2.642 | 87.50% / 2.871 | 87.89% / 2.493 | 30.21% / 2.596 | 100.00% / 2.672 |
| L2 | 92.06% / 3.187 | 91.67% / 3.426 | 91.67% / 3.061 | 33.98% / 3.093 | 99.48% / 3.178 |
| L3 | 94.53% / 3.638 | 91.54% / 3.845 | 95.05% / 3.520 | 36.46% / 3.378 | 98.05% / 3.587 |
| L4 | 95.57% / 3.907 | 89.19% / 4.077 | 96.48% / 3.864 | 35.81% / 3.658 | 94.92% / 3.772 |
| L5 | 94.27% / 4.089 | 83.59% / 4.221 | 96.61% / 4.073 | 36.85% / 3.791 | 89.71% / 3.916 |
| L6 | 88.93% / 4.260 | 73.44% / 4.360 | 92.97% / 4.217 | 35.68% / 3.886 | 83.33% / 4.077 |

### Fixed-evaluation conclusions

1. The clean comparison is Baseline-original versus Ours cert-only 0.5.  The
   baseline is higher by 6.08 percentage points overall P5, uses 0.152 fewer
   successful enter steps on average, and completes successful recovery 0.058 s
   faster on average.
2. The cert-only 0.5 deficit grows with disturbance level: its P5 deficit
   relative to Baseline-original is approximately 0.52, 0.39, 2.99, 6.38,
   10.68, and 15.50 percentage points from L1 through L6.
3. The old shared+certificate run performs well, but the three shared event
   rewards and the resume/curriculum-reset history prevent attributing that
   result to the certificate reward.
4. DWAQ flat new has the highest overall P5 in this fixed protocol, but its
   successful recovery time is much longer than Baseline-original despite its
   lower touchdown count.  Touchdown cadence and wall-clock recovery time must
   therefore be reported separately.

Raw results: [`generated/three_policy_comprehensive_final/`](generated/three_policy_comprehensive_final/).

## Stage2 curriculum upgrade history

`P` means the performance window passed.  `M` means the level reached the
1800-iteration maximum and was upgraded without passing the performance
criterion.  The numbers are TensorBoard global learning iterations.

| Run | L1 -> L2 | L2 -> L3 | L3 -> L4 | L4 -> L5 | L5 -> L6 |
|---|---:|---:|---:|---:|---:|
| Old shared-0.2 Baseline | 5500 P | 6426 P | 8225 M | 10025 M | 11825 M |
| Baseline-original | 5923 P | 7722 M | 9522 M | 11322 M | 13122 M |
| Ours cert-only 0.5 | 6799 M | 8599 M | 10399 M | 12199 M | 13999 M |
| Ours cert-only 0.2, in progress at this snapshot | 5730 P | 7530 M | 9330 M | pending | pending |

The old shared+certificate Ours trajectory has two segments:

- initial segment from `model_4999.pt`: L2 at 5499 P and L3 at 6888 P;
- solver-fix segment from `model_7400.pt`, with curriculum reset to L1: L2 at
  7900 P, L3 at 8421 P, L4 at 10220 M, L5 at 12020 M, and L6 at 13820 M.

Upgrade event sources:

- [`Baseline-original`](../../logs/g1_flat_symmetric/2026-08-30_02-35-28_stage2_baseline_original_from4999/push_curriculum_upgrades.jsonl)
- [`Ours cert-only 0.5`](../../logs/g1_flat_symmetric/2026-08-30_02-39-40_stage2_ours_cert050_async_from4999/push_curriculum_upgrades.jsonl)
- [`Old shared-0.2 Baseline`](../../logs/g1_flat_symmetric/2026-08-29_13-15-09_stage2_baseline_scale02_solverfix_resume/push_curriculum_upgrades.jsonl)
- [`Old shared+certificate 0.2 Ours, initial`](../../logs/g1_flat_symmetric/2026-08-29_02-50-27_stage2_ours_scale02/push_curriculum_upgrades.jsonl)
- [`Old shared+certificate 0.2 Ours, resumed`](../../logs/g1_flat_symmetric/2026-08-29_13-12-16_stage2_ours_scale02_solverfix_resume/push_curriculum_upgrades.jsonl)

## Final online training statistics

These are final cumulative TensorBoard statistics.  Previous levels continue
to receive 20% easy replay samples after an upgrade, so these values are not
restricted to the interval when a level was the active curriculum level.

Each cell is `mean enter step / median enter step / P5`.

| Level | Old shared+certificate 0.2 Ours | Old shared-0.2 Baseline | Baseline-original | Ours cert-only 0.5 |
|---|---:|---:|---:|---:|
| L1 | 3.05 / 3 / 93.17% | 3.16 / 3 / 82.22% | 3.28 / 3 / 83.61% | 3.33 / 3 / 78.77% |
| L2 | 3.61 / 4 / 86.92% | 3.68 / 4 / 81.83% | 3.81 / 4 / 74.20% | 3.81 / 4 / 69.47% |
| L3 | 3.94 / 4 / 78.28% | 3.97 / 4 / 75.01% | 4.08 / 4 / 65.69% | 4.07 / 4 / 62.58% |
| L4 | 4.15 / 4 / 69.35% | 4.14 / 4 / 66.43% | 4.22 / 4 / 57.83% | 4.23 / 4 / 55.84% |
| L5 | 4.27 / 4 / 61.23% | 4.24 / 4 / 58.36% | 4.34 / 5 / 50.57% | 4.33 / 5 / 49.27% |
| L6 | 4.34 / 5 / 54.63% | 4.32 / 4 / 55.53% | 4.39 / 5 / 44.02% | 4.39 / 5 / 42.00% |

FALL is extremely rare in all these online runs.  Most failures are TIMEOUT,
so the main learned-performance question is whether recovery completes within
five touchdowns, not whether the robot immediately falls.

## In-progress certificate-only weight sweep

The local `event_scale=0.2` run was still active when this record was written:

- run: `2026-08-31_00-20-09_stage2_ours_certonly020_async_from4999`;
- latest inspected TensorBoard step: approximately 10129, currently L4;
- L1: mean 3.31, median 3, P5 80.85%;
- L2: mean 3.84, median 4, P5 70.30%;
- L3: mean 4.09, median 4, P5 64.21%;
- L4: mean 4.24, median 4, P5 53.54%;
- L5 and L6: no data at this snapshot.

The remote certificate-only 0.15 and 0.25 runs are not recorded here yet
because their completed logs have not been copied to this machine.

## Older exploratory outputs

These files are preserved for traceability but should not replace the final
three-seed fixed-protocol comparison because some use different checkpoints or
only one level:

- `generated/stage2_level5_baseline_model11500.json`
- `generated/stage2_level5_baseline_model13200.json`
- `generated/stage2_level5_ours_model13200.json`
- `generated/stage2_level5_ours_model13400.json`
- `generated/three_policy_level_sweep_ckpt14300/`
- `generated/stage2_certificate_level1_model6800.json`
- `generated/stage2_certificate_level2_model6800.json`

Reward and solver smoke-test records are kept separately in
`generated/stage2_reward_offline_replay.json`,
`generated/stage2_baseline_online_smoke.json`, and
`generated/stage2_ours_online_smoke.json`.

## 2026-09-01 cross-project disturbance survival suite

Five policies were tested under six disturbance families and 32 conditions:
the latest Unitree native policy, range-matched Ours certificate-only 0.20 L3,
Ours certificate-only 0.25 final, Baseline-shared-0.2 at the final L3
checkpoint, and Baseline-original trained without curriculum at fixed L6.
Each model received 6144 paired trials (3 seeds x 32 conditions x 64 trials),
for 30720 trials total.  Survival required no native termination during the
disturbance or the fixed 10 s post-release horizon.

| Model | Full-horizon survival | P5 | Successful-step mean | Failure-aware step mean |
|---|---:|---:|---:|---:|
| Unitree-17800 | 71.37% | 50.37% | 3.044 | 4.511 |
| Ours-0.20-L3 | 66.68% | 47.57% | 2.822 | 4.488 |
| Ours-0.25-final | 96.71% | 66.06% | 3.057 | 4.056 |
| Baseline-shared-0.2-L3 | 65.87% | 49.04% | 2.724 | 4.393 |
| Baseline-original-NC | 97.46% | 67.01% | 3.012 | 3.998 |

`Successful-step mean` excludes TIMEOUT/FALL.  `Failure-aware step mean` uses
the actual 1--5 enter step for SUCCESS and assigns 6 to TIMEOUT/FALL.  The
complete family and condition tables are in
[`CROSS_PROJECT_SURVIVAL_SUITE_COMPARISON.md`](CROSS_PROJECT_SURVIVAL_SUITE_COMPARISON.md).

The full-range Baseline-original-NC is 0.75 pp higher than Ours-0.25-final in
full-horizon survival (paired McNemar p=0.0048) and 0.94 pp higher in P5
(p=0.0253).  At L3, Ours-0.20 and Baseline-shared-0.2 differ by +0.81 pp in
survival (p=0.0839), while the shared Baseline is 1.46 pp higher in P5
(p=5.17e-05).  These results show that maximum training disturbance exposure
is a major confounder and should be matched before attributing the final
model's robustness to the certificate reward.

The corrected fixed component-wise +/-0.5 m/s comparison is in
[`CROSS_PROJECT_FIXED_PUSH_B050_COMPARISON.md`](CROSS_PROJECT_FIXED_PUSH_B050_COMPARISON.md).
The corrected Unitree adapter converts its policy-step counter to physics
steps before computing time.  In that protocol, Ours-0.20-L3 P5 is 94.14%
versus Unitree-17800 at 79.43%; all four evaluated models have 100% non-fall.
