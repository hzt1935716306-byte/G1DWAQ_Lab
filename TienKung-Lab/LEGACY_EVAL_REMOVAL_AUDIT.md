# Legacy evaluation removal audit

Audit date: 2026-09-12
Base branch: `formal-experiment`
Base HEAD: `1acd0e182489841ab05056172d6379be3c674903`

The v1.2 paper protocol makes `paper_eval_final/` the only admissible source for
new paper evaluation.  The paths below were audited before retirement.  No
training task or runtime module imports `tools/evaluation`.

## Deleted / retired entry points

- `tools/evaluation/` (86 tracked files): old runners, protocol YAML files,
  manifest generators, common-task detectors, robustness entry points,
  statistics/reporting code, documentation, and tests that only exercised the
  superseded protocols.
- `experiments/g1_recovery_eval_v2/STANDARD_RECOVERY_FIVE_MODEL_COMPARISON.md`:
  report generated under the superseded protocol.
- `experiments/g1_recovery_eval_v2/standard_benchmark_five_model_index.json`:
  old aggregation index.

These files are not dependencies of `paper_eval_final` and must not be used to
generate new paper claims.

## Retained

| Path/resource | Reason | Still referenced by training/runtime? |
|---|---|---|
| `legged_lab/recovery/` | Certificate mathematics, Plane runtime, state extraction, estimator/context contracts | Yes |
| `legged_lab/estimation/` | Frozen estimator implementation | Yes |
| `legged_lab/envs/` | Registered training tasks and native policy observation/runtime behavior | Yes |
| `legged_lab/assets/` | G1 robot asset | Yes |
| `rsl_rl/` | PPO/DWAQ policy and runner implementations | Yes |
| `tools/recovery/generated/g1_recovery_params.yaml` | Frozen capability parameters | Yes |
| `tools/recovery/generated/g1_plane_nominal_params.yaml` | Retained calibration | Potentially |
| `tools/recovery/generated/g1_plane_nominal_params_g1_slope_sys_d_candidate.yaml` | Matched Plane nominal gait table | Yes |
| `logs/**/model_*.pt` and estimator checkpoint | Trained/frozen artifacts; never modified by evaluation | Yes |
| training task configs | Reproduce native inference contracts | Yes |

## Legacy data policy

Existing ignored data under `experiments/g1_recovery_eval/` and
`experiments/g1_recovery_eval_v2/` is **legacy evaluation data / read-only**.
It is intentionally not moved, relabelled, or rewritten.  The new storage and
aggregation code rejects paths outside `paper_eval_final/results/` and requires
`evaluation_system: paper_eval_final`; consequently it cannot ingest this data.

## Reference check

Before retirement, a repository-wide code search excluding the legacy tree
found no imports of its modules from `legged_lab`, `rsl_rl`, training scripts,
or task registration.  Training-task import/config instantiation is rechecked
by `paper_eval_final/tests/test_training_tasks.py`.
