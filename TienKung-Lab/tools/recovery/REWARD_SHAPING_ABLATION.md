# Reward shaping ablation: three new tasks

Base: `6a5ad21710d878129970266084565f9f03094055` on `feature/g1-recovery-eval-portable`.
Existing registrations, environment implementations, PPO/DWAQ implementations,
terrain, commands, pushes, certificate and estimator are unchanged.

| New task | Parent | Only active reward differences |
|---|---|---|
| `g1_plane_v1_estimator_context_no_reward_matched_v2` | Plane Context-only matched | Add idle −0.1 and swing height −0.2 |
| `g1_dwaq_slope_nosys_d_matched_v2` | Original DWAQ matched | Disable idle penalty |
| `g1_dwaq_slope_nosys_d_matched_v3` | Original DWAQ matched | Disable swing height penalty |

Idle thresholds remain command=0.2 m/s and velocity=0.1 m/s; swing height is
0.08 m, both foot selections `.*ankle_roll.*`. Plane still uses estimator-based
certificate context, symmetry rewards, data augmentation and mirror loss, with
all legacy and V1 certificate rewards disabled. DWAQ gait phase reward stays off.

The pre-edit instantiated parent snapshots are in
`configs/reward_shaping_parent_6a5ad21.json`. The audit checks the full canonical
environment and agent dictionaries, not just selected weights. Only the stated
reward terms and agent experiment/project/run names (resume=False) may differ.
The snapshots bind this workspace's resolved native resource paths.

## Audit and smoke

Activate `g1` and run from the TienKung-Lab directory. Explicitly use this
checkout's bundled RSL-RL; Isaac can otherwise resolve a different project copy.

```bash
export PYTHONPATH="$PWD/rsl_rl:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -B tools/recovery/audit_reward_shaping_ablation.py --headless --output /tmp/reward_ablation_audit.json
```

`smoke_reward_shaping_ablation.py --task TASK --output NEW_JSON` runs the unchanged
native training entry for 32 environments and two iterations from random policy
initialization. It checks network inputs, active terms, finite tensors and
optimizer updates. A separate post-training native push probe checks the original
push function/distribution without shortening interval timing or changing terrain.
Plane additionally checks estimator freezing and certificate queries; DWAQ checks
encoder/decoder optimizer state. Smoke logs/checkpoints live in `TASK_smoke` and
are never loaded by formal training.

## Formal training

All three inherit seed=42, num_envs=4096, steps/env=24 and 10000 iterations. The
native final checkpoint is `model_9999.pt` (iterations 0 through 9999). Never pass
`--resume False`: the old CLI uses Python bool parsing; omit `--resume` entirely.

```bash
python -u -B legged_lab/scripts/train.py \
  --task g1_plane_v1_estimator_context_no_reward_matched_v2 \
  --estimator_checkpoint_path logs/g1_com_velocity_estimator/v2_iteration_long_5000_random_init_fixed/com_velocity_estimator_v2_long_best.pt \
  --seed 42 --num_envs 4096 --max_iterations 10000 --device cuda:0 --headless

python -u -B legged_lab/scripts/train.py \
  --task g1_dwaq_slope_nosys_d_matched_v2 \
  --seed 42 --num_envs 4096 --max_iterations 10000 --device cuda:0 --headless

python -u -B legged_lab/scripts/train.py \
  --task g1_dwaq_slope_nosys_d_matched_v3 \
  --seed 42 --num_envs 4096 --max_iterations 10000 --device cuda:0 --headless
```

The serial host queue `run_reward_shaping_training.py --plan PLAN_JSON` executes
these commands in order on one GPU. It pins the commit and source hashes, requires
all three smoke results, starts only empty new formal log roots, and stops on
execution errors. It never resumes/retries training or loads any policy weights.
Only the required frozen estimator is loaded for Plane.

A verified final checkpoint must contain 983040000 training transitions, iteration
9999, no parent training run, matching task/config hashes and the fixed estimator
hash. Only after successful completion may the queue terminate a stalled Isaac
shutdown. A final `REWARD_SHAPING_ABLATION.md` and training records are maintained
under `experiments/g1_reward_shaping_ablation_v1/`. Training completion does not
establish improved recovery performance; evaluating new models is a separate task.

## Implementation validation (2026-09-09)

Full instantiated parent/child audit: PASS. Relevant CPU tests: 60 passed.
All three native smoke tests passed (32 environments, two optimizer iterations).
Plane actor/critic inputs: 483/1010, context width 3; DWAQ actor/critic inputs:
115/307, encoder input 480, actions 29 for all. Original parent policy and
estimator files were only read. Development logs and accepted smoke JSON are
retained in the separate ablation experiment directory.
