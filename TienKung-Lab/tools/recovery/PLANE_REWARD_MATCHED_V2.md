# Plane reward-on V2

Base HEAD: `d9c034deee7d04cdc8bdfb3410653693c7f5621c`.

The new `g1_plane_v1_estimator_context_reward_matched_v2` inherits
`G1PlaneV1EstimatorContextRewardMatchedEnvCfg` directly and reuses the exact
`G1PlaneContextIdleSwingRewardCfg` already used by no-reward V2.

| Task suffix (all `g1_plane_v1_estimator_…`) | Context | Recoverability reward | Idle weight | Swing weight |
|---|---|---|---|---|
| context_no_reward_matched | on | off | absent | absent |
| context_reward_matched | on | on | absent | absent |
| context_no_reward_matched_v2 | on | off | -0.1 | -0.2 |
| context_reward_matched_v2 | on | on | -0.1 | -0.2 |

Only idle and swing terms differ from the reward-on parent. Full environment
comparison with no-reward V2 differs only at `plane_v1_reward.enabled`; agent
logging names differ, with identical algorithm, normalization and resume=False.
Existing registered configs retain their pre-edit canonical hashes, saved in
`configs/plane_reward_v2_parents_d9c034d.json`. Full pre-edit snapshots and the
executed audit live in `experiments/g1_plane_reward_v2_training/`.

Recoverability coefficients remain 0.25 progress, 2.0 progress clip, -0.05
unrecovered touchdown, -0.25 TD5 penalty, target N<=1 and horizon five touchdowns.
All legacy Stage2 reward channels stay disabled. Idle thresholds are 0.2/0.1 m/s;
swing target height is 0.08 m with the same ankle-roll foot selections.

Validation: 67 relevant CPU tests passed. Native smoke used 32 environments,
two fresh PPO iterations (1536 transitions), actor/critic 483/1010 and 29 actions.
A separately labelled post-training native push probe produced one nonzero
recoverability event of -0.05 within seven steps. Across observed steps there
were 27 nonzero idle values and 1011 nonzero swing values; maxima in absolute
value were 0.1 and 0.0304442. A controlled idle probe returned -0.1 for all32
environments; command/velocity were restored before the native rollout probe.
Estimator was frozen, context was (32,3), certificate queries163, policy and
optimizer tensors finite. Smoke checkpoints are not loaded into formal training.

Estimator SHA256:
`8574d845f28cbc7908e437250de46ba865898056483741e8fb72a81281f9e319`.

Run from this checkout in the `g1` environment:

```bash
PYTHONPATH="$PWD/rsl_rl:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python -u -B legged_lab/scripts/train.py \
  --task g1_plane_v1_estimator_context_reward_matched_v2 \
  --estimator_checkpoint_path logs/g1_com_velocity_estimator/v2_iteration_long_5000_random_init_fixed/com_velocity_estimator_v2_long_best.pt \
  --seed 42 --num_envs 4096 --max_iterations 10000 \
  --device cuda:0 --headless
```

This starts iteration0 with random policy initialization, no resume or warm start.
The training budget, terrain/curriculum, commands, reset, push, physics, symmetry,
solver and estimator settings remain inherited from the reward-on parent.
