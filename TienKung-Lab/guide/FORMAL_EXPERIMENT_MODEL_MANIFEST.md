# Formal Experiment Model Manifest

Validated on 2026-09-12 on branch `formal-experiment`, based on commit
`3a11f44c647234c03d918e07d18c57b51c66312d`.

Only the final `model_9999.pt` checkpoint and its exact training parameter files
are archived. Intermediate checkpoints, TensorBoard event files, exported
policies, and fallback logs are intentionally excluded.

## Validation semantics

All six checkpoints below were successfully loaded with PyTorch and report
`iter = 9999`. Their saved agent/environment configurations report:

- `max_iterations = 10000`
- `seed = 42`
- `scene.num_envs = 4096`
- `num_steps_per_env = 24`

This means each entry is a completed 10,000-iteration training result rather
than a smoke-test or intermediate checkpoint.

## Checkpoints

| Role | Task / run | Network input dimensions | Model SHA256 |
|---|---|---:|---|
| Matched main baseline | `g1_dwaq_slope_nosys_d_matched/2026-09-05_18-18-46_dwaq_slope_nosys_d_matched_seed42` | DWAQ actor 115, critic 307, encoder 480 | `9b09227f6b997e0c89cb87844eeb05ffbafdfdb845701c390250fca284430361` |
| Matched main baseline | `g1_slope_nosys_d_matched/2026-09-05_18-18-07_slope_nosys_d_matched_seed42` | actor 960, critic 1010 | `15167ca59f001a3f1888c894d43c14a2cb7321f48aab1b0a8eaabdd9e74ce848` |
| Matched Plane V1 experiment | `g1_plane_v1_estimator_context_no_reward_matched_v2/2026-09-09_01-09-21_estimator_context_no_reward_matched_v2_idle01_swing02` | actor 483, critic 1010 | `9bde392d2a91ef7e20bfd0b6e84625e2663caf205577f02f0b720f1830a879c1` |
| Matched DWAQ reward ablation | `g1_dwaq_slope_nosys_d_matched_v4/2026-09-11_02-24-46_dwaq_nosys_matched_no_idle_no_swing` | DWAQ actor 115, critic 307, encoder 480 | `f836513565badc8429a0bc212cbf984b706885b9d508badd76c91cc19b147e07` |
| Matched DWAQ reward ablation | `g1_dwaq_slope_nosys_d_matched_v3/2026-09-09_01-13-06_dwaq_nosys_matched_no_swing_height` | DWAQ actor 115, critic 307, encoder 480 | `0a340c0f10b15c95348d5568ba4ac06abebe252bdf353d7cfae5021a214a2e6e` |
| Supplementary legacy slope result (not matched) | `g1_dwaq_slope/2026-09-03_13-22-08` | DWAQ actor 119, critic 311, encoder 500 | `8e5e6e45f5dd861b343f3a3542ee80a2a5f76d8d581f046f2c231d71ef8266cb` |

The legacy `g1_dwaq_slope` result uses a different observation/network shape
and predates the frozen matched protocol. It is valid as a trained checkpoint
and may be used as a supplementary comparison, but it must not be presented as
a strictly matched primary baseline.

## Parameter-file integrity

| Task / run | Parameter file | SHA256 |
|---|---|---|
| `g1_dwaq_slope_nosys_d_matched/2026-09-05_18-18-46_dwaq_slope_nosys_d_matched_seed42` | `agent.yaml` | `9f7eab10222d4e406b9dd10227e3473c40f1bb42e07a0c9646587b891165a0d5` |
|  | `env.yaml` | `fb124e0a18188a0f7c08593273580962c559bcd69a1b9726589b1e1b595785b5` |
| `g1_slope_nosys_d_matched/2026-09-05_18-18-07_slope_nosys_d_matched_seed42` | `agent.yaml` | `298464b5b8de2f3d0bfa8658354d1fe8be28b9d477f27ab78baea1d1654f1b43` |
|  | `env.yaml` | `88957dd9e8dc3e13282aae0ef213314863a12753e562a2915f6c8b00e107104d` |
| `g1_plane_v1_estimator_context_no_reward_matched_v2/2026-09-09_01-09-21_estimator_context_no_reward_matched_v2_idle01_swing02` | `agent.yaml` | `9b5507f34c2038e4e878b5e8be27c5bc34038d746d0d059749f0a458eb45d76f` |
|  | `env.yaml` | `3263e25842a05e3f5267150139073251d6a3d3d1404465befb27a559cd3fa6e1` |
|  | `estimator.yaml` | `81adc44ef80d4d75d7eb066e92e20b0eab2feff7445040e51bd802cdfe7c710c` |
| `g1_dwaq_slope_nosys_d_matched_v4/2026-09-11_02-24-46_dwaq_nosys_matched_no_idle_no_swing` | `agent.yaml` | `430771bd693e1ed50f2e4a0c80495be149071b9b3dd0ffe2ce25b8230e68be86` |
|  | `env.yaml` | `47acdd4621291d7ead1c0249c8244b668c878fa5c62b2e3b6f019e735904227c` |
| `g1_dwaq_slope_nosys_d_matched_v3/2026-09-09_01-13-06_dwaq_nosys_matched_no_swing_height` | `agent.yaml` | `a7f3f303abbcf560697f26daa35caeff5db2fe0c6e36df49dbb6c884ae8dbfaf` |
|  | `env.yaml` | `3cbc79b58f745137b5dc7caf391b9b380f9b55f3903ac1336729f2c5638843f2` |
| `g1_dwaq_slope/2026-09-03_13-22-08` | `agent.yaml` | `43b8e8875985f2745f46be1283e5adfb6d775cf5e68fd188a3df40984ba131db` |
|  | `env.yaml` | `0a9bfd059307f59f218124f6d6db46d4906c981a9e4d0447083c5b0cfde42203` |
