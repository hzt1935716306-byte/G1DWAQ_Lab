# Plane V1 Baseline-Matched 正式协议收口报告

日期：2026-09-05
分支：`feat/multi-terrain-walking`
本轮基准提交：`a79fc47a71d18ddaee335483bab90df2718c0d41`

## 结论

最终 7 个 matched task 的 terrain、command、reset yaw、push 和训练预算已经对齐。Plane V1 的普通 locomotion reward 已与 `g1_slope_sys_d_matched` 完全一致；reward-on 只额外启用独立 recoverability touchdown-event reward，reward-off 的该通道严格为零。

Moving command 的 nominal lookup 覆盖率为 100%，standing 被明确标记为 intentional not-applicable，不提交 LP、不产生恢复奖励，也不计为 geometry/solver/theory failure。因此新的 Gate A–F 全部通过，已经具备开始 formal training 的协议条件。本轮没有启动 10000-iteration 训练。

## 1. 最终 7 个 task

Baseline：

- `g1_slope_nosys_d_matched`：plain PPO，无 symmetry reward/mirror loss。
- `g1_slope_sys_d_matched`：symmetric PPO，保留 symmetry reward/mirror loss。
- `g1_dwaq_slope_nosys_d_matched`：DWAQ，保留 encoder/latent/AE 和无固定 gait phase 定义。

Plane V1：

- `g1_plane_v1_estimator_context_no_reward_matched`
- `g1_plane_v1_estimator_context_reward_matched`
- `g1_plane_v1_privileged_context_no_reward_matched`
- `g1_plane_v1_privileged_context_reward_matched`

所有旧 baseline、旧 Plane V1 task 和旧约 ±20° terrain 均保留，未覆盖。

## 2. Terrain 与 curriculum 协议

- continuous x-aligned coplanar plane，不使用 pyramid terrain；
- 10 rows × 20 cols；
- flat/uphill/downhill = 40%/30%/30%，对应 8/6/6 列；
- 最大坡度 ±15°，最大 slope coefficient 为 `tan(15°)`；
- difficulty 到 slope magnitude 的映射与原 baseline 相同；
- performance-based move-up/move-down curriculum；
- physical plane 为 64 m，curriculum reference length 显式保持 8 m；
- move-up threshold 为 4 m；
- initial max terrain level 为 5。

seed 42 的 200 个实际 mesh slope 范围为 `[-14.932512°, +14.945409°]`。mesh normal 与 row×column metadata normal 的最大绝对误差为 `1.11e-16`。

## 3. Command 协议

7 个 matched task 现在使用同一个 `BaselineMatchedCardinalVelocityCommand` 实现：

- command resampling time：10 s；
- standing：20%，`vx=vy=yaw=0`；
- moving：80%；
- moving 内 +x/-x/+y/-y 条件概率各 25%；
- diagonal probability：0；
- +x：`U[0.2, 1.0] m/s`；
- -x：`U[0.2, 0.6] m/s`；
- ±y：`U[0.2, 0.5] m/s`；
- moving speed 不低于 0.2 m/s；
- yaw command 恒为 0，`heading_command=False`，`rel_heading_envs=0`。

100000 次 seed-42 采样结果：

- standing：19.8960%；
- moving 条件概率：+x 25.2147%、-x 24.8564%、+y 24.9813%、-y 24.9476%；
- diagonal：0%；
- yaw：严格为 0；
- moving minimum speed：不低于 0.2 m/s；
- vx/vy 全部位于上述方向范围内。

## 4. Standing certificate 语义

Moving command（speed ≥ 0.2 m/s）继续使用完整 Plane V1 certificate、N/m context 和可选 recovery reward。

Standing command 被定义为 `intentional_not_applicable`：

- 不向 LP solver 提交 query；
- recovery context 固定为 `[0, 0, 0]`；
- `current_n_min=-1`（N/A）；
- `current_margin=0`；
- `current_certificate_valid=False`；
- 不给 delta-Phi reward；
- 不给 unrecovered touchdown cost；
- 不给 TD5 penalty；
- 不统计为 solver failure、geometry failure 或 theory failure。

Standing 环境仍正常接受 velocity-setting push，继续运行普通 locomotion reward、termination 和 PPO。若 moving recovery 期间 command 被重新采样为 standing，只结束 certificate/recovery-reward bookkeeping，不改变物理环境或普通训练。

32-env reward-on 集成检查中强制全部 standing 并触发 push：LP submissions=0、recovery reward=0、recovery active=0、context 全零。

## 5. Reset yaw 协议

7 个 matched task 均为：

- reset x/y：`[-0.5, 0.5]`；
- reset yaw：严格为 0；
- reset yaw velocity：严格为 0；
- yaw command：严格为 0。

原有 linear velocity、roll/pitch velocity、joint、friction、mass、observation noise 和 action-delay randomization 未改。

## 6. Push 协议

7 个 matched task 从 iteration 0 开启：

- delta-v x/y 独立 `U[-1, 1] m/s`；
- interval `U[10, 15] s`；
- 无 push curriculum；
- 无 adaptive upgrade；
- 无 easy-sample mixture。

Plane V1 的 wrapper 只增加 recovery bookkeeping，实际扰动仍为相同的 velocity-setting push。此前 10000 次 push 采样已验证范围、均值和标准差符合独立均匀分布。

## 7. Locomotion reward 对齐

自动比较配置对象的完整 `reward.to_dict()`：

- `g1_slope_sys_d_matched` 与 4 个 Plane V1 matched 的所有普通 locomotion reward term、function、parameters 和 weight 完全一致；
- `track_lin_vel_xy_exp=1.0`；
- `track_ang_vel_z_exp=1.0`；
- `joint_deviation_hip=-0.15`；
- reward-off 的 recoverability event channel 严格关闭；
- reward-on 只额外启用 recoverability touchdown-event reward。

DWAQ 自身的方法性 reward/encoder/latent/AE 定义没有被改成普通 PPO。

## 8. Gate A–F 结果

| Gate | 结果 | 证据 |
|---|---|---|
| A Terrain geometry | PASS | 200 meshes；row×column slope/normal 一致；最大坡度不超过 ±15° |
| B Terrain curriculum | PASS | performance move-up/down；8 m reference；4 m move-up；initial max level 5 |
| C Command + reset | PASS | 100000 samples；standing/cardinal/min-speed/ranges/zero-yaw；7/7 reset yaw checks |
| D Push | PASS | x/y `U[-1,1]`、10–15 s、iteration 0、无 push curriculum |
| E Certificate support | PASS | standing N/A 单列；moving lookup 100%；OOB 0；representative solver failure 0 |
| F Reward/regression/smoke | PASS | reward dict 4/4 相等；82 tests；7/7 32-env direct-step smoke |

`legged_lab/recovery/certificate.py` 未修改，SHA256 仍为：

`7fbef67ba3faa4bc6fdaa4d6b0de0262f85cb178d0964afe8d2a385c105234ef`

## 9. Moving certificate support 覆盖率

使用正式共享 command sampler、seed-42 真实 row×column slope table 和 candidate nominal YAML，随机检查 100000 个 command/terrain pair：

- standing N/A count：19896；
- moving count：80104；
- moving lookup valid：80104/80104 = 100%；
- geometry invalid rate：0%；
- speed out-of-bounds rate：0%；
- slope out-of-bounds rate：0%；
- other lookup invalid rate：0%。

另对 200 个覆盖实际 slope/direction/speed 的 periodic nominal query 运行 certificate：solver failure 0/200。

使用的 nominal 文件：

`tools/recovery/generated/g1_plane_nominal_params_g1_slope_sys_d_candidate.yaml`

## 10. Standing N/A 比例

- 100000 次正式 command 采样：19.8960%；
- 目标协议：约 20%；
- 它不进入 moving lookup valid 分母，也不计为 Gate E failure。

运行时日志已分别提供 intentional-not-applicable、geometry-invalid 和 solver-failure 计数/比例，三种语义不会静默混合。

## 11. 32-env direct-step smoke

7 个 matched task 均完成真实 Isaac Sim 环境创建和一步 simulation：

| task | actor observation | 结果 |
|---|---:|---|
| g1_slope_nosys_d_matched | 32×960 | finite PASS |
| g1_slope_sys_d_matched | 32×960 | finite PASS |
| g1_dwaq_slope_nosys_d_matched | 32×96 | finite PASS |
| g1_plane_v1_estimator_context_no_reward_matched | 32×483 | finite PASS |
| g1_plane_v1_estimator_context_reward_matched | 32×483 | finite PASS |
| g1_plane_v1_privileged_context_no_reward_matched | 32×483 | finite PASS |
| g1_plane_v1_privileged_context_reward_matched | 32×483 | finite PASS |

所有 observation、action 和 reward 均为 finite，无 NaN、无 solver crash。Estimator smoke 的第一步仍处于 5-frame warm-up，因此没有提交 certificate；privileged smoke 提交了 11 个 moving touchdown query，standing 环境未提交。

## 12. Formal training readiness

协议与本轮要求的 Gate 已全部通过，代码已具备 formal training 条件。正式冻结配置为：4096 env、24 steps/env、10000 iterations。

本轮没有启动训练、没有 resume 旧 pilot、没有修改 certificate、没有混入 certificate 性能优化。正式长训需等待用户确认后再开始。
