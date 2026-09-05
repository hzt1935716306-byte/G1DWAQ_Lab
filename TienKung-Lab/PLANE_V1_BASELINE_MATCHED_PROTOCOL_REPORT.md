# Plane V1 Baseline-Matched 正式协议检查报告

日期：2026-09-05
分支：`feat/multi-terrain-walking`
检查基准 HEAD：`1fdfa88e0777a0488b5108f6cb3a335f7504d4d3`

## 结论

新的 ±15° continuous coplanar terrain、baseline performance curriculum、三组 matched baseline 和四组 matched Plane V1 已实现。Gate A、B、C、D、F 通过，32-env 环境直接 step smoke 通过。

**目前不允许开始 formal 10000-iteration training。** Gate E 的坡度和上限速度已经覆盖，但 `0～0.2 m/s` 低速区以及 20% standing 没有完整、可由严格 gait-cycle 数据支持的 nominal 参数。不能用 clamp、外推或伪造零速步态解决。

## 1. 修改和新增文件

- `legged_lab/terrains/plane_terrain_cfg.py`
  - 保留旧 `PLANE_RECOVERY_TERRAINS_CFG`。
  - 新增 10×20 continuous coplanar curriculum terrain。
  - 采用 Isaac Lab 原始 difficulty 插值语义，上限改为 `tan(15°)`。
- `legged_lab/recovery/plane_terrain_math.py`
  - 独立复现列分配、difficulty RNG 顺序、signed alpha table 和 plane normal。
- `legged_lab/recovery/baseline_matched_protocol.py`
  - 4 m curriculum 判据、cardinal command 采样和 row×col slope lookup。
- `legged_lab/envs/g1/g1_slope_matched_config.py`
  - 三个旧 baseline 的 ±15° matched 配置；旧任务不变。
- `legged_lab/envs/g1/g1_slope_matched_env.py`
  - 64 m physical plane 与 8 m curriculum reference 解耦。
- `legged_lab/envs/g1/g1_plane_v1_matched_config.py`
  - 四个 Plane V1 matched 配置。
- `legged_lab/envs/g1/g1_plane_v1_matched_env.py`
  - performance curriculum、20% standing cardinal command、row×col exact geometry。
- `legged_lab/envs/__init__.py`
  - 注册七个新 matched task。
- `legged_lab/recovery/plane_nominal_params.py`
  - nominal schema/lookup 可显式表达 standing 和 speed-zero calibration node；没有自动生成节点。
- `tools/recovery/collect_dwaq_plane_nominal.py`
  - 支持 standing/zero-speed 诊断和从候选 YAML 增量 bootstrap；不覆盖旧 YAML。
- `tests/test_plane_baseline_matched_protocol.py`
  - Gate A–D 的协议测试。

`legged_lab/recovery/certificate.py` 未修改，SHA256 仍为：

`7fbef67ba3faa4bc6fdaa4d6b0de0262f85cb178d0964afe8d2a385c105234ef`

## 2. 新增 task

正式主对比候选：

- `g1_slope_nosys_d_matched`
- `g1_slope_sys_d_matched`
- `g1_dwaq_slope_nosys_d_matched`
- `g1_plane_v1_estimator_context_no_reward_matched`
- `g1_plane_v1_estimator_context_reward_matched`
- `g1_plane_v1_privileged_context_no_reward_matched`
- `g1_plane_v1_privileged_context_reward_matched`

七个任务均配置：4096 env、24 steps/env、10000 iterations、10×20 terrain、`max_init_terrain_level=5`、seed 42 默认值。

旧的约 ±20° baseline、旧 Plane terrain 和旧 Plane V1 task 均保留，未覆盖。

## 3. 原 baseline recipe 与 matched terrain

| 项目 | 旧 baseline | 新 matched 主对比 |
|---|---:|---:|
| curriculum | performance-based | performance-based |
| rows × cols | 10 × 20 | 10 × 20 |
| flat/up/down | 40/30/30% | 40/30/30% |
| columns | 8/6/6 | 8/6/6 |
| slope coefficient | `[0, 0.364]` | `[0, tan(15°)]` |
| 最大坡度 | 约 ±20° | 理论上限 ±15° |
| geometry | pyramid heightfield | continuous x-aligned coplanar plane |
| physical tile | 8×8 m | 64×32 m |
| curriculum reference | 8 m | 显式 8 m |
| move-up threshold | 4 m | 4 m |
| initial max level | 5 | 5 |

difficulty 映射严格复现 Isaac Lab：

`difficulty = (row + U[0,1)) / 10`

`coefficient = difficulty * tan(15°)`，downhill 取负，`alpha = atan(coefficient)`。

## 4. Seed 42 的 exact row × column slope table（degree）

列 0–7 为 flat，8–13 为 uphill，14–19 为 downhill。随机抖动是 Isaac TerrainGenerator 原语义，因此同一 row 的六个 slope column 不完全相等。

| row | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 | c11 | c12 | c13 | c14 | c15 | c16 | c17 | c18 | c19 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +1.020 | +0.234 | +1.395 | +0.700 | +0.897 | +0.262 | -0.747 | -1.268 | -0.165 | -0.447 | -1.367 | -1.438 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +2.158 | +2.602 | +2.608 | +1.845 | +2.531 | +2.953 | -2.287 | -2.909 | -2.939 | -2.325 | -2.904 | -1.904 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +4.312 | +3.750 | +3.474 | +3.536 | +3.197 | +3.956 | -4.501 | -3.282 | -3.420 | -3.459 | -3.861 | -3.255 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +4.850 | +5.177 | +6.071 | +5.478 | +5.229 | +5.125 | -5.467 | -5.440 | -4.653 | -6.020 | -5.077 | -5.861 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +6.152 | +6.575 | +7.297 | +6.386 | +6.181 | +7.013 | -6.835 | -6.282 | -6.958 | -6.367 | -7.287 | -6.350 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +7.767 | +8.579 | +8.709 | +8.919 | +8.374 | +7.665 | -8.033 | -8.642 | -8.189 | -7.698 | -8.626 | -7.901 |
| 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +10.211 | +9.674 | +9.804 | +10.265 | +9.626 | +10.562 | -9.629 | -9.554 | -10.371 | -9.783 | -9.692 | -10.028 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +11.307 | +10.753 | +11.026 | +11.686 | +10.837 | +11.337 | -11.393 | -11.598 | -11.817 | -12.088 | -10.763 | -11.915 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +12.335 | +12.272 | +12.240 | +12.731 | +12.250 | +13.243 | -12.741 | -13.162 | -12.563 | -13.401 | -13.190 | -12.387 |
| 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +14.283 | +14.945 | +14.860 | +14.465 | +14.408 | +13.678 | -13.590 | -14.668 | -14.933 | -14.639 | -13.938 | -14.008 |

实际范围为 `[-14.932512°, +14.945409°]`，不超过 ±15°。

## 5. Command distribution

Plane V1 matched 的 100000 次 seed-42 采样：

- standing：19.8960%
- moving 条件概率：+x 25.2147%、-x 24.8564%、+y 24.9813%、-y 24.9476%
- 无条件 moving：+x 20.1980%、-x 19.9110%、+y 20.0110%、-y 19.9840%
- diagonal：0%
- 实测范围：vx `[-0.599970, 0.999973]`，vy `[-0.499992, 0.499972]`
- yaw：恒为 0
- resampling：10 s

matched baseline 保留 baseline 自身的 UniformVelocityCommand（可能同时有 vx/vy）；Plane V1 因理论适用域使用 cardinal-only。这是明确的方法必要差异。

## 6. Push

七个 matched task 从 iteration 0 使用：

- delta-v x/y：独立 `U[-1,1] m/s`
- interval：`U[10,15] s`
- 无 push curriculum、adaptive push 或 easy sample

Plane wrapper 最终仍调用同一个 Isaac Lab `push_by_setting_velocity`，只增加 recovery bookkeeping。

10000 次采样结果：

| axis | min | max | mean | std |
|---|---:|---:|---:|---:|
| x | -0.999606 | 0.999975 | -0.001155 | 0.578430 |
| y | -0.999248 | 0.999378 | -0.003823 | 0.580607 |

理论 uniform std 为 `1/sqrt(3)=0.577350`。

## 7. Domain randomization 差异

完全相同：friction/material、base mass、joint reset、linear/roll/pitch reset velocity、observation noise、action delay 设置、velocity-jump physical operation。

Plane V1 已恢复 baseline 的 reset x/y `[-0.5,0.5]`。保留的必要差异：

- matched baseline：旧 reset yaw/yaw velocity randomization。
- Plane V1：reset yaw = 0、yaw velocity = 0、yaw command = 0，并保留 5° heading applicability gate。
- Plane V1 estimator task 多一个 IMU sensor；这是 estimator 输入需要，不改变 baseline。
- Plane V1 actor history/context 结构保持方法自身定义，不为课程对齐而改变。

## 8. Locomotion reward 差异

`g1_slope_sys_d` 与 Plane V1 的 reward term 集合相同，以下三个权重不同：

| term | g1_slope_sys_d | Plane V1 |
|---|---:|---:|
| track_lin_vel_xy_exp | 1.0 | 2.0 |
| track_ang_vel_z_exp | 1.0 | 2.0 |
| joint_deviation_hip | -0.15 | -0.30 |

其余普通 locomotion terms 和权重相同：lin_vel_z -1、ang_vel_xy -0.05、energy -0.001、dof_acc -2.5e-7、action_rate -0.01、undesired_contacts -1、fly -1、body_orientation -2、flat_orientation -1、termination -200、feet_air_time 0.15、feet_slide -0.25、feet_force -0.003、feet_too_near -2、feet_stumble -2、dof_pos_limits -2、arms -0.2、legs -0.02，以及三项 symmetry reward -1/-1/-2。

本轮没有修改任何 reward。是否为最严格因果比较统一这三个权重，需要单独决定。

## 9. Certificate/nominal support

当前候选：`tools/recovery/generated/g1_plane_nominal_params_g1_slope_sys_d_candidate.yaml`

- 140 nodes，±15°、四 cardinal directions、0.2–1.0 m/s。
- 新 terrain 的所有实际 slope 都在候选范围内，可做 bounded interpolation。
- 正式命令的上限均已覆盖：+x 1.0、-x 0.6、±y 0.5。
- 不覆盖 moving speed `<0.2` 和 standing。

在新 terrain + 新 command 上随机 20000 次 lookup：

- valid：53.22%
- moving low-speed 超出标定下界：26.58%
- standing 无节点：20.20%
- slope-out-of-bounds：0%

低速实采诊断（flat，0.05/0.10/0.15 m/s，四方向，每节点目标 40 strict cycles）：

- 可采：±x@0.10/0.15、±y@0.15，共 6/12 nodes。
- 0 cycle：±x@0.05、±y@0.05/0.10，共 6/12 nodes。
- exact standing：7 slopes×4 labels 在 1000 policy steps 内均为 0 complete gait cycle。

这说明严格周期步态参数在接近零速时并不存在于当前教师轨迹中。必须先决定 standing/near-zero 的 certificate 语义，再生成正式 candidate YAML；不能把 0.2 节点静默外推到 0。

## 10. Gate 结果

| Gate | 结果 | 证据 |
|---|---|---|
| A Terrain geometry | PASS | 实际生成 200 meshes；mesh normal vs row×col alpha 最大误差 `1.11e-16` |
| B Terrain curriculum | PASS | move-up 严格 `distance>4m`；move-down 与 BaseEnv 公式一致且 `&=~move_up` |
| C Command | PASS | 100000 samples；standing/cardinal/ranges/zero-yaw 全部通过 |
| D Push | PASS | 10000 samples；range/moments/interval/function 与 baseline 对齐 |
| E Certificate support | **FAIL** | slope 和 upper speed 通过；near-zero/standing 仅 53.22% overall lookup valid |
| F Regression | PASS | 80 tests passed；包含 dynamics、F1–F5、margin、flat/plane fixtures；certificate SHA 不变 |

32-env direct environment smoke：环境成功创建；Actor observation `(32,483)`；一步 simulation 后 observation/reward finite，shape 保持 `(32,483)`。RSL logging smoke 另发现 runner 会扫描另一份旧仓库的 Git diff；这与环境/certificate 无关，direct smoke 已绕开。

## 11. 是否可以开始 formal training

**否。** 代码层面的 matched terrain/task 已准备好，但 Gate E 尚未通过。需要用户决定以下其中一种、且必须作为明确协议：

1. 为 standing/near-zero 定义独立的静态平衡 terminal nominal（需要新的标定与 validation）；或
2. 明确 certificate 只在有完整步态的 moving support 生效，并规定 standing/near-zero 时 context/reward 的行为；或
3. 改变 command 支持下界/standing 比例，但这会偏离当前冻结目标，不能由代码自动决定。

在该决定完成、正式 candidate YAML 生成且 Gate E 重跑通过之前，不应启动七组 formal 10000-iteration training。
