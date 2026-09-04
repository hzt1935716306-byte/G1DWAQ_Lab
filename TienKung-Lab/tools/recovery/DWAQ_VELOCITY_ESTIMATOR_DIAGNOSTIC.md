# DWAQ 速度估计器精度与 recoverability certificate 敏感性诊断

## 结论

当前 `g1_dwaq_slope_d.pt` 中的速度估计头可以直接从完整 checkpoint 加载和调用，但**不适合未经再训练就作为当前 recoverability certificate 的 CoM 速度输入**。

主要原因不是 base velocity 与 whole-body CoM velocity 的定义差异，而是现有 mean velocity head 本身存在明显低估，并在速度跳变后的 TD0/TD1 严重滞后：

- nominal 前向速度 bias 为 `-0.127 ~ -0.196 m/s`；
- TD0 的 base velocity 水平向量误差 P95 为 `1.021 ~ 1.097 m/s`；
- 直接当作 CoM velocity 后，TD0 DCM 误差 P95 为 `24.4 ~ 26.4 cm`；
- TD0 的 `N` exact agreement 只有 `32.8% ~ 43.8%`，within-one agreement 为 `54.8% ~ 65.6%`；
- TD0 margin Spearman 接近 0。

因此，后续若要从无特权观测获得 certificate 所需速度，应考虑从当前 encoder/mean head 初始化一个独立 estimator，再使用 supervised velocity loss 单独训练或 fine-tune。本轮没有执行训练。

## 实验边界

- checkpoint：`logs/g1_dwaq_slope_d.pt`
- checkpoint SHA256：`ccc420c1195810330f103ba0e64379ddea1907a4eeceb502b50487a3b8d882c4`
- iteration：9999
- 坡度：`-10° / 0° / +10°`
- command：`vx=0.4 m/s, vy=0, wz=0`
- nominal：每个坡度 1000 个同步 policy env-frame 样本
- recovery：每个坡度 64 trials，共 192 trials
- push：`±x / ±y × 0.25/0.50/0.75/1.00 m/s × phase 0.25/0.75 × repeat 2`
- TD 样本：`321 / 384 / 344`，合计 1049
- 使用既有 Plane V1 nominal 参数，只读且未修改
- 没有修改 certificate、theory、reward、curriculum、Actor 或训练流程

注意：Plane V1 nominal 表来自 checkpoint SHA 前缀 `8e5e6e45f5dd`，本次 estimator checkpoint SHA 前缀为 `ccc420c11958`。这不影响同一真实状态下的 GT/direct velocity-only certificate 对照，但会使本次绝对 Gate B terminal correlation 不适合单独解释为该 policy 的正式 Gate B 结论。

## Checkpoint 与估计语义核查

| 项目 | 结果 |
|---|---|
| Actor observation dimension | 96 |
| obs history length | 5 |
| encoder input | 480 |
| encoder | `480 -> 128 -> 64` |
| deterministic velocity head | `encode_mean_vel: 64 -> 3` |
| estimator inference | `encode_mean_vel(encoder(obs_history))` |
| stochastic `code_vel` | diagnostic 未使用 |
| target | `critic_obs[:, obs_dim:obs_dim+3]` |
| target physical meaning | `root_lin_vel_b`，root rigid-body CoM velocity，body frame |
| `obs_scales.lin_vel` | 1.0 |
| observation normalizer | runner 关闭；checkpoint 无 normalizer state |
| time alignment | 最新 history frame 与同一 post-step simulator state |

checkpoint 中确认存在：

- `encoder.0.{weight,bias}`
- `encoder.2.{weight,bias}`
- `encode_mean_vel.{weight,bias}`
- `encode_logvar_vel.{weight,bias}`

估计值先按 `obs_scales.lin_vel` 反缩放，再由完整 base orientation 做 body-to-world 转换，最后由 heading quaternion 做 world-to-heading 转换。

## Nominal base velocity accuracy

单位均为 m/s。`xy P95` 是水平向量误差的 P95。

| slope | RMSE x | RMSE y | RMSE z | bias x | bias y | bias z | xy P95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -10° | 0.195 | 0.067 | 0.092 | -0.181 | -0.058 | +0.065 | 0.268 |
| 0° | 0.201 | 0.058 | 0.076 | -0.196 | -0.050 | -0.009 | 0.242 |
| +10° | 0.152 | 0.068 | 0.102 | -0.127 | -0.058 | -0.049 | 0.214 |

直接把估计的 base velocity 当作 whole-body CoM velocity 后，nominal 水平向量误差 P95 分别为：

- -10°：0.323 m/s
- 0°：0.246 m/s
- +10°：0.232 m/s

## Recovery TD0/TD1 accuracy

### TD0

| slope | base RMSE x | base RMSE y | base RMSE z | base xy P95 | direct-CoM xy P95 | DCM xy P50 | DCM xy P95 | DCM xy max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -10° | 0.488 | 0.406 | 0.144 | 1.097 m/s | 0.988 m/s | 13.6 cm | 26.3 cm | 29.9 cm |
| 0° | 0.470 | 0.400 | 0.113 | 1.021 m/s | 0.999 m/s | 12.9 cm | 26.4 cm | 28.7 cm |
| +10° | 0.470 | 0.424 | 0.099 | 1.026 m/s | 0.929 m/s | 12.5 cm | 24.4 cm | 27.2 cm |

### TD1 与 TD2～TD5 变化

下表是 direct base-as-CoM 的水平向量 RMSE，单位 m/s。

| slope | TD0 | TD1 | TD2 | TD3 | TD4 | TD5 |
|---:|---:|---:|---:|---:|---:|---:|
| -10° | 0.598 | 0.535 | 0.390 | 0.203 | 0.191 | 0.201 |
| 0° | 0.583 | 0.501 | 0.249 | 0.208 | 0.181 | 0.202 |
| +10° | 0.557 | 0.495 | 0.235 | 0.211 | 0.136 | 0.196 |

误差通常到 TD3 才明显回落。因此 nominal accuracy 不能代表 certificate 实际最关心的 TD0 分布。

## Base 与 whole-body CoM 的结构差异

为了分离“估计器误差”和“base 不是 whole-body CoM”两种来源，额外比较了 GT base velocity 与 GT whole-body CoM velocity。

TD0 水平向量误差 P95：

| slope | estimator vs GT base | GT base vs GT CoM | direct estimate vs GT CoM |
|---:|---:|---:|---:|
| -10° | 1.097 m/s | 0.149 m/s | 0.988 m/s |
| 0° | 1.021 m/s | 0.268 m/s | 0.999 m/s |
| +10° | 1.026 m/s | 0.291 m/s | 0.929 m/s |

说明 kinematic correction 可能有帮助，但无法修复当前约 1 m/s 量级的 estimator TD0 误差；主要问题仍是 velocity head 对突变后的真实 base velocity 跟踪不足。

当前仓库和 IsaacLab articulation wrapper 中没有找到可直接复用、且适合部署的 whole-body CoM Jacobian/centroidal kinematics API，因此没有为了本次诊断临时手写质量加权 Jacobian。`v_com_kin` 保留为 pending。

## Certificate sensitivity

对每一个 touchdown，GT query 中的 CoM position、feet、q、support、slope、T、h 和 nominal 参数全部保持不变；direct query 只做：

```text
b_direct = b_GT + (v_direct - v_com_GT) / omega
```

### TD0 agreement

| slope | samples | N exact | N within-one | mean abs N error | margin Spearman | margin sign agreement |
|---:|---:|---:|---:|---:|---:|---:|
| -10° | 62 | 35.5% | 54.8% | 1.968 | -0.038 | 56.5% |
| 0° | 64 | 43.8% | 65.6% | 1.438 | +0.079 | 71.9% |
| +10° | 64 | 32.8% | 56.3% | 1.906 | +0.027 | 59.4% |

三个坡度合计 TD0：`N exact=71/190=37.4%`，`within-one=112/190=58.9%`。

### 所有 TD0～TD5 合法 touchdown

| slope | samples | N exact | N within-one | mean abs N error | margin Spearman | margin sign agreement |
|---:|---:|---:|---:|---:|---:|---:|
| -10° | 321 | 61.7% | 82.6% | 0.907 | -0.047 | 84.4% |
| 0° | 384 | 72.9% | 85.7% | 0.643 | -0.159 | 89.3% |
| +10° | 344 | 61.9% | 83.1% | 0.826 | -0.248 | 87.8% |

所有 touchdown 的结果好于 TD0，主要因为速度误差在 TD2～TD5 回落；这不能抵消部署时首次 touchdown certificate 失真的问题。

## DCM 误差与 epsilon_b 量级

TD0 的 DCM 水平向量误差 P95 为 `24.4～26.4 cm`。作为量级参考，现有 Plane V1 节点的 `epsilon_b` 为：

| slope | epsilon_b_x | epsilon_b_y |
|---:|---:|---:|
| -10° | 3.16 cm | 4.54 cm |
| 0° | 1.63 cm | 4.76 cm |
| +10° | 1.72 cm | 5.56 cm |

估计速度引起的 TD0 DCM 误差明显大于当前 terminal 区域容差。本诊断没有据此修改 epsilon。

## Gate B terminal ordering

| slope | GT N rho | direct N rho | GT margin rho | direct margin rho |
|---:|---:|---:|---:|---:|
| -10° | +0.048 | 不可定义（direct N 退化为常量） | -0.129 | -0.107 |
| 0° | +0.083 | -0.211 | -0.085 | +0.133 |
| +10° | +0.146 | +0.028 | -0.218 | -0.053 |

direct estimator 没有稳定保留 GT ordering。不过这批小样本的 GT `N_0 vs N_actual_terminal` 本身也很弱，且本次 checkpoint 与只读 nominal 表的 calibration checkpoint 不同，因此这里不把 rho 差值解释成新的理论结论。TD0 的 paired certificate agreement 和 margin consistency 已足以说明直接替换不可接受。

## 最终判断

1. **能否直接迁移 checkpoint 中的 estimator？** 权重和推理链路可以直接迁移；精度上不能直接用于当前 certificate。
2. **估计的是 base 还是 CoM？** 是 body-frame root rigid-body CoM velocity，不是 whole-robot CoM velocity。
3. **nominal 有多准？** 前向 RMSE 约 `0.152～0.201 m/s`，并有 `-0.127～-0.196 m/s` 系统低估。
4. **TD0 有多准？** 水平向量 RMSE 约 `0.617～0.634 m/s`，P95 约 `1.02～1.10 m/s`。
5. **直接把 base 当 CoM 是否可接受？** 对当前 certificate 不可接受；TD0 DCM P95 达 `24～26 cm`。
6. **kinematic correction 是否明显改善？** 尚未实现；GT 对照表明它最多能修复其中较小的结构误差，不能解决主要 estimator 误差。
7. **N/m correlation 是否下降？** TD0 N agreement 和 margin consistency 显著不足；terminal ordering 未被稳定保留，但绝对 rho 因小样本和 nominal checkpoint mismatch 只作诊断参考。
8. **是否需要单独 fine-tune？** 需要。建议后续从现有 encoder + `encode_mean_vel` 初始化独立 estimator，针对 nominal 与 post-push/TD0 分布做 supervised fine-tune，再重复同一 sensitivity test。

## 产物与验证

原始报告：

- `tools/recovery/generated/g1_dwaq_estimator_diagnostic_slope_minus10.yaml`
- `tools/recovery/generated/g1_dwaq_estimator_diagnostic_slope_0.yaml`
- `tools/recovery/generated/g1_dwaq_estimator_diagnostic_slope_plus10.yaml`

代码：

- `legged_lab/recovery/dwaq_estimator_diagnostic.py`
- `tools/recovery/validate_g1_recoverability.py` 的默认关闭 `--estimator_diagnostic` 模式
- `tests/test_dwaq_estimator_diagnostic.py`

验证结果：

- diagnostic helper tests：4 passed
- diagnostic + plane certificate regression：20 passed
- Python compile：passed
- `git diff --check`：passed
- `certificate.py` SHA256 保持 `7fbef67ba3faa4bc6fdaa4d6b0de0262f85cb178d0964afe8d2a385c105234ef`
