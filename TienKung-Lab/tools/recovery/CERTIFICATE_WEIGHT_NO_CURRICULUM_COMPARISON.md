# Certificate 权重与无课程训练全面对比

更新时间：2026-09-01（Asia/Shanghai）

> 范围说明：本文件只记录 certificate 权重与无课程5模型消融。包含全部11个正式最终模型的总报告见 [`ALL_TRAINED_MODEL_COMPREHENSIVE_COMPARISON.md`](ALL_TRAINED_MODEL_COMPREHENSIVE_COMPARISON.md)。

## 1. 结论摘要

1. 固定协议总体 P5 最高的是 **Ours-0.25（有课程）**：92.40%；三个有课程权重中最高的是 **Ours-0.25（有课程）**（92.40%）。
2. 在有课程权重消融中，0.25 相对 0.20 的 P5 高 2.43 pp，但双方成功 trial 上平均多 0.053 步；0.20 表现为更强的步数效率，0.25 表现为更强的成功率。
3. 在两份无课程模型的直接对照中，Ours-0.20 相对 Baseline 的 P5 差为 +2.06 pp（McNemar p=6.70e-06），双方成功 trial 上的平均 enter-step 差为 +0.009。
4. 有课程 Ours-0.20 相对无课程 Ours-0.20 的 P5 差为 -1.02 pp（p=0.0154）。该比较同时包含 checkpoint iteration/训练历史差异，不能解释为纯课程因果效应。
5. 五个模型均为 0 FALL；差异全部来自能否在五次 touchdown/10 s 内进入严格合格窗口，不是避免倒地能力的差异。
6. 所有 enter-step 均只在 SUCCESS episode 上统计；它必须与 TIMEOUT/FALL/P5 一起看，不能用更低的条件均值掩盖更多失败。

## 2. 模型身份与可比性

下表 reward 指训练期间的 reward；本次评测为 inference-only，不计算训练 reward，也不运行 certificate/LIPM solver。

| 模型 | checkpoint | iter 字段 | 训练课程 | 训练 reward | SHA256 |
|---|---|---:|---|---|---|
| Ours-0.15（有课程） | `logs/our0.15_model_14998_no_sharereware.pt` | 14998 | 有（用户提供的训练身份；独立 checkpoint 无 params） | certificate-only，event_scale=0.15；无 shared rewards | `f4ec25ed4651…` |
| Ours-0.20（有课程） | `logs/g1_flat_symmetric/2026-08-31_21-43-19_stage2_ours_certonly020_resume_L5_from11700/model_14998.pt` | 14998 | 有；resume 时 L5，iterations_in_level=570 | certificate-only，event_scale=0.20；无 shared rewards | `4337ee3144df…` |
| Ours-0.25（有课程） | `logs/our0.25_model_14998_no_sharereward.pt` | 14998 | 有（用户提供的训练身份；独立 checkpoint 无 params） | certificate-only，event_scale=0.25；无 shared rewards | `e3f25452b394…` |
| Ours-0.20（无课程） | `logs/g1_flat_symmetric/2026-08-31_23-44-06_stage2_ours_certonly020_no_curriculum_from4999/model_9998.pt` | 9998 | 无；全程固定完整 L6 随机扰动 | certificate-only，event_scale=0.20；无 shared rewards | `011b1f6a9956…` |
| Baseline（无课程） | `logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt` | 9998 | 无；全程固定完整 L6 随机扰动 | 原始 locomotion reward；无 shared、无 certificate | `8557d443d045…` |

两份独立的 0.15/0.25 checkpoint 不附带对应 `params/env.yaml`；其权重和课程身份来自用户提供的训练记录。网络结构、checkpoint iter 和可加载性已经验证，但无法仅从 `.pt` 反向证明其完整课程轨迹。

两份无课程 checkpoint 的 `iter=9998`，三份有课程 checkpoint 的 `iter=14998`。按用户说明可正常横向评测，但涉及有/无课程的差异必须同时注明训练轮次/历史不完全相同。

## 3. 固定测试协议与完整性

- seed：42、123、2026；L1--L6；每等级每 seed 256 个 episode；
- 每模型 4608 个 episode，五模型共 23040 个；
- 每个 seed 内五个模型共享完全相同的 command、扰动、`trial_id` 和 trial-plan hash；
- 8 种 command；flat plane；关闭 observation noise 和 physics randomization；
- 最大 5 次 recovery touchdown，最大恢复时间 10 s；
- actor 使用各自 checkpoint，但本组均为 symmetric policy，actor observation dim 均为 960。

| Seed | 每模型 planned/completed/pending | trial hash | 五模型异常 reset 合计 |
|---:|---:|---|---:|
| 42 | 1536 / 1536 / 0 | `9c5af1ca86b10c320e786f3f4187146b75deef33a4e05da21a79599bf6141b2d` | 0 |
| 123 | 1536 / 1536 / 0 | `03edb164b65c72fbaf66d798881136915ae94ac7927dba0eea4a365b86605ccf` | 0 |
| 2026 | 1536 / 1536 / 0 | `0b0735fe6ec5a853a12e22a58e5f00ab808e408c2967d3c3205b3c565b948b24` | 0 |

## 4. 总体结果

P5 的 95% CI 为 Wilson interval。步数和时间为三个 seed 合并后的 SUCCESS episode 统计。

| 模型 | SUCCESS / TIMEOUT / FALL | P5 [95% CI] | 成功步数 mean | median/P75/P90 | 成功时间 mean | 全 episode 时间 mean |
|---|---:|---:|---:|---:|---:|---:|
| Ours-0.25（有课程） | 4258 / 350 / 0 | **92.40%** [91.60%, 93.13%] | 3.752 | 4.0 / 4.0 / 5.0 | 0.785 s | 1.195 s |
| Ours-0.20（无课程） | 4193 / 415 / 0 | **90.99%** [90.13%, 91.79%] | 3.806 | 4.0 / 5.0 / 5.0 | 0.791 s | 1.218 s |
| Ours-0.20（有课程） | 4146 / 462 / 0 | **89.97%** [89.07%, 90.81%] | 3.687 | 4.0 / 4.0 / 5.0 | 0.773 s | 1.246 s |
| Baseline（无课程） | 4098 / 510 / 0 | **88.93%** [87.99%, 89.81%] | 3.811 | 4.0 / 5.0 / 5.0 | 0.760 s | 1.265 s |
| Ours-0.15（有课程） | 4047 / 561 / 0 | **87.83%** [86.85%, 88.74%] | 3.820 | 4.0 / 5.0 / 5.0 | 0.785 s | 1.277 s |

### 4.1 跨 seed 稳定性

| Seed | Ours-0.15-C | Ours-0.20-C | Ours-0.25-C | Ours-0.20-NC | Baseline-NC |
|---:|---:|---:|---:|---:|---:|
| 42 | 87.76% | 90.17% | 92.90% | 91.99% | 89.19% |
| 123 | 87.11% | 89.65% | 91.15% | 89.84% | 88.09% |
| 2026 | 88.61% | 90.10% | 93.16% | 91.15% | 89.52% |
| population SD | 0.61 pp | 0.23 pp | 0.90 pp | 0.88 pp | 0.61 pp |

### 4.2 成功 episode 的 enter-step 分布

| 模型 | 1 步 | 2 步 | 3 步 | 4 步 | 5 步 |
|---|---:|---:|---:|---:|---:|
| Ours-0.15（有课程） | 0 | 524 | 812 | 1579 | 1132 |
| Ours-0.20（有课程） | 0 | 571 | 1042 | 1646 | 887 |
| Ours-0.25（有课程） | 0 | 523 | 975 | 1794 | 966 |
| Ours-0.20（无课程） | 0 | 513 | 904 | 1659 | 1117 |
| Baseline（无课程） | 0 | 523 | 792 | 1719 | 1064 |

## 5. 逐等级结果

### 5.1 严格恢复成功率 P5

每格为三个 seed 合并后的成功率。

| Level | Ours-0.15-C | Ours-0.20-C | Ours-0.25-C | Ours-0.20-NC | Baseline-NC |
|---|---:|---:|---:|---:|---:|
| L1 | 86.72% | 88.15% | 88.54% | 89.06% | 84.38% |
| L2 | 90.62% | 92.19% | 94.01% | 93.23% | 89.97% |
| L3 | 93.10% | 95.18% | 95.83% | 94.79% | 93.36% |
| L4 | 89.71% | 93.88% | 95.18% | 92.71% | 93.36% |
| L5 | 86.20% | 88.41% | 93.75% | 91.80% | 88.80% |
| L6 | 80.60% | 82.03% | 87.11% | 84.38% | 83.72% |

### 5.2 成功 episode 平均 enter step

| Level | Ours-0.15-C | Ours-0.20-C | Ours-0.25-C | Ours-0.20-NC | Baseline-NC |
|---|---:|---:|---:|---:|---:|
| L1 | 2.893 | 2.742 | 2.825 | 2.838 | 2.821 |
| L2 | 3.483 | 3.305 | 3.375 | 3.451 | 3.469 |
| L3 | 3.853 | 3.700 | 3.777 | 3.854 | 3.827 |
| L4 | 4.094 | 3.951 | 4.007 | 4.090 | 4.124 |
| L5 | 4.260 | 4.146 | 4.214 | 4.255 | 4.248 |
| L6 | 4.383 | 4.321 | 4.299 | 4.366 | 4.347 |

### 5.3 成功 episode 平均恢复时间

| Level | Ours-0.15-C | Ours-0.20-C | Ours-0.25-C | Ours-0.20-NC | Baseline-NC |
|---|---:|---:|---:|---:|---:|
| L1 | 0.594 s | 0.580 s | 0.597 s | 0.601 s | 0.565 s |
| L2 | 0.725 s | 0.700 s | 0.718 s | 0.731 s | 0.709 s |
| L3 | 0.796 s | 0.780 s | 0.798 s | 0.804 s | 0.766 s |
| L4 | 0.840 s | 0.823 s | 0.834 s | 0.848 s | 0.818 s |
| L5 | 0.874 s | 0.863 s | 0.873 s | 0.871 s | 0.838 s |
| L6 | 0.890 s | 0.899 s | 0.886 s | 0.896 s | 0.856 s |

## 6. 分 command 的严格 P5

每种 command 每模型合计 576 个 episode，并混合 L1--L6。

| Command `[vx, vy, wz]` | Ours-0.15-C | Ours-0.20-C | Ours-0.25-C | Ours-0.20-NC | Baseline-NC |
|---|---:|---:|---:|---:|---:|
| `[0.4, 0, 0]` | 96.01% | 97.57% | 98.44% | 97.74% | 95.83% |
| `[0.8, 0, 0]` | 88.54% | 89.93% | 92.19% | 90.62% | 92.19% |
| `[-0.3, 0, 0]` | 87.67% | 90.97% | 95.49% | 93.06% | 85.94% |
| `[0.4, 0.25, 0]` | 96.18% | 93.75% | 96.88% | 97.22% | 95.49% |
| `[0.4, -0.25, 0]` | 95.31% | 97.74% | 97.74% | 95.14% | 96.35% |
| `[0.4, 0, 0.5]` | 90.10% | 95.31% | 96.88% | 95.83% | 94.10% |
| `[0.4, 0, -0.5]` | 92.01% | 96.70% | 98.09% | 96.01% | 95.14% |
| `[0, 0, 0]` | 56.77% | 57.81% | 63.54% | 62.33% | 56.42% |

## 7. 分归一化扰动强度四分位的 P5

强度定义为 `sqrt(nx^2 + ny^2)`，其中 `nx, ny` 是各等级范围内的归一化扰动。Q1/Q2/Q3 边界为 0.550 / 0.812 / 0.969。

| 强度组 | Ours-0.15-C | Ours-0.20-C | Ours-0.25-C | Ours-0.20-NC | Baseline-NC |
|---|---:|---:|---:|---:|---:|
| Q1 | 88.63% | 89.50% | 89.84% | 90.02% | 86.63% |
| Q2 | 91.84% | 94.53% | 95.83% | 94.44% | 92.36% |
| Q3 | 89.24% | 91.67% | 94.44% | 92.62% | 92.10% |
| Q4 | 81.60% | 84.20% | 89.50% | 86.89% | 84.64% |

## 8. 相同 trial 的全 pair 配对统计

delta 均为 A-B。`A only/B only` 是成功结果不一致的 trial；McNemar 为双侧 exact test，并报告 10 个模型对比的 Holm 校正 p。step/time 只在双方都 SUCCESS 的 trial 上做双侧 Wilcoxon。

| A | B | delta P5 | A only / B only | McNemar p | Holm p | joint success | delta step | step p | delta time | time p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours-0.15-C | Ours-0.20-C | -2.15 pp | 180 / 279 | 4.41e-06 | 2.65e-05 | 3867 | +0.148 | 2.78e-34 | +0.016 s | 3.35e-04 |
| Ours-0.15-C | Ours-0.25-C | -4.58 pp | 117 / 328 | 3.17e-24 | 3.17e-23 | 3930 | +0.090 | 3.14e-13 | +0.005 s | 0.1133 |
| Ours-0.15-C | Ours-0.20-NC | -3.17 pp | 158 / 304 | 1.03e-11 | 8.24e-11 | 3889 | +0.025 | 0.0510 | -0.004 s | 0.5085 |
| Ours-0.15-C | Baseline-NC | -1.11 pp | 220 / 271 | 0.0239 | 0.0479 | 3827 | +0.030 | 0.0224 | +0.030 s | 6.85e-23 |
| Ours-0.20-C | Ours-0.25-C | -2.43 pp | 116 / 228 | 1.57e-09 | 1.10e-08 | 4030 | -0.053 | 2.67e-06 | -0.010 s | 0.1601 |
| Ours-0.20-C | Ours-0.20-NC | -1.02 pp | 157 / 204 | 0.0154 | 0.0461 | 3989 | -0.120 | 1.00e-25 | -0.019 s | 1.40e-05 |
| Ours-0.20-C | Baseline-NC | +1.04 pp | 253 / 205 | 0.0280 | 0.0479 | 3893 | -0.115 | 6.65e-20 | +0.016 s | 4.23e-06 |
| Ours-0.25-C | Ours-0.20-NC | +1.41 pp | 195 / 130 | 3.70e-04 | 0.0015 | 4063 | -0.067 | 9.59e-09 | -0.008 s | 0.0104 |
| Ours-0.25-C | Baseline-NC | +3.47 pp | 279 / 119 | 6.43e-16 | 5.79e-15 | 3979 | -0.055 | 1.22e-05 | +0.026 s | 3.88e-21 |
| Ours-0.20-NC | Baseline-NC | +2.06 pp | 267 / 172 | 6.70e-06 | 3.35e-05 | 3926 | +0.009 | 0.4599 | +0.035 s | 2.41e-41 |

### 8.1 主要假设的逐等级配对结果

这里不做逐等级多重比较校正，因此 p 值是探索性的。

| A | B | Level | delta P5 | A only / B only | McNemar p |
|---|---|---:|---:|---:|---:|
| Ours-0.20-C | Ours-0.15-C | L1 | +1.43 pp | 12 / 1 | 0.0034 |
| Ours-0.20-C | Ours-0.15-C | L2 | +1.56 pp | 24 / 12 | 0.0652 |
| Ours-0.20-C | Ours-0.15-C | L3 | +2.08 pp | 24 / 8 | 0.0070 |
| Ours-0.20-C | Ours-0.15-C | L4 | +4.17 pp | 59 / 27 | 7.32e-04 |
| Ours-0.20-C | Ours-0.15-C | L5 | +2.21 pp | 73 / 56 | 0.1587 |
| Ours-0.20-C | Ours-0.15-C | L6 | +1.43 pp | 87 / 76 | 0.4336 |
| Ours-0.20-C | Ours-0.25-C | L1 | -0.39 pp | 3 / 6 | 0.5078 |
| Ours-0.20-C | Ours-0.25-C | L2 | -1.82 pp | 5 / 19 | 0.0066 |
| Ours-0.20-C | Ours-0.25-C | L3 | -0.65 pp | 8 / 13 | 0.3833 |
| Ours-0.20-C | Ours-0.25-C | L4 | -1.30 pp | 18 / 28 | 0.1839 |
| Ours-0.20-C | Ours-0.25-C | L5 | -5.34 pp | 27 / 68 | 3.11e-05 |
| Ours-0.20-C | Ours-0.25-C | L6 | -5.08 pp | 55 / 94 | 0.0018 |
| Ours-0.20-NC | Baseline-NC | L1 | +4.69 pp | 38 / 2 | 1.49e-09 |
| Ours-0.20-NC | Baseline-NC | L2 | +3.26 pp | 35 / 10 | 2.47e-04 |
| Ours-0.20-NC | Baseline-NC | L3 | +1.43 pp | 25 / 14 | 0.1081 |
| Ours-0.20-NC | Baseline-NC | L4 | -0.65 pp | 28 / 33 | 0.6089 |
| Ours-0.20-NC | Baseline-NC | L5 | +2.99 pp | 62 / 39 | 0.0281 |
| Ours-0.20-NC | Baseline-NC | L6 | +0.65 pp | 79 / 74 | 0.7465 |
| Ours-0.20-C | Ours-0.20-NC | L1 | -0.91 pp | 2 / 9 | 0.0654 |
| Ours-0.20-C | Ours-0.20-NC | L2 | -1.04 pp | 6 / 14 | 0.1153 |
| Ours-0.20-C | Ours-0.20-NC | L3 | +0.39 pp | 16 / 13 | 0.7111 |
| Ours-0.20-C | Ours-0.20-NC | L4 | +1.17 pp | 31 / 22 | 0.2717 |
| Ours-0.20-C | Ours-0.20-NC | L5 | -3.39 pp | 37 / 63 | 0.0120 |
| Ours-0.20-C | Ours-0.20-NC | L6 | -2.34 pp | 65 / 83 | 0.1621 |

## 9. 解释边界与后续价值

### 9.1 这组实验实际说明了什么

- **certificate reward 有可测的正作用，但首先体现在成功率。**无课程 Ours-0.20 比同条件 Baseline 高 2.06 pp，少 95 个 TIMEOUT；但配对 enter-step 只差 +0.009（p=0.4599），没有证据表明它在这组无课程训练中减少踏步。
- 无课程 Ours 相对 Baseline 的增益并非覆盖所有区域：L1/L2/L5 分别为 +4.69 / +3.26 / +2.99 pp，但 L6 仅 +0.65 pp 且不显著。按 command，后退 `[-0.3,0,0]` 和静止 `[0,0,0]` 分别提高 7.12 / 5.90 pp。
- **0.25 最值得作为高扰动鲁棒性方向继续拓展。**它相对有课程 0.20 在 L5/L6 分别高 5.34 / 5.08 pp，强扰动 Q4 高 5.30 pp。
- **0.20 最值得作为少踏步方向保留。**它相对 0.15 同时提高 P5 2.15 pp，并在双方成功 trial 上少 0.148 步；相对 0.25 则少 0.053 步，但牺牲 2.43 pp P5。
- **无课程训练呈现成功率与步数效率的取舍。**无课程 0.20 比有课程 0.20 高 1.02 pp P5，但双方成功时平均多 0.120 步、多 0.019 s；由于训练迭代和历史不同，这只是现象，不是纯课程因果结论。
- **时间效率仍是 Ours 的短板。**无课程 Ours 与 Baseline 的成功 enter-step 基本相同，但成功恢复时间平均多 0.035 s（p=2.41e-41）。certificate 奖励按 touchdown 推动进展，未直接鼓励缩短两次 touchdown 之间的物理时间。
- 所有模型的 1 步成功数均为 0，是因为 practical-good-cycle 指标必须先形成一个完整 touchdown interval 才能判定，不是证明机器人在动力学上绝对不可能一步恢复。

### 9.2 建议优先保留的指标

1. 主指标：总体 P5，并单独报告 L5/L6 和强扰动 Q4 P5；它最能体现 0.25 的价值。
2. 第二主指标：双方成功 trial 的配对 enter-step，而不是各模型各自成功样本的非配对均值；它最能体现 0.20 的步数效率。
3. 必须同时报告 SUCCESS/TIMEOUT/FALL，当前所有差异都来自 TIMEOUT，不能写成防摔提升。
4. 保留成功恢复时间；无课程 Ours 虽提高 P5，但比 Baseline 慢，后续若要真实快速恢复需要单独解决。
5. 分 command 重点看静止和后退命令，分等级重点看 L5/L6；这些区域对 certificate 权重最敏感。

### 9.3 解释边界

不能直接得出：

- 有课程与无课程模型的差异不能全部归因于课程，因为 checkpoint iter 和训练历史不同；
- 0.15/0.25 的 standalone `.pt` 没有 env/agent YAML，课程轨迹身份不能从权重文件独立复核；
- 本测试关闭观测噪声和物理随机化，不能代替 sim-to-real 或随机动力学泛化测试；
- 成功样本中的低 enter-step 不能代替总体 P5，二者必须联合解释；
- 统计 p 值不等于实际效应大小，尤其在 4608 个配对 trial 下，小差异也可能显著。

## 10. 原始数据与复现

原始目录：[`generated/five_policy_no_shared_comprehensive_2026-09-01/`](generated/five_policy_no_shared_comprehensive_2026-09-01/)

每个模型包含 `seed42/123/2026.json` 和对应 Isaac 运行日志。

复现脚本：[`run_five_policy_no_shared_comprehensive_sweep.sh`](run_five_policy_no_shared_comprehensive_sweep.sh)

汇总脚本：[`analyze_five_policy_no_shared_comparison.py`](analyze_five_policy_no_shared_comparison.py)
