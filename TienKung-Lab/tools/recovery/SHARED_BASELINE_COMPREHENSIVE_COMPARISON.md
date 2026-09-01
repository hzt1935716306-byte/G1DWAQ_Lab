# Shared Baseline 0.2 全面固定协议对比

更新时间：2026-08-31（Asia/Shanghai）

## 1. 结论摘要

这次新增测试的是训练时包含三个 shared event rewards、但不包含 certificate reward 的
Baseline 最终模型。测试协议与此前 comprehensive sweep 完全相同。

主要结论：

1. `Baseline-shared-0.2` 的严格五步恢复成功率 P5 为 **94.47%**，高于
   `Baseline-original` 的 **92.23%**。相同 trial 的配对检验显著
   （McNemar exact p=`5.75e-13`）。
2. Shared rewards 的主要作用是减少 TIMEOUT；它没有明显改变成功 episode 的平均
   touchdown 数：3.627 对 3.627，配对差为 -0.005 步，p=0.627。
3. `Baseline-shared-0.2` 与 `Ours-shared+cert-0.2` 相比，P5 高 1.02 个百分点；
   但 Ours 在双方都成功的 trial 上平均少 0.084 步。这表现为“成功率”和“成功后步数”
   之间的权衡，不能只看其中一个指标。
4. `Baseline-shared-0.2` 与新 DWAQ 的总体 P5 分别为 94.47% 和 94.25%，
   差异不显著（p=0.682）。新 DWAQ 在 L1--L3 更强，Shared Baseline 在 L4--L6
   更强；Shared Baseline 的成功恢复时间明显更短。
5. `Ours-cert-only-0.5` 的 P5 为 86.15%，在这组固定协议中明显弱于两个 Baseline。
6. 当前尚未完成的 `Ours-cert-only-0.2` 不在本报告中，不能与历史
   `Ours-shared+cert-0.2` 混为同一个模型。

## 2. 对比模型与训练奖励身份

下表中的 reward 描述的是模型在训练期间收到的 reward。固定协议测试本身只做推理，
不会再把这些 reward 加入策略。

| 报告名称 | 精确 checkpoint | Actor observation dim | Stage2 训练 reward | 用途 |
|---|---|---:|---|---|
| Baseline-original | `logs/g1_flat_symmetric/2026-08-30_02-35-28_stage2_baseline_original_from4999/model_14998.pt` | 960 | 原始 locomotion reward；无 shared、无 certificate | 正确原始 Baseline |
| Baseline-shared-0.2 | `logs/g1_flat_symmetric/2026-08-29_13-15-09_stage2_baseline_scale02_solverfix_resume/model_14999.pt` | 960 | touchdown cost、practical success bonus、timeout penalty，统一 `event_scale=0.2`；无 certificate | 本次新增测试 |
| Ours-shared+cert-0.2 | `logs/g1_flat_symmetric/2026-08-29_13-12-16_stage2_ours_scale02_solverfix_resume/model_14999.pt` | 960 | 三个 shared rewards + certificate，统一 `event_scale=0.2` | 历史混合奖励 Ours |
| Ours-cert-only-0.5 | `logs/g1_flat_symmetric/2026-08-30_02-39-40_stage2_ours_cert050_async_from4999/model_14998.pt` | 960 | 仅 certificate，`event_scale=0.5` | 纯 certificate Ours |
| DWAQ-flat-new | `logs/model_9999.pt` | 100 | 新的纯平地 DWAQ 训练 | 新 DWAQ 参考 |
| DWAQ-old | `logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt` | 100 | 旧 DWAQ 训练 | 历史参考 |

历史 `Ours-shared+cert-0.2` 的训练曾在 `model_7400.pt` 处恢复，恢复时课程状态重新
从 L1 开始。因此它与连续训练的 Shared Baseline 不是严格的单变量 certificate 消融。

## 3. 固定测试协议与完整性

每个模型都使用：

- seed：42、123、2026；
- 课程等级：L1--L6，对应扰动比例 0.25、0.40、0.55、0.70、0.85、1.00；
- 每个等级 256 个 episode；
- 每个 seed 1536 个 episode；每个模型合计 4608 个 episode；
- 每个 seed 内所有模型使用完全相同的 command、扰动和 `trial_id`；
- 8 种 command；
- flat plane；observation noise 关闭；physics randomization 关闭；
- 最大 recovery touchdown 数为 5；
- 最大 recovery 时间为 10 s；
- inference-only，不运行 certificate solver，不计算训练 reward。

完整性检查：

| Seed | 计划/完成/pending | 所有模型 trial hash 一致 | 异常 reset |
|---:|---:|---|---:|
| 42 | 1536 / 1536 / 0 | `9c5af1ca86b10c320e786f3f4187146b75deef33a4e05da21a79599bf6141b2d` | 0 |
| 123 | 1536 / 1536 / 0 | `03edb164b65c72fbaf66d798881136915ae94ac7927dba0eea4a365b86605ccf` | 0 |
| 2026 | 1536 / 1536 / 0 | `0b0735fe6ec5a853a12e22a58e5f00ab808e408c2967d3c3205b3c565b948b24` | 0 |

## 4. Shared Baseline 三个 seed 的直接结果

| Seed | SUCCESS | TIMEOUT | FALL | P5 | 成功平均步数 | 成功步数 median/P75/P90 | 成功平均时间 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1458 | 78 | 0 | 94.92% | 3.626 | 4 / 4 / 5 | 0.766 s |
| 123 | 1440 | 96 | 0 | 93.75% | 3.666 | 4 / 4 / 5 | 0.776 s |
| 2026 | 1455 | 81 | 0 | 94.73% | 3.588 | 4 / 4 / 5 | 0.753 s |
| 合计/均值 | 4353 | 255 | 0 | **94.47%** | **3.627** | **4 / 4 / 5** | **0.765 s** |

P5 的跨 seed population standard deviation 为 0.51 个百分点。

## 5. 所有模型总体结果

步数和时间分位数由三个 seed 的成功 episode 合并后计算。`全 episode 平均时间`
包含 TIMEOUT，因此同时反映失败带来的 10 s 延迟。

| 模型 | SUCCESS/TIMEOUT/FALL | P5 | 成功平均步数 | 步数 median/P75/P90 | 成功平均时间 | 全 episode 平均时间 |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline-shared-0.2** | **4353 / 255 / 0** | **94.47%** | 3.627 | 4 / 4 / 5 | 0.765 s | 1.176 s |
| DWAQ-flat-new | 4343 / 265 / 0 | 94.25% | **3.509** | 4 / 4 / 5 | 1.188 s | 1.204 s |
| Ours-shared+cert-0.2 | 4306 / 302 / 0 | 93.45% | 3.554 | 4 / 4 / 5 | 0.756 s | 1.260 s |
| Baseline-original | 4250 / 358 / 0 | 92.23% | 3.627 | 4 / 4 / 5 | **0.744 s** | 1.238 s |
| Ours-cert-only-0.5 | 3970 / 637 / 1 | 86.15% | 3.780 | 4 / 5 / 5 | 0.801 s | 1.298 s |
| DWAQ-old | 1605 / 3003 / 0 | 34.83% | 3.426 | 3 / 4 / 5 | 1.183 s | 2.022 s |

旧 DWAQ 的成功平均步数看起来较低，是因为只有 34.83% 的 episode 成功；这是条件于
成功样本的选择偏差，不能据此认定旧 DWAQ 恢复更好。

## 6. 分课程等级结果

### 6.1 P5（跨三个 seed 的均值 +/- population SD）

| Level | Baseline-original | Baseline-shared-0.2 | Ours-shared+cert-0.2 | Ours-cert-only-0.5 | DWAQ-flat-new | DWAQ-old |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 88.02 +/- 0.49% | 88.54 +/- 0.49% | 87.89 +/- 0.32% | 87.50 +/- 0.32% | **100.00 +/- 0.00%** | 30.21 +/- 1.92% |
| L2 | 92.06 +/- 0.66% | 94.01 +/- 1.03% | 91.67 +/- 0.66% | 91.67 +/- 0.66% | **99.48 +/- 0.49%** | 33.98 +/- 0.84% |
| L3 | 94.53 +/- 0.84% | 95.83 +/- 1.47% | 95.05 +/- 0.92% | 91.54 +/- 1.03% | **98.05 +/- 0.32%** | 36.46 +/- 0.66% |
| L4 | 95.57 +/- 0.37% | **97.53 +/- 0.66%** | 96.48 +/- 1.10% | 89.19 +/- 0.49% | 94.92 +/- 1.15% | 35.81 +/- 1.47% |
| L5 | 94.27 +/- 2.24% | 96.48 +/- 0.96% | **96.61 +/- 1.03%** | 83.59 +/- 1.15% | 89.71 +/- 2.79% | 36.85 +/- 1.33% |
| L6 | 88.93 +/- 1.21% | **94.40 +/- 0.66%** | 92.97 +/- 1.15% | 73.44 +/- 1.94% | 83.33 +/- 1.92% | 35.68 +/- 1.57% |

### 6.2 成功 episode 的平均 enter step（跨 seed 均值 +/- SD）

| Level | Baseline-original | Baseline-shared-0.2 | Ours-shared+cert-0.2 | Ours-cert-only-0.5 | DWAQ-flat-new | DWAQ-old |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 2.642 +/- 0.030 | 2.594 +/- 0.033 | **2.493 +/- 0.005** | 2.871 +/- 0.030 | 2.672 +/- 0.014 | 2.596 +/- 0.034 |
| L2 | 3.187 +/- 0.034 | 3.204 +/- 0.076 | **3.061 +/- 0.055** | 3.426 +/- 0.067 | 3.178 +/- 0.037 | 3.093 +/- 0.082 |
| L3 | 3.638 +/- 0.059 | 3.613 +/- 0.047 | 3.520 +/- 0.074 | 3.845 +/- 0.077 | 3.587 +/- 0.023 | **3.378 +/- 0.068** |
| L4 | 3.907 +/- 0.015 | 3.911 +/- 0.020 | 3.864 +/- 0.021 | 4.077 +/- 0.042 | 3.772 +/- 0.005 | **3.658 +/- 0.005** |
| L5 | 4.089 +/- 0.063 | 4.090 +/- 0.020 | 4.073 +/- 0.051 | 4.221 +/- 0.045 | 3.916 +/- 0.044 | **3.791 +/- 0.020** |
| L6 | 4.260 +/- 0.081 | 4.264 +/- 0.033 | 4.217 +/- 0.041 | 4.360 +/- 0.056 | 4.077 +/- 0.046 | **3.886 +/- 0.032** |

该表只覆盖成功 episode。尤其不能用 DWAQ-old 的低步数掩盖其约 65% TIMEOUT。

### 6.3 成功 episode 的平均恢复时间（秒，跨 seed 均值 +/- SD）

| Level | Baseline-original | Baseline-shared-0.2 | Ours-shared+cert-0.2 | Ours-cert-only-0.5 | DWAQ-flat-new | DWAQ-old |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 0.549 +/- 0.009 | 0.528 +/- 0.009 | **0.519 +/- 0.005** | 0.609 +/- 0.008 | 0.875 +/- 0.005 | 0.898 +/- 0.031 |
| L2 | 0.665 +/- 0.008 | 0.679 +/- 0.022 | **0.660 +/- 0.025** | 0.730 +/- 0.013 | 1.079 +/- 0.013 | 1.057 +/- 0.038 |
| L3 | **0.750 +/- 0.010** | 0.758 +/- 0.014 | 0.752 +/- 0.016 | 0.811 +/- 0.015 | 1.239 +/- 0.015 | 1.160 +/- 0.040 |
| L4 | **0.797 +/- 0.006** | 0.824 +/- 0.010 | 0.816 +/- 0.007 | 0.861 +/- 0.014 | 1.304 +/- 0.003 | 1.231 +/- 0.028 |
| L5 | **0.829 +/- 0.012** | 0.867 +/- 0.006 | 0.869 +/- 0.013 | 0.899 +/- 0.021 | 1.320 +/- 0.015 | 1.342 +/- 0.043 |
| L6 | **0.865 +/- 0.017** | 0.915 +/- 0.005 | 0.902 +/- 0.017 | 0.926 +/- 0.016 | 1.359 +/- 0.016 | 1.352 +/- 0.014 |

## 7. 分 command 的严格 P5

每个 command 汇总三个 seed，共 576 个 episode，并混合 L1--L6。

| Command `[vx, vy, wz]` | Baseline-original | Baseline-shared-0.2 | Ours-shared+cert-0.2 | Ours-cert-only-0.5 | DWAQ-flat-new | DWAQ-old |
|---|---:|---:|---:|---:|---:|---:|
| `[0.4, 0, 0]` | 99.31% | 98.96% | **99.48%** | 92.19% | 97.05% | 22.40% |
| `[0.8, 0, 0]` | 93.23% | **97.92%** | 96.53% | 85.94% | 95.49% | 35.07% |
| `[-0.3, 0, 0]` | 97.40% | **99.65%** | 98.78% | 95.14% | 89.93% | 81.42% |
| `[0.4, 0.25, 0]` | 98.44% | **99.31%** | 98.78% | 91.84% | 94.97% | 31.42% |
| `[0.4, -0.25, 0]` | 95.14% | 98.78% | **99.48%** | 90.97% | 94.97% | 36.98% |
| `[0.4, 0, 0.5]` | 98.61% | 98.09% | **99.48%** | 90.28% | 93.06% | 27.60% |
| `[0.4, 0, -0.5]` | 98.78% | 98.78% | **99.31%** | 87.15% | 96.88% | 18.06% |
| `[0, 0, 0]` | 56.94% | 64.24% | 55.73% | 55.73% | **91.67%** | 25.69% |

静止 command `[0,0,0]` 是这些 G1 symmetric policies 的共同弱项，也是新 DWAQ 总体
P5 较高的重要来源。Shared Baseline 相对原始 Baseline 在该 command 上提高了 7.29
个百分点，但仍明显低于新 DWAQ。

## 8. 相同 trial 的配对统计

设 A=`Baseline-shared-0.2`。`A only` 表示相同 trial 上只有 Shared Baseline 成功，
`B only` 表示只有对方成功。McNemar 使用 exact two-sided binomial test。

`delta step` 和 `delta time` 均为 A-B，只统计双方都成功的 trial，并使用双侧
Wilcoxon signed-rank test。负数表示 Shared Baseline 步数更少或时间更短。

| B | delta P5 | A only / B only | McNemar p | 双方成功数 | delta step | step p | delta time | time p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline-original | +2.24 pp | 156 / 53 | `5.75e-13` | 4197 | -0.005 | 0.627 | +0.016 s | `1.99e-14` |
| Ours-shared+cert-0.2 | +1.02 pp | 98 / 51 | `1.46e-4` | 4255 | +0.084 | `3.15e-15` | +0.008 s | `9.32e-8` |
| Ours-cert-only-0.5 | +8.31 pp | 419 / 36 | `7.44e-84` | 3934 | -0.218 | `2.31e-66` | -0.053 s | `1.05e-67` |
| DWAQ-flat-new | +0.22 pp | 246 / 236 | 0.682 | 4107 | +0.030 | 0.012 | -0.451 s | `<1e-300` |
| DWAQ-old | +59.64 pp | 2816 / 68 | `<1e-300` | 1537 | +0.235 | `6.70e-22` | -0.387 s | `1.79e-228` |

这些 p 值是探索性配对检验，未做多重比较校正。步数/时间检验还条件于“双方都成功”，
不能替代对总体 P5 的判断。

### Shared Baseline 相对 Baseline-original 的逐等级 P5 变化

| Level | delta P5 | Shared only / Original only | McNemar p |
|---|---:|---:|---:|
| L1 | +0.52 pp | 6 / 2 | 0.289 |
| L2 | +1.95 pp | 19 / 4 | 0.0026 |
| L3 | +1.30 pp | 16 / 6 | 0.0525 |
| L4 | +1.95 pp | 16 / 1 | 0.000275 |
| L5 | +2.21 pp | 31 / 14 | 0.0161 |
| L6 | +5.47 pp | 68 / 26 | `1.73e-5` |

Shared rewards 的提升随扰动等级总体增大，L6 的收益最明显。

## 9. 可以与不可以得出的结论

可以得出：

- 在这套固定协议下，三个 shared event rewards 确实提高了原始 Baseline 的五步恢复
  成功率，主要通过减少 TIMEOUT，而不是减少成功 episode 的 touchdown 数。
- Shared Baseline 与新 DWAQ 的总体 P5 统计上持平，但二者优势区域不同：新 DWAQ
  偏向低等级和静止 command，Shared Baseline 偏向 L4--L6 且恢复时间更短。
- 历史 shared+certificate Ours 的平均成功步数更少，但严格成功率低于 Shared Baseline。

不可以得出：

- 不能把 `Ours-shared+cert-0.2` 的结果解释为纯 certificate 效果，因为它同时使用了
  shared rewards，并经历过不同的 resume/curriculum 历史。
- 不能用成功 episode 的平均步数单独排序策略，因为 TIMEOUT/FALL 被排除在该均值之外。
- 不能把本报告当作噪声、随机物理或真实机器人泛化结论；测试协议明确关闭了这些因素。
- DWAQ 与 symmetric policy 保留各自原生 actor 输入（100 维和 960 维），因此这里只能
  比较相同物理 trial 下的最终策略表现，不能把差异全部归因于某一个 reward。
- 不能把当前未完成的 certificate-only 0.2 训练结果代入本表。

## 10. 原始数据与复现入口

本次新增原始报告：

- [`baseline_shared02_seed42.json`](generated/three_policy_comprehensive_final/baseline_shared02_seed42.json)
- [`baseline_shared02_seed123.json`](generated/three_policy_comprehensive_final/baseline_shared02_seed123.json)
- [`baseline_shared02_seed2026.json`](generated/three_policy_comprehensive_final/baseline_shared02_seed2026.json)

对应运行日志位于同目录的 `baseline_shared02_seed*.log`。

复现脚本：

- [`run_shared_baseline_comprehensive_sweep.sh`](run_shared_baseline_comprehensive_sweep.sh)

既有对照原始数据：

- [`generated/three_policy_comprehensive_final/`](generated/three_policy_comprehensive_final/)
