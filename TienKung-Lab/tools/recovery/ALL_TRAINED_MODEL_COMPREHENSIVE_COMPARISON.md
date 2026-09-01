# G1 Stage2 全部正式训练版本全面对比

更新时间：2026-09-01（Asia/Shanghai）

## 1. 范围与结论摘要

本报告合并此前6个正式最终版本和本次5个版本，共11个模型。Stage1起点、smoke、未完成运行和每100轮中间 checkpoint 不作为独立方法版本。

1. 总体 P5 第一名是 **Baseline-shared-0.2（有课程）**（94.47%），第二名是 **DWAQ-flat-new**（94.25%）；二者差异不显著（paired McNemar p=0.6819）。
2. pure-certificate 第一名是 **Ours-cert-only-0.25（有课程）**（92.40%）；它与原始有课程 Baseline 相差 +0.17 pp，差异不显著（p=0.6618）。
3. 最干净的 certificate 直接证据来自无课程对照：Ours-0.20 比 Baseline 高 +2.06 pp （p=6.70e-06），但没有减少成功 trial 的踏步数。
4. shared events 对 Baseline 的 P5 提升为 +2.24 pp；这是目前成功率最强的 reward 版本，但它不是纯 certificate 方法。
5. 旧shared+certificate Ours排第三且成功平均步数较少，但它包含三个shared rewards，不能当作pure-certificate结果。
6. 0.25 偏向困难扰动成功率，0.20 偏向成功后的少踏步；0.50 权重明显过大，0.15 也弱于0.20。
7. 新 DWAQ 成功率很高且平均踏步少，但恢复时间明显慢于 symmetric policies；旧 DWAQ 仅作为历史失败参考。

## 2. 全部模型身份

| 模型 | Actor族/输入维度 | 精确 checkpoint | 训练身份 | SHA256 |
|---|---:|---|---|---|
| Baseline-original（有课程） | Symmetric / 960 | `logs/g1_flat_symmetric/2026-08-30_02-35-28_stage2_baseline_original_from4999/model_14998.pt` | 有课程；原始 locomotion；无 shared、无 certificate | `51d28d971321…` |
| Baseline-shared-0.2（有课程） | Symmetric / 960 | `logs/g1_flat_symmetric/2026-08-29_13-15-09_stage2_baseline_scale02_solverfix_resume/model_14999.pt` | 有课程；三个 shared events，scale=0.2；无 certificate | `25d436ba522a…` |
| Baseline-original（无课程） | Symmetric / 960 | `logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt` | 无课程；完整L6随机扰动；原始 locomotion reward | `8557d443d045…` |
| Ours-shared+cert-0.2（有课程） | Symmetric / 960 | `logs/g1_flat_symmetric/2026-08-29_13-12-16_stage2_ours_scale02_solverfix_resume/model_14999.pt` | 有课程；三个 shared events + certificate，scale=0.2；曾从7400恢复并重置课程 | `700bede22ad6…` |
| Ours-cert-only-0.15（有课程） | Symmetric / 960 | `logs/our0.15_model_14998_no_sharereware.pt` | 有课程；certificate-only，scale=0.15（身份来自用户记录） | `f4ec25ed4651…` |
| Ours-cert-only-0.20（有课程） | Symmetric / 960 | `logs/g1_flat_symmetric/2026-08-31_21-43-19_stage2_ours_certonly020_resume_L5_from11700/model_14998.pt` | 有课程；certificate-only，scale=0.20；resume L5/570 | `4337ee3144df…` |
| Ours-cert-only-0.25（有课程） | Symmetric / 960 | `logs/our0.25_model_14998_no_sharereward.pt` | 有课程；certificate-only，scale=0.25（身份来自用户记录） | `e3f25452b394…` |
| Ours-cert-only-0.50（有课程） | Symmetric / 960 | `logs/g1_flat_symmetric/2026-08-30_02-39-40_stage2_ours_cert050_async_from4999/model_14998.pt` | 有课程；certificate-only，scale=0.50 | `aa05dfd0e726…` |
| Ours-cert-only-0.20（无课程） | Symmetric / 960 | `logs/g1_flat_symmetric/2026-08-31_23-44-06_stage2_ours_certonly020_no_curriculum_from4999/model_9998.pt` | 无课程；完整L6随机扰动；certificate-only，scale=0.20 | `011b1f6a9956…` |
| DWAQ-flat-new | DWAQ / 100 | `logs/model_9999.pt` | 新纯平地 DWAQ；无 Stage2 certificate | `a90c15e657d1…` |
| DWAQ-old | DWAQ / 100 | `logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt` | 旧 DWAQ 训练；历史参考 | `5042017a558b…` |

0.15/0.25 的独立 `.pt` 没有配套 params YAML；其训练身份来自用户记录。旧 shared+certificate Ours 在7400恢复时课程曾重置，因此不能作为纯 certificate 因果消融。

## 3. 统一测试协议与完整性

- seed：42、123、2026；L1--L6；每级每 seed 256 episode；
- 每模型4608 episode；11模型合计 **50,688 episode**；
- 每个 seed 内11模型拥有完全相同的 command、扰动、trial ID 和 trial-plan hash；
- flat plane，关闭 observation noise 和 physics randomization；最多5次 touchdown/10 s；
- inference-only；不计算训练 reward，不运行 certificate/LIPM solver；
- DWAQ 保留100维原生 actor 输入，symmetric policies 保留960维原生输入。

| Seed | 每模型 planned/completed/pending | trial hash | 11模型异常reset合计 |
|---:|---:|---|---:|
| 42 | 1536 / 1536 / 0 | `9c5af1ca86b10c320e786f3f4187146b75deef33a4e05da21a79599bf6141b2d` | 0 |
| 123 | 1536 / 1536 / 0 | `03edb164b65c72fbaf66d798881136915ae94ac7927dba0eea4a365b86605ccf` | 0 |
| 2026 | 1536 / 1536 / 0 | `0b0735fe6ec5a853a12e22a58e5f00ab808e408c2967d3c3205b3c565b948b24` | 0 |

## 4. 11模型总体排名

步数/成功时间只统计 SUCCESS；全episode时间包含TIMEOUT。P5 CI为Wilson 95%区间。

| 排名 | 模型 | SUCCESS/TIMEOUT/FALL | P5 [95% CI] | 成功步数 mean | median/P75/P90 | 成功时间 | 全episode时间 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Baseline-shared-0.2（有课程） | 4353/255/0 | **94.47%** [93.77%, 95.09%] | 3.627 | 4.0/4.0/5.0 | 0.765 s | 1.176 s |
| 2 | DWAQ-flat-new | 4343/265/0 | **94.25%** [93.54%, 94.89%] | 3.509 | 4.0/4.0/5.0 | 1.188 s | 1.204 s |
| 3 | Ours-shared+cert-0.2（有课程） | 4306/302/0 | **93.45%** [92.69%, 94.13%] | 3.554 | 4.0/4.0/5.0 | 0.756 s | 1.260 s |
| 4 | Ours-cert-only-0.25（有课程） | 4258/350/0 | **92.40%** [91.60%, 93.13%] | 3.752 | 4.0/4.0/5.0 | 0.785 s | 1.195 s |
| 5 | Baseline-original（有课程） | 4250/358/0 | **92.23%** [91.42%, 92.97%] | 3.627 | 4.0/4.0/5.0 | 0.744 s | 1.238 s |
| 6 | Ours-cert-only-0.20（无课程） | 4193/415/0 | **90.99%** [90.13%, 91.79%] | 3.806 | 4.0/5.0/5.0 | 0.791 s | 1.218 s |
| 7 | Ours-cert-only-0.20（有课程） | 4146/462/0 | **89.97%** [89.07%, 90.81%] | 3.687 | 4.0/4.0/5.0 | 0.773 s | 1.246 s |
| 8 | Baseline-original（无课程） | 4098/510/0 | **88.93%** [87.99%, 89.81%] | 3.811 | 4.0/5.0/5.0 | 0.760 s | 1.265 s |
| 9 | Ours-cert-only-0.15（有课程） | 4047/561/0 | **87.83%** [86.85%, 88.74%] | 3.820 | 4.0/5.0/5.0 | 0.785 s | 1.277 s |
| 10 | Ours-cert-only-0.50（有课程） | 3970/637/1 | **86.15%** [85.13%, 87.12%] | 3.780 | 4.0/5.0/5.0 | 0.801 s | 1.298 s |
| 11 | DWAQ-old | 1605/3003/0 | **34.83%** [33.47%, 36.22%] | 3.426 | 3.0/4.0/5.0 | 1.183 s | 2.022 s |

## 5. 跨 seed P5 稳定性

| 模型 | seed42 | seed123 | seed2026 | population SD |
|---|---:|---:|---:|---:|
| Baseline-shared-0.2（有课程） | 94.92% | 93.75% | 94.73% | 0.51 pp |
| DWAQ-flat-new | 93.49% | 94.86% | 94.40% | 0.57 pp |
| Ours-shared+cert-0.2（有课程） | 94.14% | 92.58% | 93.62% | 0.65 pp |
| Ours-cert-only-0.25（有课程） | 92.90% | 91.15% | 93.16% | 0.90 pp |
| Baseline-original（有课程） | 92.32% | 91.47% | 92.90% | 0.59 pp |
| Ours-cert-only-0.20（无课程） | 91.99% | 89.84% | 91.15% | 0.88 pp |
| Ours-cert-only-0.20（有课程） | 90.17% | 89.65% | 90.10% | 0.23 pp |
| Baseline-original（无课程） | 89.19% | 88.09% | 89.52% | 0.61 pp |
| Ours-cert-only-0.15（有课程） | 87.76% | 87.11% | 88.61% | 0.61 pp |
| Ours-cert-only-0.50（有课程） | 86.91% | 85.68% | 85.87% | 0.54 pp |
| DWAQ-old | 34.96% | 35.42% | 34.11% | 0.54 pp |

## 6. 逐等级严格P5

| Level | B-Orig-C | B-Shared02-C | B-Orig-NC | O-Shared+Cert02-C | O-Cert015-C | O-Cert020-C | O-Cert025-C | O-Cert050-C | O-Cert020-NC | DWAQ-New | DWAQ-Old |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 88.02% | 88.54% | 84.38% | 87.89% | 86.72% | 88.15% | 88.54% | 87.50% | 89.06% | 100.00% | 30.21% |
| L2 | 92.06% | 94.01% | 89.97% | 91.67% | 90.62% | 92.19% | 94.01% | 91.67% | 93.23% | 99.48% | 33.98% |
| L3 | 94.53% | 95.83% | 93.36% | 95.05% | 93.10% | 95.18% | 95.83% | 91.54% | 94.79% | 98.05% | 36.46% |
| L4 | 95.57% | 97.53% | 93.36% | 96.48% | 89.71% | 93.88% | 95.18% | 89.19% | 92.71% | 94.92% | 35.81% |
| L5 | 94.27% | 96.48% | 88.80% | 96.61% | 86.20% | 88.41% | 93.75% | 83.59% | 91.80% | 89.71% | 36.85% |
| L6 | 88.93% | 94.40% | 83.72% | 92.97% | 80.60% | 82.03% | 87.11% | 73.44% | 84.38% | 83.33% | 35.68% |

## 7. 逐等级成功episode平均enter-step

| Level | B-Orig-C | B-Shared02-C | B-Orig-NC | O-Shared+Cert02-C | O-Cert015-C | O-Cert020-C | O-Cert025-C | O-Cert050-C | O-Cert020-NC | DWAQ-New | DWAQ-Old |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 2.642 | 2.594 | 2.821 | 2.493 | 2.893 | 2.742 | 2.825 | 2.871 | 2.838 | 2.672 | 2.595 |
| L2 | 3.187 | 3.204 | 3.469 | 3.061 | 3.483 | 3.305 | 3.375 | 3.426 | 3.451 | 3.178 | 3.092 |
| L3 | 3.638 | 3.613 | 3.827 | 3.519 | 3.853 | 3.700 | 3.777 | 3.845 | 3.854 | 3.587 | 3.379 |
| L4 | 3.907 | 3.911 | 4.124 | 3.864 | 4.094 | 3.951 | 4.007 | 4.077 | 4.090 | 3.772 | 3.658 |
| L5 | 4.087 | 4.090 | 4.248 | 4.073 | 4.260 | 4.146 | 4.214 | 4.221 | 4.255 | 3.917 | 3.792 |
| L6 | 4.259 | 4.263 | 4.347 | 4.217 | 4.383 | 4.321 | 4.299 | 4.360 | 4.366 | 4.077 | 3.887 |

## 8. 逐等级成功episode平均恢复时间

| Level | B-Orig-C | B-Shared02-C | B-Orig-NC | O-Shared+Cert02-C | O-Cert015-C | O-Cert020-C | O-Cert025-C | O-Cert050-C | O-Cert020-NC | DWAQ-New | DWAQ-Old |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 0.549 s | 0.528 s | 0.565 s | 0.519 s | 0.594 s | 0.580 s | 0.597 s | 0.609 s | 0.601 s | 0.875 s | 0.899 s |
| L2 | 0.665 s | 0.679 s | 0.709 s | 0.660 s | 0.725 s | 0.700 s | 0.718 s | 0.730 s | 0.731 s | 1.079 s | 1.057 s |
| L3 | 0.750 s | 0.758 s | 0.766 s | 0.751 s | 0.796 s | 0.780 s | 0.798 s | 0.811 s | 0.804 s | 1.239 s | 1.160 s |
| L4 | 0.797 s | 0.824 s | 0.818 s | 0.816 s | 0.840 s | 0.823 s | 0.834 s | 0.861 s | 0.848 s | 1.304 s | 1.232 s |
| L5 | 0.828 s | 0.867 s | 0.838 s | 0.869 s | 0.874 s | 0.863 s | 0.873 s | 0.899 s | 0.871 s | 1.320 s | 1.342 s |
| L6 | 0.865 s | 0.915 s | 0.856 s | 0.902 s | 0.890 s | 0.899 s | 0.886 s | 0.926 s | 0.896 s | 1.359 s | 1.352 s |

## 9. 分command严格P5

| Command | B-Orig-C | B-Shared02-C | B-Orig-NC | O-Shared+Cert02-C | O-Cert015-C | O-Cert020-C | O-Cert025-C | O-Cert050-C | O-Cert020-NC | DWAQ-New | DWAQ-Old |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `[0.4, 0, 0]` | 99.31% | 98.96% | 95.83% | 99.48% | 96.01% | 97.57% | 98.44% | 92.19% | 97.74% | 97.05% | 22.40% |
| `[0.8, 0, 0]` | 93.23% | 97.92% | 92.19% | 96.53% | 88.54% | 89.93% | 92.19% | 85.94% | 90.62% | 95.49% | 35.07% |
| `[-0.3, 0, 0]` | 97.40% | 99.65% | 85.94% | 98.78% | 87.67% | 90.97% | 95.49% | 95.14% | 93.06% | 89.93% | 81.42% |
| `[0.4, 0.25, 0]` | 98.44% | 99.31% | 95.49% | 98.78% | 96.18% | 93.75% | 96.88% | 91.84% | 97.22% | 94.97% | 31.42% |
| `[0.4, -0.25, 0]` | 95.14% | 98.78% | 96.35% | 99.48% | 95.31% | 97.74% | 97.74% | 90.97% | 95.14% | 94.97% | 36.98% |
| `[0.4, 0, 0.5]` | 98.61% | 98.09% | 94.10% | 99.48% | 90.10% | 95.31% | 96.88% | 90.28% | 95.83% | 93.06% | 27.60% |
| `[0.4, 0, -0.5]` | 98.78% | 98.78% | 95.14% | 99.31% | 92.01% | 96.70% | 98.09% | 87.15% | 96.01% | 96.88% | 18.06% |
| `[0, 0, 0]` | 56.94% | 64.24% | 56.42% | 55.73% | 56.77% | 57.81% | 63.54% | 55.73% | 62.33% | 91.67% | 25.69% |

## 10. 分归一化扰动强度四分位P5

边界：0.550/0.812/0.969。

| 强度 | B-Orig-C | B-Shared02-C | B-Orig-NC | O-Shared+Cert02-C | O-Cert015-C | O-Cert020-C | O-Cert025-C | O-Cert050-C | O-Cert020-NC | DWAQ-New | DWAQ-Old |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 88.63% | 90.02% | 86.63% | 89.24% | 88.63% | 89.50% | 89.84% | 88.37% | 90.02% | 99.83% | 27.17% |
| Q2 | 95.49% | 96.09% | 92.36% | 94.79% | 91.84% | 94.53% | 95.83% | 91.06% | 94.44% | 97.31% | 37.07% |
| Q3 | 94.70% | 97.14% | 92.10% | 96.35% | 89.24% | 91.67% | 94.44% | 86.11% | 92.62% | 92.88% | 39.06% |
| Q4 | 90.10% | 94.62% | 84.64% | 93.40% | 81.60% | 84.20% | 89.50% | 79.08% | 86.89% | 86.98% | 36.02% |

## 11. 成功episode的enter-step分布

| 模型 | 1步 | 2步 | 3步 | 4步 | 5步 |
|---|---:|---:|---:|---:|---:|
| Baseline-shared-0.2（有课程） | 0 | 641 | 1122 | 1811 | 779 |
| DWAQ-flat-new | 0 | 710 | 1352 | 1642 | 639 |
| Ours-shared+cert-0.2（有课程） | 0 | 755 | 1109 | 1744 | 698 |
| Ours-cert-only-0.25（有课程） | 0 | 523 | 975 | 1794 | 966 |
| Baseline-original（有课程） | 0 | 639 | 1098 | 1721 | 792 |
| Ours-cert-only-0.20（无课程） | 0 | 513 | 904 | 1659 | 1117 |
| Ours-cert-only-0.20（有课程） | 0 | 571 | 1042 | 1646 | 887 |
| Baseline-original（无课程） | 0 | 523 | 792 | 1719 | 1064 |
| Ours-cert-only-0.15（有课程） | 0 | 524 | 812 | 1579 | 1132 |
| Ours-cert-only-0.50（有课程） | 0 | 528 | 835 | 1590 | 1017 |
| DWAQ-old | 0 | 275 | 584 | 533 | 213 |

## 12. 主要假设的相同trial配对检验

delta均为A-B。McNemar为双侧exact；step/time为双方都成功trial上的双侧Wilcoxon。

| 对比 | A | B | delta P5 | A only/B only | McNemar p | joint | delta step | step p | delta time | time p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared events 对原始 Baseline | B-Shared02-C | B-Orig-C | +2.24 pp | 156/53 | 5.75e-13 | 4197 | -0.005 | 0.6273 | +0.016 s | 1.99e-14 |
| 旧混合 Ours 对 shared Baseline | O-Shared+Cert02-C | B-Shared02-C | -1.02 pp | 51/98 | 1.46e-04 | 4255 | -0.084 | 3.15e-15 | -0.008 s | 9.32e-08 |
| 最佳 pure-certificate 对原始 Baseline | O-Cert025-C | B-Orig-C | +0.17 pp | 132/124 | 0.6618 | 4126 | +0.141 | 2.76e-36 | +0.043 s | 1.87e-67 |
| 无课程 pure-certificate 直接消融 | O-Cert020-NC | B-Orig-NC | +2.06 pp | 267/172 | 6.70e-06 | 3926 | +0.009 | 0.4599 | +0.035 s | 2.41e-41 |
| 0.20 对 0.15 | O-Cert020-C | O-Cert015-C | +2.15 pp | 279/180 | 4.41e-06 | 3867 | -0.148 | 2.78e-34 | -0.016 s | 3.35e-04 |
| 0.25 对 0.20 | O-Cert025-C | O-Cert020-C | +2.43 pp | 228/116 | 1.57e-09 | 4030 | +0.053 | 2.67e-06 | +0.010 s | 0.1601 |
| 0.20 对 0.50 | O-Cert020-C | O-Cert050-C | +3.82 pp | 344/168 | 5.66e-15 | 3802 | -0.122 | 1.70e-21 | -0.034 s | 3.30e-34 |
| 有课程/无课程 0.20（有混杂） | O-Cert020-C | O-Cert020-NC | -1.02 pp | 157/204 | 0.0154 | 3989 | -0.120 | 1.00e-25 | -0.019 s | 1.40e-05 |
| 最佳 Baseline 对新 DWAQ | B-Shared02-C | DWAQ-New | +0.22 pp | 246/236 | 0.6819 | 4107 | +0.030 | 0.0123 | -0.451 s | <1e-300 |
| 最佳 pure-certificate 对新 DWAQ | O-Cert025-C | DWAQ-New | -1.84 pp | 225/310 | 2.74e-04 | 4033 | +0.178 | 6.47e-39 | -0.422 s | <1e-300 |

## 13. 全55组模型对的P5配对附录

Holm p对本节55个McNemar检验进行校正。完整step/time配对结果保留在汇总脚本中，主假设见上一节。

| A | B | delta P5 | A only/B only | raw p | Holm p |
|---|---|---:|---:|---:|---:|
| B-Orig-C | B-Shared02-C | -2.24 pp | 53/156 | 5.75e-13 | 1.38e-11 |
| B-Orig-C | B-Orig-NC | +3.30 pp | 258/106 | 9.45e-16 | 2.46e-14 |
| B-Orig-C | O-Shared+Cert02-C | -1.22 pp | 71/127 | 8.39e-05 | 0.0012 |
| B-Orig-C | O-Cert015-C | +4.41 pp | 305/102 | 1.35e-24 | 4.60e-23 |
| B-Orig-C | O-Cert020-C | +2.26 pp | 209/105 | 4.48e-09 | 8.07e-08 |
| B-Orig-C | O-Cert025-C | -0.17 pp | 124/132 | 0.6618 | 1.0000 |
| B-Orig-C | O-Cert050-C | +6.08 pp | 380/100 | 1.85e-39 | 7.03e-38 |
| B-Orig-C | O-Cert020-NC | +1.24 pp | 189/132 | 0.0017 | 0.0137 |
| B-Orig-C | DWAQ-New | -2.02 pp | 238/331 | 1.11e-04 | 0.0014 |
| B-Orig-C | DWAQ-Old | +57.40 pp | 2749/104 | <1e-300 | <1e-300 |
| B-Shared02-C | B-Orig-NC | +5.53 pp | 296/41 | 8.41e-49 | 3.53e-47 |
| B-Shared02-C | O-Shared+Cert02-C | +1.02 pp | 98/51 | 1.46e-04 | 0.0018 |
| B-Shared02-C | O-Cert015-C | +6.64 pp | 349/43 | 1.11e-60 | 4.77e-59 |
| B-Shared02-C | O-Cert020-C | +4.49 pp | 246/39 | 6.72e-38 | 2.49e-36 |
| B-Shared02-C | O-Cert025-C | +2.06 pp | 157/62 | 1.09e-10 | 2.29e-09 |
| B-Shared02-C | O-Cert050-C | +8.31 pp | 419/36 | 7.44e-84 | 3.35e-82 |
| B-Shared02-C | O-Cert020-NC | +3.47 pp | 212/52 | 4.24e-24 | 1.36e-22 |
| B-Shared02-C | DWAQ-New | +0.22 pp | 246/236 | 0.6819 | 1.0000 |
| B-Shared02-C | DWAQ-Old | +59.64 pp | 2816/68 | <1e-300 | <1e-300 |
| B-Orig-NC | O-Shared+Cert02-C | -4.51 pp | 72/280 | 4.35e-30 | 1.57e-28 |
| B-Orig-NC | O-Cert015-C | +1.11 pp | 271/220 | 0.0239 | 0.1197 |
| B-Orig-NC | O-Cert020-C | -1.04 pp | 205/253 | 0.0280 | 0.1197 |
| B-Orig-NC | O-Cert025-C | -3.47 pp | 119/279 | 6.43e-16 | 1.79e-14 |
| B-Orig-NC | O-Cert050-C | +2.78 pp | 339/211 | 5.36e-08 | 9.11e-07 |
| B-Orig-NC | O-Cert020-NC | -2.06 pp | 172/267 | 6.70e-06 | 1.01e-04 |
| B-Orig-NC | DWAQ-New | -5.32 pp | 214/459 | 1.86e-21 | 5.58e-20 |
| B-Orig-NC | DWAQ-Old | +54.10 pp | 2698/205 | <1e-300 | <1e-300 |
| O-Shared+Cert02-C | O-Cert015-C | +5.62 pp | 326/67 | 5.80e-42 | 2.32e-40 |
| O-Shared+Cert02-C | O-Cert020-C | +3.47 pp | 231/71 | 6.43e-21 | 1.87e-19 |
| O-Shared+Cert02-C | O-Cert025-C | +1.04 pp | 137/89 | 0.0017 | 0.0137 |
| O-Shared+Cert02-C | O-Cert050-C | +7.29 pp | 396/60 | 9.03e-62 | 3.97e-60 |
| O-Shared+Cert02-C | O-Cert020-NC | +2.45 pp | 200/87 | 2.11e-11 | 4.65e-10 |
| O-Shared+Cert02-C | DWAQ-New | -0.80 pp | 247/284 | 0.1181 | 0.3544 |
| O-Shared+Cert02-C | DWAQ-Old | +58.62 pp | 2787/86 | <1e-300 | <1e-300 |
| O-Cert015-C | O-Cert020-C | -2.15 pp | 180/279 | 4.41e-06 | 7.06e-05 |
| O-Cert015-C | O-Cert025-C | -4.58 pp | 117/328 | 3.17e-24 | 1.05e-22 |
| O-Cert015-C | O-Cert050-C | +1.67 pp | 323/246 | 0.0014 | 0.0128 |
| O-Cert015-C | O-Cert020-NC | -3.17 pp | 158/304 | 1.03e-11 | 2.37e-10 |
| O-Cert015-C | DWAQ-New | -6.42 pp | 207/503 | 2.60e-29 | 9.09e-28 |
| O-Cert015-C | DWAQ-Old | +52.99 pp | 2639/197 | <1e-300 | <1e-300 |
| O-Cert020-C | O-Cert025-C | -2.43 pp | 116/228 | 1.57e-09 | 2.98e-08 |
| O-Cert020-C | O-Cert050-C | +3.82 pp | 344/168 | 5.66e-15 | 1.41e-13 |
| O-Cert020-C | O-Cert020-NC | -1.02 pp | 157/204 | 0.0154 | 0.0922 |
| O-Cert020-C | DWAQ-New | -4.28 pp | 201/398 | 6.39e-16 | 1.79e-14 |
| O-Cert020-C | DWAQ-Old | +55.14 pp | 2700/159 | <1e-300 | <1e-300 |
| O-Cert025-C | O-Cert050-C | +6.25 pp | 391/103 | 1.61e-40 | 6.28e-39 |
| O-Cert025-C | O-Cert020-NC | +1.41 pp | 195/130 | 3.70e-04 | 0.0037 |
| O-Cert025-C | DWAQ-New | -1.84 pp | 225/310 | 2.74e-04 | 0.0030 |
| O-Cert025-C | DWAQ-Old | +57.57 pp | 2752/99 | <1e-300 | <1e-300 |
| O-Cert050-C | O-Cert020-NC | -4.84 pp | 141/364 | 9.33e-24 | 2.89e-22 |
| O-Cert050-C | DWAQ-New | -8.09 pp | 191/564 | 1.50e-43 | 6.16e-42 |
| O-Cert050-C | DWAQ-Old | +51.32 pp | 2582/217 | <1e-300 | <1e-300 |
| O-Cert020-NC | DWAQ-New | -3.26 pp | 215/365 | 4.94e-10 | 9.89e-09 |
| O-Cert020-NC | DWAQ-Old | +56.16 pp | 2711/123 | <1e-300 | <1e-300 |
| DWAQ-New | DWAQ-Old | +59.42 pp | 2841/103 | <1e-300 | <1e-300 |

## 14. 总体解释

### 14.1 Reward结论

- 三个shared events对严格P5的提升最稳定：shared Baseline是总体第一，但成功步数并未改善。
- 旧shared+certificate Ours的P5低于shared Baseline，但双方成功时踏步更少；由于它同时含shared reward且课程恢复历史不同，不能代表纯certificate收益。
- pure-certificate存在合理权重区间：0.15偏弱，0.20更强调少踏步，0.25更强调L5/L6/Q4成功率，0.50出现明显性能退化。
- 无课程Ours/Baseline是最干净的certificate消融：certificate提高P5，但不减少成功trial踏步，并让恢复时间稍慢。

### 14.2 哪些方向值得继续

1. 若论文主张是困难扰动鲁棒性，优先扩展0.25，并主报L5/L6、Q4和TIMEOUT下降。
2. 若主张是更少恢复踏步，保留0.20，并使用双方成功trial的配对enter-step；不要只报各自成功样本均值。
3. 若希望得到最强总体P5，shared reward仍是当前最强工程方案，但它不能证明certificate理论本身有效。
4. 新DWAQ应作为强参考基线：成功率接近第一且步数少，但其100维输入和较慢恢复时间必须单独说明。
5. 后续应补一组相同训练预算、相同课程轨迹、多个训练seed的Baseline与pure-certificate0.25，才能做最干净的因果结论。

### 14.3 解释限制

- 有/无课程模型训练轮次和历史不同；不能把差异全部归因于课程。
- 0.15/0.25缺少params YAML；身份依赖训练记录。
- 固定协议关闭噪声和物理随机化，不是sim-to-real结论。
- DWAQ与symmetric actor输入结构不同，只能比较最终物理trial表现。
- 1步成功为0与practical-good-cycle判据需要完整touchdown interval有关，不能解释为动力学绝对不可能。

## 15. 原始数据与复现

历史6模型：[`generated/three_policy_comprehensive_final/`](generated/three_policy_comprehensive_final/)

本次5模型：[`generated/five_policy_no_shared_comprehensive_2026-09-01/`](generated/five_policy_no_shared_comprehensive_2026-09-01/)

统一汇总脚本：[`analyze_all_trained_models_comparison.py`](analyze_all_trained_models_comparison.py)
