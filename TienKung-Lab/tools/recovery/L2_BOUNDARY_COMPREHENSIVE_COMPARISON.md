# L2边界 checkpoint 与无课程/DWAQ：统一六等级恢复测试

## 1. 对比范围

有课程模型取各自训练中仍处于L2的最后一个已保存checkpoint；另外加入两个无课程最终模型和新旧DWAQ作为参考。测试协议与 `ALL_TRAINED_MODEL_COMPREHENSIVE_COMPARISON.md` 相同：seed=42/123/2026，L1--L6，每级每seed 256 episode，每模型共4608 episode。

0.15和0.25只有独立最终权重，缺少L2中间checkpoint，因此未作为L2边界模型纳入。无课程模型和DWAQ没有课程阶段，结果仅作为最终能力参考。

## 2. checkpoint身份

| 模型 | checkpoint | Actor输入 | L2→L3迭代 | 距升级轮数 | SHA256 | 训练身份 |
|---|---|---:|---:|---:|---|---|
| Input-context（L2边界） | `/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab/logs/g1_flat_symmetric/2026-09-02_10-41-16_pilot_input_only_256_seed42/model_3500.pt` | 963 | 3600 | 100 | `4d6f6f186997…` | 963维Actor；真实certificate context；shared events scale=0.5 |
| Baseline-original（L2边界） | `/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab/logs/g1_flat_symmetric/2026-08-30_02-35-28_stage2_baseline_original_from4999/model_7700.pt` | 960 | 2723 | 23 | `b8d37ecc8cb6…` | 960维Actor；原始locomotion reward |
| Baseline-shared-0.2（L2边界） | `/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab/logs/g1_flat_symmetric/2026-08-29_13-15-09_stage2_baseline_scale02_solverfix_resume/model_6400.pt` | 960 | 1425 | 25 | `5b091c402fa0…` | 960维Actor；三个shared events，scale=0.2 |
| Ours-shared+cert-0.2（L2边界） | `/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab/logs/g1_flat_symmetric/2026-08-29_13-12-16_stage2_ours_scale02_solverfix_resume/model_8400.pt` | 960 | 1020 | 20 | `e7eced3f81cd…` | 960维Actor；shared events + certificate reward，scale=0.2 |
| Ours-cert-only-0.20（L2边界） | `/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab/logs/g1_flat_symmetric/2026-08-31_00-20-09_stage2_ours_certonly020_async_from4999/model_7500.pt` | 960 | 2531 | 31 | `b0ed251cf547…` | 960维Actor；certificate-only，scale=0.20 |
| Ours-cert-only-0.50（L2边界） | `/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab/logs/g1_flat_symmetric/2026-08-30_02-39-40_stage2_ours_cert050_async_from4999/model_8500.pt` | 960 | 3600 | 100 | `670eb45ae96c…` | 960维Actor；certificate-only，scale=0.50 |
| Baseline-original（无课程最终） | `/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab/logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt` | 960 | -- | -- | `8557d443d045…` | 960维Actor；无课程；固定L6；原始locomotion reward |
| Ours-cert-only-0.20（无课程最终） | `/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab/logs/g1_flat_symmetric/2026-08-31_23-44-06_stage2_ours_certonly020_no_curriculum_from4999/model_9998.pt` | 960 | -- | -- | `011b1f6a9956…` | 960维Actor；无课程；固定L6；certificate-only，scale=0.20 |
| DWAQ-flat-new | `/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab/logs/model_9999.pt` | 100 | -- | -- | `a90c15e657d1…` | 100维DWAQ Actor；新纯平地策略 |
| DWAQ-old | `/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab/logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt` | 100 | -- | -- | `5042017a558b…` | 100维DWAQ Actor；历史参考策略 |

## 3. 总体结果

| 模型 | SUCCESS/TIMEOUT/FALL | P5 | timeout | fall | 成功步数mean/median | 成功时间mean |
|---|---:|---:|---:|---:|---:|---:|
| DWAQ-flat-new | 4343/265/0 | 94.25% | 5.75% | 0.00% | 3.509/4.0 | 1.188 s |
| Ours-cert-only-0.20（无课程最终） | 4193/415/0 | 90.99% | 9.01% | 0.00% | 3.806/4.0 | 0.791 s |
| Baseline-original（无课程最终） | 4098/510/0 | 88.93% | 11.07% | 0.00% | 3.811/4.0 | 0.760 s |
| Ours-shared+cert-0.2（L2边界） | 3549/996/63 | 77.02% | 21.61% | 1.37% | 3.583/4.0 | 0.740 s |
| Ours-cert-only-0.50（L2边界） | 3528/1019/61 | 76.56% | 22.11% | 1.32% | 3.867/4.0 | 0.791 s |
| Baseline-shared-0.2（L2边界） | 3359/1165/84 | 72.89% | 25.28% | 1.82% | 3.643/4.0 | 0.738 s |
| Ours-cert-only-0.20（L2边界） | 3246/1268/94 | 70.44% | 27.52% | 2.04% | 3.821/4.0 | 0.776 s |
| Baseline-original（L2边界） | 3023/1499/86 | 65.60% | 32.53% | 1.87% | 3.837/4.0 | 0.773 s |
| DWAQ-old | 1605/3003/0 | 34.83% | 65.17% | 0.00% | 3.426/3.0 | 1.183 s |
| Input-context（L2边界） | 560/3860/188 | 12.15% | 83.77% | 4.08% | 3.454/3.0 | 1.296 s |

## 4. 逐等级P5

| Level | Input-context（L2边界） | Baseline-original（L2边界） | Baseline-shared-0.2（L2边界） | Ours-shared+cert-0.2（L2边界） | Ours-cert-only-0.20（L2边界） | Ours-cert-only-0.50（L2边界） | Baseline-original（无课程最终） | Ours-cert-only-0.20（无课程最终） | DWAQ-flat-new | DWAQ-old |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 16.15% | 88.80% | 88.93% | 89.32% | 89.58% | 88.80% | 84.38% | 89.06% | 100.00% | 30.21% |
| L2 | 16.28% | 90.10% | 93.62% | 94.92% | 92.45% | 93.49% | 89.97% | 93.23% | 99.48% | 33.98% |
| L3 | 14.06% | 76.82% | 87.76% | 93.23% | 84.51% | 89.45% | 93.36% | 94.79% | 98.05% | 36.46% |
| L4 | 10.94% | 62.24% | 73.05% | 79.95% | 67.19% | 79.04% | 93.36% | 92.71% | 94.92% | 35.81% |
| L5 | 8.07% | 45.05% | 51.82% | 59.51% | 50.26% | 60.81% | 88.80% | 91.80% | 89.71% | 36.85% |
| L6 | 7.42% | 30.60% | 42.19% | 45.18% | 38.67% | 47.79% | 83.72% | 84.38% | 83.33% | 35.68% |

## 5. Input-context与其他模型的相同trial配对

| 对照模型 | delta P5（Input-context - 对照） | Input-only/对照-only成功 | exact McNemar p |
|---|---:|---:|---:|
| Baseline-original（L2边界） | -53.45 pp | 255/2718 | <1e-300 |
| Baseline-shared-0.2（L2边界） | -60.74 pp | 209/3008 | <1e-300 |
| Ours-shared+cert-0.2（L2边界） | -64.87 pp | 190/3179 | <1e-300 |
| Ours-cert-only-0.20（L2边界） | -58.29 pp | 202/2888 | <1e-300 |
| Ours-cert-only-0.50（L2边界） | -64.41 pp | 218/3186 | <1e-300 |
| Baseline-original（无课程最终） | -76.78 pp | 248/3786 | <1e-300 |
| Ours-cert-only-0.20（无课程最终） | -78.84 pp | 184/3817 | <1e-300 |
| DWAQ-flat-new | -82.10 pp | 15/3798 | <1e-300 |
| DWAQ-old | -22.68 pp | 281/1326 | 3.593e-162 |

## 6. 解释边界

- 所有测试均为inference-only，训练reward关闭。Input-context模型仍运行certificate solver，因为这是其Actor输入的一部分；其余960维模型不运行solver。
- 有课程checkpoint都属于L2，但到达L2边界所经历的训练轮数不同，这是各自自适应课程轨迹的一部分；无课程模型和DWAQ只作最终参考。
- L2边界组回答‘各方法在各自L2结束时的策略能力’，不等于相同训练样本预算的单变量消融；与无课程/DWAQ的差异更不能归因于单一机制。
- 每个seed的trial ID、command和速度跳变完全一致，可进行相同trial配对比较。

原始结果：`/home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab/tools/recovery/generated/l2_boundary_comprehensive_2026-09-02`
