# E01–E08 修复记录（2026-09-07）

基于 `feature/g1-recovery-eval-portable` 的框架修复。没有启动 Isaac Sim、物理评测或训练；没有改写已训练的两个 baseline、原始训练 YAML、历史实验结果或公共裁判阈值。协议继续为 `1.0-dev`，真实检测器验收仍为 NOT_RUN。

| 项目 | 实现与离线证据 |
|---|---|
| E01 | task 元数据、experiment、source/reward/context、网络维度与 native strict 加载共同确认身份；run_name 仅展示。合成 RL-only/context-only/context-reward 权重改名均可加载；错误来源、开关、context、形状、task 和缺参数分别拒绝。旧证据不足要求 SHA 绑定清单显式确认。 |
| E02 | 逻辑资源角色映射必须给 path 与 SHA；工程外输入只能来自显式 CLI/映射。跨目录移动后保留 checkpoint/原始配置字节；资源通过 SHA 验证。不存在路径、错误 SHA 和训练证据冲突均失败，不做 basename/旧根猜测。 |
| E03 | nominal、capability、estimator 独立内容快照；solver/context 设置独立 JSON/SHA；构建原生环境前绑定快照，创建 worker 的工厂入口再次校验。A/B 路径测试确认实际读 A，强制重新绑定 B 被拒绝。有效配置声明、归档资源、重复快照和 snapshot 篡改均有回归测试。公共 metrics_reference 独立保留。 |
| E04 | 主表仅 final，并展示 transitions；intermediate/unknown 在演进章节。新训练 checkpoint 保存明确 task、UUID training_run_id、累计 transitions 与原始配置/资源 SHA；计数不依赖日志。新调用/分支续训新建 ID并记录 parent，累计预算已知则延续，旧预算未知保持 null。等预算消融必须 final、同 seed、正整数预算相等；前后 checkpoint 必须同一训练 ID。两类 runner 保存/恢复元数据的 CPU 测试通过。 |
| E05 | 延迟选择 trapezoid/trapz，避免 fallback 提前访问缺失属性。错误收尾使用 result(include_metrics=False)，避免重复执行已失败指标。真实 NumPy 1.24.4、1.26.0、2.4.0 分别检查积分、轨迹指标、分位数和区间。 |
| E06 | count/median/P90/Q1/Q3/IQR；各比例显式分子、分母名称与 Wilson 95%。pending、已执行失败单列。PARTIAL 指定分母的未观察结局不算失败，也不伪造该分母的置信区间；已执行样本另有描述区间，始终不参与排名。 |
| E07 | 重建前备份 Markdown 引用图片至该历史版本 assets/，重写链接；多次更新后删除当前 figures，历史 Markdown 的 PNG 仍完整可读。DOCX 保持图片内嵌。 |
| E08 | 在原生 fallback 之前传递 geometry/lookup/adapter/数值/通信/运行异常等真实类别；原生计数只把 numerical 计入 solver failure。实际查询事件与 ZOH 无效持有帧分开。CPU 原生 refresh 测试和十帧轨迹测试确认一次 query failure、十帧 invalid；standing/lookup 不计数值失败。 |

联跑另外发现 DWAQ 的 `Normal.set_default_validate_args = False` 会覆盖 PyTorch 方法，使同进程后续 PPO/Plane 审计失败。已改为方法调用，网络结构、state_dict 和原生推理路径不变；PPO/DWAQ 两份现有权重继续通过 CPU strict 加载和前向检查。

## 可重复验证

评测回归与相关原生契约测试合计 **135 项通过**；另外通过三个真实 NumPy 版本的独立兼容脚本、修改文件语法检查和 `git diff --check`。CPU stub、合成权重及 LP 单元测试均不替代真实 USD/PhysX 验收。

从 `TienKung-Lab` 执行：

```bash
OMP_NUM_THREADS=1 python -m pytest -q tools/evaluation/tests \
  tests/test_plane_certificate_runtime.py tests/test_plane_v1.py \
  tests/test_plane_v1_rl_only_contract.py tests/test_plane_baseline_matched_protocol.py
```

NumPy 独立兼容检查（在待验证 NumPy 环境中运行，不需要 Torch、Isaac 或 matplotlib）：

```bash
python tools/evaluation/tests/numpy_compat_check.py
```

| NumPy | trapz | trapezoid | 恒定速度误差积分（2 秒 × 0.2 m/s） |
|---|---|---|---|
| 1.24.4 | 存在 | 不存在 | 0.3999999999999999，PASS |
| 1.26.0 | 存在 | 不存在 | 0.3999999999999999，PASS |
| 2.4.0 | 不存在 | 存在 | 0.3999999999999999，PASS |

NumPy 1.24.4 与 2.4.0 使用官方 PyPI wheel、校验官方 SHA 后安装在临时隔离目录。现有训练环境仍为 NumPy 1.26.0。这是指标模块兼容验证，不表示 Isaac Sim 的整套依赖支持任意 NumPy 升级。

## 旧数据边界

`identity_schema_version=2` 与 `summary_schema_version=2` 标记新的资源绑定/统计格式；检测器阈值和 `metrics_version` 未改变。旧 Plane 结果没有实际原生输入绑定证据时从正式表排除，需重新评测；缺 query 分类的旧轨迹诊断记 N/A。历史文件不自动修补成新证据。

旧模型没有训练批次/预算时保持未知，不从展示名、seed 或 checkpoint 文件名推断。同工程旧资源首次登记的 SHA 标记为 observed_at_registration；它只证明登记时的字节内容，不能倒推历史训练输入。迁移或补充身份使用 `configs/identity.example.yaml` 和 `--identity_manifest`。

新任务身份、资源和预算均进入评测身份；原生环境、求解器、网络/runner 与评测源代码也纳入代码内容 SHA，代码或资源更改不能静默续入旧 run。
