# G1 可重复评测与实验文档管理框架

以下命令从仓库的 `TienKung-Lab` 子目录执行。框架持久化的路径均相对于该目录，不记录开发机器的绝对路径。
本次交付为 **1.0-dev 开发框架**：已执行离线测试及两个真实 checkpoint 的 CPU 严格加载/前向检查，**没有启动 Isaac Sim、仿真实验或训练**。固定地形、物理推扰和真实恢复检测仍须按任务书 C/D 阶段验收，不能把单元测试当成仿真通过。

## 入口与文件

- `g1_recovery_eval.py`：CLI、只读权重快照、原生推理、评测专用环境、批量 trial 调度。
- `g1_recovery_protocol.py`：固定初态 manifest、精确共同参考、身份、SHA、原子结果与导入。
- `g1_recovery_metrics.py`：独立 GT 裁判、触地去抖、双间隔恢复、统计与配对比较。
- `g1_recovery_report.py`：兼容组、累计曲线、固定 Markdown/Word、历史备份、备注迁移、离线重分析和可选 W&B。
- `configs/g1_recovery_eval_lite_v1.yaml`：唯一开发协议；E0 150 + E1 240 = 390 trial/model。
- `configs/models.example.yaml`：六种 task 的登记示例；缺 checkpoint 保持 `PENDING_CHECKPOINT`。
- `tests/`：纯 CPU 合成契约测试；合成结果禁止导入正式账本。
- `docs/DEVELOPMENT_AUDIT.md`、`docs/BASELINE_IDENTITIES.json`：审计与交付时身份快照。

独立入口不会 import 旧评测脚本。训练环境、奖励、课程、网络和 certificate 数学均未改动。旧 `tools/push_test/ALL_MODELS_PUSH_LIMITS.md` 不参与本协议比较。

## 已登记模型

| Method | Task | 当前状态 |
|---|---|---|
| ppo_plain | g1_slope_nosys_d_matched | 已核对，待评测 |
| ppo_symmetric | g1_slope_sys_d_matched | PENDING_CHECKPOINT |
| dwaq | g1_dwaq_slope_nosys_d_matched | 已核对，待评测 |
| rl_only | g1_plane_v1_rl_only_matched | PENDING_CHECKPOINT |
| context_only | g1_plane_v1_estimator_context_no_reward_matched | PENDING_CHECKPOINT |
| context_reward | g1_plane_v1_estimator_context_reward_matched | PENDING_CHECKPOINT |

两个 privileged task 不接受登记或运行。新模型必须附训练目录 `params/agent.yaml` 和 `params/env.yaml`；不能仅凭孤立权重或文件名确定 task。Plane/RL-only 共用 experiment 名时还核对 run_name、source/reward 和原生维度。

## 离线命令（无需 GPU）

先激活项目使用的 Python/Isaac Lab 环境。基础离线依赖见 `requirements-offline.txt`；Word 由标准 OOXML 生成，不依赖 python-docx。所有默认路径相对代码位置定位，显式相对路径相对当前目录。checkpoint 和 estimator 必须放在 `TienKung-Lab` 目录内。

```bash
cd TienKung-Lab
python tools/evaluation/g1_recovery_eval.py prepare \
  --protocol tools/evaluation/configs/g1_recovery_eval_lite_v1.yaml \
  --output_root experiments/g1_recovery_eval

python tools/evaluation/g1_recovery_eval.py report \
  --output_root experiments/g1_recovery_eval --format md docx

python tools/evaluation/g1_recovery_eval.py detector-validation \
  --output_root experiments/g1_recovery_eval

python -m pytest tools/evaluation/tests -q
```

`prepare` 幂等；初态按 trial 抽样后落盘，运行直接读取。准备后修改协议、manifest 或共同阈值会报错，不会静默重建或覆盖。Standing 不走名义步态恢复检测，报告速度残差与姿态波动。

只登记 checkpoint、不执行仿真：

```bash
python tools/evaluation/g1_recovery_eval.py register \
  --task g1_slope_nosys_d_matched \
  --checkpoint logs/training_run/model.pt \
  --model_alias ppo_plain_seed42_final --checkpoint_stage final \
  --output_root experiments/g1_recovery_eval
```

## 后续运行一个新 checkpoint

以下为后续使用命令，**本次没有执行**。先确认有空闲计算资源；不要停止或修改训练进程。

```bash
python tools/evaluation/g1_recovery_eval.py run \
  --task g1_plane_v1_estimator_context_no_reward_matched \
  --checkpoint logs/training_run/model.pt \
  --estimator_checkpoint checkpoints/com_velocity_estimator_v2_long_best.pt \
  --model_alias context_only_seed42_final --checkpoint_stage final \
  --suite lite --num_envs 32 --headless \
  --output_root experiments/g1_recovery_eval --update_report
```

Baseline/RL-only 不传 `--estimator_checkpoint`，不创建估计器或 certificate worker。PPO 使用原生 ActorCritic；DWAQ 使用原生 encoder/latent/history/Actor；Plane 使用原生冻结估计器和 context 更新链。全部 `eval()` / 无梯度，权重严格加载并检查没有变化。

小规模开发验收可附 `--trial_limit 2`，表示 **E0、E1 各取固定前 2 条**，形成单独 manifest hash，不会伪装成完整 Lite 或并入全量表。完整协议保持 `1.0-dev`，直到人工审查至少 20 条覆盖两个 baseline 的推扰轨迹和无扰动片段。若正常行走大量不达标，停止冻结，不自动扩大阈值。

同身份完整评测默认直接返回 `ALREADY_COMPLETE`。中断只补尚未提交的 trial；已有 trial 不作为额外样本。显式复测使用 `--new_attempt`，旧 attempt 保留在历史中，主表只取一个 attempt。含执行错误的 attempt 不能写完成标记，重试错误 trial 应创建新 attempt。

## 结果、文档与合并

```
experiments/g1_recovery_eval/
  models.yaml                 # 登记和 pending 清单
  registry.json               # 通过完整性检查的结果
  protocol.yaml, prepared.json, metrics_reference.yaml, metrics_nodes.json
  manifests/                  # 150 + 240 条初态，含每格参考脚数量
  checkpoints/<sha>/          # 运行前 CPU 校验的只读权重快照
  runs/<evaluation_id>/
    identity.json, protocol_snapshot.yaml, manifest_snapshot.jsonl
    metrics_reference.yaml, effective_env_config.yaml
    trial_records/, trial_events/, traces/   # 每条提交与完整 50Hz NPZ
    trials.csv, events.csv, summary.json, run.log
    completion.json           # 完整且无执行错误后才创建，逐文件 SHA
  report/
    G1_RECOVERY_EXPERIMENTS.md
    G1_RECOVERY_EXPERIMENTS.docx
    DETECTOR_VALIDATION.md
    notes.yaml, history/, figures/
```

运行结果属于仓库已有 `.gitignore` 的 `experiments/`，需要按结果归档机制备份；代码、配置、测试和审计文档纳入 Git。

第二台机器应复制首台已准备的 `protocol.yaml`、`prepared.json`、`metrics_reference.yaml`、`metrics_nodes.json` 和整个 `manifests/`，直接读取相同初态；不要仅靠相同 seed 假定跨软件版本生成一致。

两机合并：

```bash
python tools/evaluation/g1_recovery_eval.py import-results \
  --source imported_results \
  --output_root experiments/g1_recovery_eval --update_report
```

只接收有完整 SHA 索引的真实结果，按身份与内容去重；同名不覆盖。不同协议、指标、manifest、实际物理参数或推理模式分别成组；PARTIAL/INVALID 单列，不参与排名。同阶段、同 seed 的消融组和同方法上一 checkpoint 缺失时明确标注。恢复时间只比较双方成功的 trial 交集；累计曲线保留失败分母。W&B 可附 `--wandb`，网络失败只写独立上传状态，不破坏已封存结果。

人工分析写到 `report/notes.yaml`，以 model_alias/evaluation_id 关联。每次重建备份旧 MD/Word；检测到 Word/MD 被直接修改时，先备份并把原文迁移到 notes 的 `migrated_document_edits`，避免丢失。

离线重算（原 trial/轨迹保留，新分析放 `runs/<id>/analyses/<hash>/`）：

```bash
python tools/evaluation/g1_recovery_eval.py reanalyze \
  --source experiments/g1_recovery_eval/runs/<evaluation_id> \
  --protocol tools/evaluation/configs/new_metrics_protocol.yaml
```

不允许借重算改变实际坡度、扰动或初态。更改指标/阈值需增加协议或指标版本；当前代码只实现 `practical_interval_confirm2_v1` 检测语义，新检测定义必须先开发对应实现。离线分析保留独立身份，不自动冒充一次新的物理评测。

## 开发版补充约定

- 固定 64×64 m 坡面；spawn 高度为默认 root 高度加本地坡面高度，关节采用 matched scale 分布、原生限位裁剪；实际数值记录在 manifest 与 trace。
- E1 readiness 最多 6 秒；之后等待指定参考脚最多 3 秒，否则 `PRECONDITION_FAILED`。触地力 5 N、释放 3 N、连续 2 帧确认、80 ms 去抖；同脚重复另计诊断，不增加有效落脚数。
- 仅初始观测调用一次 `get_observations()`；每 policy step 断言原生 history 只推进一次。GT 从 reset 前原始物理量读取，不调用观测函数。
- DWAQ 原生 VAE 推理仍随机采样；以 reset_seed、trial 内步号和 `native_dwaq_v1` 固定每次随机流，保留原生结构与采样，避免 batch/续测改变 latent 样本。
- TerrainImporter 用确定列生成 mesh，生成后关闭 curriculum；评测子类绕过 matched 的地形重建构造函数，每次 reset 校验 USD mesh、origin、法向量和 signed slope。
- 物理参数实际快照含资产、质量、惯量、驱动刚度/阻尼、接触材料和仿真步长。记录 context 的有效性，不因 certificate 理论域退出删除物理 trial。
- 冻结版必须另附 `detector_validation.json`：`status: PASSED`、reviewer、至少 20 条 reviewed_traces 和 manifest/metrics/reference/physics 哈希。本框架不会自动把开发测试标成实测验收通过。
