# G1 可重复评测与实验文档管理框架

以下命令从仓库的 `TienKung-Lab` 子目录执行。登记路径相对工程或输出目录；显式资源映射中的相对路径相对映射 YAML。原始训练 YAML 按字节归档，其中历史绝对路径保留为证据，不作为迁移后的运行路径。
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
- `docs/REVIEW_FIXES.md`：E01–E08 修复、NumPy 三版本验证与旧数据处理规则。

独立入口不会 import 旧评测脚本。此次修复增加训练身份/预算记账、修正原生查询诊断分类及 DWAQ 的 Normal 方法误赋值；奖励公式、课程和 certificate 数学保持原有定义。旧 `tools/push_test/ALL_MODELS_PUSH_LIMITS.md` 不参与本协议比较。

## 已登记模型

| Method | Task | 当前状态 |
|---|---|---|
| ppo_plain | g1_slope_nosys_d_matched | 已核对，待评测 |
| ppo_symmetric | g1_slope_sys_d_matched | PENDING_CHECKPOINT |
| dwaq | g1_dwaq_slope_nosys_d_matched | 已核对，待评测 |
| rl_only | g1_plane_v1_rl_only_matched | PENDING_CHECKPOINT |
| context_only | g1_plane_v1_estimator_context_no_reward_matched | PENDING_CHECKPOINT |
| context_reward | g1_plane_v1_estimator_context_reward_matched | PENDING_CHECKPOINT |

两个 privileged task 不接受登记或运行。新模型必须附训练目录 `params/agent.yaml` 和 `params/env.yaml`；不能仅凭孤立权重或文件名确定 task。身份核对显式 task、experiment、source/reward/context、原生维度及完整 strict 权重。`run_name` 只记录为展示证据，任意名称均可。旧配置不足以确定 task 时，要求 SHA 绑定的 `--identity_manifest` 中显式填写 `task_name` 和 `task_confirmed: true`；确认不绕过 source、reward、context 或 strict 权重校验。

## 离线命令（无需 GPU）

先激活项目使用的 Python/Isaac Lab 环境。基础离线依赖见 `requirements-offline.txt`；Word 由标准 OOXML 生成，不依赖 python-docx。所有默认路径相对代码位置定位，显式 CLI 相对路径相对当前目录。允许用户显式提供工程外的 checkpoint、estimator 和资源映射；登记时将其复制为输出目录中的只读内容快照。

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

小规模开发验收可附 `--trial_limit 2`，表示 **E0、E1 各取固定前 2 条**，形成单独 manifest hash。`first_N_per_experiment` 的 COMPLETE 只显示为 `DEV_SUBSET_COMPLETE`，进入 `Development / Smoke Evaluation`，不进入正式 E0/E1 主表、累计曲线、baseline 比较或内部消融。只有 `checkpoint_stage=final`、`subset=lite_full` 且 manifest 与 prepared 中 150 个 E0 + 240 个 E1 逐项完全相同的结果才是 `FULL_EVAL_COMPLETE`。完整协议保持 `1.0-dev`，直到人工审查至少 20 条覆盖两个 baseline 的推扰轨迹和无扰动片段。若正常行走大量不达标，停止冻结，不自动扩大阈值。

同身份完整评测默认直接返回 `ALREADY_COMPLETE`。中断只补尚未提交的 trial；已有 trial 不作为额外样本。显式复测使用 `--new_attempt`，旧 attempt 保留在历史中，主表只取一个 attempt。含执行错误的 attempt 不能写完成标记，重试错误 trial 应创建新 attempt。

## 结果、文档与合并

```
experiments/g1_recovery_eval/
  models.yaml                 # 登记和 pending 清单
  registry.json               # 通过完整性检查的结果
  protocol.yaml, prepared.json, metrics_reference.yaml, metrics_nodes.json
  manifests/                  # 150 + 240 条初态，含每格参考脚数量
  checkpoints/<sha>/<config-hash>/  # 权重与原始 params/，同权重不同配置各自归档
    resources/<sha>/          # 原生 nominal、capability、estimator 分别快照
    native_configuration.json # 原生 solver/context 设置，独立 SHA
  runs/<evaluation_id>/
    identity.json, protocol_snapshot.yaml, manifest_snapshot.jsonl
    metrics_reference.yaml, effective_env_config.yaml, native_configuration.json
    resources/<sha>/          # 原生 nominal/capability 随结果归档，导入时验证
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

只接收有完整 SHA 索引的真实结果，按身份与内容去重；同名不覆盖。不同协议、指标、manifest、实际物理参数、推理模式或 evaluation runtime 分别成组；PARTIAL/INVALID 单列，不参与排名。正式主表只取显式 final、`lite_full` 且逐项匹配 prepared 完整 manifest 的结果；intermediate、unknown 和开发子集在各自章节展示。等预算消融要求 final、相同 seed、已知且相等的累计 transition 数；不同/未知预算明确标注。上一 checkpoint 必须属于同一 `training_run_id`、方法和 seed。恢复时间只比较双方成功的 trial 交集；累计曲线保留失败分母。W&B 可附 `--wandb`，网络失败只写独立上传状态，不破坏已封存结果。

人工分析写到 `report/notes.yaml`，以 model_alias/evaluation_id 关联。每次重建备份旧 MD/Word；Markdown 引用的本地图片复制到该历史版本的 `assets/`，链接随之改写，删除当前 `figures/` 后历史仍独立可读。Word 图片原本即内嵌。检测到 Word/MD 被直接修改时，先备份并把原文迁移到 notes 的 `migrated_document_edits`。

## 跨机器资源与旧 checkpoint 元数据

参考 `configs/identity.example.yaml`。把 checkpoint、原始 `params/`、原生 nominal 和 capability 文件一起迁移；在原始可信文件上记录 SHA，复制后核对内容。不要把训练 YAML 中的旧根目录替换成新根目录，也不按文件名猜测资源。

```bash
sha256sum logs/training_run/model.pt logs/training_run/params/agent.yaml logs/training_run/params/env.yaml
sha256sum tools/recovery/generated/g1_plane_nominal_params_g1_slope_sys_d_candidate.yaml tools/recovery/generated/g1_recovery_params.yaml

python tools/evaluation/g1_recovery_eval.py register \
  --task g1_plane_v1_estimator_context_no_reward_matched \
  --checkpoint logs/training_run/model.pt \
  --estimator_checkpoint checkpoints/com_velocity_estimator_v2_long_best.pt \
  --identity_manifest logs/training_run/evaluation_identity.yaml \
  --model_alias context_only_seed42_final --checkpoint_stage final \
  --output_root experiments/g1_recovery_eval
```

`run` 接受相同的 `--identity_manifest`。映射必须包含 checkpoint/agent/env 三份 SHA，资源按 `native_nominal`、`native_capability` 逻辑角色指定 path 和 SHA。缺文件、SHA 冲突或与 checkpoint 内训练证据冲突都会失败。未记录历史资源 SHA 的旧模型，在原工程直接登记时标记 `observed_at_registration`；这是登记时内容身份，不能证明训练当时使用过该字节内容。迁移时需明确提供已确认资源的 SHA 映射。

运行环境显式绑定快照文件；创建证书 worker 前再次核验实际输入 SHA。registry 默认路径不再决定运行时读哪个 nominal/capability。相关 solver/context 设置与训练快照不同则拒绝。公共裁判 `metrics_reference.yaml` 与策略内部 `native_nominal` 分开保存和校验，不能互换。快照篡改会失败；不同资源内容产生不同评测身份，不能补入原 run。

新的训练 checkpoint 保存 `training_provenance`：明确 task、随机 UUID 训练批次 ID、原始配置及资源 SHA、累计 `training_transitions`。计数在 rollout 完成后增加，独立于日志和 checkpoint 文件名；多 GPU 按全局环境数计数。每次新训练调用（包括从旧模型分支续训）产生新 ID并记录 parent ID；已知累计预算继续计数，旧模型和 warm-start 的未知历史预算保持 null。同一调用保存的多个 checkpoint 共享 ID。

旧 checkpoint 的训练批次/累计预算可在 SHA 绑定清单中补充；相同训练目录的多个 checkpoint 填同一唯一训练 ID，不同重训必须不同。不能只根据 `model_9999.pt`、seed 或展示名推断。两份已提供 baseline 的旧权重不改写，未知训练批次和预算保持未知。

## 统计与诊断字段

- 分位数统一输出 count、median、P90、Q1、Q3、IQR=Q3−Q1。比例输出 numerator、denominator、denominator_name、Wilson 95% 区间。
- `E1.pending = designated - executed`，`failures_executed = executed - successes`；指定分母失败率为已执行失败数 / designated。PARTIAL 的指定分母还有未观察结局，不计算该分母的 Wilson 区间；另给 executed 和 pushed 分母的描述统计，仍不参与正式排名。
- 原生查询在 fallback 前记录 geometry、lookup、adapter、invalid_input、numerical、communication、runtime_exception、constraint_builder 等类别。Standing 和 estimator warmup 不视为求解失败。
- `invalid_context_frames` 是非 standing 的无效持有帧数；`query_failures` 按 query ID 对实际查询事件计数；`numerical_failure_events` 只计真正数值失败。一次失败保持十帧是一次 query failure、十帧 invalid。旧轨迹缺原始分类时显示 N/A，不重新猜测原因。
- 积分兼容 NumPy 1.24 的 `trapz` 与新版本的 `trapezoid`；异常收尾保留错误与轨迹，跳过已失败的指标计算。

离线重算（原 trial/轨迹保留，新分析放 `runs/<id>/analyses/<hash>/`）：

```bash
python tools/evaluation/g1_recovery_eval.py reanalyze \
  --source experiments/g1_recovery_eval/runs/<evaluation_id> \
  --protocol tools/evaluation/configs/new_metrics_protocol.yaml
```

不允许借重算改变实际坡度、扰动或初态。上面的 `reanalyze` 命令保留 `practical_interval_confirm2_v1` 语义；Common Task v2 的独立诊断与回放见下方说明。离线分析保留独立身份，不自动冒充一次新的物理评测。

## V1 开发版补充约定

- 固定 64×64 m 坡面；spawn 高度为默认 root 高度加本地坡面高度，关节采用 matched scale 分布、原生限位裁剪；实际数值记录在 manifest 与 trace。
- E1 readiness 最多 6 秒；之后等待指定参考脚最多 3 秒，否则 `PRECONDITION_FAILED`。触地力 5 N、释放 3 N、连续 2 帧确认、80 ms 去抖；同脚重复另计诊断，不增加有效落脚数。
- 仅初始观测调用一次 `get_observations()`；每 policy step 断言原生 history 只推进一次。GT 从 reset 前原始物理量读取，不调用观测函数。
- DWAQ 原生 VAE 推理仍随机采样；以 reset_seed、trial 内步号和 `native_dwaq_v1` 固定每次随机流，保留原生结构与采样，避免 batch/续测改变 latent 样本。
- TerrainImporter 用确定列生成 mesh，生成后关闭 curriculum；评测子类绕过 matched 的地形重建构造函数，每次 reset 校验 USD mesh、origin、法向量和 signed slope。
- 物理参数实际快照含资产、质量、惯量、驱动刚度/阻尼、接触材料和仿真步长。记录 context 的有效性，不因 certificate 理论域退出删除物理 trial。
- `evaluation_runtime_sha256` 只覆盖显式列出的原生推理、trial 执行、物理判断和恢复指标源文件，并进入 evaluation key 与严格兼容性；`report_code_sha256` 覆盖报告、文档和测试，仅记录 provenance，不触发 390 条物理 trial 重跑。
- 冻结版必须另附 `detector_validation.json`：`status: PASSED`、reviewer、至少 20 条来自两个 baseline 的真实 reviewed trace SHA，以及 protocol、metrics、reference、physics、inference 和 evaluation runtime 身份。每个 source evaluation ID、subset、manifest hash 和实际物理 hash 都会重新打开归档验证；开发子集必须逐项等于当前 prepared full manifest 中 E0/E1 各自确定的前 N 条。1.0-dev 到 1.0 只允许这个精确的协议版本冻结差异。本框架不会自动把开发测试标成实测验收通过。

## Common Task v2 (candidate development)

The independent `common_task_window_v1` implementation and full predeclared
parameters are documented in [COMMON_TASK_V2.md](docs/COMMON_TASK_V2.md).
Use `configs/g1_recovery_eval_lite_v2.yaml` with the separate result root
`experiments/g1_recovery_eval_v2`. V1 remains readable; its teacher-based results
are not directly comparable with v2. `2.0-dev` is **candidate_unvalidated** and
cannot be promoted to `2.0` by passing software tests.

The pinned 16-observation design now has a dedicated `run-development` entry.
See [development execution contract](docs/DEVELOPMENT_VALIDATION_V2.md) for exact
assignment selection, sham-only marker semantics, strict warmup, E0 acceptance,
realized environment identity and the separate development report. These changes
do not execute that validation list or change the candidate parameters.
