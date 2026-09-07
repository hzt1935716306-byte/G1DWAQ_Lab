# G1 Recovery Eval 开发审计与验收边界

日期：2026-09-06。仓库根目录为当前 Git checkout；开发基准 `d593b403a97d734aceba9e9646b34e84c232da12`。

本文件保留初次交付的审计记录。2026-09-07 的 E01–E08 修复、训练记账改动和新增验收记录见 [REVIEW_FIXES.md](REVIEW_FIXES.md)；下文“训练路径没有修改”和初版测试数量仅描述 2026-09-06 的交付状态。

用户明确要求先搭完整框架、不跑实验。本次未启动 Isaac Sim 或训练；没有停止或修改训练进程。初始工作区存在未跟踪 `TienKung-Lab/tb_compare_runs/`，保持原状。

## 阶段 A：审计与原生加载链

| 审计来源 | 发现与处理 |
|---|---|
| tools/recovery/validate_g1_recoverability.py | 包含已有加载、推扰、Plane 验证逻辑；顶层启动仿真，不 import。只参考成熟路径。 |
| tools/push_test/evaluate_external_push.py | 旧 standing 外力协议，终端数据有 post-reset 风险。新协议独立状态机，不复用其物理成功率。 |
| tools/push_test/summarize_push_test.py | 参考结构化汇总做法，旧结果不混排。 |
| legged_lab/recovery/practical_metrics.py | 按逐帧速度误差范数和逐帧绝对姿态误差再取间隔平均。CPU 测试直接对照现有实现。 |
| legged_lab/recovery/state_extractor.py | 复用实际质量加权 whole-body CoM 计算、heading frame 约定；指标从状态读取，不推进 Actor history。 |
| legged_lab/recovery/plane_nominal_params.py | 原生证书保持原表；共同裁判独立读取候选表精确 slope/direction/speed 节点，不插值/最近邻/clamp。 |
| g1_plane_v1_env.py / g1_plane_v1_matched_env.py | 保留冻结 estimator → LIPM/DCM/context → Actor 原生更新。绕过仅负责重建地形的 matched 构造函数，使用评测子类和 matched task 的原生配置。 |
| g1_slope_matched_env.py / g1_matched_command.py | matched 构造函数会覆盖传入 terrain；评测直接派生 BaseEnv / G1DwaqEnv / G1PlaneV1Env，命令工厂固定 manifest 命令。 |
| legged_lab/scripts/play.py / train.py | 使用原生 OnPolicyRunner / DWAQOnPolicyRunner；新入口不调用训练更新、不导入脚本。 |
| 当前 IsaacLab 的 events.py | `isaaclab.envs.mdp.events.push_by_setting_velocity` 文档称设置速度，但所审计版本实现为 `vel_w += sample_uniform(...)`。新入口显式实现 heading 旋转后的单次加性跳变，记录前后值并校验。 |
| rsl_rl/modules/actor_critic_DWAQ.py | `act_inference` 的 VAE 仍使用 `randn_like`，`eval()` 不消除该随机性。保留原生采样，以 trial reset_seed 和步号隔离 RNG，续测不依赖批次随机调用顺序。 |
| BaseEnv.step / G1DwaqEnv.step | `check_reset` 后自动 reset。评测覆盖 check_reset 抓取 reset 前快照；done/timeout 优先处理，绝不把 reset 后站姿判作恢复。 |

训练路径没有修改。新增主要实现严格限制在四个评测 Python 模块，另有配置、测试、说明。

## 两个 baseline 的身份

完整路径、训练配置 SHA、checkpoint SHA、维度和审计字段见 `BASELINE_IDENTITIES.json`。

| 模型 | checkpoint SHA256 | checkpoint iter | seed（agent.yaml） | Actor/Critic | 动作 | History |
|---|---|---|---|---|---|---|
| PPO plain | 15167ca59f001a3f1888c894d43c14a2cb7321f48aab1b0a8eaabdd9e74ce848 | 9999 | 42 | 960 / 1010 | 29 | actor/critic 10/10 |
| DWAQ | 9b09227f6b997e0c89cb87844eeb05ffbafdfdb845701c390250fca284430361 | 9999 | 42 | 115（96+19）/307 | 29 | actor/critic 1/1，encoder 5 |

task 由 agent experiment_name、runner/policy 类型、env 输入配置与完整 checkpoint 参数交叉核对。`training_commit` 在 checkpoint 元数据中无法确认，记 `unknown`；不将当前仓库 commit 冒充训练 commit。`final` 为用户已完成 baseline 的显式登记阶段，不从文件名猜测。两个日志保存的机器人资产配置一致。

两个 checkpoint 已做 **CPU 原生类 strict=True 加载和合成观测前向检查**，不是仿真试验；同时生成 SHA 地址的只读评测快照，复制后再次核验 SHA 和 CPU 反序列化。

## 阶段 B：离线基础设施

- `protocol_id: G1-Recovery-Eval-Lite`，`protocol_version: 1.0-dev`。
- 共同参考 SHA：`0e1e68b0f2d91dd33698de3cb40e60c18461fbe34999a32c96e86adda288f73a`。
- protocol hash：`a9e0f800dfbdcf804b182a134a824b036ca997c0f86ecba7be7728a5400d0874`。
- manifest hash：`f35dea8b1267318cc09a667865dbea8b45bfc593033c1040ff1200e594070f98`。
- metrics config hash：`7150ef415a2651903d6182eca5c6a723204d110a69c79cb766cdd675b0527ae4`。
- E0 150 条，E1 240 条；实际初态逐条保存。E1 参考脚整体 left/right=120/120；每格 2/3 或 3/2，不宣称每格均衡。
- 共同名义候选表支持所有 3 坡度 × 4 moving 命令精确节点；standing 不套用步态恢复。
- 50 Hz NPZ、逐 trial 原子提交、派生 CSV、事件、完整性完成标记、显式 attempt、按缺失 trial 续测。
- 导入严格验证完成索引和 SHA；真实账本拒绝 `synthetic: true`。测试中使用的验证允许项仅为 Python API 的显式测试参数，不提供 CLI 绕过入口。
- 报告按 protocol/metrics/manifest/physics/inference 划分兼容组，另核对实际物理参数 hash。重复 attempt 不增加样本。
- MD/Word 由结构化结果重建、备份旧版、读取 notes、迁移尚未结构化的人工修改。

## 测试与文档检查

离线测试 **40 项通过**（约 37 秒，无仿真）。离线验收使用项目 Python 环境运行 `python -m pytest tools/evaluation/tests -q`。覆盖：

- 身份错误：task、维度、normalizer、缺 estimator、原生 state_dict 缺参数。
- manifest 计数、脚侧平衡、初态一致、幂等、篡改拒绝；共同参考精确匹配与必需字段。
- 逐帧误差与现有 practical_metrics 一致；加性速度旋转与其他分量保持。
- 仅一次达标不确认；连续确认、8 秒 entry/10 秒 confirm、晚跌倒保留事件但主失败；恢复步数不截成 5/6。
- 接触抖动、重复同脚、固定命令、历史单次推进、reset 前终态归属、GT 不作为 Actor 参数。
- 评测子类 CPU stub 检验 mesh/origin/normal、坡度课程不更新；**这不等于真实 USD/PhysX 验收**。
- 指定/已推分母、失败累计曲线、共同成功 trial 配对、指标与物理不兼容分组。
- 原子提交、重复提交、孤立 trace 中断恢复、完整结果 SHA 校验、合成数据拒绝、重复导入。
- 缺模型报告、合成数值表/曲线/Word XML、备注保留及 Word 人工内容备份迁移。

Word 验收：使用 LibreOffice 7.3 将正式空结果文档及临时目录中的 `SYNTHETIC_QA.docx` 转 PDF。正式文档 3 页、合成测试文档 5 页；已打开检查正式首页和合成累计曲线页，表格在页宽内自动换行、中文正常、图片/图注/页码可见。测试文档明确标记 SYNTHETIC，不在正式 runs/registry 中。

## 尚未进行的实测验收

阶段 C/D/E **NOT_RUN**：用户本轮明确不跑实验。固定 terrain 的实际 USD 与 PhysX 一致性、实际终端与推扰回写、32 环境运行、真实模型步态和至少 20 条推扰轨迹检测器验收均待后续小规模仿真验证。

本地 `report/DETECTOR_VALIDATION.md` 状态为 `NOT_RUN`。不宣称两个 baseline 已得到任何真实恢复率，不冻结为 1.0，不自动放宽旧阈值。

交付时的公共 MD/Word 中全部结果表为待评测；可使用 README 中的新 checkpoint 命令继续运行同一套开发协议，完成后自动更新固定文档。
