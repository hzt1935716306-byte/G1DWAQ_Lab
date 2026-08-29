# G1 Flat Symmetric 两阶段训练与递增速度扰动课程开发文档

> 文档用途：在修改代码前锁定本轮开发边界、继承关系、速度扰动课程、训练方式和最小验收标准。
> 状态：待审核；本文档通过后再逐项实现。
> 仓库：`hzt1935716306-byte/G1DWAQ_Lab`
> 基础分支：`feat/g1-symmetric-gait`
> 基础任务：`g1_flat_symmetric`

---

## 1. 本轮目标

整体保持“基线训练→正式恢复训练”两个大阶段，其中基线阶段再分为两个子阶段：

1. **Stage 1A — clean symmetric baseline**：使用 `g1_flat_symmetric`，在真正的平面上训练稳定、左右对称行走，不施加外部 push。
2. **Stage 1B — robust symmetric baseline**：新增 `g1_flat_symmetric_robust`，从 Stage 1A checkpoint 继续训练，只加入小幅随机 root 水平速度跳变，得到 `robust_symmetric_baseline`。
3. **Stage 2 — recovery curriculum**：新增 `g1_flat_symmetric_recovery`，从 `robust_symmetric_baseline` checkpoint 继续训练，再按 policy step 逐渐提高 root 水平速度扰动上限。

三个任务必须共享完全相同的：

- Actor observation 布局和维度；
- Critic observation 布局和维度；
- action 布局和维度；
- Actor/Critic 网络结构；
- observation history length；
- 原有 locomotion reward 和对称 reward；
- `RslRlSymmetryCfg` 的 data augmentation、mirror loss 和镜像映射。

三个任务的差异只能是 push event：

$$
\boxed{
\text{无 push}
\rightarrow
\text{小幅固定范围随机 push}
\rightarrow
\text{递增强度随机 push}
}
$$

三个任务应在长时间训练前一次性定义和注册完成。阶段切换时停止当前训练进程，使用 checkpoint 启动下一个任务，不在训练完成后临时修改同一个任务的配置。

---

## 2. 本轮不实现的内容

- DCM/LIPM 动力学；
- 1～5 步 LP 恢复认证器；
- 恢复状态机和恢复奖励；
- 新 reward 或新 observation；
- 状态估计网络和 certificate surrogate；
- 自定义 PPO、runner、Actor 或 Critic；
- 新的环境基类或自定义 Env 类；
- sim-to-sim 或实机部署；
- `g1_dwaq` 及任何 DWAQ 相关代码。

本轮不改变现有 `g1_flat`、`g1_rough` 和 `g1_dwaq` 的行为。

---

## 3. 当前代码现状

### 3.1 `g1_flat_symmetric` 当前不是真正平地且带 push

`G1FlatSymmetricEnvCfg` 当前只覆盖 reward。它继承的 `G1FlatEnvCfg.__post_init__()` 会设置：

```python
self.scene.terrain_type = "generator"
self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
```

`BaseEnvCfg` 同时默认配置了 `push_robot` interval event。因此 Stage 1A 必须在 `G1FlatSymmetricEnvCfg` 中同时覆盖地形和 push。

### 3.2 对称配置已独立存在

`G1FlatSymmetricAgentCfg` 已配置：

```python
RslRlSymmetryCfg(
    use_data_augmentation=True,
    use_mirror_loss=True,
    data_augmentation_func=compute_symmetric_states,
    mirror_loss_coeff=0.1,
)
```

robust 和 recovery 任务都直接继承该 Agent 配置，不重新定义对称逻辑、不修改 `g1_symmetry.py`。

### 3.3 policy step 计数与 `BaseEnv` 一致

`BaseEnv.step()` 在一个 policy step 内执行 `decimation` 次物理步，每个物理步增加一次 `sim_step_counter`。因此课程使用：

```python
policy_step = env.sim_step_counter // env.cfg.sim.decimation
```

当前 `dt=0.005`、`decimation=4`，一个 policy step 对应 $0.02\,\mathrm{s}$。课程计数不乘以并行环境数。

### 3.4 `train.py` 已支持目标 resume 路径

`train.py` 通过 `agent_cfg.experiment_name` 构造日志根目录，再用 `load_run` 和 `load_checkpoint` 定位 checkpoint。因此恢复 Agent 继续使用：

```python
experiment_name = "g1_flat_symmetric"
```

即可在 `logs/g1_flat_symmetric/` 中依次查找 Stage 1A 和 Stage 1B checkpoint，不需要修改 runner 或 checkpoint 格式。

---

## 4. 最终继承关系

```text
BaseEnvCfg
  └─ G1FlatEnvCfg
       └─ G1FlatSymmetricEnvCfg
            └─ G1FlatSymmetricRobustEnvCfg
                 └─ G1FlatSymmetricRecoveryEnvCfg

BaseAgentCfg
  └─ G1FlatAgentCfg
       └─ G1FlatSymmetricAgentCfg
            └─ G1FlatSymmetricRobustAgentCfg
                 └─ G1FlatSymmetricRecoveryAgentCfg
```

新任务使用现有 `BaseEnv`，不新增 `G1RecoveryEnv`。

---

## 5. 计划修改的文件

| 文件                                         | 计划 | 作用                                                    |
| -------------------------------------------- | ---- | ------------------------------------------------------- |
| `legged_lab/envs/g1/g1_config.py`          | 修改 | 将`g1_flat_symmetric` 锁定为 plane 并禁用 push        |
| `legged_lab/envs/g1/g1_recovery_config.py` | 新增 | 定义小扰动 robust 和递增扰动 recovery 的环境/Agent 配置 |
| `legged_lab/mdp/events.py`                 | 修改 | 增加递增 root 速度扰动 event                            |
| `legged_lab/envs/__init__.py`              | 修改 | 导入配置并注册新任务                                    |
| `tools/recovery/check_push_curriculum.py`  | 新增 | 检查课程 0%/50%/100% 的扰动上限                         |

以下文件需要核对，但预期不修改：

| 文件                                        | 不修改原因                                                   |
| ------------------------------------------- | ------------------------------------------------------------ |
| `legged_lab/envs/g1/g1_symmetry.py`       | 观测、动作和 history 不变，镜像映射应原样保留                |
| `legged_lab/envs/base/base_env_config.py` | 可在 G1 子类局部关闭/恢复 push，无需改变所有任务的基类默认值 |
| `legged_lab/scripts/train.py`             | 现有 experiment、run name 和 resume 逻辑已满足需求           |

如果实现时发现必须修改上述文件，应先报告具体原因，不自行扩大范围。

---

## 6. Stage 1A：修正 `g1_flat_symmetric`

在 `G1FlatSymmetricEnvCfg` 中加入：

```python
def __post_init__(self):
    super().__post_init__()
    self.scene.terrain_type = "plane"
    self.scene.terrain_generator = None
    self.domain_rand.events.push_robot = None
```

`super().__post_init__()` 保留 `G1FlatEnvCfg` 对 G1 asset、脚部 body、终止接触和 torso 质量随机化的配置；随后只覆盖地形和 push。

必须保持不变：

- `reward = G1SymmetricRewardCfg()`；
- 原 locomotion reward 和三个对称 reward；
- `compute_symmetric_states`；
- data augmentation、mirror loss 和其系数；
- push 以外的 domain randomization；
- observation noise、action delay 和 command 配置；
- Actor/Critic observation 代码和 history buffer。

配置验收：

```python
cfg = G1FlatSymmetricEnvCfg()
assert cfg.scene.terrain_type == "plane"
assert cfg.scene.terrain_generator is None
assert cfg.domain_rand.events.push_robot is None
```

---

## 7. Stage 1B 与 Stage 2 的任务配置

新建 `g1_recovery_config.py`，在同一个文件中定义 robust 和 recovery 两组薄配置，不为它们新建 Env 类。

### 7.1 Stage 1B：`g1_flat_symmetric_robust`

```python
@configclass
class G1FlatSymmetricRobustEnvCfg(G1FlatSymmetricEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.domain_rand.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={
                "velocity_range": {
                    "x": (-0.15, 0.15),
                    "y": (-0.10, 0.10),
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
```

该任务的目标是普通小扰动鲁棒性，不是极限恢复训练：

- 地形仍然是真正 plane；
- 每 8～12 秒产生一次小幅速度跳变；
- $\Delta v_x\in[-0.15,0.15]\,\mathrm{m/s}$；
- $\Delta v_y\in[-0.10,0.10]\,\mathrm{m/s}$；
- 扰动范围在 Stage 1B 内保持固定，不增长到恢复训练上限；
- 不增加 recovery reward 或 observation。

Agent 配置：

```python
@configclass
class G1FlatSymmetricRobustAgentCfg(G1FlatSymmetricAgentCfg):
    experiment_name: str = "g1_flat_symmetric"
    run_name: str = "robust_symmetric"
```

Stage 1B 完成并通过验收的 checkpoint 固定称为：

```text
robust_symmetric_baseline
```

### 7.2 Stage 1B 的训练完成条件

不仅根据迭代数命名 baseline，还要对比 Stage 1A 和 Stage 1B：

- 无 push 时的速度跟踪和 episode 存活率没有明显退化；
- 无 push 区间的左右对称性保持；
- 小 push 后的立即摔倒率明显降低；
- push 后能重新回到稳定速度跟踪；
- 脚滑、躯干振荡和非法接触没有不可接受的恶化。

侧向 push 之后的短暂左右不对称是正常恢复动作，不应要求扰动后每一帧都严格镜像对称。对称性应主要在无 push 区间和恢复后稳态区间统计。

### 7.3 Stage 2：`g1_flat_symmetric_recovery`

```python
@configclass
class G1FlatSymmetricRecoveryEnvCfg(G1FlatSymmetricRobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.domain_rand.events.push_robot = EventTerm(
            func=mdp.curriculum_push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={
                "warmup_policy_steps": 5_000,
                "ramp_policy_steps": 120_000,
                "start_abs_delta_v_xy": (0.15, 0.10),
                "end_abs_delta_v_xy": (1.00, 0.80),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
```

继承调用顺序是：

1. `G1FlatSymmetricEnvCfg` 确保 plane 且无 push；
2. `G1FlatSymmetricRobustEnvCfg` 安装小幅固定范围 push；
3. `G1FlatSymmetricRecoveryEnvCfg` 将它覆盖成递增强度 push。

这保证三个任务始终使用 plane，且只有 push event 不同。

Agent 配置：

```python
@configclass
class G1FlatSymmetricRecoveryAgentCfg(G1FlatSymmetricRobustAgentCfg):
    experiment_name: str = "g1_flat_symmetric"
    run_name: str = "push_curriculum"
```

`experiment_name` 始终保持 `g1_flat_symmetric`，使 Stage 1A、Stage 1B 和 Stage 2 可以在同一日志根目录中依次 resume。

### 7.4 单位和参数性质

- `interval_range_s`：秒；
- warm-up/ramp：policy step；
- 扰动上限：$\mathrm{m/s}$，表示 root 水平速度增量绝对值的采样上限。

`(0.15, 0.10)` 和 `(1.00, 0.80)` 都是首轮训练值，不是 G1 的最终物理极限。

---

## 8. 递增速度扰动 event

### 8.1 函数签名

```python
def curriculum_push_by_setting_velocity(
    env,
    env_ids,
    warmup_policy_steps,
    ramp_policy_steps,
    start_abs_delta_v_xy,
    end_abs_delta_v_xy,
    asset_cfg=SceneEntityCfg("robot"),
):
    ...
```

函数不命名为 `force_curriculum`，因为它不是持续力 event。

### 8.2 课程公式

```python
policy_step = env.sim_step_counter // env.cfg.sim.decimation
progress = clamp(
    (policy_step - warmup_policy_steps) / ramp_policy_steps,
    0.0,
    1.0,
)
current_max_xy = start_xy + progress * (end_xy - start_xy)
```

- warm-up 期间使用起始上限；
- ramp 期间线性插值；
- ramp 结束后保持结束上限。

### 8.3 采样与执行

各环境的两轴独立采样：

$$
\Delta v_x\sim\mathcal U(-v_{x,\max},v_{x,\max}),
\qquad
\Delta v_y\sim\mathcal U(-v_{y,\max},v_{y,\max}).
$$

包装函数只计算当前上限，然后复用 Isaac Lab 现有实现：

```python
push_by_setting_velocity(
    env,
    env_ids,
    velocity_range={
        "x": (-current_max_x, current_max_x),
        "y": (-current_max_y, current_max_y),
    },
    asset_cfg=asset_cfg,
)
```

不复制 root velocity 的采样和写入逻辑。现有 Isaac Lab 实现在当前 root velocity 上加入采样量，因此本文统一称其为“速度扰动/速度增量”，不称为持续外力。`x/y` 按 Isaac Lab 接口作用于 world-frame root 线速度分量。

### 8.4 调试属性

event 只保存：

| 属性                             | 建议类型/形状           | 含义                         |
| -------------------------------- | ----------------------- | ---------------------------- |
| `env.push_curriculum_progress` | Python`float`         | 全局课程进度`[0, 1]`       |
| `env.push_curriculum_max_xy`   | tensor`(2,)`          | 当前 (x/y) 绝对扰动上限      |
| `env.last_push_delta_v_xy`     | tensor`(num_envs, 2)` | 各环境最近一次实际施加的增量 |

为保留实际增量且不复制 Isaac Lab 采样逻辑，包装函数在调用前复制目标环境 root (x/y) 速度，调用后以速度差更新本次 `env_ids` 对应的调试 buffer。

这三个量不加入 observation 或 reward。

### 8.5 输入检查

实现应快速拒绝：

- `warmup_policy_steps < 0`；
- `ramp_policy_steps <= 0`；
- start/end 不是长度 2；
- start/end 中存在负数、NaN 或 Inf；
- 任一 end 分量小于对应 start 分量。

`env_ids is None` 时必须正确处理为所有环境。

---

## 9. 新任务注册

在 `legged_lab/envs/__init__.py` 导入：

```python
from legged_lab.envs.g1.g1_recovery_config import (
    G1FlatSymmetricRobustAgentCfg,
    G1FlatSymmetricRobustEnvCfg,
    G1FlatSymmetricRecoveryAgentCfg,
    G1FlatSymmetricRecoveryEnvCfg,
)
```

并注册两个新任务：

```python
task_registry.register(
    "g1_flat_symmetric_robust",
    BaseEnv,
    G1FlatSymmetricRobustEnvCfg(),
    G1FlatSymmetricRobustAgentCfg(),
)

task_registry.register(
    "g1_flat_symmetric_recovery",
    BaseEnv,
    G1FlatSymmetricRecoveryEnvCfg(),
    G1FlatSymmetricRecoveryAgentCfg(),
)
```

---

## 10. checkpoint 兼容性

开发中必须比较三个任务的：

```text
num_actor_obs
num_critic_obs
num_actions
actor_obs_history_length
critic_obs_history_length
policy.class_name
policy.actor_hidden_dims
policy.critic_hidden_dims
```

上述项必须完全一致。Stage 1A checkpoint 必须能加载到 Stage 1B，Stage 1B checkpoint 必须能加载到 Stage 2，最终以 resume smoke test 为准。

---

## 11. 训练命令

### 11.1 Stage 1A：clean symmetric baseline

```bash
cd /home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab

python legged_lab/scripts/train.py \
  --task g1_flat_symmetric \
  --headless \
  --num_envs 4096 \
  --run_name clean_symmetric \
  --max_iterations <Stage-1A迭代数>
```

### 11.2 Stage 1B：robust symmetric baseline

```bash
cd /home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab

python legged_lab/scripts/train.py \
  --task g1_flat_symmetric_robust \
  --headless \
  --num_envs 4096 \
  --resume True \
  --load_run <Stage-1A运行目录名> \
  --checkpoint model_<Stage-1A迭代数>.pt \
  --run_name robust_symmetric \
  --max_iterations <Stage-1B额外迭代数>
```

Stage 1B 完成后，先用第 7.2 节的条件验收，通过后再将选定 checkpoint 标记为 `robust_symmetric_baseline`。

### 11.3 Stage 2：recovery curriculum

```bash
cd /home/zt/project/myproject/G1DWAQ_Lab/TienKung-Lab

python legged_lab/scripts/train.py \
  --task g1_flat_symmetric_recovery \
  --headless \
  --num_envs 4096 \
  --resume True \
  --load_run <Stage-1B robust_symmetric运行目录名> \
  --checkpoint model_<robust-baseline迭代数>.pt \
  --run_name push_curriculum \
  --max_iterations <Stage-2额外迭代数>
```

三个任务都使用 `logs/g1_flat_symmetric/`。`run_name` 只是新运行目录后缀，不改变 experiment 根目录。

### 11.4 训练停止和继续方式

- 优先使用有限 `--max_iterations` 让训练进程正常结束；
- runner 正常结束时会保存最后一个 checkpoint；
- 不使用 `kill -9` 作为常规阶段切换方式；
- 若当前阶段未收敛，继续 resume 同一任务；
- 只有通过当前阶段验收后，才 resume 到下一个任务；
- 当前 runner 中 resume 后的 `max_iterations` 表示本次运行额外训练的 iteration 数。

---

## 12. 最小检查脚本

新建：

```text
TienKung-Lab/tools/recovery/check_push_curriculum.py
```

脚本使用与 event 相同的课程计算 helper，不复制插值公式。默认配置的期望值是：

| 检查点 |                                  `policy_step` | `progress` | `current_max_xy` |
| ------ | -----------------------------------------------: | -----------: | -----------------: |
| 0%     |                          `warmup_policy_steps` |      `0.0` |   `(0.15, 0.10)` |
| 50%    | `warmup_policy_steps + ramp_policy_steps // 2` |      `0.5` |  `(0.575, 0.45)` |
| 100%   |      `warmup_policy_steps + ramp_policy_steps` |      `1.0` |   `(1.00, 0.80)` |

脚本用数值容差比较，断言失败时以非零状态退出。

---

## 13. 最小 smoke test

实现后只完成以下检查，不运行全仓库 lint 或 4096 环境长时间训练。

### 13.1 导入与配置

- task registry 同时包含 `g1_flat_symmetric`、`g1_flat_symmetric_robust` 和 `g1_flat_symmetric_recovery`；
- Stage 1A 是 plane、generator 为 `None`、push 为 `None`；
- Stage 1B 仍是 plane，小幅固定范围 push 已启用；
- Stage 1B 的速度跳变范围为 $x\in[-0.15,0.15]\,\mathrm{m/s}$、$y\in[-0.10,0.10]\,\mathrm{m/s}$；
- Stage 2 仍是 plane，递增强度 push event 已启用；
- interval、warm-up、ramp、start 和 end 参数可读；
- 实例化 robust/recovery cfg 后再检查 symmetric cfg 的 push 仍为 `None`，排除嵌套配置意外共享。

### 13.2 课程检查

运行 `check_push_curriculum.py`，检查 0%/50%/100% 分别输出：

```text
(0.15, 0.10)
(0.575, 0.45)
(1.00, 0.80)
```

### 13.3 Stage 1A → Stage 1B resume smoke test

使用一个 Stage 1A checkpoint：

```bash
python legged_lab/scripts/train.py \
  --task g1_flat_symmetric_robust \
  --headless \
  --num_envs 32 \
  --resume True \
  --load_run <Stage-1A运行目录名> \
  --checkpoint model_<Stage-1A迭代数>.pt \
  --run_name robust_symmetric_smoke \
  --max_iterations 5
```

### 13.4 Stage 1B → Stage 2 resume smoke test

使用上一项产生的 Stage 1B smoke checkpoint：

```bash
python legged_lab/scripts/train.py \
  --task g1_flat_symmetric_recovery \
  --headless \
  --num_envs 32 \
  --resume True \
  --load_run <Stage-1B-smoke运行目录名> \
  --checkpoint model_<Stage-1B-smoke迭代数>.pt \
  --run_name push_curriculum_smoke \
  --max_iterations 5
```

当 `num_steps_per_env=24` 时，每个 5-iteration smoke test 约覆盖每个环境 120 个 environment step，同时检查 checkpoint 加载、observation shape、Actor/Critic 前向、PPO 更新、data augmentation 和 mirror loss。

验收条件：

- checkpoint 成功加载并完成 5 次 iteration；
- 无 observation/action shape 异常；
- symmetry augmentation 和 mirror loss 无异常；
- 无 event 参数或 device 异常。

---

## 14. 后续恢复算法的预留工程结构

> 本节是整个项目的后续目录规划，解决“算法定义写在哪里”和“从仿真获取特权真值写在哪里”的问题。本轮只审核这个结构，不创建、不实现其中的 LIPM/DCM、LP、状态机和奖励文件。

### 14.1 总体目录

```text
TienKung-Lab/
├── legged_lab/
│   ├── recovery/                           # 恢复算法和特权教师模块
│   │   ├── __init__.py
│   │   ├── types.py                       # 状态、证书、求解状态等数据类型
│   │   ├── privileged_state.py            # 从 Isaac Lab 提取仿真特权真值
│   │   ├── lipm_dcm.py                     # 纯 LIPM/DCM 动力学和触地映射
│   │   ├── certificate.py                  # 0～5 步 LP、N_min、margin 和见证检查
│   │   ├── manager.py                      # touchdown、缓存、恢复状态机与调用编排
│   │   └── params/
│   │       └── g1_flat_symmetric.yaml      # h、T、CoP、落脚域、速度上限等标定参数
│   │
│   └── envs/g1/
│       ├── g1_recovery_config.py            # 环境、扰动、后续恢复模块配置
│       └── g1_recovery_env.py               # 后续才增加的薄集成层
│
├── tools/recovery/
│   ├── check_push_curriculum.py             # 本轮新增
│   ├── calibrate_privileged_state.py        # 后续标定 CoM、步时、相位和足端能力
│   ├── validate_certificate.py              # 后续对比理论证书与全身仿真
│   └── benchmark_certificate.py             # 后续求解速度 benchmark
│
└── tests/recovery/
    ├── test_lipm_dcm.py
    ├── test_privileged_state.py
    ├── test_certificate.py
    └── test_recovery_manager.py
```

### 14.2 算法定义文件

#### `lipm_dcm.py`

该文件只实现纯数学，不导入 Isaac Lab，包括：

- 由 $c,\dot c,h$ 计算 $\omega$ 和 DCM $\xi$；
- $b=\xi-p_{\mathrm{sup}}$ 与 $q=p_{\mathrm{sw}}-p_{\mathrm{sup}}$；
- 常值 CoP 下的 DCM 解析传播；
- 触地映射 $b_{k+1}$ 和 $q_{k+1}$；
- 左右支撑切换；
- 行走周期点的计算与检查。

这一层应能在不启动 Isaac Sim 的情况下做单元测试。

#### `certificate.py`

该文件实现恢复认证算法，包括：

- 任意 horizon 的通用联合 LP 构造器；
- $F_0\rightarrow F_1\rightarrow\cdots\rightarrow F_5$ 顺序查询；
- $N_{\min}\in\{0,1,2,3,4,5,>5\}$；
- 可行内缩 margin 和不可行放松 margin；
- LP 见证的独立前向重放和残差检查；
- `FINITE`、`OVER_HORIZON`、`INVALID_INPUT`、`SOLVER_FAILURE` 的区分。

该文件的输入必须是已经整理好的数值状态和参数，不得直接从 `env.scene` 读数据。这样可以保持算法层与仿真器解耦。

#### `types.py`

集中定义：

```python
PrivilegedRecoveryState
RecoverabilityCertificate
HorizonPlan
CertificateStatus
SupportSide
```

避免数据类型分散在 environment、reward 和 LP 求解器中。

### 14.3 仿真特权真值提取文件

#### `privileged_state.py`

该文件是 Isaac Lab 与纯数学认证器之间的数据适配层，专门负责从仿真环境获取：

- 所有刚体当前质量；
- 全身质量加权 CoM 位置 $c$；
- 全身质量加权 CoM 速度 $\dot c$；
- 左右脚世界位置和速度；
- contact sensor 的首次接触、接触力和 air time；
- 当前支撑脚、摆动脚和 touchdown 事件；
- base heading 和冻结 heading frame；
- 当前步态相位、完整步时 $T$ 和剩余时间 $T_{\mathrm{rem}}$；
- CoM 有效高度 $h$；
- 转换到统一 heading frame 后的 $c,\dot c,p_{\mathrm{sup}},p_{\mathrm{sw}}$。

全身 CoM 必须使用当前刚体质量加权：

$$
c=\frac{\sum_jm_jc_j}{\sum_jm_j},
\qquad
\dot c=\frac{\sum_jm_j\dot c_j}{\sum_jm_j}.
$$

不得用 pelvis/root 的位置或 `root_lin_vel_b` 代替全身 CoM 与全身 CoM 速度。质量随机化开启时，必须使用随机化后的当前质量。

该文件输出 `PrivilegedRecoveryState`，而不直接计算 reward，也不直接修改 Actor observation。

### 14.4 环境集成文件

#### `manager.py`

该文件后续负责：

- 调用 `privileged_state.py` 刷新特权状态；
- 检测 push 结束、touchdown、success、timeout 和 fall；
- 管理 `dense_prev` 和 `td_prev` 两套独立缓存；
- 调用 `certificate.py` 获得 $N_{\min}$ 和 margin；
- 维护 recovery trial 的实际 touchdown 计数和日志。

#### `g1_recovery_env.py`

该文件只在后续接入恢复算法时新增，作为薄集成层：

- 初始化 recovery manager；
- 在固定时机更新特权状态；
- reset 时清理恢复 buffer；
- 将后续恢复奖励以独立分量接入。

本轮 `g1_flat_symmetric_recovery` 仍注册为 `BaseEnv`，不创建这个文件。只有开始“特权状态 + 认证器 + 状态机”集成阶段时，才将任务的 Env 类切换到该薄集成层。

### 14.5 数据流

```text
Isaac Lab 全身仿真真值
        │
        ▼
privileged_state.py
        │  PrivilegedRecoveryState
        ▼
lipm_dcm.py + certificate.py
        │  RecoverabilityCertificate(N_min, margin, plan)
        ▼
manager.py
        ├──> 日志/标签
        ├──> 后续恢复奖励
        └──> 后续恢复状态机
```

本轮只实现上述数据流之前的“无扰动平地对称基线 + 小扰动 robust baseline + 递增速度扰动”，Actor/Critic observation 不接入任何特权恢复量。

---

## 15. 已知边界

### 15.1 恢复训练二次 resume 时课程会重新 warm-up

`env.sim_step_counter` 在新建环境时从 0 开始：

- 从 Stage 1B checkpoint 首次进入 Stage 2 时，课程从 warm-up 开始，这是预期行为；
- 如果 Stage 2 中断后再 resume，课程也会重新 warm-up，因为环境计数器不在 PPO checkpoint 中恢复。

本轮按指定公式实现，不自行增加 runner 状态或 checkpoint 字段。如需跨运行保存课程进度，后续再单独设计 offset 或持久化方案。

### 15.2 课程是全局的，不随单个 episode reset

同一训练运行中，所有并行环境共享相同的 `progress` 和 `current_max_xy`，但在各自 interval event 触发时独立采样扰动。episode reset 不会重置课程。

### 15.3 旧 checkpoint 可用于 smoke test，不等于新 Stage 1A 基线

修改前的 `g1_flat_symmetric` 实际使用 gravel generator 和默认 push。因此旧 checkpoint 可验证 shape 和 resume 链路，但不应被当成在新的真平地、无 push 配置下完成的 Stage 1A checkpoint。

---

## 16. 实施顺序

1. 修改 `G1FlatSymmetricEnvCfg.__post_init__()`，建立 Stage 1A 纯平地无扰动任务；
2. 验证 Stage 1A 的 plane 和 `push_robot is None`；
3. 在 `g1_recovery_config.py` 定义 Stage 1B 小幅固定范围 push 任务；
4. 在 `events.py` 实现 Stage 2 的课程上限 helper 和 push wrapper；
5. 在 `g1_recovery_config.py` 定义 Stage 2 递增 push 任务；
6. 注册 `g1_flat_symmetric_robust` 和 `g1_flat_symmetric_recovery`；
7. 新增并运行 `check_push_curriculum.py`；
8. 比较三个任务的 observation/action/history/network；
9. 使用 32 环境分别执行 Stage 1A→1B 和 Stage 1B→2 的 5-iteration resume smoke test；
10. 汇报文件、命令、结果和阻塞项；
11. 停止，不继续实现 LIPM/DCM 或其他恢复模块。

如果任一步要求修改 observation、镜像映射、runner 或 DWAQ 代码，应停止并排查，不扩大改动绕过问题。

---

## 17. 最终验收清单

- [ ] 分支为 `feat/g1-symmetric-gait`。
- [ ] `g1_flat_symmetric` 是 plane、无 terrain generator、无 push。
- [ ] 所有对称 reward、data augmentation 和 mirror loss 保留。
- [ ] `g1_symmetry.py` 没有修改。
- [ ] `g1_flat_symmetric_robust` 使用 `BaseEnv` 成功注册。
- [ ] robust 任务是 plane，且只启用 `(0.15, 0.10)` 小幅固定范围速度跳变。
- [ ] Stage 1B 的无 push 行走性能没有相对 Stage 1A 明显退化。
- [ ] Stage 1B 的小 push 存活和恢复表现明显改善。
- [ ] 通过验收的 Stage 1B checkpoint 被明确标记为 `robust_symmetric_baseline`。
- [ ] `g1_flat_symmetric_recovery` 使用 `BaseEnv` 成功注册。
- [ ] Stage 2 recovery 任务仍是 plane，递增 push event 已启用且参数可配置。
- [ ] push 是 root 水平速度增量，未冒充持续外力。
- [ ] 课程 0%/50%/100% 分别为 `(0.15, 0.10)`、`(0.575, 0.45)`、`(1.00, 0.80)`。
- [ ] 三个任务的 observation、action、history 和网络结构完全一致。
- [ ] robust Agent 的 `experiment_name` 是 `g1_flat_symmetric`，默认 `run_name` 是 `robust_symmetric`。
- [ ] 恢复 Agent 的 `experiment_name` 是 `g1_flat_symmetric`，默认 `run_name` 是 `push_curriculum`。
- [ ] Stage 1A→1B、Stage 1B→2 的 checkpoint 均可 resume，32 环境、5 iteration smoke test 无异常。
- [ ] 未修改 `g1_flat`、`g1_dwaq` 或 DWAQ 相关代码。
- [ ] 未新增 reward、observation、PPO、runner、网络或 Env 基类。
- [ ] 未实现 LIPM/DCM、LP 认证器、恢复状态机或估计网络。

---

## 18. 完成后的固定汇报格式

实现完成后只汇报：

1. 修改和新增的文件；
2. 每个文件的修改内容；
3. Stage 1A、Stage 1B 和 Stage 2 三个任务的训练命令；
4. 最小 smoke test 的实际结果；
5. 是否存在阻塞问题。

不自动开始后续 LIPM/DCM 开发。
