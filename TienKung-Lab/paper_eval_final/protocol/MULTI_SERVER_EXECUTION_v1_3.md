# 协议 v1.3 三服务器按模型执行说明

本说明是运行层补充，不改变协议 v1.3 的实验条件、样本量、判据、margin
边界或协议哈希。所有正式评测服务器必须使用同一个冻结版本：

```bash
git fetch origin --tags
git checkout paper-eval-v1.3-frozen
git rev-parse HEAD
```

期望提交为：
`4434d384cea959025a31c4fad1fd056b3fc017f5`。

## 模型所有权

| 服务器 | 模型 | 方法 | 正式任务 |
|---|---|---|---|
| A | M1 | Baseline | 断点续跑实验一；运行实验二、三 |
| B | M2 | DWAQ | 运行实验二、三 |
| C | M5 | Ours | 运行实验二、三 |

M3、M4 尚无冻结 checkpoint，不得启动其正式实验。checkpoint 可用后，将每个
模型指派给唯一服务器，不能由两台机器同时写同一个 model-experiment shard。

## Seed 规则

```text
eval_seed = stage_seed_namespace + experiment_id * 1000000 + global_sequence
screening namespace = 1301000
pilot namespace     = 1302000
formal namespace    = 1303000
```

正式实验一、二、三的首个 eval seed 分别为 2303000、3303000、4303000。
服务器编号、主机名和 GPU 编号都不参与 seed 或 trial id。不同模型在同一
global sequence 上使用相同 eval seed 是配对设计，不能为每台服务器增加 offset。

## 三台服务器命令

在每台服务器的 `TienKung-Lab` 目录中运行。先确认对应 checkpoint 文件存在且
SHA-256 与 `configs/models.yaml` 一致。

服务器 A：

```bash
conda run -n g1 python paper_eval_final/run.py \
  --stage formal --experiment 1 --baseline slope_nosys_d_matched --num-envs 2048
conda run -n g1 python paper_eval_final/run.py \
  --stage formal --model M1 --experiments 2 3 --num-envs 2048
```

实验一的断点数据不在 GitHub 中。服务器 A 必须保留当前
`results/formal/protocol_1_3_b71a1ddf9293/experiment_1/M1/train_seed_42`
目录；若迁移，必须复制整个目录后再恢复。

服务器 B：

```bash
conda run -n g1 python paper_eval_final/run.py \
  --stage formal --model M2 --experiments 2 3 --num-envs 2048
```

服务器 C：

```bash
conda run -n g1 python paper_eval_final/run.py \
  --stage formal --model M5 --experiments 2 3 --num-envs 2048
```

## 数据回收与聚合

只复制完整的模型目录，保留目录层级和文件名。推荐使用带校验的 `rsync`，并用
`--ignore-existing` 防止覆盖中央机已有数据。每个 shard 必须包含
`identity.json`、`completion.json`、`records/`、`traces/` 和 `events/`；物理失败
时还应包含 `pre_reset_snapshots/`。

全部完整 shard 回收到中央机后运行：

```bash
conda run -n g1 python paper_eval_final/run.py \
  --aggregate --stage formal --experiments 2 3
conda run -n g1 python paper_eval_final/run.py \
  --full-audit --stage formal
```

聚合器会拒绝协议、manifest、指标、物理配置、代码提交、资产或仿真器版本不一致
的数据，也会拒绝相同 `(model_id, train_seed, trial_id)` 的重复记录。三台服务器
是计算资源分工，不是三个训练 seed 或三个统计重复。
