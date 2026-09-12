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

## 模型分配

三台服务器按配置与权限等价处理，不设置主服务器，也不规定服务器与模型的对应
关系。每台服务器评测哪个模型由用户在运行前自行选择；A、B、C 任意一台均可运行
M1、M2 或 M5。同一个模型也可以在多台服务器上运行，只要它们负责不同的 stage
或 experiment shard。

不同服务器可以使用同一份 manifest 和配对 eval seed 分别评测相同或不同模型。
唯一强制约束是：同一个 `(stage, experiment_id, model_id, train_seed,
manifest_hash)` shard 在任一时刻只能由一台服务器写入。不要让两台服务器同时
运行同一模型的同一实验。M3、M4 尚无冻结 checkpoint，在 checkpoint 注册并冻结
前不能启动其正式实验。

## 用户任务指令

协议不规定固定的任务范围、命令菜单或服务器角色。用户可以用自然语言告诉任意一台
服务器要使用哪个模型、运行哪个阶段、哪个实验或任意实验组合。服务器不依赖 A、B、C
编号决定行为，而是把用户指令映射到本仓库已有 CLI，并读取冻结协议、implementation
freeze、模型注册表和已有结果目录，完成 manifest/checkpoint 哈希检查、断点恢复、运行、
封账及结果保存。已有匹配 shard 必须续跑，不能重新生成相同 trial。

如果自然语言任务与冻结协议冲突，服务器必须说明冲突并拒绝执行，不能自行更换模型、
baseline、seed、margin 边界或统计规则。pilot 与 formal 始终分开保存。

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

## 命令示例

以下仅是自然语言任务映射后的命令示例，不是固定任务菜单。在每台服务器的
`TienKung-Lab` 目录中运行前，应确认对应 checkpoint 文件存在且 SHA-256 与
`configs/models.yaml` 一致。

例如，对用户指定的模型运行实验二和实验三：

```bash
conda run -n g1 python paper_eval_final/run.py \
  --stage formal --model <MODEL_ID> --experiments 2 3 --num-envs 2048
```

实验一只使用冻结的 M1 baseline。它可以在三台服务器中的任意一台运行：

```bash
conda run -n g1 python paper_eval_final/run.py \
  --stage formal --experiment 1 --baseline slope_nosys_d_matched --num-envs 2048
```

已有断点数据不在 GitHub 中。恢复时必须先把当前
`results/formal/protocol_1_3_b71a1ddf9293/experiment_1/M1/train_seed_42`
完整复制到你选择的任意一台服务器并完成校验，再由该机器独占恢复；也可以直接在
当前保存该目录的机器上继续。协议不指定由哪台服务器持有或接管。

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
