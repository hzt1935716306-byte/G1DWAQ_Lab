# 协议 v1.3 三服务器按模型执行说明

本说明是运行层补充，不改变协议 v1.3 的实验条件、样本量、判据、margin
边界或协议哈希。`paper-eval-v1.3-frozen` 是最低评测基线；后续只修改文档、模型
登记或运行封装的提交可以使用，只要运行器计算出的 `evaluation_code_hash` 一致。
仓库 `HEAD` 仅作来源记录，不再作为跨服务器合并门禁：

```bash
git fetch origin --tags
git checkout formal-experiment
git rev-parse HEAD
```

正式运行前仍须确认协议哈希、manifest 哈希和实际可执行代码哈希一致。说明文档提交
不同但可执行代码哈希相同，不会使结果失去兼容性。

本版灵活 checkpoint 运行器的可执行代码哈希为：
`bf8599df7510c496ca6d45687a321c0e54236ece76c35da30e4eb5d770e34304`。

## 模型分配

三台服务器按配置与权限等价处理，不设置主服务器，也不规定服务器与模型的对应
关系。每台服务器评测哪个模型和 checkpoint 由用户在运行前自行选择；A、B、C
任意一台均可运行任意已支持的模型。同一个模型也可以在多台服务器上运行，只要
它们负责不同的 checkpoint、stage 或 experiment shard。

不同服务器可以使用同一份 manifest 和配对 eval seed 分别评测相同或不同模型。
唯一强制约束是：同一个 `(stage, experiment_id, model_id, train_seed,
checkpoint_sha256, estimator_sha256, manifest_hash)` shard 在任一时刻只能由一台服务器写入。不要
让两台服务器同时写同一 checkpoint 的同一实验。

## Checkpoint 规则

Checkpoint 不是全局冻结配置，而是用户每次下达任务时选择的实验输入。
`configs/models.yaml` 中的 checkpoint 只是可省略命令行参数时使用的默认值，不是
某个模型唯一允许的 checkpoint。M3、M4 即使没有默认 checkpoint，也可以由用户
直接指定文件运行。

运行器读取用户指定的文件并自动计算 SHA-256；结果按 checkpoint SHA 分目录保存。
同一 `model_id`、同一训练 seed 下的两个不同 checkpoint 不会互相覆盖，也不会在
汇总时被默认混为同一个统计样本。若模型使用速度 estimator，可以同样在任务中
指定 estimator 文件，并记录其实际 SHA-256。

主协议中的 `checkpoints_per_method: 1` 在这里解释为“每个独立汇总变体只能包含一个
checkpoint SHA”，用于禁止把不同 checkpoint 冒充同一个统计样本；它不是全局
checkpoint 白名单，也不限制用户创建和比较多个 checkpoint 变体。

## 用户任务指令

协议不规定固定的任务范围、命令菜单或服务器角色。用户可以用自然语言告诉任意一台
服务器要使用哪个模型/checkpoint、运行哪个阶段、哪个实验或任意实验组合。服务器不依赖 A、B、C
编号决定行为，而是把用户指令映射到本仓库已有 CLI，并读取冻结协议、implementation
freeze、模型任务配置和已有结果目录，完成 manifest/checkpoint 哈希计算、断点恢复、运行、
封账及结果保存。已有匹配 shard 必须续跑，不能重新生成相同 trial。

如果自然语言任务与协议冲突，服务器必须说明冲突并拒绝执行，不能自行更换用户指定的
checkpoint、baseline、seed、margin 边界或统计规则。pilot 与 formal 始终分开保存。

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
`TienKung-Lab` 目录中运行。用户直接指定 checkpoint 时，SHA-256 由运行器计算，
无需预先写入 `configs/models.yaml`。

例如，对用户指定的模型运行实验二和实验三：

```bash
conda run -n g1 python paper_eval_final/run.py \
  --stage formal --model <MODEL_ID> --checkpoint <CHECKPOINT_PATH> \
  --train-seed <TRAIN_SEED> --experiments 2 3 --num-envs 2048
```

使用自选 estimator 时追加：

```bash
--estimator <ESTIMATOR_PATH>
```

如果省略 `--checkpoint`，运行器使用模型注册表中与 `--train-seed` 对应的默认
checkpoint；存在多个默认训练 seed 且未指定 seed 时，会分别创建独立 shard。

实验一只使用冻结的 M1 baseline。它可以在三台服务器中的任意一台运行：

```bash
conda run -n g1 python paper_eval_final/run.py \
  --stage formal --experiment 1 --baseline slope_nosys_d_matched --num-envs 2048
```

已有断点数据不在 GitHub 中。恢复时必须先把当前
`results/formal/protocol_1_3_b71a1ddf9293/experiment_1/M1/train_seed_42`
完整复制到你选择的任意一台服务器并完成校验，再由该机器独占恢复；也可以直接在
当前保存该目录的机器上继续。该既有 shard 的代码哈希是
`1f35c261c44e280292a94947805c54bd300f036eee1fbcab043f9d6389a0cc45`，必须继续使用
`paper-eval-v1.3-frozen`，不能用本版灵活 checkpoint 运行器接写。协议不指定由哪台
服务器持有或接管。

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

聚合器会拒绝协议、manifest、指标、物理配置、可执行代码哈希、资产或仿真器版本不一致
的数据，也会拒绝相同 `(model_id, checkpoint_sha256, estimator_sha256, train_seed,
trial_id)` 的重复记录。
不同仓库提交如果实际可执行代码哈希一致，可以安全合并。三台服务器
是计算资源分工，不是三个训练 seed 或三个统计重复。
