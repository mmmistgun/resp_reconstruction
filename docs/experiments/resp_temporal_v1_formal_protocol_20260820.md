# RTM-v1 formal training 协议

日期：2026-08-20

状态：**用户已授权formal training实现与用户手动执行；冻结runner/config已建立但15项均尚未运行。Validation只允许作为训练内checkpoint selector与最终validation summary；research-test继续关闭。**

## 1. 证据与不可变边界

Formal阶段形成`validation-development evidence`，不是无偏held-out证据。以下内容不得改变：research-v2数据与train/validation split、subject/session隔离、BCG input、target、Pi、`L_sync+0.25 L_effort`、metrics、full-validation Local-RR strict-`<` selector、五项locked candidate、三个seed及共同substrate/decoder identity。

GPU engineering lock SHA-256=`5632b0e404f943e61c646a8ddcf1761c2391884412915ef1d79aa342f8bc0183`，decision=`all_five_hardware_feasible_at_128x1`。Formal五项统一physical batch=128、accumulation=1、effective batch=128；资源指标不得删除、替换、缩容或排序候选。

训练只允许train与validation。禁止读取research-test、历史checkpoint、旧run prediction或任何test cache；禁止resume、early stopping、根据先完成seed的validation结果取消后续seed，也禁止新LR/width/depth/grid/band搜索。

## 2. 冻结矩阵与优化

- candidates：T0、TCN d9/H384、BiMamba2 D96×6、BiLSTM H96×2、multiscale 10/2/1-Hz H384；
- seeds：`20260811 / 20260812 / 20260813`；
- runs：`5×3=15`；
- 完整train/validation固定为`10141/2675` windows、`32/7`个`samp_id`、row-id SHA-256=`f290e569140a2ff7745cf1a5cfa6a4da943644d76498c9b85517d3ae0702c45e / b68a51b101bb80033c4de18c9c21cdfbb8924d1bfe134330f047617cf3b0915a`；
- epochs：80；batch128且drop-last=false，固定train/validation loader为`80/21` batches；
- total updates：6400/run，核心矩阵总计96000；
- optimizer：AdamW，max/min LR=`3e-4/3e-5`，warmup=5% updates，exact update-index cosine；
- betas=`0.9/0.999`，eps=`1e-8`，weight decay=`1e-4`，grad clip=1.0；
- AMP=`bfloat16`、TF32=false、cuDNN benchmark=false；
- early stopping=false、resume=false。

每次命令只允许一个candidate/seed。不存在`--all`、batch/device/LR/epoch/output override或自动重跑入口。15项一旦开始，除OOM、非有限、lifecycle、identity、数据或环境失败外均须完成。

## 3. Checkpoint、validation与产物

每个epoch完整执行train后读取完整validation：

1. 计算冻结validation core loss；
2. 收集完整validation prediction并计算sample-direct Local RR MAE；
3. 仅当新Local RR严格小于历史best时更新`checkpoint_best_local_rr.pt`；相等不更新；
4. 每epoch原子更新`train_history.csv`与`lifecycle.json`；
5. 80 epochs结束保存`checkpoint_final.pt`；随后只加载本run的best checkpoint做一次完整validation metrics/summary。

固定run目录：

```text
runs/resp_temporal_v1/formal/<candidate_id>/seed_<seed>/
```

目录必须预先不存在，禁止覆盖与resume。至少保存resolved config、run/data manifest、仅含train/validation metadata的audit、optimizer groups、history、best/final checkpoints、2675行逐sample validation metrics、summary、runtime、formal receipt及receipt SHA-256。Receipt必须闭合count、artifact hash、history/metrics/checkpoint finite/null计数及access flags；允许指标按冻结eligibility产生null，但不得出现Inf，history与checkpoint不得有NaN/Inf。Failure保留目录与失败lifecycle，不删除、不用同identity重跑；后续处理必须先修订协议。

## 4. Fail-closed preflight

创建run目录前必须全部通过：

- 干净Git；formal protocol/plan、signal/candidate/CPU/GPU locks及candidate config hashes闭合；
- v2 GPU receipt/manifest hashes与`5/5 passed at 128×1`闭合；
- `LD_LIBRARY_PATH`与`LD_PRELOAD`未设置；CUDA设备、cuDNN=92000与LSTM canary通过；
- candidate/seed属于15项allowlist；对应run目录不存在；
- 依赖版本满足冻结要求。

数据只在preflight通过、run目录建立并写入`running` lifecycle后读取。允许读取共享index metadata；只有train/validation行可进入research-v2适配器与NPZ signal reader。数据加载后必须确认上述window/samp/batch/row-hash identity及row IDs不重叠；否则fail closed。

## 5. 执行与开放条件

唯一入口：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python scripts/train_resp_temporal_v1.py --plan configs/resp_temporal_v1/formal_v1.yaml --candidate-id <LOCKED_ID> --seed <LOCKED_SEED>
```

Codex不执行命令。用户按提供顺序逐run执行并返回receipt。完成15/15前不得汇总家族质量，不得依据partial结果形成winner或停止其他正常arm。15/15完成并验收后才实现validation summarizer；research-test不自动开放。
