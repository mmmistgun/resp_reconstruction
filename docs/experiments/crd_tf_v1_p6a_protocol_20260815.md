# CRD-TF v1 P6a 三路加法与有界门控实验协议

日期：2026-08-15

状态：**协议与工程实现已建立；CPU 定向测试通过；CUDA synthetic / batch-128 acceptance 未执行，9-run formal 关闭**

协议标识：`crd-tf-v1-p6a-validation-development-20260815`

## 1. 权威性与证据边界

本文由 `docs/experiments/loss_metrics_restart_plan_20260729.md` 第 47 节纳入当前唯一实验协议，是 `docs/experiments/crd_tf_v1_protocol_20260812.md` 的 result-informed P6a 后续。发生冲突时以主协议为准。

P5 冻结候选并集为 `M / W / MS`。P6a 只研究这三个已选表示是否能通过三路加法或有界门控形成进一步收益，不重新选择 P4 arm，不读取 research-test，不改变数据、split、target、loss、五项 primary、Local-RR checkpoint selector 或固定 cache。全部证据仍属于现有 validation-development evidence，不是无偏 held-out 或强泛化证据。

Local cross-attention、其他 gate、更多表示组合、LR/batch 搜索和 research-test 归为 P6b 或后续独立协议，当前关闭。

## 2. 第一性问题与 9-run 矩阵

P6a 固定三个 arm，每个使用 seed `20260811 / 20260812 / 20260813`，共 9 个独立 run：

| variant | 简称 | condition 来源 | 机制 | 直接回答的问题 |
|---|---|---|---|---|
| `crd_tf401_mws_add` | MWS-ADD | M/W/S | 原 Stage-1 独立加法 FiLM | 在已成功 MS 上加入 W 是否有增量 |
| `crd_tf402_mws_gate` | MWS-GATE | M/W/S | temporal-latent 驱动的三路有界 gate | gate 是否优于相同三路表示的直接加法 |
| `crd_tf403_ctrl_gate` | CTRL-GATE | 3 个 temporal-only capacity branch | 与 MWS-GATE 完全相同的 gate | MWS-GATE 收益是否超出匹配容量与 gate 机制 |

既有 `crd_tf_ctrl3` 继续作为 MWS-ADD 的历史匹配容量对照，不重训。P4 的 15-arm 集合与 P5 完整性定义保持不变；P6a variant 使用独立注册表，不能被加入或回填到 P4/P5 矩阵。

## 3. 冻结结构与参数匹配

三项都复用 C201 from-scratch anchor、M/W/S 既有 branch、相同 cache、FiLM 位置和 branch checkpoint `chunk=8`。每路 condition 仍为：

```text
delta_gamma_r = 0.5 * tanh(gamma_r)
delta_beta_r  = 0.5 * tanh(beta_r)
```

MWS-ADD 保持：

```text
z' = z * (1 + sum_r delta_gamma_r) + sum_r delta_beta_r
```

两个 gated arms 的 gate 只读取共享 temporal latent `z`，不得读取 cache 或 target：

```text
GroupNorm(12,96)
-> depthwise Conv1d(96,96,k=5,pad=2,bias=false)
-> Conv1d(96,128,k=1,bias=false)
-> SiLU
-> Conv1d(128,3,k=1,bias=true,zero-init)
```

第 `r` 路 factor 与条件和固定为：

```text
g_r = 1 + 0.5 * tanh(logit_r),  g_r in [0.5, 1.5]
z' = z * (1 + sum_r g_r * delta_gamma_r) + sum_r g_r * delta_beta_r
```

Gate 共 `13,347` 个 trainable parameters，zero-init 时 `g_r` 逐点严格为 1。因此 MWS-GATE 与 MWS-ADD、CTRL-GATE 与既有 CTRL3 在初始化时分别同函数；所有 arm 的 branch final projection 仍为 zero-init，初始 waveform 与相同 seed C201 一致。

冻结的 C201 增量参数为：

| arm | condition branches | gate | 总增量 |
|---|---:|---:|---:|
| MWS-ADD | 450,080 | 0 | 450,080 |
| MWS-GATE | 450,080 | 13,347 | 463,427 |
| CTRL-GATE | 449,856 | 13,347 | 463,203 |

两个 gated arms 只差 `224` parameters（`0.048%`），且 gate 参数完全相同。MWS-ADD 与既有 CTRL3 也只差 `224` parameters。

## 4. Batch、LR 与 early stopping

P6a 不调整 batch 或 LR：physical batch `128`、gradient accumulation `1`、effective batch `128`；AdamW、`3e-4 -> 3e-5`、5% exact-update warmup + cosine 均沿用 P4。梯度累计不是当前优化变量，只在统一 batch-128 工程验收失败后由用户决定是否对全部 P6a arm 使用同一 fallback。

最大预算仍为 `80 epochs / 6400 planned updates`。LR schedule 始终按完整 6400 updates 计算；提前停止不会把 cosine 压缩到较短区间。Checkpoint selector 仍是每 epoch 完整 validation 的 Local RR MAE。

对 P4 全部 45 条、每条 80 epochs 的 history 离线回放结果为：

| patience | 错过原全程最优的 runs | 最大 Local-RR 差 | 平均停止 epoch | 最晚停止 epoch |
|---:|---:|---:|---:|---:|
| 10 | 5 | 0.017200 | 22.09 | 34 |
| 15 | 4 | 0.017200 | 27.38 | 39 |
| 20 | 3 | 0.008923 | 32.73 | 46 |
| 25 | 1 | 0.008923 | 38.71 | 56 |
| 30 | 0 | 0.000000 | 44.31 | 66 |

因此 P6a 冻结 `early_stopping_enabled=true / patience=30 / min_delta=0.0`。相等值不算改善；连续 30 个 epoch 未严格改善后停止。该策略在 P4 回放中不改变任何 selected checkpoint，但新的三路结构仍可能产生 P4 未覆盖的更晚最优，这是明确残余风险。Patience、LR schedule、最大 epoch 不得根据任一 P6a 中间结果修改。

## 5. 工程准入与正式执行开关

正式训练前必须在同一个干净 commit 完成：

1. 三个 variant 的 CUDA bf16 batch-1 forward/loss/backward，要求 output/input/parameter gradients finite、active branch projection gradient nonzero、dropout=0、gate 初始 factor=1、精确参数增量通过；
2. `MWS-GATE` 与 `CTRL-GATE` 各一次 `128 train / 32 validation / one update / physical batch 128×1` acceptance，要求完整 lifecycle、finite checkpoint/metrics、prediction nondegenerate 与显存可接受；
3. 将 receipt、run 路径、commit 与显存结果写回本文并建立 formal preflight 后，才开放 9-run 队列。

当前 `train_crd.py` 对 P6a formal 主动拒绝执行，避免在 acceptance 前误启动。CUDA 工程任务由用户执行；Codex 不代跑长时间 GPU 工作。

## 6. 正式配置、输出与执行纪律

三个正式配置彼此独立：

```text
configs/crd_tf_v1/crd_tf401_mws_add_formal.yaml
configs/crd_tf_v1/crd_tf402_mws_gate_formal.yaml
configs/crd_tf_v1/crd_tf403_ctrl_gate_formal.yaml
```

正式输出预留为：

```text
runs/crd_tf_v1/p6a_formal/<variant>/seed_<seed>/<timestamp>/
```

不建立统一矩阵 runner。用户可在两张 GPU 上各自用 shell `for` 依次运行一组 seeds；任一失败保留 partial 并先审计，不覆盖、不自动跳过、不因 validation 中间结果改变剩余计划。

## 7. P6a 结果判定

五项 primary 分别报告 mean、sample SD、paired-seed 差和方向计数，不构造总分，也不用 secondary 打破平局。比较固定为：

- MWS-ADD vs MS：W 在 MS 上的增量；
- MWS-ADD vs 既有 CTRL3：三表示相对匹配容量；
- MWS-GATE vs MWS-ADD：gate 机制增量；
- MWS-GATE vs CTRL-GATE：表示信息相对匹配 gate+容量；
- CTRL-GATE vs 既有 CTRL3：gate 本身在 temporal-only control 上的效应。

资格门槛沿用 P5：相对 C201 的 Local RR 不恶化超过 `1.5%`，trajectory 不恶化超过 `1.5%`，signed PCC 下降不超过 `0.005`；实质改善按 error metric 至少 `0.5%` 或 PCC 至少 `0.002`，并要求至少 `2/3` paired seeds 同方向。结果只能选择未来锁定候选，不能自动触发 research-test。

## 8. 需要用户介入的节点

当前下一节点是执行一次三-arm CUDA synthetic，以及两个独立 batch-128 acceptance。用户回传三个输出路径后，Codex负责审计、写回协议并解除或维持 formal gate。只有工程验收通过后，才会给出两张卡上的 9-run `for` 命令。

