# W0 时频条件功能证据整理协议

协议 ID：`paper-p5-w0-functional-evidence-v1-20260905`；日期：2026-09-05。

状态：**P5 已完成并冻结；既有 P−1 足以形成论文功能证据，不启动可选局部频率×时间干预。**

## 1. 目标与边界

本阶段只整理已经冻结的 CRD-TF-W v2 P−1 validation 功能干预，不重新运行 checkpoint inference。目标是为论文提供：

- FULL 相对 FiLM 路径、频带仅保留 view 和时间负对照的五主指标变化；
- 三 seed 方向与 arithmetic mean ± sample SD；
- correction 后的 FiLM gamma/beta 幅值、时间变化和 saturation statistics；
- 一张 error 相对恶化与 PCC 绝对下降分轴显示的功能证据图。

所有结果属于已联合训练 W0 的冻结功能敏感性证据，不解释为重新训练机制或频带的因果效果，不参与模型、checkpoint、
窗口或频带选择。

## 2. 冻结输入

| 输入 | SHA-256 |
|---|---|
| P−1 source manifest | `249c761799b1f8020a77ed51875985776718d9cf1e9701900ce5dae783492f3f` |
| `intervention_summary.csv` | `e93881e56441759ebbf4b8c78c0fb6e91921bf2e7333ae66464d3c9feefd8d38` |
| `p_minus_1_decision.json` | `6aae72dd32b09f0ac5c5a5151f78510ec292d7b2c8942feddd0b751bc0787f4a` |
| correction manifest | `1c6f7218c280b2a1579b2169a2b4752d7c10416d89de67ed5e9a1cc58d104463` |
| `film_statistics_corrected.csv` | `2b7a4edef8e9828356a336c3c1ec7880235914207b00d164bf016ce1cd7a5203` |

输入固定为 validation、三个 W0 checkpoints、2675 rows/checkpoint。完整干预为 FULL、BETA_ONLY、GAMMA_ONLY、
CONDITION_OFF、RESP、CARRIER、CARRIER_L、CARRIER_H、TIME_MEAN、TIME_SHIFT_30S。

论文主表至少保留 CONDITION_OFF、四个频带仅保留 view 与两个时间负对照；BETA_ONLY/GAMMA_ONLY 在完整 delta 表和图中保留。

四个 view 的操作语义以实际实现为准：`RESP` 仅保留 mapped indices `0..55`（`≤0.8 Hz`），`CARRIER` 仅保留
indices `56..96`（`>0.8 Hz`），`CARRIER_L` 仅保留 indices `56..71`（`0.8–2 Hz`），`CARRIER_H` 仅保留
indices `72..96`（`>2 Hz`）；其余位置置零。因此结果描述统一采用“仅保留某频带”，不能改写成“移除该频带”。

## 3. 与已有重训练消融的证据关系

当前论文证据分为三个互补层次，不能把它们合并成同一种消融：

1. **模型与容量收益：C201 / W / CTRL1。** CRD-TF v1 的完整正式训练与冻结汇总已经证明 W 通过主体质量保护线和
   匹配容量对照的实质改善门槛；CTRL1 只读取 C201 temporal latent，以匹配分支容量而不读取 W。该层回答“引入 W
   条件分支的收益能否仅由新增容量解释”，证据范围和冻结结论见
   `docs/experiments/crd_tf_v1_protocol_20260812.md` 的 §9.2 与 §22。
2. **频带选择后的性能权衡：W1 / W2 重训练。** CRD-TF-W v2 已完成 W1（仅 `≤0.8 Hz`）、W2（仅
   `>0.8 Hz`）和 W3（全频带 49 scales）各三 seed，共 9/9 formal；每项均为 80 epochs / 6400 updates。W1/W2
   从训练阶段应用静态 input-only mask，同时保持 `[97,360]` 输入 shape、W encoder、FiLM、主干、decoder 与
   `1,219,850` 参数不变。该层允许模型适应受限频带，回答“频带选择后可达到的 validation 性能权衡”；W1/W2
   没有进入后续 research-test allowlist，只有 W3 获得独立测试集评价。
3. **冻结模型的功能依赖：当前 P5。** P5 不重训练，而是在三个冻结 W0 checkpoint 上施加 FiLM 路径、频带仅保留和
   时间操作，回答“已训练 W0 的输出对条件路径和时间对齐是否敏感”。它是功能依赖证据，不代替 W1/W2 的适应后性能
   比较，也不用于重新选择频带。

W1/W2 相对 W0 的冻结 validation 变化如下；四项 error 的负值表示改善，PCC 列为绝对变化：

| 重训练模型 | Whole RR | Local RR | trajectory | global | PCC 绝对变化 |
|---|---:|---:|---:|---:|---:|
| W1，仅 `≤0.8 Hz` | `+1.2564%` | `+1.2168%` | `+0.4302%` | `−2.3242%` | `−0.003938` |
| W2，仅 `>0.8 Hz` | `−6.7179%` | `−2.4838%` | `+0.6640%` | `+1.7038%` | `−0.003158` |

W2 的 Whole/Local RR 均为 `3/3 seeds` 改善，但 trajectory/global/PCC 体现另一方向的代价；所以该结果是多属性权衡，
不是单一频带的全面优胜。

P−1 的 BETA_ONLY/GAMMA_ONLY 属于冻结 W0 的路径干预。两者均未满足预设开放条件，冻结决策为
`retain_film_no_p2_training`，ADD/SCALE 重训练没有执行。P5 只沿用该路径干预结果，不能把它写成 ADD/SCALE
重训练消融。

## 4. 统计合同

每个 intervention 与同 seed FULL 配对：

- 四个 error：`(intervention − FULL) / FULL`，正值表示恶化；
- PCC：`FULL − intervention`，正值表示下降；
- 报告 aggregate three-seed mean 的变化、paired-seed 变化 mean ± sample SD，以及恶化/改善 seed 数；
- 不构造加权总分、p-value 或 intervention 排名。

FiLM correction 表在每个 seed 内按 2675 samples 直接均值，再对三个 seed 报告 arithmetic mean 与 sample SD (`ddof=1`)；
不做 `samp_id` 分析。

## 5. 产物与执行

固定不可覆盖目录：

```text
runs/paper_evidence_v1/p5_w0_functional_evidence/
```

产物包括 seed/aggregate delta、含 aggregate 变化与 paired-seed mean ± sample SD 的论文宽表、corrected FiLM
seed/three-seed 表、PNG panel、receipt 与 manifest。

```bash
./.venv/bin/python -m pytest tests/test_paper_p5_functional_evidence.py -q
./.venv/bin/python scripts/summarize_paper_p5_functional_evidence_v1.py
```

实现与汇总只读取第 2 节文件；不读取逐样本 prediction metrics、checkpoint、dataset/index、signal/target、cache 或 test，
不执行模型 inference、训练、GPU 或 benchmark。

## 6. 冻结回执

正式汇总从干净 commit `4105e53f3a3b7b8ac164f0e2d54cb5ca20162e38` 执行，输出目录为第 5 节固定目录。
`summary_receipt.json` SHA-256 为
`d93e54ca6798671d068a0c82c97a530af3cb83df525d1068b3563b093c19292d`，`artifact_manifest.json` SHA-256 为
`82e4c10c10e85390b2eb3f61657c036c180e97a3927a0d0c15ddf666aea51187`。五项冻结输入哈希均保持不变；共生成
135 行 paired-seed delta、45 行 aggregate delta、7 行论文主表及 8 项 corrected FiLM statistics。

相对 FULL 的 aggregate three-seed mean 变化如下；四个 error 为相对变化，PCC 为绝对下降，括号中为恶化 seed 数：

| 干预 | Whole RR | Local RR | trajectory | global | PCC 下降 |
|---|---:|---:|---:|---:|---:|
| CONDITION_OFF | `−4.45% (0/3)` | `−0.63% (1/3)` | `+1.95% (2/3)` | `+20.40% (3/3)` | `+0.00822 (3/3)` |
| RESP | `+3.75% (3/3)` | `+5.47% (3/3)` | `+8.39% (3/3)` | `+0.84% (2/3)` | `+0.00934 (3/3)` |
| CARRIER | `−7.04% (0/3)` | `−1.36% (1/3)` | `+2.73% (3/3)` | `+10.41% (2/3)` | `+0.01478 (3/3)` |
| CARRIER_L | `−1.59% (1/3)` | `+1.23% (2/3)` | `+4.55% (2/3)` | `+5.03% (2/3)` | `+0.01633 (3/3)` |
| CARRIER_H | `−3.20% (0/3)` | `+0.15% (1/3)` | `+4.39% (3/3)` | `+16.41% (3/3)` | `+0.02018 (3/3)` |
| TIME_MEAN | `−2.46% (1/3)` | `+1.15% (2/3)` | `+0.25% (2/3)` | `+25.97% (3/3)` | `+0.01210 (3/3)` |
| TIME_SHIFT_30S | `+3.99% (3/3)` | `+6.60% (3/3)` | `+23.05% (3/3)` | `+24.10% (3/3)` | `+0.02382 (3/3)` |

Corrected FiLM statistics 的 sample-direct three-seed mean ± sample SD 为：`mean|gamma|=0.266855±0.028878`、
`median|gamma|=0.280095±0.039790`、`mean|beta|=0.243023±0.028429`、`median|beta|=0.245505±0.036252`、
`mean|Δgamma|=0.022615±0.002356`、`mean|Δbeta|=0.023967±0.004887`、gamma/beta saturation fraction=
`0.026338±0.009668 / 0.020325±0.011592`。

## 7. 证据边界与决定

时间平移在五项主指标上均为 3/3 seed 恶化，时间均值化也稳定损害 global modulation 与 PCC，支持模型确实利用 W 的
时间排列。仅保留 RESP 低频时 RR、trajectory 与 PCC 稳定变差；仅保留 CARRIER 高频时 Whole RR 通常改善而
trajectory/global/PCC 变差，CARRIER_L 与 CARRIER_H 子 view 也呈指标间不同方向。CONDITION_OFF 同样改善 Whole RR，
但稳定损害 global/PCC。因此这些结果是冻结模型对条件信息的功能敏感性证据，不是所有指标同向获益；keep-only view
相对 FULL 的变化也不能反推“移除某频带”的损害，更不能作为频带因果归因。

现有十项干预已经覆盖 FiLM 路径、时间排列、时间均值及预注册频率 views，并给出三 seed 方向和 corrected FiLM 分布；
论文表/图需求已满足。故 P5 在此关闭，不授权总协议第 8.2 节可选局部 `4×6` inference。
