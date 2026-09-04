# W0 时频条件功能证据整理协议

协议 ID：`paper-p5-w0-functional-evidence-v1-20260905`；日期：2026-09-05。

状态：**P5 已完成并冻结；既有 P−1 足以形成论文功能证据，不启动可选局部频率×时间干预。**

## 1. 目标与边界

本阶段只整理已经冻结的 CRD-TF-W v2 P−1 validation 功能干预，不重新运行 checkpoint inference。目标是为论文提供：

- FULL 相对 FiLM 路径、频率遮挡和时间负对照的五主指标变化；
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

论文主表至少保留 CONDITION_OFF、四个频率 view 与两个时间负对照；BETA_ONLY/GAMMA_ONLY 在完整 delta 表和图中保留。

## 3. 统计合同

每个 intervention 与同 seed FULL 配对：

- 四个 error：`(intervention − FULL) / FULL`，正值表示恶化；
- PCC：`FULL − intervention`，正值表示下降；
- 报告 aggregate three-seed mean 的变化、paired-seed 变化 mean ± sample SD，以及恶化/改善 seed 数；
- 不构造加权总分、p-value 或 intervention 排名。

FiLM correction 表在每个 seed 内按 2675 samples 直接均值，再对三个 seed 报告 arithmetic mean 与 sample SD (`ddof=1`)；
不做 `samp_id` 分析。

## 4. 产物与执行

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

## 5. 冻结回执

正式汇总从干净 commit `4105e53f3a3b7b8ac164f0e2d54cb5ca20162e38` 执行，输出目录为第 4 节固定目录。
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

## 6. 证据边界与决定

时间平移在五项主指标上均为 3/3 seed 恶化，时间均值化也稳定损害 global modulation 与 PCC，支持模型确实利用 W 的
时间排列。RESP 遮挡稳定损害 RR、trajectory 与 PCC；carrier 遮挡则通常改善 Whole RR、同时损害 trajectory/global/PCC，
且 CARRIER_H 对 global/PCC 的影响最强，说明频率信息呈任务属性依赖的 trade-off。CONDITION_OFF 同样改善 Whole RR，
但稳定损害 global/PCC，因此这些结果是冻结模型对条件信息的功能敏感性证据，不是所有指标同向获益或频带因果归因。

现有十项干预已经覆盖 FiLM 路径、时间排列、时间均值及预注册频率 views，并给出三 seed 方向和 corrected FiLM 分布；
论文表/图需求已满足。故 P5 在此关闭，不授权第 8.2 节可选局部 `4×6` inference。
