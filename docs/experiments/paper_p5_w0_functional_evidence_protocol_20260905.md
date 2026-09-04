# W0 时频条件功能证据整理协议

协议 ID：`paper-p5-w0-functional-evidence-v1-20260905`；日期：2026-09-05。

状态：**P5 只读实现与定向 CPU 验收通过，等待从干净 commit 执行正式冻结汇总。**

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
