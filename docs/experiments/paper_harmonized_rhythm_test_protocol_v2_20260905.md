# 独立测试集节律指标统一补充汇总 v2

协议 ID：`paper-harmonized-rhythm-test-v2-20260905`；日期：2026-09-05。

状态：**实现已完成，等待从干净 commit 执行一次正式只读汇总。**

## 1. 修正范围

v1 的数值统计、输入 hash、35-arm / 101-record 矩阵、逐 row identity、IBI 支撑集合分析和描述性结论均已通过核验。
但 v1 为 center30/60/90 输出了归一化的 checkpoint selector 标签，没有保留 formal manifest 中的精确字符串。

v2 只修正三项 selector provenance：

| 任务 | 精确 selector |
|---|---|
| center30 | `full_validation_center30_rr_mae_bpm_strict_lower_tie_earlier` |
| center60 | `full_validation_center_rr_mae_bpm_strict_lower_tie_earlier` |
| center90 | `full_validation_center90_rr_mae_bpm_strict_lower_tie_earlier` |

实现从每项冻结 evaluation receipt 读取 formal run path 与 formal manifest SHA-256，再从 formal manifest 核验并写出
selector。任何路径、hash 或 selector 不一致均使汇总失败。

## 2. 输出与证据身份

v1 目录保持冻结。v2 使用新目录：

```text
runs/paper_evidence_v1/harmonized_rhythm_test_summary_v2/
```

Receipt 与 manifest 显式登记 supersedes 范围为三个中心任务的 selector 标签，并固定
`numeric_metrics_changed=false`。v2 必须核对：

- `native_whole_rr_ibi_all_arms.csv`、`equal_input_output_rhythm_ladder.csv` 与
  `per_checkpoint_rhythm_summary.csv` 中除 `checkpoint_selector` 外的共有列与 v1 逐值相同；
- `equal_io_support_overlap.csv` 与 v1 byte-identical；
- 35 arms、9 条等长轨迹、101 条评价记录、233310 条逐样本输入保持一致；
- 原 v1 的 IBI/RR、支撑集合和解释边界全部延续。

## 3. 执行边界

固定命令：

```bash
./.venv/bin/python -m pytest tests/test_paper_harmonized_rhythm_test_summary.py -q
./.venv/bin/python scripts/summarize_paper_harmonized_rhythm_test_v2.py
```

正式执行只读取冻结 CSV、summary 与 provenance JSON/YAML，包括 evaluation receipt 和由其 hash 锁定的 formal manifest。
不读取 checkpoint、dataset/index、波形或 target array，不执行训练、推理、GPU、cache 或 benchmark。
