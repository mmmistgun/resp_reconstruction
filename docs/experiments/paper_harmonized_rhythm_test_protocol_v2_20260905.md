# 独立测试集节律指标统一补充汇总 v2

协议 ID：`paper-harmonized-rhythm-test-v2-20260905`；日期：2026-09-05。

状态：**已从干净实现 commit 完成一次正式只读汇总并冻结。v2 精确恢复三个中心任务的 formal selector；
全部数值、矩阵、支撑集合与 v1 一致。**

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

## 4. 正式执行与冻结身份

正式命令从干净实现 commit `7c32f4e` 成功执行。V2 新增读取 60 份 evaluation receipt 与由其 SHA-256 锁定的
60 份 formal artifact manifest，共登记 496 个唯一输入文件；输入在汇总前后保持不变。

冻结输出为：

```text
runs/paper_evidence_v1/harmonized_rhythm_test_summary_v2/
```

| 产物 | SHA-256 |
|---|---|
| `native_whole_rr_ibi_all_arms.csv` | `c7fb06803420180afe0f474e639adb75601c70a8d0947500f4dc6715040382b1` |
| `equal_input_output_rhythm_ladder.csv` | `9fdb28dec7b43c57cfa34359d658208883d0bd3ce83e856ec4a595e72e66b50b` |
| `per_checkpoint_rhythm_summary.csv` | `4246687e266e049192d9e7c3f4d3df76ece36eeaf0ada8300373fd59f2fcfe68` |
| `equal_io_support_overlap.csv` | `2013462eb376fbc5995accb3c48ad1b43748f7bc1bf9babb69e9b17c0d1e8796` |
| `rhythm_comparability.json` | `68ee464929e16561d54e0427c54a869d4b592bf14c59abd58c3225a4940880b0` |
| `summary_receipt.json` | `f70a72bf5081a06b0b13c8697cbcc99f52a7f7fbb183dbd6f070f66b4314205b` |
| `artifact_manifest.json` | `0cc7a0a7c55a842c54ab49484541e97063d6863eec96f0ff91b85f1ba54f3107` |

独立核验确认：

- 三份含 selector 的统计 CSV 删除 `checkpoint_selector` 列后与 v1 逐值相同；
- `equal_io_support_overlap.csv` 与 v1 byte-identical；
- 35 arms、9 条等长轨迹、101 条评价记录、99 个学习 checkpoints、2 个确定性记录和 233310 条逐样本输入一致；
- selector 唯一值精确包含 center30/60/90 三个 formal selector、历史 Local RR selector 与确定性方法的
  `not_applicable_deterministic`；
- manifest 全部文件 size/SHA-256 复核通过；定向 CPU 测试为 `21 passed`。

V2 作为当前节律统一汇总引用；v1 保留为原始审计记录。数值结论继续采用 v1 第 6 节的表格与解释边界。
