# P6 多属性波形与 RR 区间协议

协议 ID：`paper-p6-multi-attribute-v1-20260905`；日期：2026-09-05。

状态：**P0 实现与定向 CPU 验收通过；等待从干净 commit 生成 train/validation target-only 属性，test 访问保持关闭。**

## 1. 研究问题与证据边界

P6 回答两个描述性问题：冻结 W0 在不同失败属性下的波形表现是什么，以及预注册方法的五主指标如何随 target RR 区间
变化。波形例子只从 validation 按数值规则选择；RR 区间边界只由完整 admitted train targets 冻结。重叠窗口仍按
sample-direct mean 汇总，不解释为相互独立的临床样本。

P6 不训练、不重选 checkpoint、模型、方法、频带或窗口，也不构造总分或显著性检验。Test target-only 属性属于新的 test
访问，在 train cutpoints、实现、10 方法 allowlist 和输出 schema 冻结后仍须单独授权。

## 2. 固定合同

机器可读合同固定为 `configs/paper_evidence_v1/p6_multi_attribute_v1.json`，SHA-256=
`0c22bb56b8d5c54b604a4b7e2f2064f82c699df20d703744e6cea0cc243742ff`。它锁定：

- admitted train/validation 为 `10141 / 2675` rows、`32 / 7` 个 `samp_id`；
- 100 Hz、18000 点 target、现行 `0.05–0.70 Hz` dominant spectral Whole RR；
- train RR 的 `1/3 / 2/3` linear quantiles，区间为 `low≤q1`、`q1<medium≤q2`、`high>q2`；
- P0 primary table 的全部 10 个方法，顺序不变；
- W0 三个 validation-selected checkpoints：seed `20260811/12/13`，epoch `13/15/14`；
- 五个示例类别与 `5 rows × 3 checkpoints = 15` 次 validation forward，不构造 ensemble。

## 3. Target-only 属性

专用入口只读取 target NPZ，不读取 BCG、checkpoint 或 W cache。每个 row 保存：`dataset_row_id`、split、`samp_id`、
float32 target SHA-256、target RR、RR eligibility、target envelope modulation 和 train-frozen RR stratum。完整 train 的全部
target RR 必须 finite/eligible，否则停止，不静默过滤后重估 cutpoints。

固定不可覆盖输出为 `runs/paper_evidence_v1/p6_target_attributes/`：

```bash
./.venv/bin/python scripts/build_paper_p6_target_attributes_v1.py \
  --confirm-target-attribute-build
```

该命令读取约 12816 个 180 s target，属于全量 CPU 数据生成，由用户执行。读取统一 dataset index 时会看到 split metadata，
但本阶段不打开 test target NPZ；receipt 必须分别记录二者。

## 4. Validation 波形选样

三个 W0 冻结 `metrics.csv` 按 `dataset_row_id` 一对一对齐并取 arithmetic seed mean。定义：

- `rate_score`：Whole RR 与 Local RR error 的 normalized average ranks 均值；
- `effort_score`：trajectory error、global error 与 `−PCC` 的 normalized average ranks 均值；
- normalized rank 为 `(average_rank−1)/(N−1)`，越大表示越困难。

按固定顺序自适应排除已选 row：典型取距 `(0.5,0.5)` 最近；RR 困难取距 `(1,0)` 最近；努力困难取距 `(0,1)`
最近；RR—努力不一致取 `|rate−effort|` 最大；联合失败取距 `(1,1)` 最近。所有 tie 取较小 `dataset_row_id`。
保存五类的全部候选、目标距离/分数、排除状态、最终 row、选择原因和 rule SHA-256；禁止浏览波形后换样本。

```bash
./.venv/bin/python scripts/select_paper_p6_validation_waveforms_v1.py
```

固定不可覆盖输出为 `runs/paper_evidence_v1/p6_waveform_selection/`。

## 5. Validation 波形导出

同五个 rows 对三个冻结 W0 checkpoint 全部执行 forward。输出冻结 `bcg_rawish_segment_soft_z_key` 对应的 BCG input、
whole 180 s target、中心 `[60,120) s`
target、三个 seed prediction、canonical log-RMS envelope、target RR、五主指标、PNG panel、source/row/checkpoint hashes 和
access flags。逐 seed 新计算五主指标必须以绝对容差 `1e-6` 锚定原冻结 validation metrics。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
CUDA_VISIBLE_DEVICES=0 \
./.venv/bin/python scripts/export_paper_p6_validation_waveforms_v1.py \
  --device cuda:0 \
  --confirm-gpu-export
```

固定不可覆盖输出为 `runs/paper_evidence_v1/p6_waveform_export/`。该命令只做 15 个 validation row-checkpoint pairs，
由用户执行；不读取 test，不形成新的模型评价或选择。

## 6. Test RR 区间阶段

后续阶段固定应用 train cutpoints，并把 target-only test 属性按 `dataset_row_id` 一对一连接 10 个冻结方法结果。必须审计
2310-row 集合、重复、缺失、split、target hash、`samp_id` coverage 和每区间窗口/`samp_id` 数；不得调用任何已关闭
checkpoint evaluator。该入口和产物在获得新的 test 授权前保持未实现/关闭，避免实现阶段误触 test。

## 7. 当前验收命令

```bash
./.venv/bin/python -m pytest tests/test_paper_p6_multi_attribute.py -q
./.venv/bin/python -m py_compile \
  resp_train/paper_evidence/p6_multi_attribute.py \
  scripts/build_paper_p6_target_attributes_v1.py \
  scripts/select_paper_p6_validation_waveforms_v1.py \
  scripts/export_paper_p6_validation_waveforms_v1.py
```
