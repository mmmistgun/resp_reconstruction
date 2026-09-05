# P6 多属性波形与 RR 区间协议

协议 ID：`paper-p6-multi-attribute-v1-20260905`；日期：2026-09-05。

状态：**P1 target-only 属性与确定性 validation 选样已冻结；等待 15-pair validation GPU 波形导出，test 访问保持关闭。**

## 1. 研究问题与证据边界

P6 回答两个描述性问题：冻结 W0 在不同失败属性下的波形表现是什么，以及预注册方法的五主指标如何随 target RR 区间
变化。波形例子只从 validation 按数值规则选择；RR 区间边界只由完整 admitted train targets 冻结。重叠窗口仍按
sample-direct mean 汇总，不解释为相互独立的临床样本。

P6 不训练、不重选 checkpoint、模型、方法、频带或窗口，也不构造总分或显著性检验。Test target-only 属性属于新的 test
访问，在 train cutpoints、实现、10 方法 allowlist 和输出 schema 冻结后仍须单独授权。

## 2. 固定合同

机器可读合同固定为 `configs/paper_evidence_v1/p6_multi_attribute_v1.json`，SHA-256=
`0b9c861d3934d1d5f247afb5737838a2d885ac0e1c454400dd7114d79e42af2c`。它锁定：

- admitted train/validation 为 `10141 / 2675` rows、`32 / 7` 个 `samp_id`；
- 100 Hz、18000 点 target、现行 `0.05–0.70 Hz` dominant spectral Whole RR；
- train RR 的 `1/3 / 2/3` linear quantiles，区间为 `low≤q1`、`q1<medium≤q2`、`high>q2`；
- P0 primary table 的全部 10 个方法，顺序不变；
- W0 三个 validation-selected checkpoints：seed `20260811/12/13`，epoch `13/15/14`；
- 五个示例类别与 `5 rows × 3 checkpoints = 15` 个保存的 validation 输出，不构造 ensemble。

该合同替代实现初版 SHA-256=`0c22bb56b8d5c54b604a4b7e2f2064f82c699df20d703744e6cea0cc243742ff`；唯一变更是第 5 节
历史 batch-shape 数值重放。Train/validation target attributes、RR cutpoints、方法/checkpoint 矩阵、五个 rows、selection
rule 和 `1e-6` 锚点均不变，初版合同下已冻结的 target/selection 产物继续有效。

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

历史冻结 metrics 使用 `batch_size=128`，最后一个 validation batch 为 115。为保持 BF16/Mamba 数值路径，导出使用两个
replay batches/checkpoint：128-shape batch 固定 slots `37/81/95/105`，115-shape batch 固定 slot `5`。Padding 只重复
五个已选 rows，不读取其他 validation rows；只保存五个目标输出，padding 输出丢弃。实际处理量为每 checkpoint
`128+115=243` batch elements、三 checkpoints 共 729，但科学矩阵仍为 15 个 row-checkpoint pairs。

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

## 8. P1 冻结结果

Target-only 属性从干净 commit `6220d63c794ff51c0aaa3f9da36492dcd18d4973` 完成。Receipt/manifest SHA-256 为
`887374b0d99c21901f143f2a13bb28ee8ab8658f989faa21289576b464de03fe / bfd5ca354bbd3a84e5f5d2f979ebdf2716297036719db212fbc0377fcbb888ee`；
10141 train 与 2675 validation rows 全部 target RR eligible 且属性 finite，row/`samp_id`/source hashes 通过。

Train-frozen cutpoints 为 `q1=14.327967747931218 bpm`、`q2=17.188694745285627 bpm`。Train 的 low/medium/high
rows 为 `3381 / 3380 / 3380`；应用相同边界后的 validation rows 为 `1011 / 474 / 1190`。Train/validation target RR
观测范围分别为 `3.0000–24.8585 / 3.0000–24.7733 bpm`；`3 bpm` 是冻结 `0.05 Hz` 下边界，不在结果可见后追加过滤。
本阶段读取 train/validation target 和统一 index metadata，没有读取 test target、BCG、checkpoint 或 W cache。

确定性选样随后从同一干净 commit 完成。Receipt/manifest SHA-256 为
`c5f72423ff236364555bfac6869117fcd0ff11e8249ea1537817e1f5d7dc91e3 / 7937c9d085c35139993bb7a482797d6f9ffb932d5da92d634a61890b96e8a784`，
selection rule SHA-256=`23bd6ab081d457393bf9298b64ba8450ca7c54472cc0ac45ef3e6ce842e11e84`。五类固定 rows 为：

| 类别 | dataset_row_id | rate score | effort score | target RR | stratum |
|---|---:|---:|---:|---:|---|
| typical | 12429 | 0.491773 | 0.510347 | 18.566685 | high |
| rr_difficult | 9700 | 0.761219 | 0.308028 | 21.230867 | high |
| effort_difficult | 16442 | 0.184368 | 0.749065 | 13.067503 | low |
| rr_effort_inconsistent | 16273 | 0.244390 | 0.805285 | 14.076622 | low |
| joint_failure | 10787 | 0.995886 | 0.992645 | 12.568559 | low |

候选表保存 `5×2675=13375` rows，每类恰有一个 selected row，五个 rows 互异；候选和选中表不携带 `samp_id`。
选样只读取冻结 W0 validation metrics 与 target attributes，没有读取任何 waveform、checkpoint 或 test，也没有执行 inference。

## 9. 首次 GPU lifecycle 与修订

初版导出从干净 commit `2a7adbb` 启动，五个 rows 直接组成 batch=5。Seed `20260811` 在 row `16442` 的
`local_rr_mae_bpm` 未通过 `1e-6` 历史锚点，失败记录保留于
`runs/paper_evidence_v1/.p6_waveform_export.incomplete_yqlz_snw/failure.json`。此前 checkpoint/config/source/row/target/W
身份均已通过；失败发生在派生 Local RR 锚点，而历史 metrics 来自 batch=128 的 BF16/Mamba validation 数值路径，现有
证据与 batch-shape 数值路径差异一致，但须由 replay 是否恢复锚点进一步验证。

没有放宽容差、换 row、重选 checkpoint 或删除失败记录。用户确认后，执行合同仅改为第 5 节历史 batch-shape replay；
padding 不增加数据访问范围。相关 P6/respiration-metrics 定向 CPU 测试现为 25 项通过。

Replay 首次执行从干净 commit `abdeb61` 启动，仍在 seed `20260811`、row `16442` 的 Local RR 锚点停止：
observed=`0.039869667187182621`、expected=`0.039871286044205499`、绝对差=`1.6188570228781174e-6`，略高于
`1e-6`；失败记录保留于 `.p6_waveform_export.incomplete_ggddr8_o/failure.json`。该数值很小，但不能只根据首个超限值
事后把门槛改为 `2e-6`。

因此下一次执行保持同一合同、rows、replay 和门槛，改为完成三个 checkpoints 的全部 `15×5=75` 项锚点比较后统一
判定。若存在超限，incomplete lifecycle 将保存 `validation_anchor_deltas.csv`，报告全矩阵 failed count、最大绝对差及
对应 seed/row/metric；在看到完整误差分布前不改变验收口径。
