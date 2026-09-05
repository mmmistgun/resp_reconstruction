# P6 多属性波形与 RR 区间协议

协议 ID：`paper-p6-multi-attribute-v1-20260905`；日期：2026-09-05。

状态：**Validation 波形与 test RR 区间结果均已冻结；P6 完成并关闭。**

## 1. 研究问题与证据边界

P6 回答两个描述性问题：冻结 W0 在不同失败属性下的波形表现是什么，以及预注册方法的五主指标如何随 target RR 区间
变化。波形例子只从 validation 按数值规则选择；RR 区间边界只由完整 admitted train targets 冻结。重叠窗口仍按
sample-direct mean 汇总，不解释为相互独立的临床样本。

P6 不训练、不重选 checkpoint、模型、方法、频带或窗口，也不构造总分或显著性检验。Test target-only 属性属于新的 test
访问，在 train cutpoints、实现、10 方法 allowlist 和输出 schema 冻结后仍须单独授权。

## 2. 固定合同

机器可读合同固定为 `configs/paper_evidence_v1/p6_multi_attribute_v1.json`，SHA-256=
`c9da7b3407135b4051a13219221ccd9da348c892d9d2bee0abaa229a16f05f02`。它锁定：

- admitted train/validation 为 `10141 / 2675` rows、`32 / 7` 个 `samp_id`；
- 100 Hz、18000 点 target、现行 `0.05–0.70 Hz` dominant spectral Whole RR；
- train RR 的 `1/3 / 2/3` linear quantiles，区间为 `low≤q1`、`q1<medium≤q2`、`high>q2`；
- P0 primary table 的全部 10 个方法，顺序不变；
- W0 三个 validation-selected checkpoints：seed `20260811/12/13`，epoch `13/15/14`；
- 五个示例类别与 `5 rows × 3 checkpoints = 15` 个 validation 输出，不构造 ensemble；
- 导出 batch size 为 5，五主指标与冻结 validation metrics 的绝对一致性容差为 `0.01`。

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
access flags。逐 seed 新计算五主指标与原冻结 validation metrics 做完整 `15×5=75` 项绝对差检查，容差为 `0.01`；
差值表随成功产物保存。导出只读取五个已选 validation rows，每 checkpoint 单个 batch，三 checkpoints 共 15 个 forward
elements。

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

用户已授权该描述性阶段。独立机器合同为 `configs/paper_evidence_v1/p6_rr_strata_v1.json`，SHA-256=
`393ec90c4c582af35bddc8fdfd878bdebbba0fbb804b06922822fbc3c0434ccf`。它固定 2310 个 test windows、8 个
`samp_id`、row-ID SHA-256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`，并直接应用
train-frozen cutpoints `14.327967747931218 / 17.188694745285627 bpm`。

第一阶段只读取 2310 个 test targets，计算 target RR、target envelope modulation 与 target SHA-256，同时核验 test rows
和 train/validation rows 零交集。配置中的 target selector 为 `target_waveform_segment_soft_z_key`，索引解析后的 NPZ array key
固定为 `tho_waveform_segment_soft_z`。它不读取 BCG、W cache、checkpoint 或现有模型指标：

```bash
./.venv/bin/python scripts/build_paper_p6_test_target_attributes_v1.py \
  --confirm-test-target-attribute-build
```

固定输出为 `runs/paper_evidence_v1/p6_test_target_attributes/`。该命令属于全量 test target CPU 读取，由用户执行。

该阶段已从干净 commit `a305bd861385d81d09cef2ba4018cb4c7b992bef` 完成。Receipt/manifest SHA-256=
`d0ecdaa982919a160c73eaec3ac57a7d757d20e2d562e85e9654902457412b53 / eebd4fd38a6f5f142c15914c7a1ea40aac85f59d5df09f26da8b7ef2c68d7a2f`。
2310 rows 全部唯一、RR eligible 且属性 finite；row hash 与冻结身份一致。Low/medium/high 分别包含
`1656 / 462 / 192` windows，覆盖 `8 / 6 / 7` 个 `samp_id`。

第二阶段只读取上述 target-only artifact 与 P0 已冻结的 10 个 primary methods test metrics。来源矩阵为 2 个
deterministic records 加 8 个 learned methods × 3 seeds，共 26 个完整 2310-row records；每个来源文件都由 P0 artifact
audit 的 SHA-256/size 锚定。连接逐项核验 row、split、input set、`samp_id`、coupling state 和 target modulation。

每个 record 在 low/medium/high 内先做 sample-direct mean；learned methods 再对三个 seed means 取 arithmetic mean 与
sample SD (`ddof=1`)，deterministic methods 的 seed SD 保持未定义。输出 78 条 record-stratum rows、30 条
method-stratum rows，以及三个区间的 window/`samp_id` coverage：

```bash
./.venv/bin/python scripts/summarize_paper_p6_test_rr_strata_v1.py
```

固定输出为 `runs/paper_evidence_v1/p6_rr_strata_summary/`。该阶段不重新运行 checkpoint evaluator，不产生 prediction，
不进行模型选择、显著性检验或跨区间总分。

## 7. 当前验收命令

```bash
./.venv/bin/python -m pytest tests/test_paper_p6_multi_attribute.py -q
./.venv/bin/python -m pytest tests/test_paper_p6_rr_strata.py -q
./.venv/bin/python -m py_compile \
  resp_train/paper_evidence/p6_multi_attribute.py \
  resp_train/paper_evidence/p6_rr_strata.py \
  scripts/build_paper_p6_target_attributes_v1.py \
  scripts/build_paper_p6_test_target_attributes_v1.py \
  scripts/select_paper_p6_validation_waveforms_v1.py \
  scripts/export_paper_p6_validation_waveforms_v1.py \
  scripts/summarize_paper_p6_test_rr_strata_v1.py
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

## 9. 导出数值一致性

从干净 commit `582f360fcb0f52910915dc8ec1479d38b2cae1c2` 完成导出。完整 75 项差值全部通过，最大绝对差为
`0.00116557263923589`，低于冻结容差 `0.01`。Receipt/manifest SHA-256 为
`64d8d60cfece8606674e9ecb11e6ca662f6a12f6e4e69f2374680b0fd113b950 / 13ba01f4116385f72dbb2a4f2291ab46bb699f746a2858f49bff0479e43a37be`。

冻结 NPZ 包含 5 个 18000-point BCG/target、3×5 个 18000-point predictions、中心 6000-point targets 与对应 35-point
log-RMS envelopes；全部数值 finite，row/seed 顺序与合同一致。波形 panel 已完成视觉核验，五类标题、中心区间、三 seed
轨迹和 envelope 面板均完整可读。该一致性检查只用于确认 row、checkpoint 与指标管线，不参与模型比较、显著性判断或
结论选择。

## 10. Test RR 区间冻结结果

只读汇总从干净 commit `5cf6acf9328f6c0cc050f9f0f2ad44924255f535` 完成。Receipt/manifest SHA-256=
`05a84f191c8af98e244e6ea1ba2276a87c514b172af9266598ca63a2e136016b / b0cd08d510adebcfed5255623ae9b659bb251960fda2b4a3af1b9945fa8dc946`。
34 个输入文件均通过 SHA-256/size 复核；26 个 source records、78 条 record-stratum rows 与 30 条 method-stratum rows
完整且唯一，所有五主指标 means finite。

W0 的分区绝对指标如下。前四项越低越好，PCC 越高越好：

| RR 区间 | windows / samp_id | Whole RR | Local RR | trajectory | global | PCC |
|---|---:|---:|---:|---:|---:|---:|
| low | 1656 / 8 | 0.656562 | 0.636570 | 0.136343 | 0.162823 | 0.884978 |
| medium | 462 / 6 | 0.301508 | 0.328935 | 0.149061 | 0.212232 | 0.860423 |
| high | 192 / 7 | 1.037746 | 1.051931 | 0.144285 | 0.171408 | 0.842986 |

相对 medium，W0 在 low 的 Whole/Local RR error 增加 `117.759% / 93.524%`，在 high 增加
`244.185% / 219.799%`，呈明显的区间依赖。Trajectory/global error 在 low 分别低 `8.532% / 23.280%`，在 high
分别低 `3.204% / 19.235%`；PCC 相对 medium 在 low 增加 `+0.024555`，在 high 降低 `−0.017437`。因此 RR 误差与
形态/包络指标没有共同的单调区间趋势。

W3 相对 W0 的描述性变化如下；error 的负值表示改善，PCC 为绝对变化：

| RR 区间 | Whole RR | Local RR | trajectory | global | PCC |
|---|---:|---:|---:|---:|---:|
| low | +16.325% | +9.974% | −1.831% | −4.215% | +0.001842 |
| medium | −2.650% | −1.941% | −1.368% | −3.528% | +0.002930 |
| high | −16.912% | −0.892% | −1.943% | −3.598% | +0.002530 |

W3 在三个区间的 trajectory/global/PCC 方向一致，但 RR 收益随区间改变：low RR 下 Whole/Local RR 变差，medium/high
持平或改善，其中 high RR 的 Whole RR 改善最大。C201 相对 W0 也呈混合权衡：medium 的 Local RR/global 分别改善
`4.783% / 11.432%`，high 的 Whole RR/global 分别改善 `8.451% / 8.457%`，但其余组合并不一致。完整 10 方法结果固定在
`runs/paper_evidence_v1/p6_rr_strata_summary/rr_stratum_method_summary.csv`。

这些结果支持“性能权衡随 target RR 区间变化”，不支持单一方法跨全部区间和属性统一占优。Low/medium/high 的样本量与
subject coverage 不平衡，且统计单位仍为重叠窗口的 sample-direct mean，因此只作描述性异质性证据。
