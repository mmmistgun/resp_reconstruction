# Patch-aligned TF-Mamba 时长敏感性 research-test 结果

2026-10-06：完整 9-checkpoint research-test 评价及汇总已完成并关闭。当前状态以本记录为准。所有 checkpoint 均按此前完整 validation Local RR MAE 的既定 selector 固定，时长为 1/2/4 秒，seeds 为 20260811/20260812/20260813。

## 主指标

各 cell 覆盖相同的 2310 个 test 窗口、8 名受试者，共 20790 行逐窗口评价。主指标均先按窗口直接平均，再计算三个 seed 的均值 ± 样本标准差。RR 单位为 bpm。

| Patch | Whole RR MAE ↓ | Local RR MAE ↓ | 包络轨迹 MAE ↓ | 全局包络调制误差 ↓ | Lag-aware signed PCC ↑ |
|---|---:|---:|---:|---:|---:|
| 1 秒 | 0.622583 ± 0.068362 | 0.624839 ± 0.071472 | 0.145423 ± 0.005739 | 0.176193 ± 0.008681 | 0.869947 ± 0.002672 |
| 2 秒 | 0.640956 ± 0.027124 | 0.636419 ± 0.038752 | 0.141952 ± 0.002575 | 0.167359 ± 0.004646 | 0.877433 ± 0.001646 |
| 4 秒 | 0.707975 ± 0.143151 | 0.753873 ± 0.174376 | 0.150672 ± 0.011405 | 0.173373 ± 0.009423 | 0.861830 ± 0.012890 |

1 秒的两项 RR 误差均值最低；相对 2 秒，Whole RR 低 0.018373 bpm、Local RR 低 0.011580 bpm，均为 2/3 seeds 改善。2 秒在两项包络误差及 PCC 上优于 1 秒，三项均为 3/3 seeds 同向；其跨 seed 标准差也更小。2 秒相对 4 秒的五项主指标窗口均值均更优。

4 秒的 RR 跨 seed 波动较大：seed 20260813 的 Whole RR 为 0.860217、Local RR 为 0.950820 bpm。该 cell 与其余八组的 prediction degeneracy 标记均为零。

## 跨 split 与聚合口径

Validation 中 1 秒的 Local RR 与 PCC 均值领先；test 中 1 秒仍具有较低的 Local RR 均值，而 PCC 由 2 秒领先。Validation 中 4 秒的 Whole RR 均值最低，该排序在 test 上未延续。整体呈现 RR 估计与波形/包络恢复之间的取舍。

按受试者等权的 subject-macro 三 seed 均值如下：

| Patch | Whole RR ↓ | Local RR ↓ | 包络轨迹 ↓ | 全局调制 ↓ | PCC ↑ |
|---|---:|---:|---:|---:|---:|
| 1 秒 | 1.252676 | 1.106794 | 0.166975 | 0.217379 | 0.833672 |
| 2 秒 | 1.391521 | 1.162815 | 0.162064 | 0.207358 | 0.838300 |
| 4 秒 | 1.420973 | 1.283783 | 0.170684 | 0.201517 | 0.823719 |

subject-macro 下，1 秒仍在 RR 上领先，2 秒仍在包络轨迹和 PCC 上领先；全局调制误差则以 4 秒最低，说明该项排序受窗口权重与受试者等权口径影响。主表保持预设的逐窗口平均口径。

## 辅助指标与质量

以下为原生辅助指标逐 seed 汇总值的三 seed 均值：

| Patch | IBI MedAE（秒）↓ | IBI coverage ↑ | 呼吸频带 coherence ↑ | constrained nDTW ↓ |
|---|---:|---:|---:|---:|
| 1 秒 | 0.095206 | 0.787676 | 0.560693 | 0.217361 |
| 2 秒 | 0.090733 | 0.803816 | 0.582822 | 0.207563 |
| 4 秒 | 0.095372 | 0.792764 | 0.577892 | 0.229426 |

IBI MedAE 受可解释窗口资格约束，各 cell 的可解释窗口数为 1308–1522，需结合 coverage 理解。完整辅助资格、分层 envelope Spearman、Local RR tail 和受试者结果保存在汇总 CSV。

全部 cell 的五项主指标分母均为 2310，joint/envelope Spearman prediction degeneracy 数均为零，quality acceptance 均通过。汇总核对了每组产物哈希、固定 checkpoint 来源、跨 cell 样本顺序与 target 资格。未重跑模型推理。

## 证据边界与来源

该 test 集曾用于既有研究开发，证据角色为 **reused research/development evidence**。本结果是固定矩阵的跨 split 复核；尚未进行显著性检验，也不据此声称未经开发使用的独立确认集改善。

- Validation 结果：`docs/experiments/patch_aligned_tf_mamba_validation_results_20261006.md`。
- Research-test 协议：`docs/experiments/patch_aligned_tf_mamba_research_test_v1_20261006.md`，运行时文件保持冻结。
- Allowlist 根目录：`runs/patch_aligned_tf_mamba/research_test_v1_20261006/`。
- Allowlist SHA-256：`0d0e72b36ed8b760aeee938385120d8f6cc1ff14abdd865e6570025a861245e7`。
- 汇总目录：上述根目录内的 `summary/attempt_1c0c1aa770314c8fb4beda50246792dd/`。
- 汇总 receipt SHA-256：`2fb0d2e529c6133d978cec72c96fdab753eb58919aa953fb91744be529524070`。
- 汇总产物：`per_seed.csv`、`across_seed.csv`、`paired_delta.csv`、`native_summary.csv`、`per_subject.csv`、`subject_macro.csv`、`local_rr_tail.csv`、`denominators.csv`、`sources.json`、`receipt.json`。

运行产物保持原位并不进入 Git。已完成的 train/validation、research-test 和汇总不自行重跑；新增研究问题使用独立协议或明确修订。
