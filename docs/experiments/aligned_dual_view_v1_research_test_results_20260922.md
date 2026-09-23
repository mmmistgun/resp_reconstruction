# ADV-v1：固定候选的 research-test 结果

日期：2026-09-22。协议 ID：`aligned-dual-view-v1-research-test-20260922`。

**状态：用户已完成 test cache、四配置 × 三 seed 的 12 项评价和完整汇总；本次只读核对通过。** 汇总完成时间为 2026-09-22 11:06:42（Asia/Shanghai）。本文件记录实际完成状态，接续实现协议中的历史进度描述；冻结的实现协议、源码和运行产物保持原状。本轮已完成阶段不自行重跑。

主要判断：尺度均值对照相对完整尺度投影，在 Whole RR、Local RR、包络轨迹误差三个指标上均为三个 seed 改善，PCC 均值也更高；代价是全局包络调制误差均值更大。完整尺度投影的优越性仍未得到本轮结果支持。联合模型相对时域单视图的 PCC 收益持续存在，其余属性有取舍。CWT 单视图在当前结构与训练协议下明显较弱。

本次 test split 曾用于既有研究，结果属于 **reused research/development evidence**。它不构成未触及测试集上的独立确认，也不用于重选本轮 checkpoint。

## 1. 固定来源与核对范围

- 协议：[research-test 实现协议](aligned_dual_view_v1_research_test_protocol_20260922.md)。
- 候选锁：`configs/aligned_dual_view_research_test_v1/candidate_lock_20260922.json`；文件 SHA-256：`146e89f4c196fd1b2742ba66bb3ad600f11e25a6ccf98ebfcbc7ece012947418`。
- Lock ID：`3ded8ebbe0e91fed9bf83e053adf1805d013c3f6036f4ed6018813378944c6d2`。
- Cache：`runs/aligned_dual_view_v1_research_test/cache_20260922_r1`；manifest SHA-256：`83f288ee019a353bac97144df42680d0a722dd8706437d19f95a1c44851dc486`；cache ID：`6dfcd44a82891b996f27e2ddd83b3f1bbbb05998488dc82f8cfbd450ae673ec7`。
- 评价：`runs/aligned_dual_view_v1_research_test/cache_20260922_r1_evaluations/{view}_seed{seed}/attempt_r1`。
- 完整汇总：[summary_20260922_r1/manifest.json](../../runs/aligned_dual_view_v1_research_test/summary_20260922_r1/manifest.json)；SHA-256：`b21da372dfca62807831e97362c1a3cd81fe0b84657c1b5bca7972cd74b12bc3`。
- 共享 comparison ID：`dca2fdcaa6bf31f6aeaa776d0268dee7e01352fdfd66f2706b324dc3d719dd5c`。
- 训练代码身份：`19b92747cb2a26a5cf45d6954aeebaae1ce70b3de5a01a266677de4761284040`。
- Test 执行代码身份：`3a2b0c877c4e6832ff3d997ddafa96048cf9d074994e70e5a59ba948fce66d55`，与核对时工作树一致。

只读核对覆盖：

1. Cache、12 项评价、summary 的完成回执和 manifest 哈希；所有候选均只有 `attempt_r1`，未发现失败或未闭合尝试。
2. 全部候选的原训练 manifest、resolved config 身份及所选 checkpoint 文件哈希；view、seed、selected epoch 与冻结锁一致。
3. 各阶段记录的源码身份、依赖版本、候选锁及 started/completed run ID 一致。评价均为 CUDA、bf16、batch 32，使用共享 cache 和 comparison ID。
4. 每项 `metrics.csv`、`metrics_summary.csv`、`samples.json`、`access.jsonl` 的文件哈希及 sample 内容身份一致。每项 2,310 个严格递增且唯一的 test row ID 与冻结哈希一致，共核对 27,720 行指标记录。
5. 每项访问日志均按 input-only 推理开始、全部推理完成、target 读取开始、target 读取完成排列，计数均为 2,310；未重选 checkpoint。
6. 五主指标每项均有 2,310 个有限值，`joint_prediction_degenerate_fraction` 全部为零。逐 sample 指标按原函数汇总后与已保存单项 summary 一致；完整矩阵的 per-seed、across-seed、paired-deltas 与来源数值一致。

本次没有读取原始 input/target 波形、重放模型或重新生成 summary；只对存储的指标进行数值一致性核对。Cache 核对限于完成回执与身份元数据，没有重新读取约 1.5 GiB 的特征 payload；checkpoint 采用文件哈希核对，没有载入模型。日志能够核对记录的访问顺序，不等同于独立重放验证。

## 2. 五主指标

任务为完整 180 s 重建；沿用冻结 dataset/admission、target、Pi、eligibility、loss 和指标口径。每 seed 先按窗口 direct mean，再对三个训练 seed 等权汇总。下表为 mean ± sample SD；SD 描述训练随机性，不是人群置信区间。Test 为 2,310 窗口、8 个 samp_id，窗口之间不能视为独立人群样本。

| 配置 | Whole RR error ↓，bpm | Local RR MAE ↓，bpm | 包络轨迹误差 ↓ | 全局包络调制误差 ↓ | Signed PCC ↑ |
|---|---:|---:|---:|---:|---:|
| joint | 0.7807 ± 0.0390 | 0.7483 ± 0.0430 | 0.14335 ± 0.00182 | 0.17968 ± 0.01399 | 0.85883 ± 0.00315 |
| waveform | 0.7583 ± 0.0342 | 0.7614 ± 0.0338 | 0.14903 ± 0.00558 | 0.17851 ± 0.00455 | 0.85015 ± 0.00483 |
| cwt | 1.4660 ± 0.1210 | 1.7845 ± 0.0683 | 0.24208 ± 0.00191 | 0.30825 ± 0.01086 | 0.26336 ± 0.00814 |
| joint_scale_mean | 0.7251 ± 0.0016 | 0.6861 ± 0.0079 | 0.13749 ± 0.00171 | 0.18918 ± 0.00406 | 0.86213 ± 0.00258 |

完整数值：[per_seed.csv](../../runs/aligned_dual_view_v1_research_test/summary_20260922_r1/per_seed.csv)、[across_seed.csv](../../runs/aligned_dual_view_v1_research_test/summary_20260922_r1/across_seed.csv)、[paired_deltas.csv](../../runs/aligned_dual_view_v1_research_test/summary_20260922_r1/paired_deltas.csv)。

Checkpoint 均按 validation Local RR 最小值选取，并沿用最早并列规则：

| 配置 | seed20260811 | seed20260812 | seed20260813 |
|---|---:|---:|---:|
| joint | 12 | 7 | 5 |
| waveform | 8 | 9 | 15 |
| cwt | 22 | 7 | 21 |
| joint_scale_mean | 7 | 7 | 8 |

## 3. 相同 seed 的配对比较

误差百分比为 `(候选三 seed 均值 / 参照三 seed 均值 − 1) × 100%`，PCC 为绝对差。括号表示改善 seed 数；不据此作显著性判断。

| 指标 | joint 相对 waveform | joint_scale_mean 相对 joint | joint_scale_mean 相对 waveform |
|---|---:|---:|---:|
| Whole RR | +2.95%（0/3） | −7.11%（3/3） | −4.37%（3/3） |
| Local RR | −1.73%（2/3） | −8.30%（3/3） | −9.89%（3/3） |
| 包络轨迹误差 | −3.81%（2/3） | −4.08%（3/3） | −7.74%（3/3） |
| 全局包络调制误差 | +0.65%（2/3） | +5.29%（1/3） | +5.97%（0/3） |
| Signed PCC | +0.00868（3/3） | +0.00330（2/3） | +0.01197（3/3） |

与 [validation 结果](aligned_dual_view_v1_results_20260922.md)相比，以上三组对照的五指标均值变化方向全部保持一致。尺度均值相对 joint 的 Local RR 改善由 validation 的约 0.28% 增至本次约 8.30%；PCC 改善从 validation 的 3/3 seed 变为 2/3。该变化是不同 split 上的描述性结果，不能视为额外独立重复实验。

Joint 的 PCC 收益方向较稳定，但 Whole RR 在本次三个 seed 均差于 waveform。尺度均值对照的 RR 与包络轨迹改善更一致，全局包络调制代价也持续存在。当前比较支持保留这些属性取舍，不支持用单个“总分赢家”替代完整结果。

## 4. 辅助指标与诊断边界

以下为三个 seed 等权均值；对应 SD 与有限 seed 数见完整 CSV。

| 配置 | IBI MedAE ↓，s | IBI coverage ↑ | 呼吸带 coherence ↑ | constrained nDTW ↓ | 包络 Spearman 低 / 中 / 高 |
|---|---:|---:|---:|---:|---|
| joint | 0.11487 | 0.76599 | 0.53631 | 0.22690 | 0.40805 / 0.52307 / 0.71428 |
| waveform | 0.12755 | 0.73045 | 0.51197 | 0.23863 | 0.29433 / 0.47048 / 0.69655 |
| cwt | 0.16157 | 0.18003 | 0.31505 | 0.75672 | 0.15718 / 0.26905 / 0.56538 |
| joint_scale_mean | 0.11105 | 0.80629 | 0.56370 | 0.23012 | 0.35662 / 0.52222 / 0.73548 |

辅助指标同样存在取舍：尺度均值的 coherence 和 IBI coverage 均值更高，joint 的 nDTW 更低、低幅值层包络 Spearman 更高。不能把五主指标中四项均值改善外推为所有重建属性全面改善。

**IBI 误差存在可解释窗口筛选，必须同时阅读 coverage 与计数。** 各 seed 的 `ibi_medae_sec_n` 分别为：joint 1178/1342/1416，waveform 1224/1168/1102，cwt 38/55/44，joint_scale_mean 1480/1461/1431。CWT 对应可解释窗口比例仅 1.65%/2.38%/1.90%；其 IBI 误差均值只描述极小且模型相关的子集，不能作为全体窗口表现。IBI coverage 与可解释窗口比例是不同指标。

CWT 的最佳滞后中位数三个 seed 均为 0.30 s，滞后搜索边界命中率为 71.08%/71.30%/80.04%；同时 PCC 很低且 nDTW 较高。这提示当前重建与 target 的形态/对齐匹配较弱，不能单凭这些诊断认定存在一个可修复的固定时间偏移，也不据此修改本轮评价滞后范围。

## 5. 结论与后续决策范围

- 完整尺度投影相对尺度均值的预期优势，在 validation 与本次 reused research-test 中均未得到支持。这个结论只适用于当前表示、前端、主干、优化和预算，不能推断尺度信息普遍无用。
- 当前结构下 CWT 单视图不能替代波形输入。联合输入的部分属性收益与完整尺度投影是否必要，应分别讨论。
- 已有四配置 × 三 seed 提供了本轮结构对照。当前证据不足以支持立即扩大深度、宽度或投影维数网格；若继续研究，宜先明确全局包络调制代价或两路幅值平衡等机制问题，预定义新的 train/validation 比较。
- 后续若受本次 test 结果启发设计新结构，应明确标注为 test-informed 开发；不能继续把这个 split 当作未触及确认集。新增数据评价或实验阶段需要匹配的新协议与执行授权。
- 本次没有新增 W0/C201 评价，也未将不同任务区间结果混表；不作优于既有基线的结论。

本次改动仅为新增结果记录和协议入口更新；数据、loss、指标、选点、源码与原始产物身份均未改变。核对使用既有存储指标及 provenance，不涉及新的 GPU/真实数据实验。
