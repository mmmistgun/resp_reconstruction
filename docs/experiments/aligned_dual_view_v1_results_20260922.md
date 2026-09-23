# ADV-v1：四种输入配置的 validation 结果

日期：2026-09-22。执行契约：`aligned-dual-view-v1-train-val-20260920`。

**状态：用户执行的四配置 × 三 seed 共 12 次 formal 训练已完成；本次完成已有产物的只读核对和首次完整矩阵汇总。** 本文件更新实现协议中关于实际执行进度的历史描述。训练源码、原协议快照、checkpoint、缓存和各 run 的原始文件保持原状；本轮训练与汇总不自行重复执行。

主要判断：联合模型对时域单视图的收益集中于 PCC，其他属性存在取舍。尺度均值对照在多项指标上优于完整尺度投影，当前结果不支持 `97→32` 固定尺度投影优于尺度均值的主张。CWT 单视图在本轮结构和训练协议下明显较弱。

## 1. 来源和核对范围

- 训练目录：`runs/aligned_dual_view_v1/{view}_seed{seed}_formal_20260921_r1`。
- `view`：`joint`、`waveform`、`cwt`、`joint_scale_mean`；`seed`：20260811、20260812、20260813。
- 首次完整汇总：[summary_full_20260922_r1](../../runs/aligned_dual_view_v1/summary_full_20260922_r1/manifest.json)。Manifest SHA-256：`cdc03f23f299b24bbf6234cd73c5c1678d7e9d7b22065e7f8cef58f9a0d1577d`。
- 共享 comparison ID：`b68bc418a55375b0159964b39873631d21b3a72a37eba7395268862f380b0609`。
- 执行代码身份：`19b92747cb2a26a5cf45d6954aeebaae1ce70b3de5a01a266677de4761284040`，与核对时工作树一致。
- Cache ID：`04a5db6b544815780bc7af21a50af14bffd61dfdd0da943f08a2d9430da34869`；本次未重新读取或计算缓存特征。

核对已覆盖：

1. 十二个完整 formal lifecycle；每个 run 的 manifest 与完成回执匹配。
2. 80 个连续 epoch、最终 6,400 次 optimizer 更新；selected epoch 与 history 中最早的 Local RR 最小值一致。
3. selected/final checkpoint 文件哈希，以及 config、history、samples、逐 sample metrics、summary 的哈希。
4. 同一输入/target/mask 身份、同一 validation row ID 与顺序；每个 run 均有 2,675 个不重复的 validation 窗口。
5. resolved config 与 manifest identity 匹配；共享 comparison ID 核对数据、预算、执行代码和依赖口径。
6. eligible 五主指标有限、summary 与逐 sample direct mean 一致；所有 run 的 `joint_prediction_degenerate` 标记计数为零。

本次只读取既有运行产物并计算汇总，未执行模型前向、训练或 test 评价。没有重新读取真实输入/target 波形，也未重算指标定义。Checkpoint 核对采用文件哈希，未另行加载模型重放。

## 2. 三 seed 的完整五主指标

先在每个 seed 内按冻结规则计算 sample direct mean，再对三个 seed 等权求均值。下表为 mean ± sample SD；SD 描述训练随机性，不是人群置信区间。

| 配置 | Whole RR error ↓，bpm | Local RR MAE ↓，bpm | 包络轨迹误差 ↓ | 全局包络调制误差 ↓ | Signed PCC ↑ |
|---|---:|---:|---:|---:|---:|
| 双视图 joint | 0.4844 ± 0.0125 | 0.5489 ± 0.0171 | 0.15052 ± 0.00196 | 0.21553 ± 0.03098 | 0.84231 ± 0.00058 |
| 时域单视图 waveform | 0.4705 ± 0.0065 | 0.5550 ± 0.0117 | 0.15597 ± 0.00638 | 0.21367 ± 0.00819 | 0.83274 ± 0.00549 |
| CWT 单视图 cwt | 1.0053 ± 0.0689 | 1.3767 ± 0.1098 | 0.22310 ± 0.00210 | 0.28972 ± 0.01194 | 0.26235 ± 0.00597 |
| 双视图＋尺度均值 joint_scale_mean | 0.4629 ± 0.0090 | 0.5473 ± 0.0059 | 0.14768 ± 0.00089 | 0.22562 ± 0.01867 | 0.84598 ± 0.00371 |

逐 seed 数值：[per_seed.csv](../../runs/aligned_dual_view_v1/summary_full_20260922_r1/per_seed.csv)。完整均值/SD：[across_seed.csv](../../runs/aligned_dual_view_v1/summary_full_20260922_r1/across_seed.csv)。配对差：[paired_deltas.csv](../../runs/aligned_dual_view_v1/summary_full_20260922_r1/paired_deltas.csv)。

## 3. 对照的解释

以下误差百分比均定义为 `(候选三 seed 均值 / 参照三 seed 均值 − 1) × 100%`；PCC 使用绝对差。改善 seed 数按相同 seed 配对计算，因此总体均值与多数 seed 的方向可能不同。

### 联合模型相对时域单视图

| 指标 | 三 seed 均值变化 | 改善 seed 数 |
|---|---:|---:|
| Whole RR | +2.96% | 1/3 |
| Local RR | −1.10% | 2/3 |
| 包络轨迹误差 | −3.50% | 2/3 |
| 全局包络调制误差 | +0.87% | 2/3 |
| Signed PCC | +0.00957 | 3/3 |

PCC 的改善在三个 seed 中方向一致；Local RR 与包络轨迹的均值改善较小且非全部 seed 同向。Whole RR 与全局包络调制的均值更差，因此不能将联合模型描述为五指标全面优于时域单视图。

### 尺度均值对照相对完整尺度投影

| 指标 | 三 seed 均值变化 | 改善 seed 数 |
|---|---:|---:|
| Whole RR | −4.43% | 3/3 |
| Local RR | −0.28% | 2/3 |
| 包络轨迹误差 | −1.89% | 3/3 |
| 全局包络调制误差 | +4.68% | 1/3 |
| Signed PCC | +0.00367 | 3/3 |

尺度均值在 Whole RR、包络轨迹和 PCC 上三个 seed 均改善，Local RR 均值接近，全局包络调制误差存在代价。这削弱了本轮固定尺度投影优越性的设计预期；不能从该结果继续推断尺度信息普遍无用，或其他尺度编码方法必然无效。

尺度均值对照相对时域单视图，Local RR、包络轨迹和 PCC 均在三个 seed 中改善；均值分别为 −1.38%、−5.32% 和 +0.01324。其全局包络调制误差均值仍恶化 5.60%。因此加入 CWT 的部分属性收益与完整尺度投影是否值得保留，是两个不同问题。

CWT 单视图在五项均值上均弱于联合模型和时域单视图。这支持在当前网络/优化条件下保留波形输入；本轮并未检验所有可能的 CWT 单视图结构或训练方案。

## 4. 选点和训练后期表现

| 配置 | seed20260811 | seed20260812 | seed20260813 |
|---|---:|---:|---:|
| joint | 12 | 7 | 5 |
| waveform | 8 | 9 | 15 |
| cwt | 22 | 7 | 21 |
| joint_scale_mean | 7 | 7 | 8 |

全部 run 都执行完 80 epochs，结果使用上述 validation-selected checkpoint。十二个 run 的最后 epoch Local RR 均高于所选最小值。这个观察需要结合训练曲线解释，不能单凭选点偏早认定前端容量不足，也不能据此追溯性修改本轮预算或选点规则。

## 5. 证据边界与后续决策

- 这些结论来自完整 180 s 任务、固定 train/validation split 和本轮三 seed；不扩展为独立 test 结论或统计显著性结论。
- 四配置共享 D64、六层主干和训练预算；双路各 32 通道、单路 64 通道及前端自然参数差仍是解释边界。
- 本次只比较新结构内部四种输入配置，没有据此作出优于 W0/C201 的结论，也没有与论文中心区间指标混表。
- 现有结果还不足以支持直接扩大深度/宽度网格来维护完整尺度投影方案。后续若开展诊断，应先明确要解释的是全局包络调制的代价、两路幅值平衡、边界误差还是优化后期变化，再单独预定义比较；本次未开启后续实验。

本文件是新增结果记录；原始 run、已选 checkpoint 和本次完整汇总保留，后续解释更新可引用这些固定来源。
