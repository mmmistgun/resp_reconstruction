# Patch-aligned TF-Mamba、APOR v2 H-only 与 W0 的 test 对比

日期：2026-10-06。本记录整理三类模型已有的完整 research-test 汇总，比较节律、包络与波形重建表现。

## 比较口径

- Test full：每个 cell 为 2310 个窗口、8 名受试者，包含受试者 670。
- Seeds：20260811、20260812、20260813。先对每个 seed 的窗口直接平均，再计算三个 seed 的均值和样本标准差。
- Checkpoint：各自实验按完整 validation Local RR MAE 选定的固定 checkpoint。
- W0 指原始冻结的 `crd_tf102_w` 基线；APOR v2 H-only 指 CWT-APOR v2 的 `H` 配置，CWT 实际频率范围为 `(0.8,8] Hz`。
- Patch-aligned TF-Mamba 包含完整的 1/2/4 秒时长矩阵。其训练使用微批量 64、累积 2 次；各实验的结构、初始化和执行合同存在差异，因此本表属于跨实验描述性比较，不能作为单一模块的因果消融。

## 五项主指标

下表为三 seed 均值，RR 单位为 bpm。粗体表示表内最佳均值。

| 模型 | Whole RR MAE ↓ | Local RR MAE ↓ | 包络轨迹 MAE ↓ | 全局包络调制误差 ↓ | Lag-aware signed PCC ↑ |
|---|---:|---:|---:|---:|---:|
| W0 | **0.617234** | 0.609566 | **0.139546** | 0.173418 | 0.876577 |
| APOR v2 H-only | 0.620714 | **0.607657** | 0.143393 | **0.164118** | **0.881586** |
| Patch TF-Mamba：1 秒 | 0.622583 | 0.624839 | 0.145423 | 0.176193 | 0.869947 |
| Patch TF-Mamba：2 秒 | 0.640956 | 0.636419 | 0.141952 | 0.167359 | 0.877433 |
| Patch TF-Mamba：4 秒 | 0.707975 | 0.753873 | 0.150672 | 0.173373 | 0.861830 |

对应的跨 seed 样本标准差如下；这些标准差描述训练随机性的波动，不是受试者层面的置信区间。

| 模型 | Whole RR SD | Local RR SD | 包络轨迹 SD | 全局调制 SD | PCC SD |
|---|---:|---:|---:|---:|---:|
| W0 | 0.027788 | 0.018472 | 0.001037 | 0.006625 | 0.001329 |
| APOR v2 H-only | 0.046771 | 0.037996 | 0.006331 | 0.009295 | 0.003776 |
| Patch TF-Mamba：1 秒 | 0.068362 | 0.071472 | 0.005739 | 0.008681 | 0.002672 |
| Patch TF-Mamba：2 秒 | 0.027124 | 0.038752 | 0.002575 | 0.004646 | 0.001646 |
| Patch TF-Mamba：4 秒 | 0.143151 | 0.174376 | 0.011405 | 0.009423 | 0.012890 |

## 2 秒版本相对两条基线的变化

误差类变化按 `100×(新模型均值/基线均值−1)` 计算，正值表示误差增加；PCC 使用新模型减去基线的绝对差。计算使用源 CSV 的完整精度，非逐 seed 相对变化的平均。

| 参照基线 | Whole RR 变化 | Local RR 变化 | 包络轨迹变化 | 全局调制变化 | PCC 差值 |
|---|---:|---:|---:|---:|---:|
| APOR v2 H-only | +3.26% | +4.73% | −1.00% | +1.97% | −0.004153 |
| W0 | +3.84% | +4.41% | +1.72% | −3.49% | +0.000856 |

## 结果解读

1. **两条基线仍保有明确的指标优势。** W0 的 Whole RR 与包络轨迹误差均值最低；APOR v2 H-only 的 Local RR、全局调制误差与 PCC 均值最好。
2. **1 秒版本偏向节律，但未形成对基线的均值优势。** 它在新模型内部具有最低的两项 RR 误差；与 H-only、W0 分别比较时，五项均值均较差。
3. **2 秒版本体现形态质量取舍。** 相对 H-only，仅包络轨迹误差均值改善；相对 W0，全局调制误差改善、PCC 小幅提高，同时 RR 与包络轨迹误差增加。在新模型三种时长之间，2 秒的五项跨 seed 标准差均最小。
4. **4 秒版本的当前结果较弱。** 五项均值都逊于 H-only，也都逊于新模型 2 秒版本；其 RR 跨 seed 波动最大。

当前结果支持“不同结构在节律与形态恢复之间存在取舍”，尚未支持 Patch-aligned TF-Mamba 整体优于 APOR v2 H-only 或 W0。均值排序不等于统计显著性，本比较未进行显著性检验。该 test 集已被用于研究开发，证据角色为 reused research/development evidence；本整理不改变候选矩阵、checkpoint 选择或后续评价定义。

## 来源

本次只读取既有汇总 CSV 与结果文档，未运行训练或模型推理。源文件及 SHA-256 为：

1. **W0**：[既有 test 协议](crd_tf_v1_research_test_protocol_20260816.md)；数值取自 [performance_cost.csv](e4_closeout_data_20260922/performance_cost.csv) 的 `benchmark_group=A, arm=W0_FULL, split=test` 行。文件 SHA-256：`c34d8e77b43fe7f2713752493e98a866db0ee6392494c3e572af92751e086571`。
2. **APOR v2 H-only**：[实验总览](cwt_apor_v2_results_summary_20261002.md)；数值取自 `runs/cwt_apor_v2/session_20261001T042244Z_e5e53ad16731/research_test/summary/attempt_20261002T054624Z_e22f113cfb30/full/across_seed.csv` 的 `arm=H` 五行，产物位于 `cwt-time-frequency-v1` worktree。文件 SHA-256：`ec5a71f588261f5beeff30bc3b97d5766d7516a91e128bb8da3cd3abca9ef175`。
3. **Patch TF-Mamba**：数值取自 `runs/patch_aligned_tf_mamba/research_test_v1_20261006/summary/attempt_1c0c1aa770314c8fb4beda50246792dd/across_seed.csv` 的 `p1s/p2s/p4s` 行。文件 SHA-256：`add5c599acffd90af8eee818df276b1c1d4dae83c2fac78d4f5ff9d8834baa8d`。

原始运行产物继续保存在各自 session 中，本文件作为跨实验对照记录。
