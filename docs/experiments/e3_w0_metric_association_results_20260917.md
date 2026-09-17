# E3 五指标关联与 RR—努力不一致比例：完成记录

状态：**三个冻结 W0 的 validation/test 再分析已完成，结果核对通过。** 运行日期为 2026-09-16，整理日期为 2026-09-17。本记录依据 completed manifest 登记正式再分析完成状态；执行定义以冻结专项协议为准。

## 1. 冻结身份与统计对象

- 协议：`e3-w0-metric-association-v1-20260916`；定义见 [E3 专项协议](e3_w0_metric_association_protocol_20260916.md)。
- 执行 commit：`d7beea5bcb9237a76bf22da8d79faaf378111efd`，运行时工作树干净。
- 实现锁 SHA-256：`f9fadfc620ab07b6d58bb7508ed441f923def01dc1a948a029190fbdc9e094bc`。
- manifest SHA-256：`506f1faab57c0d5bab6a62313e00c9930a7aadc49bedc75ffdb5279f5565e068`。
- 运行目录：`/mnt/disk_code/marques/resp_reconstruction/runs/e3_w0_metric_association_v1/analysis/analysis_f9fadfc620ab_20260916T145437Z_cc3454d75c60`。
- 模型为 W0 完整目标，seed 20260811/20260812/20260813，validation-selected epoch 13/15/14。输入为既有五主指标 CSV 和时间元数据。

| Split | 完整窗口 / samp_id | 不重叠窗口 / samp_id | 五指标合格分母 | 排除 / 预测退化 |
|---|---:|---:|---:|---:|
| validation | 2675 / 7 | 530 / 7 | 每 seed 2675 | 0 / 0 |
| test | 2310 / 8 | 452 / 8 | 每 seed 2310 | 0 / 0 |

同一窗口由三个固定模型分别评价。先在每个 seed 内计算相关或比例，再报告三个估计的均值 ± 样本 SD（ddof=1）；SD 描述模型训练随机性。窗口存在重叠，test 为既有受试者隔离且具有重复访问历史的研究测试集。

## 2. 描述性阈值与分母

RR 条件为 Whole RR absolute error 与 Local RR MAE 同时 ≤1 bpm。三项其他属性分别使用 validation 参考分布的第 75 百分位，高误差判定为严格大于阈值。参考分布由同一窗口三个 seed 的指标均值构成，对 2675 个唯一 validation 窗口使用线性插值分位数；同一阈值应用于两个 split、三个 seed 和两种窗口视图。

| 属性轴 | 第 75 百分位阈值 |
|---|---:|
| 包络轨迹误差 | 0.2248641915 |
| 全局包络调制误差 | 0.2568864307 |
| 形态误差 `1−PCC` | 0.2052912807 |

最后一项等价于 signed PCC < 0.7947087193。这些阈值描述相对参照分布中的高误差，RR 门槛是操作定义。

对每个 seed，N 为视图内全部合格窗口，R 为 RR 条件满足的窗口，H 为该属性高误差窗口，D 为两者同时成立的窗口。结果分别保留 D/R（条件比例）、D/N（总体不一致占比）和 H/N（总体高误差占比）。

## 3. 主设置条件比例与重叠敏感性

下表均为 **D/R 的百分数**，均值 ± seed 样本 SD；SD 单位为百分点。

| Split / 窗口视图 | 包络轨迹误差 | 全局包络调制误差 | 形态误差 `1−PCC` |
|---|---:|---:|---:|
| validation / 完整 | 14.201 ± 0.286 | 19.361 ± 0.522 | 12.833 ± 0.397 |
| validation / 不重叠 | 15.790 ± 1.238 | 20.755 ± 0.972 | 14.946 ± 0.859 |
| test / 完整 | 6.823 ± 0.007 | 17.709 ± 1.983 | 10.222 ± 0.562 |
| test / 不重叠 | 9.173 ± 1.085 | 19.932 ± 4.064 | 13.353 ± 0.680 |

完整 validation 的 R 分别为 2098/2099/2084；完整 test 的 R 分别为 1891/1922/1918。test 的轨迹 D 为 129/131/131，全局调制 D 为 324/309/382，形态 D 为 182/198/206。

test 完整窗口的三项 H/N 均值分别为 14.502%、22.568%、19.639%；D/N 分别为 5.642%、14.647%、8.456%。满足 RR 条件后的高误差比例低于对应总体高误差占比，但仍有窗口超过指定属性阈值。

不重叠选择按各 samp_id 全部 segment 的绝对时间排序，贪心保留半开区间互不重叠的窗口。两种视图均存在 RR 条件满足与其他属性高误差共存的窗口；具体比例随视图改变。

## 4. 五指标相关

各项为完整窗口上的 Spearman 相关，再对三个 seed 求均值。形态轴为 `1−PCC`，五个轴均越大误差越大。完整 SD、逐 seed 和逐 samp_id 结果见来源表。

| 两个属性轴 | Validation | Test |
|---|---:|---:|
| Whole RR / Local RR | 0.635 | 0.640 |
| Whole RR / 轨迹 | 0.646 | 0.591 |
| Whole RR / 全局调制 | 0.412 | 0.377 |
| Whole RR / 形态 | 0.617 | 0.562 |
| Local RR / 轨迹 | 0.796 | 0.775 |
| Local RR / 全局调制 | 0.506 | 0.472 |
| Local RR / 形态 | 0.841 | 0.801 |
| 轨迹 / 全局调制 | 0.677 | 0.697 |
| 轨迹 / 形态 | 0.848 | 0.782 |
| 全局调制 / 形态 | 0.533 | 0.495 |

RR 与包络、形态误差总体相关。逐 samp_id 的 Local RR—轨迹相关存在异质性：test 的 seed 均值范围约为 −0.010 至 0.817，validation 为 0.337 至 0.883。相关及条件比例共同支持在 RR 之外报告努力与形态属性。

## 5. 阈值与受试者敏感性

完整矩阵为 RR `{0.5,1,2}` bpm × validation 分位点 `{0.50,0.75,0.90}`。以下摘录 test 条件比例的三 seed 均值（%），对应完整窗口：

| RR 门槛 | 参照分位点 | 轨迹 | 全局调制 | 形态 |
|---|---:|---:|---:|---:|
| 0.5 bpm | 0.75 | 4.58 | 15.60 | 7.94 |
| 1.0 bpm | 0.50 | 42.22 | 41.25 | 32.12 |
| 1.0 bpm | 0.75 | 6.82 | 17.71 | 10.22 |
| 1.0 bpm | 0.90 | 1.41 | 4.90 | 3.37 |
| 2.0 bpm | 0.75 | 9.10 | 19.65 | 13.42 |

比例与阈值定义相关，引用时须同时提供 RR 条件、属性阈值和分母。结果为描述性分布证据，证据范围限定于当前固定模型与数据。

test 主设置下，samp_id 等权条件比例的三 seed 均值为 16.36%、29.32%、31.13%，高于窗口加权结果。其中 samp_id=670 的三个 seed 只有 2/3/1 个 RR 合格窗口，轨迹条件比例为 1/2、2/3、1/1；稀疏条件分母会放大等权宏平均的波动。完整窗口主设置的 8 个 test samp_id 均有定义；不重叠主设置在 seed 20260812 和 20260813 中，两者均为 8 个 samp_id 中有 7 个的条件比例有定义。

全部 2754 行比例结果中，72 行因 R=0 而将条件比例记为空，并保留 `no_rr_qualified_windows` 状态；这些行属于 test 的逐 samp_id 敏感性结果。相关系数 1140 行全部有定义。

## 6. 论文解释与完成验收

建议表述：呼吸率误差与包络及形态误差存在相关性，但较低的呼吸率误差仍可伴随较高的其他属性误差，因此需要联合评价节律、相对努力与波形形态。主设置比例与不重叠、阈值和个体敏感性应一并解释。该比例不对应临床事件发生率；seed SD 不是受试者级置信区间。

2026-09-17 收尾只读验证：19 个来源文件、8 个代码/协议文件、13 个完成产物及 manifest 身份均通过；复算 2754 行 D/R、D/N、H/N 与计数关系通过。前次结果核对已检查 4985 行窗口成员元数据及所选区间互不重叠。实现阶段 synthetic CPU 定向测试为 19 passed。本轮仅整理文档与结果索引。

完成产物：source audit 6 行、窗口成员 4985 行、阈值 9 行、相关 1140 行、比例 2754 行、宏平均 324 行、相关 seed 汇总 380 行、比例 seed 汇总 648 行。

论文证据已同步至内部底稿 5.6.1、正文 V-D / Table VI(c)、实验计划及待补清单、结果核验、科学证据账本及检查、全部实验汇总和贡献证据索引，共 10 份文档。候选经独立只读审核通过，安装前核对原文件与候选 SHA，安装后回读一致；既有论文编辑保留。实验执行定义及冻结产物沿用原身份。

- [相关及 seed 波动](/mnt/disk_code/marques/resp_reconstruction/runs/e3_w0_metric_association_v1/analysis/analysis_f9fadfc620ab_20260916T145437Z_cc3454d75c60/association_seed_summary.csv)
- [条件比例、总体比例与敏感性](/mnt/disk_code/marques/resp_reconstruction/runs/e3_w0_metric_association_v1/analysis/analysis_f9fadfc620ab_20260916T145437Z_cc3454d75c60/discordance_seed_summary.csv)
- [逐 seed / samp_id 计数](/mnt/disk_code/marques/resp_reconstruction/runs/e3_w0_metric_association_v1/analysis/analysis_f9fadfc620ab_20260916T145437Z_cc3454d75c60/discordance.csv)
- [运行回执与来源身份](/mnt/disk_code/marques/resp_reconstruction/runs/e3_w0_metric_association_v1/analysis/analysis_f9fadfc620ab_20260916T145437Z_cc3454d75c60/analysis_receipt.json)
