# M4 池化评分/值位置路径 test 结果与 validation 对照

截至 2026-10-17，ABCD × 三 seed 的十二个固定最佳 checkpoint research-test 及完整自动汇总均已完成并关闭，后台任务退出码 0。执行定义见 [test 协议](m4_pool_position_paths_research_test_v1_20261017.md)，前序结果见 [validation 结果](m4_pool_position_paths_v1_validation_results_20261016.md)。

**test 上 A 的两项 RR 平均误差最低，C 的包络轨迹误差最低，D 的全局包络调制误差最低且 PCC 最高。** 相比 validation，RR 的配置排序发生变化；最一致的跨 split 主效应是值位置降低全局包络调制误差。评分主效应仍缺少稳定收益，交互方向依赖指标与 split。

## 完成与核验

- 十二项均为固定 2310 窗口/8 人，seed=20260811/20260812/20260813；五项主指标每 cell 分母均为 2310。
- 每项 quality acceptance 通过，joint 和 envelope Spearman prediction-degenerate 均为 0；失败 attempt 为 0。
- 核验 allowlist、源码快照、十二项 checkpoint/config/completion 身份、96 项评价产物及 22 项 test 汇总产物；validation 汇总的 42 项产物也通过哈希核验。
- 从各 arm 的已保存逐 seed 均值独立重算评分主效应、值主效应和交互，与两种聚合下的冻结汇总一致。本次仅读取保存结果，没有重新训练、选点或评价。
- 结果为每 seed 的窗口均值再跨 seed mean ± sample SD；另列先对受试者等权的宏平均。三个 seed 共用同一人群，SD 表示 seed 间波动。

## 四条件 test 表现

A=(评分位置1，值位置1)，B=(0,0)，C=(1,0)，D=(0,1)。

| 配置 | Local RR ↓，bpm | Whole RR ↓，bpm | 包络轨迹 MAE ↓ | 全局包络调制误差 ↓ | Lag-aware PCC ↑ |
|---|---:|---:|---:|---:|---:|
| A：评分＋值 | 0.642953 ± 0.059778 | 0.598100 ± 0.055319 | 0.145594 ± 0.009340 | 0.168686 ± 0.007070 | 0.871995 ± 0.006282 |
| B：内容池化 | 0.664745 ± 0.041971 | 0.684944 ± 0.028963 | 0.141315 ± 0.009464 | 0.177773 ± 0.008360 | 0.871086 ± 0.007395 |
| C：仅评分位置 | 0.700412 ± 0.036800 | 0.724955 ± 0.046711 | 0.136289 ± 0.001632 | 0.174854 ± 0.004369 | 0.874963 ± 0.001550 |
| D：仅值位置 | 0.710043 ± 0.083355 | 0.685498 ± 0.093595 | 0.139012 ± 0.006378 | 0.164515 ± 0.003173 | 0.875660 ± 0.001819 |

validation 的最低 RR 均值来自 B，而 test 来自 A。A−B 的 test Whole RR 为 −0.086844 bpm，三个 seed 均改善；Local RR 为 −0.021792 bpm，但只有 1/3 seed 改善，均值优势主要由 seed 20260811 贡献。两项 RR 的一致性需要分别解释。

C 的包络轨迹窗口均值在两个 split 都最低，D 的 PCC 在两个 split 都最高。全局包络调制最优均值从 validation 的 A 变为 test 的 D。D−A 的 test PCC 为 +0.003665、全局包络调制误差为 −0.004171，两者均为 3/3 seed 改善。

## 条件路径效应

delta 为开启路径后减去关闭路径，误差负值/PCC 正值为改善。test 三 seed 的平均条件效应如下：

| 条件效应 | Local RR | Whole RR | 包络轨迹 | 全局包络调制 | PCC |
|---|---:|---:|---:|---:|---:|
| 评分，值关闭 C−B | +0.035667 | +0.040011 | −0.005027 | −0.002919 | +0.003878 |
| 评分，值开启 A−D | −0.067090 | −0.087398 | +0.006583 | +0.004171 | −0.003665 |
| 值，评分关闭 D−B | +0.045298 | +0.000554 | −0.002303 | −0.013258 | +0.004575 |
| 值，评分开启 A−C | −0.057459 | −0.126855 | +0.009306 | −0.006168 | −0.002968 |

评分路径的 test 作用随值路径开关明显不同：C−B 的 Whole RR 在 3/3 seed 变差，而 A−D 的均值改善但只有 2/3 seed 同向；A−D 的 PCC 在两个 split 都为 3/3 seed 变差。test 上 C−B 改善全局包络调制的三个 seed，与 A−D 的三个 seed 均变差形成相反的条件方向。

值路径在评分关闭时，D−B 的 test 全局包络调制误差在 3/3 seed 降低，但 Local RR 在 3/3 seed 升高。评分开启时 A−C 的 test Whole RR 在 3/3 seed 降低；全局包络调制则从 validation 的 3/3 seed 改善降为 test 的 2/3。因此，值位置的贡献需要按指标和评分状态表述。

## 平均主效应：validation 对照 test

评分主效应 `S=[(C−B)+(A−D)]/2`；值主效应 `V=[(D−B)+(A−C)]/2`。下表括号为改善 seed 数，所有数值保留原指标方向。

| 指标 | S validation | S test | V validation | V test |
|---|---:|---:|---:|---:|
| Local RR | −0.000437（2/3） | −0.015712（1/3） | +0.000513（1/3） | −0.006081（2/3） |
| Whole RR | +0.013308（0/3） | −0.023694（1/3） | +0.020127（1/3） | −0.063151（3/3） |
| 包络轨迹 | +0.000117（1/3） | +0.000778（1/3） | +0.003196（1/3） | +0.003501（1/3） |
| 全局包络调制 | +0.003028（2/3） | +0.000626（1/3） | −0.013621（2/3） | −0.009713（2/3） |
| PCC | −0.002891（0/3） | +0.000106（1/3） | +0.002400（2/3） | +0.000803（1/3） |

**评分主效应没有形成稳定的跨 split 收益。** validation 中 Whole RR/PCC 三个 seed 同向变差的现象，在 test 中转为混合方向；test RR 的负均值主要来自 seed 20260811 较大的改善，另外两个 seed 的评分主效应均使两项 RR 变差。不能依据平均值符号转换就断言评分位置稳定提升 RR。

**值主效应对全局包络调制的改善最能跨 split 保持。** 窗口加权下两个 split 都是 2/3 seed 改善；受试者宏平均下均为 3/3 seed 改善（validation −0.023243，test −0.016187）。Whole RR 的值主效应则发生方向变化：test 为 −0.063151 bpm、3/3 seed 改善，validation 为 +0.020127 bpm、1/3 seed 改善。

## 交互效应

`I=A−C−D+B=(A−D)−(C−B)`，用于衡量联合结果偏离单独路径可加参照 `C+D−B` 的程度。误差上的正交互/PCC 上的负交互意味着偏离方向不利于该指标；交互自身不等同于某一实际 arm 间的改进。

| 指标 | I validation mean ± SD | 符号 11/12/13 | I test mean ± SD | 符号 11/12/13 |
|---|---:|---|---:|---|
| Local RR | −0.003311 ± 0.004628 | −/+/− | −0.102757 ± 0.115233 | −/+/− |
| Whole RR | +0.006723 ± 0.039637 | +/+/− | −0.127409 ± 0.130277 | −/−/− |
| 包络轨迹 | +0.013985 ± 0.009988 | +/+/+ | +0.011609 ± 0.010775 | +/−/+ |
| 全局包络调制 | −0.013660 ± 0.016896 | −/+/− | +0.007090 ± 0.004081 | +/+/+ |
| PCC | +0.002512 ± 0.008711 | −/−/+ | −0.007543 ± 0.006233 | −/−/− |

- Whole RR：test 三个 seed 均为负交互，受试者宏平均也同向；该一致性未出现在 validation。
- 全局包络调制与 PCC：test 分别为三个 seed 正交互、三个 seed 负交互，两种聚合均保持。两条路径组合后的结果在这两项上低于单独路径可加参照的预期。
- 包络轨迹：正交互的均值从 validation 保持到 test，但一致性由 3/3 降为 2/3；test seed 20260812 的交互为 −0.000673。
- Local RR：两个 split 都为 −/+/−，test 幅度增大，但没有三个 seed 一致的协同证据。

test 中，联合路径对 Whole RR 的有利交互与对全局包络调制/PCC 的不利交互并存，支持按任务指标讨论结构取舍。

## 受试者宏平均

test 先对每人窗口求均值，再对八人等权，最后跨 seed 平均：

| 配置 | Local RR ↓ | Whole RR ↓ | 包络轨迹 ↓ | 全局包络调制 ↓ | PCC ↑ |
|---|---:|---:|---:|---:|---:|
| A | 1.238068 | 1.231432 | 0.166485 | 0.199007 | 0.830749 |
| B | 1.258455 | 1.520930 | 0.162068 | 0.215925 | 0.832492 |
| C | 1.306067 | 1.589228 | 0.154847 | 0.210302 | 0.836200 |
| D | 1.294648 | 1.415933 | 0.157985 | 0.194847 | 0.836481 |

test 的各指标最低误差/最高 PCC 配置与窗口加权相同，但 RR 绝对数值明显更高，显示聚合权重对人群表现评价有实质影响。

test 值主效应的 Whole RR 为 −0.231397 bpm、3/3 seed 改善，而 validation 宏平均为 +0.049081 bpm、3/3 seed 变差。test 的 Whole RR、全局包络调制、PCC 交互分别为 −0.252799、+0.009783、−0.009440，均保持各自三个 seed 同向。Local RR 交互为 −0.104191，仍只有 2/3 seed 为负。

总体方向不能代表每个受试者均受益；逐人效应保留在原汇总 `subject_path_effects.csv`，逐人三 seed 均值另见本轮对照产物。

## 综合判断与证据边界

此前“B 适合 RR 优先”的判断局限于 validation。固定 test 结果中，A 的 Whole RR 优于 B 且三个 seed 一致，A 的 Local RR 均值优势则缺少同样的一致性。D 在两个 split 的 PCC 均值均最高，test 还具有最好的全局包络调制均值；其 Local RR 代价也需保留。C 在两个 split 都保留最低的窗口加权包络轨迹均值。

可以保留的跨 split 结构结论是：值位置与全局包络调制保真有关；评分位置的平均收益依赖条件、指标和 split；两条路径存在明显的非加性，其方向具有指标特异性。实际意义阈值未预设，不作等效或统计显著性判断。

validation 为 2675 窗口/7 人且参与 checkpoint 选择，test 为另一组 2310 窗口/8 人；跨 split 绝对差异同时包含人群构成与选点影响。这份 test 已参与研究开发，证据角色为 **reused research/development evidence**，本轮用于固定矩阵复核。结论来自各结构分别训练并按相同 validation 规则选点后的系统表现。

## 产物与冻结身份

- test 根目录：`runs/m4_pool_position_paths_v1/research_test_v1_20261017/`。
- allowlist SHA-256：`75ed8644e008912bbf2ee6d2e11b0a9f6cc68c851f5d73bad45ffa04b0a759f9`。
- test summary：`summary/attempt_b271e58ec1e949daba7bc65255b5b194/`；receipt SHA-256：`47417b0fa34b92539840a0af35c6a4a66964095f90b8e2e1ce34c5924550a3ed`。
- [自动完整 test 报告](../../runs/m4_pool_position_paths_v1/research_test_v1_20261017/summary/attempt_b271e58ec1e949daba7bc65255b5b194/report.md)、[窗口加权交互图](../../runs/m4_pool_position_paths_v1/research_test_v1_20261017/summary/attempt_b271e58ec1e949daba7bc65255b5b194/interactions.png)、[受试者宏平均交互图](../../runs/m4_pool_position_paths_v1/research_test_v1_20261017/summary/attempt_b271e58ec1e949daba7bc65255b5b194/subject_macro_interactions.png)。图中为三个 seed 的 mean ± sample SD。
- 跨 split 分析：`runs/m4_pool_position_paths_v1/test_analysis_v1_20261017/`；含独立核验脚本、`effects_validation_vs_test.csv`、`performance_validation_vs_test.csv`、`test_subject_seed_means.csv` 和核验 receipt。
- 原生辅助指标、尾部、分母、参数量和全部条件效应见 test summary 的 CSV。训练协议、评价入口、allowlist、历史结果和源码快照继续保持冻结身份。
