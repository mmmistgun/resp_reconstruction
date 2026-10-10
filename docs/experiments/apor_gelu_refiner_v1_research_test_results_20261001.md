# APOR GELU/H64/Direct：固定checkpoint research-test完成记录

日期：2026-10-01（Asia/Shanghai）。协议ID：`apor-gelu-refiner-v1-research-test-20261001`。

状态：**9/9新增固定checkpoint评价与完整12-cell三视图汇总均已完成并冻结，失败0，两worker退出码均为0。** B0复用原A0三份test。本轮train/validation与research-test均关闭，validation的`eligible_arms=[B0]`决定保留。

## 1. 主要结果

**H64在test窗口平均RR上获得本轮最明确的改善，尤其在排除670后，两项RR均为3/3 seed改善；同时PCC下降，完整人群的受试者等权Local RR也退化。** 该结果支持把H64保留为节律方向的结构候选，但不能将其表述为跨split、跨聚合口径的全面替代。

统一GELU保持了调制误差下降与RR均值上升的取舍。Direct删除条件残差后，窗口RR均值有所改善，包络和PCC却明显变差；这些损失在排除670后仍存在，支持保留条件非线性重组的形态与调制作用。

这是已经用于研究开发的test split，证据角色为`reused research/development evidence`。三seed用于描述优化随机性，不将结果当作新的独立人群确认，也不据test重新选择checkpoint或改写validation保护线。

## 2. 矩阵、样本与统计

- Seeds：20260811/20260812/20260813。GELU best epoch=10/13/11，H64=8/13/14，DIRECT=14/13/61；B0固定epoch=8/13/13。
- Full：2310窗口、8人；exclude670：2231窗口、7人；subject670：79窗口、1人。三个视图来自同一完整推理及逐窗口指标。
- 新增九cell共20790行metrics；包含B0复用参照的完整12-cell矩阵共27720行。
- 以下主表先计算各seed窗口均值，再取三seed均值；四项error越低越好，PCC越高越好，RR单位bpm。sample SD、逐seed、逐受试者、subject-macro、配对差及RR尾部均保存在summary。
- 全部cell的五项主指标有限，样本顺序、target资格与窗口分母一致，joint/envelope Spearman prediction degeneracy均为0。

## 3. 完整test

| Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| B0 | 0.663993 | 0.651348 | 0.142169 | 0.161903 | 0.881917 |
| GELU | 0.684256 | 0.671514 | 0.140308 | 0.158787 | 0.882235 |
| H64 | 0.633429 | 0.623685 | 0.142160 | 0.160932 | 0.880363 |
| DIRECT | 0.655539 | 0.625604 | 0.149066 | 0.174441 | 0.877758 |

相对B0的error变化为百分比，PCC为绝对差：

| Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| GELU | +3.052% | +3.096% | −1.309% | −1.925% | +0.000319 |
| H64 | −4.603% | −4.247% | −0.006% | −0.600% | −0.001553 |
| DIRECT | −1.273% | −3.953% | +4.851% | +7.744% | −0.004158 |

GELU调制误差3/3 seed改善，Local RR仅1/3改善。H64 Whole RR为3/3改善、Local RR为2/3改善，PCC仅1/3改善。Direct的Local RR虽然均值降低，但只有1/3 seed改善；轨迹、调制与PCC均3/3退化，不能把RR均值收益概括为整体波形质量提升。

## 4. 排除670

| Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| B0 | 0.378613 | 0.447043 | 0.137637 | 0.156584 | 0.889405 |
| GELU | 0.386608 | 0.465625 | 0.135497 | 0.153418 | 0.890005 |
| H64 | 0.343110 | 0.406453 | 0.137378 | 0.155267 | 0.888061 |
| DIRECT | 0.355260 | 0.406721 | 0.143224 | 0.166153 | 0.886500 |

| 相对B0 | Whole RR变化 | Local RR变化 | Whole/Local RR改善seed数 | 包络轨迹变化 | 全局调制变化 | PCC差 |
|---|---:|---:|---|---:|---:|---:|
| GELU | +2.112% | +4.157% | 2/3、2/3 | −1.555% | −2.022% | +0.000601 |
| H64 | −9.377% | −9.080% | 3/3、3/3 | −0.188% | −0.841% | −0.001344 |
| DIRECT | −6.168% | −9.020% | 2/3、2/3 | +4.060% | +6.111% | −0.002904 |

H64的RR改善在该视图最一致，包络窗口误差接近基准。Direct的RR收益伴随轨迹与调制3/3退化；保留H64条件残差比直接删除提供了更好的包络均值表现。GELU仍是调制改善与RR均值代价并存，不能只看两个seed的改善票数忽略第三个seed的幅度。

## 5. 受试者等权与670单病例

Subject-macro先在每seed内对受试者等权，再取三seed均值。相对B0：

| 视图 | Arm | Whole RR变化 | Local RR变化 | 包络轨迹变化 | 全局调制变化 | PCC差 |
|---|---|---:|---:|---:|---:|---:|
| full | GELU | +3.697% | +1.497% | −0.387% | −1.391% | −0.001850 |
| full | H64 | −0.366% | +1.765% | +0.588% | +0.280% | −0.002980 |
| full | DIRECT | +2.951% | +2.579% | +6.567% | +8.122% | −0.010165 |
| exclude670 | GELU | +2.046% | +2.551% | −1.088% | −1.608% | −0.001023 |
| exclude670 | H64 | −5.626% | −5.807% | +0.044% | −0.376% | −0.002338 |
| exclude670 | DIRECT | −2.834% | −4.869% | +3.971% | +2.240% | −0.005963 |

H64在exclude670的subject-macro两项RR仍均3/3改善，但PCC在full与exclude670两种macro视图均3/3退化。全量macro Local RR均值增加1.765%，与窗口均值改善方向不同，说明人群构成与受试者权重是结论的重要组成部分。GELU窗口PCC均值微升也未在macro口径保留。

670单病例的三seed窗口均值：

| Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| B0 | 8.723295 | 6.421021 | 0.270149 | 0.312115 | 0.670451 |
| GELU | 9.089981 | 6.485933 | 0.276156 | 0.310406 | 0.662809 |
| H64 | 8.832202 | 6.758408 | 0.277218 | 0.320920 | 0.662978 |
| DIRECT | 9.135571 | 6.806946 | 0.314032 | 0.408523 | 0.630874 |

H64在670上的Local RR增加5.254%，轨迹/调制增加2.617%/2.821%，PCC下降0.007473。Direct在该病例的轨迹/调制误差增加16.244%/30.889%，PCC下降0.039576；轨迹与PCC三个seed均退化，调制均值退化但仅两个seed同向。这些结果只描述该病例，不外推为重度OSA亚组结论，也不凭指标认定具体生理或信号机制。

## 6. 与validation的关系

| Local RR相对B0变化 | Validation窗口均值 | Full test窗口均值 | Exclude670窗口均值 |
|---|---:|---:|---:|
| GELU | +0.213% | +3.096% | +4.157% |
| H64 | +0.515% | −4.247% | −9.080% |
| DIRECT | +0.079% | −3.953% | −9.020% |

GELU的RR代价与调制收益具有跨split均值方向一致性。H64的Local RR方向跨split反转，test提供了节律改善证据，同时保留PCC及670代价。Direct的包络与PCC退化在validation和test均出现，因此条件残差的形态保护作用得到延续。

维持validation已冻结的B0决定。若以后以节律为优先目标研究H64，应将其作为新问题，预先约定评价口径和独立验证；本轮test结果不用于追溯改选既有checkpoint或把三种独立因素组合成新模型。

## 7. 来源与验收

- 训练与validation来源：[validation完成记录](apor_gelu_refiner_v1_validation_results_20261001.md)。
- Test合同：[固定checkpoint附件](apor_gelu_refiner_v1_research_test_protocol_20261001.md)。
- Allowlist：`runs/apor_gelu_refiner_v1/research_test/allowlist_20260930T180254Z_beab6c619808`。
- Allowlist SHA256：`2ea95faaf110079f698b7d7fad1e581b4182df79cd5034ff210b5870591bf7ac`。
- Summary：allowlist下`summary/attempt_20260930T181130Z_bcf7686dd1a7`，于北京时间2026-10-01 02:11创建。
- Summary manifest SHA256：`7b38d042c649f19d745a3c24db57ee7097061e2a146fc9996684c46f76fbb32d`。
- Dispatch：allowlist下`dispatch/attempt_20260930T180324Z_656546693357`，worker退出码`[0,0]`。

完成后只读校验allowlist、dispatch、summary及全部12个评价来源的manifest/freeze；核对九项评价的checkpoint哈希与selected epoch。使用保存的逐窗口metrics独立复算三视图的三seedmean/sample SD、subject-macro、配对均值与改善seed数，核对窗口数、样本顺序、target资格、五指标有限性和退化标志，均通过。未重新推理、重训或改选checkpoint。

原生辅助指标与分母保存在`native_metrics_per_seed.csv`、`denominators.csv`，本记录的主要解释基于冻结五项主指标。本轮评价完成并关闭。
