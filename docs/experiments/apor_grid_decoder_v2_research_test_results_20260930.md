# APOR v2：固定checkpoint research-test三视图结果

日期：2026-09-30。协议：`apor-grid-decoder-v2-research-test-20260930`。

状态：**六项新增N1/D1评价与完整12-cell三视图汇总均已完成并冻结。** N0/D0复用六份既有test；两路worker退出码0，新增成功6、失败0。验证期决定保持N0作为激活规范化的U0来源，不据test重新选择模型或epoch。

## 1. 综合判断

**本轮test没有为将该调制后时间细化块加入APOR提供跨属性一致的支持。** 原生网格N1的Local RR在validation与两种多受试者test视图均值上都变差。密集网格D1的全量test Local RR均值略有改善，但排除670后方向反转，且其轨迹、调制与PCC均值均不利。

该结果明确了人群构成的作用：D1在670上的RR改善伴随包络及PCC代价，不能把该病例的RR收益直接概括为完整波形追踪改善。当前结论针对已固定的时间残差块与checkpoint，具体频谱或相位机制仍需诊断。

## 2. 样本、checkpoint与统计

- Seeds：20260811、20260812、20260813。
- 新评价N1 selected epoch=8/9/22，D1=29/19/10；N0/D0绑定既有A0/A3的同seed固定checkpoint。
- Full：2310窗口、8人；exclude670：2231窗口、7人；subject670：79窗口、1人。
- 完整12-cell共27720行窗口指标，其中新增13860行。子集视图从相同指标派生，不作为额外独立样本重复。
- 所有cell的五主指标完整有限，prediction degeneracy为0，row ID、顺序及target资格一致。

以下先按各seed窗口平均，再取三seed均值。前四项越低越好，PCC越高越好，RR单位bpm；sample SD、逐seed及subject-macro完整保存在summary中。该test属于reused research/development evidence，排除670为已声明的病例敏感性视图。

## 3. 完整test

| Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| N0 | 0.663993 | 0.651348 | 0.142169 | 0.161903 | 0.881917 |
| N1 | 0.687045 | 0.684599 | 0.142326 | 0.162496 | 0.882129 |
| D0 | 0.635045 | 0.623972 | 0.140419 | 0.160208 | 0.877345 |
| D1 | 0.640608 | 0.614279 | 0.144577 | 0.166619 | 0.876401 |

N1相对N0：Whole/Local RR均值增加3.47%/5.10%；D1相对D0：Whole RR增加0.88%、Local RR下降1.55%。两组Local RR均仅1/3 seed改善，不能把D1的均值收益描述为跨seed一致。D1的轨迹和调制误差分别增加2.96%/4.00%，PCC下降0.000944。

## 4. 排除670的test

| Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| N0 | 0.378613 | 0.447043 | 0.137637 | 0.156584 | 0.889405 |
| N1 | 0.389172 | 0.470853 | 0.137461 | 0.156760 | 0.889789 |
| D0 | 0.331549 | 0.396090 | 0.135669 | 0.154001 | 0.884564 |
| D1 | 0.355295 | 0.408664 | 0.139213 | 0.158363 | 0.884308 |

| 相对匹配参照 | Whole RR变化 | Local RR变化 | Local RR改善seed数 | Subject-macro Whole/Local RR变化 |
|---|---:|---:|---:|---:|
| N1−N0 | +2.79% | +5.33% | 1/3 | +3.19% / +4.61% |
| D1−D0 | +7.16% | +3.17% | 1/3 | +11.07% / +3.79% |

D1的Whole RR在三个seed均更差；五项窗口均值全部不利。N1的轨迹与PCC均值略有改善，各仅1/3 seed同向；RR代价在窗口与受试者等权下保留。因此，排除670并未使新增时间细化获得一致收益。

## 5. 670单病例

| Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| N0 | 8.723295 | 6.421021 | 0.270149 | 0.312115 | 0.670451 |
| N1 | 9.099133 | 6.720871 | 0.279715 | 0.324507 | 0.665788 |
| D0 | 9.205931 | 7.059460 | 0.274539 | 0.335488 | 0.673475 |
| D1 | 8.697984 | 6.420944 | 0.296049 | 0.399774 | 0.653112 |

D1相对D0的Whole/Local RR改善5.52%/9.04%，各2/3 seed同向；轨迹和全局调制误差却增加7.84%/19.16%，均3/3退化，PCC下降0.020363。670的RR下降足以抵消剩余窗口的部分Local RR增加，导致全量Local RR均值方向翻转。

用户提供670为重度OSA伴一定体动的病例背景；本结果只描述该病例的模型属性取舍，不外推为重度OSA亚组效果，也不由这些指标认定具体失败机制。

## 6. 与validation的关系及后续

| Local RR变化 | Validation | Full test | Test排除670 |
|---|---:|---:|---:|
| N1相对N0 | +0.83% | +5.10% | +5.33% |
| D1相对D0 | +2.72% | −1.55% | +3.17% |

Validation预设质量规则没有使N1/D1晋级，当前test也未形成加入该模块的跨属性依据。后续激活规范化继续使用test前已固定的N0，精确比较四处GELU→SiLU，不改变网格、归一化、条件或解码结构。

本轮train/validation与research-test执行关闭。新增训练预测和重叠标量已保存，但完整N0/D0预测导出及配对频谱、片段一致性图尚待机制诊断；不能将其写成已验证的解释。

## 7. 来源与核对

- Allowlist：`runs/apor_grid_decoder_v2/research_test/allowlist_20260930T115416Z_3dc620df72a1`。
- Allowlist SHA256：`20d1fd1d96839bd32ec5b7cf7c09bd1f8d053f560ef71de9fd8e5dfe032dee7a`。
- Summary：上述allowlist下`summary/attempt_20260930T120143Z_d828889cf244`。
- Summary manifest SHA256：`2c38233f481a7f493602a2629f9d07a0e9f75c281ff3cf906b280f07469f9d80`。
- Dispatch：上述allowlist下`dispatch/attempt_20260930T115637Z_4e3c49c78a37/result.json`。
- 来源合同见[test附件](apor_grid_decoder_v2_research_test_protocol_20260930.md)，训练结果见[validation记录](apor_grid_decoder_v2_validation_results_20260930.md)。

完成后核对allowlist、summary与12个评价来源manifest/freeze及96项来源文件哈希；三视图分母、五指标有限值、三seedmean/SD及subject-macro独立复算通过。全部来源一次性汇总，未重新推理或改选checkpoint。原生辅助指标、资格与分母保存在`native_metrics_per_seed.csv`和`denominators.csv`。
