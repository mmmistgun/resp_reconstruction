# APOR激活规范化：validation与research-test完成记录

日期：2026-09-30。训练协议 `apor-activation-v1-20260930`，test协议 `apor-activation-v1-research-test-20260930`。

状态：**U1三seed训练、完整validation决定及固定三checkpoint research-test均已完成并冻结。** U0复用原A0/N0三seed训练与test；test双worker退出码均为0。预设validation规则保留U0。

## 1. 结论

**本轮不支持将四处GELU统一替换为SiLU。** U1在validation上的Whole RR和全局包络调制误差超过预设退化保护线。Test呈现属性取舍：包络误差略降、PCC略升，同时Whole/Local RR误差增大；排除670后RR代价仍存在。保留U0是test前完整validation已经冻结的决定。

结论针对固定N0结构、四处激活替换与三个训练seed，不外推为SiLU在其他结构上普遍更差。本test属于复用的研究开发证据；三seed方向和sample SD描述训练随机性，不构成独立人群确认或统计显著性检验。

## 2. 比较与统计口径

- U0：原A0/N0，PatchMixer中的GELU保留；U1：两个PatchMixerBlock各两处GELU替换为SiLU，共四处。归一化和其他结构不变，两臂均1,077,640参数。
- Seeds：20260811、20260812、20260813。U1均在第30 epoch早停，best epoch为8/13/9；U0固定best epoch为8/13/13。
- Validation：每cell 2675窗口、7人；test full为2310窗口、8人，exclude670为2231窗口、7人，subject670为79窗口、1人。子集从同一完整评价派生。
- 下表先计算每seed的窗口均值，再平均三个seed。前四项越低越好，PCC越高越好，RR单位bpm。sample SD、逐seed、配对差、subject-macro、逐受试者和分母均保存在summary。
- 新增test 6930行，包含复用U0的完整科学矩阵为13860行。五项主指标全部有限，逐cell行数与row顺序一致，joint/envelope Spearman prediction degeneracy均为0。

## 3. Validation及预设决定

| Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| U0 | 0.453363 | 0.526085 | 0.146643 | 0.165625 | 0.871033 |
| U1 | 0.469584 | 0.523613 | 0.146073 | 0.169947 | 0.870570 |

U1的Whole RR增加3.58%、全局调制误差增加2.61%，超过四项error各自最多退化0.5%的预设工程保护线。Local RR和轨迹误差分别下降0.47%/0.39%，PCC下降0.000462；全局调制误差三个seed均更差。

Subject-macro下，Whole RR和全局调制误差分别增加6.40%/3.00%，同样未通过保护线。因此冻结的`validation_decision.json`为`candidate=U0`、`eligible_arms=[U0]`、`test_used=false`。保护线是工程规则，不是统计非劣效界值。

## 4. 完整test

| Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| U0 | 0.663993 | 0.651348 | 0.142169 | 0.161903 | 0.881917 |
| U1 | 0.689915 | 0.671748 | 0.138986 | 0.158371 | 0.883156 |

U1的Whole/Local RR误差增加3.90%/3.13%，包络轨迹/全局调制误差下降2.24%/2.18%，PCC增加0.001239。Whole RR三个seed均更差，Local RR仅1/3 seed改善；轨迹误差3/3改善，调制与PCC各2/3改善。

Subject-macro下，Whole/Local RR误差增加1.16%/1.03%，轨迹/调制误差下降1.91%/0.32%，PCC增加0.002808。RR与包络之间的取舍在受试者等权后仍存在。

## 5. 排除670及单病例视图

| 视图 | Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---|---:|---:|---:|---:|---:|
| exclude670 | U0 | 0.378613 | 0.447043 | 0.137637 | 0.156584 | 0.889405 |
| exclude670 | U1 | 0.406758 | 0.468661 | 0.134434 | 0.152576 | 0.890377 |
| subject670 | U0 | 8.723295 | 6.421021 | 0.270149 | 0.312115 | 0.670451 |
| subject670 | U1 | 8.686438 | 6.407022 | 0.267544 | 0.322013 | 0.679214 |

排除670后，Whole/Local RR误差增加7.43%/4.84%，改善seed数仍为0/3和1/3；轨迹/调制误差下降2.33%/2.56%，均3/3改善，PCC增加0.000973。Subject-macro Whole/Local RR也增加6.30%/3.73%。因此，U1的RR代价并非仅由670驱动。

670单病例中，Whole/Local RR分别下降0.42%/0.22%，各2/3 seed改善；轨迹误差下降0.96%，调制误差增加3.17%，PCC均值增加0.008763但仅1/3 seed改善。它是单病例属性描述，不能据此推断重度OSA亚组收益或具体失败机制。

## 6. 故障恢复与验收

原并行训练中seed 20260812已完成训练，但最终validation导出在共享参考文件锁处失败；seed 20260813随后在来源审计时被中断。恢复依据[工程附件](apor_activation_v1_recovery_20260930.md)执行：保留失败现场，seed 20260812在新attempt补齐导出，seed 20260813串行完成原定训练。

seed 20260812恢复期间optimizer更新数为0；派生checkpoint仅修改内嵌输出目录，权重、optimizer及其余训练状态精确一致，恢复预测与故障前保存的预测逐元素完全一致。恢复代码3项CPU合成测试通过。原训练峰值内存未保存，明确记为未知。

完成后只读核对两份summary、12个来源cell的manifest/freeze及allowlist，共228次文件身份校验通过；从保存的逐窗口metrics独立复算三seedmean/sample SD及subject-macro通过，核对窗口分母、row顺序、五指标有限性与退化标志通过。本次核对没有重新训练或推理，也未改选checkpoint。

## 7. 冻结来源

- Training session：`runs/apor_activation_v1/session_20260930T120541Z_324e56049c8a`。
- Session SHA256：`b49dae4138aaec8b51072f70b0b2b1caf02735c901317d00e70b92d6c264df4d`。
- Validation summary：session下`summary/attempt_20260930T142735Z_fcf878c8003c`。
- Validation manifest SHA256：`a7501b7805513485ba67859c6e93522a932b8072638b3dd3a25a427a3445f740`。
- Test allowlist：`runs/apor_activation_v1/research_test/allowlist_20260930T142738Z_e3f7830e2e3d`。
- Allowlist SHA256：`c79e10fcff3c17abc468a84480c99c938a2e6a143bb5b33e3a95b77a0fb7d1cc`。
- Test summary：allowlist下`summary/attempt_20260930T143110Z_b5f3489d02c5`。
- Test manifest SHA256：`4eb39e43a34485586a687f0b1369a4d6dd73f0b5e712dca45ae6123648f4af6c`。
- 端到端完成receipt：training session下`pipeline_result.json`。

本轮train/validation与research-test执行关闭。前序网格/解码实验的完整配对波形和频谱机制诊断仍是独立待办；当前指标结果不足以确认RR与包络取舍的具体机制。
