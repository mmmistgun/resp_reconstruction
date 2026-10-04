# PATCH/APOR：validation 与排除670的test综合结论

日期：2026-09-30。本文按用户要求，对已完成实验进行结果整理。

## 1. 综合结论

**M0保留了接近W0的RR水平并改善包络重建；完整APOR的稳定增量是PCC，validation中的RR优势未在排除670的test延续；密集主干A3在该test子集上恢复了RR表现，但与A0的排序跨split反转。现有证据支持围绕时间网格与调制后解码开展机制研究，尚未形成一个跨split、五指标均占优的新结构。**

模块证据中，Mamba对节律和形态的贡献最明确；W条件、尺度邻域编码及H65各有属性作用。解码残差的validation删除收益未复现。FiLM双路径的PCC均值优势保留，其Local RR优势则依赖比较对象和人群。

## 2. 分析口径

- Validation：全部2675窗口、7个samp_id，保留原始口径。
- Test子集：仅排除`samp_id=670`的79窗口，剩余2231窗口、7个samp_id（220、229、286、671、704、726、1006）；所有模型使用同一子集。
- 用户提供670为重度OSA并伴一定体动的病例背景。本分析未独立核实疾病分级或量化体动；排除670用于病例敏感性分析，剩余7人不直接命名为正常或非重度OSA人群。
- Seeds：20260811、20260812、20260813。Checkpoint沿用完整validation Local RR严格最小、并列最早的既定选择。
- 主表在每个seed内按窗口平均，再对三seed等权；subject-macro先按samp_id聚合，再对7人等权、对三seed等权。
- 前四项误差越低越好，PCC越高越好。误差变化统一为`100×(候选三seed均值/参照三seed均值−1)`；PCC报绝对差。Seed方向描述训练波动，不代表人群显著性。

排除670是看到全量结果后、由用户指定的事后敏感性分析；research-test本身属于复用研究开发证据。本文与[完整test结果](patch_apor_v1_research_test_results_20260930.md)并列保存，原始test样本集合、训练合同、selector和冻结产物不变。

## 3. 主要模型的两阶段表现

三seed窗口均值；RR单位bpm。各阶段内比较模型，不把不同人群的validation/test绝对值差当作泛化改善。

| 阶段 | 模型 | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---|---:|---:|---:|---:|---:|
| Validation | 原始W0 | 0.499298 | 0.551309 | 0.152729 | 0.191051 | 0.865300 |
| Validation | M0 | 0.504329 | 0.549437 | 0.147534 | 0.178862 | 0.863972 |
| Validation | A0 完整APOR | 0.453363 | 0.526085 | 0.146643 | 0.165625 | 0.871033 |
| Validation | A3 密集主干 | 0.498222 | 0.555214 | 0.149623 | 0.170746 | 0.865254 |
| Test排除670 | 原始W0 | 0.359658 | 0.401268 | 0.134765 | 0.167819 | 0.884349 |
| Test排除670 | M0 | 0.358590 | 0.412134 | 0.133851 | 0.157226 | 0.886799 |
| Test排除670 | A0 完整APOR | 0.378613 | 0.447043 | 0.137637 | 0.156584 | 0.889405 |
| Test排除670 | A3 密集主干 | 0.331549 | 0.396090 | 0.135669 | 0.154001 | 0.884564 |

### 3.1 M0：接近W0的RR与包络改善

相对原始W0，validation Whole/Local RR变化为+1.01%/−0.34%，test子集为−0.30%/+2.71%。M0的test Local RR绝对增加0.0109 bpm，三个seed均更差；subject-macro增加1.15%。因此应表述为RR接近并伴有局部节律代价，而非保持全部RR质量。

M0的包络轨迹与全局调制均值在两个split均改善；全局调制分别降低6.38%/6.31%，均3/3同向。PCC在validation下降0.001329、test子集提高0.002449，不能将其写成跨split稳定收益。

### 3.2 A0：PCC收益稳定，RR与轨迹收益依赖split

| A0相对M0 | Validation变化 | 改善seed | Test排除670变化 | 改善seed |
|---|---:|---:|---:|---:|
| Whole RR | −10.11% | 3/3 | +5.58% | 1/3 |
| Local RR | −4.25% | 3/3 | +8.47% | 0/3 |
| 包络轨迹 | −0.60% | 2/3 | +2.83% | 1/3 |
| 全局调制 | −7.40% | 3/3 | −0.41% | 1/3 |
| PCC | +0.007061 | 3/3 | +0.002606 | 3/3 |

完整APOR的PCC优势在两个split、两种聚合口径均保留。Test子集全局调制均值改善较小且仅1/3 seed同向，RR与轨迹的validation优势发生反转。相对原始W0，A0的test子集Local RR窗口均值增加11.41%、subject-macro增加4.92%，两种口径均3/3退化；故其Local RR问题并非只由670驱动。

### 3.3 A3：密集网格具有RR研究价值，尚无跨split优越性

相对A0，A3在validation的Whole/Local RR增加9.89%/5.54%，均3/3退化；test子集则降低12.43%/11.40%，均2/3改善。A3的PCC在两个split均3/3低于A0。时间网格改变形成了明显的节律—形态取舍，且RR排序依赖split。

相对原始W0，A3的validation Whole/Local RR变化仅−0.22%/+0.71%；test子集为−7.82%/−1.29%，均2/3改善。其test Local RR仅降低0.0052 bpm，适合描述为接近W0；Whole RR均值收益更明显。A3的全局调制在两个split均3/3优于W0，但test轨迹误差增加0.67%，同为3/3。

Test子集subject-macro对W0的Whole/Local RR变化分别为：M0 −3.02%/+1.15%，A0 −2.40%/+4.92%，A3 −9.36%/−0.89%。A3的RR均值方向在两种聚合下保持，不能据此消除validation排序反转的事实。

## 4. 模块结论

M1–M7相对M0，A1–A4相对A0；下列百分比为窗口均值变化。

| 模块 | Validation证据 | Test排除670证据 | 综合判断 |
|---|---|---|---|
| Mamba（M2删除） | Whole/Local RR +12.54%/+17.17%，PCC −0.061733，均3/3退化 | Whole/Local RR +25.95%/+21.26%，PCC −0.054260，均3/3退化 | 主干对RR/PCC的贡献最稳定 |
| W条件（M1删除） | Local RR +2.30%、轨迹+3.71%，均3/3退化 | Local RR +4.48%、轨迹+2.86%，各2/3退化 | 保留条件分支的证据集中在局部节律与轨迹；test一致性弱于validation |
| 尺度邻域编码（M3删除） | 轨迹+2.27%、调制+1.84%，均3/3退化 | 轨迹+1.57%、调制+2.83%，均3/3退化 | 对两项包络窗口指标有跨split稳定贡献；Whole RR不支持逐项优势 |
| H65（M4删除） | 两项RR均值均恶化 | Local RR +1.70%，Whole RR反而−1.44% | Local RR均值收益保留，Whole RR方向依赖split |
| 解码残差（M5删除） | 两项RR与其余三项窗口均值均改善 | 两项RR +4.53%/+7.33%，均3/3退化，五项窗口均值均不利 | 删除收益未跨split复现，当前支持保留残差 |
| 双路径FiLM（M6/M7改为单路径） | 双路径PCC均值更好；Local RR均值也更好 | 双路径PCC均值更好；M6的Local RR比M0低0.39%，差异很小 | PCC均值证据保留，不能把双路径Local RR优势写成普遍结论 |
| 五点条件（A1改中心单点） | 五点条件的RR均值、调制和PCC均值更好 | 五点条件两项RR均值更好，中心单点的轨迹、调制和PCC均值更好 | 两项RR均值方向保留，其他属性优势依赖split及seed |
| 非线性Patch decoder（A2改线性） | 线性版窗口RR略好，轨迹/调制/PCC均值较差 | 线性版两项RR和PCC均值较差，轨迹/调制较好 | 非线性版PCC均值优势跨split保留，RR和包络效应反转 |
| 重叠权重（A4改均匀） | 均匀权重PCC +0.000616，3/3改善；轨迹较差 | 均匀权重PCC +0.000806，3/3改善；Local RR +3.22% | 均匀权重的PCC收益稳定，Hann在test子集保留Local RR均值优势 |

上述结构消融回答的是当前训练与checkpoint选择合同下的比较，不把重训练差异解释为已证明的单一信号机制。

## 5. TM3×REF2：作为专项机制问题保留

使用历史三因素实验的同轮Patch四臂，而非把其结果与本轮M0混合成一个实验。排除670后：

| 结构 | Whole RR | Local RR |
|---|---:|---:|
| TM0/REF0 | 0.358053 | 0.411881 |
| TM3/REF0 | 0.362381 | 0.412010 |
| TM0/REF2 | 0.343625 | 0.408014 |
| TM3/REF2 | 0.358667 | 0.396865 |

完整组合相对TM0/REF0：validation Whole/Local RR均值约−0.91%/+0.19%；test子集约+0.17%/−3.65%，后者Local RR为3/3改善。但是，test子集subject-macro Local RR基本持平（+0.012%），Whole RR增加7.85%。因此，两模块对窗口Local RR的联合价值值得研究，但当前证据不足以认定恢复两模块是跨split、跨受试者稳定改善。

## 6. 下一步研究的含义

1. **优先解释APOR的时间网格与FiLM后解码。** 固定结构比较已显示RR排序反转；671的已有配对结果提供了具体诊断对象。应区分时间网格、条件注入与片段读出对主频选择的影响，当前尚未测量峰值切换、倍频/半频或相位漂移。
2. **TM3×REF2作为局部节律协同问题继续保留。** 同时观察窗口均值与受试者等权，明确它更可能帮助哪些病例，而不是按全量平均效应直接恢复。
3. **结构基线保持清晰。** W0为原始参考，M0为本轮消融基准；A0用于研究PCC与高效Patch重建，A3用于研究节律保持。当前综合结果不自动产生模型或checkpoint替换决定。

本文只整理既有证据，不启动下一阶段实验。APOR训练峰值显存收益已由validation完成记录保存，其测量是训练/validation流程记录，不作为本次RR机制的解释变量。

## 7. 来源与核对

- [Validation完成记录](patch_apor_v1_validation_results_20260930.md)。原始summary：`runs/patch_apor_v1/session_20260929T181607Z_77e89165fef0/summary/attempt_20260930T033450Z_da517e1ea2f1`。
- [完整research-test结果](patch_apor_v1_research_test_results_20260930.md)。原始summary：`runs/patch_apor_v1/research_test/allowlist_20260930T071732Z_9b85989bdc0b/summary/attempt_20260930T074924Z_e2a86b361eb4`，manifest SHA256=`79aa5d4deaeae62d2bbfe2925130035196300e053aa690b1e376aac5481a3a6d`。
- 原始W0：`docs/experiments/w0_film_gamma_test_lock_20260919.json`中的三份`baseline_test`；同一训练目录的`metrics.csv`提供validation。Test seed mean与[W0冻结结果](w0_film_gamma_test_results_20260919.md)一致。
- [W0三因素结项](w0_structural_factorial_v1_closeout_20260925.md)；test子集复算输入为其冻结summary中的`subject_stratified_metrics.csv`。

排除670的复算方法：对每份既有test指标表保留`samp_id != 670`，逐seed复算五指标窗口均值和subject-macro，再计算三seed均值、sample SD与同seed方向。前次子集核对已验证42份原始W0/本轮test指标文件哈希及完整2310-row顺序一致；过滤后均为2231行、7人。三因素按逐受试者`eligible_n`加权恢复窗口均值，分母同为2231。整理时再次核对主要模型的两阶段数值及配对方向，全部来自冻结指标，不读取原始波形或checkpoint，不执行推理。
