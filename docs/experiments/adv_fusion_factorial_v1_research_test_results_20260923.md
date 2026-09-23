# ADV 融合方式 × 位置：research-test 结果

日期：2026-09-23。协议：`adv-fusion-factorial-v1-es30p15-research-test-20260923`。

**状态：用户完成独立test cache、18项固定checkpoint评价及完整汇总；只读来源与数值核对通过。** 本文件记录实际完成状态，接续实现协议中的历史进度描述。已完成阶段不自行重跑，原始checkpoint、cache、CSV、manifest及协议快照保持原状。

主要判断：后融合FiLM（D）的Whole/Local RR均值最低，后融合attention（F）的trajectory误差均值最低、PCC均值最高，前融合拼接（A）的global modulation误差均值最低。各属性存在取舍。部分位置效应延续validation，部分方式排序与交互发生变化；不能将validation中的所有交互方向当作稳定结论。

Test split已参与既有研究，证据属性为 **reused research/development evidence**，不称为未触及held-out的独立确认。

## 1. 来源与核对

- [专项协议](adv_fusion_factorial_v1_research_test_protocol_20260923.md)、[validation结果](adv_fusion_factorial_v1_results_20260923.md)。
- [完整test汇总manifest](../../runs/adv_fusion_factorial_v1_research_test/summary_full_es30p15_20260923_r1/manifest.json)，完成于2026-09-23 02:29:00，Asia/Shanghai。
- Summary manifest SHA-256：`892d6735f3d4dba7d9a79c187fe0c7a6d3f608230b7817009299997e1aede640`。
- 候选锁文件SHA-256：`2c80d570168b0de907a348bd1a1bdd8b476203623fc83ba64462377dd0ea8add`；lock ID：`f15640f215d21c9cea67d9b3ca1949fd19eacc4eb575a208dddfc66993fa9f72`。
- Test执行代码SHA-256：`d723d279a4ec38d89f61c2f4ccfaeb9e1ae947dd8016d40a19ec9be171b3cb32`；训练代码保持`f9381a7d1bf48a736f4a34e549d2bb440c9eb057e2f2e789a4e5364737870bce`。
- Cache ID：`f3971cfd3841a859033cd75de92916b53f7cc9191fba3ed0a1b21522dab5fcb3`；cache manifest SHA：`1d39f4be68c96abc117e405797703cfd9bd89a57b845eef8bb1ec00a5f5a0f13`。
- Comparison ID：`e1c51b9743362e579736562e4cd1bae1f0f59c064ed6b36da3d7ed023b2d89f1`。
- 共享test sample内容身份：`e1bd52f2cdc315eb46aa13ba9863d7d96de1a6af8a2572b33981b151a32decef`。

核对18个唯一arm/seed、成功回执与源manifest、原训练config、selected checkpoint文件哈希、selected/stopped epoch及方式/位置标签。所有评价均使用同一cache、CUDA、bf16、batch32；当前代码与执行身份一致，18项源码快照逐文件哈希通过。

每项均为同序2310个唯一test窗口、8个samp_id，row-ID哈希与候选锁一致。输入/target/mask内容身份一致；五主指标每项分母均为2310且有限，共核对41580行指标，prediction degeneracy fraction全部为0。

每项`metrics.csv`、`metrics_summary.csv`、`samples.json`与`access.jsonl`哈希通过。访问记录均按input-only推理开始、全部推理完成、target开始、target完成排列，四阶段计数完整。Checkpoint未重选。所有项仅有成功的`attempt_r1`，未发现失败或未闭合尝试。

逐sample指标到单项summary、18行per-seed、78行across-seed以及240/80行因子差分表的数值一致性通过。Cache核对限于完成回执、身份与表示元数据；本次未读取特征payload或原始波形、未运行模型、未重建cache或summary。Checkpoint采用文件哈希核对。

## 2. 五主指标

完整180 s任务，沿用冻结target、Pi、指标及eligibility。每seed按窗口direct mean，三个seed等权汇总为mean ± sample SD。SD只描述训练随机性；三个seed不作为三个独立人群，窗口之间也不能视为独立样本。

| 组别 | Whole RR ↓，bpm | Local RR ↓，bpm | 包络轨迹 ↓ | 全局包络调制 ↓ | Signed PCC ↑ |
|---|---:|---:|---:|---:|---:|
| A 前融合拼接 | 0.77570 ± 0.07231 | 0.81669 ± 0.09891 | 0.14702 ± 0.00968 | 0.16799 ± 0.00776 | 0.85748 ± 0.00610 |
| B 后融合拼接 | 0.76015 ± 0.03321 | 0.72283 ± 0.04305 | 0.14237 ± 0.00249 | 0.19850 ± 0.01812 | 0.85905 ± 0.00131 |
| C 前融合FiLM | 0.76414 ± 0.02929 | 0.74322 ± 0.07359 | 0.14297 ± 0.00542 | 0.17387 ± 0.00582 | 0.86054 ± 0.00708 |
| D 后融合FiLM | 0.71861 ± 0.01213 | 0.70308 ± 0.05755 | 0.14306 ± 0.00999 | 0.17797 ± 0.00302 | 0.85709 ± 0.01141 |
| E 前融合attention | 0.74205 ± 0.04601 | 0.73663 ± 0.04938 | 0.14424 ± 0.00693 | 0.17233 ± 0.00317 | 0.85939 ± 0.01087 |
| F 后融合attention | 0.78258 ± 0.00681 | 0.75487 ± 0.04803 | 0.13830 ± 0.00104 | 0.19191 ± 0.03598 | 0.86134 ± 0.00578 |

完整逐seed数据：[per_seed.csv](../../runs/adv_fusion_factorial_v1_research_test/summary_full_es30p15_20260923_r1/per_seed.csv)；全部均值、SD及分母：[across_seed.csv](../../runs/adv_fusion_factorial_v1_research_test/summary_full_es30p15_20260923_r1/across_seed.csv)。

D与F体现不同质量取舍。F相对D的Whole RR和Local RR在三个seed均更差，尽管F的trajectory和PCC均值更好。F的PCC均值最高并不表示它全面优于其他组；本轮不由test结果重选模型或构造总分。

## 3. 同位置的方式差异

候选减参照，表中为三seed均值差；括号为改善seed数。四误差指标负值为改善，PCC正值为改善。全部配对SD及逐seed差见[差分明细](../../runs/adv_fusion_factorial_v1_research_test/summary_full_es30p15_20260923_r1/contrasts_per_seed.csv)和[差分汇总](../../runs/adv_fusion_factorial_v1_research_test/summary_full_es30p15_20260923_r1/contrasts_across_seed.csv)。

| 对比 | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| C−A：前置FiLM−拼接 | −0.011560（2/3） | −0.073462（2/3） | −0.004050（1/3） | +0.005884（1/3） | +0.003061（2/3） |
| D−B：后置FiLM−拼接 | −0.041534（3/3） | −0.019750（2/3） | +0.000689（2/3） | −0.020526（3/3） | −0.001961（2/3） |
| E−A：前置attention−拼接 | −0.033652（2/3） | −0.080061（2/3） | −0.002785（1/3） | +0.004348（1/3） | +0.001909（2/3） |
| F−B：后置attention−拼接 | +0.022430（1/3） | +0.032044（0/3） | −0.004065（3/3） | −0.006595（2/3） | +0.002292（2/3） |
| E−C：前置attention−FiLM | −0.022093（3/3） | −0.006599（1/3） | +0.001265（0/3） | −0.001536（1/3） | −0.001152（2/3） |
| F−D：后置attention−FiLM | +0.063964（0/3） | +0.051794（0/3） | −0.004755（1/3） | +0.013931（2/3） | +0.004253（2/3） |

后置FiLM相对后置拼接的Whole RR与global modulation在三个seed均改善，但PCC均值下降。前置attention相对前置FiLM的Whole RR三个seed均改善，而后置attention相对后置FiLM三个seed均更差，呈现方式排序随位置变化的描述性交叉。

部分均值变化由少数seed主导，例如F−D的trajectory均值改善但只有1/3 seed改善；应同时阅读配对方向，避免只报告均值优势。

## 4. 同方式的位置效应

定义为post−pre，格式同上。

| 对比 | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| 拼接 B−A | −0.015551（2/3） | −0.093856（2/3） | −0.004655（2/3） | +0.030513（0/3） | +0.001569（2/3） |
| FiLM D−C | −0.045525（3/3） | −0.040145（2/3） | +0.000085（2/3） | +0.004103（1/3） | −0.003453（0/3） |
| attention F−E | +0.040532（1/3） | +0.018249（0/3） | −0.005936（3/3） | +0.019571（2/3） | +0.001952（2/3） |

后移拼接的Local RR均值改善较大，同时global modulation三个seed均恶化。FiLM后移使Whole RR三个seed均改善，PCC三个seed均下降。Attention后移使trajectory三个seed均改善，但Local RR也在三个seed均恶化；global modulation均值变差主要由seed20260812推动，另两个seed略改善。

因此，“后融合更好”缺少统一的属性含义；方式与位置需共同解释。

## 5. 方式 × 位置交互与validation对照

交互定义为 `(m2_post−m2_pre)−(m1_post−m1_pre)`，表中为mean ± sample SD。交互正负表示位置效应之差，不直接代表总体优劣。

| 方式对 | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| FiLM−拼接 | −0.029975 ± 0.074118 | +0.053712 ± 0.112048 | +0.004739 ± 0.013406 | −0.026410 ± 0.006470 | −0.005022 ± 0.008841 |
| attention−拼接 | +0.056082 ± 0.049422 | +0.112105 ± 0.083401 | −0.001281 ± 0.011284 | −0.010942 ± 0.038343 | +0.000383 ± 0.004573 |
| attention−FiLM | +0.086057 ± 0.032920 | +0.058393 ± 0.038029 | −0.006020 ± 0.010819 | +0.015468 ± 0.033540 | +0.005405 ± 0.009556 |

延续较好的描述性信号：

- 拼接后移的global modulation代价、FiLM后移的PCC下降、attention后移的trajectory改善，在validation和test均为三个seed同向。
- attention−拼接的Whole RR交互在两个split均为三个seed正值；test上Local RR交互也为三个seed正值。
- FiLM−拼接的global modulation交互在两个split均为三个seed负值，即FiLM的位置变化带来的调制误差增量小于拼接；不等于FiLM后移本身改善global modulation。

需要修正或限定的validation解释：

- Whole RR的FiLM−拼接交互由validation的+0.020999变为test的−0.029975，test的配对SD为0.074118且seed方向混合。不能把“拼接后移比FiLM后移更有利于Whole RR”作为跨split稳定结论。
- Validation最低RR均值来自B，test来自D。Attention后移的global modulation在validation三个seed均恶化，在test则只有一个seed恶化而主导均值。
- Attention后移的PCC均值由validation略降变为test略升；两边均不能概括为稳定的PCC优势。

Test上的attention−FiLM Whole/Local RR交互均为三个seed正值，结合pre/post两端的方式排序支持位置相关响应的描述。但仅有三个训练seed，且test复用，不能据此作人群显著性或因果生理解释。三组两两交互只有两个线性独立，不能作为三个独立证据。边际效应也已完整保存在差分表中，不替代上述简单效应。

## 6. 辅助指标

以下为三个seed等权均值，SD与全部合法NA分母保留在原CSV。

| 组别 | IBI MedAE ↓，s | IBI coverage ↑ | Coherence ↑ | nDTW ↓ | 包络Spearman 低 / 中 / 高 |
|---|---:|---:|---:|---:|---|
| A | 0.12170 | 0.74395 | 0.51737 | 0.22811 | 0.35138 / 0.50319 / 0.71190 |
| B | 0.11263 | 0.78059 | 0.55013 | 0.22974 | 0.33282 / 0.50043 / 0.71378 |
| C | 0.11300 | 0.76095 | 0.53841 | 0.22605 | 0.39407 / 0.52513 / 0.71913 |
| D | 0.12036 | 0.75565 | 0.53533 | 0.23129 | 0.37300 / 0.50872 / 0.71939 |
| E | 0.11510 | 0.75948 | 0.53755 | 0.22731 | 0.38532 / 0.51860 / 0.71709 |
| F | 0.10803 | 0.79432 | 0.57257 | 0.22977 | 0.41863 / 0.53829 / 0.72828 |

F在多项辅助均值上较好，但nDTW均值最低的是C，RR均值最低的是D。Attention结果仍代表整个融合模块的效果，而不是权重的单独贡献。

IBI MedAE具有可解释窗口筛选。各seed的有效窗口数分别为：A 1400/1158/1196，B 1326/1344/1367，C 1397/1138/1353，D 1412/967/1383，E 1360/1142/1374，F 1421/1449/1366。不同模型的IBI误差不能脱离coverage和这些计数比较，IBI coverage也不等于可解释窗口比例。

## 7. 历史参照与证据边界

既有W0 research-test均值为Whole RR0.617234、Local RR0.609566、trajectory0.139546、global modulation0.173418、PCC0.876577。本轮D的RR均值虽在六组中最低，仍高于W0；F的PCC均值虽然本轮最高，仍低于W0。F的trajectory均值与A的global modulation均值则分别低于W0。这些是存在结构、表示和训练合同差异的历史参照，不是单因素归因。

本轮18个checkpoint全部沿用原validation Local RR选点和30/15早停合同，test未用于重选epoch、修改停止规则或追加候选。核心比较限于固定卷积前端、97尺度编码、D64六层主干及统一停止策略下的六组；不能外推到其他前端或所有attention/FiLM实现。

数据、target、Pi、loss、指标、聚合和checkpoint selector均未因本次报告改变。本次只新增结果记录与入口，未开启后续实验或改写历史结论表。
