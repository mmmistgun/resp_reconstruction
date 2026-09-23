# ADV 融合方式 × 位置：validation 结果

日期：2026-09-23。协议：`adv-fusion-factorial-v1-es30p15-train-val-20260922`。

**状态：六组 GPU 合成验收、六组有限 smoke、18 项 formal 训练和完整 validation 汇总已完成；本次只读核对通过。** 当前结果与来源以本文件及原始完成回执为准；实现协议中的未执行描述保留为实现阶段历史。已完成阶段不自行重跑。

主要判断：融合方式和位置呈现属性相关的取舍。后融合拼接的 Whole RR 均值最低，但全局包络调制误差明显增加；后融合 attention 的包络轨迹误差均值最低，也伴随全局调制代价。FiLM 与 attention 均未带来五主指标的稳定全面改善。部分配对交互在三个 seed 方向一致，但不据此宣称人群统计显著性。

## 1. 来源与只读核对

- 执行契约：[实验协议](adv_fusion_factorial_v1_protocol_20260922.md)。
- 汇总：[summary_full_es30p15_20260922_r1](../../runs/adv_fusion_factorial_v1/summary_full_es30p15_20260922_r1/manifest.json)。完成时间：2026-09-23 01:09:31，Asia/Shanghai。
- Summary manifest SHA-256：`ab47bb367b74c476605d70d8e2ab7b17940fbc905dbd4d1ae610be6a72c76e11`。
- 执行代码 SHA-256：`f9381a7d1bf48a736f4a34e549d2bb440c9eb057e2f2e789a4e5364737870bce`，与当前工作树及 es30p15 实现锁一致。
- Comparison ID：`2c27ebf37f35947a5a78b7070cd94dae3da5b6243bf24076212da31234168f75`。
- 复用 ADV formal cache ID：`04a5db6b544815780bc7af21a50af14bffd61dfdd0da943f08a2d9430da34869`；其 manifest SHA：`ecce7f6c67bd4bc585e98e20f8c441ad19c328cde6c73c83e6e84ba6b6c1f424`。
- Train / validation sample 内容身份：`a1e2bd035a7cd1515ce5011093bbacde2eac7d91600a5f71bbff3d9c47c1f543` / `646d36b96009b060dd31fbd580239b8d7c217f2d69b7e8126c9b1a75bbaaade3`。

核对覆盖18项唯一 arm/seed、完成回执、源 manifest、config、history、sample provenance、初始化和参数报告、selected/final checkpoint 文件哈希；18项及汇总的源码快照逐文件哈希均通过。相同 seed 的六组共享模块，以及同方式 pre/post 的融合模块，初始化身份一致。

每项包含相同的10141 train和2675 validation窗口，row ID唯一且有序，输入/target/mask内容身份一致。五主指标每项均有2675个有限值，共核对48150行validation指标；所有`joint_prediction_degenerate_fraction`为0。逐窗口指标到单项summary、18行per-seed、30行across-seed、240行contrasts-per-seed和80行contrasts-across-seed数值一致。

停止回执逐history回放通过；每epoch首末学习率与6400更新计划一致。六组GPU报告的文件哈希、输出shape、loss和模块梯度记录通过；六组smoke有完整成功回执。当前产物目录未发现`failed.json`。

本次只读取已保存产物，未读取原始波形或CWT数组、加载checkpoint运行模型、访问本轮test或再次生成汇总。Checkpoint核对采用文件哈希；cache payload未重新扫描。本次新增结果文档及入口，不改变训练代码、协议哈希、数据、指标或选点规则。

## 2. 完整五主指标

任务为完整180 s重建。每seed按窗口sample direct mean，再对三个训练seed等权计算mean ± sample SD。SD描述训练随机性，不是人群置信区间；validation窗口之间存在依赖。

| 组别 | Whole RR ↓，bpm | Local RR ↓，bpm | 包络轨迹 ↓ | 全局包络调制 ↓ | Signed PCC ↑ |
|---|---:|---:|---:|---:|---:|
| A 前融合拼接 | 0.48201 ± 0.00666 | 0.54864 ± 0.00745 | 0.15619 ± 0.00946 | 0.19146 ± 0.01218 | 0.84166 ± 0.01071 |
| B 后融合拼接 | 0.45316 ± 0.01322 | 0.54742 ± 0.00823 | 0.15122 ± 0.00122 | 0.23604 ± 0.03619 | 0.83887 ± 0.00218 |
| C 前融合 FiLM | 0.48041 ± 0.01770 | 0.54903 ± 0.01013 | 0.15002 ± 0.00625 | 0.21470 ± 0.01303 | 0.84162 ± 0.00940 |
| D 后融合 FiLM | 0.47256 ± 0.03127 | 0.54871 ± 0.00338 | 0.14982 ± 0.00704 | 0.21716 ± 0.01808 | 0.83871 ± 0.00877 |
| E 前融合 attention | 0.47496 ± 0.01101 | 0.54933 ± 0.00920 | 0.15147 ± 0.00790 | 0.21555 ± 0.01517 | 0.84024 ± 0.00882 |
| F 后融合 attention | 0.47530 ± 0.02868 | 0.55206 ± 0.01051 | 0.14827 ± 0.00625 | 0.25162 ± 0.05502 | 0.84013 ± 0.00695 |

Local RR六组均值范围为0.54742–0.55206 bpm；PCC为0.83871–0.84166。仅看这些均值的细小排序不足以证明稳定优势。B的Whole RR与Local RR均值最低，F的trajectory最低，A的global modulation最低、PCC最高；这些是各属性描述，不构成总分或唯一模型选择。

完整18行逐seed数据：[per_seed.csv](../../runs/adv_fusion_factorial_v1/summary_full_es30p15_20260922_r1/per_seed.csv)。均值及SD：[across_seed.csv](../../runs/adv_fusion_factorial_v1/summary_full_es30p15_20260922_r1/across_seed.csv)。

## 3. 同位置的融合方式差异

下表为候选减参照的三seed均值差，括号为改善seed数。四项误差负值为改善，PCC正值为改善。全部逐seed差及差值SD见[配对明细](../../runs/adv_fusion_factorial_v1/summary_full_es30p15_20260922_r1/contrasts_per_seed.csv)和[配对汇总](../../runs/adv_fusion_factorial_v1/summary_full_es30p15_20260922_r1/contrasts_across_seed.csv)。

| 对比 | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| FiLM−拼接，pre：C−A | −0.001599（2/3） | +0.000394（1/3） | −0.006171（3/3） | +0.023233（1/3） | −0.000047（1/3） |
| FiLM−拼接，post：D−B | +0.019401（1/3） | +0.001295（1/3） | −0.001399（2/3） | −0.018881（2/3） | −0.000157（2/3） |
| attention−拼接，pre：E−A | −0.007049（3/3） | +0.000690（1/3） | −0.004722（2/3） | +0.024088（0/3） | −0.001428（1/3） |
| attention−拼接，post：F−B | +0.022144（0/3） | +0.004643（1/3） | −0.002953（2/3） | +0.015580（1/3） | +0.001260（2/3） |
| attention−FiLM，pre：E−C | −0.005450（2/3） | +0.000296（2/3） | +0.001449（1/3） | +0.000855（1/3） | −0.001381（0/3） |
| attention−FiLM，post：F−D | +0.002743（1/3） | +0.003348（2/3） | −0.001554（3/3） | +0.034461（1/3） | +0.001417（2/3） |

前融合FiLM的trajectory相对拼接在三个seed均改善，但PCC均值近乎相同、global modulation均值更差。前融合attention的Whole RR相对拼接三个seed均改善，同时global modulation三个seed均恶化；放在后融合时，其Whole RR相对拼接则三个seed均恶化。这说明融合方式的排序依赖位置和所考察属性。

## 4. 同方式的位置效应

定义为post−pre，格式同上。

| 对比 | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| 拼接 B−A | −0.028849（3/3） | −0.001223（2/3） | −0.004968（2/3） | +0.044578（0/3） | −0.002798（2/3） |
| FiLM D−C | −0.007850（1/3） | −0.000322（1/3） | −0.000196（2/3） | +0.002463（1/3） | −0.002908（0/3） |
| attention F−E | +0.000344（2/3） | +0.002731（2/3） | −0.003199（3/3） | +0.036069（0/3） | −0.000111（2/3） |

后融合拼接的Whole RR均值降低约5.99%，三个seed一致；global modulation均值增加约23.28%，也在三个seed一致。其PCC有2/3 seed改善，但seed20260811的下降使均值变差，因此均值方向不能替代逐seed描述。

FiLM后移的PCC三个seed均下降，其他属性的位置效应较小或方向混合。Attention后移的trajectory三个seed均改善，global modulation三个seed均恶化；Whole/Local RR均值没有改善。

等权平均三种方式的位置效应：Whole RR −0.012118、Local RR +0.000395、trajectory −0.002788、global modulation +0.027703、PCC −0.001939。Whole RR与global modulation在三个seed均分别降低/升高；该边际平均不能替代上面的方式特异结果。

## 5. 方式 × 位置交互

对方式m2/m1，交互为 `(m2_post−m2_pre)−(m1_post−m1_pre)`。表中为mean ± sample SD；符号表示位置效应的差异，不直接表示一个模型总体更好。

| 方式对 | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| FiLM−拼接 | +0.020999 ± 0.014523 | +0.000901 ± 0.009303 | +0.004773 ± 0.010145 | −0.042115 ± 0.029317 | −0.000110 ± 0.007174 |
| attention−拼接 | +0.029193 ± 0.023804 | +0.003953 ± 0.013851 | +0.001769 ± 0.009670 | −0.008509 ± 0.066037 | +0.002688 ± 0.006854 |
| attention−FiLM | +0.008194 ± 0.012360 | +0.003052 ± 0.007597 | −0.003003 ± 0.002674 | +0.033606 ± 0.061565 | +0.002798 ± 0.001842 |

较一致的描述性信号：

- Whole RR：FiLM−拼接、attention−拼接的交互均为三个seed正值，说明后移所带来的Whole RR变化在拼接方式下更有利。
- Global modulation：FiLM−拼接的交互三个seed均为负值，说明FiLM后移的调制误差增量小于拼接；这不表示FiLM后移本身改善了global modulation。
- attention−FiLM：trajectory交互三个seed为负、PCC交互三个seed为正，体现两者对位置的响应不同。

三组两两交互只有两个线性独立，不能作为三个独立证据。当前只有三个训练seed，结论限于描述性配对交互；不作人群显著性检验或生理机制归因。

## 6. Early stopping 与训练预算

每格为selected epoch / stopped epoch，seed依次为20260811、20260812、20260813。

| 组别 | seed20260811 | seed20260812 | seed20260813 |
|---|---:|---:|---:|
| A | 8 / 30 | 17 / 32 | 18 / 33 |
| B | 12 / 30 | 7 / 30 | 4 / 30 |
| C | 9 / 30 | 17 / 32 | 4 / 30 |
| D | 9 / 30 | 16 / 31 | 4 / 30 |
| E | 9 / 30 | 17 / 32 | 4 / 30 |
| F | 9 / 30 | 3 / 30 | 5 / 30 |

18项全部按patience_exhausted结束，均满足min_epochs30、patience15、min_delta0、max_epochs80。实际总计550epochs、44000更新，平均30.56epochs；相对最大1440epochs/115200更新，减少61.81%的epoch/更新数。这不是端到端耗时节省率。

停止时的Local RR均高于各自selected值，最终评价均重载最佳checkpoint。由于后续epoch没有执行，不能从本轮产物证明继续到80epochs不会再改善；本轮比较是统一停止策略下的质量和计算取舍，不能描述为相同实际计算预算。

## 7. GPU 工程数据

设备：NVIDIA GeForce RTX 4070 Ti SUPER；batch32、bf16、实际合成BCG的CWT、原生Mamba与任务loss。每组5步，前2步预热，后3步中位耗时；计时包含有限性/梯度检查和同步，排除离线CWT与I/O，不作为正式吞吐benchmark。

| 组别 | 总参数 | 峰值allocated，GiB | 峰值reserved，GiB | step中位，ms |
|---|---:|---:|---:|---:|
| A | 480593 | 1.838 | 2.195 | 157.32 |
| B | 480593 | 1.831 | 2.195 | 138.57 |
| C | 480657 | 1.859 | 2.182 | 139.13 |
| D | 480657 | 1.831 | 2.191 | 142.01 |
| E | 488785 | 1.909 | 2.242 | 159.86 |
| F | 488785 | 1.837 | 2.260 | 163.26 |

来源为`runs/adv_fusion_factorial_v1/gpu_{arm}_es30p15_20260922_r1/report.json`。Attention融合解析MAC为44,697,600/窗口，拼接与FiLM为14,745,600；算子统计口径见原协议。Attention结果解释为整个融合模块的效果，不能全部归因于注意力权重，也不能将权重作为生理贡献或可靠性。

## 8. 历史参照与结论边界

W0历史validation均值为Whole RR0.499298、Local RR0.551309、trajectory0.152729、global modulation0.191051、PCC0.865300。相较之下，本轮六组的PCC均值仍约0.839–0.842，未恢复到W0水平；A的global modulation接近W0，B/F则更差。已有ADV joint的PCC0.842308和global modulation0.215527可作为另一历史参照。

这些历史模型的前端、宽度、表示或训练合同不同；不能将跨系列差异归因于单一融合因子。尤其本轮后置FiLM（D）并未表现出恢复W0 PCC的效果，现有证据不支持仅更换为后置FiLM即可解释或消除历史差距。

本轮核心结论限于固定卷积前端、97尺度投影、D64六层主干及30/15早停合同：融合位置与方式呈现属性相关的响应差异，但没有一致的全面优选方案。前端替代方案的价值、不同尺度编码及其他训练预算均未由本轮检验。本轮仅完成validation；未开启test或新增实验。
