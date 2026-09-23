# ADV 与 W0 训练过程及 early stopping 评估

日期：2026-09-22。范围：只读已完成的 train/validation history；回顾性分析，不读取test指标、波形或重新执行模型。

结论：历史曲线显示明显的提前停止空间，但短patience存在漏选最佳点的实际反例。本轮统一采用 `min_epochs=30、patience=15、min_delta=0、max_epochs=80`；该规则在当前21条历史曲线上保留了全部原选点，但不保证新attention/融合结构不会出现更晚改善。

## 1. 来源与方法

读取完整80epochs/6400更新的21条历史：ADV四配置三seed共12条，W0原基线3条，以及已完成的W0 gamma030/gamma040各3条作为敏感性补充。每条校验epoch/update连续且全部数值有限，保存原文件路径和SHA；没有改写历史checkpoint或summary。

分析脚本：`scripts/review_adv_fusion_training_history.py`。结果目录：`runs/adv_fusion_factorial_v1/history_review_20260922_r1`，含 `histories.csv`、`replay_per_run.csv`、`replay_summary.csv`、`training_curves.png` 和来源manifest。

回放按原始full-validation Local RR严格改善重置patience；并列不重置，min_delta=0。达到min_epochs且连续patience个epoch未改善时停止，只在已观察的前缀中选择最早最小值。回放始终沿用原80epoch cosine曲线，不把学习率压缩到模拟停止点。考察min_epochs20/30及patience10/15/20/25，属于探索性敏感性检查。

## 2. 训练过程

| 系列/配置 | 三seed最佳epoch | 最佳Local RR均值约 | 第80epoch Local RR均值约 |
|---|---|---:|---:|
| ADV joint | 12 / 7 / 5 | 0.548861 | 0.671128 |
| ADV waveform | 8 / 9 / 15 | 0.554955 | 0.736640 |
| ADV cwt | 22 / 7 / 21 | 1.376742 | 1.652691 |
| ADV joint_scale_mean | 7 / 7 / 8 | 0.547298 | 0.669942 |
| W0 gamma050 | 13 / 15 / 14 | 0.551309 | 0.615623 |
| W0 gamma030 | 13 / 5 / 14 | 0.546693 | 0.592024 |
| W0 gamma040 | 13 / 30 / 14 | 0.542056 | 0.604317 |

所有21条历史在最佳Local RR之后，最终train loss均继续下降，最终Local RR均升高；多数validation loss也随训练后期上升。这与训练后期过拟合或loss/任务指标目标差异相容，不能只凭选点早就认定模型容量或优化器有问题。既有流程重载最佳checkpoint，继续训练并不会直接把第80epoch作为最终评价模型。

完整曲线：[training_curves.png](../../runs/adv_fusion_factorial_v1/history_review_20260922_r1/training_curves.png)。逐run记录：[histories.csv](../../runs/adv_fusion_factorial_v1/history_review_20260922_r1/histories.csv)。

## 3. 早停回放

| min_epochs / patience | ADV：平均停止epoch；漏选数 | W0基线 | W0 gamma补充 |
|---|---|---|---|
| 20 / 10 | 22.50；0/12 | 24.00；0/3 | 23.17；1/6 |
| 20 / 15 | 25.67；0/12 | 29.00；0/3 | 29.83；0/6 |
| 20 / 20 | 30.67；0/12 | 34.00；0/3 | 34.83；0/6 |
| 30 / 15 | 31.08；0/12 | 30.00；0/3 | 32.50；0/6 |
| 30 / 20 | 32.50；0/12 | 34.00；0/3 | 35.67；0/6 |

反例为 gamma040、seed20260812：20/10规则在epoch25停止，只能选epoch15，会漏掉epoch30的最佳点，Local RR增加0.002485 bpm。仅观察ADV和W0基线的15条历史，会漏掉这个风险。

本轮采用的30/15规则在21条历史均保留原最早最佳点，平均停止31.33epochs，合计减少1022/1680个epoch，约60.8%。30/20参考规则平均停止33.62epochs，减少974个epoch，约58.0%。这些仅为epoch数量节省，实际时间还包括数据核对、cache、逐epoch validation、I/O和最终指标，不能直接当作端到端耗时节省。

## 4. 对本轮实验的决定

本轮三种融合方式、位置与交互统一使用30/15/0/80停止策略；停止策略本身不作为额外实验因子。Attention的零输出初始化及后续优化轨迹尚未有完整训练证据，回顾性证据不能消除晚期改善被漏掉的风险。

全部18组开始前统一冻结：监控full-validation Local RR、mode=min、min_delta=0、min_epochs30、patience15、max_epochs80、原80epoch学习率轨迹、最早最佳checkpoint，记录实际停止epoch/原因及实际更新数。结果解释为统一停止策略下的质量和计算取舍；完整矩阵验收按实际停止点回放，使用独立`es30p15`实验身份。

回顾性规则由已见历史启发；上述分析没有逐epoch的五主指标，因此只能说明回放保留了历史Local-RR选点，不能证明所有指标都在早期达到最优。配置与汇总拒绝更改统一停止参数、压缩学习率预算或无依据截断formal训练。
