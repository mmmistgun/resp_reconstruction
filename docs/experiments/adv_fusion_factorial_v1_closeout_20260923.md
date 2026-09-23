# ADV 融合因子实验收尾索引

日期：2026-09-23。状态：**本轮六组工程验收、18项formal训练与validation、18项固定checkpoint的research-test及两份完整矩阵汇总均已完成，阶段关闭。**

本文件为本轮当前状态入口。执行协议与实现锁中的“尚未执行”等描述属于对应实现阶段的历史状态；实际完成情况以本索引、两份结果报告和原始完成回执为准。协议及执行源码已纳入哈希身份，保持原文以便复核。已完成的cache、训练、评价和汇总不重复执行。

## 1. 结果与协议入口

| 内容 | 入口 |
|---|---|
| Validation完整五指标、位置/方式效应与交互 | [validation结果报告](adv_fusion_factorial_v1_results_20260923.md) |
| Research-test完整结果及跨split比较 | [research-test结果报告](adv_fusion_factorial_v1_research_test_results_20260923.md) |
| 网络定义、配对初始化和es30p15训练合同 | [训练协议](adv_fusion_factorial_v1_protocol_20260922.md) |
| 固定test候选、访问顺序及指标合同 | [research-test协议](adv_fusion_factorial_v1_research_test_protocol_20260923.md) |
| GPU、smoke、formal与汇总命令记录 | [执行命令](adv_fusion_factorial_v1_commands_20260922.md) |
| 21条历史曲线的停止规则回放 | [early stopping评估](adv_fusion_factorial_v1_earlystop_review_20260922.md) |

## 2. 固定实验范围

- 分支：`codex/adv-fusion-factorial-v1`。
- 工作树：`/mnt/disk_code/marques/resp_reconstruction/.worktrees/adv_fusion_factorial_v1`。
- 开发基点：ADV-v1收尾提交`f1a4236`。实际运行时的Git状态、逐文件哈希与源码快照保存在各产物目录；收尾提交保存实现，实际运行来源以这些记录为准。
- 三方式为拼接投影、FiLM、逐时刻双视图attention；两位置为六层Mamba主干前与主干后、统一读出前。A/B为拼接pre/post，C/D为FiLM pre/post，E/F为attention pre/post。
- 固定Conv前端、97尺度CWT编码、两路32→64适配、D64六层双向Mamba-2及读出；同seed共享模块初始化配对。Seeds为20260811/20260812/20260813。
- Train/validation分别10141/2675窗口；test为2310窗口、8个samp_id。任务均为完整180 s重建；数据、target、Pi、loss、指标与窗口direct mean遵循各专项合同。
- 停止规则为`min_epochs=30、patience=15、min_delta=0、max_epochs=80`，学习率始终按80epochs/6400更新计划计算；实际总训练550epochs、44000更新，18项均在30–33epochs合法停止。实际epoch数比最大预算减少61.81%，不等同于端到端耗时节省率。
- Checkpoint按已执行的完整validation Local RR最小值选择，并列取最早；test沿用全部18个选点。

## 3. 主要结果与解释边界

1. Validation中B的Whole/Local RR均值最低，F的trajectory误差均值最低，A的global modulation误差均值最低、PCC均值最高。
2. Research-test中D的Whole/Local RR均值最低（0.71861/0.70308 bpm），F的trajectory误差均值最低、PCC均值最高（0.13830/0.86134），A的global modulation误差均值最低（0.16799）。这些是属性取舍，不构成唯一总分赢家。
3. 拼接后移的global modulation代价、FiLM后移的PCC下降、attention后移的trajectory改善，在两个split内均为三个seed同向。
4. Whole RR的FiLM−拼接交互均值由validation的+0.020999变为test的−0.029975，test中seed方向混合；因此不能把validation中的全部交互方向作为稳定结论。
5. Test中F相对D的Whole/Local RR在三个seed均更差；F的PCC和trajectory均值优势不等于全面优势。本轮D的RR、F的PCC均值仍未达到历史W0水平，跨系列差异不作单因素归因。

完整三seed均值/SD、配对方向和辅助指标分母见两份结果报告及其CSV来源。三个seed描述训练随机性，不作为独立人群；三组两两交互只有两个线性独立。Test为 **reused research/development evidence**，不作为未触及held-out的独立确认。结论限于本轮固定前端、表示和停止策略；attention效应代表整个融合模块，权重不直接解释为生理贡献或可靠性。

## 4. 完成产物

以下路径相对于当前工作树；`{arm}`为A–F，`{seed}`为上述三个训练seed。

| 阶段 | 位置 | 完成情况 |
|---|---|---|
| GPU合成验收 | `runs/adv_fusion_factorial_v1/gpu_{arm}_es30p15_20260922_r1` | 6/6 |
| 有限smoke及cache | `runs/adv_fusion_factorial_v1/{arm}_smoke_es30p15_20260922_r1`、`runs/adv_fusion_factorial_v1/cache_smoke_es30p15_20260922_r1` | 6/6及cache完成 |
| Formal训练 | `runs/adv_fusion_factorial_v1/{arm}_seed{seed}_formal_es30p15_20260922_r1` | 18/18 |
| Validation完整汇总 | `runs/adv_fusion_factorial_v1/summary_full_es30p15_20260922_r1` | 完成 |
| Test input-only cache | `runs/adv_fusion_factorial_v1_research_test/cache_es30p15_20260923_r1` | 完成 |
| Test评价 | `runs/adv_fusion_factorial_v1_research_test/cache_es30p15_20260923_r1_evaluations/{arm}_seed{seed}/attempt_r1` | 18/18 |
| Test完整汇总 | `runs/adv_fusion_factorial_v1_research_test/summary_full_es30p15_20260923_r1` | 完成 |
| 历史训练曲线回放 | `runs/adv_fusion_factorial_v1/history_review_20260922_r1` | 完成 |

Formal训练复用的cache原位保留在`/mnt/disk_code/marques/resp_reconstruction/.worktrees/aligned_dual_view_v1/runs/aligned_dual_view_v1/cache_formal_20260921_r1`。各结果报告列出相应cache身份与核对范围。

每份summary manifest列出完整来源。Validation的per-seed/across-seed/contrasts-per-seed/contrasts-across-seed分别18/30/240/80行；test分别18/78/240/80行，额外保留辅助指标与分母。

`runs/`、checkpoint、cache、日志和生成图原位保留，不进入Git。恢复实验记录需要代码提交及当前工作树、原ADV工作树中的相关运行产物；单独克隆Git不包含这些数据。成功与失败生命周期均按原协议保留，不向已完成目录补写。

## 5. 来源锁与验证

| 身份 | SHA-256 |
|---|---|
| es30p15训练执行代码 | `f9381a7d1bf48a736f4a34e549d2bb440c9eb057e2f2e789a4e5364737870bce` |
| Research-test执行代码 | `d723d279a4ec38d89f61c2f4ccfaeb9e1ae947dd8016d40a19ec9be171b3cb32` |
| Validation summary manifest | `ab47bb367b74c476605d70d8e2ab7b17940fbc905dbd4d1ae610be6a72c76e11` |
| Test summary manifest | `892d6735f3d4dba7d9a79c187fe0c7a6d3f608230b7817009299997e1aede640` |
| Test candidate lock文件 | `2c80d570168b0de907a348bd1a1bdd8b476203623fc83ba64462377dd0ea8add` |

- 当前训练实现与CPU验证：[es30p15 implementation lock](adv_fusion_factorial_v1_implementation_lock_es30p15_20260922.json)，23项通过，50.39 s。
- 当前test实现与CPU验证：[research-test implementation lock](adv_fusion_factorial_v1_research_test_implementation_lock_20260923.json)，13项通过，44.95 s。其中包含训练集合中的一项因子差分测试，两个数字不直接相加为独立测试数。
- 早期工程阶段记录：[初始implementation lock](adv_fusion_factorial_v1_implementation_lock_20260922.json)，保留为es30p15修订前的实现历史，不作为本轮实际训练身份。
- 固定checkpoint路径、哈希及selected/stopped epoch：[candidate lock](../../configs/adv_fusion_research_test_v1/candidate_lock_20260923.json)。

GPU原生前后向、有限真实数据smoke、formal和test均由用户完成；完成后的只读核对覆盖样本身份、配对初始化、选点/停止回放、学习率计划、checkpoint文件哈希、源码快照、访问记录和存储指标到汇总的数值一致性。详细范围及未重放项目见两份结果报告。执行协议中的命令保留用于复现记录，已完成阶段维持关闭状态。
