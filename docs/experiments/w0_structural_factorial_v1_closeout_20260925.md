# W0 三因素结构对照 v1：结项报告

日期：2026-09-25。状态：**工程验收、24 项 formal train/validation、validation 汇总、固定 checkpoint research-test 24 项评价及完整汇总均已完成，本轮阶段关闭。**

本文件是本实验的当前状态与证据入口。训练协议和 research-test 协议记录各阶段当时的合同与状态，且已进入实现锁或 allowlist 文件身份，保留原文用于复核。已完成的工程验收、训练、评价与汇总不重复执行。

## 1. 结项判断

本轮没有找到同时保持 RR、PCC、包络轨迹和全局调制质量的结构简化。保留完整 `PATCH/TM3/REF2` 作为 W0 结构身份；其余组合用于解释结构作用和属性取舍，不替换既定主模型。

1. **前端 A：**在本轮固定合同和所测试的 Conv20 package 下，Conv20 不能替代 Patch。Validation 中 Conv20 的 Whole RR 与轨迹存在局部收益，但 Local RR、全局调制和 PCC 受损；research-test 中 Conv20 的五项等权主效应均不利，且 RR 差距明显。
2. **CWT 后时间块 B：**TM3 的作用依赖前端和 refinement。Validation 的 W0 邻域主要表现为 PCC 收益与其余属性代价；research-test 的 W0 邻域转为 RR/轨迹收益与全局调制、PCC代价并存。
3. **FiLM 后细化 C：**REF2 同样是属性取舍。Research-test 的 W0 邻域中，REF2 改善 Whole/Local RR，并对轨迹和 PCC 有小幅收益，但全局调制误差增加。
4. **交互：**AB、AC、BC 和 ABC 在多个指标上达到预设材料性阈值，B/C 的条件效应也出现反号。三个因素均按预设规则归类为 `context_dependent_continue_research`，不能用单一主效应作普遍因果解释。
5. **结构选择：**完整 W0 在 research-test 上取得最低 Whole RR 和 Local RR；`PATCH/TM0/REF0` 取得最低轨迹误差、最低全局调制误差和最高 PCC。两者形成清楚的 RR—形态/PCC 取舍，不构成唯一跨属性赢家。

Research-test 已参与既有研究开发，证据属性为 **reused research/development evidence**。它用于描述固定 validation-selected checkpoint 的跨 split 表现，不作为新的独立 held-out 确认，也不触发 checkpoint 或模型重选。

## 2. 固定范围与完成情况

- 因子：A=`PATCH/CONV20`，B=`TM3/TM0`，C=`REF2/REF0`；完整 `2×2×2` 矩阵。
- Seeds：`20260811`、`20260812`、`20260813`；24 个 arm×seed formal run 和 24 个固定 checkpoint test evaluation。
- Train/validation：10,141/2,675 窗口，32/7 个 `samp_id`；test：2,310 窗口，8 个 `samp_id`。
- 训练：physical batch 128、accumulation 1、BF16、AdamW；`min_epochs=30`、`patience=15`、`min_delta=0`、`max_epochs=80`，学习率始终按 6,400 updates 计划。
- Selector：完整 validation Local RR 严格最小值，并列取最早；test 使用全部 24 个已选 checkpoint。
- 实际 formal 合计 763 epochs、61,040 optimizer updates；各 run 完成 30–44 epochs，selected epoch 为 2–29。逐 run wall-time 合计约 14.64 h，此数是进程 wall-time 求和，不代表并行调度或端到端成本。
- Validation 汇总 64,200 行指标；research-test 汇总 55,440 行指标。三个 seed 描述训练随机性，不作为三个独立人群；重叠窗口不作为独立样本。

## 3. 五主指标

表中为三个 seed 的 arithmetic mean ± sample SD。前四项越低越好，PCC 越高越好。

### 3.1 Validation

| Arm | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| PATCH/TM3/REF2 | 0.497327 ± 0.018359 | 0.551939 ± 0.012166 | 0.152166 ± 0.000924 | 0.187206 ± 0.006818 | 0.863919 ± 0.001341 |
| PATCH/TM3/REF0 | 0.490598 ± 0.018174 | 0.550302 ± 0.010775 | 0.155391 ± 0.003466 | 0.184808 ± 0.003622 | 0.860114 ± 0.004466 |
| PATCH/TM0/REF2 | 0.479769 ± 0.015471 | 0.550577 ± 0.005512 | 0.150745 ± 0.000810 | 0.183457 ± 0.006723 | 0.860219 ± 0.004342 |
| PATCH/TM0/REF0 | 0.501902 ± 0.011076 | 0.550870 ± 0.015931 | 0.147635 ± 0.003007 | 0.179050 ± 0.003901 | 0.863949 ± 0.001181 |
| CONV20/TM3/REF2 | 0.484075 ± 0.017206 | 0.571927 ± 0.002700 | 0.145145 ± 0.003716 | 0.208071 ± 0.005082 | 0.842676 ± 0.006972 |
| CONV20/TM3/REF0 | 0.450785 ± 0.000568 | 0.563464 ± 0.005667 | 0.144966 ± 0.004049 | 0.212046 ± 0.010467 | 0.842311 ± 0.006830 |
| CONV20/TM0/REF2 | 0.463614 ± 0.020678 | 0.565916 ± 0.003772 | 0.149844 ± 0.004927 | 0.234525 ± 0.020813 | 0.837509 ± 0.005869 |
| CONV20/TM0/REF0 | 0.486831 ± 0.017878 | 0.560067 ± 0.004500 | 0.152177 ± 0.008256 | 0.207121 ± 0.018034 | 0.840013 ± 0.002555 |

### 3.2 Research-test

| Arm | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| PATCH/TM3/REF2 | 0.637269 ± 0.007575 | 0.611730 ± 0.015189 | 0.140746 ± 0.002399 | 0.175462 ± 0.004317 | 0.873767 ± 0.003729 |
| PATCH/TM3/REF0 | 0.696070 ± 0.040995 | 0.651828 ± 0.032328 | 0.141287 ± 0.002041 | 0.171036 ± 0.007693 | 0.872616 ± 0.005492 |
| PATCH/TM0/REF2 | 0.660426 ± 0.042373 | 0.639607 ± 0.022028 | 0.143471 ± 0.003391 | 0.172356 ± 0.011414 | 0.875059 ± 0.004561 |
| PATCH/TM0/REF0 | 0.670981 ± 0.027393 | 0.630742 ± 0.047708 | 0.139557 ± 0.003130 | 0.165216 ± 0.001018 | 0.878331 ± 0.000649 |
| CONV20/TM3/REF2 | 0.779043 ± 0.070827 | 0.760957 ± 0.084455 | 0.139940 ± 0.001568 | 0.206895 ± 0.015497 | 0.858897 ± 0.003937 |
| CONV20/TM3/REF0 | 0.778460 ± 0.046993 | 0.757452 ± 0.060326 | 0.140866 ± 0.000683 | 0.203486 ± 0.014753 | 0.861524 ± 0.002349 |
| CONV20/TM0/REF2 | 0.784783 ± 0.042812 | 0.746402 ± 0.061637 | 0.144856 ± 0.001289 | 0.208451 ± 0.013977 | 0.859911 ± 0.001369 |
| CONV20/TM0/REF0 | 0.804836 ± 0.047523 | 0.785333 ± 0.064912 | 0.144475 ± 0.000489 | 0.189152 ± 0.015542 | 0.859167 ± 0.001599 |

## 4. 简化、资源与尾部误差

Validation 的预设容差检查和 research-test 的同口径复核均没有产生质量保持型简化。最接近明确工程取舍的是 `PATCH/TM0/REF0`：

- 参数由 1,219,850 降至 1,031,690，减少 188,160（15.43%）；
- 协议覆盖的 MACs 由 339,806,080 降至 3,710,080，减少 98.91%；该字段不是完整模型 FLOPs；
- synthetic cached-input benchmark 的 train throughput 提高 10.57%，peak allocated 降低 8.56%；
- research-test 相对完整 W0 的 Whole RR 和 Local RR 分别增加 0.033712 bpm（5.29%）与 0.019011 bpm（3.11%）；轨迹误差降低 0.001189，全局调制误差降低 0.010246，PCC 增加 0.004564。

因此该结构是高效率的属性取舍候选，不是保持 W0 全部质量的替代项。资源测量只覆盖同硬件上的 model-only cached-input 路径，不包含 CWT/cache 生成，不外推为端到端时延。

Research-test Local RR 尾部进一步显示这种取舍不是由单一均值概括：

| Arm | Median | P90 | P95 | >2 bpm | >5 bpm |
|---|---:|---:|---:|---:|---:|
| PATCH/TM3/REF2 | 0.08360 | 1.46115 | 3.12076 | 7.5036% | 3.0159% |
| PATCH/TM0/REF0 | 0.07940 | 1.49073 | 3.11667 | 7.6335% | 3.2179% |

## 5. 分母、受试者与解释边界

- 24 个 test cell 均为 2,310 行、2,310 个唯一 row ID、8 个 `samp_id`；Whole RR、Local RR、PCC、包络分层 target eligibility 均为 2,310。
- `joint_prediction_degenerate` 与 `envelope_spearman_prediction_degenerate` 在全部 test cell 均为 0。
- IBI target eligibility 为 2,310；逐 cell 的 interpretable IBI 窗口为 1,300–1,513，IBI 结果必须连同 coverage 与分母解释。
- 完整 validation/test 逐受试者表、8-subject test macro、Local RR P90/P95 与 `>2/>5 bpm`、辅助 coherence/nDTW 和分层 envelope Spearman 均保存在对应 summary。
- Local RR 是窗口级频谱主峰误差，不等同于逐呼吸相位跟踪；因窗口重叠和 seed 角色，本轮不报告窗口级或 seed-level 人群显著性。
- A 比较的是两个完整前端 package；其差异包含表示、归一化、激活、采样与容量。B 只针对二维 CWT 编码和尺度聚合后的 TM3；C 只针对 FiLM 后 REF2，统一读出的时间卷积保持固定。

## 6. 冻结产物与身份

所有路径相对于主仓库 `/mnt/disk_code/marques/resp_reconstruction`；运行产物原位保留且不进入 Git。

| 内容 | 路径或身份 |
|---|---|
| 主协议 | `docs/experiments/w0_structural_factorial_v1_protocol_20260923.md` |
| Research-test 协议 | `docs/experiments/w0_structural_factorial_v1_research_test_protocol_20260925.md` |
| P2 工程回执 | `docs/experiments/w0_structural_factorial_v1_p2_engineering_receipt_20260924.md` |
| 实现锁 SHA-256 | `32141eab672ea41435055c45cbd7ec96325f8ddef2481a210db2941222c9e9f3` |
| Formal 运行源码提交 | `03957e068d5cac90322ccdbe7a560ffd2754c43d` |
| Formal 产物 | `runs/w0_structural_factorial_v1_es30p15/formal/{arm}/seed_{seed}/{attempt}`，24/24 完成 |
| Validation summary | `runs/w0_structural_factorial_v1_es30p15/summary/summary_32141eab672e_20260925T072509Z_dbcda95df316` |
| Validation manifest SHA-256 | `4ffbcece5b07756bf99342b764e438f5ef67bef16cf76cf76e3f6ceb7a880065` |
| Validation decision SHA-256 | `d829f65eaf464a3bfde9ff9364acd5afb936ecd78395278d335f6648c6c6f1f3` |
| Research-test 运行源码提交 | `68a44daf88b9bd1f1fcad49bf126cbdd954afd8f` |
| Checkpoint allowlist | `runs/w0_structural_factorial_v1_es30p15/research_test/allowlist/allowlist_e3fa7f6d0a8d_20260925T080614Z_e05b0ac352d2` |
| Allowlist SHA-256 | `7f664c538abf68c65e40a222e03cce4434244e8f5fb84664dfd2721d0fd3037b` |
| Research-test evaluation | `runs/w0_structural_factorial_v1_es30p15/research_test/evaluation/{arm}/seed_{seed}/{attempt}`，24/24 完成 |
| Research-test summary | `runs/w0_structural_factorial_v1_es30p15/research_test/summary/summary_7f664c538abf_20260925T094542Z_3bd17f17c3e4` |
| Research-test manifest SHA-256 | `3728bdc121b675d859e348e00c0e4ed84912483ad0f0a31fbd34bbcdba9db54c` |
| Research-test decision SHA-256 | `3d167041ceb736bc863a31027ba7bec863b57741cbdfcc909a57bea0cc410fb1` |

两份 summary 的 manifest 各包含 22 个受管文件。完成后的只读核验覆盖 summary manifest/freeze receipt、24 个 formal 来源、24 个 test evaluation manifest、allowlist、样本行数、指标分母、finite 与冻结 decision。Validation 状态为 24/24 完成；research-test 状态为 completed=24、pending/running/failed=0。

## 7. 关闭决定

本轮结构因子实验执行工作关闭。冻结产物、成功与失败 lifecycle 均按原协议保留，不覆盖、不补写、不使用相同 identity 重跑。后续论文使用应同时披露五主指标、seed SD、尾部误差、受试者分层和证据属性；不得按单项均值把属性取舍改写为全面优越。

若后续继续研究 TM3/REF2 的条件依赖、设计新的轻量前端，或需要新的独立确认，应另立科学问题、协议和输出 identity；本结项不自动开放新训练或新 test 访问。
