# CWT-APOR v2机制分析完成记录

日期：2026-10-02。状态：三seed、18条件机制评价及full / exclude670 / subject670汇总全部完成，本阶段关闭。后台任务`j-ezqgwd`正常退出0。本文为机制阶段的最终状态入口，验收合同见[同batch修订r2](cwt_apor_v2_mechanisms_r2_20261002.md)。

## 完成与验收

正式会话：`runs/cwt_apor_v2/session_20261001T042244Z_e5e53ad16731`。

最终机制版本：`research_test/mechanisms_r2/revision_485f7f748f79`。

| seed | 条件数 | 每条件窗口数 | 同次运行普通前向/FULL逐点最大差 | 原机制结果复用 |
|---|---:|---:|---:|---|
| 20260811 | 18 | 2310 | 0 | 是，已完成产物 |
| 20260812 | 18 | 2310 | 0 | 是，原跨batch门槛失败产物，经独立验收认证 |
| 20260813 | 18 | 2310 | 0 | 否，首次完成 |

三个seed共6930次逐窗波形检查全部通过；54个条件×seed实例均完整。FULL/NAT和FULL/FIXED五项主指标逐窗相同。完成回执、summary文件和三个新验收阶段的manifest、metrics引用身份已只读核对通过。

三个视图的每条件分母分别为full=2310窗/8受试者、exclude670=2231窗/7受试者、subject670=79窗/1受试者。完整视图与预定敏感性视图同时报告。

已完成60-cell常规research-test结果保持原身份。历史失败attempt、原pipeline失败记录及跨batch/跨运行诊断均保留；本轮成功状态由新的r2验收与汇总链提供。前两个seed的18条件指标和案例文件没有重算或改写。

## 主要结果

以下数值为各seed内逐窗均值再跨三个seed取均值；误差越低越好。机制比较的参考是同流程batch8的FULL/NAT。

| 条件 | full Local RR MAE（bpm） | exclude670 Local RR MAE（bpm） | full包络轨迹MAE | exclude670包络轨迹MAE |
|---|---:|---:|---:|---:|
| FULL/NAT | 0.649895 | 0.445329 | 0.142223 | 0.137689 |
| H窗口均值/NAT | 0.680513 | 0.445613 | 0.143148 | 0.138520 |
| H共同SHIFT1/NAT | 0.685025 | 0.441116 | 0.145453 | 0.140456 |
| H共同SHIFT2/NAT | 0.681808 | 0.442192 | 0.145645 | 0.140573 |
| H共同SHIFT3/NAT | 0.688375 | 0.442719 | 0.145655 | 0.140694 |
| H2窗口均值/NAT | 0.662457 | 0.445498 | 0.143425 | 0.138863 |
| H2共同SHIFT1/NAT | 0.665791 | 0.444388 | 0.144338 | 0.139867 |
| H2共同SHIFT2/NAT | 0.662140 | 0.445228 | 0.144458 | 0.139947 |
| H2共同SHIFT3/NAT | 0.665460 | 0.444223 | 0.144494 | 0.140009 |

H=(0.8,8] Hz，H2=(2,8] Hz。三个SHIFT是预定的三个共同时间平移；范围描述其三项结果，不把它们当作额外独立seed。

### 高频时间对应关系与包络

H共同错位使full视图包络轨迹MAE增加2.27%–2.41%，exclude670视图增加2.01%–2.18%；三个SHIFT在三个seed中均增加包络误差。H2对应为full增加1.49%–1.60%，exclude670增加1.58%–1.68%。该现象同时出现在完整视图和预定敏感性视图。

固定GN统计后，H错位的包络误差增幅仍为full的2.29%–2.43%、exclude670的2.03%–2.20%；H2亦保留同方向变化。因此，固定统计条件下仍有高频时间错位的功能影响，不能仅用GN重新计算统计解释。这里支持的是本模型对条件时间对应关系的功能响应，尚未单独分解整体强度与跨尺度相对结构的贡献。

### RR的受试者差异

full视图中，H窗口均值替换使Local RR MAE增加4.71%，H共同错位增加4.91%–5.92%；H2共同错位增加1.88%–2.45%。

在exclude670视图中，H窗口均值变化约+0.064%，H共同错位反而降低Local RR MAE约0.59%–0.95%；H2共同错位降低约0.02%–0.25%。相应地，subject670视图的FULL/NAT Local RR MAE为6.426940 bpm，H窗口均值后为7.314208 bpm，H共同错位后为7.448668–7.625825 bpm。

因此，完整视图的RR退化主要由受试者670贡献，不支持把RR收益概括为所有受试者共有。相比之下，包络误差对高频错位的响应在两个主视图和三个seed中方向一致。以上为描述性机制证据，不将seed间一致性当作独立受试者层面的显著性检验。

## 最终证据入口

以下路径相对于正式会话：

- pipeline：`research_test/mechanisms_r2/revision_485f7f748f79/pipeline/attempt_20261002T082854Z_7b0c6ad722e7`。
- seed20260811验收：`research_test/mechanisms_r2/revision_485f7f748f79/seed_20260811/attempt_20261002T082909Z_fb3e586a2af1`。
- seed20260812验收：`research_test/mechanisms_r2/revision_485f7f748f79/seed_20260812/attempt_20261002T082921Z_c2263c67cafa`。
- seed20260813结果与验收：`research_test/mechanisms_r2/revision_485f7f748f79/seed_20260813/attempt_20261002T083041Z_5972f1dfc3dc`。
- 最终三视图汇总：`research_test/mechanisms_r2/revision_485f7f748f79/summary/attempt_20261002T084725Z_430651990a29`，包含逐seed、逐受试者、受试者宏平均、配对差值和尾部误差表；`sources.json`绑定三个独立验收manifest。
- 60-cell常规research-test汇总：`research_test/summary/attempt_20261002T054624Z_e22f113cfb30`。

每个seed的`result.json`定位实际指标与案例来源，`same_batch_waveform_check.csv`给出逐窗硬检查，`cross_batch_diagnostic.csv`记录与常规test的数值差异；复用seed的`historical_replay_diagnostic.csv`另保留跨运行复现差异。
