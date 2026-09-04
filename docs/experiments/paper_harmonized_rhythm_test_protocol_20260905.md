# 独立测试集节律指标统一补充汇总

协议 ID：`paper-harmonized-rhythm-test-v1-20260905`；日期：2026-09-05。

状态：实现与定向 CPU 验证阶段；用户已授权从干净实现 commit 执行一次正式只读汇总。

## 1. 问题、授权与证据边界

本附件独立整理 center30/60/90 全部 input/model arms 与历史 180→180 已审计独立测试集方法。
用户本次明确授权描述性跨输出长度节律比较；这只扩展本附件的报告范围，不回写三个原任务的独立 panel、
结论或关闭状态，不构造跨任务总分、模型排名，不据此重选 checkpoint、模型、窗口、seed 或阈值。

外侧输入上下文收益与输出本身持续时间是两个不同问题。等长输入输出轨迹同时改变输入、监督、输出和评价时长，
不能解释为控制其他因素后的纯输出长度因果效应。重叠的 2310 个父窗口不解释为 2310 个独立受试者。

本附件只读取既有冻结 test CSV、summary 和 provenance JSON/YAML；不读取 checkpoint 内容、dataset/index、
原始 signal/target array，不执行训练、GPU、cache、benchmark 或 checkpoint inference。数据、split、标签、Pi、
loss、指标算法、selector 均保持原合同。旧结果、receipt、manifest 和旧协议保持原样。

## 2. 冻结来源与完整范围

| 来源 | manifest SHA-256 | 范围 |
|---|---|---|
| `context_length_research_test_summary/` | `d69d47a80e514a9556000296938dff642d2642656c61ecff82d57c1a1baeeb66` | center30 8 arms、center60 6 arms，42 checkpoints |
| `center90_context_research_test_summary/` | `2da65f83dc93d6a65b9c9e2c5024c67aa00a0520f2cf36338e35c77589d9f50b` | center90 6 arms，18 checkpoints |
| `p0_comparison_audit/` | `1e05c3de9a75c08ee016af379e075adc1b71b4d1162f7bbf81d716b4942d2542` | 历史 10 个论文主表方法与 RTM 五候选 |

上述目录均位于 `runs/paper_evidence_v1/`。历史路径取自
`configs/paper_evidence_v1/p0_comparison_sources.json`，SHA-256=
`c1fcf7d55a25763013a9604d544a81c6306b3752e41cbfba2447a8dbb46f5b82`。

预检确认：历史主表 10 方法的逐样本 CSV 均含 `ibi_medae_sec / ibi_coverage / ibi_interpretable / ibi_target_eligible`，
stored summary 均含 IBI mean、有效数、coverage mean 和 interpretable fraction。历史 summary 的
`ibi_interpretable_n` 实际表示 fraction 分母（target-eligible 数），新表区分 `ibi_target_eligible_n` 与真正的
`ibi_interpretable_n`。RTM 五候选逐样本 CSV 也有四项 IBI 字段，但原 seed summary 只有五主指标；
其 IBI 从既有逐样本 CSV 汇总，明确标记来源。五候选保留 `pending_user_confirmation_audit_only` 状态，
不改动论文已冻结主表或替用户确认 RTM 结论锁。D4 只有 validation，不进入本 test 汇总。

合计 35 arms、99 个学习 checkpoint 记录与 2 个 deterministic 记录；101×2310=233310 条逐样本指标。
学习方法保留 seeds `20260811/20260812/20260813`，确定性方法保留 `deterministic-no-seed`、seed_count=0、SD 未定义。

每份 CSV 必须经冻结摘要 hash 链或 P0 source artifact audit 的 observed hash 核验。P0 当时首次登记 observed hash，
本附件将该已冻结值作为 expected hash，不能将其追溯表述为 P0 前已预注册的 source hash。
每项必须为同一 2310 个唯一 test row IDs，row SHA-256=
`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`；按 row 排序检查 split/input_set/samp_id/coupling_state_id。
本次不重读 train/validation 或数据索引；subject/session 隔离沿用原冻结验收，不能声称重新做了原始数据隔离审计。
中心60项评价配置逐项由冻结manifest核验，RTM配置由冻结test manifest→checkpoint_inputs→config hash核验。
其他历史配置与W3定位配置所用JSON只读登记本次observed hash并核对参数，不宣称其具有同样的先验hash链。
所有来源均核对 IBI 三参数、100 Hz与0.3 s lag，历史同时核对Local RR的60s/15s；当前共享metric源码hash也写入receipt。

## 3. 指标与统计合同

1. Native Whole RR：各模型完整输出上的 dominant-frequency RR 绝对误差。center30/60/90 的原字段分别为
   `center30_rr_mae_bpm / center_rr_mae_bpm / center90_rr_mae_bpm`；历史为 `whole_rr_abs_error_bpm`。
   去均值、symmetric Hann、N 点 rFFT、0.05–0.70 Hz 内取峰，理论 RR bin 间距 `60/T` 为
   `2 / 1 / 0.666667 / 0.333333 bpm`。该间距不是 MAE 下界，也不等于完整频谱分辨能力。
2. IBI：均调用 `resp_train.metrics.task._ibi_metrics`，100 Hz、最小峰距142 samples、匹配容差0.5 s、
   coverage≥0.8；先按 signed PCC 在 ±0.3 s 对齐，使用去除两端各30 samples的支撑。
   峰 prominence=`max(0.2×std(ddof=0), 0.08×(P95−P5))`；有序匹配先最大化匹配数，再最小化总时间差。
   仅相邻 target/prediction 峰对形成周期误差。每个样本先取周期绝对误差 median，再对 interpretable 样本取直接均值。
3. Coverage=有效相邻匹配周期数/(target峰数−1)，在 target-eligible 样本上直接均值；interpretable fraction
   分母也是 target-eligible 样本数。MedAE 有限集必须恰等于 interpretable 集合，coverage 有限集恰等于 eligible 集合；
   Inf、缺字段、错误布尔值和不符合合同的 NaN 均失败，不静默过滤。
4. 单 checkpoint 使用 sample-direct mean；学习模型完整三 seed arithmetic mean 与 sample SD (`ddof=1`)，
   不按有效样本数加权或把三 seed 样本池化。缺失 seed 不参与部分均值；保留有效样本数与有限 seed 记录数。
5. 检查等长轨迹相邻任务的逐 row eligible/interpretable 交集、并集、改变数和 Jaccard。
   coverage/fraction 接近只能排除明显比例漂移，不能自动排除可解释子集成员或周期组成变化。

Native RR 不必随输出时长平滑：FFT 网格、target时间支撑、Pi_T 投影、训练监督范围共同变化；中心任务由 native RR
选择 checkpoint，历史由60s Local RR选择，W-reduced/W3/W0结构身份也不同。这里只记录可解释因素，不能估计各自因果贡献。
IBI 使用同一事件算法，是当前更直接的跨输出节律桥梁，但不同支撑上的事件、边界、投影和可解释集合仍有差异。

## 4. Local RR 的可比范围

历史180→180固定60s窗口、15s step，共9窗，先样本内平均；历史方法内部可比较。
Center60 native RR为完整中心60s单窗，虽与历史 Local RR 观测长度相同，但时间位置、Pi作用域、窗口聚合不同，
不直接等同历史9窗均值。Center30不足60s；center90冻结metrics没有统一60s Local RR。本阶段不新增该列。

如果后续需要共同FFT观测尺度，可独立设计 `common_center30_rr_mae_bpm`：所有输出取父窗口同一中心30s，
统一Pi_30、同一target时间位置、同一FFT和每样本一次贡献。现有evaluator没有预测波形产物，该指标需重新推理。
本阶段不需要该追加指标；实施/运行前必须先报告 checkpoint矩阵、test访问范围、两GPU任务数及成本、全部context arms
或仅等长轨迹、历史180模型名单，待用户确认后另立附件。

## 5. 产物、执行与失败规则

固定新目录 `runs/paper_evidence_v1/harmonized_rhythm_test_summary/`，存在即拒绝，失败写出目录也保留且不可复用。

- `native_whole_rr_ibi_all_arms.csv`：全部35 arms，保留准确模型名、时长、seed/selector/结论锁身份、mean/SD和有效数。
- `equal_input_output_rhythm_ladder.csv`：9行，C201四点，W-reduced-center30/60/90三点与历史W3/W0分列。
- `per_checkpoint_rhythm_summary.csv`：101行，原字段映射、来源与逐seed有效数。
- `equal_io_support_overlap.csv`：21个相邻任务/seed集合比较。
- `rhythm_comparability.json`：指标、统计、Local RR范围、描述性边界和支持集合诊断。
- `summary_receipt.json`：干净commit、命令、依赖、输入hash、counts、access和输出hash。
- `artifact_manifest.json`：最终文件size/hash（manifest不自哈希，其hash写回本协议）。

授权验证与正式命令：

```bash
./.venv/bin/python -m pytest tests/test_paper_harmonized_rhythm_test_summary.py -q
./.venv/bin/python scripts/summarize_paper_harmonized_rhythm_test_v1.py
```

先完成定向CPU测试并中文提交实现；工作树干净后只执行一次正式只读汇总，再把产物hash、精确结果、验证记录写回本协议并提交。
定向测试包括纯合成聚合/失败路径与冻结CSV的只读集成验证，不访问波形或执行推理。原有关闭的summarizer/evaluator不调用。

## 6. 正式执行结果

待从干净实现 commit 一次性生成后登记。
