from __future__ import annotations

from dataclasses import dataclass


REPORT = "docs/stage_reports/2026-07-08-tho-research-v2-group-meeting-stage-report.md"
METRIC_SCHEMA = "docs/experiments/metric_schema.md"
EVIDENCE_LEDGER = "docs/experiments/current_evidence_ledger.md"
G_SERIES_ROOT = "runs/test_eval_g_series_20260709_local_rr_canonical"
HARMONIC_ROOT = "runs/bcg_second_harmonic_20260710"


@dataclass(frozen=True)
class SlideSpec:
    key: str
    title: str
    section: str
    takeaway: str
    bullets: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    contains_numeric_evidence: bool = False
    placeholder: str | None = None


MAIN_SLIDES = (
    SlideSpec(
        "title",
        "THO research v2 阶段进展",
        "研究问题",
        "已确定当前 STFT 输入基准，下一阶段转向选择性修正",
        sources=(REPORT,),
        placeholder="【待补：汇报人、汇报日期】",
    ),
    SlideSpec(
        "three_questions",
        "本次汇报回答三个问题",
        "研究问题",
        "从“STFT 是否有用”推进到“如何选择性使用”",
        bullets=(
            "STFT 辅助信息是否为纯时序模型提供了额外信息？",
            "哪一种 STFT 表示适合作为本阶段基准？",
            "为什么下一阶段不继续无差别扩大输入？",
        ),
        sources=(REPORT,),
    ),
    SlideSpec(
        "motivation",
        "BCG 中包含呼吸信息，但不能直接替代胸带呼吸信号",
        "研究问题",
        "目标是从非接触 BCG 恢复 THO-like 呼吸信息",
        bullets=(
            "BCG 同时包含呼吸运动、心冲击、体动、姿态和接触变化。",
            "THO 也可能存在局部质量下降、相位延迟和方向不一致。",
            "应用价值在于从床垫 BCG 中获得更连续的呼吸节律与状态信息。",
        ),
        sources=(REPORT,),
    ),
    SlideSpec(
        "task_and_difficulties",
        "任务不是波形照抄，而是恢复可信的呼吸节律与形态",
        "研究问题",
        "BCG → THO-like 波形 / 呼吸状态",
        bullets=("心冲击与高频干扰", "体动和接触条件变化", "方向、相位与局部形态不一致", "低质量和节律模糊窗口"),
        sources=(REPORT,),
        placeholder="【待补：典型 BCG / THO 对照波形；建议来源：当前 research v2 诊断图】",
    ),
    SlideSpec(
        "dataset_split",
        "受试者隔离的独立测试集用于阶段决策",
        "实验可信性",
        "32 / 7 / 8 名受试者对应 10,141 / 2,675 / 2,310 个可用窗口",
        bullets=("每个窗口 180 秒，采样率 100 Hz。", "训练、验证、测试受试者不重叠。", "同一受试者内窗口重叠且窗口数不均衡。"),
        sources=(REPORT,),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "preprocessing",
        "输入和目标使用分段归一化与 soft-z 压缩",
        "实验可信性",
        "模型学习宽频 BCG soft-z 到 THO soft-z 的映射",
        bullets=("BCG 宽频带通：0.03–20 Hz。", "状态对齐后按段估计 robust center / scale。", "soft-z 压缩极端体动尾部，避免尺度被少数异常值主导。", "推理阶段不使用 THO 目标构造时频输入。"),
        sources=(REPORT,),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "model_flow",
        "时频分支只改变辅助表示，不改变时序主干和输出目标",
        "实验可信性",
        "同一 BCG 同时形成时序 token 与可选时频 token，并在 pre-mixer 融合",
        sources=(REPORT,),
    ),
    SlideSpec(
        "training_supervision",
        "训练约束覆盖节律、低频形态、相对强弱与方向一致性",
        "实验可信性",
        "当前损失不是单一逐点 MSE",
        bullets=("训练集更新参数，验证集用于早停和方向一致性筛选。", "每个方案保留 3 次独立训练。", "测试集只在当前比较方案确定后复评。", "本轮未启用目标侧 STFT 损失。"),
        sources=(REPORT,),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "four_way_control",
        "四方案对照只改变时频辅助信息",
        "实验可信性",
        "数据、划分、时序主干、输出目标、训练设置和评价方式保持一致",
        bullets=("G0_time_only：纯时序参照。", "G0_f0_native_stft_pre_mixer：上一版 STFT 参照。", "G3_C_wide_8p0：宽频高时间分辨率 STFT。", "G3_C_bandenergy：5 条重叠频带能量序列。"),
        sources=(REPORT,),
    ),
    SlideSpec(
        "metric_framework",
        "最终排序依据是呼吸任务指标，而非验证损失",
        "实验可信性",
        "整窗节律、周期数量、时延校正形态、相对强弱和局部节律共同读表",
        bullets=("稳健带通 RR 误差：越低越好。", "周期计数 bpm 误差：越低越好。", "4 秒时延校正相关：越高越好。", "相对包络相关：越高越好。", "局部 RR MAE / 相关：低 / 高更好。"),
        sources=(REPORT, METRIC_SCHEMA),
    ),
    SlideSpec(
        "overall_test_results",
        "独立测试集结果支持宽频 STFT 作为当前基准",
        "基准方案选择",
        "wide 的稳健 RR 和时延校正形态更好，bandenergy 的计数和局部 RR 略优",
        sources=(REPORT, f"{G_SERIES_ROOT}/g_series_local_rr_canonical_label_summary.csv"),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "stft_adds_information",
        "STFT 辅助信息补充了纯时序模型的节律判断",
        "基准方案选择",
        "问题已从“是否需要 STFT”转向“哪种表示适合作为基准”",
        bullets=("上一版 STFT 已改善部分节律和形态指标。", "高时间分辨率 wide 进一步降低稳健 RR 与计数误差。", "STFT 的作用是辅助时序主干，而不是替代它。"),
        sources=(REPORT, f"{G_SERIES_ROOT}/g_series_local_rr_canonical_label_summary.csv"),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "wide_as_baseline",
        "宽频、高时间分辨率 STFT 更适合作为本阶段基准",
        "基准方案选择",
        "G3_C_wide_8p0 在核心稳健 RR、形态与综合稳定性上更均衡",
        bullets=("20 秒窗、2.5 秒步长，提高时频辅助信息的时间分辨率。", "稳健带通 RR 误差最低。", "4 秒时延校正相关最高。", "bandenergy 保留为选择性修正候选，而非统一替代。"),
        sources=(REPORT,),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "tail_errors",
        "wide 更稳地压低 RR 长尾，bandenergy 更擅长计数与局部 RR 长尾",
        "基准方案选择",
        "均值之外的 p95 与灾难性错误比例揭示不同角色",
        sources=(REPORT, f"{G_SERIES_ROOT}/stratified_analysis_20260709/tail_summary.csv"),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "stratification_scope",
        "分层分析用于理解收益来源，不能直接作为推理时 gate",
        "收益来源",
        "hard / easy、count-error 和 low-spectrum 均含目标侧评价信息",
        bullets=("同一分层内横向比较同一批窗口。", "分层不重新推理、不重评 checkpoint。", "下一阶段必须构造 BCG-only、无泄漏的选择判据。"),
        sources=(REPORT,),
    ),
    SlideSpec(
        "hard_easy_low_spectrum",
        "wide 的收益主要来自 hard 和 low-spectrum 窗口",
        "收益来源",
        "easy 窗口可能被时频方案误伤，选择性修正必须保护已正确窗口",
        sources=(REPORT, f"{G_SERIES_ROOT}/stratified_analysis_20260709/strata_summary.csv"),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "count_error",
        "bandenergy 在已有周期计数错误的窗口中呈现修正价值",
        "收益来源",
        "clean-count 窗口需要保护，不能全窗口开放更强修正",
        sources=(REPORT, f"{G_SERIES_ROOT}/stratified_analysis_20260709/strata_summary.csv"),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "rr_roles",
        "不同 RR 区间呈现 wide 与 bandenergy 的角色分化",
        "收益来源",
        "wide 在慢 RR 和 14–18 bpm 更稳，bandenergy 在部分中快 RR 区间有局部优势",
        sources=(REPORT, f"{G_SERIES_ROOT}/stratified_analysis_20260709/rr_bin_summary.csv"),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "harmonic_motivation",
        "二次谐波分层把困难窗口连接到可观察的输入机制",
        "二次谐波困难窗口",
        "BCG 双峰或倍频结构可能使主峰落在 THO 基频的二倍附近",
        bullets=("比单纯按模型误差定义 hard window 更接近输入机制。", "当前冻结规则仍使用 THO 参考，只能用于离线分析。", "不参与训练、checkpoint 选择或推理 gate。"),
        sources=(REPORT,),
    ),
    SlideSpec(
        "harmonic_coverage",
        "二次谐波显著窗口覆盖测试集 19.57%，但集中在少数受试者",
        "二次谐波困难窗口",
        "452 个阳性窗口来自 6 名受试者；strong 层 214 个窗口",
        sources=(REPORT, f"{HARMONIC_ROOT}/test_v2/coverage_summary.csv"),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "harmonic_model_results",
        "四类网络大多能纠正倍频，但任务表现仍有差异",
        "二次谐波困难窗口",
        "纠正率均约 95% 以上；bandenergy 的节律误差更低，wide 的形态更好",
        sources=(REPORT, f"{HARMONIC_ROOT}/model_metrics/model_stratified_metrics_summary.csv", f"{HARMONIC_ROOT}/corrections/model_harmonic_correction_summary.csv"),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "balanced_cases",
        "平衡案例用于展示共有能力、共同失败和模型分歧",
        "二次谐波困难窗口",
        "案例选择预先覆盖四模型均纠正、均失败、模型分歧和阈值附近",
        bullets=("row 640：四模型均纠正。", "row 873：四模型均未纠正。", "row 1353：模型间分歧。", "row 3584：阈值附近。"),
        sources=(REPORT, f"{HARMONIC_ROOT}/figures/model_case_manifest.csv"),
        contains_numeric_evidence=True,
    ),
    SlideSpec(
        "stage_decision",
        "阶段决策：wide 为当前基准，bandenergy 用于条件修正探索",
        "阶段决策",
        "不继续无差别扩大输入，也不补评不服务当前问题的历史分支",
        bullets=("基准：G3_C_wide_8p0。", "候选：G3_C_bandenergy。", "停止：非重点历史分支测试补评。", "转向：难恢复窗口的选择性修正。"),
        sources=(REPORT, EVIDENCE_LEDGER),
    ),
    SlideSpec(
        "risks",
        "当前证据支持阶段判断，但不能外推为跨受试者稳定结论",
        "阶段决策",
        "结论边界来自 soft-z 口径、重叠窗口、目标侧分层和受试者集中性",
        bullets=("只适用于 THO research v2 soft-z。", "统计单位首先是重叠窗口，而非独立受试者。", "二次谐波阳性集中在少数受试者。", "现有分层不能直接作为无泄漏 gate。", "波形回归可能不是最终输出空间。"),
        sources=(REPORT, METRIC_SCHEMA, EVIDENCE_LEDGER),
    ),
    SlideSpec(
        "next_stage_takeaways",
        "下一阶段：建立 BCG-only 判据并验证条件修正",
        "阶段决策",
        "STFT 值得保留，wide 是当前基准，bandenergy 用于条件修正探索",
        bullets=("建立 BCG-only 难度 / 不确定性判据。", "验证 wide 默认 + 条件 bandenergy 修正。", "做受试者级复核或重采样。", "探索 respiratory state、局部 RR、事件或不确定性辅助目标。"),
        sources=(REPORT, EVIDENCE_LEDGER),
    ),
)


BACKUP_SLIDES = (
    SlideSpec("balanced_cases", "备份：二次谐波平衡案例", "备份", "四类预定义案例共同约束定性解释", sources=(REPORT, f"{HARMONIC_ROOT}/figures/"), contains_numeric_evidence=True),
    SlideSpec("model_details", "备份：模型结构与融合参数", "备份", "四方案共享时序主干，差异仅在时频表示", sources=(REPORT,)),
    SlideSpec("training_details", "备份：训练配置与损失权重", "备份", "3 次独立训练，早停与方向一致性筛选保持一致", sources=(REPORT,), contains_numeric_evidence=True),
    SlideSpec("metric_formulas", "备份：主要评价指标公式", "备份", "指标分别回答节律、计数、形态、包络和局部 RR 问题", sources=(REPORT, METRIC_SCHEMA)),
    SlideSpec("full_stratified_results", "备份：完整分层结果", "备份", "分层结果仅用于回顾性解释收益来源", sources=(REPORT, f"{G_SERIES_ROOT}/stratified_analysis_20260709/"), contains_numeric_evidence=True),
    SlideSpec("subject_results", "备份：8 名测试受试者结果", "备份", "wide 的稳健 RR 收益并非由单一受试者拉动", sources=(REPORT,), contains_numeric_evidence=True),
    SlideSpec("harmonic_subgroups", "备份：二次谐波子层结果", "备份", "strong、peak-doubling 与 harmonic-prominent 需分开解释", sources=(REPORT, HARMONIC_ROOT), contains_numeric_evidence=True),
    SlideSpec("data_provenance", "备份：数据 provenance", "备份", "以导出包保存的配置快照和 metadata 为准", sources=(REPORT,), contains_numeric_evidence=True),
    SlideSpec("reproduction_commands", "备份：复现实验命令与证据路径", "备份", "命令与 manifest 用于追溯，不在汇报制作阶段重新运行", sources=(REPORT,)),
    SlideSpec("qa_notes", "备份：现场问答提示", "备份", "主动说明独立性、soft-z、排序依据和泄漏边界", sources=(REPORT, METRIC_SCHEMA, EVIDENCE_LEDGER)),
)

