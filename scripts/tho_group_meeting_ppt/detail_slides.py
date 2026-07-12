"""研究讨论型正文的显式页组与原生 PowerPoint builders。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from .discussion_content import DISCUSSION_UNITS, DiscussionUnit
from .theme import (
    BODY_BLACK,
    SCNU_BLUE,
    SECONDARY_ORANGE,
    WHITE,
    add_discussion_box,
    add_evidence_boundary,
    add_evidence_gap,
    add_method_panel,
    add_multi_panel,
    add_source_note,
    add_text_box,
    add_three_line_table,
    new_content_slide,
)


@dataclass(frozen=True)
class SlideSpec:
    """人工审查后的页定义；绝不按字符数自动分页。"""

    key: str
    section: str
    title: str
    unit_keys: tuple[str, ...]
    builder: str
    asset_key: str | None = None


def _s(key: str, section: str, title: str, *unit_keys: str, builder: str = "method", asset_key: str | None = None) -> SlideSpec:
    return SlideSpec(key, section, title, tuple(unit_keys), builder, asset_key)


# 48 个语义单元被 43 个显式页面定义恰好消费一次；五组互补问题人工合并。
SLIDE_GROUPS = (
    _s("task_observation", "任务与信号直觉", "从混合观测到 THO-like 呼吸表示", "task_mapping", "bcg_signal_mixture", builder="signal", asset_key="signal_overview"),
    _s("reference_boundary", "任务与信号直觉", "参考信号也有边界：时延、方向与局部质量", "tho_reference_limits", builder="discussion"),
    _s("window_timescale", "任务与信号直觉", "三分钟样本如何兼顾稳定估计与局部变化", "window_180s_tradeoff", builder="method"),
    _s("sample_formation", "数据来源与样本形成", "样本如何从整晚数据形成", "data_provenance", "index_to_window", builder="method"),
    _s("window_filter", "数据来源与样本形成", "进入模型前：waveform 窗口筛选", "waveform_window_filter", builder="method"),
    _s("split_boundary", "数据来源与样本形成", "受试者隔离定义数据泄漏边界", "subject_split_boundary", builder="discussion"),
    _s("wideband_input", "输入与目标预处理", "为什么保留 0.03–20 Hz 宽频 BCG", "bcg_wideband_input", builder="signal"),
    _s("robust_softz", "输入与目标预处理", "从 robust-z 到 soft-z：本窗未触发压缩", "bcg_segment_robust_z", "bcg_soft_z", builder="signal", asset_key="softz_mapping"),
    _s("target_mask", "输入与目标预处理", "目标、整窗筛选与 RR mask 是三条链路", "target_soft_z_and_mask", builder="method"),
    _s("patch_contract", "模型输入与计算图", "PatchMixer 张量契约：256 / 128 / 140", "model_tensor_contract", "patchmixer_token_shape", builder="model", asset_key="token_geometry"),
    _s("stft_shapes", "模型输入与计算图", "PatchMixer 与 STFT：F0 / wide / bandenergy shape", "stft_resolution", builder="model", asset_key="stft_branch_shapes"),
    _s("fusion", "模型输入与计算图", "pre-mixer 加法融合如何进入计算图", "pre_mixer_injection", builder="model"),
    _s("bandenergy", "模型输入与计算图", "五条重叠频带的表示能力与 tradeoff", "bandenergy_encoder", builder="signal", asset_key="bandenergy_response"),
    _s("dataset_roles", "训练与损失", "训练、验证与独立测试各自决定什么", "train_val_test_roles", builder="discussion"),
    _s("loss_system", "训练与损失", "完整 WeakSyncLoss：任务代理与权重", "composite_loss_goal", "loss_component_weights", builder="formula"),
    _s("signed_loss", "训练与损失", "L_signed_corr 与方向调度", "polarity_schedule", builder="formula", asset_key="loss_schedule"),
    _s("checkpoint", "训练与损失", "checkpoint 选择：方向 gate、val-loss top-k 与任务复评", "checkpoint_selection", builder="discussion"),
    _s("robust_rr", "指标计算与失效场景", "稳健 RR 算法：谱引导、峰间距与双峰护栏", "metric_robust_rr", builder="formula", asset_key="metric_robust_rr"),
    _s("cycle", "指标计算与失效场景", "cycle：双向过零如何变成周期计数", "metric_breath_count", builder="formula", asset_key="metric_cycle_count"),
    _s("lag4", "指标计算与失效场景", "lag4：在 ±4 秒内分离形态与时延", "metric_lag_corr", builder="formula", asset_key="metric_lag_corr"),
    _s("relative_envelope", "指标计算与失效场景", "relative envelope：比较局部增强与减弱", "metric_relative_envelope", builder="formula", asset_key="metric_relative_envelope"),
    _s("local_rr", "指标计算与失效场景", "local RR：正式 40 秒窗 / 10 秒步长", "metric_local_rr", builder="formula", asset_key="metric_local_rr"),
    _s("controls", "对照实验设计", "四个对照只改变时频辅助表示", "controlled_variables", builder="result"),
    _s("time_only", "对照实验设计", "time-only substrate 的计算边界", "time_only_baseline", builder="model"),
    _s("f0_wide", "对照实验设计", "F0 与 wide：频率分辨力和时间更新的取舍", "f0_vs_wide", builder="result", asset_key="stft_resolution_comparison"),
    _s("wide_band", "对照实验设计", "完整谱图还是低维频带摘要", "wide_vs_bandenergy", builder="result"),
    _s("overall", "整体结果与稳定性", "整体结果：主护栏与局部指标不指向同一方案", "overall_metric_values", builder="result"),
    _s("paired", "整体结果与稳定性", "paired delta：同 seed 隔离方案差异", "paired_delta", builder="result", asset_key="overall_delta"),
    _s("seed_subject", "整体结果与稳定性", "seed × subject：方向稳定不等于总体稳定", "seed_stability", builder="result", asset_key="seed_subject_stability"),
    _s("decision_rule", "整体结果与稳定性", "结论顺序：主护栏、形态、局部诊断", "result_decision_rule", builder="discussion"),
    _s("subject", "分层分析与完整案例", "受试者分层：总体均值由谁贡献", "subject_stratification", builder="result"),
    _s("quality", "分层分析与完整案例", "质量与难度分层：收益发生在哪里", "quality_difficulty_strata", builder="result", asset_key="strata_tradeoffs"),
    _s("rr_bins", "分层分析与完整案例", "RR 分层：慢、常规与快呼吸的角色差异", "rr_bin_strata", builder="result"),
    _s("cases", "分层分析与完整案例", "完整案例复核：波形、频谱与指标同页读", "complete_case_review", builder="case"),
    _s("harmonic_definition", "BCG 二次谐波问题", "二次谐波：BCG 主峰为什么落在 2×THO", "harmonic_definition", builder="formula"),
    _s("harmonic_threshold", "BCG 二次谐波问题", "二次谐波判据：阈值如何冻结", "harmonic_thresholds", builder="formula"),
    _s("harmonic_types", "BCG 二次谐波问题", "三类谐波标签不能混为一个结论", "harmonic_subtypes", builder="result"),
    _s("harmonic_correction", "BCG 二次谐波问题", "纠偏判据：回到基频且压低相对二倍频", "harmonic_correction", builder="result"),
    _s("harmonic_cases", "BCG 二次谐波问题", "边界案例约束二次谐波结论外推", "harmonic_boundary_cases", builder="discussion"),
    _s("output_space", "研究议题与下一步", "输出空间：继续波形回归还是任务表示", "decision_output_space", builder="discussion"),
    _s("bcg_gate", "研究议题与下一步", "选择性修正需要无泄漏 BCG-only gate", "decision_bcg_only_gate", builder="discussion"),
    _s("stat_unit", "研究议题与下一步", "统计单位从重叠窗提升到受试者与时间块", "decision_statistical_unit", builder="discussion"),
    _s("minimal_next", "研究议题与下一步", "下一轮最小判别性实验", "decision_minimal_next_experiment", builder="discussion"),
)


UNIT_BY_KEY = {unit.key: unit for unit in DISCUSSION_UNITS}


ASSET_FILES = {
    key: f"{key}.png" for key in (
        "signal_overview", "softz_mapping", "token_geometry", "stft_branch_shapes",
        "bandenergy_response", "loss_schedule", "metric_robust_rr", "metric_cycle_count",
        "metric_lag_corr", "metric_relative_envelope", "metric_local_rr", "overall_delta",
        "seed_subject_stability", "strata_tradeoffs", "stft_resolution_comparison",
    )
}


def resolve_discussion_assets(repo_root: Path) -> dict[str, Path]:
    root = repo_root / "docs/stage_reports/20260708/generated_assets/discussion"
    return {key: root / filename for key, filename in ASSET_FILES.items() if (root / filename).is_file()}


def _joined(units: tuple[DiscussionUnit, ...], field: str, *, limit: int = 4) -> str:
    values = [value for unit in units for value in getattr(unit, field)]
    return "\n".join(f"• {value.replace('更优', '更合适')}" for value in values[:limit])


def _picture(slide, path: Path, *, x=Inches(0.65), y=Inches(2.00), width=Inches(6.20), height=Inches(3.50)):
    with Image.open(path) as image:
        ratio = image.width / image.height
    target_ratio = width / height
    if ratio > target_ratio:
        draw_width, draw_height = width, int(width / ratio)
    else:
        draw_height, draw_width = height, int(height * ratio)
    pic = slide.shapes.add_picture(
        str(path), int(x + (width - draw_width) / 2), int(y + (height - draw_height) / 2),
        draw_width, draw_height,
    )
    pic.name = f"真实证据图：{path.stem}"
    return pic


def _base(prs: Presentation, spec: SlideSpec, page: int, units: tuple[DiscussionUnit, ...]):
    slide = new_content_slide(prs, spec.title, page_number=page)
    question = " / ".join(unit.question for unit in units)
    add_text_box(
        slide, f"研究问题｜{question}", x=Inches(0.76), y=Inches(1.23),
        width=Inches(11.78), height=Inches(0.56), font_size=18, bold=True,
        name="正文研究问题", align=PP_ALIGN.CENTER,
    )
    add_source_note(slide, tuple(dict.fromkeys(source for unit in units for source in unit.sources)))
    return slide


def _native_flow(slide, steps: tuple[str, ...], *, y=2.08):
    selected = steps[:4]
    width = 2.72 if len(selected) == 4 else 3.45
    for index, text in enumerate(selected):
        x = 0.70 + index * (width + 0.38)
        node = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(width), Inches(1.25))
        node.name = "流程节点"
        node.fill.solid(); node.fill.fore_color.rgb = WHITE
        node.line.color.rgb = SCNU_BLUE
        add_text_box(slide, text, x=Inches(x + 0.10), y=Inches(y + 0.08), width=Inches(width - 0.20), height=Inches(1.05), font_size=18, name="流程节点文字", align=PP_ALIGN.CENTER)
        if index:
            line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x - 0.34), Inches(y + 0.62), Inches(x - 0.04), Inches(y + 0.62))
            line.name = "流程连接线"; line.line.color.rgb = SECONDARY_ORANGE; line.line.end_arrowhead = True


def _generic_native(slide, units: tuple[DiscussionUnit, ...], spec: SlideSpec):
    steps = tuple(step for unit in units for step in unit.method_steps)
    _native_flow(slide, steps)
    panels = []
    parameters = _joined(units, "parameters", limit=3)
    rationale = _joined(units, "rationale", limit=3)
    evidence = _joined(units, "evidence", limit=3)
    limits = _joined(units, "limits", limit=3)
    if parameters:
        panels.append(("参数 / shape", parameters))
    if rationale:
        panels.append(("为什么", rationale))
    if evidence:
        panels.append(("证据", evidence))
    if limits:
        panels.append(("限制", limits))
    if len(panels) >= 2:
        add_multi_panel(slide, panels[:3], x=Inches(0.72), y=Inches(3.62), width=Inches(11.82), height=Inches(1.62))
    elif panels:
        add_method_panel(slide, *panels[0], x=Inches(0.72), y=Inches(3.62), width=Inches(11.82), height=Inches(1.62))
    prompt = next((unit.discussion_prompt for unit in units if unit.discussion_prompt), None)
    if prompt:
        add_discussion_box(slide, prompt, x=Inches(1.18), y=Inches(5.47), width=Inches(10.90), height=Inches(1.20))


def _asset_native(slide, units: tuple[DiscussionUnit, ...], path: Path):
    _picture(slide, path)
    params = _joined(units, "parameters", limit=3) or _joined(units, "method_steps", limit=3)
    add_method_panel(slide, "步骤与参数", params, x=Inches(7.10), y=Inches(2.00), width=Inches(5.15), height=Inches(1.75))
    evidence = _joined(units, "evidence", limit=2)
    if evidence:
        add_evidence_boundary(slide, evidence, x=Inches(7.10), y=Inches(3.95), width=Inches(5.15), height=Inches(1.25))
    else:
        add_evidence_boundary(slide, _joined(units, "limits", limit=2), x=Inches(7.10), y=Inches(3.95), width=Inches(5.15), height=Inches(1.25))
    prompt = next((unit.discussion_prompt for unit in units if unit.discussion_prompt), "本页证据是否足以支持当前设计？")
    add_discussion_box(slide, prompt, x=Inches(0.90), y=Inches(5.48), width=Inches(11.45), height=Inches(1.18))


def _formula_text(spec: SlideSpec) -> str:
    formulas = {
        "loss_system": "L = L_env + 0.2L_spec + 0.1L_smooth + 0.2L_high + 0.03L_rel_env + L_signed",
        "signed_loss": "L_signed_corr: 0.6 → 0.2；L_signed_cos: 0.1 → 0（epoch 1–6）",
        "robust_rr": "x → bandpass → Welch 主峰 → find_peaks → median(Δt_peak) → 60 / Δt",
        "cycle": "cycle_bpm_error = |N_cycle(ŷ) − N_cycle(y)| / 3 min",
        "lag4": "corr_lag4 = max corr(ŷ(t+τ), y(t)), |τ| ≤ 4 s",
        "relative_envelope": "R(x) = RMS₂ₛ(x) / smooth₂₀ₛ(RMS₂ₛ(x))",
        "local_rr": "RR_local: 40 秒窗 / 10 秒步长；不要与 STFT 的 20 秒分析窗混淆",
        "harmonic_definition": "f_BCG,peak ≈ 2 × f_THO,fundamental",
        "harmonic_threshold": "ratio₂f/f + peak-distance + valid-spectrum → frozen label",
    }
    return formulas.get(spec.key, "可编辑公式：按本页步骤计算")


def _formula_native(slide, units: tuple[DiscussionUnit, ...], spec: SlideSpec):
    add_text_box(slide, _formula_text(spec), x=Inches(0.82), y=Inches(2.05), width=Inches(11.65), height=Inches(0.85), font_size=23, bold=True, name="公式", align=PP_ALIGN.CENTER)
    panels = (
        ("具体步骤", _joined(units, "method_steps", limit=3)),
        ("参数", _joined(units, "parameters", limit=3) or "• 参数由当前正式口径固定"),
        ("限制", _joined(units, "limits", limit=2)),
    )
    add_multi_panel(slide, panels, x=Inches(0.72), y=Inches(3.20), width=Inches(11.82), height=Inches(1.82))
    prompt = next((unit.discussion_prompt for unit in units if unit.discussion_prompt), "本页算法边界是否充分？")
    add_discussion_box(slide, prompt, x=Inches(1.18), y=Inches(5.43), width=Inches(10.90), height=Inches(1.22))


def _result_native(slide, units: tuple[DiscussionUnit, ...], spec: SlideSpec):
    if spec.key == "controls":
        add_three_line_table(slide, ("变量", "固定/改变", "审查点"), (
            ("数据与 split", "固定", "相同受试者边界"), ("时序主干与输出", "固定", "PatchMixer / THO soft-z"),
            ("训练与评价", "固定", "相同损失与指标"), ("时频辅助表示", "唯一改变", "time-only / F0 / wide / bandenergy"),
        ), x=Inches(0.75), y=Inches(2.05), width=Inches(11.80), height=Inches(3.15), font_size=16)
    elif spec.key == "wide_band":
        add_three_line_table(slide, ("表示", "输入 shape", "保留信息", "tradeoff"), (
            ("F0 STFT", "B×239×37", "较细频率网格", "5 秒更新"),
            ("wide STFT", "B×160×73", "完整 0.05–8 Hz 谱图", "频率分辨力下降"),
            ("bandenergy", "B×5×73", "重叠频带强弱", "丢失带内峰位置"),
        ), x=Inches(0.70), y=Inches(2.05), width=Inches(11.90), height=Inches(3.15), font_size=16)
        add_discussion_box(slide, units[0].discussion_prompt or "选择哪种表示？", x=Inches(1.15), y=Inches(5.48), width=Inches(10.95), height=Inches(1.16))
    elif spec.key == "overall":
        add_three_line_table(slide, ("方案", "稳健 RR MAE", "lag4 corr", "周期数绝对误差", "计数 bpm 误差", "local RR MAE"), (
            ("wide STFT", "0.712186 bpm", "0.870774", "—", "—", "—"),
            ("bandenergy", "—", "—", "2.054401", "0.684800 bpm", "0.773299 bpm"),
        ), x=Inches(0.55), y=Inches(2.10), width=Inches(12.20), height=Inches(2.65), font_size=14)
        add_evidence_boundary(slide, "周期数绝对误差与 bpm 误差必须区分；数值仅按已核验 DiscussionUnit 口径解读。", x=Inches(1.20), y=Inches(5.00), width=Inches(10.90), height=Inches(1.20))
    else:
        _generic_native(slide, units, spec)


def _case_slides(prs: Presentation, spec: SlideSpec, page: int, repo_root: Path):
    rows = ((640, "四模型均纠正"), (873, "四模型均失败"), (1353, "模型间分歧"), (3584, "阈值边界"))
    root = repo_root / "docs/stage_reports/20260708/generated_assets/discussion"
    slides = []
    units = tuple(UNIT_BY_KEY[key] for key in spec.unit_keys)
    for offset, (row_id, label) in enumerate(rows):
        slide = _base(prs, spec, page + offset, units)
        title = next(shape for shape in slide.shapes if shape.name == "页面标题")
        title.text_frame.paragraphs[0].runs[0].text = f"{spec.title}｜row {row_id}：{label}"
        path = root / f"case_row_{row_id}.png"
        if path.is_file():
            _picture(slide, path, x=Inches(0.62), y=Inches(2.00), width=Inches(7.10), height=Inches(4.00))
        add_evidence_boundary(slide, f"row {row_id} 仅代表一种案例类型；不能替代 seed×subject 与总体分层统计。", x=Inches(7.98), y=Inches(2.05), width=Inches(4.35), height=Inches(1.40))
        add_method_panel(slide, "完整复核", "输入 BCG → THO 目标 → 四模型预测 → 频谱主峰 → 单窗指标", x=Inches(7.98), y=Inches(3.72), width=Inches(4.35), height=Inches(1.55))
        add_discussion_box(slide, units[0].discussion_prompt or "如何判定该案例？", x=Inches(7.98), y=Inches(5.52), width=Inches(4.35), height=Inches(1.05))
        slides.append(slide)
    return slides


def build_section_slide(prs: Presentation, section: str, number: int, page: int):
    slide = new_content_slide(prs, f"{number:02d}｜{section}", page_number=page)
    add_text_box(slide, "本章沿“问题 → 步骤 → 参数/shape → 证据 → 边界 → 决策”展开。", x=Inches(1.15), y=Inches(2.55), width=Inches(11.05), height=Inches(1.05), font_size=25, bold=True, name="章节导航", align=PP_ALIGN.CENTER)
    add_text_box(slide, "章节导航", x=Inches(4.65), y=Inches(4.35), width=Inches(4.05), height=Inches(0.70), font_size=20, color=SCNU_BLUE, bold=True, name="章节标签", align=PP_ALIGN.CENTER)
    return slide


def build_discussion_slide(prs: Presentation, spec: SlideSpec, page: int, *, repo_root: Path, assets: Mapping[str, Path]):
    if spec.builder == "case":
        return _case_slides(prs, spec, page, repo_root)
    units = tuple(UNIT_BY_KEY[key] for key in spec.unit_keys)
    slide = _base(prs, spec, page, units)
    path = assets.get(spec.asset_key or "")
    if path and path.is_file():
        _asset_native(slide, units, path)
        if spec.builder == "formula":
            add_text_box(slide, _formula_text(spec), x=Inches(0.84), y=Inches(1.80), width=Inches(11.60), height=Inches(0.30), font_size=18, bold=True, name="公式", align=PP_ALIGN.CENTER)
    elif spec.builder == "formula":
        _formula_native(slide, units, spec)
    elif spec.builder == "result":
        _result_native(slide, units, spec)
    else:
        _generic_native(slide, units, spec)
    if spec.key == "robust_softz":
        add_text_box(slide, "本窗未触发压缩：0 samples changed；不能据此推断其他窗口。", x=Inches(7.10), y=Inches(5.15), width=Inches(5.15), height=Inches(0.38), font_size=18, name="证据说明", align=PP_ALIGN.CENTER)
    if spec.key == "local_rr":
        add_evidence_boundary(slide, "20 秒局部 RR 窗属于旧/其他流程，不是当前正式 local RR；当前正式口径为 40 秒窗 / 10 秒步长。", x=Inches(7.10), y=Inches(3.95), width=Inches(5.15), height=Inches(1.25))
    if spec.key == "sample_formation":
        add_evidence_gap(slide, missing_fields="dirty=false 的数据制作 commit 与重导出 manifest", suggested_entrypoint="在数据制作工作区重新导出并冻结 provenance", x=Inches(0.95), y=Inches(5.47), width=Inches(11.35), height=Inches(1.18))
    return [slide]
