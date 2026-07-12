from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from .charts import ChartData
from .content import BACKUP_SLIDES, SlideSpec
from .theme import (
    BODY_BLACK,
    NOTE_GRAY,
    SCNU_BLUE,
    SECONDARY_ORANGE,
    add_body_text,
    add_card,
    add_placeholder,
    add_source_note,
    add_takeaway,
    add_text_box,
    add_three_line_table,
    new_content_slide,
)


CASE_DIR = Path("runs/bcg_second_harmonic_20260710/figures")


def _add_contained_picture(slide, path: Path, *, x, y, width, height, name: str):
    with Image.open(path) as image:
        image_width, image_height = image.size
    scale = min(width / image_width, height / image_height)
    draw_width = int(image_width * scale)
    draw_height = int(image_height * scale)
    picture = slide.shapes.add_picture(
        str(path),
        int(x + (width - draw_width) / 2),
        int(y + (height - draw_height) / 2),
        draw_width,
        draw_height,
    )
    picture.name = name
    return picture


def _base_slide(prs, spec: SlideSpec, page_number: int, *, title: str | None = None):
    slide = new_content_slide(prs, title or spec.title, page_number=page_number)
    add_takeaway(slide, spec.takeaway)
    add_source_note(slide, spec.sources)
    return slide


def _build_balanced_cases(slide, repo_root: Path) -> None:
    cases = (
        ("四模型均纠正", 640, "all_corrected_seed_20260837_row_640.png"),
        ("四模型均失败", 873, "all_not_corrected_seed_20260837_row_873.png"),
        ("模型间分歧", 1353, "model_disagreement_seed_20260837_row_1353.png"),
        ("阈值附近", 3584, "threshold_boundary_seed_20260837_row_3584.png"),
    )
    for index, (title, row_id, filename) in enumerate(cases):
        column, row = index % 2, index // 2
        x = Inches(0.55 + 6.22 * column)
        y = Inches(2.00 + 2.38 * row)
        add_text_box(
            slide,
            f"{title}｜dataset_row_id={row_id}",
            x=x,
            y=y,
            width=Inches(5.85),
            height=Inches(0.32),
            font_size=13,
            bold=True,
            name="案例标题",
            align=PP_ALIGN.CENTER,
        )
        path = repo_root / CASE_DIR / filename
        if path.exists():
            _add_contained_picture(slide, path, x=x, y=y + Inches(0.35), width=Inches(5.85), height=Inches(1.82), name=f"案例图：{row_id}")
        else:
            add_placeholder(slide, f"【待补：{title}；建议来源：{path}】", x=x, y=y + Inches(0.35), width=Inches(5.85), height=Inches(1.82))


def _build_model_details(slide) -> None:
    rows = [
        ("G0_time_only", "patch_mixer1d", "无", "—", "—"),
        ("G0_f0_native", "patch_mixer1d", "conv2d STFT", "30 s / 5 s", "pre-mixer"),
        ("G3_C_wide_8p0", "patch_mixer1d", "conv2d STFT", "20 s / 2.5 s", "pre-mixer"),
        ("G3_C_bandenergy", "patch_mixer1d", "5 频带序列", "20 s / 2.5 s", "pre-mixer"),
    ]
    add_three_line_table(slide, ("方案", "时序主干", "时频编码", "窗 / 步长", "融合位置"), rows, x=Inches(0.65), y=Inches(2.10), width=Inches(12.0), height=Inches(3.35), font_size=14)
    add_text_box(slide, "所有方案输出同一 THO-like soft-z 波形；THO 目标不作为推理阶段输入。", x=Inches(1.0), y=Inches(5.70), width=Inches(11.3), height=Inches(0.52), font_size=17, bold=True, name="正文提示", align=PP_ALIGN.CENTER)


def _build_training_details(slide) -> None:
    rows = [
        ("最大训练轮数", "50", "早停决定是否提前结束"),
        ("批量大小", "128", "四方案一致"),
        ("优化器 / 学习率", "Adam / 1e-3", "无学习率 warmup"),
        ("早停", "patience=8, min_delta=0.001", "基于验证损失"),
        ("方向一致性筛选", "auto_direction, max=0.5", "过滤明显方向异常"),
        ("独立训练", "3 次", "方案内均值 ± 标准差"),
    ]
    add_three_line_table(slide, ("设置", "当前值", "汇报含义"), rows, x=Inches(0.70), y=Inches(2.02), width=Inches(11.9), height=Inches(3.55), font_size=14)
    add_text_box(slide, "启用损失权重：L_env 1.0；L_spec 0.2；L_smooth 0.1；L_high 0.2；L_rel_env 0.03；signed corr / cos 使用早期权重调度。", x=Inches(0.92), y=Inches(5.75), width=Inches(11.45), height=Inches(0.62), font_size=14, name="正文损失权重", align=PP_ALIGN.CENTER)


def _build_metric_formulas(slide) -> None:
    formulas = (
        ("稳健带通 RR", "|RR_robust(B(ŷ)) − RR_robust(B(y))|"),
        ("周期计数 bpm", "|N_cycle(B(ŷ)) − N_cycle(B(y))| / 3"),
        ("4 秒时延相关", "max corr(B(ŷ(t+τ)), B(y(t))), |τ| ≤ 4 s"),
        ("相对包络相关", "corr(R(ŷ), R(y))，R 为 2 秒包络去除约 20 秒趋势"),
        ("局部 RR", "30 秒局部窗口、10 秒步长上的 RR MAE / 相关"),
    )
    for index, (name, formula) in enumerate(formulas):
        y = Inches(2.00 + index * 0.86)
        add_text_box(slide, name, x=Inches(0.80), y=y, width=Inches(2.35), height=Inches(0.50), font_size=16, color=SCNU_BLUE, bold=True, name="公式名称")
        add_text_box(slide, formula, x=Inches(3.15), y=y, width=Inches(9.20), height=Inches(0.50), font_size=15, name="正文公式")


def _build_full_strata(slide, part: int) -> None:
    all_rows = [
        ("overall", "纯时序", "0.773", "0.778", "0.865", "0.787"),
        ("overall", "wide", "0.712", "0.717", "0.871", "0.780"),
        ("overall", "bandenergy", "0.730", "0.685", "0.867", "0.773"),
        ("baseline hard", "纯时序", "4.379", "1.891", "0.738", "2.576"),
        ("baseline hard", "wide", "3.808", "1.845", "0.741", "2.570"),
        ("baseline hard", "bandenergy", "3.903", "1.836", "0.746", "2.572"),
        ("low spectrum", "纯时序", "1.328", "1.074", "0.809", "1.263"),
        ("low spectrum", "wide", "1.236", "1.037", "0.814", "1.255"),
        ("low spectrum", "bandenergy", "1.259", "0.987", "0.813", "1.246"),
    ]
    rows = all_rows[:5] if part == 1 else all_rows[5:]
    add_three_line_table(slide, ("分层", "方案", "稳健 RR", "计数 bpm", "lag4 corr", "local RR MAE"), rows, x=Inches(0.58), y=Inches(2.18), width=Inches(12.15), height=Inches(3.55), font_size=15)
    note = "整体结果与 baseline hard 前半" if part == 1 else "baseline hard 后半与 low-spectrum"
    add_text_box(slide, note, x=Inches(1.0), y=Inches(5.95), width=Inches(11.3), height=Inches(0.42), font_size=15, color=NOTE_GRAY, name="来源注释", align=PP_ALIGN.CENTER)


def _build_subject_results(slide, part: int) -> None:
    all_rows = [
        ("220", "283", "0.708 / 0.692", "0.276 / 0.275", "0.860 / 0.865", "0.586 / 0.575"),
        ("229", "298", "0.276 / 0.260", "0.186 / 0.181", "0.921 / 0.923", "0.268 / 0.263"),
        ("286", "302", "0.730 / 0.642", "0.572 / 0.553", "0.869 / 0.871", "0.592 / 0.571"),
        ("670", "79", "7.458 / 7.164", "2.039 / 1.961", "0.623 / 0.638", "5.369 / 5.649"),
        ("671", "504", "0.748 / 0.674", "2.112 / 1.825", "0.806 / 0.820", "0.980 / 0.933"),
        ("704", "575", "0.347 / 0.275", "0.348 / 0.377", "0.911 / 0.913", "0.520 / 0.523"),
        ("726", "214", "0.384 / 0.377", "0.206 / 0.206", "0.898 / 0.899", "0.602 / 0.587"),
        ("1006", "55", "0.639 / 0.607", "0.394 / 0.400", "0.859 / 0.862", "0.857 / 0.829"),
    ]
    rows = all_rows[:4] if part == 1 else all_rows[4:]
    add_text_box(slide, "每项按“纯时序 / wide”顺序列出，不表示 delta。", x=Inches(0.80), y=Inches(1.92), width=Inches(11.8), height=Inches(0.36), font_size=13, color=NOTE_GRAY, name="来源注释", align=PP_ALIGN.CENTER)
    add_three_line_table(slide, ("受试者", "窗口数", "稳健 RR", "计数 bpm", "lag4 corr", "local RR MAE"), rows, x=Inches(0.55), y=Inches(2.42), width=Inches(12.2), height=Inches(3.35), font_size=14)


def _build_harmonic_subgroups(slide) -> None:
    rows = [
        ("阳性并集", "452", "6", "19.57%", "主要汇总层"),
        ("strong harmonic", "214", "5", "9.26%", "证据最严格"),
        ("peak doubling", "5", "2", "0.22%", "样本过少"),
        ("harmonic prominent", "233", "5", "10.09%", "能量显著"),
        ("harmonic negative", "1,472", "8", "63.72%", "参照层"),
    ]
    add_three_line_table(slide, ("分层", "窗口数", "受试者数", "占全部", "解释"), rows, x=Inches(0.70), y=Inches(2.10), width=Inches(11.9), height=Inches(3.25), font_size=14)
    add_card(slide, "证据边界", "452 个阳性窗口中 285 个来自受试者 671；214 个 strong 窗口中 175 个来自该受试者。", x=Inches(2.10), y=Inches(5.55), width=Inches(9.15), height=Inches(0.85), accent=SECONDARY_ORANGE)


def _build_harmonic_metrics(slide, data: ChartData) -> None:
    rows = [
        (
            {"g0_time_only": "纯时序", "g0_f0_native_stft_pre_mixer": "上一版 STFT", "g3_c_wide_8p0": "宽频 STFT", "g3_c_bandenergy": "频带能量"}[model],
            f"{data.harmonic.loc[model, 'robust_rr']:.3f}",
            f"{data.harmonic.loc[model, 'count_bpm']:.3f}",
            f"{data.harmonic.loc[model, 'lag4_corr']:.3f}",
            f"{data.harmonic.loc[model, 'local_rr_mae']:.3f}",
            f"{data.harmonic.loc[model, 'correction_rate']:.2%}",
        )
        for model in ("g0_time_only", "g0_f0_native_stft_pre_mixer", "g3_c_wide_8p0", "g3_c_bandenergy")
    ]
    add_three_line_table(
        slide,
        ("方案", "稳健 RR", "计数 bpm", "lag4 corr", "local RR MAE", "纠正率"),
        rows,
        x=Inches(0.58),
        y=Inches(2.25),
        width=Inches(12.15),
        height=Inches(3.45),
        bold_cells={(4, 1), (4, 2), (3, 3), (4, 4), (4, 5)},
        font_size=15,
    )
    add_text_box(slide, "bandenergy 的节律误差更低；wide 的时延校正后形态更好。", x=Inches(1.0), y=Inches(5.95), width=Inches(11.3), height=Inches(0.42), font_size=16, bold=True, name="正文结果解读", align=PP_ALIGN.CENTER)


def _build_provenance(slide) -> None:
    bullets = (
        "导出包：20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf。",
        "数据制作仓库 commit：7ab93efa0d4c8394b0fdf7c57972f75d2a98eec8。",
        "导出时 configs/default.yaml 等文件存在未提交修改。",
        "因此以导出包 provenance 中保存的配置快照和实际 metadata 为准。",
        "本 PPT 生成过程不改数据、split、标签、指标或实验产物。",
    )
    add_body_text(slide, bullets, x=Inches(0.90), y=Inches(2.03), width=Inches(11.55), height=Inches(4.50), font_size=17, bullet=True)


def build_backup_slides(prs, *, repo_root: Path, data: ChartData, assets: dict[str, Path], start_page: int) -> None:
    del assets
    simple_builders = {
        "balanced_cases": lambda slide: _build_balanced_cases(slide, repo_root),
        "model_details": _build_model_details,
        "training_details": _build_training_details,
        "metric_formulas": _build_metric_formulas,
        "harmonic_subgroups": _build_harmonic_subgroups,
        "harmonic_metrics": lambda slide: _build_harmonic_metrics(slide, data),
        "data_provenance": _build_provenance,
    }
    page_number = start_page
    for spec in BACKUP_SLIDES:
        if spec.key == "full_stratified_results":
            for part in (1, 2):
                slide = _base_slide(prs, spec, page_number, title=f"{spec.title}（{part}/2）")
                _build_full_strata(slide, part)
                page_number += 1
            continue
        if spec.key == "subject_results":
            for part in (1, 2):
                slide = _base_slide(prs, spec, page_number, title=f"备份：8 名测试受试者（{part}/2）")
                _build_subject_results(slide, part)
                page_number += 1
            continue
        slide = _base_slide(prs, spec, page_number)
        simple_builders[spec.key](slide)
        page_number += 1
