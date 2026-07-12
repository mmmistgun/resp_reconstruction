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


def _base_slide(prs, spec: SlideSpec, page_number: int):
    slide = new_content_slide(prs, spec.title, page_number=page_number)
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


def _build_full_strata(slide) -> None:
    rows = [
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
    add_three_line_table(slide, ("分层", "方案", "稳健 RR", "计数 bpm", "lag4 corr", "local RR MAE"), rows, x=Inches(0.48), y=Inches(1.97), width=Inches(12.35), height=Inches(4.65), font_size=11)


def _build_subject_results(slide) -> None:
    rows = [
        ("220", "283", "0.708 / 0.692", "0.276 / 0.275", "0.860 / 0.865", "0.586 / 0.575"),
        ("229", "298", "0.276 / 0.260", "0.186 / 0.181", "0.921 / 0.923", "0.268 / 0.263"),
        ("286", "302", "0.730 / 0.642", "0.572 / 0.553", "0.869 / 0.871", "0.592 / 0.571"),
        ("670", "79", "7.458 / 7.164", "2.039 / 1.961", "0.623 / 0.638", "5.369 / 5.649"),
        ("671", "504", "0.748 / 0.674", "2.112 / 1.825", "0.806 / 0.820", "0.980 / 0.933"),
        ("704", "575", "0.347 / 0.275", "0.348 / 0.377", "0.911 / 0.913", "0.520 / 0.523"),
        ("726", "214", "0.384 / 0.377", "0.206 / 0.206", "0.898 / 0.899", "0.602 / 0.587"),
        ("1006", "55", "0.639 / 0.607", "0.394 / 0.400", "0.859 / 0.862", "0.857 / 0.829"),
    ]
    add_text_box(slide, "每项按“纯时序 / wide”顺序列出，不表示 delta。", x=Inches(0.80), y=Inches(1.92), width=Inches(11.8), height=Inches(0.36), font_size=13, color=NOTE_GRAY, name="来源注释", align=PP_ALIGN.CENTER)
    add_three_line_table(slide, ("受试者", "窗口数", "稳健 RR", "计数 bpm", "lag4 corr", "local RR MAE"), rows, x=Inches(0.50), y=Inches(2.32), width=Inches(12.3), height=Inches(4.18), font_size=11)


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


def _build_provenance(slide) -> None:
    bullets = (
        "导出包：20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf。",
        "数据制作仓库 commit：7ab93efa0d4c8394b0fdf7c57972f75d2a98eec8。",
        "导出时 configs/default.yaml 等文件存在未提交修改。",
        "因此以导出包 provenance 中保存的配置快照和实际 metadata 为准。",
        "本 PPT 生成过程不改数据、split、标签、指标或实验产物。",
    )
    add_body_text(slide, bullets, x=Inches(0.90), y=Inches(2.03), width=Inches(11.55), height=Inches(4.50), font_size=17, bullet=True)


def _build_commands(slide) -> None:
    commands = (
        "# Top3 checkpoint 复评（正式长任务由用户执行）\n"
        "./.venv/bin/python scripts/eval_topk_checkpoints.py --runs-root runs/<root> --top-k 3 "
        "--device cuda:0 --device cuda:1 --max-parallel 4 --metric-workers 4 --metrics-chunk-size 128\n\n"
        "# 当前 G 系列汇总\n"
        "./.venv/bin/python scripts/summarize_tho_runs.py --runs-root runs/<root> --output /tmp/tho_runs_summary.csv\n\n"
        "# 生成本 PPT\n"
        "./.venv/bin/python scripts/build_tho_group_meeting_ppt.py"
    )
    add_text_box(slide, commands, x=Inches(0.72), y=Inches(2.00), width=Inches(11.90), height=Inches(4.55), font_size=13, name="正文命令")


def _build_qa(slide) -> None:
    items = (
        ("受试者独立吗？", "train / val / test 受试者不重叠；窗口在受试者内重叠。"),
        ("soft-z 是什么？", "分段 robust-z 后对极端尾部做连续软压缩。"),
        ("为什么不用 val_loss 排序？", "阶段判断按独立测试集呼吸任务指标展开。"),
        ("为什么不补非重点方案？", "它们不回答当前阶段问题，补评不会改变决策。"),
        ("如何避免泄漏？", "现有目标侧分层仅用于回顾；下一阶段 gate 必须由 BCG-only 判据构造。"),
    )
    for index, (question, answer) in enumerate(items):
        row, column = divmod(index, 2)
        add_card(slide, question, answer, x=Inches(0.65 + column * 6.20), y=Inches(2.00 + row * 1.48), width=Inches(5.80), height=Inches(1.18), accent=(SECONDARY_ORANGE if index == 4 else SCNU_BLUE))


def build_backup_slides(prs, *, repo_root: Path, data: ChartData, assets: dict[str, Path], start_page: int) -> None:
    del data, assets
    builders = {
        "balanced_cases": lambda slide: _build_balanced_cases(slide, repo_root),
        "model_details": _build_model_details,
        "training_details": _build_training_details,
        "metric_formulas": _build_metric_formulas,
        "full_stratified_results": _build_full_strata,
        "subject_results": _build_subject_results,
        "harmonic_subgroups": _build_harmonic_subgroups,
        "data_provenance": _build_provenance,
        "reproduction_commands": _build_commands,
        "qa_notes": _build_qa,
    }
    for offset, spec in enumerate(BACKUP_SLIDES):
        slide = _base_slide(prs, spec, start_page + offset)
        builders[spec.key](slide)

