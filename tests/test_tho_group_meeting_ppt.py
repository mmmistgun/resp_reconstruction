from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from pptx import Presentation
from pptx.util import Inches
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "docs/stage_reports/20260708/组会汇报.pptx"
FINAL_DECK = REPO_ROOT / "docs/stage_reports/20260708/THO_research_v2_阶段进展_组会汇报.pptx"


def test_discussion_evidence_catalog_resolves_formal_configs_and_signal_sources():
    from scripts.tho_group_meeting_ppt.evidence import build_evidence_catalog

    catalog = build_evidence_catalog(REPO_ROOT)

    assert catalog.dataset_root.exists()
    assert catalog.dataset_index.exists()
    assert catalog.general_signal_npz.name == "f0_visual_sample_signals.npz"
    assert catalog.general_signal_npz.exists()
    assert catalog.general_sample_row_id == 8025
    assert (catalog.result_root / "g_series_test_eval_manifest.csv").exists()
    assert (catalog.harmonic_root / "model_metrics" / "model_stratified_metrics_summary.csv").exists()
    assert catalog.run_configs["g3_c_wide_8p0"].stft_win == 2000
    assert catalog.run_configs["g3_c_wide_8p0"].stft_hop == 250
    assert catalog.run_configs["g3_c_bandenergy"].stft_encoder_type == "bandenergy"
    assert all(config.path.name == "config.yaml" for config in catalog.run_configs.values())
    assert all("g_series_stft_input" in config.path.parts for config in catalog.run_configs.values())
    assert catalog.case_row_ids == (640, 873, 1353, 3584)


def test_read_run_config_evidence_reports_missing_field_with_path(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.evidence import read_run_config

    config = tmp_path / "config.yaml"
    config.write_text("model:\n  patch_len: 256\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"stft_win.*config\.yaml"):
        read_run_config(config)


def _evidence_config_text(*, dataset_root: str = "data/dataset", field_override: tuple[str, str] | None = None) -> str:
    model = {
        "patch_len": "256",
        "patch_stride": "128",
        "mixer_layers": "2",
        "base_channels": "16",
        "stft_win": "2000",
        "stft_hop": "250",
        "stft_low_hz": "0.05",
        "stft_high_hz": "8.0",
        "stft_encoder_type": "conv2d",
        "stft_inject_position": "pre_mixer",
    }
    if field_override is not None:
        model[field_override[0]] = field_override[1]
    model_yaml = "\n".join(f"  {key}: {value}" for key, value in model.items())
    return (
        "data:\n"
        f"  dataset_root: {dataset_root}\n"
        "  index_csv: training/dataset_index.csv\n"
        "  format: research_v2\n"
        "  input_set: research_v2_waveform\n"
        "  train_split: train\n"
        "  val_split: val\n"
        "  target_task: waveform\n"
        "  bcg_input_key: bcg_rawish_segment_soft_z_key\n"
        "  target_key: target_waveform_segment_soft_z_key\n"
        f"model:\n{model_yaml}\n"
    )


def _synthetic_evidence_repo(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    dataset = repo / "data" / "dataset"
    (dataset / "training").mkdir(parents=True)
    (dataset / "training" / "dataset_index.csv").write_text("row_id\n1\n", encoding="utf-8")

    signal = repo / "assets" / "f0_visual_sample_signals.npz"
    signal.parent.mkdir(parents=True)
    signal.touch()
    signal.with_name("f0_visual_sample_metadata.json").write_text(
        json.dumps({"dataset_row_id": 8025, "files": {"signal_arrays_npz": signal.name}}),
        encoding="utf-8",
    )

    harmonic = repo / "harmonic"
    artifact_files = (
        harmonic / "test_v2" / "test_harmonic_labels.csv",
        harmonic / "model_metrics" / "model_stratified_metrics_summary.csv",
        harmonic / "corrections" / "model_harmonic_correction_summary.csv",
    )
    for path in artifact_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("status\ncomplete\n", encoding="utf-8")

    labels = ("g0_time_only", "g0_f0_native_stft_pre_mixer", "g3_c_wide_8p0", "g3_c_bandenergy")
    seeds = (20260700, 20260837, 20260901)
    labels_path = harmonic / "test_v2" / "test_harmonic_labels.csv"
    shared_labels_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    thresholds_path = harmonic / "harmonic_thresholds.json"
    thresholds_path.write_text(json.dumps({"threshold_version": "synthetic-v1"}), encoding="utf-8")
    thresholds_hash = hashlib.sha256(thresholds_path.read_bytes()).hexdigest()
    (harmonic / "model_metrics" / "analysis_manifest.json").write_text(
        json.dumps({
            "operation": "summarize-metrics",
            "labels_path": str(labels_path.relative_to(repo)),
            "labels_sha256": shared_labels_hash,
            "eval_root": "results",
            "dataset_index": "data/dataset/training/dataset_index.csv",
            "model_labels": list(labels),
            "seeds": list(seeds),
        }),
        encoding="utf-8",
    )
    (harmonic / "test_v2" / "analysis_manifest.json").write_text(
        json.dumps({
            "operation": "apply",
            "dataset_root": "data/dataset",
            "index_csv": "training/dataset_index.csv",
            "split": "test",
            "thresholds_path": str(thresholds_path.relative_to(repo)),
            "thresholds_sha256": thresholds_hash,
            "threshold_version": "synthetic-v1",
            "outputs": {"labels": str(labels_path.relative_to(repo))},
        }),
        encoding="utf-8",
    )
    (harmonic / "corrections" / "analysis_manifest.json").write_text(
        json.dumps({
            "operation": "summarize-corrections",
            "labels_path": str(labels_path.relative_to(repo)),
            "labels_sha256": shared_labels_hash,
            "threshold_version": "synthetic-v1",
            "runs": [{"label": label, "seed": seed} for label in labels for seed in seeds],
        }),
        encoding="utf-8",
    )
    case_manifest = harmonic / "figures" / "model_case_manifest.csv"
    case_manifest.parent.mkdir(parents=True)
    case_manifest.write_text(
        "dataset_row_id,case_category\n"
        "640,all_corrected\n"
        "873,all_not_corrected\n"
        "1353,model_disagreement\n"
        "3584,threshold_boundary\n",
        encoding="utf-8",
    )

    manifest = repo / "results" / "g_series_test_eval_manifest.csv"
    manifest.parent.mkdir(parents=True)
    rows = []
    for label in labels:
        for seed in seeds:
            run_dir = repo / "runs" / label / str(seed)
            run_dir.mkdir(parents=True)
            (run_dir / "checkpoint_top1.pt").touch()
            (run_dir / "config.yaml").write_text(_evidence_config_text(), encoding="utf-8")
            output_stem = repo / "results" / f"{label}_{seed}"
            for suffix in ("_metrics.csv", "_summary.csv", "_manifest.csv"):
                Path(f"{output_stem}{suffix}").touch()
            rows.append({
                "label": label,
                "seed": str(seed),
                "checkpoint": str((run_dir / "checkpoint_top1.pt").relative_to(repo)),
                "metrics_output": str(Path(f"{output_stem}_metrics.csv").relative_to(repo)),
                "summary_output": str(Path(f"{output_stem}_summary.csv").relative_to(repo)),
                "manifest_output": str(Path(f"{output_stem}_manifest.csv").relative_to(repo)),
            })
    with manifest.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=("label", "seed", "checkpoint", "metrics_output", "summary_output", "manifest_output"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return repo, manifest, harmonic, signal


def _enable_default_evidence_discovery(
    repo: Path,
    manifest: Path,
    harmonic: Path,
    signal: Path,
) -> tuple[Path, Path, Path]:
    discovered_manifest = repo / "runs" / "test_eval_g_series_synthetic_canonical" / manifest.name
    discovered_manifest.parent.mkdir(parents=True)
    manifest.rename(discovered_manifest)

    discovered_harmonic = repo / "runs" / "synthetic_harmonic_analysis"
    harmonic.rename(discovered_harmonic)
    relocated_labels = str(
        (discovered_harmonic / "test_v2" / "test_harmonic_labels.csv").relative_to(repo)
    )
    for relative in (
        Path("model_metrics/analysis_manifest.json"),
        Path("corrections/analysis_manifest.json"),
    ):
        provenance = discovered_harmonic / relative
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        payload["labels_path"] = relocated_labels
        if relative == Path("model_metrics/analysis_manifest.json"):
            payload["eval_root"] = str(discovered_manifest.parent.relative_to(repo))
        provenance.write_text(json.dumps(payload), encoding="utf-8")
    test_manifest = discovered_harmonic / "test_v2" / "analysis_manifest.json"
    test_payload = json.loads(test_manifest.read_text(encoding="utf-8"))
    test_payload["outputs"]["labels"] = relocated_labels
    test_payload["thresholds_path"] = str(
        (discovered_harmonic / "harmonic_thresholds.json").relative_to(repo)
    )
    test_manifest.write_text(json.dumps(test_payload), encoding="utf-8")

    discovered_signal = repo / "docs" / "figure" / "synthetic" / signal.name
    discovered_signal.parent.mkdir(parents=True)
    signal.rename(discovered_signal)
    signal.with_name("f0_visual_sample_metadata.json").rename(
        discovered_signal.with_name("f0_visual_sample_metadata.json")
    )
    return discovered_manifest, discovered_harmonic, discovered_signal


def test_discussion_evidence_catalog_discovers_unique_structured_candidates(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.evidence import build_evidence_catalog

    repo, manifest, harmonic, signal = _synthetic_evidence_repo(tmp_path)
    discovered = _enable_default_evidence_discovery(repo, manifest, harmonic, signal)

    catalog = build_evidence_catalog(repo)

    assert catalog.result_root == discovered[0].parent.resolve()
    assert catalog.harmonic_root == discovered[1].resolve()
    assert catalog.general_signal_npz == discovered[2].resolve()
    assert catalog.general_sample_row_id == 8025


def test_discussion_evidence_catalog_requires_explicit_manifest_when_candidates_are_ambiguous(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.evidence import build_evidence_catalog

    repo, manifest, harmonic, signal = _synthetic_evidence_repo(tmp_path)
    discovered_manifest, _, _ = _enable_default_evidence_discovery(repo, manifest, harmonic, signal)
    second = repo / "runs" / "test_eval_g_series_second_canonical" / discovered_manifest.name
    second.parent.mkdir(parents=True)
    second.write_text(discovered_manifest.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"manifest.*候选.*(?:synthetic_canonical.*second_canonical|second_canonical.*synthetic_canonical).*显式",
    ):
        build_evidence_catalog(repo)


def test_discussion_evidence_catalog_accepts_explicit_paths_relative_to_repo_root(tmp_path: Path, monkeypatch):
    from scripts.tho_group_meeting_ppt.evidence import build_evidence_catalog

    repo, manifest, harmonic, signal = _synthetic_evidence_repo(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    catalog = build_evidence_catalog(
        repo,
        manifest_path=manifest.relative_to(repo),
        harmonic_root=harmonic.relative_to(repo),
        general_signal_npz=signal.relative_to(repo),
    )

    assert catalog.dataset_root == (repo / "data" / "dataset").resolve()
    assert catalog.dataset_index == (repo / "data" / "dataset" / "training" / "dataset_index.csv").resolve()
    assert catalog.result_root == manifest.parent.resolve()
    assert catalog.harmonic_root == harmonic.resolve()
    assert catalog.general_signal_npz == signal.resolve()


def test_evidence_manifest_rejects_duplicate_label_seed(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.evidence import build_evidence_catalog

    repo, manifest, harmonic, signal = _synthetic_evidence_repo(tmp_path)
    with manifest.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    rows.append(rows[0].copy())
    with manifest.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=r"重复.*g0_time_only.*20260700.*g_series_test_eval_manifest\.csv"):
        build_evidence_catalog(repo, manifest_path=manifest, harmonic_root=harmonic, general_signal_npz=signal)


def test_evidence_manifest_checks_nonfirst_seed_data_provenance(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.evidence import build_evidence_catalog

    repo, manifest, harmonic, signal = _synthetic_evidence_repo(tmp_path)
    mismatched = repo / "runs" / "g3_c_wide_8p0" / "20260837" / "config.yaml"
    other_dataset = repo / "data" / "other"
    (other_dataset / "training").mkdir(parents=True)
    (other_dataset / "training" / "dataset_index.csv").touch()
    mismatched.write_text(_evidence_config_text(dataset_root="data/other"), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"g3_c_wide_8p0.*20260837.*data\.dataset_root.*config\.yaml",
    ):
        build_evidence_catalog(repo, manifest_path=manifest, harmonic_root=harmonic, general_signal_npz=signal)


def test_evidence_harmonic_provenance_rejects_mismatched_eval_root(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.evidence import build_evidence_catalog

    repo, manifest, harmonic, signal = _synthetic_evidence_repo(tmp_path)
    provenance = harmonic / "model_metrics" / "analysis_manifest.json"
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["eval_root"] = "results/wrong_eval"
    provenance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"model_metrics/analysis_manifest\.json.*eval_root.*wrong_eval.*results",
    ):
        build_evidence_catalog(repo, manifest_path=manifest, harmonic_root=harmonic, general_signal_npz=signal)


def test_evidence_harmonic_provenance_rejects_wrong_case_category(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.evidence import build_evidence_catalog

    repo, manifest, harmonic, signal = _synthetic_evidence_repo(tmp_path)
    case_manifest = harmonic / "figures" / "model_case_manifest.csv"
    case_manifest.write_text(
        case_manifest.read_text(encoding="utf-8").replace("640,all_corrected", "640,wrong_category"),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"model_case_manifest\.csv.*dataset_row_id=640.*case_category.*wrong_category.*all_corrected",
    ):
        build_evidence_catalog(repo, manifest_path=manifest, harmonic_root=harmonic, general_signal_npz=signal)


def test_evidence_harmonic_provenance_rejects_tampered_labels_hash(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.evidence import build_evidence_catalog

    repo, manifest, harmonic, signal = _synthetic_evidence_repo(tmp_path)
    labels = harmonic / "test_v2" / "test_harmonic_labels.csv"
    labels.write_text(labels.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"test_harmonic_labels\.csv.*labels_sha256.*实际.*期望",
    ):
        build_evidence_catalog(repo, manifest_path=manifest, harmonic_root=harmonic, general_signal_npz=signal)


def test_evidence_harmonic_provenance_rejects_tampered_thresholds_hash(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.evidence import build_evidence_catalog

    repo, manifest, harmonic, signal = _synthetic_evidence_repo(tmp_path)
    thresholds = harmonic / "harmonic_thresholds.json"
    thresholds.write_text('{"tampered": true}', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"harmonic_thresholds\.json.*thresholds_sha256.*实际.*期望",
    ):
        build_evidence_catalog(repo, manifest_path=manifest, harmonic_root=harmonic, general_signal_npz=signal)


@pytest.mark.parametrize(
    ("field", "yaml_value"),
    (
        ("patch_len", "true"),
        ("patch_len", "2.9"),
        ("stft_low_hz", ".nan"),
        ("stft_encoder_type", "[conv2d]"),
    ),
)
def test_read_run_config_evidence_rejects_invalid_model_field_types(
    tmp_path: Path,
    field: str,
    yaml_value: str,
):
    from scripts.tho_group_meeting_ppt.evidence import read_run_config

    config = tmp_path / "config.yaml"
    config.write_text(_evidence_config_text(field_override=(field, yaml_value)), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"model\.{field}.*config\.yaml"):
        read_run_config(config)


def test_body_text_is_black_and_table_has_only_three_horizontal_rules():
    from scripts.tho_group_meeting_ppt.theme import (
        BODY_BLACK,
        WHITE,
        add_body_text,
        add_three_line_table,
        is_three_line_table,
        new_content_slide,
    )

    prs = Presentation()
    slide = new_content_slide(prs, "测试标题", page_number=1)
    box = add_body_text(
        slide,
        ["正文内容"],
        x=Inches(1.0),
        y=Inches(1.5),
        width=Inches(5.0),
        height=Inches(1.0),
    )
    table = add_three_line_table(
        slide,
        ["方案", "指标"],
        [["A", "0.1"]],
        x=Inches(1.0),
        y=Inches(2.8),
        width=Inches(6.0),
        height=Inches(1.2),
    )

    assert box.text_frame.paragraphs[0].runs[0].font.color.rgb == BODY_BLACK
    assert is_three_line_table(table)
    assert all(cell.fill.fore_color.rgb == WHITE for row in table.rows for cell in row.cells)
    assert not table.first_row
    assert not table.first_col
    assert not table.horz_banding
    assert not table.vert_banding
    header_tags = [child.tag.rsplit("}", 1)[-1] for child in table.cell(0, 0)._tc.get_or_add_tcPr()]
    assert header_tags[:3] == ["lnT", "lnB", "solidFill"]


def test_main_deck_has_exactly_25_ordered_slides_and_numeric_sources():
    from scripts.tho_group_meeting_ppt.content import MAIN_SLIDES

    assert len(MAIN_SLIDES) == 25
    assert MAIN_SLIDES[0].key == "title"
    assert MAIN_SLIDES[10].key == "overall_test_results"
    assert MAIN_SLIDES[24].key == "next_stage_takeaways"
    assert all(slide.sources for slide in MAIN_SLIDES if slide.contains_numeric_evidence)
    assert MAIN_SLIDES[0].placeholder == "【待补：汇报人、汇报日期】"


EXPECTED_DISCUSSION_SECTIONS = {
    "任务与信号直觉",
    "数据来源与样本形成",
    "输入与目标预处理",
    "模型输入与计算图",
    "训练与损失",
    "指标计算与失效场景",
    "对照实验设计",
    "整体结果与稳定性",
    "分层分析与完整案例",
    "BCG 二次谐波问题",
    "研究议题与下一步",
}


def test_discussion_units_cover_complete_causal_chain():
    from scripts.tho_group_meeting_ppt.content import DISCUSSION_UNITS

    assert {unit.section for unit in DISCUSSION_UNITS} == EXPECTED_DISCUSSION_SECTIONS
    assert len(DISCUSSION_UNITS) >= 42
    assert len({unit.key for unit in DISCUSSION_UNITS}) == len(DISCUSSION_UNITS)


def test_discussion_unit_schema_is_frozen_and_uses_tuples():
    from dataclasses import FrozenInstanceError, fields
    from typing import get_type_hints

    from scripts.tho_group_meeting_ppt.discussion_content import DiscussionUnit

    required_fields = {
        "key", "section", "title", "kind", "question", "method_steps", "parameters",
        "rationale", "evidence", "limits", "discussion_prompt", "visual_keys", "sources",
    }
    assert required_fields <= {field.name for field in fields(DiscussionUnit)}
    hints = get_type_hints(DiscussionUnit)
    for name in ("method_steps", "parameters", "rationale", "evidence", "limits", "visual_keys", "sources"):
        assert hints[name] == tuple[str, ...]
    assert hints["discussion_prompt"] == str | None
    unit = DiscussionUnit(key="k", section="s", title="t", kind="technical", question="q")
    with pytest.raises(FrozenInstanceError):
        unit.title = "changed"
    for name in ("method_steps", "parameters", "rationale", "evidence", "limits", "visual_keys", "sources"):
        assert isinstance(getattr(unit, name), tuple)


def test_discussion_technical_units_are_actionable_and_traceable():
    from scripts.tho_group_meeting_ppt.content import DISCUSSION_UNITS

    allowed_kinds = {"title", "section", "technical"}
    assert {unit.kind for unit in DISCUSSION_UNITS} <= allowed_kinds
    technical = [unit for unit in DISCUSSION_UNITS if unit.kind == "technical"]
    assert all(unit.method_steps or unit.visual_keys for unit in technical)
    assert all(unit.sources for unit in technical)


def test_discussion_units_have_no_blank_scalar_or_tuple_content():
    from scripts.tho_group_meeting_ppt.content import DISCUSSION_UNITS

    scalar_fields = ("key", "section", "title", "kind", "question")
    tuple_fields = ("method_steps", "parameters", "rationale", "evidence", "limits", "visual_keys", "sources")
    for unit in DISCUSSION_UNITS:
        assert all(isinstance(getattr(unit, name), str) and getattr(unit, name).strip() for name in scalar_fields), unit.key
        for name in tuple_fields:
            assert all(isinstance(value, str) and value.strip() for value in getattr(unit, name)), (unit.key, name)
        if unit.discussion_prompt is not None:
            assert isinstance(unit.discussion_prompt, str) and unit.discussion_prompt.strip(), unit.key


def test_high_risk_discussion_units_encode_required_domain_semantics():
    from scripts.tho_group_meeting_ppt.content import DISCUSSION_UNITS

    by_key = {unit.key: unit for unit in DISCUSSION_UNITS}
    patch = "\n".join((*by_key["patchmixer_token_shape"].method_steps, *by_key["patchmixer_token_shape"].parameters))
    assert "patch_len" in patch and "stride" in patch and "token" in patch

    loss = "\n".join((*by_key["composite_loss_goal"].method_steps, *by_key["composite_loss_goal"].rationale))
    assert "包络" in loss and "频谱" in loss and "方向" in loss

    robust_rr = "\n".join((*by_key["metric_robust_rr"].method_steps, *by_key["metric_robust_rr"].limits))
    assert "Welch" in robust_rr and "峰间距" in robust_rr and "双峰" in robust_rr

    paired = "\n".join((*by_key["paired_delta"].method_steps, *by_key["paired_delta"].limits))
    seed = "\n".join((*by_key["seed_stability"].method_steps, *by_key["seed_stability"].limits))
    assert "配对" in paired and "三个 seed" in paired
    assert "seed" in seed and "样本量小" in seed

    correction = "\n".join((*by_key["harmonic_correction"].method_steps, *by_key["harmonic_correction"].limits))
    assert "10%" in correction and "20%" in correction and "高纠正率不等于" in correction


def test_target_supervision_unit_distinguishes_window_filter_loss_and_rr_mask():
    from scripts.tho_group_meeting_ppt.content import DISCUSSION_UNITS

    unit = next(unit for unit in DISCUSSION_UNITS if unit.key == "target_soft_z_and_mask")
    text = "\n".join((*unit.method_steps, *unit.parameters, *unit.rationale, *unit.evidence, *unit.limits))
    assert "整窗筛选" in text
    assert "完整 target 波形" in text
    assert "rr_peak_valid_mask" in text
    assert "RR" in text and "评价" in text
    assert "仅在允许位置形成监督" not in text


def test_overall_count_result_states_cycle_and_bpm_units():
    from scripts.tho_group_meeting_ppt.content import DISCUSSION_UNITS

    unit = next(unit for unit in DISCUSSION_UNITS if unit.key == "overall_metric_values")
    parameters = "\n".join(unit.parameters)
    assert "0.712186 bpm" in parameters
    assert "0.870774" in parameters and "无量纲" in parameters
    assert "2.054401" in parameters and "周期数绝对误差" in parameters
    assert "0.684800" in parameters and "bpm" in parameters
    assert "0.773299 bpm" in parameters


def test_checkpoint_selection_describes_actual_g_series_topk_and_reevaluation():
    from scripts.tho_group_meeting_ppt.content import DISCUSSION_UNITS

    unit = next(unit for unit in DISCUSSION_UNITS if unit.key == "checkpoint_selection")
    text = "\n".join((*unit.method_steps, *unit.parameters, *unit.evidence, *unit.limits))
    assert "方向 gate" in text
    assert "val-loss top-k" in text
    assert "旁路复评" in text
    assert "runs/test_eval_g_series_20260709_local_rr_canonical/g_series_test_eval_manifest.csv" in text
    assert "checkpoint_best_task.pt" in text and "新版可选机制" in text
    assert "本轮保存 checkpoint_best_task.pt" not in text


def test_content_module_reexports_discussion_units_without_changing_legacy_build_input():
    from scripts.tho_group_meeting_ppt import build, content, discussion_content

    assert content.DISCUSSION_UNITS is discussion_content.DISCUSSION_UNITS
    assert build.MAIN_SLIDES is content.MAIN_SLIDES
    assert build.MAIN_SLIDES is not content.DISCUSSION_UNITS


def test_discussion_sources_are_existing_repository_files():
    from scripts.tho_group_meeting_ppt.content import DISCUSSION_UNITS

    for unit in DISCUSSION_UNITS:
        for source in unit.sources:
            base_path = source.split("#", 1)[0]
            assert not Path(base_path).is_absolute(), (unit.key, source)
            assert (REPO_ROOT / base_path).is_file(), (unit.key, source)


def test_discussion_content_avoids_deck_and_status_artifacts():
    from dataclasses import fields

    from scripts.tho_group_meeting_ppt.content import DISCUSSION_UNITS

    forbidden = ("工作进度", "实验编号汇总", "复现实验命令", "现场问答", "固定页数")
    text = "\n".join(
        str(value)
        for unit in DISCUSSION_UNITS
        for value in (getattr(unit, field.name) for field in fields(unit))
    )
    assert not any(phrase in text for phrase in forbidden)
    assert all(not hasattr(unit, "slide_number") for unit in DISCUSSION_UNITS)


def test_backup_deck_covers_required_materials():
    from scripts.tho_group_meeting_ppt.content import BACKUP_SLIDES

    keys = {slide.key for slide in BACKUP_SLIDES}
    assert {
        "balanced_cases",
        "model_details",
        "training_details",
        "metric_formulas",
        "full_stratified_results",
        "subject_results",
        "harmonic_subgroups",
        "harmonic_metrics",
        "data_provenance",
    } <= keys
    assert "reproduction_commands" not in keys
    assert "qa_notes" not in keys


def test_chart_data_matches_canonical_report_values():
    from scripts.tho_group_meeting_ppt.charts import load_chart_data

    data = load_chart_data(REPO_ROOT)

    assert data.overall.loc["g3_c_wide_8p0", "robust_rr"] == pytest.approx(0.712186)
    assert data.overall.loc["g3_c_bandenergy", "count_bpm"] == pytest.approx(0.684800)
    assert data.harmonic.loc["g3_c_bandenergy", "correction_rate"] == pytest.approx(0.9720, abs=5e-5)
    assert data.harmonic.loc["g3_c_wide_8p0", "lag4_corr"] == pytest.approx(0.847504)
    assert data.harmonic_coverage["positive_union_windows"] == 452


def _shape_text(slide, name: str) -> str:
    return next(shape.text for shape in slide.shapes if shape.name == name)


def test_generated_main_deck_has_25_slides_black_body_and_editable_diagrams(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.build import build_presentation
    from scripts.tho_group_meeting_ppt.theme import BODY_BLACK

    output = tmp_path / "main.pptx"
    build_presentation(template=TEMPLATE, output=output, include_backup=False)
    prs = Presentation(output)

    assert len(prs.slides) == 25
    assert _shape_text(prs.slides[10], "页面标题") == "独立测试集结果支持宽频 STFT 作为当前基准"
    assert all(slide.slide_layout.name != "空白" for slide in list(prs.slides)[1:])
    assert sum(1 for shape in prs.slides[6].shapes if shape.name.startswith("流程节点")) >= 8
    assert sum(1 for slide in prs.slides for shape in slide.shapes if shape.has_table) >= 3
    assert sum(1 for shape in prs.slides[10].shapes if shape.shape_type == 13) == 0
    assert sum(1 for shape in prs.slides[20].shapes if shape.has_table) == 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.name.startswith(("正文", "核心结论")) or not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    assert run.font.color.rgb == BODY_BLACK


def test_backup_contains_balanced_cases_formulas_subject_table_and_commands(tmp_path: Path):
    from scripts.tho_group_meeting_ppt.build import build_presentation

    output = tmp_path / "full.pptx"
    build_presentation(template=TEMPLATE, output=output, include_backup=True)
    prs = Presentation(output)
    text = "\n".join(
        shape.text
        for slide in prs.slides
        for shape in slide.shapes
        if shape.has_text_frame
    )

    assert len(prs.slides) >= 36
    assert "dataset_row_id=640" in text
    assert "dataset_row_id=873" in text
    assert "dataset_row_id=1353" in text
    assert "dataset_row_id=3584" in text
    assert "8 名测试受试者" in text
    assert "8 名测试受试者（1/2）" in text
    assert "8 名测试受试者（2/2）" in text
    assert "复现实验命令与证据路径" not in text
    assert "现场问答提示" not in text
    assert "eval_topk_checkpoints.py" not in text


def test_final_deck_has_no_sample_text_gray_body_or_out_of_bounds_shapes():
    from scripts.tho_group_meeting_ppt.theme import BODY_BLACK, is_three_line_table

    assert FINAL_DECK.exists()
    prs = Presentation(FINAL_DECK)
    text = "\n".join(
        shape.text
        for slide in prs.slides
        for shape in slide.shapes
        if shape.has_text_frame
    )
    assert len(prs.slides) >= 36
    assert "论文分享" not in text
    assert "2021/12/21" not in text
    assert "汇报人：xxx" not in text
    assert "【待补：汇报人、汇报日期】" in text

    for slide in prs.slides:
        for shape in slide.shapes:
            assert shape.left >= 0
            assert shape.top >= 0
            assert shape.left + shape.width <= prs.slide_width
            assert shape.top + shape.height <= prs.slide_height
            if shape.has_table:
                assert is_three_line_table(shape.table)
            if shape.name.startswith(("正文", "核心结论")) and shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        assert run.font.color.rgb == BODY_BLACK


def test_cli_can_run_from_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/build_tho_group_meeting_ppt.py", "--charts-only"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
