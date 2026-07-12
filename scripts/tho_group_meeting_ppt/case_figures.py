from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from omegaconf import OmegaConf

from .evidence import FORMAL_LABELS, build_evidence_catalog


CASE_ROW_IDS = (640, 873, 1353, 3584)
CASE_SEED = 20260837
SAMPLE_RATE_HZ = 100.0
SIGNAL_LENGTH = 18_000
MODEL_COLORS = {
    "g0_time_only": "#555555",
    "g0_f0_native_stft_pre_mixer": "#4C78A8",
    "g3_c_wide_8p0": "#E45756",
    "g3_c_bandenergy": "#54A24B",
}
DISPLAY_LABELS = {
    "g0_time_only": "Time-only",
    "g0_f0_native_stft_pre_mixer": "F0 STFT",
    "g3_c_wide_8p0": "Wide STFT",
    "g3_c_bandenergy": "Band-energy",
}
CASE_DESCRIPTIONS = {
    "all_corrected": ("四模型均纠偏", "展示一致现象；不能证明跨 subject 或跨 seed 的必然性。"),
    "all_not_corrected": ("四模型均未纠偏", "展示共同失败；不能据此归因于单一结构。"),
    "model_disagreement": ("模型间纠偏分歧", "展示结构敏感性；不能据此选择线上 gate。"),
    "threshold_boundary": ("阈值边界案例", "展示规则边界敏感性；不能证明阈值可部署。"),
}


def _plot_style() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(font_path)
        chinese_font = font_manager.FontProperties(fname=font_path).get_name()
    else:
        chinese_font = "Noto Sans CJK SC"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chinese_font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_for_prediction(path: Path) -> Path:
    return path.with_name(f"{path.stem}_manifest.json")


def load_case_predictions(
    prediction_paths: Iterable[str | Path],
    row_ids: Iterable[int],
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """按 dataset_row_id 连接预测；绝不依赖各 NPZ 的物理行顺序。"""
    requested = tuple(int(row_id) for row_id in row_ids)
    joined: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    label_paths: dict[str, Path] = {}
    for raw_path in prediction_paths:
        path = Path(raw_path).resolve()
        manifest_path = _manifest_for_prediction(path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"prediction manifest 不存在: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        label = str(manifest.get("label", "")).strip()
        if not label:
            raise ValueError(f"prediction manifest label 为空: {manifest_path}")
        if label in joined:
            raise ValueError(f"模型 {label} 重复 prediction 文件: {path}")
        label_paths[label] = path
        with np.load(path) as archive:
            required = {"dataset_row_id", "r_tho_hat", "tho_ref"}
            missing_arrays = required - set(archive.files)
            if missing_arrays:
                raise ValueError(f"模型 {label} 缺数组 {sorted(missing_arrays)}: {path}")
            ids = np.asarray(archive["dataset_row_id"]).reshape(-1).copy()
            prediction = np.asarray(archive["r_tho_hat"])
            reference = np.asarray(archive["tho_ref"])
        if prediction.shape != reference.shape or prediction.shape != (len(ids), SIGNAL_LENGTH):
            raise ValueError(
                f"模型 {label} prediction/reference shape 无效 {prediction.shape}/{reference.shape}; path: {path}"
            )
        positions: dict[int, int] = {}
        for index, raw_id in enumerate(ids):
            row_id = int(raw_id)
            if row_id in positions:
                raise ValueError(f"模型 {label} 重复 row {row_id}; path: {path}")
            positions[row_id] = index
        selected: dict[int, dict[str, np.ndarray]] = {}
        for row_id in requested:
            if row_id not in positions:
                raise ValueError(f"模型 {label} 缺 row {row_id}; path: {path}")
            index = positions[row_id]
            selected[row_id] = {
                # 只保留 4 个案例的独立数组，避免切片继续持有完整 (452, 18000) 矩阵。
                "prediction": np.array(prediction[index], dtype=np.float32, copy=True),
                "reference": np.array(reference[index], dtype=np.float32, copy=True),
            }
        joined[label] = selected
        del prediction, reference, ids
    anchor_label = next(iter(joined), None)
    if anchor_label is not None:
        for label, selected in joined.items():
            if label == anchor_label:
                continue
            for row_id in requested:
                anchor = joined[anchor_label][row_id]["reference"]
                current = selected[row_id]["reference"]
                if anchor.shape != current.shape or not np.array_equal(anchor, current):
                    raise ValueError(
                        f"row {row_id} reference 不一致: source {label} ({label_paths[label]}) "
                        f"vs {anchor_label} ({label_paths[anchor_label]})"
                    )
    return joined


def validate_case_identity(
    row_id: int,
    records: dict[str, dict[str, str]],
    *,
    expected_seed: int,
) -> None:
    """核对案例链条实际共有的 ID 字段；字段不存在时不臆造身份。"""
    aliases = {
        "dataset_row_id": ("dataset_row_id",),
        "samp_id": ("samp_id", "subject", "subject_id"),
        "split": ("split",),
        "window_start_s": ("window_start_s",),
        "source_npz": ("source_npz",),
        "stratum": ("stratum", "input_stratum"),
    }
    anchors: dict[str, tuple[str, str]] = {}
    for source, record in records.items():
        actual_row = record.get("dataset_row_id")
        if actual_row is not None and int(actual_row) != int(row_id):
            raise ValueError(
                f"row {row_id} source {source} field dataset_row_id={actual_row!r} 不一致"
            )
        if record.get("seed") not in (None, "") and int(record["seed"]) != int(expected_seed):
            raise ValueError(
                f"row {row_id} source {source} field seed={record['seed']!r}, expected={expected_seed}"
            )
        for canonical, candidates in aliases.items():
            key = next((name for name in candidates if record.get(name) not in (None, "")), None)
            if key is None:
                continue
            value = str(record[key]).strip()
            if canonical in anchors and anchors[canonical][1] != value:
                anchor_source, expected = anchors[canonical]
                raise ValueError(
                    f"row {row_id} source {source} field {canonical}={value!r}, "
                    f"expected={expected!r} from {anchor_source}"
                )
            anchors.setdefault(canonical, (source, value))


def _required_sha256(payload: dict[str, Any], field: str, *, label: str, path: Path) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"模型 {label} {field} hash 缺失或不是合法 64hex; path: {path}")
    return value.lower()


def _canonical_manifest_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows = _read_csv(path)
    return {(row["label"], int(row["seed"])): row for row in rows}


def _validate_prediction_provenance(
    repo_root: Path,
    prediction_path: Path,
    canonical_row: dict[str, str],
    labels_path: Path,
) -> Path:
    manifest_path = _manifest_for_prediction(prediction_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    label = canonical_row["label"]
    seed = int(canonical_row["seed"])
    for field, expected in (("label", label), ("seed", seed), ("split", "test")):
        if payload.get(field) != expected:
            raise ValueError(
                f"模型 {label} row provenance {field}={payload.get(field)!r}, expected {expected!r}; path: {manifest_path}"
            )
    expected_checkpoint = (repo_root / canonical_row["checkpoint"]).resolve()
    actual_checkpoint = (repo_root / str(payload.get("checkpoint", ""))).resolve()
    if actual_checkpoint != expected_checkpoint:
        raise ValueError(
            f"模型 {label} checkpoint 不一致: {actual_checkpoint} != {expected_checkpoint}; path: {manifest_path}"
        )
    if not expected_checkpoint.is_file():
        raise FileNotFoundError(f"模型 {label} checkpoint 不存在: {expected_checkpoint}")
    expected_checkpoint_hash = _required_sha256(
        payload, "checkpoint_sha256", label=label, path=manifest_path
    )
    if _sha256(expected_checkpoint) != expected_checkpoint_hash:
        raise ValueError(
            f"模型 {label} checkpoint_sha256 hash 不一致: checkpoint {expected_checkpoint}; path: {manifest_path}"
        )
    actual_labels = (repo_root / str(payload.get("labels_path", ""))).resolve()
    if actual_labels != labels_path.resolve():
        raise ValueError(f"模型 {label} labels_path 不一致: {actual_labels}; path: {manifest_path}")
    expected_labels_hash = _required_sha256(payload, "labels_sha256", label=label, path=manifest_path)
    if _sha256(labels_path) != expected_labels_hash:
        raise ValueError(
            f"模型 {label} labels_sha256 hash 不一致: labels {labels_path}; path: {manifest_path}"
        )
    actual_output = (repo_root / str(payload.get("output_path", ""))).resolve()
    if actual_output != prediction_path.resolve():
        raise ValueError(f"模型 {label} output_path 不一致: {actual_output}; path: {manifest_path}")
    config_path = (repo_root / str(payload.get("config", ""))).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"模型 {label} config 不存在: {config_path}; path: {manifest_path}")
    if config_path.parent != expected_checkpoint.parent:
        raise ValueError(
            f"模型 {label} config 与 checkpoint 不在同一 run: config {config_path}; checkpoint {expected_checkpoint}; "
            f"path: {manifest_path}"
        )
    config_hash = payload.get("config_sha256")
    if config_hash is not None:
        expected_config_hash = _required_sha256(payload, "config_sha256", label=label, path=manifest_path)
        if _sha256(config_path) != expected_config_hash:
            raise ValueError(f"模型 {label} config_sha256 hash 不一致: {config_path}; path: {manifest_path}")
    config_seed = OmegaConf.select(OmegaConf.load(config_path), "training.seed")
    if config_seed is None or int(config_seed) != seed:
        raise ValueError(
            f"模型 {label} config training.seed={config_seed!r}, expected={seed}; config yaml: {config_path}"
        )

    eval_manifest_path = (repo_root / canonical_row["manifest_output"]).resolve()
    eval_rows = _read_csv(eval_manifest_path)
    if len(eval_rows) != 1:
        raise ValueError(f"模型 {label} canonical eval manifest 需要恰好一行: {eval_manifest_path}")
    eval_row = eval_rows[0]
    checks = {
        "checkpoint": (repo_root / eval_row["checkpoint"]).resolve() == expected_checkpoint,
        "config": (repo_root / eval_row["config"]).resolve() == config_path,
        "split": eval_row.get("split") == "test",
        "sample_seed": int(eval_row.get("sample_seed", -1)) == seed,
    }
    for field, valid in checks.items():
        if not valid:
            raise ValueError(
                f"模型 {label} canonical eval manifest field {field} 不一致: {eval_manifest_path}"
            )
    return manifest_path


def _unique_rows(path: Path, key: str, requested: Iterable[int]) -> dict[int, dict[str, str]]:
    wanted = set(requested)
    found: dict[int, dict[str, str]] = {}
    for row in _read_csv(path):
        row_id = int(row[key])
        if row_id not in wanted:
            continue
        if row_id in found:
            raise ValueError(f"重复 row {row_id}; path: {path}")
        found[row_id] = row
    for row_id in wanted:
        if row_id not in found:
            raise ValueError(f"缺 row {row_id}; path: {path}")
    return found


def _load_bcg_window(
    dataset_index: Path,
    row: dict[str, str],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    source = (dataset_index.parent / row["source_npz"]).resolve()
    key_field = "bcg_rawish_segment_soft_z_key"
    key = row.get(key_field, "")
    start = int(round(float(row["window_start_s"]) * SAMPLE_RATE_HZ))
    gap = {"source_npz": str(source), "key_field": key_field, "key": key, "start_sample": start}
    try:
        with np.load(source) as archive:
            if key not in archive.files:
                gap["reason"] = f"missing key {key!r}"
                return None, gap
            signal = np.asarray(archive[key][start : start + SIGNAL_LENGTH], dtype=np.float32)
    except (OSError, ValueError) as exc:
        gap["reason"] = str(exc)
        return None, gap
    if signal.shape != (SIGNAL_LENGTH,) or not np.isfinite(signal).all():
        gap["reason"] = f"invalid shape/finite state: {signal.shape}"
        return None, gap
    gap["status"] = "available"
    return signal, gap


def _load_target_window(dataset_index: Path, row: dict[str, str]) -> tuple[np.ndarray, Path]:
    source = (dataset_index.parent / row["target_source_npz"]).resolve()
    key_field = "target_waveform_segment_soft_z_key"
    key = row.get(key_field, "")
    start = int(round(float(row["window_start_s"]) * SAMPLE_RATE_HZ))
    with np.load(source) as archive:
        if key not in archive.files:
            raise ValueError(
                f"row {row['dataset_row_id']} dataset target field {key_field}={key!r} 不存在; source: {source}"
            )
        target = np.array(
            archive[key][start : start + SIGNAL_LENGTH], dtype=np.float32, copy=True
        )
    if target.shape != (SIGNAL_LENGTH,) or not np.isfinite(target).all():
        raise ValueError(
            f"row {row['dataset_row_id']} dataset target shape/finite 无效 {target.shape}; source: {source}"
        )
    return target, source


def _style_axis(axis: plt.Axes) -> None:
    axis.tick_params(labelsize=8, colors="black")
    axis.xaxis.label.set_color("black")
    axis.yaxis.label.set_color("black")
    axis.title.set_color("black")
    axis.grid(alpha=0.18, linewidth=0.6)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _reserve_footer(fig: plt.Figure) -> None:
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.045, 1.0, 0.94))


def _overall_figure(
    label_rows: dict[str, dict[str, str]],
    delta_rows: dict[str, dict[str, str]],
    output: Path,
) -> None:
    metrics = (
        ("robust RR error", "rr_peak_band_robust_abs_error", "bpm", "error: Δ<0 表示数值下降"),
        ("count error", "breath_count_zero_cross_bpm_error", "bpm", "error: Δ<0 表示数值下降"),
        ("lag-aware corr", "best_lag_corr_4s", "corr", "correlation: Δ>0 表示数值上升"),
        ("local RR MAE", "local_rr_mae", "bpm", "error: Δ<0 表示数值下降"),
    )
    comparisons = ("f0_vs_time", "wide_vs_time", "bandenergy_vs_time")
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("Held-out test：paired delta vs Time-only（3 seeds）", fontsize=18, color="black")
    for axis, (title, prefix, unit, direction) in zip(axes.flat, metrics, strict=True):
        values = [float(delta_rows[name][f"delta_{prefix}_mean"]) for name in comparisons]
        errors = [float(delta_rows[name][f"delta_{prefix}_mean_std"]) for name in comparisons]
        labels = []
        for name in comparisons:
            target = delta_rows[name]["target"]
            absolute = float(delta_rows[name][f"target_{prefix}_mean"])
            labels.append(f"{DISPLAY_LABELS[target]}\nΔ {float(delta_rows[name][f'delta_{prefix}_mean']):+.3f} | abs {absolute:.3f}")
        colors = [MODEL_COLORS[delta_rows[name]["target"]] for name in comparisons]
        axis.barh(labels, values, xerr=errors, color=colors, alpha=0.84, capsize=4)
        axis.axvline(0, color="black", linewidth=1)
        axis.set_title(f"{title} Δ ({unit})")
        axis.set_xlabel(f"target − Time-only；{direction}")
        _style_axis(axis)
    _reserve_footer(fig)
    fig.text(
        0.01,
        0.005,
        "误差条=3 个训练 seed 的 paired-delta 标准差；2310 个重叠窗口没有被当作独立样本构造 CI。数值方向已标注，不作排序断言。",
        fontsize=9,
        color="black",
    )
    _save(fig, output)


def _stability_figure(subject_rows: list[dict[str, str]], output: Path) -> None:
    rows = [row for row in subject_rows if row["comparison"] == "wide_vs_time"]
    seeds = sorted({int(row["seed"]) for row in rows})
    subjects = sorted({int(row["samp_id"]) for row in rows})
    matrix = np.full((len(seeds), len(subjects)), np.nan)
    counts = np.zeros_like(matrix)
    for row in rows:
        i, j = seeds.index(int(row["seed"])), subjects.index(int(row["samp_id"]))
        matrix[i, j] = float(row["delta_rr_peak_band_robust_abs_error_mean"])
        counts[i, j] = float(row["n_windows"])
    fig, (heat, dist) = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={"width_ratios": [1.5, 1]}, constrained_layout=True)
    bound = float(np.nanmax(np.abs(matrix)))
    image = heat.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    heat.set_xticks(range(len(subjects)), [str(subject) for subject in subjects])
    heat.set_yticks(range(len(seeds)), [str(seed) for seed in seeds])
    heat.set_xlabel("Subject (samp_id)")
    heat.set_ylabel("Training seed")
    heat.set_title("Wide − Time-only：subject 内 robust RR error Δ (bpm)")
    for i in range(len(seeds)):
        for j in range(len(subjects)):
            heat.text(j, i, f"{matrix[i,j]:+.2f}\nn={int(counts[i,j])}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(image, ax=heat, label="Δ bpm；负值=误差数值下降")
    for i, seed in enumerate(seeds):
        dist.scatter(np.full(len(subjects), i) + np.linspace(-0.08, 0.08, len(subjects)), matrix[i], s=50, alpha=0.75, label=str(seed))
        dist.plot([i - 0.18, i + 0.18], [np.nanmean(matrix[i])] * 2, color="black", linewidth=2)
    dist.axhline(0, color="black", linewidth=1)
    dist.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
    dist.set_xlabel("Training seed")
    dist.set_ylabel("Subject-level Δ robust RR error (bpm)")
    dist.set_title("每个点=一个 subject 聚合；横线=subject 均值")
    _style_axis(heat)
    _style_axis(dist)
    fig.suptitle("稳定性不是窗口误差条：seed × subject 的真实聚合", fontsize=18, color="black")
    _reserve_footer(fig)
    fig.text(0.01, 0.01, "窗口在时间上重叠；这里只展示 seed/subject 聚合，不把窗口视为独立重复。", fontsize=10, color="black")
    _save(fig, output)


def _strata_figure(rows: list[dict[str, str]], output: Path) -> None:
    selected = ("harmonic_negative", "harmonic_positive_union", "harmonic_prominent", "peak_doubling")
    comparisons = ("wide_vs_time", "bandenergy_vs_time")
    keyed = {(row["comparison"], row["stratum"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)
    for axis, (metric, title, direction) in zip(
        axes,
        (
            ("delta_rr_peak_band_robust_abs_error_mean", "Robust RR error Δ (bpm)", "负值=误差数值下降"),
            ("delta_breath_count_zero_cross_bpm_error_mean", "Count error Δ (bpm)", "负值=误差数值下降"),
        ),
        strict=True,
    ):
        x = np.arange(len(selected))
        for offset, comparison in zip((-0.18, 0.18), comparisons, strict=True):
            vals = [float(keyed[(comparison, stratum)][metric]) for stratum in selected]
            axis.bar(x + offset, vals, width=0.34, label=DISPLAY_LABELS[keyed[(comparison, selected[0])]["target"]])
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(x, [name.replace("_", "\n") for name in selected])
        axis.set_ylabel(title)
        axis.set_xlabel(direction)
        axis.legend(frameon=False)
        _style_axis(axis)
    fig.suptitle("真实分层中的取舍：二次谐波标签用于事后诊断", fontsize=18, color="black")
    _reserve_footer(fig)
    fig.text(
        0.01,
        0.01,
        "这些 strata 使用 THO reference/target-informed harmonic label，不能直接部署。可部署的 BCG-only gate 尚未验证；图中不暗示线上筛选能力。",
        fontsize=10,
        color="black",
    )
    _save(fig, output)


def _spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = signal - np.mean(signal)
    amplitude = np.abs(np.fft.rfft(centered * np.hanning(len(centered))))
    frequency = np.fft.rfftfreq(len(centered), 1.0 / SAMPLE_RATE_HZ)
    mask = (frequency >= 0.05) & (frequency <= 1.0)
    scale = max(float(np.max(amplitude[mask])), 1e-12)
    return frequency[mask], amplitude[mask] / scale


def _case_figure(
    row_id: int,
    category: str,
    case_row: dict[str, str],
    harmonic_row: dict[str, str],
    predictions: dict[str, dict[int, dict[str, np.ndarray]]],
    metrics: dict[str, dict[str, str]],
    corrections: dict[str, dict[str, str]],
    thresholds: dict[str, float],
    bcg: np.ndarray | None,
    gap: dict[str, Any],
    output: Path,
) -> None:
    reference = predictions[FORMAL_LABELS[0]][row_id]["reference"]
    time = np.arange(SIGNAL_LENGTH) / SAMPLE_RATE_HZ
    zoom = (time >= 30) & (time <= 60)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.55, 1), height_ratios=(1, 1, 1))
    overview = fig.add_subplot(grid[0, 0])
    local = fig.add_subplot(grid[1, 0])
    spectra = fig.add_subplot(grid[2, 0])
    table_axis = fig.add_subplot(grid[:2, 1])
    note_axis = fig.add_subplot(grid[2, 1])
    title, boundary = CASE_DESCRIPTIONS[category]
    fig.suptitle(f"Case row {row_id} | {title} | subject {case_row['samp_id']}", fontsize=17, color="black")

    if bcg is None:
        overview.text(0.5, 0.5, f"BCG evidence gap\n{gap.get('reason', 'unavailable')}", ha="center", va="center", transform=overview.transAxes, color="black")
    else:
        overview.plot(time, bcg, color="#9C755F", linewidth=0.55, alpha=0.8, label="BCG input (soft-z)")
    overview.plot(time, reference, color="black", linewidth=0.9, label="THO reference")
    for label in FORMAL_LABELS:
        overview.plot(time, predictions[label][row_id]["prediction"], color=MODEL_COLORS[label], linewidth=0.55, alpha=0.8, label=DISPLAY_LABELS[label])
    overview.set_xlim(0, 180)
    overview.set_xlabel("Time in 180 s window (s)")
    overview.set_ylabel("Normalized amplitude")
    overview.set_title("整窗：可用 BCG 输入、唯一 THO reference 与四模型预测")
    overview.legend(ncol=3, fontsize=7, frameon=False)

    local.plot(time[zoom], reference[zoom], color="black", linewidth=1.4, label="THO reference")
    for label in FORMAL_LABELS:
        local.plot(time[zoom], predictions[label][row_id]["prediction"][zoom], color=MODEL_COLORS[label], linewidth=0.9, label=DISPLAY_LABELS[label])
    local.set_xlim(30, 60)
    local.set_xlabel("Local time (s)")
    local.set_ylabel("Normalized amplitude")
    local.set_title("30–60 s 局部放大")

    if bcg is not None:
        freq, amp = _spectrum(bcg)
        spectra.plot(freq, amp, color="#9C755F", linewidth=1.1, label="BCG input")
    freq, amp = _spectrum(reference)
    spectra.plot(freq, amp, color="black", linewidth=1.3, label="THO reference")
    for label in FORMAL_LABELS:
        freq, amp = _spectrum(predictions[label][row_id]["prediction"])
        spectra.plot(
            freq,
            amp,
            color=MODEL_COLORS[label],
            linewidth=0.8,
            alpha=0.8,
            label=DISPLAY_LABELS[label],
        )
    f0 = float(harmonic_row["tho_reference_hz"])
    spectra.axvline(f0, color="black", linestyle="--", linewidth=1, label=f"THO f0={f0:.3f} Hz")
    spectra.axvline(2 * f0, color="#E45756", linestyle=":", linewidth=1.4, label="2×f0")
    spectra.set_xlim(0.05, 1.0)
    spectra.set_xlabel("Frequency (Hz)")
    spectra.set_ylabel("Normalized magnitude")
    spectra.set_title("整窗频谱与二次谐波证据")
    spectra.legend(fontsize=7, frameon=False)

    table_axis.axis("off")
    columns = [
        "Model",
        "RR err\nbpm",
        "Lag corr",
        "H ratio\nin→out",
        "Rel. drop",
        "Peak out\n|−1|≤tol",
        "Correction",
    ]
    cells = []
    for label in FORMAL_LABELS:
        row = metrics[label]
        correction_row = corrections[label]
        correction = correction_row["correction_status"]
        input_ratio = float(correction_row["input_harmonic_to_fundamental_ratio"])
        output_ratio = float(correction_row["output_harmonic_to_fundamental_ratio"])
        relative_drop = float(correction_row["harmonic_ratio_relative_drop"])
        output_peak = float(correction_row["output_peak_to_tho_ratio"])
        drop_hit = relative_drop >= thresholds["correction_ratio_drop_min"]
        peak_hit = abs(output_peak - 1.0) <= thresholds["peak_relative_tolerance"]
        cells.append(
            [
                DISPLAY_LABELS[label],
                f"{float(row['rr_peak_band_robust_abs_error']):.2f}",
                f"{float(row['best_lag_corr_4s']):.2f}",
                f"{input_ratio:.3f}→{output_ratio:.3f}",
                f"{relative_drop:+.3f} {'达标' if drop_hit else '未达'}",
                f"{output_peak:.3f} {'达标' if peak_hit else '未达'}",
                correction,
            ]
        )
    table = table_axis.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1, 2.0)
    for cell in table.get_celld().values():
        cell.get_text().set_color("black")
        cell.set_edgecolor("#BBBBBB")
    table_axis.set_title(
        "逐模型指标与纠偏判据（seed 20260837）\n"
        f"drop≥{thresholds['correction_ratio_drop_min']:.2f} 且 |output peak/THO−1|≤{thresholds['peak_relative_tolerance']:.2f} → corrected",
        fontsize=10,
        color="black",
    )

    note_axis.axis("off")
    ratio = float(harmonic_row["harmonic_to_fundamental_ratio"])
    boundary_distance = (
        abs(float(harmonic_row["peak_second_harmonic_relative_error"]) / thresholds["peak_relative_tolerance"] - 1.0)
        + abs(ratio / thresholds["harmonic_to_fundamental_min"] - 1.0)
        + abs(float(harmonic_row["harmonic_band_fraction"]) / thresholds["harmonic_band_fraction_min"] - 1.0)
    )
    note = (
        f"Category: {category}\n"
        f"Input stratum: {harmonic_row['stratum']}\n"
        f"Threshold version: {harmonic_row['threshold_version']}\n"
        f"Harmonic/fundamental energy: {ratio:.3f}\n"
        f"Boundary distance (3 normalized terms): {boundary_distance:.3f}\n"
        f"Distance to H-ratio threshold: {ratio - thresholds['harmonic_to_fundamental_min']:+.3f}\n"
        f"Distance to band-fraction threshold: {float(harmonic_row['harmonic_band_fraction']) - thresholds['harmonic_band_fraction_min']:+.3f}\n"
        f"Distance to peak-error tolerance: {float(harmonic_row['peak_second_harmonic_relative_error']) - thresholds['peak_relative_tolerance']:+.3f}\n"
        f"Case seed: {case_row['seed']}\n\n"
        f"这例说明什么：{title}，可用于形成可复核讨论。\n"
        f"不能说明什么：{boundary}"
    )
    note_axis.text(0, 1, note, va="top", fontsize=9, linespacing=1.45, color="black", transform=note_axis.transAxes)
    for axis in (overview, local, spectra):
        _style_axis(axis)
    _reserve_footer(fig)
    fig.text(0.01, 0.005, "呼吸带只作为 THO reference 展示一次；BCG 输入来自 dataset row 对应 source NPZ，预测按 dataset_row_id 严格连接。", fontsize=8.5, color="black")
    _save(fig, output)


def build_case_assets(
    repo_root: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _plot_style()
    catalog = build_evidence_catalog(root)
    harmonic = catalog.harmonic_root
    canonical_manifest = catalog.result_root / "g_series_test_eval_manifest.csv"
    canonical_rows = _canonical_manifest_rows(canonical_manifest)
    labels_path = harmonic / "test_v2" / "test_harmonic_labels.csv"
    threshold_path = harmonic / "harmonic_thresholds.json"
    threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    thresholds = {
        key: float(value) for key, value in threshold_payload["thresholds"].items()
    }

    prediction_paths: list[Path] = []
    prediction_manifests: list[Path] = []
    prediction_configs: list[Path] = []
    prediction_checkpoints: list[Path] = []
    prediction_eval_manifests: list[Path] = []
    for label in FORMAL_LABELS:
        path = harmonic / "predictions" / f"{label}_{CASE_SEED}_harmonic_predictions.npz"
        if not path.is_file():
            raise FileNotFoundError(f"模型 {label} seed {CASE_SEED} prediction 不存在: {path}")
        prediction_paths.append(path)
        prediction_manifests.append(
            _validate_prediction_provenance(root, path, canonical_rows[(label, CASE_SEED)], labels_path)
        )
        prediction_payload = json.loads(prediction_manifests[-1].read_text(encoding="utf-8"))
        prediction_configs.append((root / prediction_payload["config"]).resolve())
        prediction_checkpoints.append((root / prediction_payload["checkpoint"]).resolve())
        prediction_eval_manifests.append(
            (root / canonical_rows[(label, CASE_SEED)]["manifest_output"]).resolve()
        )
    predictions = load_case_predictions(prediction_paths, CASE_ROW_IDS)
    if set(predictions) != set(FORMAL_LABELS):
        raise ValueError(f"case prediction 模型集合不完整: {sorted(predictions)}")

    case_manifest_path = harmonic / "figures" / "model_case_manifest.csv"
    case_rows = _unique_rows(case_manifest_path, "dataset_row_id", CASE_ROW_IDS)
    harmonic_rows = _unique_rows(labels_path, "dataset_row_id", CASE_ROW_IDS)
    index_rows = _unique_rows(catalog.dataset_index, "dataset_row_id", CASE_ROW_IDS)
    correction_path = harmonic / "corrections" / "model_harmonic_correction.csv"
    correction_rows = _read_csv(correction_path)

    label_summary_path = catalog.result_root / "g_series_local_rr_canonical_label_summary.csv"
    label_summary = {row["label"]: row for row in _read_csv(label_summary_path)}
    paired_path = harmonic / "model_metrics" / "paired_delta_vs_time_summary.csv"
    paired_all = {
        row["comparison"]: row
        for row in _read_csv(paired_path)
        if row["stratum"] == "all_windows"
    }
    comparison_names = {
        "g0_f0_native_stft_pre_mixer": "f0_vs_time",
        "g3_c_wide_8p0": "wide_vs_time",
        "g3_c_bandenergy": "bandenergy_vs_time",
    }
    delta_rows = {comparison_names[row["target"]]: row for row in paired_all.values() if row["target"] in comparison_names}

    assets: dict[str, Path] = {
        "overall_delta": output / "overall_delta.png",
        "seed_subject_stability": output / "seed_subject_stability.png",
        "strata_tradeoffs": output / "strata_tradeoffs.png",
        **{f"case_row_{row_id}": output / f"case_row_{row_id}.png" for row_id in CASE_ROW_IDS},
    }
    _overall_figure(label_summary, delta_rows, assets["overall_delta"])

    stratified_root = catalog.result_root / "stratified_analysis_20260709"
    subject_path = stratified_root / "subject_seed_summary.csv"
    strata_path = harmonic / "model_metrics" / "paired_delta_vs_time_summary.csv"
    _stability_figure(_read_csv(subject_path), assets["seed_subject_stability"])
    _strata_figure(_read_csv(strata_path), assets["strata_tradeoffs"])

    metric_paths: list[Path] = []
    dataset_sources: list[Path] = []
    gaps: dict[str, Any] = {}
    categories: dict[str, str] = {}
    prediction_lengths: dict[str, int] = {}
    case_threshold_evidence: dict[str, Any] = {}
    for row_id in CASE_ROW_IDS:
        category = case_rows[row_id]["case_category"]
        categories[str(row_id)] = category
        metric_rows: dict[str, dict[str, str]] = {}
        correction_by_model: dict[str, dict[str, str]] = {}
        for label in FORMAL_LABELS:
            metric_path = catalog.result_root / f"{label}_{CASE_SEED}_test_metrics.csv"
            metric_paths.append(metric_path)
            metric_rows[label] = _unique_rows(metric_path, "dataset_row_id", (row_id,))[row_id]
            matches = [
                row
                for row in correction_rows
                if row["label"] == label and int(row["seed"]) == CASE_SEED and int(row["dataset_row_id"]) == row_id
            ]
            if len(matches) != 1:
                raise ValueError(f"模型 {label} row {row_id} correction 需要恰好一行; path: {correction_path}")
            correction_by_model[label] = matches[0]
            prediction_lengths[f"{label}:{row_id}"] = len(predictions[label][row_id]["prediction"])
        identity_records = {
            "dataset_index": index_rows[row_id],
            "harmonic_labels": harmonic_rows[row_id],
            "case_manifest": case_rows[row_id],
            **{f"correction:{label}": row for label, row in correction_by_model.items()},
            **{f"metrics:{label}": row for label, row in metric_rows.items()},
        }
        validate_case_identity(row_id, identity_records, expected_seed=CASE_SEED)
        dataset_target, target_source = _load_target_window(catalog.dataset_index, index_rows[row_id])
        dataset_sources.append(target_source)
        for label in FORMAL_LABELS:
            reference = predictions[label][row_id]["reference"]
            if reference.shape != dataset_target.shape or not np.array_equal(reference, dataset_target):
                raise ValueError(
                    f"row {row_id} reference 与 dataset target 不一致: source {label}; target source: {target_source}"
                )
        bcg, gap = _load_bcg_window(catalog.dataset_index, index_rows[row_id])
        source = (catalog.dataset_index.parent / index_rows[row_id]["source_npz"]).resolve()
        dataset_sources.append(source)
        if bcg is None:
            gap_path = output / f"evidence_gap_case_row_{row_id}.json"
            gap_path.write_text(json.dumps({"dataset_row_id": row_id, **gap}, indent=2, ensure_ascii=False), encoding="utf-8")
            gaps[str(row_id)] = {"path": str(gap_path), **gap}
        _case_figure(
            row_id,
            category,
            case_rows[row_id],
            harmonic_rows[row_id],
            predictions,
            metric_rows,
            correction_by_model,
            thresholds,
            bcg,
            gap,
            assets[f"case_row_{row_id}"],
        )
        harmonic_ratio = float(harmonic_rows[row_id]["harmonic_to_fundamental_ratio"])
        boundary_distance = (
            abs(
                float(harmonic_rows[row_id]["peak_second_harmonic_relative_error"])
                / thresholds["peak_relative_tolerance"]
                - 1.0
            )
            + abs(harmonic_ratio / thresholds["harmonic_to_fundamental_min"] - 1.0)
            + abs(
                float(harmonic_rows[row_id]["harmonic_band_fraction"])
                / thresholds["harmonic_band_fraction_min"]
                - 1.0
            )
        )
        case_threshold_evidence[str(row_id)] = {
            "boundary_distance": boundary_distance,
            "distance_to_thresholds": {
                "harmonic_to_fundamental": harmonic_ratio
                - thresholds["harmonic_to_fundamental_min"],
                "harmonic_band_fraction": float(harmonic_rows[row_id]["harmonic_band_fraction"])
                - thresholds["harmonic_band_fraction_min"],
                "peak_second_harmonic_relative_error": float(
                    harmonic_rows[row_id]["peak_second_harmonic_relative_error"]
                )
                - thresholds["peak_relative_tolerance"],
            },
            "models": {
                label: {
                    "input_harmonic_to_fundamental_ratio": float(
                        correction_by_model[label]["input_harmonic_to_fundamental_ratio"]
                    ),
                    "output_harmonic_to_fundamental_ratio": float(
                        correction_by_model[label]["output_harmonic_to_fundamental_ratio"]
                    ),
                    "harmonic_ratio_relative_drop": float(
                        correction_by_model[label]["harmonic_ratio_relative_drop"]
                    ),
                    "output_peak_to_tho_ratio": float(
                        correction_by_model[label]["output_peak_to_tho_ratio"]
                    ),
                    "correction_condition_met": correction_by_model[label]["correction_status"]
                    == "corrected",
                    "correction_status": correction_by_model[label]["correction_status"],
                }
                for label in FORMAL_LABELS
            },
        }

    metadata: dict[str, Any] = {
        "case_row_ids": list(CASE_ROW_IDS),
        "models_per_case": len(FORMAL_LABELS),
        "signal_length": SIGNAL_LENGTH,
        "case_categories": categories,
        "case_titles": {str(row_id): CASE_DESCRIPTIONS[categories[str(row_id)]][0] for row_id in CASE_ROW_IDS},
        "case_prediction_lengths": prediction_lengths,
        "deltas": {
            "wide_vs_time": {
                "robust_rr_bpm": float(label_summary["g3_c_wide_8p0"]["rr_peak_band_robust_abs_error_mean_mean"])
                - float(label_summary["g0_time_only"]["rr_peak_band_robust_abs_error_mean_mean"]),
            },
            "bandenergy_vs_time": {
                "count_bpm": float(delta_rows["bandenergy_vs_time"]["delta_breath_count_zero_cross_bpm_error_mean"]),
            },
        },
        "stability": {
            "uncertainty_unit": "seed",
            "subject_aggregation_source": str(subject_path),
            "window_overlap_used_as_independent_ci": False,
        },
        "strata_interpretation": "target-informed retrospective diagnosis; not a validated deployable BCG-only gate",
        "correction_thresholds": thresholds,
        "case_threshold_evidence": case_threshold_evidence,
        "evidence_gaps": gaps,
    }
    source_paths = [
        canonical_manifest,
        label_summary_path,
        catalog.dataset_index,
        case_manifest_path,
        labels_path,
        threshold_path,
        paired_path,
        subject_path,
        harmonic / "model_metrics" / "model_stratified_metrics_summary.csv",
        harmonic / "model_metrics" / "model_stratified_metrics_seed.csv",
        harmonic / "corrections" / "model_harmonic_correction_summary.csv",
        correction_path,
        harmonic / "model_metrics" / "analysis_manifest.json",
        harmonic / "test_v2" / "analysis_manifest.json",
        harmonic / "corrections" / "analysis_manifest.json",
        *prediction_paths,
        *prediction_manifests,
        *prediction_configs,
        *prediction_checkpoints,
        *prediction_eval_manifests,
        *metric_paths,
        *dataset_sources,
        Path(__file__).resolve(),
    ]
    for row in canonical_rows.values():
        source_paths.extend(
            [
                (root / row["metrics_output"]).resolve(),
                (root / row["summary_output"]).resolve(),
                (root / row["manifest_output"]).resolve(),
            ]
        )
    source_paths.extend(
        [
            catalog.result_root / "g_series_local_rr_canonical_seed_summary.csv",
            catalog.result_root / "g_series_local_rr_canonical_delta_summary.csv",
            stratified_root / "analysis_manifest.json",
            stratified_root / "strata_seed_summary.csv",
            stratified_root / "strata_summary.csv",
            stratified_root / "subject_summary.csv",
        ]
    )
    unique_sources = sorted({path.resolve() for path in source_paths})
    manifest = {
        "generator": "scripts.tho_group_meeting_ppt.case_figures.build_case_assets",
        "metadata": metadata,
        "sources": [
            {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in unique_sources
        ],
        "assets": {
            key: {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for key, path in assets.items()
        },
    }
    (output / "case_assets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return assets, metadata
