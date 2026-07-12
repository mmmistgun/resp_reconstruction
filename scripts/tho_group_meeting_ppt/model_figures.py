"""生成可追溯的模型张量、STFT 分支与正式训练损失证据图。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from omegaconf import OmegaConf
import torch

from resp_train.losses.weak import WeakSyncLoss
from resp_train.models.registry import build_model
from resp_train.models.stft_branch import STFTEncoder, align_to_time
from resp_train.models.timeseries import PatchMixer1D
from scripts.tho_group_meeting_ppt.evidence import EvidenceCatalog, build_evidence_catalog


_ASSET_KEYS = ("token_geometry", "stft_branch_shapes", "loss_schedule")
_BLUE = "#DCEEFF"
_ORANGE = "#FCE4D6"
_GREEN = "#E2F0D9"
_PURPLE = "#E4DFEC"
_GRAY = "#F2F2F2"


def _plot_style() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(font_path)
        chinese_font = font_manager.FontProperties(fname=font_path).get_name()
    else:
        chinese_font = "Droid Sans Fallback"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chinese_font, "DejaVu Sans"],
            "font.size": 12,
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


def _box(ax, xy: tuple[float, float], size: tuple[float, float], text: str, color: str):
    x, y = xy
    width, height = size
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.3, edgecolor="#333333", facecolor=color,
            transform=ax.transAxes,
        )
    )
    artist = ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", color="black", transform=ax.transAxes,
        linespacing=1.35, fontsize=10.5,
    )
    artist.set_gid("layout-key")
    return artist


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=16,
            linewidth=1.5, color="#333333", transform=ax.transAxes,
        )
    )


def _shape_text(shape: list[Any]) -> str:
    return "[" + ", ".join(str(value) for value in shape) + "]"


def _layout_report(fig: plt.Figure) -> dict[str, Any]:
    """检查主要文本边界与彼此重叠；使用 bbox 几何而非像素内容断言。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    tagged = [artist for artist in fig.findobj() if getattr(artist, "get_gid", lambda: None)() == "layout-key"]
    boxes = [artist.get_window_extent(renderer=renderer) for artist in tagged if artist.get_visible()]
    inside = all(
        box.x0 >= figure_box.x0 and box.y0 >= figure_box.y0
        and box.x1 <= figure_box.x1 and box.y1 <= figure_box.y1
        for box in boxes
    )
    overlaps = 0
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            if left.overlaps(right):
                overlaps += 1
    return {
        "all_key_text_inside": bool(inside),
        "text_overlap_count": int(overlaps),
        "checked_text_count": len(boxes),
    }


def _save_figure(fig: plt.Figure, path: Path) -> dict[str, Any]:
    report = _layout_report(fig)
    if not report["all_key_text_inside"] or report["text_overlap_count"]:
        raise RuntimeError(f"图中文字布局检查失败: {path.name}: {report}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches=None, metadata={"Software": "matplotlib"})
    plt.close(fig)
    return report


def _formal_config(catalog: EvidenceCatalog):
    """G 系列四个正式配置共享主干与 loss；使用 time-only 作为共同计算图锚点。"""
    configs = catalog.run_configs
    anchor = configs["g0_time_only"]
    shared_fields = ("patch_len", "patch_stride", "mixer_layers", "base_channels")
    for label, config in configs.items():
        for field in shared_fields:
            if getattr(config, field) != getattr(anchor, field):
                raise ValueError(f"正式 G 配置的共享主干字段不一致: {label}.{field}")
    return anchor


def _flatten_config(value: Any, prefix: str = "loss") -> dict[str, Any]:
    container = OmegaConf.to_container(value, resolve=True)
    flattened: dict[str, Any] = {}

    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                visit(item[key], f"{path}.{key}")
        elif isinstance(item, list):
            flattened[path] = tuple(item)
        else:
            flattened[path] = item

    visit(container, prefix)
    return flattened


def validate_formal_loss_configs(catalog: EvidenceCatalog) -> Path:
    """逐字段核对四份正式 G resolved config 的全部 loss 配置。"""
    anchor_label = "g0_time_only"
    anchor_path = catalog.run_configs[anchor_label].path
    anchor = _flatten_config(OmegaConf.load(anchor_path).loss)
    for label, config in catalog.run_configs.items():
        current = _flatten_config(OmegaConf.load(config.path).loss)
        for field in sorted(set(anchor) | set(current)):
            expected = anchor.get(field, "<missing>")
            actual = current.get(field, "<missing>")
            if actual != expected:
                raise ValueError(
                    f"正式 G loss 配置漂移: label={label}; field={field}; "
                    f"actual={actual!r}; expected={expected!r}; path={config.path}"
                )
    return anchor_path


def _token_metadata(catalog: EvidenceCatalog) -> dict[str, Any]:
    config = _formal_config(catalog)
    model = PatchMixer1D(
        base_channels=config.base_channels,
        patch_len=config.patch_len,
        patch_stride=config.patch_stride,
        mixer_layers=config.mixer_layers,
    )
    length = 18_000
    token_count = model.token_count_for_length(length)
    probe, original_length = model.tokenize_input(torch.zeros(1, 1, length))
    padded_length = config.patch_len + (token_count - 1) * config.patch_stride
    formula_count = (padded_length - config.patch_len) // config.patch_stride + 1
    if tuple(probe.shape) != (1, config.base_channels, token_count) or formula_count != token_count:
        raise RuntimeError("PatchMixer1D 的 forward、token_count_for_length 与公式核对失败")
    return {
        "input_shape": ["B", 1, length],
        "sample_rate_hz": 100.0,
        "patch_len": config.patch_len,
        "patch_stride": config.patch_stride,
        "patch_duration_sec": config.patch_len / 100.0,
        "stride_duration_sec": config.patch_stride / 100.0,
        "patch_count": token_count,
        "padded_length": padded_length,
        "mixer_layers": config.mixer_layers,
        "token_shape": ["B", config.base_channels, token_count],
        "output_shape": ["B", 1, original_length],
    }


def _stft_probe(config, token_count: int) -> dict[str, Any]:
    cfg = OmegaConf.load(config.path)
    model = build_model(cfg).cpu().eval()
    encoder = model.stft_encoder
    if not isinstance(encoder, STFTEncoder):
        raise TypeError(f"正式 G STFT encoder 类型异常: {config.path}: {type(encoder).__name__}")
    if model.fusion_mode != "native_inject" or model.stft_inject_position != "pre_mixer":
        raise ValueError(
            f"正式 G 融合语义异常: {config.path}: "
            f"fusion_mode={model.fusion_mode}, inject={model.stft_inject_position}"
        )
    projection = model.stft_proj
    if projection is None or projection.kernel_size != (1,):
        raise ValueError(f"正式 G STFT projection 必须是 Conv1d kernel=1: {config.path}")
    projection_zero = bool(
        torch.count_nonzero(projection.weight).item() == 0
        and torch.count_nonzero(projection.bias).item() == 0
    )
    if not projection_zero:
        raise ValueError(f"正式 G STFT projection 初始化并非全零: {config.path}")
    zero = torch.zeros(1, 1, 18_000)
    time = torch.arange(18_000, dtype=torch.float32) / 100.0
    real = (torch.sin(2 * torch.pi * 0.23 * time) + 0.2 * torch.sin(2 * torch.pi * 1.1 * time))[None, None]
    window = torch.hann_window(config.stft_win)
    raw_zero = torch.stft(
        zero[:, 0], n_fft=config.stft_win, hop_length=config.stft_hop,
        win_length=config.stft_win, window=window, center=True, return_complex=True,
    )
    raw_real = torch.stft(
        real[:, 0], n_fft=config.stft_win, hop_length=config.stft_hop,
        win_length=config.stft_win, window=window, center=True, return_complex=True,
    )
    observed: dict[str, list[int]] = {}

    def record_encoder(_module, _inputs, output) -> None:
        observed["encoder"] = list(output.shape)

    def record_projection(_module, inputs, output) -> None:
        observed["aligned"] = list(inputs[0].shape)
        observed["projection"] = list(output.shape)

    def record_patch_head(_module, inputs) -> None:
        observed["decoder_tokens"] = list(inputs[0].shape)

    hooks = (
        encoder.register_forward_hook(record_encoder),
        projection.register_forward_hook(record_projection),
        model.time_backbone.patch_head.register_forward_pre_hook(record_patch_head),
    )
    with torch.no_grad():
        encoded_zero = encoder(zero)
        encoded_real = encoder(real)
        aligned = align_to_time(encoded_real, token_count)
        full_output = model(real)
    for hook in hooks:
        hook.remove()
    expected_observed = {
        "encoder": [1, config.base_channels, raw_real.shape[-1]],
        "aligned": [1, config.base_channels, token_count],
        "projection": [1, config.base_channels, token_count],
        "decoder_tokens": [1, token_count, config.base_channels],
    }
    if observed != expected_observed or list(full_output.shape) != [1, 1, 18_000]:
        raise RuntimeError(
            f"正式 G 完整模型 forward 张量链不一致: {config.path}: "
            f"observed={observed}, output={list(full_output.shape)}"
        )
    cropped_shape = ["B", encoder.band_bin_count(), raw_real.shape[-1]]
    encoder_input_channels = encoder.energy_band_count() if config.stft_encoder_type == "bandenergy" else 1
    encoder_input_shape = (
        ["B", encoder_input_channels, raw_real.shape[-1]]
        if config.stft_encoder_type == "bandenergy"
        else ["B", 1, encoder.band_bin_count(), raw_real.shape[-1]]
    )
    return {
        "win_samples": config.stft_win,
        "hop_samples": config.stft_hop,
        "frequency_range_hz": [config.stft_low_hz, config.stft_high_hz],
        "center": True,
        "raw_stft_shape": ["B", raw_real.shape[1], raw_real.shape[2]],
        "cropped_shape": cropped_shape,
        "encoder_type": config.stft_encoder_type,
        "encoder_input_shape": encoder_input_shape,
        "encoder_output_shape": ["B", encoded_real.shape[1], encoded_real.shape[2]],
        "aligned_token_shape": ["B", aligned.shape[1], aligned.shape[2]],
        "inject_position": config.stft_inject_position,
        "fusion_mode": model.fusion_mode,
        "projection_kernel_size": list(projection.kernel_size),
        "projection_zero_initialized": projection_zero,
        "decoder_token_shape": ["B", token_count, config.base_channels],
        "full_model_output_shape": ["B", 1, 18_000],
        "fusion_semantics": "1x1 projection, zero-initialized, additive token injection",
        "zero_and_real_probe_shapes_match": tuple(raw_zero.shape) == tuple(raw_real.shape)
        and tuple(encoded_zero.shape) == tuple(encoded_real.shape),
        "config_path": str(config.path),
    }


def _loss_metadata(formal_config: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(formal_config)
    loss = WeakSyncLoss(cfg)
    signed_corr: list[float] = []
    signed_cosine: list[float] = []
    for epoch in range(1, 11):
        loss.set_epoch(epoch)
        weights = loss.current_weights()
        signed_corr.append(float(weights["signed_corr"]))
        signed_cosine.append(float(weights["signed_cosine"]))
    fixed_fields = {
        "envelope": "envelope_weight",
        "spectrum": "spectrum_weight",
        "smooth": "smooth_weight",
        "high_freq": "high_freq_weight",
        "relative_envelope": "relative_envelope_weight",
    }
    fixed = {name: float(OmegaConf.select(cfg, f"loss.{field}")) for name, field in fixed_fields.items()}
    optional_fields = {
        "phase_alignment": "phase_alignment_weight",
        "band_waveform": "band_waveform_weight",
        "curvature": "curvature_weight",
        "rhythm": "rhythm_weight",
        "signed_rms_envelope": "signed_rms_envelope_weight",
        "signed_mean": "signed_mean_weight",
        "si_sdr": "si_sdr_weight",
        "stft_dist": "stft_dist_weight",
        "stft_band_energy": "stft_band_energy_weight",
        "stft_peak_anchor": "stft_peak_anchor_weight",
        "fb_aux": "fb_aux_weight",
        "fb_consistency": "fb_consistency_weight",
    }
    disabled = [
        name for name, field in optional_fields.items()
        if float(OmegaConf.select(cfg, f"loss.{field}", default=0.0)) == 0.0
    ]
    scheduled = {"signed_corr": "signed_corr_schedule", "signed_cosine": "signed_cosine_schedule"}
    all_terms = [*fixed, *scheduled, *disabled]
    return {
        "epochs": list(range(1, 11)),
        "signed_corr": signed_corr,
        "signed_cosine": signed_cosine,
        "fixed_weights": fixed,
        "scheduled_terms": scheduled,
        "all_weighted_terms": all_terms,
        "disabled_terms": disabled,
        "objects": {
            "envelope": "prediction/target RMS envelope",
            "spectrum": "whole-window respiratory-band power distribution",
            "smooth": "prediction first difference",
            "high_freq": "prediction power above respiratory band",
            "relative_envelope": "slow-normalized envelope trace",
            "signed_corr": "signed waveform correlation (polarity)",
            "signed_cosine": "signed waveform direction (polarity warm-up)",
        },
    }


def _plot_token_geometry(metadata: dict[str, Any], path: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(14.4, 8.0))
    ax.set_axis_off()
    title = ax.text(0.5, 0.95, "PatchMixer1D：180 s 窗口如何变成 140 个 token", ha="center", va="center", fontsize=20, fontweight="bold", transform=ax.transAxes)
    title.set_gid("layout-key")
    labels = [
        ("输入 waveform\n[B, 1, 18,000]\n100 Hz × 180 s", _BLUE),
        (f"重叠切片\npatch = {metadata['patch_len']}\nstride = {metadata['patch_stride']}\n单位：samples", _ORANGE),
        (f"Linear embedding\n256 → {metadata['token_shape'][1]} dims\nTokens [B, C, N]\n= [B, 16, 140]", _GREEN),
        (f"{metadata['mixer_layers']} × mixer block\nToken mixing\nChannel mixing\n[B, 16, 140]", _PURPLE),
        ("Patch head\nOverlap-add (Hann)\n输出 waveform\n[B, 1, 18,000]", _BLUE),
    ]
    xs = [0.035, 0.235, 0.435, 0.635, 0.835]
    for x, (label, color) in zip(xs, labels):
        _box(ax, (x, 0.52), (0.13, 0.22), label, color)
    for x in xs[:-1]:
        _arrow(ax, (x + 0.132, 0.63), (x + 0.195, 0.63))

    ax.text(0.08, 0.39, "全窗比例与局部放大", fontsize=13, fontweight="bold", transform=ax.transAxes, color="black")
    base_x, base_y, scale = 0.20, 0.29, 0.62
    ax.add_patch(Rectangle((base_x, base_y), scale, 0.045, facecolor="#E8E8E8", edgecolor="#555555", transform=ax.transAxes))
    patch_width = scale * metadata["patch_len"] / metadata["padded_length"]
    stride_width = scale * metadata["patch_stride"] / metadata["padded_length"]
    for idx, color in enumerate(("#5B9BD5", "#ED7D31")):
        ax.add_patch(Rectangle((base_x + idx * stride_width, base_y), patch_width, 0.045, facecolor=color, alpha=0.82, edgecolor="#333333", transform=ax.transAxes))
    full_scale_note = ax.text(0.51, 0.35, "180.48 s padded window；彩色 patch/stride 按真实全窗比例", ha="center", va="center", fontsize=9.5, transform=ax.transAxes, color="black")
    full_scale_note.set_gid("layout-key")

    inset_x, inset_y, local_patch, local_stride = 0.38, 0.225, 0.18, 0.09
    ax.add_patch(Rectangle((inset_x - 0.03, inset_y - 0.025), 0.43, 0.10, facecolor="white", edgecolor="#777777", linestyle="--", transform=ax.transAxes))
    for idx, color in enumerate(("#5B9BD5", "#ED7D31", "#70AD47")):
        ax.add_patch(Rectangle((inset_x + idx * local_stride, inset_y), local_patch, 0.05, facecolor=color, alpha=0.62, edgecolor="#333333", transform=ax.transAxes))
    local_note = ax.text(0.565, 0.205, f"局部放大（非按全窗比例）：patch={metadata['patch_duration_sec']:.2f} s，stride={metadata['stride_duration_sec']:.2f} s", ha="center", va="top", fontsize=9.5, transform=ax.transAxes)
    local_note.set_gid("layout-key")
    note = ax.text(0.5, 0.14, f"padded length = {metadata['padded_length']:,}；N = ({metadata['padded_length']} − {metadata['patch_len']}) / {metadata['patch_stride']} + 1 = {metadata['patch_count']}", ha="center", va="center", fontsize=14, transform=ax.transAxes)
    note.set_gid("layout-key")
    footer = ax.text(0.02, 0.045, "证据：正式 G 系列 resolved config + PatchMixer1D.token_count_for_length + 实际 tokenize_input forward；维度未按示意图猜测。", ha="left", va="bottom", fontsize=10, transform=ax.transAxes)
    footer.set_gid("layout-key")
    return _save_figure(fig, path)


def _plot_stft_shapes(branches: dict[str, dict[str, Any]], path: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(14.4, 8.0))
    ax.set_axis_off()
    title = ax.text(0.5, 0.95, "正式 G 系列 STFT 分支：真实张量形状与注入位置", ha="center", va="center", fontsize=20, fontweight="bold", transform=ax.transAxes)
    title.set_gid("layout-key")
    display = (("F0 native", "f0"), ("Wide", "wide"), ("Bandenergy", "bandenergy"))
    y_positions = (0.69, 0.43, 0.17)
    for (name, key), y in zip(display, y_positions):
        item = branches[key]
        raw = _shape_text(item["raw_stft_shape"])
        crop = _shape_text(item["cropped_shape"])
        enc_in = _shape_text(item["encoder_input_shape"])
        _box(ax, (0.025, y), (0.13, 0.16), f"{name}\nwin/hop={item['win_samples']}/{item['hop_samples']}\n{item['frequency_range_hz'][0]}–{item['frequency_range_hz'][1]} Hz", _BLUE)
        _box(ax, (0.205, y), (0.14, 0.16), f"torch.stft\ncenter=True\n{raw}", _ORANGE)
        _box(ax, (0.395, y), (0.14, 0.16), f"频带裁剪\n{crop}\nencoder input {enc_in}", _GREEN)
        semantics = "2 × Conv2d\n频率维求均" if item["encoder_type"] == "conv2d" else "5 个重叠带求均\n2 × Conv1d"
        _box(ax, (0.585, y), (0.14, 0.16), f"{item['encoder_type']}\n{semantics}\n→ {_shape_text(item['encoder_output_shape'])}", _PURPLE)
        _box(ax, (0.775, y), (0.19, 0.16), f"Linear align\n→ {_shape_text(item['aligned_token_shape'])}\n1×1 projection（零初始化）\n{item['inject_position']} additive injection", _GRAY)
        for x0, x1 in ((0.155, 0.205), (0.345, 0.395), (0.535, 0.585), (0.725, 0.775)):
            _arrow(ax, (x0, y + 0.08), (x1, y + 0.08))
    footer = ax.text(0.02, 0.035, "本图只陈述已运行的 G 系列结构。F0/wide 使用 Conv2d 后沿频率求均；bandenergy 先压成 5 条带能量序列。新版可选 gate/cross-attention 不作为本轮事实。", ha="left", va="bottom", fontsize=10, transform=ax.transAxes)
    footer.set_gid("layout-key")
    return _save_figure(fig, path)


def _plot_loss_schedule(schedule: dict[str, Any], path: Path) -> dict[str, Any]:
    fig = plt.figure(figsize=(14.4, 8.0))
    grid = fig.add_gridspec(2, 2, height_ratios=(3.0, 1.25), width_ratios=(1.25, 1.0), left=0.07, right=0.97, top=0.88, bottom=0.12, hspace=0.34, wspace=0.25)
    ax = fig.add_subplot(grid[0, 0])
    info = fig.add_subplot(grid[0, 1])
    objects = fig.add_subplot(grid[1, :])
    fig.suptitle("正式 G 系列训练目标：固定项、调度项与关闭项", fontsize=20, fontweight="bold", y=0.96, color="black")
    epochs = schedule["epochs"]
    ax.plot(epochs, schedule["signed_corr"], marker="o", linewidth=2.5, label="signed corr（调度）", color="#C00000")
    ax.plot(epochs, schedule["signed_cosine"], marker="s", linewidth=2.5, label="signed cosine（调度）", color="#4472C4")
    ax.set_title("Epoch 1–10 的有效权重（由 WeakSyncLoss.current_weights 计算）")
    ax.set_xlabel("训练 epoch")
    ax.set_ylabel("loss weight（无量纲）")
    ax.set_xticks(epochs)
    ax.set_ylim(-0.02, 0.66)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    info.set_axis_off()
    fixed_lines = "\n".join(f"{name}: {weight:g}" for name, weight in schedule["fixed_weights"].items())
    disabled_terms = schedule["disabled_terms"]
    disabled = "\n".join(
        ", ".join(disabled_terms[start : start + 3])
        for start in range(0, len(disabled_terms), 3)
    )
    _box(info, (0.03, 0.52), (0.94, 0.42), f"固定启用项\n{fixed_lines}", _GREEN)
    _box(info, (0.03, 0.02), (0.94, 0.39), f"本轮 G 系列关闭（weight=0；完整枚举）\n{disabled}\n它们是代码中的可选机制，不是本轮已训练机制", _GRAY)

    objects.set_axis_off()
    summary = (
        "约束对象与理由　｜　envelope：匹配慢变幅度　｜　spectrum：匹配呼吸带整体分布\n"
        "smooth/high_freq：抑制输出抖动与带外能量　｜　relative envelope：匹配相对幅度轨迹\n"
        "signed corr/cosine：显式约束极性；前 6 epoch 退火，之后保持 corr=0.2、cosine=0"
    )
    text = objects.text(0.01, 0.62, summary, ha="left", va="center", fontsize=11.5, wrap=True, transform=objects.transAxes)
    text.set_gid("layout-key")
    footer = objects.text(0.01, 0.06, "证据：正式 G resolved config 的 loss 字段与 resp_train/losses/weak.py 的线性 schedule 实现。本图不声称 checkpoint_best_task 属于该轮 G 实验。", ha="left", va="bottom", fontsize=10, transform=objects.transAxes)
    footer.set_gid("layout-key")
    return _save_figure(fig, path)


def _file_record(repo_root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(repo_root))
    except ValueError:
        display = str(resolved)
    payload = resolved.read_bytes()
    return {"path": display, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _write_manifest(repo_root: Path, output_dir: Path, catalog: EvidenceCatalog, assets: dict[str, Path], metadata: dict[str, Any]) -> None:
    config_records = {
        label: _file_record(repo_root, config.path)
        for label, config in catalog.run_configs.items()
    }
    sources = {
        "model_figure_code": _file_record(repo_root, Path(__file__)),
        "formal_configs": config_records,
        "patch_mixer_code": _file_record(repo_root, repo_root / "resp_train/models/timeseries.py"),
        "stft_branch_code": _file_record(repo_root, repo_root / "resp_train/models/stft_branch.py"),
        "loss_code": _file_record(repo_root, repo_root / "resp_train/losses/weak.py"),
    }
    manifest = {
        "schema_version": 1,
        "sources": sources,
        "parameters": metadata,
        "assets": {key: _file_record(repo_root, path) for key, path in assets.items()},
    }
    (output_dir / "model_assets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_model_assets(repo_root: str | Path, output_dir: str | Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """生成三张模型机制证据图，并返回恰好三个稳定 asset key 与 metadata。"""
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalog = build_evidence_catalog(root)
    anchor = _formal_config(catalog)
    validated_loss_path = validate_formal_loss_configs(catalog)
    if validated_loss_path != anchor.path:
        raise RuntimeError("正式 G loss 锚点与共享模型锚点不一致")
    token = _token_metadata(catalog)
    branches = {
        "f0": _stft_probe(catalog.run_configs["g0_f0_native_stft_pre_mixer"], token["patch_count"]),
        "wide": _stft_probe(catalog.run_configs["g3_c_wide_8p0"], token["patch_count"]),
        "bandenergy": _stft_probe(catalog.run_configs["g3_c_bandenergy"], token["patch_count"]),
    }
    schedule = _loss_metadata(anchor.path)
    assets = {key: output / f"{key}.png" for key in _ASSET_KEYS}
    _plot_style()
    layout = {
        "token_geometry": _plot_token_geometry(token, assets["token_geometry"]),
        "stft_branch_shapes": _plot_stft_shapes(branches, assets["stft_branch_shapes"]),
        "loss_schedule": _plot_loss_schedule(schedule, assets["loss_schedule"]),
    }
    metadata = {
        **token,
        "stft_branches": branches,
        "loss_schedule": schedule,
        "layout": layout,
        "sources": {
            "formal_config": str(anchor.path),
            "patch_mixer_code": str(root / "resp_train/models/timeseries.py"),
            "stft_branch_code": str(root / "resp_train/models/stft_branch.py"),
            "loss_code": str(root / "resp_train/losses/weak.py"),
        },
    }
    _write_manifest(root, output, catalog, assets, metadata)
    return assets, metadata


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[2]
    destination = repository / "docs/stage_reports/20260708/generated_assets/discussion"
    generated, details = build_model_assets(repository, destination)
    print(json.dumps({"assets": {key: str(path) for key, path in generated.items()}, "metadata": details}, ensure_ascii=False, indent=2))
