"""冻结 W0 的 CWT-FiLM 调制统计、关联与典型窗口选择。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


PROTOCOL = "w0-cwt-film-behavior-v1-20260918"
SEEDS = (20260811, 20260812, 20260813)
WINDOW_COUNT = 2675
CHANNELS = 96
LATENT_FRAMES = 1800
TIME_BIN_FRAMES = 50
TIME_BIN_COUNT = LATENT_FRAMES // TIME_BIN_FRAMES
TAUS = (0.90, 0.95, 0.99)
MAIN_TAU = 0.95
EPSILON = 1e-12
LOW_ENERGY_RMS = 1e-6
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
QUANTILE_CHUNK = 8
RAW_HISTOGRAM_RANGE = (-8.0, 8.0)
RAW_HISTOGRAM_BINS = 160
BOUNDED_HISTOGRAM_RANGE = (-1.0, 1.0)
BOUNDED_HISTOGRAM_BINS = 200
EFFECTIVE_HISTOGRAM_RANGE = (-0.5, 0.5)
EFFECTIVE_HISTOGRAM_BINS = 200

ERROR_COLUMNS = (
    "whole_rr_abs_error_bpm",
    "local_rr_mae_bpm",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
    "one_minus_lag_aware_signed_pcc",
)
QUALITY_COLUMNS = ("waveform_confidence_score", "transient_motion_ratio")
QUALITY_GROUP_COLUMN = "waveform_confidence_level"

PRIMARY_ASSOCIATION_COLUMNS = (
    "r_scale",
    "r_shift",
    "r_total",
    "g_mean_abs",
    "b_mean_abs",
    "g_edge_abs_tau_095",
    "b_edge_abs_tau_095",
    "z_rms",
)
SATURATION_COLUMNS = tuple(
    f"{prefix}_edge_{edge}_tau_{int(round(100 * tau)):03d}"
    for prefix in ("g", "b")
    for tau in TAUS
    for edge in ("pos", "neg", "abs")
)
SUMMARY_MODULATION_COLUMNS = tuple(dict.fromkeys((*PRIMARY_ASSOCIATION_COLUMNS, *SATURATION_COLUMNS)))


@dataclass(frozen=True)
class CapturedFilmBatch:
    """原生 W0 forward 完成后保留的四个分析张量。"""

    z: torch.Tensor
    gamma_raw: torch.Tensor
    beta_raw: torch.Tensor
    z_prime: torch.Tensor


@dataclass(frozen=True)
class BatchStatistics:
    """单 batch 的紧凑统计；不持有原始 activation。"""

    window: Mapping[str, np.ndarray]
    channel: Mapping[str, np.ndarray]
    time_bin: Mapping[str, np.ndarray]
    histograms: Mapping[str, np.ndarray]


def validate_w0_contract(model: nn.Module) -> None:
    """限制到冻结的单 W 分支、六层 local trunk W0。"""

    if getattr(model, "tf_variant", None) != "crd_tf102_w":
        raise ValueError("FiLM 行为分析只接受 crd_tf102_w")
    if set(getattr(model, "branches", {})) != {"w"}:
        raise ValueError("W0 必须恰含 W condition branch")
    if len(getattr(model, "controls", ())) != 0 or getattr(model, "fusion_gate", None) is not None:
        raise ValueError("W0 不得包含 control 或 fusion gate")
    base = getattr(model, "base", None)
    if base is None or not hasattr(base, "refinement"):
        raise TypeError("W0 缺少 base/refinement")
    if len(getattr(base, "local_blocks", ())) != 6:
        raise ValueError("W0 必须使用六层 local trunk")
    if len(getattr(base, "local_tcn_blocks", ())) != 0 or getattr(base, "global_stage", None) is not None:
        raise ValueError("W0 FiLM 前路径出现未冻结的 TCN/global stage")
    for name in ("representation", "additional_representation", "capacity_control"):
        if getattr(base, name, None) is not None:
            raise ValueError(f"W0 FiLM 前路径出现未冻结模块: {name}")


def forward_with_capture(
    model: nn.Module,
    x: torch.Tensor,
    *,
    tf: Mapping[str, torch.Tensor],
) -> tuple[Mapping[str, torch.Tensor], CapturedFilmBatch]:
    """执行一次原生 forward；hook 仅保存引用，不在 decoder 前计算统计。"""

    validate_w0_contract(model)
    if model.training:
        raise ValueError("FiLM 行为采集要求 model.eval()")
    if torch.is_grad_enabled():
        raise ValueError("FiLM 行为采集要求 inference_mode/no_grad")
    if set(tf) != {"w"}:
        raise ValueError("FiLM 行为采集只接受 W feature")

    latent_tokens: list[torch.Tensor] = []
    raw_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    conditioned_inputs: list[torch.Tensor] = []

    def capture_latent(_: nn.Module, __: tuple[Any, ...], output: torch.Tensor) -> None:
        latent_tokens.append(output)

    def capture_raw(
        _: nn.Module,
        __: tuple[Any, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        raw_outputs.append(output)

    def capture_conditioned(_: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if len(inputs) != 1:
            raise RuntimeError("refinement pre-hook 输入数量错误")
        conditioned_inputs.append(inputs[0])

    handles = (
        model.base.local_blocks[-1].register_forward_hook(capture_latent),
        model.branches["w"].register_forward_hook(capture_raw),
        model.base.refinement.register_forward_pre_hook(capture_conditioned),
    )
    try:
        result = model(x, tf=tf)
    finally:
        for handle in handles:
            handle.remove()

    if len(latent_tokens) != 1 or len(raw_outputs) != 1 or len(conditioned_inputs) != 1:
        raise RuntimeError(
            "FiLM hook 捕获次数错误: "
            f"latent={len(latent_tokens)}, raw={len(raw_outputs)}, conditioned={len(conditioned_inputs)}"
        )
    tokens = latent_tokens[0]
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (LATENT_FRAMES, CHANNELS):
        raise ValueError(f"FiLM 前 token shape 错误: {tuple(tokens.shape)}")
    z = tokens.transpose(1, 2)
    gamma_raw, beta_raw = raw_outputs[0]
    z_prime = conditioned_inputs[0]
    expected = (int(x.shape[0]), CHANNELS, LATENT_FRAMES)
    if any(tuple(value.shape) != expected for value in (z, gamma_raw, beta_raw, z_prime)):
        raise ValueError(
            "FiLM 捕获 shape 错误: "
            f"Z={tuple(z.shape)}, gamma={tuple(gamma_raw.shape)}, "
            f"beta={tuple(beta_raw.shape)}, Z'={tuple(z_prime.shape)}"
        )
    return result, CapturedFilmBatch(z=z, gamma_raw=gamma_raw, beta_raw=beta_raw, z_prime=z_prime)


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"FiLM {name} 包含 NaN/Inf")


def _quantiles(flat: torch.Tensor) -> torch.Tensor:
    """分 sample 分块计算线性插值分位数，限制峰值显存。"""

    if flat.ndim != 2 or flat.shape[0] == 0:
        raise ValueError("分位数输入必须为非空二维 tensor")
    q = torch.tensor(QUANTILES, dtype=torch.float32, device=flat.device)
    outputs = []
    for start in range(0, int(flat.shape[0]), QUANTILE_CHUNK):
        chunk = flat[start : start + QUANTILE_CHUNK].float()
        outputs.append(torch.quantile(chunk, q, dim=1, interpolation="linear").transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _window_distribution(prefix: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = value.float().flatten(1)
    quantiles = _quantiles(flat)
    absolute = flat.abs()
    output = {
        f"{prefix}_mean": flat.mean(dim=1),
        f"{prefix}_std": flat.std(dim=1, unbiased=False),
        f"{prefix}_min": flat.amin(dim=1),
        f"{prefix}_max": flat.amax(dim=1),
        f"{prefix}_mean_abs": absolute.mean(dim=1),
        f"{prefix}_median_abs": _quantiles(absolute)[:, QUANTILES.index(0.50)],
        f"{prefix}_positive_fraction": (flat > 0).float().mean(dim=1),
        f"{prefix}_negative_fraction": (flat < 0).float().mean(dim=1),
        f"{prefix}_zero_fraction": (flat == 0).float().mean(dim=1),
    }
    for index, quantile in enumerate(QUANTILES):
        output[f"{prefix}_p{int(round(100 * quantile)):02d}"] = quantiles[:, index]
    return output


def _edge_statistics(prefix: str, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    bounded = torch.tanh(raw.float())
    output: dict[str, torch.Tensor] = {}
    for tau in TAUS:
        label = f"{int(round(100 * tau)):03d}"
        positive = (bounded > tau).float().mean(dim=(1, 2))
        negative = (bounded < -tau).float().mean(dim=(1, 2))
        output[f"{prefix}_edge_pos_tau_{label}"] = positive
        output[f"{prefix}_edge_neg_tau_{label}"] = negative
        output[f"{prefix}_edge_abs_tau_{label}"] = positive + negative
    return output


def _histogram(
    value: torch.Tensor,
    *,
    low: float,
    high: float,
    bins: int,
) -> tuple[np.ndarray, int, int]:
    flat = value.detach().float().flatten()
    counts = torch.histc(flat, bins=bins, min=low, max=high).to(torch.int64)
    below = int((flat < low).sum().item())
    above = int((flat > high).sum().item())
    return counts.cpu().numpy(), below, above


def _to_numpy(values: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, value in values.items():
        array = value.detach().cpu().numpy()
        if not np.isfinite(array).all():
            raise FloatingPointError(f"FiLM 统计包含 NaN/Inf: {name}")
        output[name] = array
    return output


def _frobenius(value: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    """FP32 平方、FP64 累加，避免物化完整 FP64 activation。"""

    return value.float().square().sum(dim=dim, dtype=torch.float64).sqrt()


def compute_batch_statistics(captured: CapturedFilmBatch) -> BatchStatistics:
    """从实际捕获张量计算窗口、通道、时间块和流式分布统计。"""

    z_native = captured.z
    gamma_native = captured.gamma_raw
    beta_native = captured.beta_raw
    z_prime_native = captured.z_prime
    expected = (int(z_native.shape[0]), CHANNELS, LATENT_FRAMES)
    if any(
        tuple(value.shape) != expected
        for value in (z_native, gamma_native, beta_native, z_prime_native)
    ):
        raise ValueError("FiLM batch statistics shape 不一致")
    for name, value in (
        ("Z", z_native),
        ("gamma_raw", gamma_native),
        ("beta_raw", beta_native),
        ("Z_prime", z_prime_native),
    ):
        _require_finite(name, value)

    # 保持原始 dtype 执行有效调制公式，再转 FP32 做描述性归约。
    g_native = 0.5 * torch.tanh(gamma_native)
    b_native = 0.5 * torch.tanh(beta_native)
    reconstructed_native = z_native * (1.0 + g_native) + b_native

    z = z_native.float()
    gamma = gamma_native.float()
    beta = beta_native.float()
    g = g_native.float()
    b = b_native.float()
    z_prime = z_prime_native.float()
    scale_delta = g * z
    total_delta = z_prime - z
    closure = z_prime - reconstructed_native.float()

    z_norm = _frobenius(z, dim=(1, 2))
    scale_norm = _frobenius(scale_delta, dim=(1, 2))
    shift_norm = _frobenius(b, dim=(1, 2))
    total_norm = _frobenius(total_delta, dim=(1, 2))
    denominator = z_norm + EPSILON
    z_rms = z_norm / np.sqrt(CHANNELS * LATENT_FRAMES)

    window: dict[str, torch.Tensor] = {
        "z_fro_norm": z_norm,
        "z_rms": z_rms,
        "denom_small": z_rms <= LOW_ENERGY_RMS,
        "scale_fro_norm": scale_norm,
        "shift_fro_norm": shift_norm,
        "total_fro_norm": total_norm,
        "r_scale": scale_norm / denominator,
        "r_shift": shift_norm / denominator,
        "r_total": total_norm / denominator,
        "closure_max_abs": closure.abs().flatten(1).amax(dim=1),
        "closure_mean_abs": closure.abs().flatten(1).mean(dim=1),
        "g_time_mean_abs_difference": g.diff(dim=-1).abs().mean(dim=(1, 2)),
        "b_time_mean_abs_difference": b.diff(dim=-1).abs().mean(dim=(1, 2)),
        "g_legacy_abs_ge_0p49": (g.abs() >= 0.49).float().mean(dim=(1, 2)),
        "b_legacy_abs_ge_0p49": (b.abs() >= 0.49).float().mean(dim=(1, 2)),
    }
    for prefix, value in (
        ("gamma_raw", gamma),
        ("beta_raw", beta),
        ("g", g),
        ("b", b),
    ):
        window.update(_window_distribution(prefix, value))
    window.update(_edge_statistics("g", gamma))
    window.update(_edge_statistics("b", beta))

    channel_z_norm = _frobenius(z, dim=2)
    channel_scale_norm = _frobenius(scale_delta, dim=2)
    channel_shift_norm = _frobenius(b, dim=2)
    channel_total_norm = _frobenius(total_delta, dim=2)
    channel_denominator = channel_z_norm + EPSILON
    channel_z_rms = channel_z_norm / np.sqrt(LATENT_FRAMES)
    main_label = f"{int(round(100 * MAIN_TAU)):03d}"
    gamma_bounded = torch.tanh(gamma)
    beta_bounded = torch.tanh(beta)
    channel: dict[str, torch.Tensor] = {
        "z_rms": channel_z_rms,
        "denom_small": channel_z_rms <= LOW_ENERGY_RMS,
        "r_scale": channel_scale_norm / channel_denominator,
        "r_shift": channel_shift_norm / channel_denominator,
        "r_total": channel_total_norm / channel_denominator,
        "g_mean_abs": g.abs().mean(dim=2),
        "b_mean_abs": b.abs().mean(dim=2),
        f"g_edge_pos_tau_{main_label}": (gamma_bounded > MAIN_TAU).float().mean(dim=2),
        f"g_edge_neg_tau_{main_label}": (gamma_bounded < -MAIN_TAU).float().mean(dim=2),
        f"b_edge_pos_tau_{main_label}": (beta_bounded > MAIN_TAU).float().mean(dim=2),
        f"b_edge_neg_tau_{main_label}": (beta_bounded < -MAIN_TAU).float().mean(dim=2),
    }

    def time_view(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0], CHANNELS, TIME_BIN_COUNT, TIME_BIN_FRAMES)

    z_time = time_view(z)
    scale_time = time_view(scale_delta)
    shift_time = time_view(b)
    total_time = time_view(total_delta)
    gamma_time = time_view(gamma_bounded)
    beta_time = time_view(beta_bounded)
    g_time = time_view(g)
    b_time = time_view(b)
    time_z_norm = _frobenius(z_time, dim=(1, 3))
    time_denominator = time_z_norm + EPSILON
    time_z_rms = time_z_norm / np.sqrt(CHANNELS * TIME_BIN_FRAMES)
    time_bin: dict[str, torch.Tensor] = {
        "z_rms": time_z_rms,
        "denom_small": time_z_rms <= LOW_ENERGY_RMS,
        "r_scale": _frobenius(scale_time, dim=(1, 3)) / time_denominator,
        "r_shift": _frobenius(shift_time, dim=(1, 3)) / time_denominator,
        "r_total": _frobenius(total_time, dim=(1, 3)) / time_denominator,
        "g_mean_abs": g_time.abs().mean(dim=(1, 3)),
        "b_mean_abs": b_time.abs().mean(dim=(1, 3)),
        f"g_edge_pos_tau_{main_label}": (gamma_time > MAIN_TAU).float().mean(dim=(1, 3)),
        f"g_edge_neg_tau_{main_label}": (gamma_time < -MAIN_TAU).float().mean(dim=(1, 3)),
        f"b_edge_pos_tau_{main_label}": (beta_time > MAIN_TAU).float().mean(dim=(1, 3)),
        f"b_edge_neg_tau_{main_label}": (beta_time < -MAIN_TAU).float().mean(dim=(1, 3)),
    }
    window["channel_r_total_std"] = channel["r_total"].std(dim=1, unbiased=False)
    window["time_bin_r_total_std"] = time_bin["r_total"].std(dim=1, unbiased=False)

    histograms: dict[str, np.ndarray] = {}
    for name, value, low, high, bins in (
        ("g", g, *EFFECTIVE_HISTOGRAM_RANGE, EFFECTIVE_HISTOGRAM_BINS),
        ("b", b, *EFFECTIVE_HISTOGRAM_RANGE, EFFECTIVE_HISTOGRAM_BINS),
        ("gamma_tanh", 2.0 * g, *BOUNDED_HISTOGRAM_RANGE, BOUNDED_HISTOGRAM_BINS),
        ("beta_tanh", 2.0 * b, *BOUNDED_HISTOGRAM_RANGE, BOUNDED_HISTOGRAM_BINS),
        ("gamma_raw", gamma, *RAW_HISTOGRAM_RANGE, RAW_HISTOGRAM_BINS),
        ("beta_raw", beta, *RAW_HISTOGRAM_RANGE, RAW_HISTOGRAM_BINS),
    ):
        counts, below, above = _histogram(value, low=low, high=high, bins=bins)
        histograms[f"{name}_counts"] = counts
        histograms[f"{name}_below"] = np.asarray([below], dtype=np.int64)
        histograms[f"{name}_above"] = np.asarray([above], dtype=np.int64)

    return BatchStatistics(
        window=_to_numpy(window),
        channel=_to_numpy(channel),
        time_bin=_to_numpy(time_bin),
        histograms=histograms,
    )


def validate_window_identity(frame: pd.DataFrame, *, expected_count: int = WINDOW_COUNT) -> None:
    required = {"seed", "dataset_row_id", "samp_id", "split"}
    missing = sorted(required - set(frame))
    if missing:
        raise KeyError(f"窗口统计缺少身份列: {missing}")
    if len(frame) != expected_count or frame["dataset_row_id"].duplicated().any():
        raise ValueError("窗口统计行数或 dataset_row_id 唯一性错误")
    if set(frame["split"].astype(str)) != {"val"}:
        raise ValueError("窗口统计只允许 split=val")
    if frame["dataset_row_id"].isna().any() or frame["samp_id"].isna().any():
        raise ValueError("窗口统计身份不得为空")
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("窗口统计含 NaN/Inf")


def validate_historical_film(
    observed: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    seed: int,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """逐窗口核对 P−1 corrected 有效 FiLM 统计。"""

    reference = historical.loc[
        historical["seed"].eq(int(seed)) & historical["intervention"].eq("FULL")
    ].copy()
    if len(reference) != WINDOW_COUNT or reference["dataset_row_id"].duplicated().any():
        raise ValueError("历史 corrected FiLM seed 矩阵不完整")
    left = observed.sort_values("dataset_row_id").reset_index(drop=True)
    right = reference.sort_values("dataset_row_id").reset_index(drop=True)
    for identity_column in ("dataset_row_id", "samp_id"):
        if not np.array_equal(left[identity_column].to_numpy(), right[identity_column].to_numpy()):
            raise ValueError(f"历史 FiLM identity 不一致: {identity_column}")
    mapping = {
        "mean_abs_gamma": "g_mean_abs",
        "median_abs_gamma": "g_median_abs",
        "mean_abs_beta": "b_mean_abs",
        "median_abs_beta": "b_median_abs",
        "gamma_time_mean_abs_difference": "g_time_mean_abs_difference",
        "beta_time_mean_abs_difference": "b_time_mean_abs_difference",
        "gamma_saturation_fraction": "g_legacy_abs_ge_0p49",
        "beta_saturation_fraction": "b_legacy_abs_ge_0p49",
    }
    maximums: dict[str, float] = {}
    for old, new in mapping.items():
        delta = np.abs(
            left[new].to_numpy(dtype=np.float64) - right[old].to_numpy(dtype=np.float64)
        )
        maximums[new] = float(delta.max(initial=0.0))
    passed = all(value <= atol for value in maximums.values())
    receipt = {"seed": int(seed), "atol": float(atol), "max_abs_deltas": maximums, "passed": passed}
    if not passed:
        raise RuntimeError(f"新统计未复现历史 corrected FiLM: {receipt}")
    return receipt


def join_frozen_sources(
    statistics: pd.DataFrame,
    metrics: pd.DataFrame,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """一对一连接冻结误差和独立质量元数据。"""

    validate_window_identity(statistics)
    ordered = statistics.sort_values("dataset_row_id").reset_index(drop=True)
    metric = metrics.sort_values("dataset_row_id").reset_index(drop=True)
    metadata = rows.sort_values("dataset_row_id").reset_index(drop=True)
    if len(metric) != WINDOW_COUNT or len(metadata) != WINDOW_COUNT:
        raise ValueError("冻结 metrics/metadata 行数不完整")
    for source_name, source in (("metrics", metric), ("metadata", metadata)):
        if source["dataset_row_id"].duplicated().any():
            raise ValueError(f"{source_name} dataset_row_id 重复")
        for key in ("dataset_row_id", "samp_id"):
            if not np.array_equal(ordered[key].to_numpy(), source[key].to_numpy()):
                raise ValueError(f"{source_name} identity 不一致: {key}")

    metric_required = {
        "whole_rr_abs_error_bpm",
        "local_rr_mae_bpm",
        "envelope_trajectory_mae",
        "global_envelope_modulation_error",
        "lag_aware_signed_pcc",
        "target_envelope_modulation",
        "envelope_target_stratum",
        "split",
        "whole_rr_target_eligible",
        "local_rr_target_eligible",
        "joint_target_eligible",
        "envelope_spearman_target_eligible",
        "joint_prediction_degenerate",
    }
    metadata_required = {
        QUALITY_GROUP_COLUMN,
        *QUALITY_COLUMNS,
        "coupling_state_id",
        "window_start_s",
        "window_end_s",
        "split",
        "usable",
    }
    if missing := sorted(metric_required - set(metric)):
        raise KeyError(f"冻结 metrics 缺少列: {missing}")
    if missing := sorted(metadata_required - set(metadata)):
        raise KeyError(f"冻结 metadata 缺少列: {missing}")
    if (
        metric["joint_prediction_degenerate"].isna().any()
        or not metric["joint_prediction_degenerate"].isin([True, False]).all()
        or metric["joint_prediction_degenerate"].astype(bool).any()
    ):
        raise RuntimeError("冻结 metrics 出现 prediction degeneracy")
    if set(metric["split"].astype(str)) != {"val"} or set(metadata["split"].astype(str)) != {"val"}:
        raise ValueError("冻结 metrics/metadata 不是纯 validation")
    for column in (
        "whole_rr_target_eligible",
        "local_rr_target_eligible",
        "joint_target_eligible",
        "envelope_spearman_target_eligible",
    ):
        if (
            metric[column].isna().any()
            or not metric[column].isin([True, False]).all()
            or not metric[column].astype(bool).all()
        ):
            raise RuntimeError(f"冻结主指标 target eligibility 漂移: {column}")
    if (
        metadata["usable"].isna().any()
        or not metadata["usable"].isin([True, False]).all()
        or not metadata["usable"].astype(bool).all()
    ):
        raise RuntimeError("冻结 validation metadata 出现 unusable row")
    metric_values = metric[
        [
            "whole_rr_abs_error_bpm",
            "local_rr_mae_bpm",
            "envelope_trajectory_mae",
            "global_envelope_modulation_error",
            "lag_aware_signed_pcc",
            "target_envelope_modulation",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(metric_values).all():
        raise FloatingPointError("冻结主指标含 NaN/Inf")
    quality_values = metadata[list(QUALITY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(quality_values).all():
        raise FloatingPointError("质量元数据含 NaN/Inf")

    output = ordered.copy()
    for column in (
        "whole_rr_abs_error_bpm",
        "local_rr_mae_bpm",
        "envelope_trajectory_mae",
        "global_envelope_modulation_error",
        "target_envelope_modulation",
        "envelope_target_stratum",
    ):
        output[column] = metric[column].to_numpy()
    output["one_minus_lag_aware_signed_pcc"] = 1.0 - metric[
        "lag_aware_signed_pcc"
    ].to_numpy(dtype=np.float64)
    for column in (*QUALITY_COLUMNS, QUALITY_GROUP_COLUMN, "coupling_state_id", "window_start_s", "window_end_s"):
        output[column] = metadata[column].to_numpy()
    validate_window_identity(output)
    return output


def saturation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = []
    for prefix in ("g", "b"):
        for tau in TAUS:
            label = f"{int(round(100 * tau)):03d}"
            columns.extend(
                (
                    f"{prefix}_edge_pos_tau_{label}",
                    f"{prefix}_edge_neg_tau_{label}",
                    f"{prefix}_edge_abs_tau_{label}",
                )
            )
    records = []
    for seed, group in frame.groupby("seed", sort=True):
        for column in columns:
            records.append(
                {
                    "seed": int(seed),
                    "parameter": column[0],
                    "edge": column.split("_edge_")[1].split("_tau_")[0],
                    "tau": int(column.rsplit("_", 1)[1]) / 100.0,
                    "mean_fraction": float(group[column].mean()),
                    "median_fraction": float(group[column].median()),
                    "p95_fraction": float(group[column].quantile(0.95, interpolation="linear")),
                    "windows": int(len(group)),
                }
            )
    return pd.DataFrame.from_records(records)


def subject_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *SUMMARY_MODULATION_COLUMNS,
        *ERROR_COLUMNS,
        *QUALITY_COLUMNS,
        "target_envelope_modulation",
    ]
    output = frame.groupby(["seed", "samp_id"], sort=True)[columns].agg(["mean", "median"])
    output.columns = [f"{column}_{statistic}" for column, statistic in output.columns]
    return output.reset_index()


def quality_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [*SUMMARY_MODULATION_COLUMNS, *ERROR_COLUMNS]
    records = []
    for (seed, level), group in frame.groupby(["seed", QUALITY_GROUP_COLUMN], sort=True):
        for column in columns:
            records.append(
                {
                    "seed": int(seed),
                    QUALITY_GROUP_COLUMN: str(level),
                    "variable": column,
                    "mean": float(group[column].mean()),
                    "median": float(group[column].median()),
                    "p25": float(group[column].quantile(0.25, interpolation="linear")),
                    "p75": float(group[column].quantile(0.75, interpolation="linear")),
                    "windows": int(len(group)),
                    "samp_ids": int(group["samp_id"].nunique()),
                }
            )
    return pd.DataFrame.from_records(records)


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float | None, str]:
    values = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(values) < 3:
        return None, "insufficient_units"
    if values["x"].nunique() < 2 or values["y"].nunique() < 2:
        return None, "constant_input"
    correlation = values["x"].rank(method="average").corr(values["y"].rank(method="average"))
    if correlation is None or not np.isfinite(correlation):
        return None, "undefined"
    return float(correlation), "ok"


def association_table(frame: pd.DataFrame) -> pd.DataFrame:
    """输出 pooled、逐 samp、samp mean 和 target-stratum 敏感性。"""

    targets = [(column, "error") for column in ERROR_COLUMNS]
    targets.extend((column, "quality") for column in QUALITY_COLUMNS)
    records: list[dict[str, Any]] = []

    def append(
        group: pd.DataFrame,
        *,
        seed: int,
        view: str,
        samp_id: int | None = None,
        target_stratum: str | None = None,
    ) -> None:
        for modulation in PRIMARY_ASSOCIATION_COLUMNS:
            for target, target_kind in targets:
                value, status = _spearman(group[modulation], group[target])
                records.append(
                    {
                        "seed": int(seed),
                        "view": view,
                        "samp_id": samp_id,
                        "target_stratum": target_stratum,
                        "modulation_variable": modulation,
                        "target_variable": target,
                        "target_kind": target_kind,
                        "spearman": value,
                        "status": status,
                        "units": int(len(group)),
                    }
                )

    for seed, seed_frame in frame.groupby("seed", sort=True):
        append(seed_frame, seed=int(seed), view="pooled_windows")
        for samp_id, group in seed_frame.groupby("samp_id", sort=True):
            append(group, seed=int(seed), view="within_samp", samp_id=int(samp_id))
        means = seed_frame.groupby("samp_id", sort=True)[
            [*PRIMARY_ASSOCIATION_COLUMNS, *ERROR_COLUMNS, *QUALITY_COLUMNS]
        ].mean()
        means["samp_id"] = means.index
        append(means, seed=int(seed), view="samp_means")
        for stratum, group in seed_frame.groupby("envelope_target_stratum", sort=True):
            append(
                group,
                seed=int(seed),
                view="pooled_windows_target_stratum",
                target_stratum=str(stratum),
            )
    return pd.DataFrame.from_records(records)


def summarize_seed_variation(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """对三个固定 seed 的同定义估计报告 mean/sample SD 与定义数。"""

    records = []
    for keys, group in frame.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[value_column].dropna().to_numpy(dtype=np.float64)
        record = dict(zip(group_columns, keys, strict=True))
        record.update(
            {
                f"{value_column}_seed_mean": float(values.mean()) if len(values) else None,
                f"{value_column}_seed_sd": float(values.std(ddof=1)) if len(values) >= 2 else None,
                "defined_seed_count": int(len(values)),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def select_typical_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """按协议固定的四象限与实际质量 level medoid 选择窗口。"""

    required = {
        "seed",
        "dataset_row_id",
        "samp_id",
        QUALITY_GROUP_COLUMN,
        "r_scale",
        "r_shift",
        "r_total",
        "local_rr_mae_bpm",
        "one_minus_lag_aware_signed_pcc",
    }
    if missing := sorted(required - set(frame)):
        raise KeyError(f"case selection 缺少列: {missing}")
    if set(frame["seed"].unique()) != set(SEEDS):
        raise ValueError("case selection 要求完整三个 seed")
    feature_columns = [
        "r_scale",
        "r_shift",
        "r_total",
        "local_rr_mae_bpm",
        "one_minus_lag_aware_signed_pcc",
    ]
    means = frame.groupby("dataset_row_id", sort=True)[feature_columns].mean()
    static = frame.groupby("dataset_row_id", sort=True)[["samp_id", QUALITY_GROUP_COLUMN]].first()
    combined = means.join(static, validate="one_to_one")
    q25 = means.quantile(0.25, interpolation="linear")
    q75 = means.quantile(0.75, interpolation="linear")
    scale = (q75 - q25).replace(0.0, 1.0)
    median_r = float(combined["r_total"].median())
    median_error = float(combined["local_rr_mae_bpm"].median())
    combined["r_side"] = np.where(combined["r_total"] <= median_r, "low", "high")
    combined["error_side"] = np.where(
        combined["local_rr_mae_bpm"] <= median_error, "low", "high"
    )

    records: list[dict[str, Any]] = []

    def choose(group: pd.DataFrame, label: str, kind: str) -> None:
        if group.empty:
            records.append(
                {
                    "selection_kind": kind,
                    "selection_label": label,
                    "selection_status": "empty_cell",
                    "dataset_row_id": None,
                    "samp_id": None,
                    QUALITY_GROUP_COLUMN: None,
                    "candidate_count": 0,
                    "distance": None,
                    "r_total_cutpoint": median_r,
                    "local_rr_cutpoint": median_error,
                    **{column: None for column in feature_columns},
                }
            )
            return
        center = group[feature_columns].median()
        distance = ((group[feature_columns] - center).abs() / scale).sum(axis=1)
        ordered = pd.DataFrame(
            {"dataset_row_id": group.index.to_numpy(dtype=np.int64), "distance": distance.to_numpy()}
        ).sort_values(["distance", "dataset_row_id"], kind="mergesort")
        selected_id = int(ordered.iloc[0]["dataset_row_id"])
        selected = group.loc[selected_id]
        records.append(
            {
                "selection_kind": kind,
                "selection_label": label,
                "selection_status": "selected",
                "dataset_row_id": selected_id,
                "samp_id": int(selected["samp_id"]),
                QUALITY_GROUP_COLUMN: str(selected[QUALITY_GROUP_COLUMN]),
                "candidate_count": int(len(group)),
                "distance": float(ordered.iloc[0]["distance"]),
                "r_total_cutpoint": median_r,
                "local_rr_cutpoint": median_error,
                **{column: float(selected[column]) for column in feature_columns},
            }
        )

    for r_side in ("low", "high"):
        for error_side in ("low", "high"):
            group = combined.loc[
                combined["r_side"].eq(r_side) & combined["error_side"].eq(error_side)
            ]
            choose(group, f"r_{r_side}__error_{error_side}", "modulation_error_quadrant")
    for level in sorted(combined[QUALITY_GROUP_COLUMN].astype(str).unique()):
        choose(
            combined.loc[combined[QUALITY_GROUP_COLUMN].astype(str).eq(level)],
            f"waveform_confidence_{level}",
            "quality_level",
        )
    selection = pd.DataFrame.from_records(records)
    valid = selection["dataset_row_id"].notna()
    duplicate = valid & selection["dataset_row_id"].duplicated(keep="first")
    selection["duplicate_of_dataset_row_id"] = selection["dataset_row_id"].where(duplicate)
    selection["selected_for_render"] = valid & ~duplicate
    return selection
