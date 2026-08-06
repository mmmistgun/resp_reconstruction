from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.signal.windows import hamming

from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics


IEWT_METHOD = "IEWT"
IEWT_SOURCE_COLUMN = "bcg_rawish_segment_soft_z_key"
IEWT_EXPECTED_SIGNAL_KEY = "bcg_rawish_wideband_state_aligned_segment_soft_z"


@dataclass(frozen=True)
class IEWTConfig:
    """协议化 Python IEWT 的冻结参数。"""

    fs: float = 100.0
    block_sec: float = 30.0
    context_sec: float = 35.0
    detrend_degree: int = 3
    lowpass_order: int = 3
    lowpass_hz: float = 1.0
    spectrum_high_hz: float = 1.0
    candidate_relative_height: float = 0.5
    high_band_center_bin_threshold: int = 6
    post_lowpass: bool = True


@dataclass(frozen=True)
class IEWTWindowResult:
    waveform: np.ndarray
    spectrum: np.ndarray
    upper_envelope: np.ndarray
    boundary_bins: np.ndarray
    boundary_hz: np.ndarray
    selected_band_indices: np.ndarray
    components: tuple[np.ndarray, ...] | None = None


@dataclass(frozen=True)
class IEWTResult:
    waveform: np.ndarray
    windows: tuple[IEWTWindowResult, ...]


def prepare_iewt_config(cfg: DictConfig) -> DictConfig:
    """复制当前配置并冻结 IEWT 所需的输入层级和 CPU 流式评价。"""

    if str(cfg.data.get("format", "")) != "research_v2":
        raise ValueError("IEWT 基线只支持当前 research_v2 数据集")
    if float(cfg.window.target_fs) != 100.0 or int(cfg.window.duration_samples) != 18000:
        raise ValueError("IEWT 基线协议固定为 100 Hz、180 秒/18000 点")
    prepared = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    prepared.data.bcg_input_key = IEWT_SOURCE_COLUMN
    prepared.data.preload_windows = False
    prepared.training.device = "cpu"
    return prepared


def extract_respiration_iewt(
    signal: np.ndarray,
    *,
    fs: float,
    config: IEWTConfig | None = None,
    keep_components: bool = False,
) -> IEWTResult:
    """按 35 秒上下文/30 秒输出协议提取与输入等长的呼吸波形。"""

    cfg = config or IEWTConfig(fs=float(fs))
    if not np.isclose(float(fs), cfg.fs):
        raise ValueError(f"采样率与 IEWT 配置不一致: fs={fs} config.fs={cfg.fs}")
    x = _as_finite_1d(signal, name="signal")
    block_samples = _seconds_to_samples(cfg.block_sec, cfg.fs, name="block_sec")
    context_samples = _seconds_to_samples(cfg.context_sec, cfg.fs, name="context_sec")
    left_context = context_samples - block_samples
    if left_context <= 0:
        raise ValueError("context_sec 必须大于 block_sec")
    if x.size % block_samples != 0:
        raise ValueError("IEWT 输入长度必须是 30 秒输出块的整数倍")
    if x.size < context_samples:
        raise ValueError("IEWT 输入短于首个 35 秒上下文窗")

    windows: list[IEWTWindowResult] = []
    output_blocks: list[np.ndarray] = []
    for block_index, output_start in enumerate(range(0, x.size, block_samples)):
        if block_index == 0:
            context = x[:context_samples]
            keep = slice(0, block_samples)
        else:
            context_start = output_start - left_context
            context = x[context_start : output_start + block_samples]
            keep = slice(left_context, context_samples)
        if context.size != context_samples:
            raise ValueError(
                f"第 {block_index} 个 IEWT 上下文长度错误: "
                f"expected={context_samples} actual={context.size}"
            )
        window_result = extract_iewt_window(
            context,
            fs=cfg.fs,
            config=cfg,
            keep_components=keep_components,
        )
        output_blocks.append(window_result.waveform[keep])
        windows.append(window_result)

    waveform = np.concatenate(output_blocks)
    if cfg.post_lowpass:
        waveform = _zero_phase_lowpass(
            waveform,
            fs=cfg.fs,
            cutoff_hz=cfg.lowpass_hz,
            order=cfg.lowpass_order,
        )
    if waveform.size != x.size or not np.isfinite(waveform).all():
        raise RuntimeError("IEWT 输出长度或有限值检查失败")
    return IEWTResult(waveform=waveform, windows=tuple(windows))


def extract_iewt_window(
    signal: np.ndarray,
    *,
    fs: float,
    config: IEWTConfig | None = None,
    keep_components: bool = False,
) -> IEWTWindowResult:
    """移植 ``pre -> Test_EEWT1D -> EEWT1D`` 的单上下文窗算法。"""

    cfg = config or IEWTConfig(fs=float(fs))
    if not np.isclose(float(fs), cfg.fs):
        raise ValueError(f"采样率与 IEWT 配置不一致: fs={fs} config.fs={cfg.fs}")
    x = _as_finite_1d(signal, name="signal")
    expected = _seconds_to_samples(cfg.context_sec, cfg.fs, name="context_sec")
    if x.size != expected:
        raise ValueError(f"单窗 IEWT 固定需要 {expected} 点，实际为 {x.size} 点")
    if x.size % 4:
        raise ValueError("IEWT 选带要求上下文长度可被 4 整除")

    preprocessed = _polynomial_detrend(x, degree=cfg.detrend_degree)
    preprocessed = _zero_phase_lowpass(
        preprocessed,
        fs=cfg.fs,
        cutoff_hz=cfg.lowpass_hz,
        order=cfg.lowpass_order,
    )
    spectrum = _one_hz_amplitude_spectrum(
        preprocessed,
        fs=cfg.fs,
        high_hz=cfg.spectrum_high_hz,
    )
    upper = _upper_envelope(spectrum)
    flat_labels, maximum_starts = _detect_envelope_flats(upper)
    boundary_bins = detect_boundaries(spectrum, flat_labels)
    components = _decompose_ewt(preprocessed, boundary_bins)
    selected = _select_bands(
        preprocessed,
        upper,
        flat_labels,
        maximum_starts,
        boundary_bins,
        fs=cfg.fs,
        relative_height=cfg.candidate_relative_height,
        high_band_center_bin_threshold=cfg.high_band_center_bin_threshold,
    )
    if selected.size == 0 or np.any(selected < 0) or np.any(selected >= len(components)):
        raise RuntimeError(f"IEWT 选带结果非法: {selected.tolist()}")
    waveform = np.sum(np.stack([components[index] for index in selected]), axis=0)
    if waveform.size != x.size or not np.isfinite(waveform).all():
        raise RuntimeError("单窗 IEWT 输出长度或有限值检查失败")
    return IEWTWindowResult(
        waveform=waveform,
        spectrum=spectrum,
        upper_envelope=upper,
        boundary_bins=boundary_bins,
        boundary_hz=boundary_bins.astype(np.float64) * cfg.fs / x.size,
        selected_band_indices=selected,
        components=components if keep_components else None,
    )


def detect_boundaries(spectrum: np.ndarray, flat_labels: np.ndarray) -> np.ndarray:
    """在相邻包络平台之间取局部最左谷值，返回零基频谱 bin。"""

    values = _as_finite_1d(spectrum, name="spectrum")
    labels = np.asarray(flat_labels, dtype=np.int64)
    if labels.ndim != 2 or labels.shape[1] != 2 or labels.shape[0] == 0:
        raise ValueError("flat_labels 必须是非空 [n, 2] 数组")
    if np.any(labels < 0) or np.any(labels[:, 0] > labels[:, 1]) or np.any(labels[:, 1] >= values.size):
        raise ValueError("flat_labels 越界或起止顺序错误")
    if np.any(np.diff(labels[:, 0]) <= 0):
        raise ValueError("flat_labels 必须按起点严格递增")
    if labels.shape[0] > 1 and np.any(labels[:-1, 1] > labels[1:, 0]):
        raise ValueError("flat_labels 区间不能相互交叠")

    boundaries: list[int] = []
    for index, (_, end) in enumerate(labels):
        if index + 1 < len(labels):
            next_start = int(labels[index + 1, 0])
            region_start, region_end = int(end), next_start
        elif int(end) < values.size - 1:
            region_start, region_end = int(end), values.size - 1
        else:
            continue
        region = values[region_start : region_end + 1]
        boundary = region_start + int(np.argmin(region))
        if boundary > 0:
            boundaries.append(boundary)

    result = np.asarray(boundaries, dtype=np.int64)
    if result.size == 0:
        raise ValueError("IEWT 未检测到有效频带边界")
    if np.any(np.diff(result) <= 0):
        raise ValueError(f"IEWT 边界不是严格递增序列: {result.tolist()}")
    return result


def meyer_filter_bank(boundaries: np.ndarray, n_samples: int) -> tuple[np.ndarray, ...]:
    """生成 EEWT 使用的实值 Meyer scaling/wavelet 频域滤波器组。"""

    bounds = np.asarray(boundaries, dtype=np.float64).reshape(-1)
    if n_samples < 2:
        raise ValueError("n_samples 必须至少为 2")
    if bounds.size == 0 or not np.isfinite(bounds).all():
        raise ValueError("Meyer filter bank 需要有限且非空的边界")
    if np.any(bounds <= 0.0) or np.any(bounds >= np.pi) or np.any(np.diff(bounds) <= 0.0):
        raise ValueError("Meyer 边界必须严格递增并位于 (0, pi)")

    ratios = [
        (bounds[index + 1] - bounds[index]) / (bounds[index + 1] + bounds[index])
        for index in range(bounds.size - 1)
    ]
    ratios.append((np.pi - bounds[-1]) / (np.pi + bounds[-1]))
    gamma = (1.0 - 1.0 / n_samples) * min(1.0, *ratios)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"Meyer transition ratio 非法: {gamma}")

    filters: list[np.ndarray] = [_meyer_scaling(bounds[0], gamma, n_samples)]
    filters.extend(
        _meyer_wavelet(bounds[index], bounds[index + 1], gamma, n_samples)
        for index in range(bounds.size - 1)
    )
    filters.append(_meyer_wavelet(bounds[-1], np.pi, gamma, n_samples))
    return tuple(filters)


def evaluate_iewt_loader(
    loader: Iterable[Mapping[str, Any]],
    cfg: DictConfig,
    *,
    include_test_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从 BCG 独立生成 IEWT prediction，并复用冻结任务指标。"""

    fs = float(cfg.window.target_fs)
    duration_samples = int(cfg.window.duration_samples)
    algorithm_config = IEWTConfig(fs=fs)
    frames: list[pd.DataFrame] = []
    for batch in loader:
        inputs = _as_numpy(batch["x"])
        targets = _as_numpy(batch["target"])
        if inputs.ndim == 3 and inputs.shape[1] == 1:
            inputs = inputs[:, 0, :]
        if targets.ndim == 3 and targets.shape[1] == 1:
            targets = targets[:, 0, :]
        if inputs.ndim != 2 or targets.ndim != 2 or inputs.shape != targets.shape:
            raise ValueError(f"IEWT batch shape 非法: input={inputs.shape} target={targets.shape}")
        if inputs.shape[1] != duration_samples:
            raise ValueError(
                f"IEWT sample 长度与配置不一致: expected={duration_samples} actual={inputs.shape[1]}"
            )
        results = [extract_respiration_iewt(sample, fs=fs, config=algorithm_config) for sample in inputs]
        predictions = np.stack([result.waveform for result in results])
        frame = evaluate_task_predictions(
            {
                "r_tho_hat": predictions,
                "tho_ref": targets,
                **_metric_metadata(batch.get("meta", {})),
            },
            cfg,
            include_test_only=include_test_only,
            method=IEWT_METHOD,
        )
        frame["iewt_boundary_bins_by_block"] = [
            json.dumps([window.boundary_bins.tolist() for window in result.windows], separators=(",", ":"))
            for result in results
        ]
        frame["iewt_selected_band_indices_by_block"] = [
            json.dumps(
                [window.selected_band_indices.tolist() for window in result.windows], separators=(",", ":")
            )
            for result in results
        ]
        frame["iewt_boundary_count_min"] = [
            min(window.boundary_bins.size for window in result.windows) for result in results
        ]
        frame["iewt_boundary_count_max"] = [
            max(window.boundary_bins.size for window in result.windows) for result in results
        ]
        frame["iewt_selected_mode_count_max"] = [
            max(window.selected_band_indices.size for window in result.windows) for result in results
        ]
        frames.append(frame)
    if not frames:
        raise ValueError("IEWT 基线没有可评价 sample")
    metrics = pd.concat(frames, ignore_index=True)
    summary = summarize_task_metrics(metrics)
    summary.insert(0, "method", IEWT_METHOD)
    return metrics, summary


def add_iewt_summary_metadata(summary: pd.DataFrame, metrics: pd.DataFrame, *, split: str) -> pd.DataFrame:
    enriched = summary.copy()
    enriched.insert(1, "split", str(split))
    if "bcg_signal_key" in metrics:
        keys = sorted(set(metrics["bcg_signal_key"].astype(str)))
        if keys != [IEWT_EXPECTED_SIGNAL_KEY]:
            raise ValueError(
                f"IEWT 读取了意外信号 key: expected={IEWT_EXPECTED_SIGNAL_KEY} actual={keys}"
            )
        enriched["source_signal_key"] = keys[0]
    enriched["iewt_fs_hz"] = 100.0
    enriched["iewt_context_sec"] = 35.0
    enriched["iewt_output_step_sec"] = 30.0
    enriched["iewt_post_lowpass_hz"] = 1.0
    enriched["iewt_filter_phase"] = "zero_phase"
    return enriched


def _polynomial_detrend(signal: np.ndarray, *, degree: int) -> np.ndarray:
    if degree < 0 or signal.size <= degree:
        raise ValueError("多项式去趋势阶数必须小于信号长度")
    # 归一化横坐标改善三阶最小二乘的条件数；表示的多项式子空间与 MATLAB detrend 相同。
    coordinate = np.linspace(-1.0, 1.0, signal.size, dtype=np.float64)
    coefficients = np.polynomial.polynomial.polyfit(coordinate, signal, degree)
    trend = np.polynomial.polynomial.polyval(coordinate, coefficients)
    result = signal - trend
    if not np.isfinite(result).all():
        raise ValueError("多项式去趋势产生非有限值")
    return result


def _zero_phase_lowpass(signal: np.ndarray, *, fs: float, cutoff_hz: float, order: int) -> np.ndarray:
    """使用前后向 SOS 滤波消除两处历史因果低通引入的系统群延迟。"""

    if not 0.0 < cutoff_hz < fs / 2.0:
        raise ValueError("低通截止频率必须位于 (0, Nyquist)")
    sos = butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    try:
        result = sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))
    except ValueError as error:
        raise ValueError("信号过短，无法执行 IEWT 零相位低通") from error
    if not np.isfinite(result).all():
        raise ValueError("零相位低通产生非有限值")
    return result


def _one_hz_amplitude_spectrum(signal: np.ndarray, *, fs: float, high_hz: float) -> np.ndarray:
    n_samples = signal.size
    high_bin_float = n_samples * high_hz / fs
    high_bin = int(round(high_bin_float))
    if not np.isclose(high_bin_float, high_bin):
        raise ValueError("上下文长度必须使 1 Hz 对应整数 FFT bin")
    spectrum = 2.0 * np.abs(np.fft.fft(signal)) / n_samples
    return spectrum[: high_bin + 1]


def _upper_envelope(spectrum: np.ndarray) -> np.ndarray:
    peaks, _ = find_peaks(spectrum)
    distances = np.diff(peaks)
    mirror = max(3, int(np.min(distances))) if distances.size else 3
    half_width = int(np.ceil((mirror - 1) / 2.0))
    padded = np.pad(spectrum, (half_width, half_width), mode="constant")
    width = 2 * half_width + 1
    return np.asarray(
        [np.max(padded[index : index + width]) for index in range(spectrum.size)],
        dtype=np.float64,
    )


def _detect_envelope_flats(upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    next_values = np.concatenate((upper[1:], np.zeros(1)))
    previous_values = np.concatenate((np.zeros(1), upper[:-1]))
    flat = np.flatnonzero((upper == next_values) & (upper != previous_values))
    strict_minima = np.flatnonzero((upper < next_values) & (upper < previous_values))
    flat_and_min = np.sort(np.concatenate((flat, strict_minima)))

    minimum_starts = np.empty(0, dtype=np.int64)
    if flat_and_min.size < 3:
        maximum_starts = flat_and_min.astype(np.int64)
    else:
        peak_positions, _ = find_peaks(upper[flat_and_min])
        maximum_starts = flat_and_min[peak_positions]
        if upper[flat_and_min[0]] > upper[flat_and_min[1]]:
            maximum_starts = np.concatenate(([flat_and_min[0]], maximum_starts))
        if upper[flat_and_min[-1]] > upper[flat_and_min[-2]]:
            maximum_starts = np.concatenate((maximum_starts, [flat_and_min[-1]]))
        minimum_positions, _ = find_peaks(-upper[flat_and_min])
        minimum_starts = np.setdiff1d(flat_and_min[minimum_positions], strict_minima)

    flat_starts = np.sort(np.concatenate((maximum_starts, minimum_starts))).astype(np.int64)
    if flat_starts.size == 0:
        flat_starts = np.asarray([int(np.argmax(upper))], dtype=np.int64)
    flat_ends = np.empty_like(flat_starts)
    for index, start in enumerate(flat_starts):
        limit = int(flat_starts[index + 1]) if index + 1 < flat_starts.size else upper.size
        end = int(start)
        while end + 1 < limit and upper[end + 1] == upper[start]:
            end += 1
        flat_ends[index] = end
    return np.column_stack((flat_starts, flat_ends)), maximum_starts.astype(np.int64)


def _decompose_ewt(signal: np.ndarray, boundary_bins: np.ndarray) -> tuple[np.ndarray, ...]:
    half_length = int(np.round(signal.size / 2.0))
    normalized_boundaries = boundary_bins.astype(np.float64) * np.pi / half_length
    prefix = signal[half_length - 2 :: -1]
    suffix = signal[: signal.size - half_length - 1 : -1]
    extended = np.concatenate((prefix, signal, suffix))
    filters = meyer_filter_bank(normalized_boundaries, extended.size)
    spectrum = np.fft.fft(extended)
    crop_start = half_length - 1
    crop_end = extended.size - half_length
    components = tuple(
        np.fft.ifft(np.conjugate(filter_values) * spectrum).real[crop_start:crop_end]
        for filter_values in filters
    )
    if any(component.size != signal.size or not np.isfinite(component).all() for component in components):
        raise RuntimeError("EEWT component 长度或有限值检查失败")
    return components


def _select_bands(
    signal: np.ndarray,
    upper: np.ndarray,
    flat_labels: np.ndarray,
    maximum_starts: np.ndarray,
    boundary_bins: np.ndarray,
    *,
    fs: float,
    relative_height: float,
    high_band_center_bin_threshold: int,
) -> np.ndarray:
    if maximum_starts.size == 0:
        return np.asarray([0], dtype=np.int64)

    maximum_values = upper[maximum_starts]
    dominant_maximum_position = int(np.argmax(maximum_values))
    amplitude = float(maximum_values[dominant_maximum_position])
    candidate_bands = np.flatnonzero(upper[flat_labels[:, 0]] > relative_height * amplitude)
    if candidate_bands.size == 0:
        dominant_start = int(maximum_starts[dominant_maximum_position])
        candidate_bands = np.flatnonzero(flat_labels[:, 0] == dominant_start)
    if candidate_bands.size == 1:
        return candidate_bands.astype(np.int64)

    centers = np.ceil(np.mean(flat_labels[candidate_bands], axis=1)).astype(np.int64)
    stft_magnitude = _four_block_spectrogram(signal, fs=fs)
    center_values = stft_magnitude[centers, :].T
    center_max_locations = np.argmax(center_values, axis=1)
    center_max_values = np.max(center_values, axis=1)
    center_above_half = center_values > relative_height * center_max_values[:, None]
    center_single = np.sum(center_above_half, axis=1) == 1
    interval_max = int(np.count_nonzero(center_single) > 2)

    peak_amplitudes: list[list[float | None]] = []
    peak_locations: list[list[int | None]] = []
    for frame in range(4):
        frame_amplitudes: list[float | None] = []
        frame_locations: list[int | None] = []
        for band_index in candidate_bands:
            start, stop = _candidate_peak_region(
                int(band_index), boundary_bins=boundary_bins, spectrum_size=upper.size
            )
            region = stft_magnitude[start:stop, frame]
            peak_indices, _ = find_peaks(region)
            if peak_indices.size:
                local_values = region[peak_indices]
                winner = int(np.argmax(local_values))
                frame_amplitudes.append(float(local_values[winner]))
                frame_locations.append(start + int(peak_indices[winner]))
            elif int(band_index) == 0:
                frame_amplitudes.append(float(stft_magnitude[0, frame]))
                frame_locations.append(0)
            else:
                frame_amplitudes.append(None)
                frame_locations.append(None)
        peak_amplitudes.append(frame_amplitudes)
        peak_locations.append(frame_locations)

    nonempty_counts = np.asarray(
        [sum(value is not None for value in frame) for frame in peak_amplitudes], dtype=np.int64
    )
    multi_frames = np.flatnonzero((nonempty_counts != 1) & (nonempty_counts != 0))
    useful = np.zeros((4, candidate_bands.size), dtype=bool)
    uniquely_useful = np.zeros(4, dtype=bool)
    strongest_locations = np.full(4, -1, dtype=np.int64)
    for frame in multi_frames:
        available = [value for value in peak_amplitudes[frame] if value is not None]
        strongest = max(available)
        strongest_candidate = next(
            index for index, value in enumerate(peak_amplitudes[frame]) if value == strongest
        )
        strongest_locations[frame] = int(peak_locations[frame][strongest_candidate])
        useful[frame] = np.asarray(
            [value is not None and value > relative_height * strongest for value in peak_amplitudes[frame]]
        )
        uniquely_useful[frame] = np.count_nonzero(useful[frame]) == 1

    interval_locmax = int(4 - multi_frames.size + np.count_nonzero(uniquely_useful[multi_frames]) > 2)
    flag_locmax = bool(np.all(nonempty_counts == 1) or interval_locmax)
    if interval_max or flag_locmax:
        if interval_max:
            frames = np.arange(4) if np.all(center_single) else np.flatnonzero(center_single)
            selected = np.unique(candidate_bands[center_max_locations[frames]])
            if selected.size == 1 and np.any(~center_single):
                ambiguous = candidate_bands[np.flatnonzero(np.any(center_above_half[~center_single], axis=0))]
                alternatives = np.setdiff1d(ambiguous, selected)
                if alternatives.size:
                    selected = np.unique(np.concatenate((selected, alternatives[:1])))
        else:
            selected_modes: list[int] = []
            for frame in range(4):
                if nonempty_counts[frame] == 1:
                    candidate_position = next(
                        index for index, value in enumerate(peak_locations[frame]) if value is not None
                    )
                    selected_modes.append(
                        _mode_for_peak(int(peak_locations[frame][candidate_position]), boundary_bins)
                    )
                elif uniquely_useful[frame]:
                    selected_modes.append(_mode_for_peak(int(strongest_locations[frame]), boundary_bins))
            selected = np.unique(np.asarray(selected_modes, dtype=np.int64))
            if selected.size == 1 and len(selected_modes) < 4:
                for frame in multi_frames:
                    if uniquely_useful[frame]:
                        continue
                    alternatives = sorted(
                        {
                            _mode_for_peak(int(peak_locations[frame][candidate]), boundary_bins)
                            for candidate in np.flatnonzero(useful[frame])
                            if peak_locations[frame][candidate] is not None
                        }
                        - set(selected.tolist())
                    )
                    if alternatives:
                        selected = np.unique(np.concatenate((selected, alternatives[:1])))
            if selected.size == 0:
                selected = candidate_bands[:1]
        return selected.astype(np.int64)

    high_candidates = np.flatnonzero(centers + 1 > high_band_center_bin_threshold)
    if high_candidates.size:
        # MATLAB 草稿用候选序号加首个候选 band 偏移；候选通常连续，显式写出该语义。
        selected = int(candidate_bands[0] + high_candidates[0])
    else:
        selected = int(candidate_bands[0])
    return np.asarray([selected], dtype=np.int64)


def _candidate_peak_region(
    band_index: int, *, boundary_bins: np.ndarray, spectrum_size: int
) -> tuple[int, int]:
    if band_index == 0:
        return 0, min(spectrum_size, int(boundary_bins[0]) + 2)
    if band_index == boundary_bins.size:
        return max(0, int(boundary_bins[-1]) - 1), spectrum_size
    return (
        max(0, int(boundary_bins[band_index - 1]) - 1),
        min(spectrum_size, int(boundary_bins[band_index]) + 2),
    )


def _mode_for_peak(peak_bin: int, boundary_bins: np.ndarray) -> int:
    return int(np.searchsorted(boundary_bins, peak_bin, side="left"))


def _four_block_spectrogram(signal: np.ndarray, *, fs: float) -> np.ndarray:
    block_length = signal.size // 4
    n_fft_float = (signal.size / fs + 1.0) * fs
    n_fft = int(round(n_fft_float))
    if not np.isclose(n_fft_float, n_fft):
        raise ValueError("IEWT spectrogram n_fft 必须为整数")
    window = hamming(block_length, sym=True)
    frames = signal.reshape(4, block_length)
    spectra = np.stack([np.fft.rfft(frame * window, n=n_fft) for frame in frames], axis=1)
    return np.abs(spectra)


def _meyer_frequency_grid(n_samples: int) -> np.ndarray:
    frequencies = np.fft.fftshift(2.0 * np.pi * np.arange(n_samples) / n_samples)
    frequencies[: n_samples // 2] -= 2.0 * np.pi
    return np.abs(frequencies)


def _meyer_beta(values: np.ndarray) -> np.ndarray:
    x = np.clip(values, 0.0, 1.0)
    return x**4 * (35.0 - 84.0 * x + 70.0 * x**2 - 20.0 * x**3)


def _meyer_scaling(boundary: float, gamma: float, n_samples: int) -> np.ndarray:
    absolute_frequency = _meyer_frequency_grid(n_samples)
    lower = (1.0 - gamma) * boundary
    upper = (1.0 + gamma) * boundary
    values = np.zeros(n_samples, dtype=np.float64)
    values[absolute_frequency <= lower] = 1.0
    transition = (absolute_frequency >= lower) & (absolute_frequency <= upper)
    argument = (absolute_frequency[transition] - lower) / (2.0 * gamma * boundary)
    values[transition] = np.cos(np.pi * _meyer_beta(argument) / 2.0)
    return np.fft.ifftshift(values)


def _meyer_wavelet(lower_boundary: float, upper_boundary: float, gamma: float, n_samples: int) -> np.ndarray:
    absolute_frequency = _meyer_frequency_grid(n_samples)
    lower_minus = (1.0 - gamma) * lower_boundary
    lower_plus = (1.0 + gamma) * lower_boundary
    upper_minus = (1.0 - gamma) * upper_boundary
    upper_plus = (1.0 + gamma) * upper_boundary
    values = np.zeros(n_samples, dtype=np.float64)
    passband = (absolute_frequency >= lower_plus) & (absolute_frequency <= upper_minus)
    values[passband] = 1.0
    upper_transition = (absolute_frequency >= upper_minus) & (absolute_frequency <= upper_plus)
    upper_argument = (
        absolute_frequency[upper_transition] - upper_minus
    ) / (2.0 * gamma * upper_boundary)
    values[upper_transition] = np.cos(np.pi * _meyer_beta(upper_argument) / 2.0)
    lower_transition = (absolute_frequency >= lower_minus) & (absolute_frequency <= lower_plus)
    lower_argument = (
        absolute_frequency[lower_transition] - lower_minus
    ) / (2.0 * gamma * lower_boundary)
    values[lower_transition] = np.sin(np.pi * _meyer_beta(lower_argument) / 2.0)
    return np.fft.ifftshift(values)


def _seconds_to_samples(seconds: float, fs: float, *, name: str) -> int:
    value = float(seconds) * float(fs)
    rounded = int(round(value))
    if rounded <= 0 or not np.isclose(value, rounded):
        raise ValueError(f"{name} 在当前采样率下必须对应正整数点数")
    return rounded


def _as_finite_1d(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} 必须是非空一维数组")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} 包含 NaN/Inf")
    return result


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _metric_metadata(meta: Mapping[str, Any]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key in ("dataset_row_id", "split", "input_set", "samp_id", "coupling_state_id"):
        if key in meta:
            output[key] = _as_numpy(meta[key])
    return output
