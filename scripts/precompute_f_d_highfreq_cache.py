"""F-D 高频输入上下文离线缓存。

第一批 F-D 不在训练循环里现场计算 CWT/SST。此脚本按 dataset_row_id 预计算固定
high-frequency context，并保存成现有 ResearchV2WindowDataset 可读取的 npz 结构：
`row_ids` + `sst`。这里的 `sst` 是历史字段名，实际可以是 high-CWT map 或 modulation features。
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import load_config  # noqa: E402
from resp_train.data.research_v2 import ResearchV2WindowDataset, read_research_v2_index  # noqa: E402


BAND_LOW_HZ = 1.0
BAND_HIGH_HZ = 8.0
VOICES_PER_OCTAVE = 12
N_FRAMES = 180
MODULATION_FEATURE_NAMES = [
    "band_energy_1_2hz",
    "band_energy_2_4hz",
    "band_energy_4_8hz",
    "spectral_centroid_1_8hz",
    "spectral_entropy_1_8hz",
    "peak_freq_1_8hz",
    "peak_energy_1_8hz",
    "high_band_quality_ratio",
]
ComputeFn = Callable[..., tuple[np.ndarray, np.ndarray]]


def _target_freqs(low_hz: float, high_hz: float, voices_per_octave: int = VOICES_PER_OCTAVE) -> np.ndarray:
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("low_hz/high_hz 必须满足 0 < low_hz < high_hz")
    bins = int(np.ceil(np.log2(float(high_hz) / float(low_hz)) * int(voices_per_octave)))
    freqs = float(low_hz) * np.power(2.0, np.arange(bins, dtype=np.float64) / int(voices_per_octave))
    return freqs[freqs <= float(high_hz) * (1.0 + 1e-6)].astype(np.float32)


def _pool_time(arr: np.ndarray, n_frames: int) -> np.ndarray:
    if arr.shape[1] == int(n_frames):
        return arr.astype(np.float32, copy=False)
    if arr.shape[1] < int(n_frames):
        old_x = np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float64)
        new_x = np.linspace(0.0, 1.0, int(n_frames), dtype=np.float64)
        return np.stack([np.interp(new_x, old_x, row) for row in arr], axis=0).astype(np.float32)
    chunks = np.array_split(arr, int(n_frames), axis=1)
    return np.stack([chunk.mean(axis=1) for chunk in chunks], axis=1).astype(np.float32)


def compute_high_stft_window(
    sig: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
    win_length: int = 800,
    hop_length: int = 100,
    n_fft: int = 800,
    n_frames: int = N_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """短窗 high-STFT log-magnitude，主要用于离线诊断或缓存 sanity。"""

    signal = np.asarray(sig, dtype=np.float64).reshape(-1)
    win_length = int(win_length)
    hop_length = int(hop_length)
    n_fft = int(n_fft)
    if win_length <= 0 or hop_length <= 0 or n_fft < win_length:
        raise ValueError("win_length/hop_length/n_fft 配置无效")
    starts = np.arange(0, max(1, signal.size - win_length + 1), hop_length, dtype=np.int64)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int64)
    window = np.hanning(win_length).astype(np.float64)
    spectra = []
    for start in starts:
        segment = signal[start : start + win_length]
        if segment.size < win_length:
            segment = np.pad(segment, (0, win_length - segment.size), mode="reflect")
        spectra.append(np.abs(np.fft.rfft(segment * window, n=n_fft)))
    mag = np.stack(spectra, axis=1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs)).astype(np.float32)
    band = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
    if not np.any(band):
        raise ValueError("high-STFT 频带内没有可用频点")
    feat = np.log1p(_pool_time(mag[band], n_frames)).astype(np.float32)
    return feat, freqs[band].astype(np.float32)


def compute_high_cwt_window(
    sig: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
    n_frames: int = N_FRAMES,
    voices_per_octave: int = VOICES_PER_OCTAVE,
    wavelet_cycles: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """简洁 Morlet filter-bank CWT log-magnitude。

    这里不做反变换，只生成固定 high-frequency context。频率网格按 log2 spacing，
    1-8Hz 与 12 voices/octave 时得到 36 个 scale。
    """

    signal = np.asarray(sig, dtype=np.float64).reshape(-1)
    freqs = _target_freqs(low_hz, high_hz, voices_per_octave)
    feats = []
    for freq in freqs:
        sigma_t = float(wavelet_cycles) / (2.0 * np.pi * float(freq))
        radius = max(4, int(np.ceil(4.0 * sigma_t * float(fs))))
        t = np.arange(-radius, radius + 1, dtype=np.float64) / float(fs)
        carrier = np.exp(2j * np.pi * float(freq) * t)
        envelope = np.exp(-0.5 * (t / sigma_t) ** 2)
        kernel = carrier * envelope
        kernel = kernel / (np.sqrt(np.sum(np.abs(kernel) ** 2)) + 1e-12)
        conv = np.convolve(signal, np.conj(kernel[::-1]), mode="same")
        feats.append(np.abs(conv))
    mag = np.stack(feats, axis=0)
    return np.log1p(_pool_time(mag, n_frames)).astype(np.float32), freqs


def compute_modulation_features(logmag: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """从 high-CWT/STFT log-magnitude 中提取 8 个固定调制/质量序列。"""

    feat = np.asarray(logmag, dtype=np.float32)
    freq_arr = np.asarray(freqs, dtype=np.float32).reshape(-1)
    if feat.ndim != 2 or feat.shape[0] != freq_arr.size:
        raise ValueError("logmag 必须是 (freq, time)，且 freqs 长度要匹配")
    energy = np.maximum(feat, 0.0).astype(np.float32)
    total = energy.sum(axis=0).clip(min=1e-6)

    def band_mean(lo: float, hi: float) -> np.ndarray:
        mask = (freq_arr >= lo) & (freq_arr < hi)
        if not np.any(mask):
            return np.zeros(feat.shape[1], dtype=np.float32)
        return energy[mask].mean(axis=0).astype(np.float32)

    band_1_2 = band_mean(1.0, 2.0)
    band_2_4 = band_mean(2.0, 4.0)
    band_4_8 = band_mean(4.0, 8.01)
    centroid = (energy * freq_arr[:, None]).sum(axis=0) / total
    prob = energy / total[None, :]
    entropy = -(prob * np.log(prob.clip(min=1e-8))).sum(axis=0) / np.log(max(2, freq_arr.size))
    peak_idx = np.argmax(energy, axis=0)
    peak_freq = freq_arr[peak_idx]
    peak_energy = energy[peak_idx, np.arange(energy.shape[1])]
    quality = (band_2_4 + band_4_8) / (band_1_2 + 1e-6)
    stacked = np.stack(
        [band_1_2, band_2_4, band_4_8, centroid, entropy, peak_freq, peak_energy, quality],
        axis=0,
    ).astype(np.float32)
    return stacked, list(MODULATION_FEATURE_NAMES)


def _iter_split_rows(cfg, split: str):
    index_path = Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)
    index = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows = index[index["split"] == str(split)].copy()
    dataset = ResearchV2WindowDataset(index_csv_path=index_path, rows=rows, cfg=cfg, preload_windows=False)
    for i in range(len(dataset)):
        item = dataset[i]
        yield int(item["meta"]["dataset_row_id"]), item["x"].view(-1).numpy().astype(np.float32, copy=False)


def _check_freqs(row_id: int, freqs: np.ndarray, freqs_ref: np.ndarray | None) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=np.float32)
    if freqs_ref is None:
        return freqs
    if freqs.shape != freqs_ref.shape or not np.allclose(freqs, freqs_ref, atol=1e-5, rtol=1e-5):
        raise ValueError(f"row {row_id} 频率网格与首窗不一致")
    return freqs_ref


def _feature_from_base(mode: str, base_feat: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if mode == "modulation":
        return compute_modulation_features(base_feat, freqs)
    return np.asarray(base_feat, dtype=np.float32), []


def _precompute_serial(
    cfg,
    *,
    splits: list[str],
    mode: str,
    fs: float,
    low_hz: float,
    high_hz: float,
    n_frames: int,
    workers: int,
    log_every: int,
    compute_fn: ComputeFn,
) -> tuple[list[int], np.ndarray, np.ndarray, list[str]]:
    del workers
    row_ids: list[int] = []
    feats: list[np.ndarray] = []
    seen: set[int] = set()
    freqs_ref: np.ndarray | None = None
    feature_names: list[str] = []
    for split in splits:
        print(f"=== split={split} ===", flush=True)
        count = 0
        for row_id, bcg in _iter_split_rows(cfg, split):
            if row_id in seen:
                continue
            base_feat, freqs = compute_fn(bcg, fs=fs, low_hz=low_hz, high_hz=high_hz, n_frames=n_frames)
            freqs_ref = _check_freqs(row_id, freqs, freqs_ref)
            feat, names = _feature_from_base(mode, base_feat, freqs_ref)
            feature_names = names or feature_names
            seen.add(row_id)
            row_ids.append(row_id)
            feats.append(np.asarray(feat, dtype=np.float32))
            count += 1
            if count % int(log_every) == 0:
                print(f"  {split}: {count} 窗已算", flush=True)
        print(f"  {split}: 共 {count} 窗", flush=True)
    if not feats or freqs_ref is None:
        raise RuntimeError("没有可预计算的窗口")
    return row_ids, np.stack(feats, axis=0).astype(np.float32, copy=False), freqs_ref, feature_names


def _compute_job(job: tuple[int, str, int, np.ndarray, str, float, float, float, int]):
    seq, split, row_id, bcg, mode, fs, low_hz, high_hz, n_frames = job
    base_feat, freqs = compute_high_cwt_window(bcg, fs=fs, low_hz=low_hz, high_hz=high_hz, n_frames=n_frames)
    feat, feature_names = _feature_from_base(mode, base_feat, freqs)
    return seq, split, row_id, feat, freqs, feature_names


def _precompute_parallel(
    cfg,
    *,
    splits: list[str],
    mode: str,
    fs: float,
    low_hz: float,
    high_hz: float,
    n_frames: int,
    workers: int,
    log_every: int,
    mp_start_method: str,
    max_pending: int,
) -> tuple[list[int], np.ndarray, np.ndarray, list[str]]:
    ctx = mp.get_context(mp_start_method)
    pending: set = set()
    results: dict[int, tuple[int, np.ndarray]] = {}
    count_by_split: dict[str, int] = {}
    seen: set[int] = set()
    freqs_ref: np.ndarray | None = None
    feature_names: list[str] = []
    seq = 0

    def consume(done) -> None:
        nonlocal freqs_ref, feature_names
        for future in done:
            done_seq, split, row_id, feat, freqs, names = future.result()
            freqs_ref = _check_freqs(row_id, freqs, freqs_ref)
            feature_names = names or feature_names
            results[int(done_seq)] = (int(row_id), np.asarray(feat, dtype=np.float32))
            count_by_split[split] = count_by_split.get(split, 0) + 1
            if count_by_split[split] % int(log_every) == 0:
                print(f"  {split}: {count_by_split[split]} 窗已算", flush=True)

    with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as pool:
        for split in splits:
            print(f"=== split={split} ===", flush=True)
            count_by_split[split] = 0
            for row_id, bcg in _iter_split_rows(cfg, split):
                if row_id in seen:
                    continue
                seen.add(row_id)
                job = (seq, split, row_id, bcg, mode, fs, low_hz, high_hz, int(n_frames))
                pending.add(pool.submit(_compute_job, job))
                seq += 1
                if len(pending) >= int(max_pending):
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    consume(done)
            if pending:
                done, pending = wait(pending)
                consume(done)
            print(f"  {split}: 共 {count_by_split[split]} 窗", flush=True)
    if not results or freqs_ref is None:
        raise RuntimeError("没有可预计算的窗口")
    ordered = [results[i] for i in sorted(results)]
    return (
        [row_id for row_id, _ in ordered],
        np.stack([feat for _, feat in ordered], axis=0).astype(np.float32, copy=False),
        freqs_ref,
        feature_names,
    )


def precompute_highfreq_features(
    cfg,
    *,
    splits: list[str],
    mode: str,
    fs: float,
    low_hz: float,
    high_hz: float,
    n_frames: int,
    workers: int,
    log_every: int,
    compute_fn: ComputeFn = compute_high_cwt_window,
    mp_start_method: str = "spawn",
    max_pending: int | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray, list[str]]:
    if mode not in {"cwt", "modulation", "high_stft"}:
        raise ValueError("mode 必须是 cwt、modulation 或 high_stft")
    if mode == "high_stft" and compute_fn is compute_high_cwt_window:
        compute_fn = compute_high_stft_window
    if int(workers) < 1:
        raise ValueError("workers 必须 >= 1")
    if int(workers) == 1:
        return _precompute_serial(
            cfg,
            splits=splits,
            mode="cwt" if mode == "high_stft" else mode,
            fs=fs,
            low_hz=low_hz,
            high_hz=high_hz,
            n_frames=n_frames,
            workers=workers,
            log_every=log_every,
            compute_fn=compute_fn,
        )
    if compute_fn is not compute_high_cwt_window:
        raise ValueError("workers > 1 仅支持默认 high-CWT compute_fn")
    return _precompute_parallel(
        cfg,
        splits=splits,
        mode=mode,
        fs=fs,
        low_hz=low_hz,
        high_hz=high_hz,
        n_frames=n_frames,
        workers=workers,
        log_every=log_every,
        mp_start_method=mp_start_method,
        max_pending=max_pending or int(workers) * 2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="F-D high-frequency context cache 预计算")
    parser.add_argument("--config", default="configs/tho_research_v2.yaml")
    parser.add_argument("--out", required=True, help="输出 npz 路径")
    parser.add_argument("--mode", choices=("cwt", "modulation", "high_stft"), default="cwt")
    parser.add_argument("--low-hz", type=float, default=BAND_LOW_HZ)
    parser.add_argument("--high-hz", type=float, default=BAND_HIGH_HZ)
    parser.add_argument("--n-frames", type=int, default=N_FRAMES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mp-start-method", choices=("spawn", "forkserver"), default="spawn")
    parser.add_argument("--max-pending", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    fs = float(cfg.window.target_fs)
    splits = sorted({str(cfg.data.train_split), str(cfg.data.val_split)})
    row_ids, arr, freqs, feature_names = precompute_highfreq_features(
        cfg,
        splits=splits,
        mode=args.mode,
        fs=fs,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        n_frames=args.n_frames,
        workers=args.workers,
        log_every=args.log_every,
        mp_start_method=args.mp_start_method,
        max_pending=args.max_pending,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        row_ids=np.asarray(row_ids, dtype=np.int64),
        sst=arr.astype(np.float32, copy=False),
        freqs=freqs.astype(np.float32, copy=False),
        n_frames=np.int64(args.n_frames),
        low_hz=np.float32(args.low_hz),
        high_hz=np.float32(args.high_hz),
        mode=np.array(args.mode),
        feature_names=np.asarray(feature_names, dtype=str),
        norm=np.array("log1p_n0"),
    )
    print(
        f"saved {out_path}: N={len(row_ids)} shape={arr.shape} mode={args.mode} "
        f"约 {arr.nbytes / 1024 / 1024:.1f} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
