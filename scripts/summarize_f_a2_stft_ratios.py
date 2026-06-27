from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_paired_f_a2_windows import build_prediction_cache, infer_pair_predictions, load_window_rows


DEFAULT_BANDS = {
    "low": (0.1, 0.7),
    "harm": (0.3, 1.2),
    "high": (1.2, 3.0),
}


def summarize_f_a2_stft_ratios(
    window_list: str | Path,
    *,
    output: str | Path | None = None,
    candidate_labels: list[str] | None = None,
    filter_column: str | None = None,
    sort_by: str | None = None,
    sort_ascending: bool = False,
    max_rows: int | None = None,
    fs: float = 100.0,
    win_length: int = 3000,
    hop_length: int = 500,
    n_fft: int = 3000,
    bands: Mapping[str, tuple[float, float]] | None = None,
    device: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """对 F-A2 paired 窗口补算 target/F0/candidate 的 STFT band-energy ratio。"""

    rows = load_window_rows(
        window_list,
        candidate_labels=candidate_labels,
        filter_column=filter_column,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
        max_rows=max_rows,
    )
    if rows.empty:
        raise ValueError("窗口清单筛选后为空")
    band_map = dict(bands or DEFAULT_BANDS)
    cache = build_prediction_cache(rows, device=device) if use_cache else None
    records = []
    for _, row in rows.iterrows():
        pair = infer_pair_predictions(row, cache=cache, device=device)
        record = _base_record(row)
        features = {
            "target": _stft_band_features(pair["target"], fs=fs, win_length=win_length, hop_length=hop_length, n_fft=n_fft, bands=band_map),
            "baseline": _stft_band_features(pair["baseline"], fs=fs, win_length=win_length, hop_length=hop_length, n_fft=n_fft, bands=band_map),
            "candidate": _stft_band_features(pair["candidate"], fs=fs, win_length=win_length, hop_length=hop_length, n_fft=n_fft, bands=band_map),
        }
        for role, values in features.items():
            for key, value in values.items():
                record[f"{role}_{key}"] = value
        for band in band_map:
            record[f"delta_energy_{band}"] = record[f"candidate_energy_{band}"] - record[f"baseline_energy_{band}"]
        for key in ("log_harm_over_low", "log_high_over_low"):
            record[f"delta_{key}"] = record[f"candidate_{key}"] - record[f"baseline_{key}"]
        records.append(record)
    frame = pd.DataFrame.from_records(records)
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    return frame


def _base_record(row: pd.Series) -> dict:
    keep = [
        "label",
        "seed",
        "dataset_row_id",
        "baseline_run_dir",
        "candidate_run_dir",
        "baseline_rr_peak_band_abs_error",
        "candidate_rr_peak_band_abs_error",
        "delta_rr_peak_band_abs_error",
        "baseline_pred_rr_peak_band_bpm",
        "candidate_pred_rr_peak_band_bpm",
        "target_rr_peak_band_bpm",
        "dirty_easy_lowspec",
        "clean_easy_highspec",
        "baseline_hard",
        "fast_rr",
    ]
    return {key: row[key] for key in keep if key in row}


def _stft_band_features(
    values: np.ndarray,
    *,
    fs: float,
    win_length: int,
    hop_length: int,
    n_fft: int,
    bands: Mapping[str, tuple[float, float]],
    eps: float = 1e-12,
) -> dict[str, float]:
    freqs, power = _stft_power(values, fs=fs, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
    features: dict[str, float] = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= float(low)) & (freqs < float(high))
        energy = float(power[mask].sum(axis=0).mean()) if mask.any() else float("nan")
        features[f"energy_{name}"] = energy
    low_energy = features.get("energy_low", np.nan)
    harm_energy = features.get("energy_harm", np.nan)
    high_energy = features.get("energy_high", np.nan)
    features["log_harm_over_low"] = float(np.log((harm_energy + eps) / (low_energy + eps)))
    features["log_high_over_low"] = float(np.log((high_energy + eps) / (low_energy + eps)))
    return features


def _stft_power(values: np.ndarray, *, fs: float, win_length: int, hop_length: int, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("values 为空，无法计算 STFT")
    nperseg = min(int(win_length), arr.size)
    noverlap = max(0, min(nperseg - 1, nperseg - int(hop_length)))
    nfft = max(int(n_fft), nperseg)
    freqs, _, stft = scipy_signal.stft(
        arr,
        fs=float(fs),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    return freqs, np.abs(stft) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 F-A2 paired 窗口的 STFT band-energy ratio")
    parser.add_argument("--window-list", required=True, help="window_delta/top_degraded/top_improved CSV")
    parser.add_argument("--output", required=True, help="输出 CSV")
    parser.add_argument("--candidate-label", action="append", default=None, help="可重复传入；默认不过滤")
    parser.add_argument("--filter-column", default="", help="只汇总该布尔列为 true 的窗口，例如 dirty_easy_lowspec")
    parser.add_argument("--sort-by", default="", help="筛选后按该列排序，例如 delta_rr_peak_band_abs_error")
    parser.add_argument("--sort-ascending", action="store_true", help="按升序排序；默认降序")
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--win-length", type=int, default=3000)
    parser.add_argument("--hop-length", type=int, default=500)
    parser.add_argument("--n-fft", type=int, default=3000)
    parser.add_argument("--device", default="", help="覆盖 run config 中的 training.device，例如 cuda:0")
    args = parser.parse_args()

    frame = summarize_f_a2_stft_ratios(
        args.window_list,
        output=args.output,
        candidate_labels=args.candidate_label,
        filter_column=args.filter_column or None,
        sort_by=args.sort_by or None,
        sort_ascending=args.sort_ascending,
        max_rows=args.max_rows,
        fs=args.fs,
        win_length=args.win_length,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        device=args.device or None,
    )
    print(f"写出 {args.output} rows={len(frame)}")


if __name__ == "__main__":
    main()
