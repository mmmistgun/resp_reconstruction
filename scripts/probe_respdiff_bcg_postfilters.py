#!/usr/bin/env python3
"""小规模保存输出滤波诊断；不调用模型，不替换正式 validation 结果。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt
import torch

from resp_train.crd.spectral_ops import fourier_interpolate
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.protocols.respiration import canonicalize_numpy, fft_band_project_numpy
from resp_train.respdiff_bcg.runtime import sha256, write_json
from resp_train.respdiff_bcg.signal import parent_noise

SOURCE = ROOT / "runs/respdiff_bcg_loss_repair_v1/epsilon_only_seed20260811_r01"
METRICS = ["whole_rr_abs_error_bpm", "local_rr_mae_bpm", "envelope_trajectory_mae",
           "global_envelope_modulation_error", "lag_aware_signed_pcc"]


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        analyze(output)
    except BaseException as exc:
        write_json(output / "failure.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


def analyze(output):
    torch.set_num_threads(1)
    receipt = json.loads((SOURCE / "receipt.json").read_text())
    if receipt["status"] != "complete" or receipt["objective"] != "epsilon_only":
        raise ValueError("来源必须为完成的 epsilon-only run")
    for name in ("validation_waveforms.npz", "validation_per_parent.csv", "resolved_config.yaml"):
        if sha256(SOURCE / name) != receipt["artifacts"][name]:
            raise ValueError(f"来源 hash 变化：{name}")
    cfg = OmegaConf.load(SOURCE / "resolved_config.yaml")
    with np.load(SOURCE / "validation_waveforms.npz") as data:
        all_ids = data["dataset_row_id"]
        even = np.unique(np.linspace(0, len(all_ids) - 1, 64, dtype=int))
        stress = np.array([int(np.flatnonzero(all_ids == value)[0]) for value in (10382, 12637)])
        selected = np.unique(np.concatenate([even, stress]))
        if not np.all(data["split"][selected] == "val"):
            raise ValueError("只允许 validation 保存输出")
        low = data["prediction_20hz"][selected].astype(np.float64)
        high = data["r_tho_hat"][selected].astype(np.float64)
        target = data["tho_ref"][selected]
        ids = all_ids[selected]
        subjects = data["samp_id"][selected]
    write_json(output / "plan.json", {"source": str(SOURCE), "evenly_spaced_parent_indices": even.tolist(),
        "stress_parent_indices": stress.tolist(), "selected_parent_indices": selected.tolist(),
        "selection": "64 fixed equally spaced saved positions plus known outliers 10382/12637; report separately",
        "filters": ["saved", "public_band", "butterworth_4th_1hz_zero_phase", "median_3", "median_5"],
        "median_boundary": "reflect", "filter_fs": 20, "same_target_and_public_metrics": True,
        "model_forward_calls": 0, "script_sha256": sha256(Path(__file__))})
    stages = {
        "saved": (low, high),
        "public_band": (fft_band_project_numpy(low, fs=20, low_hz=.05, high_hz=.7),
                        fft_band_project_numpy(high, fs=100, low_hz=.05, high_hz=.7)),
    }
    sos = butter(4, 1.0, fs=20, btype="lowpass", output="sos")
    for name, values in (
        ("butterworth_4th_1hz_zero_phase", sosfiltfilt(sos, low, axis=-1, padtype="odd")),
        ("median_3", median_filter(low, size=(1, 3), mode="reflect")),
        ("median_5", median_filter(low, size=(1, 5), mode="reflect")),
    ):
        # sosfiltfilt 可返回反向 stride 的视图；只整理内存布局，不改变数值。
        values = np.ascontiguousarray(values)
        stages[name] = (values, fourier_interpolate(torch.from_numpy(values), target_length=18000).numpy())
    summaries, details, metric_frames = [], [], {}
    reference = pd.read_csv(SOURCE / "validation_per_parent.csv").iloc[selected]
    for name, (values, interpolated) in stages.items():
        payload = {"r_tho_hat": interpolated, "tho_ref": target, "dataset_row_id": ids,
                   "samp_id": subjects, "split": np.full(len(ids), "val")}
        frame = evaluate_task_predictions(payload, cfg, method=f"postfilter_diagnostic/{name}")
        if not np.isfinite(frame[METRICS]).all().all():
            raise ValueError("诊断主指标非有限")
        if name == "saved":
            np.testing.assert_allclose(frame[METRICS], reference[METRICS], rtol=1e-10, atol=1e-10)
        frame["filter"] = name
        frame["raw_max_abs"] = abs(values).max(-1)
        frame["in_even_subset"] = np.isin(selected, even)
        frame["in_stress_subset"] = np.isin(selected, stress)
        details.append(frame)
        metric_frames[name] = frame
        for subset, mask in (("even64", np.isin(selected, even)), ("stress2", np.isin(selected, stress))):
            rows = frame.loc[mask]
            summaries.append({"filter": name, "subset": subset, "n": len(rows),
                              **rows[METRICS].mean().to_dict(),
                              "median_raw_max_abs": float(rows.raw_max_abs.median()),
                              "largest_raw_max_abs": float(rows.raw_max_abs.max())})
        print(f"已完成 {name}：{len(ids)} 个保存窗口", flush=True)
    difference = metric_frames["public_band"][METRICS].to_numpy() - metric_frames["saved"][METRICS].to_numpy()
    write_json(output / "public_band_idempotence.json", {"max_abs_metric_difference": float(abs(difference).max())})
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(output / "per_parent_metrics.csv", index=False)
    noise_links = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
    for row, row_id in enumerate((10382, 12637)):
        index = int(np.flatnonzero(ids == row_id)[0])
        peak = int(np.argmax(abs(low[index])))
        noise = parent_noise(split="val", row_id=row_id, seed=int(cfg.inference.noise_seed))[1:, 0, :300].numpy().reshape(-1)
        noise_links.append({"row_id": row_id, "peak_index_20hz": peak, "peak_time_sec": peak / 20,
                            "prediction_value": float(low[index, peak]), "initial_noise_at_peak": float(noise[peak])})
        start, stop = max(0, peak / 20 - 5), min(180, peak / 20 + 5)
        low_slice = slice(int(start * 20), int(stop * 20))
        high_slice = slice(int(start * 100), int(stop * 100))
        for name in ("saved", "butterworth_4th_1hz_zero_phase", "median_3", "median_5"):
            values, interpolated = stages[name]
            axes[row, 0].plot(np.arange(3600)[low_slice] / 20, values[index, low_slice], label=name, lw=.9)
            _, canonical = canonicalize_numpy(interpolated[index:index + 1], fs=100, low_hz=.05, high_hz=.7)
            axes[row, 1].plot(np.arange(18000)[high_slice] / 100, canonical[0, high_slice], label=name, lw=.9)
        _, target_pi = canonicalize_numpy(target[index:index + 1], fs=100, low_hz=.05, high_hz=.7)
        axes[row, 1].plot(np.arange(18000)[high_slice] / 100, target_pi[0, high_slice], label="target Pi", color="black", lw=1)
        axes[row, 0].set(title=f"row {row_id}: raw output around original peak", xlabel="Time (s)", ylabel="soft-z amplitude")
        axes[row, 1].set(title=f"row {row_id}: public Pi", xlabel="Time (s)", ylabel="Pi amplitude")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(output / "filter_examples.png", dpi=160)
    plt.close(fig)
    write_json(output / "initial_noise_at_remaining_peaks.json", noise_links)
    write_json(output / "manifest.json", {"source_receipt_sha256": sha256(SOURCE / "receipt.json"),
        "scope": "diagnostic subset; does not replace full validation or select a final postprocessor",
        "artifacts": {path.name: sha256(path) for path in output.iterdir() if path.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
