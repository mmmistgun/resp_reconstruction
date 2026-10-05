#!/usr/bin/env python3
"""只读比较原版同 seed 与两项 loss 修复的完整保存 validation 输出。"""

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
import pandas as pd
from scipy.signal import resample_poly

from resp_train.protocols.respiration import canonicalize_numpy
from resp_train.respdiff_bcg.runtime import sha256, write_json
from resp_train.respdiff_bcg.signal import TAPS
from scripts.analyze_respdiff_bcg_outputs import spectral_stats, quantiles

RUNS = {"original": ROOT / "runs/respdiff_bcg_v1/train_b64_seed20260811_r01",
        **{name: ROOT / f"runs/respdiff_bcg_loss_repair_v1/{name}_seed20260811_r01"
           for name in ("snr_weighted_fft", "epsilon_only")}}
METRICS = ("whole_rr_abs_error_bpm", "local_rr_mae_bpm", "envelope_trajectory_mae",
           "global_envelope_modulation_error", "lag_aware_signed_pcc")


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    summaries, frames, metric_rows, spectra, examples, sources = {}, [], [], {}, {}, {}
    common_ids, target100 = None, None
    for objective, directory in RUNS.items():
        receipt = json.loads((directory / "receipt.json").read_text())
        if receipt["status"] != "complete" or receipt["mode"] != "train":
            raise ValueError("要求完成的 train/validation 产物")
        sources[objective] = {"run": str(directory), "receipt_sha256": sha256(directory / "receipt.json")}
        for name in ("validation_waveforms.npz", "validation_per_parent.csv", "validation_summary.csv"):
            if sha256(directory / name) != receipt["artifacts"][name]:
                raise ValueError(f"来源产物 hash 变化：{directory / name}")
        metrics = pd.read_csv(directory / "validation_per_parent.csv")
        with np.load(directory / "validation_waveforms.npz") as data:
            if not np.all(data["split"] == "val"):
                raise ValueError("仅允许 validation 输出")
            ids = data["dataset_row_id"]
            prediction = data["prediction_20hz"].astype(np.float64)
            if objective == "original":
                common_ids, target100 = ids.copy(), data["tho_ref"]
                target20 = resample_poly(target100.astype(np.float64), 1, 5, axis=-1, window=TAPS, padtype="line")
                target_stats, target_power = spectral_stats(target20)
                spectra["target"] = np.median(target_power, axis=0)
            elif not np.array_equal(ids, common_ids) or not np.array_equal(data["tho_ref"], target100):
                raise ValueError("父窗口次序或参考波形与基线不一致")
            if metrics.dataset_row_id.tolist() != ids.tolist():
                raise ValueError("逐父指标身份不一致")
            if not np.isfinite(metrics[list(METRICS)]).all().all():
                raise ValueError("主指标非有限")
            stats, power = spectral_stats(prediction)
            spectra[objective] = np.median(power, axis=0)
            energy = (prediction - prediction.mean(-1, keepdims=True)) ** 2
            top_fraction = np.partition(energy, -36, axis=-1)[:, -36:].sum(-1) / energy.sum(-1)
            maximum = abs(prediction).max(-1)
            ratio = stats["std"] / target_stats["std"]
            summary = {"n_parents": len(ids), "metrics": metrics[list(METRICS)].mean().to_dict(),
                       "max_abs": quantiles(maximum), "global_max_abs": float(maximum.max()),
                       "global_max_abs_row_id": int(ids[maximum.argmax()]),
                       "resp_energy_fraction": quantiles(stats["resp_fraction"]),
                       "above_2hz_energy_fraction": quantiles(stats["above_2hz_fraction"]),
                       "top_1pct_energy_fraction": quantiles(top_fraction), "std_ratio": quantiles(ratio)}
            summaries[objective] = summary
            metric_rows.append({"objective": objective, **summary["metrics"]})
            frames.append(pd.DataFrame({"objective": objective, "dataset_row_id": ids,
                 "max_abs": maximum, "std_ratio": ratio, "resp_energy_fraction": stats["resp_fraction"],
                 "above_2hz_energy_fraction": stats["above_2hz_fraction"], "top_1pct_energy_fraction": top_fraction}))
            # 与原诊断保持同一个已知异常 row，避免按新结果选择好例子。
            index = int(np.flatnonzero(ids == 10382)[0])
            high = data["r_tho_hat"]
            _, pred_pi = canonicalize_numpy(high[index:index + 1], fs=100, low_hz=.05, high_hz=.7)
            _, target_pi = canonicalize_numpy(target100[index:index + 1], fs=100, low_hz=.05, high_hz=.7)
            examples[objective] = {"prediction": prediction[index].copy(), "target": target20[index].copy(),
                "pred_pi": pred_pi[0], "target_pi": target_pi[0],
                "pcc": float(metrics.iloc[index].lag_aware_signed_pcc)}
            summaries[objective]["old_outlier_row10382_max_abs"] = float(maximum[index])
        print(f"已分析 {objective}：{len(ids)} 个父窗口", flush=True)
        del prediction, power, energy, high
    write_json(output / "summary.json", summaries)
    pd.concat(frames, ignore_index=True).to_csv(output / "per_parent_signal_diagnostics.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "validation_comparison.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    for row, (objective, values) in enumerate(examples.items()):
        axes[row, 0].plot(np.arange(3600) / 20, values["prediction"], lw=.8, label="prediction")
        axes[row, 0].plot(np.arange(3600) / 20, values["target"], lw=.8, label="target")
        axes[row, 0].set(title=f"{objective}: raw 20 Hz", xlabel="Time (s)", ylabel="soft-z amplitude", xlim=(0, 180))
        region = slice(0, 3000)
        axes[row, 1].plot(np.arange(3000) / 100, values["pred_pi"][region], lw=1, label="prediction Pi")
        axes[row, 1].plot(np.arange(3000) / 100, values["target_pi"][region], lw=1, label="target Pi")
        axes[row, 1].set(title=f"Public Pi | parent signed PCC={values['pcc']:.3f}", xlabel="Time (s)", ylabel="Pi amplitude", xlim=(0, 30))
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Same seed 20260811, 6400 updates | fixed original outlier row 10382 | axes scale independently")
    fig.savefig(output / "fixed_outlier_waveforms.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    frequency = np.fft.rfftfreq(3600, 1 / 20)
    for objective, power in spectra.items():
        for ax in axes:
            ax.semilogy(frequency[1:], np.maximum(power[1:], 1e-14), label=objective)
    axes[0].set(xlim=(0, 10), xlabel="Frequency (Hz)", ylabel="Median normalized energy per bin")
    axes[1].set(xlim=(0, .8), xlabel="Frequency (Hz)", ylabel="Median normalized energy per bin")
    for ax in axes:
        ax.axvspan(.05, .7, color="green", alpha=.08)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("All 2675 validation parents | same targets, seed, batch order and DDIM6 initial noise")
    fig.savefig(output / "spectrum_comparison.png", dpi=160)
    plt.close(fig)
    write_json(output / "manifest.json", {"sources": sources, "script_sha256": sha256(Path(__file__)),
        "spectral_helper_sha256": sha256(ROOT / "scripts/analyze_respdiff_bcg_outputs.py"),
        "operations": "saved-output CPU comparison; no model forward or test access",
        "artifacts": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
