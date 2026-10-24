#!/usr/bin/env python3
"""只读已完成的 validation 输出，生成频谱/幅值和 FFT loss 诊断。"""

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

SEEDS = (20260811, 20260812, 20260813)


def quantiles(values):
    return dict(zip(("p05", "median", "p95"), np.quantile(values, (0.05, 0.5, 0.95)).tolist()))


def spectral_stats(values):
    if values.ndim != 2 or values.shape[1] != 3600 or not np.isfinite(values).all():
        raise ValueError("诊断要求有限的 [N,3600] 20 Hz 父窗口")
    frequency = np.fft.rfftfreq(3600, d=1 / 20)
    power = np.abs(np.fft.rfft(values - values.mean(-1, keepdims=True), norm="ortho")) ** 2
    power[:, 1:-1] *= 2
    energy = power.sum(-1)
    if np.any(energy <= 0):
        raise ValueError("零能量父窗口，不能定义能量占比")
    normalized = power / energy[:, None]
    stats = {"std": values.std(-1), "mean": values.mean(-1),
             "resp_fraction": normalized[:, (frequency >= .05) & (frequency <= .7)].sum(-1),
             "above_2hz_fraction": normalized[:, frequency > 2].sum(-1),
             "peak_hz": frequency[power.argmax(-1)]}
    return stats, normalized


def analyze(run_root, output):
    output.mkdir(parents=True, exist_ok=False)
    summaries, per_parent, histories, spectra, sources = {}, [], {}, {}, {}
    for seed in SEEDS:
        directory = run_root / f"train_b64_seed{seed}_r01"
        receipt = json.loads((directory / "receipt.json").read_text())
        if receipt["status"] != "complete" or receipt["mode"] != "train":
            raise ValueError("只能诊断已完成的 train/validation 产物")
        sources[str(seed)] = {}
        for name in ("validation_waveforms.npz", "validation_per_parent.csv", "history.jsonl"):
            actual = sha256(directory / name)
            if actual != receipt["artifacts"][name]:
                raise ValueError(f"产物 hash 不匹配：{directory / name}")
            sources[str(seed)][name] = actual
        with np.load(directory / "validation_waveforms.npz") as data:
            prediction = data["prediction_20hz"]
            reference100 = data["tho_ref"]
            row_ids = data["dataset_row_id"]
            if not np.all(data["split"] == "val"):
                raise ValueError("只允许 validation 输出")
            target = resample_poly(reference100.astype(np.float64), 1, 5, axis=-1,
                                   window=TAPS, padtype="line")
            ps, pp = spectral_stats(prediction)
            ts, tp = spectral_stats(target)
            ratio = ps["std"] / ts["std"]
            per_parent.append(pd.DataFrame({"seed": seed, "dataset_row_id": row_ids,
                **{f"prediction_{k}": v for k, v in ps.items()},
                **{f"target_{k}": v for k, v in ts.items()}, "std_ratio": ratio}))
            spectra[str(seed)] = np.median(pp, axis=0)
            if seed == SEEDS[0]:
                spectra["target"] = np.median(tp, axis=0)
                # 固定取排序后的首、中、末父窗口，不按结果优劣选择示例。
                selected = [0, len(row_ids) // 2, len(row_ids) - 1]
                saved100 = data["r_tho_hat"]
                _, pred_pi = canonicalize_numpy(saved100[selected], fs=100, low_hz=.05, high_hz=.7)
                _, target_pi = canonicalize_numpy(reference100[selected], fs=100, low_hz=.05, high_hz=.7)
                metrics = pd.read_csv(directory / "validation_per_parent.csv")
                fig, axes = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)
                for index, parent in enumerate(selected):
                    axes[index, 0].plot(np.arange(3600) / 20, prediction[parent], lw=.65, label="prediction")
                    axes[index, 0].plot(np.arange(3600) / 20, target[parent], lw=1, label="target", alpha=.85)
                    axes[index, 0].set_title(f"row={row_ids[parent]} | raw std ratio={ratio[parent]:.2f}")
                    axes[index, 0].set(xlabel="Time (s)", ylabel="soft-z amplitude", xlim=(0, 180))
                    region = slice(7500, 10500)
                    time = np.arange(7500, 10500) / 100
                    axes[index, 1].plot(time, pred_pi[index, region], label="prediction Pi", lw=1)
                    axes[index, 1].plot(time, target_pi[index, region], label="target Pi", lw=1)
                    axes[index, 1].set_title(f"Public Pi, central 30 s | parent signed PCC={metrics.iloc[parent].lag_aware_signed_pcc:.3f}")
                    axes[index, 1].set(xlabel="Time (s)", ylabel="Pi amplitude", xlim=(75, 105))
                for ax in axes.flat:
                    ax.legend(loc="upper right", fontsize=8)
                    ax.grid(alpha=.2)
                fig.suptitle("Saved validation outputs | seed 20260811 | first / middle / last parent")
                fig.savefig(output / "waveform_examples.png", dpi=160)
                plt.close(fig)
                write_json(output / "example_rows.json", {"selection": "first_middle_last_in_saved_order",
                    "seed": seed, "indices": selected, "dataset_row_ids": row_ids[selected].tolist()})
                del saved100
        history = pd.read_json(directory / "history.jsonl", lines=True)
        for key in ("loss", "loss_noise", "loss_fft"):
            if not np.isfinite(history[key]).all():
                raise ValueError(f"history {key} 非有限")
        histories[seed] = history
        losses = {}
        for label, rows in (("first100", history.iloc[:100]), ("last100", history.iloc[-100:]),
                            ("last1000", history.iloc[-1000:])):
            noise, fft = rows.loss_noise.to_numpy(), .01 * rows.loss_fft.to_numpy()
            losses[label] = {"noise_mean": float(noise.mean()), "weighted_fft_mean": float(fft.mean()),
                            "fft_share_of_sum": float(fft.sum() / (noise.sum() + fft.sum()))}
        summaries[str(seed)] = {"n_parents": len(row_ids),
            "prediction": {k: quantiles(v) for k, v in ps.items()},
            "target": {k: quantiles(v) for k, v in ts.items()},
            "std_ratio": quantiles(ratio), "training_loss": losses}
        del prediction, reference100, target, pp, tp
        print(f"已分析 seed={seed} 的 {len(row_ids)} 个保存父窗口", flush=True)

    alpha = np.cumprod(1 - np.linspace(.0001, .5, 50))
    snr = alpha / (1 - alpha)
    schedule = pd.DataFrame({"t": np.arange(50), "alpha_bar": alpha, "snr": snr,
                             "x0_error_gain": 1 / np.sqrt(snr), "fft_to_time_coefficient": .01 / snr})
    schedule.to_csv(output / "schedule.csv", index=False)
    # 反号与循环平移具有相同幅度谱；这是损失性质，不是模型前向实验。
    target = np.sin(2 * np.pi * .25 * np.arange(600) / 20)
    controls = []
    for label, candidate in (("exact", target), ("negative", -target),
                             ("circular_shift_20", np.roll(target, 20)), ("zero", np.zeros_like(target))):
        mse = float(np.mean((candidate - target) ** 2))
        fft = float(np.mean((abs(np.fft.fft(candidate, norm="ortho"))
                            - abs(np.fft.fft(target, norm="ortho"))) ** 2))
        for t in (0, 19, 29, 39, 49):
            controls.append({"candidate": label, "t": t, "time_mse": mse, "fft_magnitude_mse": fft,
                             "epsilon_mse": float(snr[t] * mse), "total_loss": float(snr[t] * mse + .01 * fft)})
    pd.DataFrame(controls).to_csv(output / "synthetic_phase_controls.csv", index=False)
    pd.concat(per_parent, ignore_index=True).to_csv(output / "per_parent_output_stats.csv", index=False)
    write_json(output / "summary.json", summaries)

    frequency = np.fft.rfftfreq(3600, 1 / 20)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for label, power in spectra.items():
        for ax in axes:
            ax.semilogy(frequency[1:], np.maximum(power[1:], 1e-14), label=label)
    axes[0].set(xlim=(0, 10), xlabel="Frequency (Hz)", ylabel="Median normalized energy / bin")
    axes[1].set(xlim=(0, .8), xlabel="Frequency (Hz)", ylabel="Median normalized energy / bin")
    for ax in axes:
        ax.axvspan(.05, .7, color="green", alpha=.08)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("All 2675 parents per seed | centered, unwindowed spectra | band: 0.05-0.70 Hz")
    fig.savefig(output / "output_spectra.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for seed, history in histories.items():
        axes[0].semilogy(history['update'], history.loss_noise.rolling(100, min_periods=100).mean(), label=str(seed))
        axes[1].semilogy(history['update'], (.01 * history.loss_fft).rolling(100, min_periods=100).mean(), label=str(seed))
    axes[0].set(title="Noise MSE | 100-update rolling mean", xlabel="Update", ylabel="Loss")
    axes[1].set(title="0.01 x FFT loss | 100-update rolling mean", xlabel="Update", ylabel="Weighted loss")
    axes[2].semilogy(schedule.t, schedule.fft_to_time_coefficient, color="firebrick")
    axes[2].set(title="Relative coefficient in x0 coordinates", xlabel="Diffusion timestep t", ylabel="0.01 / SNR(t)")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[1].legend()
    fig.savefig(output / "loss_and_schedule.png", dpi=160)
    plt.close(fig)
    write_json(output / "diagnostic_manifest.json", {"source_run_root": str(run_root.resolve()),
        "source_artifacts_sha256": sources, "script_sha256": sha256(Path(__file__)),
        "operations": "saved validation output statistics + saved history + analytic synthetic controls",
        "model_forward_calls": 0, "new_test_access": False,
        "artifacts": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/respdiff_bcg_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_root, args.output)
