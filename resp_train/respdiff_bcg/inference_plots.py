"""ε ensemble 预算图；只读取本次已保存的曲线和固定 row。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_tradeoffs(output):
    curve = pd.read_csv(output / "ensemble_curve.csv")
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    metrics = [("whole_rr_abs_error_bpm", "Whole RR MAE (bpm)"),
        ("local_rr_mae_bpm", "Local RR MAE (bpm)"), ("lag_aware_signed_pcc", "Signed PCC"),
        ("envelope_trajectory_mae", "Envelope trajectory MAE"),
        ("global_envelope_modulation_error", "Global modulation error")]
    for axis, (metric, title) in zip(axes.flat, metrics):
        for sampler, rows in curve.groupby("sampler", sort=False):
            axis.plot(rows.N, rows[f"postfiltered_{metric}"], "o-", label=sampler.upper())
        axis.set(xlabel="Independent trajectories N", ylabel=title, xscale="log", xticks=[1, 2, 4, 8, 16])
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.grid(alpha=.2)
    axes.flat[-1].axis("off")
    axes.flat[0].legend()
    figure.suptitle("Postfiltered ensemble metrics on fixed diagnostic subset")
    figure.tight_layout()
    figure.savefig(output / "metrics_vs_N.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, title in zip(axes, ("parent_max_p99", "global_max"),
                                   ("P99 parent maximum", "Global maximum")):
        for sampler, rows in curve.groupby("sampler", sort=False):
            for view, linestyle in (("raw", "--"), ("postfiltered", "-")):
                axis.plot(rows.N, rows[f"{view}_{metric}"], marker="o", linestyle=linestyle,
                          label=f"{sampler.upper()} {view}")
        axis.set(xlabel="Independent trajectories N", ylabel=title)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "max_amplitude_vs_N.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for sampler, rows in curve.groupby("sampler", sort=False):
        axes[0].plot(rows.N, rows.sampling_seconds, "o-", label=sampler.upper())
        axes[1].plot(rows.denoiser_calls_per_chunk, rows.postfiltered_parent_max_p99, "o-", label=sampler.upper())
    axes[0].set(xlabel="Independent trajectories N", ylabel="Cumulative prefix sampling seconds")
    axes[1].set(xlabel="Denoiser calls per chunk (N × NFE)", ylabel="Postfiltered P99 parent maximum")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output / "runtime_vs_N.png", dpi=160)
    plt.close(figure)

    for row_id in (10382, 12226):
        entries = []
        for row in curve.itertuples():
            path = output / f"{row.sampler}_N{row.N}" / f"row_{row_id}_waveforms.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as item:
                    entries.append((row.sampler, row.N, item["prediction_raw"],
                                    item["prediction_postfiltered"], item["tho_ref"]))
        if not entries:
            continue
        np.savez(output / f"row_{row_id}_waveforms.npz", sampler=np.array([item[0] for item in entries]),
                 N=np.array([item[1] for item in entries]), prediction_raw=np.stack([item[2] for item in entries]),
                 prediction_postfiltered=np.stack([item[3] for item in entries]), tho_ref=entries[0][4])
        time = np.arange(18000) / 100
        samplers = list(dict.fromkeys(item[0] for item in entries))
        figure, axes = plt.subplots(len(samplers), 2, figsize=(14, 3.4 * len(samplers)), squeeze=False)
        for index, sampler in enumerate(samplers):
            selected = [item for item in entries if item[0] == sampler]
            for _, count, raw, filtered, target in selected:
                axes[index, 0].plot(time, raw, lw=.65, alpha=.8, label=f"N={count}")
                axes[index, 1].plot(time, filtered, lw=.8, alpha=.8, label=f"N={count}")
            for axis in axes[index]:
                axis.plot(time, target, color="black", lw=.8, label="THO reference")
                axis.set(xlabel="Time (s)", ylabel="Soft-z amplitude")
                axis.legend(fontsize=8, ncol=3)
            axes[index, 0].set_title(f"{sampler.upper()} raw")
            axes[index, 1].set_title(f"{sampler.upper()} parent 1 Hz postfilter")
        figure.suptitle(f"Fixed validation row {row_id}: nested ensemble progression")
        figure.tight_layout()
        figure.savefig(output / f"row_{row_id}_ensemble_progression.png", dpi=160)
        plt.close(figure)
