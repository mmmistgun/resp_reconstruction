#!/usr/bin/env python3
"""从已保存 GPU 诊断生成只读分析图，不调用模型。"""

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

from resp_train.protocols.respiration import canonicalize_numpy
from resp_train.respdiff_bcg.runtime import sha256, write_json


def main(source, output):
    receipt = json.loads((source / "receipt.json").read_text())
    if receipt["status"] != "complete" or receipt["mode"] != "validation_diagnostic":
        raise ValueError("要求完成的独立诊断")
    for name, expected in receipt["artifacts"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"诊断产物变更：{name}")
    output.mkdir(parents=True, exist_ok=False)
    trace = pd.read_csv(source / "sampling_trace_summary.csv")
    probe = pd.read_csv(source / "timestep_probe_summary.csv")
    metrics = pd.read_csv(source / "selected_parent_metrics.csv")
    stats = pd.read_csv(source / "selected_parent_signal_stats.csv")
    data = {nfe: np.load(source / f"selected_parents_ddim{nfe}.npz") for nfe in (6, 50)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for nfe in (6, 50):
        part = trace[(trace.original_batch == 177) & (trace.nfe == nfe)]
        axes[0].plot(part.t, part.x0_hat_max_abs, marker="." if nfe == 6 else None, label=f"DDIM {nfe}")
    axes[0].invert_xaxis()
    axes[0].set(title="Batch 177: x0 maximum from first step", xlabel="Reverse diffusion t", ylabel="max |x0 estimate|")
    axes[0].legend()
    axes[1].semilogy(probe.t, probe.gradient_norm_ratio, marker="o")
    axes[1].set(title="Fixed forward-noise probe, original batch 0", xlabel="Diffusion timestep t", ylabel="||grad(0.01 FFT)|| / ||grad(noise MSE)||")
    with np.load(source / "probe_t49.npz") as detail:
        true = detail["true_noise"].ravel()
        predicted = detail["predicted_noise"].ravel()
        axes[2].scatter(true[::4], predicted[::4], s=2, alpha=.2)
        # 明确标出该 probe 的极端正噪声点。
        worst = np.argmax(abs(true))
        axes[2].scatter([true[worst]], [predicted[worst]], s=35, color="red", label="largest |noise|")
    axes[2].plot([-5, 5], [-5, 5], color="black", ls="--", lw=1, label="identity")
    axes[2].set(title="t=49: epsilon prediction", xlabel="True epsilon", ylabel="Predicted epsilon", xlim=(-5, 5), ylim=(-5, 5))
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output / "first_step_and_gradient.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
    for r, row_id in enumerate((8025, 10382)):
        index = int(np.flatnonzero(data[6]["dataset_row_id"] == row_id)[0])
        target = data[6]["tho_ref"][index]
        _, target_pi = canonicalize_numpy(target[None], fs=100, low_hz=.05, high_hz=.7)
        peak = int(np.argmax(abs(data[6]["prediction_20hz"][index]))) / 20
        start, stop = max(0, peak - 5), min(180, peak + 5)
        region = slice(int(start * 100), int(stop * 100))
        for nfe in (6, 50):
            low = data[nfe]["prediction_20hz"][index]
            _, projected = canonicalize_numpy(data[nfe]["r_tho_hat"][index][None], fs=100, low_hz=.05, high_hz=.7)
            axes[r, 0].plot(np.arange(3600) / 20, low, lw=.8, label=f"DDIM {nfe}")
            axes[r, 1].plot(np.arange(18000)[region] / 100, projected[0, region], lw=1, label=f"DDIM {nfe} Pi")
        axes[r, 0].plot(np.arange(18000) / 100, target, lw=.8, label="target", alpha=.8)
        axes[r, 1].plot(np.arange(18000)[region] / 100, target_pi[0, region], lw=1, label="target Pi")
        axes[r, 0].set(title=f"row {row_id}: raw waveform", xlabel="Time (s)", ylabel="soft-z amplitude", xlim=(0, 180))
        axes[r, 1].set(title=f"row {row_id}: public Pi near original largest peak", xlabel="Time (s)", ylabel="Pi amplitude", xlim=(start, stop))
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Diagnostic selection: first parent and known outlier; same checkpoint and initial noise")
    fig.savefig(output / "six_vs_fifty_waveforms.png", dpi=170)
    plt.close(fig)

    with np.load(source / "batch_177_inputs.npz") as inputs, np.load(source / "batch_177_ddim6_trace.npz") as trajectory:
        index = tuple(int(v) for v in np.unravel_index(np.argmax(abs(trajectory["states"][-1])), trajectory["states"][-1].shape))
        alpha = float(np.cumprod(1 - np.linspace(.0001, .5, 50))[-1])
        epsilon_required = (float(inputs["initial_noise"][index]) - np.sqrt(alpha) * float(inputs["target"][index])) / np.sqrt(1 - alpha)
        chain = {"global_chunk_index": int(inputs["global_chunk_indices"][index[0]]),
                 "coordinate_in_batch": index, "initial_noise": float(inputs["initial_noise"][index]),
                 "condition": float(inputs["condition"][index]), "target": float(inputs["target"][index]),
                 "epsilon_required_for_target": float(epsilon_required),
                 "epsilon_hat": float(trajectory["epsilon_hat"][0][index]),
                 "x0_first_step": float(trajectory["x0_hat"][0][index]),
                 "x0_error_gain": float(np.sqrt((1 - alpha) / alpha)),
                 "final_chunk_value": float(trajectory["states"][-1][index])}
    columns = ["whole_rr_abs_error_bpm", "local_rr_mae_bpm", "envelope_trajectory_mae",
               "global_envelope_modulation_error", "lag_aware_signed_pcc"]
    summary = {"scope": "13 deliberately selected parents; not full validation or an unbiased estimate",
               "spike_chain": chain,
               "selected_parent_metric_means": metrics.groupby("nfe")[columns].mean().to_dict("index"),
               "selected_parent_amplitude": stats.groupby("nfe").agg(
                   median_max_abs=("max_abs", "median"), largest_max_abs=("max_abs", "max")).to_dict("index")}
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {"source_diagnostic": str(source.resolve()),
               "source_receipt_sha256": sha256(source / "receipt.json"), "script_sha256": sha256(Path(__file__)),
               "artifacts": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})
    for values in data.values():
        values.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
