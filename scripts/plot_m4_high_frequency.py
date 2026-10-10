"""从已完成的 M4 机制产物生成 PNG/SVG 和完整案例图册；不运行模型。"""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from scripts import run_m4_high_frequency as run


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_effects(subjects, output):
    labels = ["Whole RR error (bpm)", "Local RR error (bpm)", "Envelope trajectory MAE",
              "Global envelope error", "PCC decrease"]
    conditions = [c for c in run.core.CONDITIONS if not c.startswith("FULL")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for ax, metric, label in zip(axes.flat, run.sf.PRIMARY, labels):
        data = subjects[subjects.metric == metric]
        means = data.groupby(["condition", "samp_id"], sort=False).deterioration.mean()
        for j, condition in enumerate(conditions):
            values = means.loc[condition]
            offset = np.linspace(-.16, .16, len(values))
            color = "#0072B2" if condition.endswith("NAT") else "#D55E00"
            ax.scatter(j + offset, values, s=22, alpha=.8, color=color)
            ax.plot([j-.22, j+.22], [values.mean()] * 2, color="black", lw=2)
        ax.axhline(0, color="gray", lw=.8)
        ax.set_xticks(range(len(conditions)), [c.replace("H_", "").replace("__", "\n") for c in conditions], fontsize=8)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.15)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.02, .9, "Each dot: one subject, mean of 3 seeds\nBlack bar: equal-subject mean\nPositive: worse than same-batch FULL\nBlue: natural GN; orange: fixed GN\nNo population confidence interval implied",
                       va="top", fontsize=11, linespacing=1.7)
    save(fig, output / "subject_paired_effects")


def plot_signals(source, output):
    with np.load(source / "modulation_psd.npz", allow_pickle=False) as z:
        count, sums = z["valid_count"], z["psd_sum"]
        frequency, carrier = z["modulation_hz"], z["carrier_hz"]
        supported = count > 0
        subject_psd = np.full_like(sums, np.nan)
        np.divide(sums, count[..., None], out=subject_psd, where=supported[..., None])
        scale_subjects = supported.sum(0)
        mean = np.full(sums.shape[1:], np.nan)
        np.divide(np.nansum(subject_psd, axis=0), scale_subjects[:, None], out=mean,
                  where=scale_subjects[:, None] > 0)
        pd.DataFrame({"carrier_hz": carrier, "valid_subjects": scale_subjects,
                      "valid_windows": count.sum(0), "total_windows": int(z["window_count"].sum())}).to_csv(
            output / "modulation_coverage.csv", index=False)
    signals = pd.read_csv(source / "signal_association.csv")
    if not signals.valid.isin([True, False]).all() or not np.array_equal(np.isfinite(signals.signed_r), signals.valid):
        raise ValueError("信号相关 finite/资格不符")
    pivot = signals.pivot(index=["dataset_row_id", "samp_id"], columns="transform", values="signed_r")
    paired_valid = pivot.notna().all(axis=1)
    coverage = pd.DataFrame({"valid": paired_valid}).groupby("samp_id").agg(eligible=("valid", "sum"), windows=("valid", "size"))
    coverage.to_csv(output / "association_coverage.csv")
    if (coverage.eligible == 0).any():
        raise ValueError("存在无有效配对相关的受试者，须检查信号产物")
    paired = pivot.loc[paired_valid].copy()
    paired["SHIFT mean"] = paired[["SHIFT_1", "SHIFT_2", "SHIFT_3"]].mean(axis=1)
    subject = paired.groupby("samp_id")[["FULL", "SHIFT mean"]].mean()
    subject.to_csv(output / "association_subjects.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    image = axes[0].pcolormesh(frequency, carrier, mean, shading="nearest", cmap="viridis")
    axes[0].set(xlim=(0, .8), yscale="log", xlabel="Modulation frequency (Hz)", ylabel="CWT carrier frequency (Hz)")
    axes[0].axhline(.8, color="white", ls="--", lw=1)
    axes[0].axhline(2, color="white", ls=":", lw=1)
    fig.colorbar(image, ax=axes[0], label="Fraction of non-DC power per bin")
    for subject_id, row in subject.iterrows():
        axes[1].plot([0, 1], row, marker="o", alpha=.75, label=str(subject_id))
    axes[1].set(xticks=[0, 1], xticklabels=["FULL", "Mean of 3 shifts"], ylabel="Signed correlation with THO envelope")
    axes[1].axhline(0, lw=.8, color="gray")
    axes[1].legend(title="Subject", fontsize=8)
    axes[1].set_title(f"Paired valid windows: {int(paired_valid.sum())}/{len(paired_valid)}")
    save(fig, output / "modulation_and_association")


def plot_case(source, row, seed, frequencies, cfg, output):
    selected = "H_SHIFT_1__FIXED"
    with np.load(source / "cases" / f"row_{row}__FULL__NAT.npz", allow_pickle=False) as z:
        full = dict(z)
    with np.load(source / "cases" / f"row_{row}__{selected}.npz", allow_pickle=False) as z:
        shifted = dict(z)
    if not np.array_equal(full["target"], shifted["target"]):
        raise ValueError("案例参考信号变化")
    target, et, target_env = run.canonical_envelopes(full["target"], cfg)
    prediction, _, envelope = run.canonical_envelopes(full["prediction"], cfg)
    changed, _, changed_env = run.canonical_envelopes(shifted["prediction"], cfg)
    fig, axes = plt.subplots(6, 1, figsize=(13, 13), constrained_layout=True)
    times = .245 + np.arange(360) * .5
    vmin, vmax = min(full["w"].min(), shifted["w"].min()), max(full["w"].max(), shifted["w"].max())
    for ax, data, title in zip(axes[:2], (full["w"], shifted["w"]), ("FULL CWT", "High-band time shift; fixed condition GN")):
        image = ax.pcolormesh(times, frequencies, data, shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set(yscale="log", ylabel="Carrier (Hz)", title=title)
        ax.axhline(.8, color="white", ls="--", lw=1)
        fig.colorbar(image, ax=ax, label="log amplitude")
    residual = shifted["residual_delta"].T
    limit = max(float(np.abs(residual).max()), 1e-12)
    im = axes[2].imshow(residual, aspect="auto", origin="lower", extent=(.245, 179.745, -.5, residual.shape[0]-.5),
                         cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[2].set(ylabel="Latent channel", title="Change in additive residual (shift minus FULL)")
    fig.colorbar(im, ax=axes[2], label="Residual difference")
    t = np.arange(len(target)) / 100
    for data, label, color in ((target, "THO", "black"), (prediction, "FULL", "#0072B2"), (changed, "H shift", "#D55E00")):
        axes[3].plot(t, data, label=label, color=color, lw=.85, alpha=.8)
    axes[3].set_ylabel("Canonical waveform")
    axes[3].legend(ncol=3)
    for data, label, color in ((target_env, "THO", "black"), (envelope, "FULL", "#0072B2"), (changed_env, "H shift", "#D55E00")):
        axes[4].plot(et, data, label=label, color=color)
    axes[4].set_ylabel("Centered log-RMS")
    error_delta = np.abs(changed_env-target_env) - np.abs(envelope-target_env)
    axes[5].plot(et, error_delta, color="#D55E00")
    axes[5].axhline(0, color="gray", lw=.8)
    axes[5].set(ylabel="Absolute error change", xlabel="Time (s)", title="Positive: envelope error increased")
    for ax in axes:
        ax.set_xlim(0, 180)
    fig.suptitle(f"M4 / seed {seed} / dataset row {row} / preselected case", fontsize=14)
    name = f"seed_{seed}_row_{row}"
    save(fig, output / name)
    return name


def render(session, output):
    session, output = Path(session).resolve(), Path(output).resolve()
    manifest = run.load_manifest(session)
    summary = run.completed(session, session / "summary")
    if summary is None:
        raise ValueError("缺少完整三 seed 汇总")
    sources = run.read_json(summary / "sources.json")
    for seed in run.core.SEEDS:
        source = Path(sources[str(seed)]["path"])
        actual = run.completed(session, session / "evaluation" / f"seed_{seed}")
        if actual != source or run.sha(source / "receipt.json") != sources[str(seed)]["receipt_sha256"]:
            raise ValueError("图件来源与汇总来源不符")
    output.mkdir(parents=True, exist_ok=False)
    try:
        subjects = pd.read_csv(summary / "paired_subjects.csv")
        plot_effects(subjects, output)
        plot_signals(Path(sources[str(run.core.SEEDS[0])]["path"]), output)
        cases = []
        for seed in run.core.SEEDS:
            source = Path(sources[str(seed)]["path"])
            cfg = OmegaConf.load(source / "evaluation_config.yaml")
            selection = pd.read_csv(source / "case_selection.csv")
            for row in selection.dataset_row_id.astype(int):
                name = plot_case(source, row, seed, manifest["frequencies_hz"], cfg, output)
                cases.append(name)
        summary_table = pd.read_csv(summary / "three_seed_summary.csv").to_html(index=False, float_format=lambda v: f"{v:.6g}")
        links = "\n".join(f'<li><a href="{html.escape(name)}.png">{html.escape(name)}</a> · <a href="{html.escape(name)}.svg">SVG</a></li>' for name in cases)
        page = '<!doctype html><meta charset="utf-8"><title>M4 高频机制</title>'
        page += '<h1>M4 高频时间结构机制</h1><p>全部预选案例；正差表示相对同批 FULL 退化。案例不替代总体统计。</p>'
        page += '<img width="100%" src="subject_paired_effects.png"><img width="100%" src="modulation_and_association.png">'
        page += summary_table + '<h2>每人首／中／末窗口，全部三个 seed</h2><ul>' + links + '</ul>'
        (output / "index.html").write_text(page, encoding="utf-8")
        run.write_json(output / "receipt.json", {"protocol": run.PROTOCOL, "manifest_sha256": run.sha(session / "manifest.json"),
            "summary_receipt_sha256": run.sha(summary / "receipt.json"), "sources": sources,
            "cases": cases, "code_sha256": run.code_identity(),
            "artifacts": {p.name: run.sha(p) for p in output.iterdir() if p.is_file()}})
    except BaseException:
        run.write_json(output / "failed.json", {"traceback": run.traceback.format_exc()})
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(render(args.session, args.output))
