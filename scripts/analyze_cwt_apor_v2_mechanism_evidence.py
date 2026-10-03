"""既存产物的机制整理：调制谱、FiLM路径、受试者及谱峰竞争诊断。"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import scipy
from scipy.signal import find_peaks
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resp_train.protocols.respiration import fft_band_project_numpy

SESSION = ROOT / "runs/cwt_apor_v2/session_20261001T042244Z_e5e53ad16731"
SIGNALS = ROOT / "runs/cwt_time_frequency_v1/session_20260930T184031Z_812b424a1c92/signals/attempt_20260930T185944Z_900055cebf4a"
DATA = SESSION / "research_test/data/attempt_20261002T045020Z_0be812da89ce"
MECH = SESSION / "research_test/mechanisms_r2/revision_485f7f748f79"
SEEDS = (20260811, 20260812, 20260813)
METRICS = ("envelope_trajectory_mae", "local_rr_mae_bpm")
BANDS = ((.03, .8), (.8, 2), (2, 4), (4, 8), (8, 12), (12, 20))
SOURCES = {}


def register(path):
    path = Path(path)
    if str(path) not in SOURCES:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        SOURCES[str(path)] = {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}
    return path


def read_csv(path):
    return pd.read_csv(register(path))


def read_json(path):
    return json.loads(register(path).read_text())


def only(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f"来源必须唯一：{paths}")
    return paths[0]


def finite(value, label):
    if not np.isfinite(np.asarray(value, dtype=float)).all():
        raise FloatingPointError(f"{label} 包含非有限值")


def check_ids(frame, expected):
    if frame.dataset_row_id.duplicated().any() or not np.array_equal(frame.dataset_row_id, expected):
        raise ValueError("逐窗身份、顺序或分母不一致")


def rms(value, axis=None):
    return np.sqrt(np.mean(np.square(np.asarray(value, dtype=float)), axis=axis))


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=160)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def normalized_psd(value):
    """DC之外逐窗逐尺度归一化；零功率不能用零谱混入平均。"""
    value = np.asarray(value, dtype=float).copy()
    finite(value, "PSD")
    if np.any(value < 0):
        raise ValueError("PSD为负")
    value[..., 0] = 0
    total = value.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("存在零功率PSD，须显式扩展有效分母分析")
    return value / total


def analyze_signals(output):
    meta = read_json(SIGNALS / "analysis.json")
    fc, fm = np.array(meta["signal_frequency_hz"]), np.array(meta["modulation_frequency_hz"])
    masks = {"slow_0_005": (fm > 0) & (fm < .05),
             "resp_005_070": (fm >= .05) & (fm <= .70), "fast_070_1": fm > .70}
    summaries, subjects, maps = [], [], {}
    for split in ("train", "val"):
        rows = read_csv(SIGNALS / f"{split}_rows.csv")
        if rows.dataset_row_id.duplicated().any():
            raise ValueError("TF-S1身份重复")
        psd = np.load(register(SIGNALS / f"{split}_modulation_psd.npy"), mmap_mode="r")
        if psd.shape != (len(rows), len(fc), len(fm)):
            raise ValueError("PSD坐标不匹配")
        means = []
        for subject, group in rows.groupby("samp_id", sort=True):
            accum = np.zeros((len(fc), len(fm)))
            for start in range(0, len(group), 64):
                accum += normalized_psd(psd[group.index[start:start+64]]).sum(axis=0)
            mean = accum / len(group)
            means.append(mean)
            for scale, carrier in enumerate(fc):
                subjects.append({"split": split, "samp_id": subject, "scale": scale,
                                 "carrier_hz": carrier, "window_n": len(group),
                                 **{name: mean[scale, mask].sum() for name, mask in masks.items()}})
        macro = np.mean(means, axis=0)
        maps[split] = macro
        np.save(output / f"{split}_subject_macro_psd_fraction.npy", macro, allow_pickle=False)
        for low, high in BANDS:
            carrier_mask = (fc >= low if low == .03 else fc > low) & (fc <= high)
            summaries.append({"split": split, "carrier_low": low, "carrier_high": high,
                              "subject_n": len(means), "window_n": len(rows), "scale_n": int(carrier_mask.sum()),
                              **{name: macro[carrier_mask][:, mask].sum(axis=1).mean() for name, mask in masks.items()}})
    pd.DataFrame(subjects).to_csv(output / "modulation_per_subject.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "modulation_by_carrier_band.csv", index=False)

    raw = read_csv(SIGNALS / "association_per_subject.csv")
    keys = ["split", "samp_id", "scale", "signal_frequency_hz", "condition"]
    grouped = raw.groupby(keys, dropna=False)[["signed_sum", "absolute_sum", "envelope_sum", "defined_n", "envelope_n", "window_n"]].sum().reset_index()
    records = []
    for metric, numerator, denominator in (("signed", "signed_sum", "defined_n"),
                                            ("absolute", "absolute_sum", "defined_n"),
                                            ("envelope", "envelope_sum", "envelope_n")):
        grouped[metric] = grouped[numerator] / grouped[denominator].replace(0, np.nan)
        for key, part in grouped.groupby(keys[:-1], dropna=False):
            part = part.set_index("condition")
            if set(part.index) != {"FULL", "SHIFT_1", "SHIFT_2", "SHIFT_3"} or len(part) != 4:
                raise ValueError("关联条件缺失或重复")
            # 原support由载频固定；FULL/SHIFT有效性须完全匹配才进行配对。
            if part[denominator].nunique() != 1 or part.window_n.nunique() != 1:
                raise ValueError("关联有效分母在配对条件间不一致")
            n = int(part.loc["FULL", denominator])
            baseline = part.loc["FULL", metric]
            shift = part.loc[["SHIFT_1", "SHIFT_2", "SHIFT_3"], metric].mean() if n else np.nan
            records.append(dict(zip(keys[:-1], key), metric=metric, full=baseline, shift=shift,
                                full_minus_shift=baseline-shift, defined_window_n=n,
                                window_n=int(part.loc["FULL", "window_n"]),
                                defined=n > 0, undefined_reason="" if n else "原固定support内无有效关联"))
    paired = pd.DataFrame(records)
    paired.to_csv(output / "association_paired_subject_scale.csv", index=False)
    macro = paired.groupby(["split", "scale", "signal_frequency_hz", "metric"]).agg(
        full=("full", "mean"), shift=("shift", "mean"), full_minus_shift=("full_minus_shift", "mean"),
        defined_subject_n=("full_minus_shift", "count"), total_subject_n=("samp_id", "size")).reset_index()
    macro.to_csv(output / "association_paired_macro.csv", index=False)
    band_rows = []
    for low, high in BANDS:
        chosen = paired[(paired.signal_frequency_hz >= low if low == .03 else paired.signal_frequency_hz > low) & (paired.signal_frequency_hz <= high)]
        # 先对每名受试者的尺度平均，再对受试者平均；保存有效尺度范围。
        by_subject = chosen.groupby(["split", "metric", "samp_id"]).agg(
            full=("full", "mean"), shift=("shift", "mean"), delta=("full_minus_shift", "mean"), nscale=("full", "count")).reset_index()
        for (split, metric), part in by_subject.groupby(["split", "metric"]):
            band_rows.append({"split": split, "metric": metric, "carrier_low": low, "carrier_high": high,
                              "full": part.full.mean(), "shift": part['shift'].mean(), "delta": part.delta.mean(),
                              "defined_subject_n": part.full.count(), "total_subject_n": len(part),
                              "valid_scale_min": part.nscale.min(), "valid_scale_max": part.nscale.max()})
    pd.DataFrame(band_rows).to_csv(output / "association_by_carrier_band.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    order = np.argsort(fc)
    for col, split in enumerate(("train", "val")):
        heat = axes[0, col].pcolormesh(fm[1:], fc[order], np.log10(np.maximum(maps[split][order, 1:], 1e-7)),
                                     shading="auto", vmin=-4, vmax=-1, cmap="magma")
        axes[0, col].set(yscale="log", xlabel="条件随时间变化的调制频率（Hz）", ylabel="CWT载频（Hz）", title=f"{split}：逐窗归一化、受试者等权PSD")
        axes[0, col].axvline(.05, color="white", lw=.6, ls="--")
        axes[0, col].axvline(.70, color="white", lw=.6, ls="--")
        fig.colorbar(heat, ax=axes[0, col], label="log10 每频点功率比例")
        for metric, label in (("absolute", "|波形相关|"), ("envelope", "包络相关")):
            part = macro[(macro.split == split) & (macro.metric == metric)].sort_values("signal_frequency_hz")
            axes[1, col].plot(part.signal_frequency_hz, part.full_minus_shift, label=f"{label}：FULL − SHIFT均值")
        axes[1, col].axhline(0, color="grey", lw=.7)
        axes[1, col].set(xscale="log", xlabel="CWT载频（Hz）", ylabel="配对相关差（受试者等权）")
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(alpha=.2)
    save_figure(fig, output, "modulation_and_association")
    print("TF-S1完成", flush=True)


def case_arrays(path):
    with np.load(register(path), allow_pickle=False) as source:
        result = {name: source[name] for name in source.files}
    for name, value in result.items():
        if name.endswith("rr_bpm"):
            eligible = result[name.replace("rr_bpm", "rr_eligible")].astype(bool)
            finite(value[eligible], name)
        else:
            finite(value, name)
    return result


def plot_case(cases, row_id, subject, seed, pdf, output, preview):
    names = ("FULL__NAT", "H_SHIFT1__NAT", "H_SHIFT1__FIXED")
    base = cases[names[0]]
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), layout="constrained")
    fc = base["frequency_hz"]
    order = np.argsort(fc)
    for ax, name in zip(axes[0], names[:2]):
        ax.pcolormesh(base["cwt_time_seconds"], fc[order], cases[name]["actual_cwt"][order],
                      shading="auto", cmap="viridis", vmin=np.min(base["actual_cwt"]), vmax=np.max(base["actual_cwt"]))
        ax.set(yscale="log", title=name+" 条件CWT", ylabel="载频（Hz）", xlabel="时间（秒）")
    t = base["latent_time_seconds"]
    for name, color in zip(names, ("#386cb0", "#e07a28", "#7f3b97")):
        c = cases[name]
        label = name.replace("H_SHIFT1", "SHIFT1")
        for ax, field in zip(axes[1], ("gamma_raw", "beta_raw")):
            ax.plot(t, rms(c[field]-base[field], axis=0), label=label, c=color, lw=1)
        axes[2, 0].plot(t, rms(c["Z_prime"]-c["Z"], axis=0), label=label, c=color, lw=1)
        axes[2, 1].plot(t, rms(c["Z_prime"]-base["Z_prime"], axis=0), label=label, c=color, lw=1)
        et = c["prediction_envelope_time_seconds"]
        axes[3, 0].plot(et, c["prediction_envelope_centered"], label=label, c=color, lw=1)
        axes[3, 1].plot(et, np.abs(c["prediction_envelope_centered"]-c["reference_envelope_centered"]), label=label, c=color, lw=1)
    axes[3, 0].plot(et, base["reference_envelope_centered"], color="black", label="参考", lw=1.4)
    for ax, title in zip(axes[1:].flat, ("RMS(γraw − FULL γraw)", "RMS(βraw − FULL βraw)",
                                             "RMS(Z′ − Z)", "RMS(Z′ − FULL Z′)", "中心化log-RMS包络", "包络逐点绝对误差")):
        ax.set(title=title, xlabel="时间（秒）")
        ax.grid(alpha=.15)
    axes[3, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"预定案例：受试者{subject} / row {row_id} / seed {seed}", fontsize=14)
    pdf.savefig(fig)
    if preview:
        fig.savefig(output / f"case_preview_{subject}_{row_id}.png", dpi=140)
    plt.close(fig)


def analyze_cases(output, rows, frames, directories):
    case_ids = read_json(DATA / "cases.json")["dataset_row_ids"]
    lookup = rows.set_index("dataset_row_id").samp_id
    records = []
    for seed in SEEDS:
        source, frame = directories[seed], frames[seed].set_index(["dataset_row_id", "condition"])
        with PdfPages(output / f"film_cases_seed_{seed}.pdf") as pdf:
            for case_index, row_id in enumerate(case_ids):
                cases = {condition: case_arrays(source / f"case_{row_id}_{condition}.npz")
                         for condition in frames[seed].condition.unique()}
                base = cases["FULL__NAT"]
                for condition, c in cases.items():
                    for key in ("Z", "reference", "reference_envelope_centered", "frequency_hz", "latent_time_seconds"):
                        if not np.array_equal(c[key], base[key]):
                            raise ValueError(f"公共{key}在案例条件间变化")
                    delta = c["Z_prime"].astype(float)-base["Z_prime"]
                    dg, db = c["g"].astype(float)-base["g"], c["b"].astype(float)-base["b"]
                    algebra = dg*base["Z"]+db
                    env = np.mean(np.abs(c["prediction_envelope_centered"]-c["reference_envelope_centered"]))
                    saved = frame.loc[(row_id, condition), "envelope_trajectory_mae"]
                    if not np.isclose(env, saved, rtol=1e-10, atol=1e-12):
                        raise ValueError("案例包络轨迹与已存逐窗指标不一致")
                    record = {"seed": seed, "dataset_row_id": row_id, "samp_id": int(lookup.loc[row_id]), "condition": condition,
                              "gamma_delta_rms": rms(c["gamma_raw"]-base["gamma_raw"]),
                              "beta_delta_rms": rms(c["beta_raw"]-base["beta_raw"]),
                              "g_delta_rms": rms(dg), "b_delta_rms": rms(db), "z_prime_delta_rms": rms(delta),
                              "modulation_rms": rms(c["Z_prime"]-c["Z"]),
                              "algebra_residual_rms": rms(delta-algebra),
                              "prediction_delta_rms": rms(c["prediction"]-base["prediction"]),
                              "envelope_prediction_delta_mae": np.mean(np.abs(c["prediction_envelope_centered"]-base["prediction_envelope_centered"]))}
                    for metric in METRICS:
                        record[metric] = frame.loc[(row_id, condition), metric]
                        record[metric+"_delta"] = record[metric]-frame.loc[(row_id, "FULL__NAT"), metric]
                    records.append(record)
                plot_case(cases, row_id, int(lookup.loc[row_id]), seed, pdf, output,
                          seed == SEEDS[0] and case_index % 3 == 1)
        print(f"FiLM案例 {seed}：24例完成", flush=True)
    frame = pd.DataFrame(records)
    numeric = frame.select_dtypes(include="number").columns.difference(["seed", "dataset_row_id", "samp_id"])
    finite(frame[numeric], "FiLM案例汇总")
    frame.to_csv(output / "film_case_statistics.csv", index=False)
    means = frame.groupby("condition")[numeric].mean()
    means.to_csv(output / "film_case_condition_means.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    chosen = frame[frame.condition.str.match(r"H_SHIFT[123]__")].copy()
    chosen["gn"] = chosen.condition.str.split("__").str[-1]
    # 每个点为一个预定窗口；先平均三个seed和三个SHIFT。
    dots = chosen.groupby(["samp_id", "dataset_row_id", "gn"])[numeric].mean().reset_index()
    for gn, marker in (("NAT", "o"), ("FIXED", "x")):
        part = dots[dots.gn == gn]
        for ax, x, y in ((axes[0], "g_delta_rms", "z_prime_delta_rms"),
                          (axes[1], "z_prime_delta_rms", "envelope_prediction_delta_mae"),
                          (axes[2], "envelope_prediction_delta_mae", "envelope_trajectory_mae_delta")):
            ax.scatter(part[x], part[y], c=part.samp_id.astype(str).map({str(s): i for i, s in enumerate(sorted(rows.samp_id.unique()))}),
                       cmap="tab10", vmin=0, vmax=9, marker=marker, label=gn, alpha=.8)
            ax.set(xlabel=x, ylabel=y)
            ax.grid(alpha=.2)
    axes[2].axhline(0, color="grey", lw=.7)
    axes[0].legend()
    fig.suptitle("FiLM预定案例：每点一个窗口，三个seed与三个H-SHIFT均值；颜色区分受试者")
    save_figure(fig, output, "film_path_overview")


def peak_competition(raw):
    """20秒局部谱的分离峰竞争；比值不构成双峰标签或正式RR指标。"""
    raw = np.asarray(raw, dtype=float).reshape(-1)
    finite(raw, "谱诊断输入")
    if len(raw) != 18000:
        raise ValueError("谱诊断要求180秒/100Hz")
    band = fft_band_project_numpy(raw, fs=100, low_hz=.05, high_hz=.70)
    chunks = band.reshape(9, 2000)
    power = np.abs(np.fft.rfft((chunks-chunks.mean(axis=1, keepdims=True))*np.hanning(2000), axis=-1))**2
    freq = np.fft.rfftfreq(2000, d=.01)
    mask = (freq >= .05) & (freq <= .70)
    records = []
    for i, p in enumerate(power[:, mask]):
        if np.max(p) <= 0:
            records.append({"local_index": i, "defined": False, "peak_ratio": np.nan, "peak_gap_bpm": np.nan, "peak_count": 0})
            continue
        peaks = find_peaks(np.r_[-np.inf, p, -np.inf])[0]-1
        peaks = peaks[np.argsort(-p[peaks], kind="stable")]
        ratio = p[peaks[1]]/p[peaks[0]] if len(peaks) > 1 else 0.
        gap = 60*abs(freq[mask][peaks[1]]-freq[mask][peaks[0]]) if len(peaks) > 1 else 0.
        records.append({"local_index": i, "defined": True, "peak_ratio": ratio,
                        "peak_gap_bpm": gap, "peak_count": len(peaks)})
    return records


def analyze_subjects(output, rows, frames):
    ids = rows.dataset_row_id.to_numpy()
    pairs = []
    for seed, frame in frames.items():
        baseline = frame[frame.condition == "FULL__NAT"].set_index("dataset_row_id")
        for condition, part in frame.groupby("condition", sort=False):
            check_ids(part, ids)
            if not part.local_rr_target_eligible.all():
                raise ValueError("发现RR不合格目标，需要按原eligible口径显式处理")
            for metric in METRICS:
                finite(part[metric], metric)
            delta = part[["dataset_row_id", "samp_id", *METRICS]].copy()
            delta["seed"], delta["condition"] = seed, condition
            for metric in METRICS:
                delta[metric+"_baseline"] = baseline.loc[ids, metric].to_numpy()
                delta[metric+"_delta"] = delta[metric]-delta[metric+"_baseline"]
            pairs.append(delta)
    pair = pd.concat(pairs, ignore_index=True)
    pair.to_csv(output / "paired_window_metrics.csv", index=False)
    columns = [m+s for m in METRICS for s in ("", "_baseline", "_delta")]
    by_seed = pair.groupby(["seed", "samp_id", "condition"])[columns].mean().reset_index()
    by_seed.to_csv(output / "subject_seed_condition.csv", index=False)
    families = []
    for prefix in ("H", "H2"):
        for gn in ("NAT", "FIXED"):
            selected = pair[pair.condition.isin([f"{prefix}_SHIFT{k}__{gn}" for k in (1, 2, 3)])]
            if len(selected) != len(rows)*9:
                raise ValueError("SHIFT或seed分母错误")
            family = selected.groupby(["samp_id", "dataset_row_id"])[columns].mean().reset_index()
            family["family"] = f"{prefix}_SHIFT_MEAN__{gn}"
            families.append(family)
    windows = pd.concat(families, ignore_index=True)
    windows.to_csv(output / "shift_family_window_effects.csv", index=False)
    summaries = []
    for (family, subject), part in windows.groupby(["family", "samp_id"]):
        rec = {"family": family, "samp_id": subject, "window_n": len(part)}
        for metric in METRICS:
            values = part[metric+"_delta"]
            rec.update({metric+"_baseline": part[metric+"_baseline"].mean(), metric+"_delta": values.mean(),
                        metric+"_worse_fraction": (values > 0).mean(), metric+"_contribution": values.sum()/len(rows)})
        summaries.append(rec)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "subject_shift_effects.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    subjects = sorted(rows.samp_id.unique())
    for j, (family, label, color) in enumerate((("H_SHIFT_MEAN__NAT", "H自然GN", "#d97732"),
                                               ("H_SHIFT_MEAN__FIXED", "H固定GN", "#775398"),
                                               ("H2_SHIFT_MEAN__NAT", "H2自然GN", "#3b8d79"))):
        part = summary[summary.family == family].set_index("samp_id").loc[subjects]
        for ax, metric, title in zip(axes, METRICS, ("包络轨迹MAE变化", "Local RR MAE变化（bpm）")):
            ax.scatter(np.arange(len(subjects))+(j-1)*.18, part[metric+"_delta"], color=color, label=label, s=38)
            ax.set(xticks=np.arange(len(subjects)), xticklabels=[f"{s}\nn={int((rows.samp_id==s).sum())}" for s in subjects], ylabel=title)
            ax.axhline(0, color="grey", lw=.8)
            ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("受试者配对效应：三个seed和三个预设SHIFT均值；正值表示误差增加")
    save_figure(fig, output, "subject_effects")

    diagnostic = windows[windows.family == "H_SHIFT_MEAN__NAT"].drop(columns="family").merge(rows, on=["dataset_row_id", "samp_id"], validate="one_to_one")
    diagnostic = diagnostic.merge(read_csv(DATA / "reference_changes.csv"), on="dataset_row_id", validate="one_to_one")
    spectral_records = []
    reference = np.load(register(DATA / "test_reference.npy"), mmap_mode="r")
    wave_sources = [("reference", 0, reference)]
    for seed in SEEDS:
        path = only((SESSION / f"research_test/evaluation/A0/seed_{seed}").glob("attempt_*/prediction.npy"))
        order = read_csv(path.parent / "prediction_rows.csv")
        check_ids(order, ids)
        wave_sources.append(("prediction_batch128", seed, np.load(register(path), mmap_mode="r")))
    for source, seed, waves in wave_sources:
        if len(waves) != len(rows):
            raise ValueError("保存波形数量不匹配")
        for i, row in rows.iterrows():
            for record in peak_competition(waves[i]):
                spectral_records.append({"source": source, "seed": seed, "dataset_row_id": row.dataset_row_id,
                                         "samp_id": row.samp_id, **record})
        print(f"保存波形谱诊断 {source}/{seed} 完成", flush=True)
    spectral = pd.DataFrame(spectral_records)
    spectral.to_csv(output / "spectral_peak_competition_local.csv", index=False)
    for source, prefix in (("reference", "reference"), ("prediction_batch128", "baseline_prediction")):
        group = spectral[spectral.source == source].groupby("dataset_row_id")
        mean = group.agg(peak_ratio=("peak_ratio", "mean"), peak_gap_bpm=("peak_gap_bpm", "mean"),
                         defined_n=("peak_ratio", "count"), total_n=("peak_ratio", "size")).add_prefix(prefix+"_").reset_index()
        diagnostic = diagnostic.merge(mean, on="dataset_row_id", validate="one_to_one")
    diagnostic.to_csv(output / "window_rr_diagnostics.csv", index=False)
    predictors = ["rr_change", "envelope_change", "reference_peak_ratio", "baseline_prediction_peak_ratio",
                  "transient_motion_ratio", "posture_transition_ratio", "amplitude_reliable_ratio", "rate_confidence_score"]
    correlations, quartiles, quality = [], [], []
    for subject, part in diagnostic.groupby("samp_id"):
        for predictor in predictors:
            for outcome in ("local_rr_mae_bpm_baseline", "local_rr_mae_bpm_delta"):
                valid = np.isfinite(part[predictor]) & np.isfinite(part[outcome])
                x, y = part.loc[valid, predictor], part.loc[valid, outcome]
                defined = len(x) >= 3 and x.nunique() > 1 and y.nunique() > 1
                correlations.append({"samp_id": subject, "predictor": predictor, "outcome": outcome, "window_n": len(part),
                                     "defined_n": len(x), "correlation_defined": defined,
                                     "spearman": spearmanr(x, y).statistic if defined else np.nan,
                                     "undefined_reason": "" if defined else "常量或有效数量不足"})
        for label in ("residual_quality_class", "rate_confidence_level", "waveform_confidence_level"):
            for value, subset in part.groupby(label, dropna=False):
                quality.append({"samp_id": subject, "label": label, "value": value, "window_n": len(subset),
                                "rr_baseline": subset.local_rr_mae_bpm_baseline.mean(), "rr_delta": subset.local_rr_mae_bpm_delta.mean()})
        for predictor in predictors:
            finite(part[predictor], predictor)
            if part[predictor].nunique() < 2:
                continue
            groups = pd.qcut(part[predictor], 4, duplicates="drop")
            for interval, subset in part.groupby(groups, observed=True):
                quartiles.append({"samp_id": subject, "predictor": predictor, "interval": str(interval), "window_n": len(subset),
                                  "predictor_min": subset[predictor].min(), "predictor_max": subset[predictor].max(),
                                  "rr_baseline": subset.local_rr_mae_bpm_baseline.mean(), "rr_delta": subset.local_rr_mae_bpm_delta.mean()})
    pd.DataFrame(correlations).to_csv(output / "within_subject_correlations.csv", index=False)
    pd.DataFrame(quartiles).to_csv(output / "within_subject_quartiles.csv", index=False)
    pd.DataFrame(quality).to_csv(output / "quality_label_diagnostics.csv", index=False)
    diagnostic.groupby("samp_id")[[*predictors, "local_rr_mae_bpm_baseline", "local_rr_mae_bpm_delta"]].mean().to_csv(output / "subject_diagnostic_means.csv")
    part = diagnostic[diagnostic.samp_id == 670].sort_values("dataset_row_id")
    part.sort_values("local_rr_mae_bpm_delta", ascending=False).to_csv(output / "subject670_windows_ranked.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, predictor, label in zip(axes.flat[:3], ("rr_change", "baseline_prediction_peak_ratio", "transient_motion_ratio"),
                                   ("参考相邻局部RR变化（bpm）", "常规A0预测次强/最强谱峰功率比", "已有transient_motion_ratio")):
        points = ax.scatter(part[predictor], part.local_rr_mae_bpm_delta, c=part.local_rr_mae_bpm_baseline, cmap="viridis", s=24)
        ax.set(xlabel=label, ylabel="H-SHIFT RR误差变化（bpm）")
        ax.axhline(0, color="grey", lw=.7)
        fig.colorbar(points, ax=ax, label="FULL基准RR MAE")
    axes[1, 1].plot(np.arange(len(part)), part.local_rr_mae_bpm_baseline, label="FULL/NAT")
    axes[1, 1].plot(np.arange(len(part)), part.local_rr_mae_bpm, label="H-SHIFT均值")
    axes[1, 1].set(xlabel="670的79窗（按dataset_row_id）", ylabel="Local RR MAE（bpm）")
    axes[1, 1].legend()
    fig.suptitle("受试者670：既存窗口诊断（描述性关联，窗口不视为独立样本）")
    save_figure(fig, output, "subject670_diagnostics")


def main():
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    output = ROOT / "runs/cwt_apor_v2/reports" / ("mechanism_evidence_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(output, flush=True)
    receipt = {"status": "running", "output": str(output), "started_at": datetime.now(timezone.utc).isoformat(),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    (output / "started.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    (output / Path(__file__).name).write_bytes(register(Path(__file__)).read_bytes())
    register(ROOT / "resp_train/protocols/respiration.py")
    register(ROOT / "docs/experiments/cwt_apor_v2_mechanism_evidence_analysis_20261003.md")
    try:
        rows = read_csv(DATA / "test_rows.csv")
        if len(rows) != 2310 or rows.samp_id.nunique() != 8 or rows.dataset_row_id.duplicated().any():
            raise ValueError("test身份不完整")
        frames, directories = {}, {}
        for seed in SEEDS:
            result = read_json(only((MECH / f"seed_{seed}").glob("attempt_*/result.json")))
            path = register(result["metrics"]["path"])
            if SOURCES[str(path)]["sha256"] != result["metrics"]["sha256"] or not result["same_batch_passed"]:
                raise ValueError("机制来源身份或同batch验收失败")
            frames[seed], directories[seed] = read_csv(path), Path(result["case_source"])
            if len(frames[seed]) != 18*2310 or frames[seed].condition.nunique() != 18:
                raise ValueError("机制条件不完整")
        analyze_signals(output)
        analyze_cases(output, rows, frames, directories)
        analyze_subjects(output, rows, frames)
        receipt.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(), source_n=len(SOURCES),
                       counts={"test_windows": len(rows), "subjects": 8, "seeds": 3, "conditions": 18, "preselected_cases_per_seed": 24})
    except BaseException as exc:
        receipt.update(status="failed", error=repr(exc), finished_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        (output / "sources.json").write_text(json.dumps(SOURCES, ensure_ascii=False, indent=2)+"\n")
        (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    print("分析完成："+str(output), flush=True)


if __name__ == "__main__":
    main()
