"""TF-S1 train/validation 信号统计及参考变化分层；不构造神经网络。"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import pearsonr
from resp_train.data.factory import build_window_data
from resp_train.metrics.task import TaskMetricConfig, _log_rms_envelope, _periodogram, _rr_from_power
from resp_train.protocols.respiration import canonicalize_numpy, centered_energy_numpy
from . import artifacts as io
from .data import CacheReader, data_receipt, raw_config, verify_sources
from .spec import config, SEEDS, COUNTS


def reference_curves(raw, cfg):
    p = TaskMetricConfig.from_config(cfg)
    band, canonical = canonicalize_numpy(np.asarray(raw).reshape(1, -1), fs=p.fs,
                                        low_hz=p.band_low_hz, high_hz=p.band_high_hz, scale_eps=p.scale_eps)
    rr, eligible = [], []
    starts = list(range(0, p.length-p.local_rr_window+1, p.local_rr_step))
    for start in starts:
        stop = start+p.local_rr_window
        valid = centered_energy_numpy(band[0, start:stop]) > p.dynamic_eps
        eligible.append(valid)
        rr.append(_rr_from_power(_periodogram(canonical[0, start:stop], p), p.local_rr_window, p) if valid else np.nan)
    envelope = _log_rms_envelope(canonical[0], p)
    return {"canonical": canonical[0], "band": band[0], "rr_bpm": np.array(rr), "rr_eligible": np.array(eligible),
            "rr_time_seconds": (np.array(starts)+(p.local_rr_window-1)/2)/p.fs,
            "envelope": envelope, "envelope_centered": envelope-np.median(envelope),
            "envelope_time_seconds": (np.arange(len(envelope))*p.envelope_step+(p.envelope_window-1)/2)/p.fs}


def reference_changes(raw, cfg):
    curves = reference_curves(raw, cfg)
    adjacent = curves["rr_eligible"][:-1] & curves["rr_eligible"][1:]
    delta = np.abs(np.diff(curves["rr_bpm"]))
    return {"rr_change": float(delta[adjacent].mean()) if adjacent.any() else np.nan,
            "rr_change_defined": bool(adjacent.any()), "rr_pairs": int(adjacent.sum()),
            "envelope_change": float(np.abs(np.diff(curves["envelope"])).mean())}


def association(a, b):
    if a.size < 3 or a.shape != b.shape:
        return np.nan, False
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("关联输入非有限")
    if np.std(a) <= 1e-8 or np.std(b) <= 1e-8:
        return np.nan, False
    value = float(pearsonr(a, b).statistic)
    if not np.isfinite(value):
        raise FloatingPointError("关联计算非有限")
    return value, True


def band_project(x, fs=2):
    frequency = np.fft.rfftfreq(x.shape[-1], 1/fs)
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * ((frequency >= .05) & (frequency <= .70)), n=x.shape[-1], axis=-1)


def run_signals(session, retry=False):
    frozen = io.load_session(session)
    key = io.binding(session, "signals")
    prior = io.completed(session / "signals", key)
    if prior:
        return prior
    data_path, data = data_receipt(session)
    arm = "C_20"
    cache = io.completed(session / "cache" / arm, io.binding(session, "cache", arm=arm,
                         data=io.identity(data_path / "manifest.json")["sha256"]))
    if cache is None:
        raise RuntimeError("signals 需要 C_20 的 train/validation 缓存")
    rep = frozen["representations"][arm]
    calibration = frozen["calibration"]
    from pathlib import Path
    calibration_path = Path(calibration["path"])
    io.verify(calibration_path / "manifest.json", calibration)
    io.verify_stage(calibration_path)
    responses = pd.read_csv(calibration_path / f"{arm}_response.csv")
    cfg = raw_config(config("B", SEEDS[0], session / "unused"))
    p2 = replace(TaskMetricConfig.from_config(cfg), fs=2, length=360, envelope_window=20, envelope_step=10)
    with io.attempt(session / "signals", key, retry) as output:
        verify_sources(data)
        changes, subject_records = [], []
        with (output / "associations.csv").open("x") as associations, (output / "band_power.csv").open("x") as powers:
            first_association, first_power = True, True
            for split, count in COUNTS.items():
                bundle = build_window_data(cfg, split=split, max_windows=None, shuffle=False,
                                          sample_strategy=cfg.data[f"{split}_sample_strategy"], sample_seed=cfg.data[f"{split}_sample_seed"])
                rows = pd.read_csv(data_path / f"{split}_rows.csv")
                if not np.array_equal(rows.dataset_row_id, bundle.rows.dataset_row_id):
                    raise ValueError("signals 样本顺序漂移")
                reader = CacheReader(cache, split, rep, rows.dataset_row_id.to_numpy(np.int64))
                psd = np.lib.format.open_memmap(output / f"{split}_modulation_psd.npy", mode="w+", dtype=np.float32,
                                              shape=(count, rep["shape"][0], 181))
                rng = np.random.Generator(np.random.PCG64(frozen["spec"]["shift_seed"] + (split == "train")))
                shifts = np.stack([rng.choice(np.arange(60, 301), 3, replace=False) for _ in range(count)])
                np.save(output / f"{split}_shift_frames.npy", shifts, allow_pickle=False)
                rows[["dataset_row_id", "samp_id", "split"]].to_csv(output / f"{split}_rows.csv", index=False)
                for i in range(count):
                    item = bundle.dataset[i]
                    raw = item["x"].numpy().reshape(-1)
                    target = item["target"].numpy().reshape(-1)
                    w = reader.get(i, item["meta"]["dataset_row_id"])["w"].numpy()
                    row = rows.iloc[i]
                    identity = {"dataset_row_id": int(row.dataset_row_id), "samp_id": int(row.samp_id), "split": split,
                                "quality": str(row.get("residual_quality_class", "unavailable"))}
                    changes.append({**identity, **reference_changes(target, cfg)})
                    fm, spectrum = periodogram(w.astype(np.float64), fs=2., window="hann", detrend="constant", axis=-1, scaling="density")
                    if not np.isfinite(spectrum).all():
                        raise FloatingPointError("调制谱非有限")
                    psd[i] = spectrum.astype(np.float32)
                    if not np.isfinite(psd[i]).all():
                        raise FloatingPointError("调制谱 float32 存储溢出")
                    fp, raw_psd = periodogram(raw.astype(np.float64), fs=100., window="hann", detrend="constant", scaling="density")
                    if not np.isfinite(raw_psd).all():
                        raise FloatingPointError("BCG 功率谱非有限")
                    power_rows = []
                    for low, high in ((.03,.8),(.8,2),(2,4),(4,8),(8,12),(12,20)):
                        mask = (fp >= low if low == .03 else fp > low) & (fp <= high)
                        power_rows.append({**identity, "low_hz": low, "high_hz": high,
                                           "power": float(raw_psd[mask].sum()*(fp[1]-fp[0]))})
                    pd.DataFrame(power_rows).to_csv(powers, index=False, header=first_power)
                    first_power = False
                    reference = reference_curves(target, cfg)["band"].reshape(360, 50).mean(-1)
                    ref_env = _log_rms_envelope(reference, p2)
                    projected = band_project(w)
                    records = []
                    for scale, frequency in enumerate(rep["frequencies_hz"]):
                        trim = int(np.ceil(float(responses.iloc[scale].energy_99_radius_seconds)*2))
                        support = slice(trim, 360-trim) if trim < 180 else slice(0, 0)
                        env_centers = np.arange(len(ref_env))*10+9.5
                        env_support = (env_centers-9.5 >= trim) & (env_centers+9.5 < 360-trim)
                        for k, offset in enumerate((0, *shifts[i])):
                            values = np.roll(projected[scale], int(offset))
                            r, defined = association(values[support], reference[support])
                            env = _log_rms_envelope(values, p2)
                            e, e_defined = association(env[env_support], ref_env[env_support])
                            records.append({**identity, "scale": scale, "signal_frequency_hz": frequency,
                                            "condition": "FULL" if k == 0 else f"SHIFT_{k}", "offset_frames": int(offset),
                                            "boundary_trim_frames": trim, "support_frames": values[support].size,
                                            "signed_r": r, "absolute_r": abs(r), "defined": defined,
                                            "envelope_r": e, "envelope_defined": e_defined})
                    frame = pd.DataFrame(records)
                    frame.to_csv(associations, index=False, header=first_association)
                    first_association = False
                    subject_records.append(frame)
                    # 每个受试者的全部窗口可能不连续；分块累计后按固定 key 统一汇总。
                    if len(subject_records) >= 64:
                        _append_aggregate(output, subject_records)
                        subject_records.clear()
                psd.flush()
                del psd
        if subject_records:
            _append_aggregate(output, subject_records)
        aggregate = pd.read_csv(output / "association_partials.csv")
        keys = ["split", "samp_id", "quality", "scale", "signal_frequency_hz", "condition"]
        sums = aggregate.groupby(keys, dropna=False)[["signed_sum", "absolute_sum", "envelope_sum", "defined_n", "envelope_n", "window_n"]].sum().reset_index()
        for name, numerator, denominator in (("signed_mean", "signed_sum", "defined_n"), ("absolute_mean", "absolute_sum", "defined_n"), ("envelope_mean", "envelope_sum", "envelope_n")):
            sums[name] = sums[numerator] / sums[denominator].replace(0, np.nan)
        sums.to_csv(output / "association_per_subject.csv", index=False)
        sums.groupby(["split", "quality", "scale", "signal_frequency_hz", "condition"])[["signed_mean", "absolute_mean", "envelope_mean"]].agg(["mean", "count"]).to_csv(output / "association_subject_macro.csv")
        changes = pd.DataFrame(changes)
        changes.to_csv(output / "reference_changes.csv", index=False)
        thresholds = {}
        for column in ("rr_change", "envelope_change"):
            finite = changes.loc[changes.split.eq("train"), column].dropna().to_numpy()
            if not len(finite) or not np.isfinite(finite).all():
                raise ValueError("分层阈值没有有效 training 参考")
            thresholds[column] = {"cuts": np.quantile(finite, [1/3, 2/3], method="linear").tolist(), "defined_n": len(finite)}
        io.write_json(output / "analysis.json", {"thresholds": thresholds, "threshold_source": "train",
                      "modulation_frequency_hz": fm.tolist(), "signal_frequency_hz": rep["frequencies_hz"],
                      "psd": "scipy.periodogram; Hann; detrend=constant; density; 180s; delta_f=1/180Hz",
                      "association": "zero_lag; fixed_per_scale_99percent_energy_support; signed_and_absolute",
                      "nonlinear_preprocessing_comparison": "未纳入；需要提供压缩前信号的独立来源合同",
                      "test_used": False})
        verify_sources(data)
        io.verify_provenance(frozen["provenance"], session)
    return output


def _append_aggregate(output, frames):
    frame = pd.concat(frames, ignore_index=True)
    frame["signed_sum"] = frame.signed_r.fillna(0)
    frame["absolute_sum"] = frame.absolute_r.fillna(0)
    frame["envelope_sum"] = frame.envelope_r.fillna(0)
    frame["defined_n"] = frame.defined.astype(int)
    frame["envelope_n"] = frame.envelope_defined.astype(int)
    frame["window_n"] = 1
    keys = ["split", "samp_id", "quality", "scale", "signal_frequency_hz", "condition"]
    columns = ["signed_sum", "absolute_sum", "envelope_sum", "defined_n", "envelope_n", "window_n"]
    table = frame.groupby(keys, dropna=False)[columns].sum().reset_index()
    path = output / "association_partials.csv"
    table.to_csv(path, mode="a", index=False, header=not path.exists())
