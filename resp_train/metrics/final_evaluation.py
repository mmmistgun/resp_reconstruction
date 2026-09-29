"""最终评价 v1：固定 100 Hz / 180 s，资格判断在归一化前完成。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from resp_train.protocols.respiration import canonicalize_numpy, centered_energy_numpy, lag_priority

METRICS = (
    "rr180_mae_bpm", "rr60_nonoverlap_mae_bpm", "aligned_waveform_mae",
    "relative_envelope_mae", "lag_signed_pcc",
)
EPS = 1e-8
LAGS = tuple(lag_priority(30))


class EvaluationFailure(ValueError):
    """预测失败必须由调用者记录 row/checkpoint/seed 后终止该次评价。"""


def project(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size not in (6000, 18000):
        raise EvaluationFailure("波形必须为一维 6000 或 18000 点")
    if not np.isfinite(values).all():
        raise EvaluationFailure("波形包含 NaN/Inf")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        band, normalized = canonicalize_numpy(values, fs=100, low_hz=.05, high_hz=.70, scale_eps=EPS)
    if not np.isfinite(band).all() or not np.isfinite(normalized).all():
        raise EvaluationFailure("FFT 投影/归一化产生非有限数值")
    return band, normalized


def rr_from_power(power: np.ndarray) -> float:
    """6000 点频谱；边界不插值，平坦三点取零偏移，谱峰并列取低频。"""
    if power.shape != (3001,) or not np.isfinite(power).all() or np.any(power < 0):
        raise EvaluationFailure("RR 周期图非法")
    frequencies = np.fft.rfftfreq(6000, d=.01)
    bins = np.flatnonzero((frequencies >= .05) & (frequencies <= .70))
    peak = int(bins[np.argmax(power[bins])])
    delta = 0.0
    if peak not in (bins[0], bins[-1]):
        left, middle, right = np.log(power[peak-1:peak+2] + 1e-12)
        denominator = left - 2 * middle + right
        if denominator != 0:
            delta = float(np.clip(.5 * (left - right) / denominator, -.5, .5))
    return float(60 * (peak + delta) * 100 / 6000)


def estimate_rr(normalized: np.ndarray) -> float:
    if normalized.ndim != 1 or normalized.size not in (6000, 18000) or not np.isfinite(normalized).all():
        raise EvaluationFailure("RR 输入长度或有限性错误")
    starts = (0,) if normalized.size == 6000 else (0, 3000, 6000, 9000, 12000)
    windows = np.stack([normalized[start:start+6000] for start in starts])
    powers = np.abs(np.fft.rfft((windows - windows.mean(axis=1, keepdims=True)) * np.hanning(6000))) ** 2
    return rr_from_power(np.median(powers, axis=0))


def select_nonoverlap_centers(rows: pd.DataFrame) -> np.ndarray:
    """按受试者和整段记录选窗；不按状态/segment 重启，避免边界重复。"""
    required = ("dataset_row_id", "samp_id", "target_source_npz", "window_start_sample", "window_end_sample")
    if any(key not in rows for key in required) or rows[list(required)].isna().any().any():
        raise EvaluationFailure("中央窗口缺少身份/记录/绝对时间")
    if rows.empty or rows.dataset_row_id.duplicated().any():
        raise EvaluationFailure("窗口集合为空或 row ID 重复")
    times = rows[["window_start_sample", "window_end_sample"]].to_numpy(dtype=np.float64)
    if (not np.isfinite(times).all() or np.any(times != np.floor(times))
            or np.any(times[:, 0] < 0) or np.any(times[:, 1] - times[:, 0] != 18000)):
        raise EvaluationFailure("绝对采样时间必须为非负整数且窗口长 18000")
    work = rows[list(required)].copy()
    work["position"] = np.arange(len(rows))
    if work.duplicated(["samp_id", "target_source_npz", "window_start_sample"]).any():
        raise EvaluationFailure("同一记录出现重复绝对窗口")
    selected = np.zeros(len(rows), dtype=bool)
    for _, group in work.groupby(["samp_id", "target_source_npz"], sort=True):
        end = -1
        for row in group.sort_values("window_start_sample", kind="stable").itertuples(index=False):
            start = int(row.window_start_sample) + 6000
            if start >= end:
                selected[row.position] = True
                end = start + 6000
    return selected


def lag_correlation(pred: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    """严格 > 更新保证完全并列时采用 0,-1,+1,...；不作符号翻转。"""
    reference = target[30:-30]
    reference = reference - reference.mean()
    reference_norm = np.sqrt(np.sum(reference ** 2) + EPS)
    best, best_lag = -np.inf, 0
    for lag in LAGS:
        paired = pred[30+lag:17970+lag]
        paired = paired - paired.mean()
        corr = float(np.sum(paired * reference) / (np.sqrt(np.sum(paired ** 2) + EPS) * reference_norm))
        if corr > best:
            best, best_lag = corr, lag
    return best, best_lag


def envelope(normalized: np.ndarray) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(normalized, 1000)[::500]
    values = .5 * np.log(np.mean(windows ** 2, axis=1) + EPS)
    return values - np.median(values)


def evaluate_window(prediction: np.ndarray, reference: np.ndarray, *, center_selected: bool) -> dict:
    """参考资格仅由参考决定；预测退化不改变分母，直接抛出失败。"""
    if np.asarray(prediction).shape != (18000,) or np.asarray(reference).shape != (18000,):
        raise EvaluationFailure("完整波形必须恰好 18000 点")
    pb, px = project(prediction)
    tb, tx = project(reference)
    if centered_energy_numpy(pb) <= EPS:
        raise EvaluationFailure("prediction: 完整 180 s 频带能量不超过 1e-8")
    # 公共配对片段的动态性在未归一化频带波形上检查。
    for lag in LAGS:
        if centered_energy_numpy(pb[30+lag:17970+lag]) <= EPS:
            raise EvaluationFailure(f"prediction: PCC 配对区间频带能量不超过 1e-8 (lag={lag})")
    whole_ok = centered_energy_numpy(tb) > EPS
    joint_ok = whole_ok and centered_energy_numpy(tb[30:-30]) > EPS
    result = {key: None for key in METRICS}
    result.update(center_selected=bool(center_selected), rr180_target_eligible=bool(whole_ok),
                  rr60_target_eligible=False, waveform_target_eligible=bool(joint_ok),
                  envelope_target_eligible=bool(whole_ok), pcc_target_eligible=bool(joint_ok),
                  best_lag_samples=None, rr180_pred_bpm=None, rr180_target_bpm=None,
                  rr60_pred_bpm=None, rr60_target_bpm=None)
    if whole_ok:
        result["rr180_pred_bpm"], result["rr180_target_bpm"] = estimate_rr(px), estimate_rr(tx)
        result[METRICS[0]] = abs(result["rr180_pred_bpm"] - result["rr180_target_bpm"])
        result[METRICS[3]] = float(np.mean(np.abs(envelope(px) - envelope(tx))))
    if joint_ok:
        corr, lag = lag_correlation(px, tx)
        result[METRICS[4]], result["best_lag_samples"] = corr, lag
        result[METRICS[2]] = float(np.mean(np.abs(px[30+lag:17970+lag] - tx[30:-30])))
    if center_selected:
        pcb, pcx = project(np.asarray(prediction)[6000:12000])
        tcb, tcx = project(np.asarray(reference)[6000:12000])
        if centered_energy_numpy(pcb) <= EPS:
            raise EvaluationFailure("prediction: 独立中央 60 s 频带能量不超过 1e-8")
        center_ok = centered_energy_numpy(tcb) > EPS
        result["rr60_target_eligible"] = bool(center_ok)
        if center_ok:
            result["rr60_pred_bpm"], result["rr60_target_bpm"] = estimate_rr(pcx), estimate_rr(tcx)
            result[METRICS[1]] = abs(result["rr60_pred_bpm"] - result["rr60_target_bpm"])
    for key, value in result.items():
        if value is not None and not np.isfinite(value):
            raise EvaluationFailure(f"指标非有限: {key}")
    return result


def summarize_windows(records: list[dict]) -> dict:
    """按显式资格聚合，绝不以 dropna/nanmean 隐式缩小评价集合。"""
    if not records:
        raise EvaluationFailure("评价集合为空")
    flags = ("rr180_target_eligible", "rr60_target_eligible", "waveform_target_eligible",
             "envelope_target_eligible", "pcc_target_eligible")
    result = {}
    for metric, flag in zip(METRICS, flags, strict=True):
        values = []
        for row in records:
            eligible = bool(row[flag])
            if metric == METRICS[1]:
                eligible = eligible and bool(row["center_selected"])
            value = row[metric]
            if eligible:
                if value is None or not np.isfinite(value):
                    raise EvaluationFailure(f"有效窗口缺失有限指标: {metric}")
                values.append(value)
            elif value is not None:
                raise EvaluationFailure(f"无资格窗口含指标: {metric}")
        if not values:
            raise EvaluationFailure(f"指标有效分母为零: {metric}")
        result[metric] = float(np.mean(values))
        result[metric + "_n"] = len(values)
    return result


def summarize_seeds(summaries: list[dict], expected_seeds: tuple[int, ...]) -> dict:
    if len(summaries) != len(expected_seeds) or sorted(s["seed"] for s in summaries) != sorted(expected_seeds):
        raise EvaluationFailure("训练 seed 不完整或重复")
    if len(expected_seeds) not in (1, 3):
        raise EvaluationFailure("仅支持确定性单结果或三个训练 seed")
    result = {}
    for key in METRICS:
        if len({s[key + "_n"] for s in summaries}) != 1:
            raise EvaluationFailure(f"seed 间分母不一致: {key}")
        values = np.asarray([s[key] for s in summaries])
        if not np.isfinite(values).all():
            raise EvaluationFailure(f"seed 汇总含非有限值: {key}")
        result[key] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) == 3 else None,
                       "n_per_seed": summaries[0][key + "_n"]}
    return result
