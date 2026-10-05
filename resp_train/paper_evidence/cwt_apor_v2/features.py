"""保留 W0 基准映射的原生 CWT 网格与时间池化。"""
from __future__ import annotations
from dataclasses import asdict
from functools import lru_cache
import numpy as np
from .spec import Arm
from resp_train.crd.tf_v1_features import (
    _require_ssqueezepy_version, morlet_scales_and_frequencies, morlet_target_frequencies,
)


@lru_cache(maxsize=12)
def base_grid(mu: float, voices: int):
    _require_ssqueezepy_version()
    from ssqueezepy import Wavelet
    from ssqueezepy.experimental import freq_to_scale, scale_to_freq
    count = 8 * voices + 1
    wavelet = Wavelet(("morlet", {"mu": mu}), N=18000, dtype="float32")
    if (mu, voices) == (13.4, 12):
        scales, _ = morlet_scales_and_frequencies(12, 97)
    else:
        scales = freq_to_scale(morlet_target_frequencies(voices_per_octave=voices, count=count),
                               wavelet, 18000, fs=100., n_search_scales=max(4096, 20 * count),
                               kind="peak", base=2)[::-1]
    scales = np.asarray(scales, dtype=np.float64)
    frequencies = np.asarray(scale_to_freq(scales, wavelet, 18000, fs=100., padtype="reflect"), dtype=np.float64)
    if scales.shape != (count,) or not np.all(np.diff(scales) > 0) or not np.isfinite(frequencies).all():
        raise ValueError("CWT 基准映射无效")
    scales.setflags(write=False)
    frequencies.setflags(write=False)
    return scales, frequencies


def representation(arm: Arm):
    from ssqueezepy import Wavelet
    from ssqueezepy.experimental import scale_to_freq
    scales, frequencies = base_grid(arm.mu, arm.voices)
    if arm.high_hz > 8:
        # 只延伸高频尺度，已有基准尺度不参与重新搜索。
        steps = np.arange(1, int(np.ceil(arm.voices * np.log2(arm.high_hz / 8))) + 2)
        extra = scales[0] * np.exp2(-steps / arm.voices)
        wavelet = Wavelet(("morlet", {"mu": arm.mu}), N=18000, dtype="float32")
        ef = np.asarray(scale_to_freq(extra[::-1], wavelet, 18000, fs=100., padtype="reflect"), dtype=np.float64)
        keep = (ef > frequencies[0]) & (ef <= arm.high_hz)
        scales = np.concatenate((extra[::-1][keep], scales))
        frequencies = np.concatenate((ef[keep], frequencies))
    if arm.high_hz < 8:
        keep = frequencies <= arm.high_hz
        scales, frequencies = scales[keep], frequencies[keep]
    if arm.high_only:
        keep = (frequencies > .8) & (frequencies <= 8)
        scales, frequencies = scales[keep], frequencies[keep]
    if len(scales) < 1 or not np.all(np.diff(scales) > 0) or not np.isfinite(scales).all():
        raise ValueError("空频带或不合法 scales")
    order = np.argsort(frequencies, kind="stable")
    f = frequencies[order]
    return {"arm": asdict(arm), "scales_cwt_order": scales.tolist(), "frequencies_cwt_order_hz": frequencies.tolist(),
            "frequencies_hz": f.tolist(), "frequency_order": order.tolist(), "shape": [len(scales), arm.frames],
            "duplicate_center_count": int(len(f) - len(np.unique(f))),
            "time_seconds": ((np.arange(arm.frames) * arm.pool_samples + (arm.pool_samples - 1) / 2) / 100).tolist(),
            "feature": "mean_pool_log1p_abs", "padtype": "reflect", "sample_rate": 100,
            "window_samples": 18000, "library": "ssqueezepy==0.6.6"}


def transform(waveform, rep):
    _require_ssqueezepy_version()
    from ssqueezepy import Wavelet, cwt
    from ssqueezepy.experimental import scale_to_freq
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if x.shape != (18000,) or not np.isfinite(x).all():
        raise ValueError("CWT 输入必须为有限的 18000 点波形")
    arm = Arm(**rep["arm"])
    wavelet = Wavelet(("morlet", {"mu": arm.mu}), N=18000, dtype="float32")
    coefficients, returned = cwt(x, wavelet=wavelet, scales=np.array(rep["scales_cwt_order"], dtype=np.float64),
                                 fs=100., padtype="reflect", rpadded=False, vectorized=True,
                                 astensor=False, nan_checks=False)
    actual = np.asarray(scale_to_freq(returned, wavelet, 18000, fs=100., padtype="reflect"), dtype=np.float64)
    if not np.array_equal(actual, rep["frequencies_cwt_order_hz"]):
        raise ValueError("实际 CWT 频率与已冻结网格不一致")
    magnitude = np.log1p(np.abs(np.asarray(coefficients)[rep["frequency_order"]])).astype(np.float32)
    if not np.isfinite(magnitude).all():
        raise FloatingPointError("CWT 包含非有限值")
    result = magnitude.reshape(rep["shape"][0], arm.frames, arm.pool_samples).mean(-1, dtype=np.float32)
    if list(result.shape) != rep["shape"] or not np.isfinite(result).all():
        raise FloatingPointError("CWT 池化 shape/finite 检查失败")
    return np.ascontiguousarray(result)
