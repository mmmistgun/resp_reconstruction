from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .signal import (
    DECIMATION, FIR_BETA, FIR_CUTOFF_HZ, FIR_TAPS, LATENT_LENGTH,
    SAMPLE_RATE, WINDOW_SAMPLES, CenteredFIRDecimate10, fir_coefficients, require_finite,
)

GRID_PATH = Path(__file__).with_name("w0_grid.json")
W0_FREQUENCY_SHA256 = "9fb164e7b09d42b31f7ee57a7d3e966af7adabc6d8c6a0e1eff3090b00b73c0c"
W0_SCALE_SHA256 = "1c6ea711d2e04a98505c49e9b2a1f79b6126e23479023a75c884c443e2eeca2d"
SCALE_COUNT = 97


def spec_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=1)
def load_w0_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取随代码保存的冻结数组；不重新反求尺度，也不访问历史数据缓存。"""
    grid = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    scales = np.asarray(grid["scales"], dtype="<f8")
    mapped = np.asarray(grid["mapped_frequencies_hz"], dtype="<f8")
    output = np.asarray(grid["output_frequencies_hz"], dtype="<f8")
    for name, values in (("scales", scales), ("mapped_frequencies_hz", mapped),
                         ("output_frequencies_hz", output)):
        if values.shape != (SCALE_COUNT,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"冻结 W0 {name} 无效")
        if hashlib.sha256(values.tobytes()).hexdigest() != grid["array_sha256"][name]:
            raise ValueError(f"冻结 W0 {name} 哈希不匹配")
    if not np.all(np.diff(scales) > 0):
        raise ValueError("冻结 W0 scales 必须严格递增")
    if hashlib.sha256(scales.tobytes()).hexdigest() != W0_SCALE_SHA256:
        raise ValueError("冻结 W0 scale 身份不匹配")
    # 低频处可能存在相同的离散中心频率；保留旧实现 stable argsort 的列次序。
    order = np.argsort(mapped, kind="stable")
    if not np.array_equal(mapped[order], output):
        raise ValueError("冻结 W0 scale/frequency 次序不匹配")
    if hashlib.sha256(output.tobytes()).hexdigest() != W0_FREQUENCY_SHA256:
        raise ValueError("冻结 W0 输出频率身份不匹配")
    for values in (scales, mapped, order):
        values.setflags(write=False)
    return scales, mapped, order


def representation_spec() -> dict[str, Any]:
    scales, mapped, order = load_w0_grid()
    spec = {
        "name": "adv-v1-cwt-log1p-fir501-stride10",
        "version": 1,
        "source_key": "bcg_rawish_segment_soft_z_key",
        "input_shape": [1, WINDOW_SAMPLES],
        "sample_rate_hz": SAMPLE_RATE,
        "shape": [SCALE_COUNT, LATENT_LENGTH],
        "dtype": "float32",
        "cwt": {
            "library": "ssqueezepy==0.6.6", "wavelet": "morlet", "mu": 13.4,
            "l1_norm": True, "padtype": "reflect", "rpadded": False,
            "dtype": "float32", "ordering": "stable ascending mapped frequency",
            "scales_sha256": hashlib.sha256(scales.tobytes()).hexdigest(),
            "frequencies_sha256": hashlib.sha256(mapped[order].tobytes()).hexdigest(),
        },
        "amplitude": "log1p(abs(CWT(x))) before temporal filtering",
        "temporal_filter": {
            "type": "centered_kaiser_fir", "taps": FIR_TAPS, "beta": FIR_BETA,
            "cutoff_hz": FIR_CUTOFF_HZ, "boundary": "reflect", "factor": DECIMATION,
            "coefficient_dtype": "float32", "accumulation_dtype": "float32",
            "cuda_conv_fp32_precision": "ieee",
            "coefficient_sha256": hashlib.sha256(fir_coefficients().astype("<f4").tobytes()).hexdigest(),
        },
        "grid": {"origin_seconds": 0.0, "step_seconds": 0.1, "last_seconds": 179.9},
        "normalization": "none",
    }
    return {**spec, "representation_id": spec_digest(spec)}


@dataclass(frozen=True)
class CWTBatch:
    """显式携带表示身份。训练数据适配器仍须按 sample/input identity 配对。"""

    values: torch.Tensor
    representation_id: str

    def to(self, device: torch.device | str) -> CWTBatch:
        return CWTBatch(self.values.to(device=device), self.representation_id)


def extract_cwt_10hz(waveform: np.ndarray | torch.Tensor) -> np.ndarray:
    """对一条输入作固定、离线 CWT 前处理；不读取文件或 target。"""
    if isinstance(waveform, torch.Tensor):
        if waveform.device.type != "cpu" or waveform.requires_grad:
            raise ValueError("固定 CWT 前处理要求 CPU、requires_grad=False 的输入")
        waveform = waveform.detach().numpy()
    values = np.asarray(waveform)
    if values.shape == (1, WINDOW_SAMPLES):
        values = values[0]
    if values.shape != (WINDOW_SAMPLES,) or np.iscomplexobj(values):
        raise ValueError("CWT 输入必须是实数 (18000,) 或 (1,18000)")
    if not np.isfinite(values).all():
        raise FloatingPointError("ADV-v1 CWT 输入包含 NaN/Inf")
    values = np.array(values, dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise FloatingPointError("ADV-v1 CWT 输入转 float32 后溢出")
    version = importlib.metadata.version("ssqueezepy")
    if version != "0.6.6":
        raise RuntimeError(f"ADV-v1 要求 ssqueezepy==0.6.6，当前 {version}")
    from ssqueezepy import Wavelet, cwt
    from ssqueezepy.configs import USE_GPU
    from ssqueezepy.experimental import scale_to_freq

    if USE_GPU():
        raise RuntimeError("固定 CWT 前处理要求 SSQ_GPU=0")
    scales, mapped, order = load_w0_grid()
    wavelet = Wavelet(("morlet", {"mu": 13.4}), N=WINDOW_SAMPLES, dtype="float32")
    coefficients, returned_scales = cwt(
        values, wavelet=wavelet, scales=scales.copy(), fs=SAMPLE_RATE,
        l1_norm=True, padtype="reflect", rpadded=False, vectorized=True,
        astensor=False, nan_checks=False,
    )
    # ssqueezepy 将尺度转为 wavelet dtype，检查实际使用的数组。
    if not np.array_equal(returned_scales, scales.astype(np.float32)):
        raise RuntimeError("CWT 实际 scales 偏离冻结数组")
    actual = np.asarray(scale_to_freq(
        returned_scales, wavelet, WINDOW_SAMPLES, fs=SAMPLE_RATE, padtype="reflect",
    ), dtype=np.float64)
    if not np.array_equal(actual, mapped):
        raise RuntimeError("CWT 实际映射频率偏离冻结数组")
    coefficients = np.asarray(coefficients)
    if coefficients.shape != (SCALE_COUNT, WINDOW_SAMPLES):
        raise RuntimeError(f"CWT coefficients shape 错误: {coefficients.shape}")
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("ADV-v1 CWT coefficients 包含 NaN/Inf")
    magnitude = np.log1p(np.abs(coefficients[order])).astype(np.float32)
    with torch.no_grad():
        result = CenteredFIRDecimate10()(torch.from_numpy(magnitude)[None])[0]
    require_finite("CWT 10 Hz 特征", result)
    return np.ascontiguousarray(result.numpy())


def prepare_cwt_batch(waveforms: torch.Tensor) -> CWTBatch:
    """供合成验证或小批推理使用；正式 cache 需另建 sample/provenance manifest。"""
    if waveforms.ndim != 3 or waveforms.shape[1:] != (1, WINDOW_SAMPLES) or not len(waveforms):
        raise ValueError("CWT batch 期望非空 (B,1,18000)")
    values = np.stack([extract_cwt_10hz(value) for value in waveforms])
    return CWTBatch(torch.from_numpy(values), representation_spec()["representation_id"])
