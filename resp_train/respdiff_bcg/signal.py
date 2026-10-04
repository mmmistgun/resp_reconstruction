"""固定抗混叠、父窗口切块、共享噪声与归一化 Hann overlap-add。"""

import hashlib
import json

import numpy as np
import torch
from scipy.signal import firwin, resample_poly

from resp_train.crd.spectral_ops import fourier_interpolate
from resp_train.respdiff.data import array, float32

TAPS = firwin(255, 9.0, fs=100.0, window=("kaiser", 8.6), scale=True)
TAPS_SHA256 = hashlib.sha256(TAPS.astype("<f8").tobytes()).hexdigest()
STARTS = tuple(range(0, 3601, 300))


def prepare_parent(values):
    values = array("parent100", values, (18000,))
    low = resample_poly(values, 1, 5, window=TAPS, padtype="line")
    return float32("parent20", array("parent20", low, (3600,)))


def chunks_from_parent(values):
    parent = array("parent20", values, (3600,))
    padded = np.pad(parent, (300, 300), mode="reflect")
    return float32("chunks", np.stack([padded[s:s + 600] for s in STARTS])[:, None])


def parent_noise(*, split, row_id, seed):
    if split not in {"train", "val", "synthetic"} or type(seed) is not int or seed < 0:
        raise ValueError("噪声 split/seed 非法")
    identity = json.dumps(["respdiff-bcg-noise-v1", split, int(row_id), seed])
    value = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % (2**63)
    generator = torch.Generator().manual_seed(value)
    # 4200 点是独立高斯场；只对信号反射，噪声不做反射。
    noise = torch.randn(4200, generator=generator, dtype=torch.float32)
    return torch.stack([noise[s:s + 600] for s in STARTS])[:, None]


def restore_parent(chunks, *, chunk_indices):
    if list(chunk_indices) != list(range(13)):
        raise ValueError("要求完整且有序的 0…12 块")
    values = array("prediction chunks", chunks, (13, 1, 600))[:, 0]
    weights = torch.hann_window(600, periodic=True, dtype=torch.float64).numpy()
    summed, denominator = np.zeros(4200), np.zeros(4200)
    for start, chunk in zip(STARTS, values):
        summed[start:start + 600] += chunk * weights
        denominator[start:start + 600] += weights
    if not np.all(denominator[300:3900] > 0):
        raise ValueError("保留区间存在零 OLA 权重")
    low = float32("prediction20", summed[300:3900] / denominator[300:3900])
    high = fourier_interpolate(torch.from_numpy(low)[None], target_length=18000)[0].numpy()
    return low, float32("prediction100", array("prediction100", high, (18000,)))
