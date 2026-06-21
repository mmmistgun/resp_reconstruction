from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import load_config
from resp_train.data.factory import build_tho_data
from resp_train.models.stft_branch import _band_indices, _validate_stft_config


def _compute_band_features(
    waveform: torch.Tensor,
    *,
    stft_win: int,
    stft_hop: int,
    band_start: int,
    band_end: int,
) -> torch.Tensor:
    """计算单个窗口的 log-STFT 频带特征，返回形状为 (freq, time)。"""

    signal = waveform.reshape(1, -1).to(dtype=torch.float32)
    window = torch.hann_window(stft_win, dtype=signal.dtype, device=signal.device)
    spectrum = torch.stft(
        signal,
        n_fft=stft_win,
        hop_length=stft_hop,
        win_length=stft_win,
        window=window,
        center=True,
        return_complex=True,
    )
    return torch.log1p(spectrum.abs())[:, band_start:band_end, :].squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="预统计 STFT 频带 N1 鲁棒尺度")
    parser.add_argument("--config", default="configs/tho_small.yaml", help="配置文件路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    parser.add_argument("--high-hz", type=float, required=True, help="频带上界 Hz")
    parser.add_argument("--low-hz", type=float, default=0.05, help="频带下界 Hz")
    parser.add_argument("--stft-win", type=int, default=3000, help="STFT 窗长")
    parser.add_argument("--stft-hop", type=int, default=500, help="STFT hop 长度")
    parser.add_argument("--max-windows", type=int, default=512, help="最多用于统计的训练窗口数")
    parser.add_argument("--output", required=True, help="输出 .npy 路径")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    fs = float(cfg.window.target_fs)
    _validate_stft_config(args.stft_win, args.stft_hop, fs, args.low_hz, args.high_hz)
    band_start, band_end = _band_indices(args.stft_win, fs, args.low_hz, args.high_hz)

    data = build_tho_data(cfg)
    dataset = data.train.dataset
    window_count = min(len(dataset), int(args.max_windows))
    if window_count <= 0:
        raise RuntimeError("没有可用于预统计的训练窗口")

    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for idx in range(window_count):
            item = dataset[idx]
            chunks.append(
                _compute_band_features(
                    torch.as_tensor(item["x"]),
                    stft_win=args.stft_win,
                    stft_hop=args.stft_hop,
                    band_start=band_start,
                    band_end=band_end,
                )
            )

    features = torch.cat(chunks, dim=1)
    q1 = torch.quantile(features, 0.25, dim=1)
    q3 = torch.quantile(features, 0.75, dim=1)
    scale = (q3 - q1).clamp_min(1e-6).cpu().numpy().astype(np.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, scale)
    print(f"保存 STFT band scale: {output} shape={scale.shape}")


if __name__ == "__main__":
    main()
