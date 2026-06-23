"""E4 前置验证：STFT / CWT / SST 幅度谱在「呼吸误拣窗口」上的谐波分离对比。

目的（不训练、只读数据）：判断 SST（同步压缩）幅度谱相对当前训练用 STFT 幅度图，
在高误拣窗口上是否真的把「呼吸基频 vs 2× 谐波」分得更开。若分离度明显更好 →
值得进 E4-SST 探针；若看不出 → 直接收，省一整轮训练。

信号特性（决定参数，不拍脑袋）：
- fs=100Hz，窗 18000 样点=180s。
- 呼吸 0.05-0.7Hz = 周期 1.4-20s = 3-42 bpm，极低频长周期。
- 训练 STFT 口径 win=3000(30s)/hop=500(5s)/hann，频率分辨率 0.0333Hz/bin
  —— 慢呼吸区(0.05-0.2Hz)分辨率严重不足，是误拣的疑似来源，也是 CWT/SST 该证明
  多分辨率优势的地方。

用法：
  python scripts/e4_timefreq_compare.py \
      --metrics-run runs/b2_native_dual_8hz/<某 run> \
      --n-hard 6 --n-clean 3 --out-dir runs/e4_timefreq_compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from resp_train.config import load_config  # noqa: E402
from resp_train.data.research_v2 import ResearchV2WindowDataset, read_research_v2_index  # noqa: E402

# ---- 信号与频带常量（来自数据集/训练口径，集中此处便于核对）----
FS = 100.0
RESP_LOW_HZ = 0.05
RESP_HIGH_HZ = 0.7
# STFT：直接沿用训练口径，作为被对标的基线，不可换。
STFT_WIN = 3000
STFT_HOP = 500
# 误拣的本质是基频↔2× 谐波混淆，因此分离度判据围绕「基频带能量 vs 2×谐波带能量」。


def _resp_band_mask(freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    return (freqs >= low) & (freqs <= high)


def compute_stft(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (freqs, times, |STFT| 幅度谱)，口径与训练一致。"""
    from scipy.signal import stft as scipy_stft

    freqs, times, zxx = scipy_stft(
        sig,
        fs=FS,
        window="hann",
        nperseg=STFT_WIN,
        noverlap=STFT_WIN - STFT_HOP,
        boundary="zeros",
    )
    return freqs, times, np.abs(zxx)


def compute_cwt(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """复 Morlet CWT，尺度覆盖呼吸频带，返回 (freqs, |CWT|)。

    参数按信号特性选：呼吸是极低频，需在 0.04-1.0Hz 上密集采样尺度；
    cmor 带宽-中心频率 B-C 取 1.5-1.0，在低频给足频率分辨率、高频保时间分辨率。
    """
    import pywt

    wavelet = "cmor1.5-1.0"
    target_freqs = np.linspace(0.04, 1.0, 128)  # 略宽于呼吸带，便于看 2× 谐波(可达 1.4Hz 外但主集中在此)
    central = pywt.central_frequency(wavelet)
    scales = central * FS / target_freqs
    coefs, freqs = pywt.cwt(sig, scales, wavelet, sampling_period=1.0 / FS)
    return freqs, np.abs(coefs)


def compute_sst(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """同步压缩（基于 CWT），返回 (freqs, |SST|)。相位被用来把能量搬到瞬时频率上。"""
    from ssqueezepy import ssq_cwt

    # ssqueezepy 默认 GMW 小波，对低频长周期信号适用；fs 传入以得到物理频率轴。
    tx, _, ssq_freqs, *_ = ssq_cwt(sig.astype(np.float64), fs=FS)
    # ssq_freqs 高频在前，统一成升序返回。
    order = np.argsort(ssq_freqs)
    return ssq_freqs[order], np.abs(tx)[order]


def harmonic_separation_ratio(
    freqs: np.ndarray, mag2d: np.ndarray, target_rr_hz: float
) -> float:
    """分离度判据：基频带能量 / 2× 谐波带能量（时间平均后）。

    越大 = 基频相对 2× 谐波越突出 = 越不容易把谐波误拣成基频。
    基频带取 target_rr_hz ± 15%，谐波带取 2*target_rr_hz ± 15%。
    """
    if not np.isfinite(target_rr_hz) or target_rr_hz <= 0:
        return float("nan")
    spec = mag2d.mean(axis=1) if mag2d.ndim == 2 else mag2d  # 时间平均 → 每频率一个能量
    fund = _resp_band_mask(freqs, target_rr_hz * 0.85, target_rr_hz * 1.15)
    harm = _resp_band_mask(freqs, 2 * target_rr_hz * 0.85, 2 * target_rr_hz * 1.15)
    fund_e = spec[fund].sum() if fund.any() else 0.0
    harm_e = spec[harm].sum() if harm.any() else np.nan
    if not np.isfinite(harm_e) or harm_e <= 1e-12:
        return float("nan")
    return float(fund_e / harm_e)


def _bandpassed_peak_rate_hz(sig: np.ndarray) -> float:
    """对 tho 真值算呼吸率（Hz），复用项目口径定位基频，用于分离度判据与画图标注。"""
    from resp_train.metrics.signal import estimate_bandpassed_peak_rate_bpm

    bpm = estimate_bandpassed_peak_rate_bpm(sig, fs=FS, low_hz=RESP_LOW_HZ, high_hz=RESP_HIGH_HZ)
    return float(bpm) / 60.0 if np.isfinite(bpm) else float("nan")


def render_window(
    row_id: int,
    bcg: np.ndarray,
    tho: np.ndarray,
    misclass_err: float,
    out_path: Path,
) -> dict:
    """对单窗画 STFT/CWT/SST 三联图 + 返回分离度数值。对 BCG 输入做时频分析。"""
    target_rr_hz = _bandpassed_peak_rate_hz(tho)

    f_stft, t_stft, m_stft = compute_stft(bcg)
    f_cwt, m_cwt = compute_cwt(bcg)
    f_sst, m_sst = compute_sst(bcg)

    sep = {
        "row_id": row_id,
        "misclass_err_bpm": round(float(misclass_err), 3),
        "target_rr_bpm": round(target_rr_hz * 60, 2) if np.isfinite(target_rr_hz) else float("nan"),
        "sep_stft": round(harmonic_separation_ratio(f_stft, m_stft, target_rr_hz), 3),
        "sep_cwt": round(harmonic_separation_ratio(f_cwt, m_cwt, target_rr_hz), 3),
        "sep_sst": round(harmonic_separation_ratio(f_sst, m_sst, target_rr_hz), 3),
    }

    # 上排：BCG / tho 原始波形（全窗 + 前 30s 放大）；下排：STFT/CWT/SST 时频图到 8Hz。
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    t_axis = np.arange(bcg.shape[0]) / FS

    # --- 上排左：BCG 全窗 ---
    ax = axes[0, 0]
    ax.plot(t_axis, bcg, color="tab:blue", lw=0.4)
    ax.set_title("BCG (full 180s)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amp")
    # --- 上排中：tho 全窗 ---
    ax = axes[0, 1]
    ax.plot(t_axis, tho, color="tab:red", lw=0.5)
    ax.set_title("tho target (full 180s)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amp")
    # --- 上排右：前 30s BCG vs tho 叠加放大（看呼吸形态与心冲击纹理）---
    ax = axes[0, 2]
    zoom = t_axis <= 30.0

    def _z(a):  # 归一化便于叠加比形态
        a = a - a.mean()
        s = a.std()
        return a / s if s > 1e-8 else a

    ax.plot(t_axis[zoom], _z(bcg[zoom]), color="tab:blue", lw=0.6, label="BCG")
    ax.plot(t_axis[zoom], _z(tho[zoom]), color="tab:red", lw=1.0, label="tho")
    ax.set_title("BCG vs tho (first 30s, z-norm)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("z")
    ax.legend(loc="upper right", fontsize=7)

    # --- 下排：三种时频图，频率轴到 8Hz（全频带，对齐训练 STFT 口径）---
    band = (0.0, 8.0)
    for ax, (name, f, m, taxis) in zip(
        axes[1],
        [
            ("STFT |X|", f_stft, m_stft, t_stft),
            ("CWT |W|", f_cwt, m_cwt, np.arange(m_cwt.shape[1]) / FS),
            ("SST |T|", f_sst, m_sst, np.arange(m_sst.shape[1]) / FS),
        ],
    ):
        fmask = (f >= band[0]) & (f <= band[1])
        ax.pcolormesh(taxis, f[fmask], m[fmask], shading="auto", cmap="magma")
        if np.isfinite(target_rr_hz):
            ax.axhline(target_rr_hz, color="cyan", lw=1.0, ls="--", label="tho fundamental")
            ax.axhline(2 * target_rr_hz, color="lime", lw=1.0, ls=":", label="2x harmonic")
        ax.set_title(f"{name} (to 8Hz)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("freq (Hz)")
        ax.set_ylim(*band)
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"row={row_id}  misclass_err={misclass_err:.1f}bpm  "
        f"tho_rr={sep['target_rr_bpm']}bpm  |  separation STFT={sep['sep_stft']} "
        f"CWT={sep['sep_cwt']} SST={sep['sep_sst']} (higher=better)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return sep


def main() -> None:
    parser = argparse.ArgumentParser(description="E4 前置：STFT/CWT/SST 幅度谱谐波分离对比")
    parser.add_argument("--config", default="configs/tho_research_v2.yaml")
    parser.add_argument("--metrics-run", required=True, help="含 metrics.csv 的 run 目录，用于挑误拣窗口")
    parser.add_argument("--n-hard", type=int, default=6, help="高误拣窗口数")
    parser.add_argument("--n-clean", type=int, default=3, help="干净窗口数（对照）")
    parser.add_argument("--out-dir", default="runs/e4_timefreq_compare")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 从 metrics.csv 挑高误拣 + 干净窗口
    metrics = pd.read_csv(Path(args.metrics_run) / "metrics.csv")
    metrics = metrics.dropna(subset=["rr_peak_band_abs_error"]).copy()
    metrics = metrics.sort_values("rr_peak_band_abs_error", ascending=False)
    hard_ids = metrics.head(args.n_hard)["dataset_row_id"].astype(int).tolist()
    clean_ids = metrics.tail(args.n_clean)["dataset_row_id"].astype(int).tolist()
    picks = [("hard", rid) for rid in hard_ids] + [("clean", rid) for rid in clean_ids]
    print(f"挑选窗口: hard={hard_ids} clean={clean_ids}", flush=True)

    # 2) 用 val split 全量 index 回溯波形（不抽样，确保目标 row_id 在内）
    cfg = load_config(args.config)
    index_path = Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)
    index = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    val_rows = index[index["split"] == str(cfg.data.val_split)].copy()
    dataset = ResearchV2WindowDataset(
        index_csv_path=index_path,
        rows=val_rows,
        cfg=cfg,
        preload_windows=False,
    )
    rowid_to_idx = {int(r): i for i, r in enumerate(dataset.rows["dataset_row_id"].tolist())}

    # 3) 逐窗渲染 + 收分离度
    records = []
    for kind, rid in picks:
        if rid not in rowid_to_idx:
            print(f"  跳过 row={rid}（不在 val split）", flush=True)
            continue
        item = dataset[rowid_to_idx[rid]]
        bcg = item["x"].view(-1).numpy().astype(np.float64)
        tho = item["target"].view(-1).numpy().astype(np.float64)
        err = float(metrics[metrics["dataset_row_id"] == rid]["rr_peak_band_abs_error"].iloc[0])
        out_path = out_dir / f"{kind}_row{rid}_err{err:.1f}.png"
        rec = render_window(rid, bcg, tho, err, out_path)
        rec["kind"] = kind
        records.append(rec)
        print(f"  {kind} row={rid}: {rec}", flush=True)

    # 4) 汇总分离度表 + 判定提示
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "separation_summary.csv", index=False)
    hard = df[df["kind"] == "hard"]
    print("\n=== 谐波分离度汇总（基频能量/2×谐波能量，越大越好）===", flush=True)
    print(df.to_string(index=False), flush=True)
    if not hard.empty:
        print(
            f"\n高误拣窗口均值: STFT={hard['sep_stft'].mean():.3f} "
            f"CWT={hard['sep_cwt'].mean():.3f} SST={hard['sep_sst'].mean():.3f}",
            flush=True,
        )
        print(
            "判定提示：若 SST 在 hard 窗口上分离度明显高于 STFT（且图上 2× 谐波带更暗）"
            "→ 值得进 E4-SST 探针；若持平/更差 → 直接收。",
            flush=True,
        )


if __name__ == "__main__":
    main()
