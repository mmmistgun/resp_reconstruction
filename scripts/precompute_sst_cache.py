"""E4-SST：离线预计算 SST 幅度谱缓存（带限 0.05-8Hz + 降采样 37 帧 + log1p N0）。

为什么离线：SST 单窗 ~357ms（STFT 的 ~300 倍），且是固定无参变换、跨 epoch 不变，
on-the-fly 会把训练拖到几天且纯重复。预计算一次存盘，训练时按 dataset_row_id 读 ~23KB/窗。

网格（与用户逐数确认、对齐 B2-0 口径）：
- 频率：全频带 0.05-8Hz，取 SST 原生对数频点（约 159 点），不砍频带（避免换表征+换频带双变量）。
- 时间：18000 → 37 帧，区间均值池化（对齐 STFT hop=500/center=True 网格，保住帧间能量峰）。
- 归一化：N0 = log1p(|SST|)，直接存为可喂模型的特征。

用法：
  python scripts/precompute_sst_cache.py --config configs/tho_research_v2.yaml \
      --out runs/sst_cache/sst_8hz_37f.npz
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import load_config  # noqa: E402
from resp_train.data.research_v2 import ResearchV2WindowDataset, read_research_v2_index  # noqa: E402

# 缓存网格常量（对齐 B2-0 STFT 口径）。
BAND_LOW_HZ = 0.05
BAND_HIGH_HZ = 8.0
N_FRAMES = 37
ComputeFn = Callable[..., tuple[np.ndarray, np.ndarray]]


def compute_sst_window(
    sig: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """单窗 SST 幅度谱 → 带限 → 时间均值池化到 n_frames → log1p（N0）。

    返回 (feat[freq_in_band, n_frames], freqs[freq_in_band])。
    """
    from ssqueezepy import ssq_cwt

    tx, _, ssq_freqs, *_ = ssq_cwt(np.asarray(sig, dtype=np.float64), fs=float(fs))
    mag = np.abs(tx)  # (n_freq, n_time)
    freqs = np.asarray(ssq_freqs, dtype=np.float64)

    # 频率轴统一升序。
    order = np.argsort(freqs)
    freqs = freqs[order]
    mag = mag[order]

    # 带限 [low, high]。
    band = (freqs >= low_hz) & (freqs <= high_hz)
    freqs_band = freqs[band]
    mag_band = mag[band]  # (freq_in_band, n_time)

    # 时间轴均值池化到 n_frames（n_time 不必整除 n_frames，用 array_split）。
    chunks = np.array_split(mag_band, int(n_frames), axis=1)
    pooled = np.stack([c.mean(axis=1) for c in chunks], axis=1)  # (freq_in_band, n_frames)

    feat = np.log1p(pooled).astype(np.float32)  # N0
    return feat, freqs_band.astype(np.float32)


def _iter_split_rows(cfg, split: str):
    """构造某 split 的全量（不抽样）dataset，逐窗产出 (row_id, bcg)。"""
    index_path = Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)
    index = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows = index[index["split"] == str(split)].copy()
    dataset = ResearchV2WindowDataset(index_csv_path=index_path, rows=rows, cfg=cfg, preload_windows=False)
    for i in range(len(dataset)):
        item = dataset[i]
        yield int(item["meta"]["dataset_row_id"]), item["x"].view(-1).numpy().astype(np.float32, copy=False)


def _check_freqs(row_id: int, freqs: np.ndarray, freqs_ref: np.ndarray | None) -> np.ndarray:
    """确认所有窗口的 SST 频率网格一致。"""
    freqs = np.asarray(freqs, dtype=np.float32)
    if freqs_ref is None:
        return freqs
    if freqs.shape != freqs_ref.shape:
        raise ValueError(f"row {row_id} 频带点数 {freqs.shape} 与首窗 {freqs_ref.shape} 不一致")
    return freqs_ref


def _init_sst_worker(disable_inner_parallel: bool) -> None:
    """多进程按窗口并行时关闭 ssqueezepy 内部并行，避免 CPU 过度竞争。"""
    if disable_inner_parallel:
        os.environ["SSQ_PARALLEL"] = "0"


def _compute_sst_job(job: tuple[int, str, int, np.ndarray, float, float, float, int]):
    """ProcessPool worker 入口；保持顶层函数以支持 spawn 序列化。"""
    seq, split, row_id, bcg, fs, low_hz, high_hz, n_frames = job
    feat, freqs = compute_sst_window(bcg, fs=fs, low_hz=low_hz, high_hz=high_hz, n_frames=n_frames)
    return seq, split, row_id, feat, freqs


def _precompute_sst_features_serial(
    cfg,
    *,
    splits: list[str],
    fs: float,
    low_hz: float,
    high_hz: float,
    n_frames: int,
    log_every: int,
    compute_fn: ComputeFn,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    row_ids: list[int] = []
    feats: list[np.ndarray] = []
    seen: set[int] = set()
    freqs_ref: np.ndarray | None = None

    for split in splits:
        print(f"=== split={split} ===", flush=True)
        count = 0
        for row_id, bcg in _iter_split_rows(cfg, split):
            if row_id in seen:
                continue
            feat, freqs = compute_fn(bcg, fs=fs, low_hz=low_hz, high_hz=high_hz, n_frames=n_frames)
            freqs_ref = _check_freqs(row_id, freqs, freqs_ref)
            seen.add(row_id)
            row_ids.append(row_id)
            feats.append(np.asarray(feat, dtype=np.float32))
            count += 1
            if count % log_every == 0:
                print(f"  {split}: {count} 窗已算", flush=True)
        print(f"  {split}: 共 {count} 窗", flush=True)

    if not feats or freqs_ref is None:
        raise RuntimeError("没有可预计算的窗口")
    return row_ids, np.stack(feats, axis=0).astype(np.float32, copy=False), freqs_ref


def _precompute_sst_features_parallel(
    cfg,
    *,
    splits: list[str],
    fs: float,
    low_hz: float,
    high_hz: float,
    n_frames: int,
    log_every: int,
    workers: int,
    mp_start_method: str,
    max_pending: int,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    if workers < 1:
        raise ValueError(f"workers 必须 >= 1，实际为 {workers}")
    if max_pending < workers:
        max_pending = workers

    seen: set[int] = set()
    pending: set = set()
    results: dict[int, tuple[int, np.ndarray]] = {}
    count_by_split: dict[str, int] = {}
    freqs_ref: np.ndarray | None = None
    seq = 0
    ctx = mp.get_context(mp_start_method)

    def consume(done) -> None:
        nonlocal freqs_ref
        for future in done:
            done_seq, split, row_id, feat, freqs = future.result()
            freqs_ref = _check_freqs(row_id, freqs, freqs_ref)
            results[int(done_seq)] = (int(row_id), np.asarray(feat, dtype=np.float32))
            count_by_split[split] = count_by_split.get(split, 0) + 1
            if count_by_split[split] % log_every == 0:
                print(f"  {split}: {count_by_split[split]} 窗已算", flush=True)

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_sst_worker,
        initargs=(True,),
    ) as pool:
        for split in splits:
            print(f"=== split={split} ===", flush=True)
            count_by_split[split] = 0
            for row_id, bcg in _iter_split_rows(cfg, split):
                if row_id in seen:
                    continue
                seen.add(row_id)
                job = (seq, split, row_id, bcg, fs, low_hz, high_hz, int(n_frames))
                pending.add(pool.submit(_compute_sst_job, job))
                seq += 1
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    consume(done)
            if pending:
                done, pending = wait(pending)
                consume(done)
            print(f"  {split}: 共 {count_by_split[split]} 窗", flush=True)

    if not results or freqs_ref is None:
        raise RuntimeError("没有可预计算的窗口")

    ordered = [results[i] for i in sorted(results)]
    row_ids = [row_id for row_id, _ in ordered]
    arr = np.stack([feat for _, feat in ordered], axis=0).astype(np.float32, copy=False)
    return row_ids, arr, freqs_ref


def precompute_sst_features(
    cfg,
    *,
    splits: list[str],
    fs: float,
    low_hz: float,
    high_hz: float,
    n_frames: int,
    workers: int,
    log_every: int,
    compute_fn: ComputeFn = compute_sst_window,
    mp_start_method: str = "spawn",
    max_pending: int | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """预计算 SST 特征，返回 row_ids、已堆叠特征和频率网格。"""
    if workers < 1:
        raise ValueError(f"workers 必须 >= 1，实际为 {workers}")
    if workers == 1:
        return _precompute_sst_features_serial(
            cfg,
            splits=splits,
            fs=fs,
            low_hz=low_hz,
            high_hz=high_hz,
            n_frames=n_frames,
            log_every=log_every,
            compute_fn=compute_fn,
        )
    if compute_fn is not compute_sst_window:
        raise ValueError("workers > 1 时不支持自定义 compute_fn；请使用单进程路径测试注入")
    return _precompute_sst_features_parallel(
        cfg,
        splits=splits,
        fs=fs,
        low_hz=low_hz,
        high_hz=high_hz,
        n_frames=n_frames,
        log_every=log_every,
        workers=workers,
        mp_start_method=mp_start_method,
        max_pending=max_pending or workers * 2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="E4-SST 幅度谱缓存离线预计算")
    parser.add_argument("--config", default="configs/tho_research_v2.yaml")
    parser.add_argument("--out", required=True, help="输出 npz 路径")
    parser.add_argument("--low-hz", type=float, default=BAND_LOW_HZ)
    parser.add_argument("--high-hz", type=float, default=BAND_HIGH_HZ)
    parser.add_argument("--n-frames", type=int, default=N_FRAMES)
    parser.add_argument("--workers", type=int, default=1, help="SST 窗口级并行进程数；1 为串行")
    parser.add_argument(
        "--mp-start-method",
        choices=("spawn", "forkserver"),
        default="spawn",
        help="多进程启动方式；避免 fork 与 OpenMP 运行时冲突",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=None,
        help="多进程待处理任务上限，默认 workers*2，限制波形数组排队占用内存",
    )
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    fs = float(cfg.window.target_fs)
    splits = sorted({str(cfg.data.train_split), str(cfg.data.val_split)})

    row_ids, arr, freqs_ref = precompute_sst_features(
        cfg,
        splits=splits,
        fs=fs,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        n_frames=args.n_frames,
        workers=args.workers,
        log_every=args.log_every,
        mp_start_method=args.mp_start_method,
        max_pending=args.max_pending,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        row_ids=np.asarray(row_ids, dtype=np.int64),
        sst=arr,
        freqs=freqs_ref.astype(np.float32),
        n_frames=np.int64(args.n_frames),
        low_hz=np.float32(args.low_hz),
        high_hz=np.float32(args.high_hz),
        norm=np.array("n0"),
    )
    print(
        f"saved {out_path}: N={len(row_ids)} 窗, sst shape={arr.shape}, "
        f"freq点={freqs_ref.shape[0]}, 帧={args.n_frames}, "
        f"约 {arr.nbytes / 1024 / 1024:.1f} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
