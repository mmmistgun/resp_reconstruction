#!/usr/bin/env python3
"""RespDiff-BCG：CPU 合成验证、GPU 合成工程检查、6400-update train/val。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.data.factory import build_tho_data
from resp_train.data.research_v2 import ResearchV2WindowDataset
from resp_train.respdiff.data import assert_split_independence
from resp_train.respdiff_bcg.config import DEFAULT_CONFIG, load_config
from resp_train.respdiff_bcg.data import ChunkDataset
from resp_train.respdiff_bcg.runtime import (
    artifact_manifest, environment, make_model, seed_all, sha256, source_file_inventory,
    source_identity, save_source_snapshot, train_updates, validate, write_json,
    LOGGER, Progress, cuda_memory_detail, run_logging,
)


def synthetic_dataset(output, cfg, split, *, n_parents=2):
    """走实际 ResearchV2 NPZ 读取器；所有文件只写入本次 disposable 输出。"""
    rows = []
    for index in range(n_parents):
        row_id = index + (0 if split == "train" else 10)
        time = np.arange(18000) / 100
        target = (1 + 0.3 * np.sin(2 * np.pi * 0.015 * time)) * np.sin(
            2 * np.pi * (0.20 + index * 0.02) * time)
        condition = target + 0.2 * np.sin(2 * np.pi * 2 * time)
        source, reference = output / f"{split}_{index}_bcg.npz", output / f"{split}_{index}_tho.npz"
        np.savez(source, bcg=condition.astype(np.float32))
        np.savez(reference, tho=target.astype(np.float32))
        rows.append({"dataset_row_id": row_id, "samp_id": row_id, "split": split,
                     "source_npz": str(source.resolve()), "target_source_npz": str(reference.resolve()),
                     "window_start_sample": 0, "window_end_sample": 18000,
                     "bcg_signal_key": "bcg", "target_signal_key": "tho",
                     "coupling_state_id": 0, "allowed_losses": "waveform"})
    rows = pd.DataFrame(rows)
    index_path = output / f"{split}_fixture_index.csv"
    rows.to_csv(index_path, index=False)
    return ChunkDataset(ResearchV2WindowDataset(index_path, rows, cfg), rows)


def gpu_check(model, cfg, output):
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("GPU 检查要求 cuda:N")
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    LOGGER.info("GPU 工程检查：%s，训练 batch=%d，推理 batch=%d",
                torch.cuda.get_device_name(device), cfg.training.batch_size, cfg.inference.batch_size)
    progress = Progress("GPU 检查 update", 3, every=1, unit="update")
    for update in range(1, 4):
        x = torch.randn(int(cfg.training.batch_size), 1, 600, device=device)
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_loss(x, torch.randn_like(x))
        losses["loss"].backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("GPU gradient 非有限")
        optimizer.step()
        progress.update(update, " ".join(f"{key}={float(value.detach()):.6g}" for key, value in losses.items())
                        + cuda_memory_detail(device))
    model.eval()
    for batch_size in (1, int(cfg.inference.batch_size)):
        LOGGER.info("GPU 六步采样检查：batch=%d", batch_size)
        x = torch.randn(batch_size, 1, 600, device=device)
        model.sample_chunks(x, torch.randn_like(x))
    torch.cuda.synchronize(device)
    LOGGER.info("GPU 检查通过：峰值 allocated=%.2f GiB", torch.cuda.max_memory_allocated(device) / 2**30)
    write_json(output / "gpu_check.json", {"device": str(device),
        "gpu": torch.cuda.get_device_name(device), "training_updates": 3,
        "train_batch_size": int(cfg.training.batch_size),
        "inference_batches": [1, int(cfg.inference.batch_size)],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})


def run(mode, output, cfg, *, confirm_training=False):
    if mode not in {"synthetic-smoke", "gpu-check", "train"}:
        raise ValueError("未知运行 mode")
    if mode == "train" and not confirm_training:
        raise ValueError("真实数据训练需要 --confirm-training；仅开放 train/val")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with run_logging(output):
        _run(mode, output, cfg)


def _run(mode, output, cfg):
    stage = "initialization"
    try:
        tiny = mode == "synthetic-smoke"
        LOGGER.info("开始 mode=%s seed=%d device=%s train_batch=%d inference_batch=%d updates=%d",
                    mode, cfg.training.seed, "cpu" if tiny else cfg.training.device,
                    2 if tiny else cfg.training.batch_size, 8 if tiny else cfg.inference.batch_size,
                    2 if tiny else (3 if mode == "gpu-check" else 6400))
        LOGGER.info("输出目录：%s；终端日志：%s", output, output / "run.log")
        identity = source_identity(cfg)
        OmegaConf.save(cfg, output / "resolved_config.yaml")
        write_json(output / "source_identity.json", identity)
        save_source_snapshot(output, identity)
        write_json(output / "environment.json", environment())
        seed_all(int(cfg.training.seed))
        device = torch.device("cpu" if tiny else str(cfg.training.device))
        LOGGER.info("初始化模型：%s", "合成小网络" if tiny else "6 层双向 RNN / hidden=1024 / FP32")
        model = make_model(cfg, tiny=tiny).to(device)
        write_json(output / "execution.json", {"mode": mode, "model_spec": vars(model.spec),
                   "device": str(device), "updates": 2 if tiny else (3 if mode == "gpu-check" else 6400),
                   "training_batch_size": 2 if tiny else int(cfg.training.batch_size),
                   "inference_batch_size": 8 if tiny else int(cfg.inference.batch_size)})
        if mode == "gpu-check":
            stage = "gpu_synthetic_check"
            gpu_check(model, cfg, output)
        else:
            stage = "data_identity"
            LOGGER.info("读取 train/val 索引并核对父窗口身份")
            if tiny:
                train, val = (synthetic_dataset(output, cfg, s) for s in ("train", "val"))
                index_path = output / "train_fixture_index.csv"
            else:
                bundle = build_tho_data(cfg)
                train, val = (ChunkDataset(part.dataset, part.rows) for part in (bundle.train, bundle.val))
                index_path = bundle.train.index_path
            assert_split_independence(train.rows, val.rows)
            LOGGER.info("数据准入：train parents=%d chunks=%d；val parents=%d chunks=%d",
                        len(train.rows), len(train), len(val.rows), len(val))
            rows = pd.concat([train.rows, val.rows], ignore_index=True)
            inventory = source_file_inventory(rows)
            index_hash = sha256(index_path)
            write_json(output / "data_identity.json", {"index_path": str(index_path),
                       "index_sha256": index_hash, "source_file_stats": inventory})
            for split, dataset in (("train", train), ("val", val)):
                dataset.rows.to_csv(output / f"{split}_parents.csv", index=False)
                dataset.manifest().to_csv(output / f"{split}_chunks.csv", index=False)
            stage = "training"
            train_updates(model, train, cfg, output, updates=2 if tiny else 6400,
                          batch_size=2 if tiny else None)
            stage = "validation"
            validate(model, val, cfg, output, batch_size=8 if tiny else None)
            if inventory != source_file_inventory(rows) or index_hash != sha256(index_path):
                raise ValueError("运行期间数据身份发生变化")
        stage = "completion"
        LOGGER.info("检查运行前后源码与配置身份")
        if source_identity(cfg) != identity:
            raise ValueError("运行期间源码或配置发生变化")
        # 此后不再向 run.log 追加内容，使 receipt 中的日志 hash 保持有效。
        LOGGER.info("生成产物校验摘要与完成回执：%s", output / "receipt.json")
        write_json(output / "receipt.json", {"status": "complete", "mode": mode,
                   "formal_evidence": mode == "train", "artifacts": artifact_manifest(output)})
        print(f"完成：{output / 'receipt.json'}", file=sys.stderr, flush=True)
    except BaseException as exc:
        LOGGER.error("运行终止：stage=%s %s: %s；详情：%s",
                     stage, type(exc).__name__, str(exc), output / "failure.json")
        LOGGER.debug("异常堆栈", exc_info=True)
        write_json(output / "failure.json", {"status": "failed", "mode": mode,
                   "stage": stage, "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("synthetic-smoke", "gpu-check", "train"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-training", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.mode == "synthetic-smoke":
        torch.set_num_threads(1)
    run(args.mode, args.output, cfg, confirm_training=args.confirm_training)


if __name__ == "__main__":
    main()
