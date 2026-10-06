#!/usr/bin/env python3
"""双 1 Hz 低通呼吸基带三臂：合成验收、GPU 检查与固定预算 train/val。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.data.factory import build_tho_data
from resp_train.respdiff.data import assert_split_independence
from resp_train.respdiff_bcg.baseband import (
    DEFAULT_CONFIG, LPF_CONTRACT, SCHEMA, baseband_dataset, load_baseband_config, make_baseband_model,
)
from resp_train.respdiff_bcg.baseband_diagnostics import TrainingProbe, validation_diagnostics
from resp_train.respdiff_bcg.runtime import (
    LOGGER, artifact_manifest, environment, run_logging, save_source_snapshot, seed_all,
    sha256, source_file_inventory, source_identity, train_updates, validate, write_json,
)
from scripts.run_respdiff_bcg_v1 import gpu_check, synthetic_dataset


def baseband_identity(cfg):
    identity = source_identity(cfg)
    paths = [Path(__file__).resolve(), ROOT / "scripts/run_respdiff_bcg_v1.py",
             ROOT / "configs/respdiff_bcg_v1/gpu_b64_b64.yaml",
             ROOT / "docs/experiments/respdiff_bcg_baseband_v1_protocol_20261006.md",
             *sorted((ROOT / "configs/respdiff_bcg_baseband_v1").glob("*.yaml"))]
    for path in paths:
        identity["files"][str(path.relative_to(ROOT))] = sha256(path)
    identity["lowpass"] = LPF_CONTRACT
    return identity


def fixture(output, cfg, split, *, n_parents=2):
    raw = synthetic_dataset(output, cfg, split, n_parents=n_parents)
    return baseband_dataset(raw.parents, raw.rows)


def run(mode, output, cfg, *, confirm_training=False):
    if mode not in {"synthetic-smoke", "gpu-check", "train"}:
        raise ValueError("未知 mode")
    if mode == "train" and not confirm_training:
        raise ValueError("训练需要 --confirm-training；仅开放 train/val")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with run_logging(output):
        stage = "initialization"
        try:
            tiny = mode == "synthetic-smoke"
            objective = str(cfg.objective.name)
            device = torch.device("cpu" if tiny else str(cfg.training.device))
            updates = 2 if tiny else (3 if mode == "gpu-check" else int(cfg.training.max_updates))
            LOGGER.info("呼吸基带：objective=%s mode=%s device=%s updates=%d", objective, mode, device, updates)
            LOGGER.info("输出：%s；训练过程日志：%s", output, output / "history.jsonl")
            identity = baseband_identity(cfg)
            OmegaConf.save(cfg, output / "resolved_config.yaml")
            write_json(output / "source_identity.json", identity)
            save_source_snapshot(output, identity)
            write_json(output / "environment.json", environment())
            seed_all(int(cfg.training.seed))
            model = make_baseband_model(cfg, tiny=tiny).to(device)
            write_json(output / "execution.json", {"mode": mode, "objective": objective,
                "model_spec": vars(model.spec), "device": str(device), "updates": updates,
                "training_batch_size": 2 if tiny else int(cfg.training.batch_size),
                "inference_batch_size": 8 if tiny else int(cfg.inference.batch_size),
                "validation_reference": "full-parent 1Hz LPF THO", "lowpass": LPF_CONTRACT})
            if mode == "gpu-check":
                stage = "gpu_synthetic_check"
                gpu_check(model, cfg, output)
                # 额外验收 native 模型双 backward probe 的显存与 cuDNN 路径。
                probe_data = fixture(output, cfg, "train", n_parents=5)
                TrainingProbe(model, probe_data, cfg, output)(0)
                write_json(output / "gpu_probe_memory.json", {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
            else:
                stage = "data_identity"
                if tiny:
                    train, val = (fixture(output, cfg, split) for split in ("train", "val"))
                    index_path = output / "train_fixture_index.csv"
                else:
                    bundle = build_tho_data(cfg)
                    train, val = (baseband_dataset(part.dataset, part.rows) for part in (bundle.train, bundle.val))
                    index_path = bundle.train.index_path
                assert_split_independence(train.rows, val.rows)
                rows = pd.concat([train.rows, val.rows], ignore_index=True)
                inventory = source_file_inventory(rows)
                index_hash = sha256(index_path)
                write_json(output / "data_identity.json", {"index_path": str(index_path),
                    "index_sha256": index_hash, "source_file_stats": inventory, "lowpass": LPF_CONTRACT})
                for split, dataset in (("train", train), ("val", val)):
                    dataset.rows.to_csv(output / f"{split}_parents.csv", index=False)
                    dataset.manifest().to_csv(output / f"{split}_chunks.csv", index=False)
                LOGGER.info("数据准入：train parents=%d chunks=%d；val parents=%d chunks=%d",
                            len(train.rows), len(train), len(val.rows), len(val))
                stage = "training"
                probe = TrainingProbe(model, train, cfg, output, tiny=tiny)
                train_updates(model, train, cfg, output, updates=updates, batch_size=2 if tiny else None,
                    checkpoint_schema=SCHEMA, checkpoint_metadata={
                        "protocol": OmegaConf.to_container(cfg.protocol), "lowpass": LPF_CONTRACT,
                        "objective": OmegaConf.to_container(cfg.objective), "seed": int(cfg.training.seed)},
                    diagnostic_callback=probe, history_extra=lambda: model.last_diagnostics)
                stage = "validation"
                validate(model, val, cfg, output, batch_size=8 if tiny else None,
                         method=f"RespDiff-BCG-baseband-v1/{objective}")
                validation_diagnostics(output)
                if inventory != source_file_inventory(rows) or index_hash != sha256(index_path):
                    raise ValueError("运行期间数据身份变化")
            stage = "completion"
            if baseband_identity(cfg) != identity:
                raise ValueError("运行期间源码或配置变化")
            LOGGER.info("保存完成回执：%s", output / "receipt.json")
            write_json(output / "receipt.json", {"status": "complete", "mode": mode,
                "objective": objective, "formal_evidence": mode == "train", "artifacts": artifact_manifest(output)})
            print(f"完成：{output / 'receipt.json'}", file=sys.stderr, flush=True)
        except BaseException as exc:
            LOGGER.error("运行终止：stage=%s %s: %s", stage, type(exc).__name__, exc)
            LOGGER.debug("异常堆栈", exc_info=True)
            write_json(output / "failure.json", {"status": "failed", "stage": stage,
                "error_type": type(exc).__name__, "error": str(exc)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("synthetic-smoke", "gpu-check", "train"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-training", action="store_true")
    args = parser.parse_args()
    cfg = load_baseband_config(args.config)
    if args.mode == "synthetic-smoke":
        torch.set_num_threads(1)
    run(args.mode, args.output, cfg, confirm_training=args.confirm_training)
