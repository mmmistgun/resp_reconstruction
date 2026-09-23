from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.crd.training import build_crd_optimizer, optimizer_updates_per_epoch, train_crd_one_epoch
from resp_train.engine import collect_predictions, validate
from resp_train.losses.task import RespirationTaskLoss
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics, validation_local_rr_mean
from resp_train.utils.run import resolve_device, set_seed, setup_logger

from .artifacts import (
    artifact_directory, read_json, require_finite_tree, array_digest,
    sha256_file, verified_manifest, write_json,
)
from .config import PROTOCOL, load_experiment_config, model_config, validate_experiment_config
from .data import build_loaders, verified_cache
from resp_train.aligned_dual_view.features import spec_digest
from .model import FusionModel, ARMS, module_report
from .stopping import stopping_state

PRIMARY = (
    "whole_rr_abs_error_bpm", "local_rr_mae_bpm", "envelope_trajectory_mae",
    "global_envelope_modulation_error", "lag_aware_signed_pcc",
)


RuntimeModel = FusionModel


def comparison_identity(cfg, samples, cache_id, code, dependencies):
    comparison = OmegaConf.to_container(cfg, resolve=True)
    comparison["model"].pop("arm")
    comparison["model"].pop("initialization_seed")
    for key in ("seed", "device", "show_progress"):
        comparison["training"].pop(key)
    return spec_digest({"config": comparison, "samples": samples, "cache_id": cache_id,
                        "code": code, "dependencies": dependencies})


def _device(cfg):
    device = resolve_device(cfg.training.device)
    if device.type == "cuda":
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    return device


@contextmanager
def _training_logger(root):
    logger = setup_logger(root)
    try:
        yield logger
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def _check_predictions(predictions, rows):
    if not np.array_equal(predictions["dataset_row_id"], rows.dataset_row_id.to_numpy()):
        raise ValueError("validation 输出 row ID/顺序不完整")
    if not np.all(predictions["split"] == "val"):
        raise ValueError("评价仅允许 validation")
    for key in ("r_tho_hat", "tho_ref"):
        if predictions[key].shape != (len(rows), 1, 18000) or not np.isfinite(predictions[key]).all():
            raise FloatingPointError(f"{key} shape/finite 不合格")


def check_primary_metrics(metrics):
    eligibility = {
        "whole_rr_abs_error_bpm": "whole_rr_target_eligible",
        "local_rr_mae_bpm": "local_rr_target_eligible",
        "lag_aware_signed_pcc": "joint_target_eligible",
    }
    for name in PRIMARY:
        values = metrics[name].to_numpy(dtype=np.float64)
        eligible = (metrics[eligibility[name]].to_numpy(dtype=bool) if name in eligibility
                    else np.ones(len(metrics), dtype=bool))
        if np.isinf(values).any() or not np.array_equal(np.isfinite(values), eligible):
            raise FloatingPointError(f"{name} 有非有限值或与 target eligibility 不一致")
        if not eligible.any():
            raise ValueError(f"{name} 没有可评价 sample")
        if name == "lag_aware_signed_pcc":
            if np.any(np.abs(values[eligible]) > 1 + 1e-6):
                raise ValueError("PCC 超出 [-1,1]")
        elif np.any(values[eligible] < 0):
            raise ValueError(f"误差指标 {name} 不得为负")


def _save_data_provenance(root, selection, loaders):
    records = {}
    for split, loader in loaders.items():
        frame = selection.rows[split]
        frame.to_csv(root / f"{split}_selection.csv", index=False)
        records[split] = [loader.dataset.provenance[index] for index in range(len(frame))]
    write_json(root / "samples.json", records)
    return {split: spec_digest(values) for split, values in records.items()}


def _save_checkpoint(path, *, model, optimizer, cfg, epoch, record, identity):
    payload = {
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "metrics": record, "config": OmegaConf.to_container(cfg, resolve=True),
        "identity": identity, "resume_supported": False,
    }
    require_finite_tree(payload)
    path.parent.mkdir(exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)


def _load_checkpoint(path, *, model, identity):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require_finite_tree(payload)
    if payload.get("identity") != identity or payload.get("resume_supported") is not False:
        raise ValueError("checkpoint identity 不匹配")
    kernel = payload["model_state_dict"]["downsample.kernel"]
    if not torch.equal(kernel.cpu(), model.downsample.kernel.detach().cpu()):
        raise ValueError("checkpoint 固定 FIR 系数漂移")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def _write_metrics(root, predictions, cfg):
    metrics = evaluate_task_predictions(predictions, cfg, include_test_only=False,
                                        method=f"adv_fusion_v1_{cfg.model.arm}")
    check_primary_metrics(metrics)
    summary = summarize_task_metrics(metrics)
    metrics.to_csv(root / "metrics.csv", index=False)
    summary.to_csv(root / "metrics_summary.csv", index=False)
    return {"metrics_sha256": sha256_file(root / "metrics.csv"),
            "summary_sha256": sha256_file(root / "metrics_summary.csv")}


def train(cfg, *, cache_root: str | Path, output: str | Path, model_factory=RuntimeModel) -> Path:
    validate_experiment_config(cfg)
    with artifact_directory(output, kind="train", cfg=cfg) as (root, started), _training_logger(root) as logger:
        set_seed(int(cfg.training.seed))
        device = _device(cfg)
        selection, loaders = build_loaders(cfg, cache_root)
        samples = _save_data_provenance(root, selection, loaders)
        cache = verified_cache(cache_root)
        identity = {
            "protocol": PROTOCOL, "config_sha256": spec_digest(OmegaConf.to_container(cfg, resolve=True)),
            "code_sha256": started["code"]["sha256"],
            "cache_manifest_sha256": sha256_file(Path(cache_root) / "manifest.json"),
            "cache_id": cache["cache_id"], "samples": samples,
        }
        model = model_factory(model_config(cfg)).to(device)
        initial = {name: array_digest(value) for name, value in model.state_dict().items()}
        shared = {name: value for name, value in initial.items() if not name.startswith("fusion.")}
        write_json(root / "initialization.json", {"state": initial, "shared_sha256": spec_digest(shared)})
        write_json(root / "model_report.json", module_report(model))
        loss_fn = RespirationTaskLoss(cfg).to(device)
        optimizer, partition = build_crd_optimizer(model, cfg)
        write_json(root / "optimizer_groups.json", {
            "decay": list(partition.decay_names), "no_decay": list(partition.no_decay_names),
            "weight_decay": float(cfg.training.weight_decay),
        })
        epochs = int(cfg.training.epochs)
        updates_per_epoch = optimizer_updates_per_epoch(len(loaders["train"]), cfg.training.gradient_accumulation_steps)
        total_updates = updates_per_epoch * epochs
        update_index = 0
        stopping = stopping_state(cfg)
        history = []
        with (root / "history.jsonl").open("x", encoding="utf-8") as history_file:
            for epoch in range(1, epochs + 1):
                train_values, update_index = train_crd_one_epoch(
                    model, loaders["train"], loss_fn, optimizer, device=device,
                    accumulation_steps=cfg.training.gradient_accumulation_steps,
                    update_index=update_index, total_updates=total_updates,
                    max_learning_rate=cfg.training.max_learning_rate, min_learning_rate=cfg.training.min_learning_rate,
                    warmup_fraction=cfg.training.warmup_fraction, grad_clip_norm=cfg.training.grad_clip_norm,
                    use_amp=cfg.training.use_amp, show_progress=cfg.training.show_progress,
                    epoch=epoch, total_epochs=epochs,
                )
                val_values, predictions = validate(
                    model, loaders["val"], loss_fn, device=device, return_predictions=True,
                    use_amp=cfg.training.use_amp, show_progress=cfg.training.show_progress,
                    epoch=epoch, total_epochs=epochs,
                )
                _check_predictions(predictions, selection.rows["val"])
                local_rr = validation_local_rr_mean(predictions, cfg)
                record = {"epoch": epoch, "optimizer_update": update_index, "val_local_rr_mae": local_rr,
                          **{f"train_{key}": value for key, value in train_values.items()},
                          **{f"val_{key}": value for key, value in val_values.items()}}
                require_finite_tree(record, "epoch history")
                history.append(record)
                history_file.write(json.dumps(record, allow_nan=False) + "\n")
                history_file.flush()
                improved = stopping.step(local_rr)
                if improved or stopping.reason is not None:
                    _save_checkpoint(root / "checkpoints" / f"epoch_{epoch:03d}.pt", model=model,
                                     optimizer=optimizer, cfg=cfg, epoch=epoch, record=record, identity=identity)
                logger.info("epoch=%d/%d update=%d/%d val_local_rr=%.6f selected=%s",
                            epoch, epochs, update_index, total_updates, local_rr, improved)
                if stopping.reason is not None:
                    logger.info("training_stopped epoch=%d reason=%s best_epoch=%d bad_epochs=%d",
                                epoch, stopping.reason, stopping.best_epoch, epoch-stopping.best_epoch)
                    break
        stop_receipt = stopping.receipt()
        actual_epochs = stopping.epoch
        best_epoch, best_value = stopping.best_epoch, stopping.best_value
        if update_index != updates_per_epoch * actual_epochs or best_epoch < 1:
            raise RuntimeError("训练更新数或 checkpoint selector 不完整")
        pd.DataFrame(history).to_csv(root / "history.csv", index=False)
        selected = root / "checkpoints" / f"epoch_{best_epoch:03d}.pt"
        _load_checkpoint(selected, model=model, identity=identity)
        predictions = collect_predictions(model, loaders["val"], device=device,
                                          max_windows=len(loaders["val"].dataset), use_amp=cfg.training.use_amp)
        _check_predictions(predictions, selection.rows["val"])
        # 所有 arm/seed 比较时要求同一数据、预算、环境和执行代码。
        comparison_id = comparison_identity(cfg, samples, cache["cache_id"], identity["code_sha256"],
                                            started["dependencies"])
        write_json(root / "manifest.json", {
            "protocol": PROTOCOL, "kind": "train", "identity": identity, "comparison_id": comparison_id,
            "run_role": cfg.protocol.run_role, "arm": cfg.model.arm, "seed": cfg.training.seed,
            "method": ARMS[cfg.model.arm][0], "position": ARMS[cfg.model.arm][1],
            "initialization_sha256": sha256_file(root / "initialization.json"),
            "shared_initialization_sha256": spec_digest(shared),
            "model_report_sha256": sha256_file(root / "model_report.json"),
            "selected_epoch": best_epoch, "selected_local_rr_mae": best_value,
            "selected_checkpoint": str(selected.relative_to(root)), "selected_sha256": sha256_file(selected),
            "final_checkpoint": f"checkpoints/epoch_{actual_epochs:03d}.pt",
            "final_sha256": sha256_file(root / f"checkpoints/epoch_{actual_epochs:03d}.pt"),
            "history_sha256": sha256_file(root / "history.csv"), "samples_sha256": sha256_file(root / "samples.json"),
            "config_sha256": sha256_file(root / "config.yaml"), "epochs_completed": actual_epochs,
            "optimizer_updates_completed": update_index, "schedule_total_updates": total_updates,
            "updates_per_epoch": updates_per_epoch, "stopping": stop_receipt,
            "parameters": sum(p.numel() for p in model.parameters()),
            "selector": "full_validation_local_rr_minimum_earliest_tie", "test_waveform_read": False,
            **_write_metrics(root, predictions, cfg),
        })
    return Path(output).resolve()


def evaluate_validation(*, run_root: str | Path, cache_root: str | Path, output: str | Path,
                        device: str | None = None, model_factory=RuntimeModel) -> Path:
    run_root = Path(run_root).resolve()
    source = verified_manifest(run_root, "train")
    if sha256_file(run_root / "config.yaml") != source["config_sha256"]:
        raise ValueError("训练配置被修改")
    cfg = load_experiment_config(run_root / "config.yaml")
    original_config_digest = spec_digest(OmegaConf.to_container(cfg, resolve=True))
    if original_config_digest != source["identity"]["config_sha256"]:
        raise ValueError("训练配置 identity 不匹配")
    if device is not None:
        cfg.training.device = device
    validate_experiment_config(cfg)
    with artifact_directory(output, kind="validation", cfg=cfg) as (root, started):
        if source["identity"]["code_sha256"] != started["code"]["sha256"]:
            raise ValueError("评价代码偏离训练执行代码；需明确修订协议")
        if read_json(run_root / "started.json")["dependencies"] != started["dependencies"]:
            raise ValueError("评价依赖版本偏离训练环境")
        if sha256_file(Path(cache_root) / "manifest.json") != source["identity"]["cache_manifest_sha256"]:
            raise ValueError("评价 cache 偏离训练 cache")
        selection, loaders = build_loaders(cfg, cache_root, splits=("val",))
        samples = _save_data_provenance(root, selection, loaders)
        if samples["val"] != source["identity"]["samples"]["val"]:
            raise ValueError("validation input/target/mask 偏离训练时内容")
        checkpoint_name = source["selected_checkpoint"]
        if not re.fullmatch(r"checkpoints/epoch_[0-9]{3}\.pt", checkpoint_name):
            raise ValueError("selected checkpoint 路径不合格")
        selected = run_root / checkpoint_name
        if sha256_file(selected) != source["selected_sha256"]:
            raise ValueError("selected checkpoint 哈希不匹配")
        actual_device = _device(cfg)
        model = model_factory(model_config(cfg)).to(actual_device)
        checkpoint = _load_checkpoint(selected, model=model, identity=source["identity"])
        if checkpoint["epoch"] != source["selected_epoch"]:
            raise ValueError("selected epoch 不匹配")
        predictions = collect_predictions(model, loaders["val"], device=actual_device,
                                          max_windows=len(loaders["val"].dataset), use_amp=cfg.training.use_amp)
        _check_predictions(predictions, selection.rows["val"])
        write_json(root / "manifest.json", {
            "kind": "validation", "protocol": PROTOCOL, "source_run": str(run_root),
            "source_manifest_sha256": sha256_file(run_root / "manifest.json"),
            "selected_epoch": checkpoint["epoch"], "checkpoint_sha256": source["selected_sha256"],
            "test_waveform_read": False, **_write_metrics(root, predictions, cfg),
        })
    return Path(output).resolve()
