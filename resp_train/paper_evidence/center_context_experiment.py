from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import crd_dependency_versions
from resp_train.crd.training import build_crd_optimizer, optimizer_updates_per_epoch, train_crd_one_epoch
from resp_train.engine import save_checkpoint, validate
from resp_train.paper_evidence.center_context_config import CENTER_CONTEXT_PROTOCOL_ID
from resp_train.paper_evidence.center_context_data import build_center_context_data
from resp_train.paper_evidence.center_context_loss import CenterContextLoss
from resp_train.paper_evidence.center_context_metrics import (
    center_rr_strictly_improved,
    evaluate_center_predictions,
    summarize_center_metrics,
    validation_center_rr_mean,
)
from resp_train.paper_evidence.center_context_model import build_center_context_model
from resp_train.utils.run import resolve_device, set_seed, setup_logger


def center_experiment_id(variant: str, input_samples: int) -> str:
    prefix = {"c201_center60": "CCV1_C201", "w_reduced_center60": "CCV1_WR"}.get(str(variant))
    if prefix is None or int(input_samples) not in {6000, 9000, 18000}:
        raise ValueError("未知中心上下文实验 arm")
    return f"{prefix}_{int(input_samples) // 100}"


class CenterContextExperiment:
    task_name = "paper_center_context_v1"

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def train(self) -> Path:
        role = str(self.cfg.protocol.run_role)
        if role == "implementation":
            raise RuntimeError("P1 implementation_only 配置不授权真实数据 lifecycle 或训练")
        if role != "formal":
            raise RuntimeError(f"当前训练入口未开放 run_role={role!r}")
        _assert_clean_repository()
        run_dir = self._create_identity_dir()
        lifecycle_path = run_dir / "lifecycle.json"
        self._write_lifecycle(lifecycle_path, status="running")
        try:
            result = self._train_into(run_dir)
        except BaseException as exc:
            self._write_lifecycle(
                lifecycle_path,
                status="failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        self._write_lifecycle(lifecycle_path, status="complete")
        return result

    def _create_identity_dir(self) -> Path:
        arm = center_experiment_id(str(self.cfg.model.variant), int(self.cfg.window.input_samples))
        output = Path(str(self.cfg.outputs.run_root)).resolve() / arm / f"seed_{int(self.cfg.training.seed)}"
        output.mkdir(parents=True, exist_ok=False)
        return output

    def _train_into(self, run_dir: Path) -> Path:
        OmegaConf.save(self.cfg, run_dir / "resolved_config.yaml")
        logger = setup_logger(run_dir)
        set_seed(int(self.cfg.training.seed))
        device = resolve_device(str(self.cfg.training.device))
        runtime = _configure_training_runtime(self.cfg, device)
        (run_dir / "runtime_identity.json").write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        data = build_center_context_data(self.cfg)
        data_identity = {
            **data.identity_audit,
            "dataset_index": str(data.index_path),
            "dataset_index_sha256": _sha256_file(data.index_path),
            "input_samples": int(self.cfg.window.input_samples),
            "input_slice": {
                6000: [6000, 12000],
                9000: [4500, 13500],
                18000: [0, 18000],
            }[int(self.cfg.window.input_samples)],
            "target_slice": [6000, 12000],
            "source_key_column": str(self.cfg.data.bcg_input_key),
            "target_key_column": str(self.cfg.data.target_key),
            "test_access": False,
        }
        (run_dir / "data_identity.json").write_text(
            json.dumps(data_identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        model = build_center_context_model(self.cfg).to(device)
        loss_fn = CenterContextLoss(self.cfg).to(device)
        optimizer, partition = build_crd_optimizer(model, self.cfg)
        (run_dir / "optimizer_parameter_groups.json").write_text(
            json.dumps(
                {"decay": list(partition.decay_names), "no_decay": list(partition.no_decay_names)},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        epochs = int(self.cfg.training.epochs)
        accumulation = int(self.cfg.training.gradient_accumulation_steps)
        updates_per_epoch = optimizer_updates_per_epoch(len(data.train_loader), accumulation)
        total_updates = epochs * updates_per_epoch
        update_index = 0
        history: list[dict[str, Any]] = []
        best_rr = float("inf")
        best_epoch: int | None = None
        for epoch in range(1, epochs + 1):
            train_summary, update_index = train_crd_one_epoch(
                model,
                data.train_loader,
                loss_fn,
                optimizer,
                device=device,
                accumulation_steps=accumulation,
                update_index=update_index,
                total_updates=total_updates,
                max_learning_rate=float(self.cfg.training.max_learning_rate),
                min_learning_rate=float(self.cfg.training.min_learning_rate),
                warmup_fraction=float(self.cfg.training.warmup_fraction),
                grad_clip_norm=float(self.cfg.training.grad_clip_norm),
                use_amp=bool(self.cfg.training.use_amp),
                show_progress=_show_progress(self.cfg),
                epoch=epoch,
                total_epochs=epochs,
            )
            val_summary, predictions = validate(
                model,
                data.val_loader,
                loss_fn,
                device=device,
                show_progress=_show_progress(self.cfg),
                epoch=epoch,
                total_epochs=epochs,
                return_predictions=True,
                use_amp=bool(self.cfg.training.use_amp),
            )
            center_rr = validation_center_rr_mean(predictions, self.cfg)
            improved = center_rr_strictly_improved(center_rr, best_rr)
            record = {
                "epoch": epoch,
                "optimizer_update": update_index,
                "train_loss_total": float(train_summary["loss"]),
                "train_loss_sync": float(train_summary["loss_sync"]),
                "train_loss_effort": float(train_summary["loss_effort"]),
                "val_loss_sync": float(val_summary["loss_sync"]),
                "val_loss_effort": float(val_summary["loss_effort"]),
                "val_center_rr_mae_bpm": center_rr,
                "strictly_improved": bool(improved),
            }
            history.append(record)
            pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
            logger.info(
                "epoch=%d/%d update=%d/%d train_loss=%.6f val_center_rr=%.6f improved=%s",
                epoch,
                epochs,
                update_index,
                total_updates,
                record["train_loss_total"],
                center_rr,
                improved,
            )
            extra = self._checkpoint_extra(update_index=update_index, total_updates=total_updates)
            if improved:
                best_rr = center_rr
                best_epoch = epoch
                save_checkpoint(
                    run_dir / "checkpoint_best_center_rr.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=record,
                    cfg=self.cfg,
                    extra_state=extra,
                )
        if update_index != total_updates or best_epoch is None:
            raise RuntimeError("中心训练未闭合 optimizer updates 或 selector checkpoint")
        save_checkpoint(
            run_dir / "checkpoint_final.pt",
            model=model,
            optimizer=optimizer,
            epoch=epochs,
            metrics=history[-1],
            cfg=self.cfg,
            extra_state=self._checkpoint_extra(update_index=update_index, total_updates=total_updates),
        )

        checkpoint = torch.load(run_dir / "checkpoint_best_center_rr.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, predictions = validate(
            model,
            data.val_loader,
            loss_fn,
            device=device,
            show_progress=_show_progress(self.cfg),
            return_predictions=True,
            use_amp=bool(self.cfg.training.use_amp),
        )
        metrics = evaluate_center_predictions(predictions, self.cfg, method=str(self.cfg.model.variant))
        metrics.to_csv(run_dir / "validation_center_metrics.csv", index=False)
        summarize_center_metrics(metrics).to_csv(run_dir / "validation_center_metrics_summary.csv", index=False)
        self._write_manifest(
            run_dir,
            model=model,
            best_epoch=best_epoch,
            best_rr=best_rr,
            runtime=runtime,
        )
        return run_dir

    def _checkpoint_extra(self, *, update_index: int, total_updates: int) -> dict[str, Any]:
        return {
            "protocol_id": CENTER_CONTEXT_PROTOCOL_ID,
            "selector": "full_validation_center_rr_mae_bpm_strict_lower_tie_earlier",
            "update_index": int(update_index),
            "total_updates": int(total_updates),
            "early_stopping": False,
            "resume_supported": False,
        }

    def _write_manifest(
        self,
        run_dir: Path,
        *,
        model: torch.nn.Module,
        best_epoch: int,
        best_rr: float,
        runtime: dict[str, Any],
    ) -> None:
        artifacts = {}
        for path in sorted(run_dir.iterdir()):
            if path.is_file() and path.name not in {"artifact_manifest.json", "lifecycle.json"}:
                artifacts[path.name] = {
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
        payload = {
            "protocol_id": CENTER_CONTEXT_PROTOCOL_ID,
            "task": self.task_name,
            "experiment_id": center_experiment_id(
                str(self.cfg.model.variant), int(self.cfg.window.input_samples)
            ),
            "seed": int(self.cfg.training.seed),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "best_epoch": int(best_epoch),
            "best_center_rr_mae_bpm": float(best_rr),
            "selector": "full_validation_center_rr_mae_bpm_strict_lower_tie_earlier",
            "train_access": True,
            "validation_access": True,
            "test_access": False,
            "test_cache_created": False,
            "outer_target_supervision": False,
            "checkpoint_initialization_used": False,
            "scientific_role": "auxiliary_context_sensitivity_outside_main_model_experiment",
            "strict_w0_equivalent": False,
            "w_scale_count": 49 if str(self.cfg.model.variant) == "w_reduced_center60" else 0,
            "early_stopping": False,
            "resume": False,
            "dependencies": _dependency_versions(),
            "runtime": runtime,
            "git": _git_identity(),
            "command": list(sys.argv),
            "artifacts": artifacts,
        }
        (run_dir / "artifact_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_lifecycle(path: Path, *, status: str, **extra: Any) -> None:
        path.write_text(
            json.dumps({"status": status, **extra}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _show_progress(cfg: DictConfig) -> bool | None:
    value = cfg.training.get("show_progress", None)
    if value in (None, "auto"):
        return None
    return bool(value)


def _configure_training_runtime(cfg: DictConfig, device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("中心 formal training runtime 要求 CUDA")
    allow_tf32 = bool(cfg.training.allow_tf32)
    cudnn_benchmark = bool(cfg.training.cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark
    observed = {
        "device_type": device.type,
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "amp_enabled": bool(cfg.training.use_amp),
        "amp_dtype": str(cfg.training.amp_dtype),
    }
    if (
        observed["matmul_allow_tf32"] is not allow_tf32
        or observed["cudnn_allow_tf32"] is not allow_tf32
        or observed["cudnn_benchmark"] is not cudnn_benchmark
    ):
        raise RuntimeError("中心 formal CUDA runtime 状态未按配置生效")
    return observed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = dict(crd_dependency_versions())
    for distribution in ("numpy", "pandas", "torch", "omegaconf", "scipy", "ssqueezepy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _git_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "error": None if commit.returncode == 0 and status.returncode == 0 else "git identity failed",
    }


def _assert_clean_repository() -> None:
    identity = _git_identity()
    if identity["error"] is not None:
        raise RuntimeError("无法检查中心任务 Git identity")
    if identity["dirty"] is not False:
        raise RuntimeError("中心任务 lifecycle/formal 要求干净 Git 工作树")


__all__ = ["CenterContextExperiment", "center_experiment_id"]
