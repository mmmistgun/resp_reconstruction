from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import crd_dependency_versions
from resp_train.crd.training import build_crd_optimizer, optimizer_updates_per_epoch, train_crd_one_epoch
from resp_train.engine import save_checkpoint, validate
from resp_train.paper_evidence.center90_config import CENTER90_INPUT_SAMPLES, CENTER90_PROTOCOL_ID
from resp_train.paper_evidence.center90_data import INPUT_SLICES, TARGET_SLICE, build_center90_data
from resp_train.paper_evidence.center90_loss import Center90ContextLoss
from resp_train.paper_evidence.center90_metrics import (
    center90_rr_strictly_improved,
    evaluate_center90_predictions,
    summarize_center90_metrics,
    validation_center90_rr_mean,
)
from resp_train.paper_evidence.center90_model import build_center90_model
from resp_train.paper_evidence.center_context_experiment import _configure_training_runtime
from resp_train.utils.run import resolve_device, set_seed, setup_logger


def center90_experiment_id(variant: str, input_samples: int) -> str:
    prefix = {"c201_center90": "C90V1_C201", "w_reduced_center90": "C90V1_WR"}.get(
        str(variant)
    )
    if prefix is None or int(input_samples) not in CENTER90_INPUT_SAMPLES:
        raise ValueError("未知 center90 arm")
    return f"{prefix}_{int(input_samples) // 100}"


class Center90Experiment:
    task_name = "paper_center90_context_v1"

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def train(self) -> Path:
        if str(self.cfg.protocol.run_role) != "formal":
            raise RuntimeError("center90 implementation config 不创建真实 lifecycle")
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
        arm = center90_experiment_id(str(self.cfg.model.variant), int(self.cfg.window.input_samples))
        output = Path(str(self.cfg.outputs.run_root)).resolve() / arm / f"seed_{int(self.cfg.training.seed)}"
        output.mkdir(parents=True, exist_ok=False)
        return output

    def _train_into(self, run_dir: Path) -> Path:
        OmegaConf.save(self.cfg, run_dir / "resolved_config.yaml")
        logger = setup_logger(run_dir)
        set_seed(int(self.cfg.training.seed))
        device = resolve_device(str(self.cfg.training.device))
        runtime = _configure_training_runtime(self.cfg, device)
        _write_json(run_dir / "runtime_identity.json", runtime)
        data = build_center90_data(self.cfg)
        input_samples = int(self.cfg.window.input_samples)
        data_identity = {
            **data.identity_audit,
            "dataset_index": str(data.index_path),
            "dataset_index_sha256": _sha256_file(data.index_path),
            "input_samples": input_samples,
            "input_slice": list(INPUT_SLICES[input_samples]),
            "target_slice": list(TARGET_SLICE),
            "source_key_column": str(self.cfg.data.bcg_input_key),
            "target_key_column": str(self.cfg.data.target_key),
            "test_access": False,
        }
        _write_json(run_dir / "data_identity.json", data_identity)
        model = build_center90_model(self.cfg).to(device)
        loss_fn = Center90ContextLoss(self.cfg).to(device)
        optimizer, partition = build_crd_optimizer(model, self.cfg)
        _write_json(
            run_dir / "optimizer_parameter_groups.json",
            {"decay": list(partition.decay_names), "no_decay": list(partition.no_decay_names)},
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
            center_rr = validation_center90_rr_mean(predictions, self.cfg)
            improved = center90_rr_strictly_improved(center_rr, best_rr)
            record = {
                "epoch": epoch,
                "optimizer_update": update_index,
                "train_loss_total": float(train_summary["loss"]),
                "train_loss_sync": float(train_summary["loss_sync"]),
                "train_loss_effort": float(train_summary["loss_effort"]),
                "val_loss_sync": float(val_summary["loss_sync"]),
                "val_loss_effort": float(val_summary["loss_effort"]),
                "val_center90_rr_mae_bpm": center_rr,
                "strictly_improved": bool(improved),
            }
            history.append(record)
            pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
            logger.info(
                "epoch=%d/%d update=%d/%d train_loss=%.6f val_center90_rr=%.6f improved=%s",
                epoch,
                epochs,
                update_index,
                total_updates,
                record["train_loss_total"],
                center_rr,
                improved,
            )
            if improved:
                best_rr = center_rr
                best_epoch = epoch
                save_checkpoint(
                    run_dir / "checkpoint_best_center90_rr.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=record,
                    cfg=self.cfg,
                    extra_state=self._checkpoint_extra(update_index=update_index, total_updates=total_updates),
                )
        if update_index != total_updates or best_epoch is None:
            raise RuntimeError("center90 optimizer updates 或 selector checkpoint 未闭合")
        save_checkpoint(
            run_dir / "checkpoint_final.pt",
            model=model,
            optimizer=optimizer,
            epoch=epochs,
            metrics=history[-1],
            cfg=self.cfg,
            extra_state=self._checkpoint_extra(update_index=update_index, total_updates=total_updates),
        )
        checkpoint = torch.load(run_dir / "checkpoint_best_center90_rr.pt", map_location=device)
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
        metrics = evaluate_center90_predictions(predictions, self.cfg, method=str(self.cfg.model.variant))
        metrics.to_csv(run_dir / "validation_center90_metrics.csv", index=False)
        summarize_center90_metrics(metrics).to_csv(
            run_dir / "validation_center90_metrics_summary.csv", index=False
        )
        self._write_manifest(run_dir, model=model, best_epoch=best_epoch, best_rr=best_rr, runtime=runtime)
        return run_dir

    def _checkpoint_extra(self, *, update_index: int, total_updates: int) -> dict[str, Any]:
        return {
            "protocol_id": CENTER90_PROTOCOL_ID,
            "selector": "full_validation_center90_rr_mae_bpm_strict_lower_tie_earlier",
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
        artifacts = {
            path.name: {"size_bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name not in {"artifact_manifest.json", "lifecycle.json"}
        }
        payload = {
            "protocol_id": CENTER90_PROTOCOL_ID,
            "task": self.task_name,
            "experiment_id": center90_experiment_id(
                str(self.cfg.model.variant), int(self.cfg.window.input_samples)
            ),
            "seed": int(self.cfg.training.seed),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "best_epoch": int(best_epoch),
            "best_center90_rr_mae_bpm": float(best_rr),
            "selector": "full_validation_center90_rr_mae_bpm_strict_lower_tie_earlier",
            "train_access": True,
            "validation_access": True,
            "test_access": False,
            "test_cache_created": False,
            "outer_target_supervision": False,
            "checkpoint_initialization_used": False,
            "scientific_role": "independent_center90_output_context_scale_experiment",
            "representation": (
                "reduced_w_49_scale" if str(self.cfg.model.variant) == "w_reduced_center90" else "time_domain"
            ),
            "w_scale_count": 49 if str(self.cfg.model.variant) == "w_reduced_center90" else 0,
            "early_stopping": False,
            "resume": False,
            "dependencies": _dependency_versions(),
            "runtime": runtime,
            "git": _git_identity(),
            "command": list(sys.argv),
            "artifacts": artifacts,
        }
        _write_json(run_dir / "artifact_manifest.json", payload)

    @staticmethod
    def _write_lifecycle(path: Path, *, status: str, **extra: Any) -> None:
        _write_json(path, {"status": status, **extra})


def _show_progress(cfg: DictConfig) -> bool | None:
    value = cfg.training.get("show_progress", None)
    return None if value in (None, "auto") else bool(value)


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
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "error": None if commit.returncode == 0 and status.returncode == 0 else "git identity failed",
    }


def _assert_clean_repository() -> None:
    identity = _git_identity()
    if identity["error"] is not None or identity["dirty"] is not False:
        raise RuntimeError("center90 formal lifecycle 要求干净 Git 工作树")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = ["Center90Experiment", "center90_experiment_id"]
