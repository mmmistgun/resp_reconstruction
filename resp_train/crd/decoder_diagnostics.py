from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from resp_train.crd.candidate_lock import sha256_file
from resp_train.crd.config import CRD_CONTROLS_PROTOCOL_VERSION, load_crd_config
from resp_train.crd.experiment import _validate_checkpoint_config
from resp_train.crd.model import build_crd_model
from resp_train.data.factory import build_window_data
from resp_train.utils.run import resolve_device, save_execution_manifest, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/crd_v1/crd_c2_decoder_diagnostics"
WINDOW_FILENAME = "c202_residual_spectral_windows.csv"
SUMMARY_FILENAME = "c202_residual_spectral_summary.json"
MANIFEST_FILENAME = "c202_residual_spectral_manifest.json"


def residual_spectral_rows(residual: torch.Tensor, *, sample_rate: float = 100.0) -> dict[str, np.ndarray]:
    """计算 C202 `Pi` 前 residual 的逐窗口带内/带外能量比例。"""

    if residual.ndim != 3 or residual.shape[1:] != (1, 18000):
        raise ValueError(f"C202 residual 期望 (B,1,18000)，实际 {tuple(residual.shape)}")
    if not bool(torch.isfinite(residual).all()):
        raise FloatingPointError("C202 residual 包含 NaN/Inf")
    with torch.amp.autocast(residual.device.type, enabled=False):
        work = residual.float().squeeze(1)
        work = work - work.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(work, dim=-1, norm="forward")
        energy = spectrum.abs().square()
        frequencies = torch.fft.rfftfreq(work.shape[-1], d=1.0 / float(sample_rate), device=work.device)
        non_dc = frequencies > 0.0
        in_band = (frequencies >= 0.05) & (frequencies <= 0.70)
        out_of_band = frequencies > 0.70
        total = energy[:, non_dc].sum(dim=-1)
        if not bool((total > 0.0).all()):
            raise FloatingPointError("C202 residual total non-DC energy 为 0")
        in_band_energy = energy[:, in_band].sum(dim=-1)
        out_of_band_energy = energy[:, out_of_band].sum(dim=-1)
        return {
            "residual_total_energy": total.cpu().numpy(),
            "residual_in_band_energy": in_band_energy.cpu().numpy(),
            "residual_out_of_band_energy": out_of_band_energy.cpu().numpy(),
            "residual_in_band_energy_fraction": (in_band_energy / total).cpu().numpy(),
            "residual_out_of_band_energy_fraction": (out_of_band_energy / total).cpu().numpy(),
        }


@torch.no_grad()
def evaluate_c202_residual_spectrum(
    *,
    checkpoint_path: str | Path,
    device: str = "cuda:0",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """只读完整 validation，生成一次 C202 selected-checkpoint residual 频谱描述。"""

    _assert_clean_repository()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists() or checkpoint_path.name != "checkpoint_best_local_rr.pt":
        raise FileNotFoundError(f"C202 selected checkpoint 不存在或文件名错误: {checkpoint_path}")
    cfg = load_crd_config(
        checkpoint_path.parent / "config.yaml",
        overrides=[f"training.device={device}", "training.show_progress=false"],
    )
    if (
        str(cfg.protocol.name) != CRD_CONTROLS_PROTOCOL_VERSION
        or str(cfg.protocol.stage) != "c2"
        or str(cfg.protocol.run_role) != "formal"
        or str(cfg.model.variant) != "crd_c202_decoder_100hz"
        or cfg.data.get("max_val_windows") is not None
    ):
        raise ValueError("residual spectrum 只允许 C202 formal selected checkpoint/完整 validation")
    seed = int(cfg.training.seed)
    final_dir = Path(output_root) / str(cfg.model.variant) / f"seed_{seed}"
    if final_dir.exists():
        raise FileExistsError(f"C202 residual diagnostic 禁止覆盖: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = final_dir.parent / f".{final_dir.name}.{uuid4().hex}.tmp"
    temporary_dir.mkdir(parents=False, exist_ok=False)
    try:
        resolved_device = resolve_device(device)
        set_seed(seed)
        model = build_crd_model(cfg).to(resolved_device).eval()
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
        _validate_checkpoint_config(checkpoint.get("config"), cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        selected_epoch = int(checkpoint["epoch"])

        validation = build_window_data(
            cfg,
            split=str(cfg.data.val_split),
            max_windows=None,
            sample_strategy=str(cfg.data.val_sample_strategy),
            sample_seed=int(cfg.data.val_sample_seed),
            shuffle=False,
        )
        rows: list[dict[str, Any]] = []
        amp_enabled = resolved_device.type == "cuda" and bool(cfg.training.use_amp)
        for batch in validation.loader:
            if "meta" not in batch:
                raise KeyError("C202 validation batch 缺少 meta")
            sensor = batch["x"].to(resolved_device, non_blocking=resolved_device.type == "cuda")
            with torch.amp.autocast(
                resolved_device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                output = model(sensor)
            residual = output.get("decoder_residual_100hz")
            if residual is None:
                raise KeyError("C202 output 缺少 decoder_residual_100hz")
            spectral = residual_spectral_rows(residual, sample_rate=float(cfg.window.target_fs))
            metadata = batch["meta"]
            batch_size = int(residual.shape[0])
            for index in range(batch_size):
                rows.append(
                    {
                        "variant": str(cfg.model.variant),
                        "seed": seed,
                        "selected_epoch": selected_epoch,
                        "dataset_row_id": int(metadata["dataset_row_id"][index]),
                        "samp_id": int(metadata["samp_id"][index]),
                        "coupling_state_id": int(metadata["coupling_state_id"][index]),
                        **{name: float(values[index]) for name, values in spectral.items()},
                    }
                )
        frame = pd.DataFrame(rows).sort_values("dataset_row_id").reset_index(drop=True)
        if (
            len(frame) != 2675
            or frame["dataset_row_id"].duplicated().any()
            or int(frame["samp_id"].nunique()) != 7
            or not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all()
        ):
            raise RuntimeError("C202 residual diagnostic validation identity/finite 检查失败")
        fractions = frame["residual_out_of_band_energy_fraction"].to_numpy(dtype=np.float64)
        summary = {
            "protocol": CRD_CONTROLS_PROTOCOL_VERSION,
            "variant": str(cfg.model.variant),
            "seed": seed,
            "selected_epoch": selected_epoch,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "validation_window_count": int(len(frame)),
            "validation_samp_id_count": int(frame["samp_id"].nunique()),
            "out_of_band_energy_fraction_mean": float(np.mean(fractions)),
            "out_of_band_energy_fraction_sd": float(np.std(fractions, ddof=1)),
            "out_of_band_energy_fraction_median": float(np.median(fractions)),
            "out_of_band_energy_fraction_p05": float(np.quantile(fractions, 0.05)),
            "out_of_band_energy_fraction_p95": float(np.quantile(fractions, 0.95)),
            "research_test_used": False,
            "selection_metric": False,
        }
        frame.to_csv(temporary_dir / WINDOW_FILENAME, index=False)
        (temporary_dir / SUMMARY_FILENAME).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        save_execution_manifest(
            temporary_dir / MANIFEST_FILENAME,
            task="crd_c202_residual_spectrum",
            phase="c2_100hz_decoder_residual_diagnostic",
            protocol=CRD_CONTROLS_PROTOCOL_VERSION,
            variant=str(cfg.model.variant),
            seed=seed,
            selected_epoch=selected_epoch,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=sha256_file(checkpoint_path),
            validation_window_count=int(len(frame)),
            validation_samp_id_count=int(frame["samp_id"].nunique()),
            research_test_used=False,
        )
        os.replace(temporary_dir, final_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return final_dir / SUMMARY_FILENAME


def _assert_clean_repository() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(f"无法检查 Git 工作树: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError("C202 residual diagnostic 要求干净 Git 工作树")

