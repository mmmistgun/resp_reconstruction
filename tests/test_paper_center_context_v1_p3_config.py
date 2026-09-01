from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from resp_train.paper_evidence.center_context_config import (
    CENTER_CONTEXT_P3_OUTPUT_ROOT,
    CENTER_CONTEXT_P3_W_CACHE_MANIFEST_SHA256,
    CENTER_CONTEXT_P3_W_CACHE_PATHS,
    load_center_context_config,
)
from resp_train.paper_evidence.center_context_experiment import (
    _configure_training_runtime,
    center_experiment_id,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/paper_evidence_v1"
EXPECTED = {
    "p3_ccv1_c201_60.yaml": ("c201_center60", 6000, "CCV1_C201_60"),
    "p3_ccv1_c201_90.yaml": ("c201_center60", 9000, "CCV1_C201_90"),
    "p3_ccv1_c201_180.yaml": ("c201_center60", 18000, "CCV1_C201_180"),
    "p3_ccv1_wr_60.yaml": ("w_reduced_center60", 6000, "CCV1_WR_60"),
    "p3_ccv1_wr_90.yaml": ("w_reduced_center60", 9000, "CCV1_WR_90"),
    "p3_ccv1_wr_180.yaml": ("w_reduced_center60", 18000, "CCV1_WR_180"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_p3_six_configs_freeze_matrix_seed_batch_and_output_identity() -> None:
    paths = sorted(CONFIG_ROOT.glob("p3_ccv1_*.yaml"))
    assert {path.name for path in paths} == set(EXPECTED)
    observed_ids = []
    for path in paths:
        variant, samples, experiment_id = EXPECTED[path.name]
        cfg = load_center_context_config(path)
        assert cfg.protocol.stage == "p3_single_seed"
        assert cfg.protocol.run_role == "formal"
        assert cfg.protocol.execution_gate == "p3_formal"
        assert cfg.model.variant == variant
        assert cfg.window.input_samples == samples
        assert cfg.window.input_sec == samples // 100
        assert cfg.training.seed == cfg.model.initialization_seed == 20260811
        assert (cfg.training.epochs, cfg.training.batch_size, cfg.training.gradient_accumulation_steps) == (
            80,
            128,
            1,
        )
        assert cfg.training.allow_tf32 is False
        assert cfg.training.cudnn_benchmark is False
        assert cfg.data.max_train_windows is None
        assert cfg.data.max_val_windows is None
        assert cfg.data.max_test_windows is None
        assert cfg.outputs.run_root == CENTER_CONTEXT_P3_OUTPUT_ROOT
        assert center_experiment_id(variant, samples) == experiment_id
        observed_ids.append(experiment_id)
    assert len(set(observed_ids)) == 6


def test_p3_w_configs_pin_complete_cache_manifest_and_common_rows() -> None:
    train_hashes = set()
    val_hashes = set()
    dataset_hashes = set()
    for samples, cache_path_text in CENTER_CONTEXT_P3_W_CACHE_PATHS.items():
        cache_path = Path(cache_path_text)
        manifest_path = cache_path / "cache_manifest.json"
        assert _sha256(manifest_path) == CENTER_CONTEXT_P3_W_CACHE_MANIFEST_SHA256[samples]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["complete"] is True
        assert manifest["input_samples"] == samples
        assert manifest["frequency_count"] == 49
        assert manifest["input_only"] is True
        assert manifest["target_read"] is False
        assert manifest["test_read"] is False
        assert manifest["splits"]["train"]["count"] == 10141
        assert manifest["splits"]["val"]["count"] == 2675
        train_hashes.add(manifest["splits"]["train"]["row_ids_sha256"])
        val_hashes.add(manifest["splits"]["val"]["row_ids_sha256"])
        dataset_hashes.add(manifest["dataset_index_sha256"])
    assert len(train_hashes) == len(val_hashes) == len(dataset_hashes) == 1


def test_p3_gate_rejects_extra_seed_output_or_cache_identity_drift() -> None:
    with pytest.raises(ValueError, match="template_only"):
        load_center_context_config(CONFIG_ROOT / "_p3_base.yaml")
    c201 = CONFIG_ROOT / "p3_ccv1_c201_60.yaml"
    with pytest.raises(ValueError, match="只开放 seed"):
        load_center_context_config(
            c201,
            overrides=["training.seed=20260812", "model.initialization_seed=20260812"],
        )
    with pytest.raises(ValueError, match="outputs.run_root"):
        load_center_context_config(c201, overrides=["outputs.run_root=runs/other"])
    with pytest.raises(ValueError, match="allow_tf32"):
        load_center_context_config(c201, overrides=["training.allow_tf32=true"])
    with pytest.raises(ValueError, match="cudnn_benchmark"):
        load_center_context_config(c201, overrides=["training.cudnn_benchmark=true"])
    wr = CONFIG_ROOT / "p3_ccv1_wr_60.yaml"
    with pytest.raises(ValueError, match="manifest SHA-256"):
        load_center_context_config(
            wr,
            overrides=["data.center_w_cache_manifest_sha256=deadbeef"],
        )


def test_p3_runtime_applies_and_reports_explicit_cuda_numeric_flags() -> None:
    cfg = load_center_context_config(CONFIG_ROOT / "p3_ccv1_c201_60.yaml")
    original = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
    )
    try:
        observed = _configure_training_runtime(cfg, torch.device("cuda:0"))
        assert observed == {
            "device_type": "cuda",
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "amp_enabled": True,
            "amp_dtype": "bfloat16",
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original[0]
        torch.backends.cudnn.allow_tf32 = original[1]
        torch.backends.cudnn.benchmark = original[2]
