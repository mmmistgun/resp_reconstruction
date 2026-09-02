from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from resp_train.paper_evidence.center30_cache import Center30WCacheReader
from resp_train.paper_evidence.center30_config import (
    CENTER30_P4S_OUTPUT_ROOT,
    CENTER30_W_CACHE_MANIFEST_SHA256,
    CENTER30_W_CACHE_PATHS,
    load_center30_config,
)
from resp_train.paper_evidence.center30_experiment import Center30Experiment, center30_experiment_id


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/paper_evidence_v1"
EXPECTED = {
    "p4s_c30v1_c201_30.yaml": ("c201_center30", 3000, "C30V1_C201_30"),
    "p4s_c30v1_c201_45.yaml": ("c201_center30", 4500, "C30V1_C201_45"),
    "p4s_c30v1_c201_60.yaml": ("c201_center30", 6000, "C30V1_C201_60"),
    "p4s_c30v1_c201_90.yaml": ("c201_center30", 9000, "C30V1_C201_90"),
    "p4s_c30v1_wr_30.yaml": ("w_reduced_center30", 3000, "C30V1_WR_30"),
    "p4s_c30v1_wr_45.yaml": ("w_reduced_center30", 4500, "C30V1_WR_45"),
    "p4s_c30v1_wr_60.yaml": ("w_reduced_center30", 6000, "C30V1_WR_60"),
    "p4s_c30v1_wr_90.yaml": ("w_reduced_center30", 9000, "C30V1_WR_90"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_p4s_eight_configs_freeze_matrix_seed_batch_and_output() -> None:
    paths = sorted(CONFIG_ROOT.glob("p4s_c30v1_*.yaml"))
    assert {path.name for path in paths} == set(EXPECTED)
    identities = set()
    for path in paths:
        variant, samples, experiment_id = EXPECTED[path.name]
        cfg = load_center30_config(path)
        assert cfg.protocol.stage == "p4s_single_seed"
        assert cfg.protocol.run_role == "formal"
        assert cfg.protocol.execution_gate == "p4s_formal"
        assert cfg.model.variant == variant
        assert cfg.window.input_samples == samples
        assert cfg.window.output_samples == 3000
        assert cfg.training.seed == cfg.model.initialization_seed == 20260811
        assert (cfg.training.epochs, cfg.training.batch_size, cfg.training.gradient_accumulation_steps) == (
            80,
            128,
            1,
        )
        assert cfg.training.device == "cuda:0"
        assert cfg.data.access_splits == ["train", "val"]
        assert cfg.data.max_train_windows is cfg.data.max_val_windows is cfg.data.max_test_windows is None
        assert cfg.outputs.run_root == CENTER30_P4S_OUTPUT_ROOT
        assert center30_experiment_id(variant, samples) == experiment_id
        identities.add((variant, samples))
    assert len(identities) == 8


def test_p4s_four_w_configs_pin_complete_cache_hashes() -> None:
    for samples, path_text in CENTER30_W_CACHE_PATHS.items():
        manifest = Path(path_text) / "cache_manifest.json"
        assert _sha256(manifest) == CENTER30_W_CACHE_MANIFEST_SHA256[samples]
        reader = Center30WCacheReader(path_text, split="val", input_samples=samples)
        assert reader.features.shape == (2675, 49, samples // 50)


def test_p4s_gate_rejects_seed_output_cache_runtime_and_test_drift() -> None:
    c201 = CONFIG_ROOT / "p4s_c30v1_c201_30.yaml"
    with pytest.raises(ValueError, match="只开放 seed"):
        load_center30_config(
            c201,
            overrides=["training.seed=20260812", "model.initialization_seed=20260812"],
        )
    with pytest.raises(ValueError, match="输出"):
        load_center30_config(c201, overrides=["outputs.run_root=runs/other"])
    with pytest.raises(ValueError, match="allow_tf32"):
        load_center30_config(c201, overrides=["training.allow_tf32=true"])
    with pytest.raises(ValueError, match="test windows"):
        load_center30_config(c201, overrides=["data.max_test_windows=1"])
    wr = CONFIG_ROOT / "p4s_c30v1_wr_30.yaml"
    with pytest.raises(ValueError, match="manifest SHA-256"):
        load_center30_config(wr, overrides=["data.center_w_cache_manifest_sha256=deadbeef"])


def test_center30_experiment_identity_and_cli_confirmation_gate() -> None:
    implementation = load_center30_config(CONFIG_ROOT / "center30_context_v1.yaml")
    with pytest.raises(RuntimeError, match="不授权"):
        Center30Experiment(implementation).train()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_paper_center30_context_v1.py"),
            "--config",
            "configs/paper_evidence_v1/p4s_c30v1_c201_30.yaml",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "confirm-formal-training" in result.stderr
