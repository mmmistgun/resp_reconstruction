from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from resp_train.paper_evidence.center90_cache import open_center90_w_cache
from resp_train.paper_evidence.center90_config import (
    CENTER90_FORMAL_OUTPUT_ROOT,
    CENTER90_FORMAL_SEEDS,
    CENTER90_W_CACHE_MANIFEST_SHA256,
    CENTER90_W_CACHE_PATHS,
    load_center90_config,
)
from resp_train.paper_evidence.center90_experiment import center90_experiment_id


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/paper_evidence_v1"
ACCEPTANCE_ROOT = (
    ROOT / "runs/paper_evidence_v1/center90_context/gpu_acceptance/2c151cba2e41"
)
ACCEPTANCE_RECEIPT_SHA256 = "36ab2e4c9c2d3276078728bf657463619b82e02776aa4ff4238101643c891b87"
ACCEPTANCE_MANIFEST_SHA256 = "721fe0a8d59460809666723daf9ba9dee9cfde1f4aab28934297d83cd1e25935"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_configs() -> dict[str, tuple[str, int, int]]:
    return {
        f"c90v1_{short}_{input_sec}_seed{seed}.yaml": (variant, input_sec, seed)
        for short, variant in (("c201", "c201_center90"), ("wr", "w_reduced_center90"))
        for input_sec in (90, 135, 180)
        for seed in CENTER90_FORMAL_SEEDS
    }


def test_center90_eighteen_formal_configs_freeze_full_matrix() -> None:
    expected = _expected_configs()
    paths = sorted(CONFIG_ROOT.glob("c90v1_*_seed*.yaml"))
    assert {path.name for path in paths} == set(expected)
    identities = set()
    for path in paths:
        variant, input_sec, seed = expected[path.name]
        cfg = load_center90_config(path)
        assert cfg.protocol.stage == "formal"
        assert cfg.protocol.run_role == "formal"
        assert cfg.protocol.execution_gate == "formal_full_matrix"
        assert cfg.model.variant == variant
        assert cfg.window.input_sec == input_sec
        assert cfg.window.input_samples == input_sec * 100
        assert cfg.window.output_samples == 9000
        assert cfg.training.seed == cfg.model.initialization_seed == seed
        assert (cfg.training.epochs, cfg.training.batch_size, cfg.training.gradient_accumulation_steps) == (
            80,
            128,
            1,
        )
        assert cfg.training.device == "cuda:0"
        assert cfg.data.access_splits == ["train", "val"]
        assert cfg.data.max_train_windows is cfg.data.max_val_windows is cfg.data.max_test_windows is None
        assert cfg.outputs.run_root == CENTER90_FORMAL_OUTPUT_ROOT
        assert center90_experiment_id(variant, input_sec * 100) == (
            f"C90V1_{'C201' if variant == 'c201_center90' else 'WR'}_{input_sec}"
        )
        if variant == "w_reduced_center90":
            assert Path(str(cfg.data.center_w_cache_path)).resolve() == Path(
                CENTER90_W_CACHE_PATHS[input_sec * 100]
            ).resolve()
            assert (
                cfg.data.center_w_cache_manifest_sha256
                == CENTER90_W_CACHE_MANIFEST_SHA256[input_sec * 100]
            )
        else:
            assert cfg.data.center_w_cache_path is cfg.data.center_w_cache_manifest_sha256 is None
        identities.add((variant, input_sec, seed))
    assert len(identities) == 18


def test_center90_all_three_w_caches_are_complete_and_hash_pinned() -> None:
    for samples, path_text in CENTER90_W_CACHE_PATHS.items():
        root = Path(path_text)
        assert _sha256(root / "cache_manifest.json") == CENTER90_W_CACHE_MANIFEST_SHA256[samples]
        reader = open_center90_w_cache(
            root,
            split="val",
            input_samples=samples,
            expected_manifest_sha256=CENTER90_W_CACHE_MANIFEST_SHA256[samples],
        )
        assert reader.features.shape == (2675, 49, samples // 50)


def test_center90_gpu_acceptance_is_frozen_and_batch128_accepted() -> None:
    receipt_path = ACCEPTANCE_ROOT / "acceptance_receipt.json"
    manifest_path = ACCEPTANCE_ROOT / "artifact_manifest.json"
    lifecycle_path = ACCEPTANCE_ROOT / "lifecycle.json"
    assert _sha256(receipt_path) == ACCEPTANCE_RECEIPT_SHA256
    assert _sha256(manifest_path) == ACCEPTANCE_MANIFEST_SHA256
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert lifecycle["status"] == "complete"
    assert lifecycle["decision"] == receipt["decision"] == "batch128_accepted"
    assert receipt["git_commit"] == "2c151cba2e41056a115dd5f7f0293fa486666c8d"
    assert receipt["git_dirty"] is False
    assert receipt["access"] == {
        "dataset_accessed": False,
        "cache_accessed": False,
        "train_split_accessed": False,
        "validation_split_accessed": False,
        "test_split_accessed": False,
        "checkpoint_accessed": False,
        "synthetic_tensor_used": True,
        "optimizer_step_used": True,
    }
    result = receipt["result"]
    assert result["waveform_shape"] == [128, 1, 9000]
    assert result["waveform_10hz_shape"] == [128, 1, 900]
    assert result["parameter_count"] == 1219850
    assert result["finite"] is True
    assert result["peak_allocated_mib"] == pytest.approx(11021.49951171875)
    assert result["peak_reserved_mib"] == pytest.approx(11628.0)
    for filename, record in manifest["files"].items():
        path = ACCEPTANCE_ROOT / filename
        assert path.stat().st_size == record["size_bytes"]
        assert _sha256(path) == record["sha256"]


def test_center90_formal_gate_and_cli_reject_identity_drift() -> None:
    c201 = CONFIG_ROOT / "c90v1_c201_90_seed20260811.yaml"
    with pytest.raises(ValueError, match="只开放 seeds"):
        load_center90_config(
            c201,
            overrides=["training.seed=1", "model.initialization_seed=1"],
        )
    with pytest.raises(ValueError, match="固定输出根"):
        load_center90_config(c201, overrides=["outputs.run_root=runs/other"])
    with pytest.raises(ValueError, match="physical batch"):
        load_center90_config(c201, overrides=["training.batch_size=64"])
    with pytest.raises(ValueError, match="allow_tf32"):
        load_center90_config(c201, overrides=["training.allow_tf32=true"])
    with pytest.raises(ValueError, match="test windows"):
        load_center90_config(c201, overrides=["data.max_test_windows=1"])
    wr = CONFIG_ROOT / "c90v1_wr_135_seed20260811.yaml"
    with pytest.raises(ValueError, match="manifest SHA-256"):
        load_center90_config(wr, overrides=["data.center_w_cache_manifest_sha256=deadbeef"])
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_paper_center90_context_v1.py"),
            "--config",
            str(c201),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "confirm-formal-training" in result.stderr
