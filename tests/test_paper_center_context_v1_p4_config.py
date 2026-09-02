from __future__ import annotations

from pathlib import Path

import pytest

from resp_train.paper_evidence.center_context_config import (
    CENTER_CONTEXT_P4_OUTPUT_ROOT,
    load_center_context_config,
)
from resp_train.paper_evidence.center_context_experiment import center_experiment_id


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/paper_evidence_v1"
VARIANTS = {
    "c201": "c201_center60",
    "wr": "w_reduced_center60",
}


def _expected_configs() -> dict[str, tuple[str, int, int]]:
    return {
        f"p4_ccv1_{short}_{input_sec}_seed{seed}.yaml": (variant, input_sec * 100, seed)
        for short, variant in VARIANTS.items()
        for input_sec in (60, 90, 180)
        for seed in (20260812, 20260813)
    }


def test_p4_twelve_configs_freeze_full_matrix_and_additional_seeds() -> None:
    expected = _expected_configs()
    paths = sorted(CONFIG_ROOT.glob("p4_ccv1_*.yaml"))
    assert {path.name for path in paths} == set(expected)
    identities = set()
    for path in paths:
        variant, input_samples, seed = expected[path.name]
        cfg = load_center_context_config(path)
        assert cfg.protocol.stage == "p4_additional_seeds"
        assert cfg.protocol.run_role == "formal"
        assert cfg.protocol.execution_gate == "p4_formal"
        assert cfg.model.variant == variant
        assert cfg.window.input_samples == input_samples
        assert cfg.window.input_sec == input_samples // 100
        assert cfg.training.seed == cfg.model.initialization_seed == seed
        assert (cfg.training.epochs, cfg.training.batch_size, cfg.training.gradient_accumulation_steps) == (
            80,
            128,
            1,
        )
        assert cfg.training.device == "cuda:0"
        assert cfg.training.allow_tf32 is False
        assert cfg.training.cudnn_benchmark is False
        assert cfg.data.access_splits == ["train", "val"]
        assert cfg.data.max_test_windows is None
        assert cfg.outputs.run_root == CENTER_CONTEXT_P4_OUTPUT_ROOT
        identities.add((center_experiment_id(variant, input_samples), seed))
    assert len(identities) == 12


def test_p4_gate_rejects_seed_stage_output_runtime_and_test_drift() -> None:
    path = CONFIG_ROOT / "p4_ccv1_c201_60_seed20260812.yaml"
    with pytest.raises(ValueError, match="只开放 seeds"):
        load_center_context_config(
            path,
            overrides=["training.seed=20260811", "model.initialization_seed=20260811"],
        )
    with pytest.raises(ValueError, match="outputs.run_root"):
        load_center_context_config(path, overrides=["outputs.run_root=runs/other"])
    with pytest.raises(ValueError, match="p4_formal"):
        load_center_context_config(path, overrides=["protocol.execution_gate=p3_formal"])
    with pytest.raises(ValueError, match="allow_tf32"):
        load_center_context_config(path, overrides=["training.allow_tf32=true"])
    with pytest.raises(ValueError, match="test windows"):
        load_center_context_config(path, overrides=["data.max_test_windows=1"])


def test_p4_w_configs_reuse_only_frozen_length_specific_caches() -> None:
    path = CONFIG_ROOT / "p4_ccv1_wr_180_seed20260813.yaml"
    with pytest.raises(ValueError, match="cache manifest"):
        load_center_context_config(
            path,
            overrides=["data.center_w_cache_manifest_sha256=deadbeef"],
        )
