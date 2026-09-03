from __future__ import annotations

from pathlib import Path

import pytest

from resp_train.paper_evidence.center30_config import (
    CENTER30_P4S3_OUTPUT_ROOT,
    CENTER30_W_CACHE_MANIFEST_SHA256,
    CENTER30_W_CACHE_PATHS,
    load_center30_config,
)
from resp_train.paper_evidence.center30_experiment import center30_experiment_id


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/paper_evidence_v1"
VARIANTS = {
    "c201": "c201_center30",
    "wr": "w_reduced_center30",
}


def _expected_configs() -> dict[str, tuple[str, int, int]]:
    return {
        f"p4s3_c30v1_{short}_{input_sec}_seed{seed}.yaml": (variant, input_sec * 100, seed)
        for short, variant in VARIANTS.items()
        for input_sec in (30, 45, 60, 90)
        for seed in (20260812, 20260813)
    }


def test_p4s3_sixteen_configs_freeze_full_matrix_and_additional_seeds() -> None:
    expected = _expected_configs()
    paths = sorted(CONFIG_ROOT.glob("p4s3_c30v1_*.yaml"))
    assert {path.name for path in paths} == set(expected)
    identities = set()
    for path in paths:
        variant, input_samples, seed = expected[path.name]
        cfg = load_center30_config(path)
        assert cfg.protocol.stage == "p4s_additional_seeds"
        assert cfg.protocol.run_role == "formal"
        assert cfg.protocol.execution_gate == "p4s_three_seed_formal"
        assert cfg.model.variant == variant
        assert cfg.window.input_samples == input_samples
        assert cfg.window.input_sec == input_samples // 100
        assert cfg.window.output_samples == 3000
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
        assert cfg.data.max_train_windows is cfg.data.max_val_windows is cfg.data.max_test_windows is None
        assert cfg.outputs.run_root == CENTER30_P4S3_OUTPUT_ROOT
        identities.add((center30_experiment_id(variant, input_samples), seed))
    assert len(identities) == 16


def test_p4s3_w_configs_reuse_frozen_length_specific_caches() -> None:
    for input_sec in (30, 45, 60, 90):
        input_samples = input_sec * 100
        path = CONFIG_ROOT / f"p4s3_c30v1_wr_{input_sec}_seed20260812.yaml"
        cfg = load_center30_config(path)
        assert Path(str(cfg.data.center_w_cache_path)).resolve() == Path(
            CENTER30_W_CACHE_PATHS[input_samples]
        ).resolve()
        assert cfg.data.center_w_cache_manifest_sha256 == CENTER30_W_CACHE_MANIFEST_SHA256[input_samples]


def test_p4s3_gate_rejects_seed_stage_output_cache_runtime_and_test_drift() -> None:
    c201 = CONFIG_ROOT / "p4s3_c30v1_c201_30_seed20260812.yaml"
    for seed in (20260811, 20260814):
        with pytest.raises(ValueError, match="只开放 seeds"):
            load_center30_config(
                c201,
                overrides=[f"training.seed={seed}", f"model.initialization_seed={seed}"],
            )
    with pytest.raises(ValueError, match="输出"):
        load_center30_config(c201, overrides=["outputs.run_root=runs/other"])
    with pytest.raises(ValueError, match="p4s_three_seed_formal"):
        load_center30_config(c201, overrides=["protocol.execution_gate=p4s_formal"])
    with pytest.raises(ValueError, match="allow_tf32"):
        load_center30_config(c201, overrides=["training.allow_tf32=true"])
    with pytest.raises(ValueError, match="test windows"):
        load_center30_config(c201, overrides=["data.max_test_windows=1"])
    wr = CONFIG_ROOT / "p4s3_c30v1_wr_30_seed20260812.yaml"
    with pytest.raises(ValueError, match="manifest SHA-256"):
        load_center30_config(wr, overrides=["data.center_w_cache_manifest_sha256=deadbeef"])


def test_p4s2_and_p4s3_gates_are_disjoint() -> None:
    p4s2 = CONFIG_ROOT / "p4s_c30v1_c201_30.yaml"
    with pytest.raises(ValueError, match="p4s_formal"):
        load_center30_config(p4s2, overrides=["protocol.execution_gate=p4s_three_seed_formal"])
    p4s3 = CONFIG_ROOT / "p4s3_c30v1_c201_30_seed20260813.yaml"
    with pytest.raises(ValueError, match="p4s_three_seed_formal"):
        load_center30_config(p4s3, overrides=["protocol.execution_gate=p4s_formal"])
