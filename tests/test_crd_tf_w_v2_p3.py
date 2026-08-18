from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from resp_train.crd.config import CRD_TF_W_V2_PROTOCOL_VERSION, _validate_crd_config, load_crd_config
from resp_train.crd.model import CRDCoarseModel, CRD_TF_W_V2_VARIANTS, build_crd_model
from resp_train.crd.tf_v1_model import CRDTfV1Model, trainable_parameter_count
from resp_train.crd.tf_w_v2 import (
    CANDIDATE_LOCK,
    CANDIDATE_LOCK_SHA256,
    FULL_12V_INDICES_SHA256,
    P1_VARIANTS,
    P3_VARIANT,
    P3_VARIANTS,
    SOURCE_CACHE_ROOT,
    TF_W_V2_VARIANTS,
    p3_variant_contract,
)
from resp_train.crd.tf_w_v2_p3 import (
    P3_LOCAL_BLOCK_COUNT,
    P3_TRAINABLE_PARAMETERS,
    _validate_p3_stress_receipt,
    validate_p3_formal_preflight,
)


STRESS_CONFIG = Path("configs/crd_tf_w_v2/crd_tfw_v2_d4_full_12v_film_stress.yaml")
FORMAL_CONFIG = Path("configs/crd_tf_w_v2/crd_tfw_v2_d4_full_12v_film_formal.yaml")


def _valid_receipt(*, commit: str = "0123456789abcdef0123456789abcdef01234567") -> dict[str, object]:
    return {
        "protocol": CRD_TF_W_V2_PROTOCOL_VERSION,
        "phase": "p3_depth_stress",
        "variant": P3_VARIANT,
        "status": "passed",
        "complete": True,
        "run_role": "stress",
        "git_commit": commit,
        "git_dirty": False,
        "candidate_lock_sha256": CANDIDATE_LOCK_SHA256,
        "source_cache_manifest_sha256": "6fb44aad2689d9426ad78dc1f054db5aaac698792af5818bc01a54563cb9f0b8",
        "physical_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "use_amp": True,
        "amp_dtype": "bfloat16",
        "trainable_parameters": P3_TRAINABLE_PARAMETERS,
        "local_bimamba2_blocks": P3_LOCAL_BLOCK_COUNT,
        "benchmark_batch_size": 1,
        "all_benchmark_input_gradients_finite": True,
        "all_benchmark_parameter_gradients_finite": True,
        "optimizer_updates": 400,
        "full_validation_count": 5,
        "all_history_finite": True,
        "all_checkpoint_finite": True,
        "all_optimizer_finite": True,
        "all_primary_finite": True,
        "prediction_degenerate_fraction": 0.0,
        "validation_peak_prediction_degenerate_fraction": 0.0,
        "oom": False,
        "training_peak_allocated_mib": 7000.0,
        "training_peak_reserved_mib": 9000.0,
        "validation_peak_allocated_mib": 2000.0,
        "validation_peak_reserved_mib": 9100.0,
        "peak_allocated_mib": 7000.0,
        "peak_reserved_mib": 9100.0,
        "total_device_memory_mib": 16000.0,
        "peak_reserved_fraction": 9100.0 / 16000.0,
        "memory_status": "passed",
        "warm_train_samples_per_second": 200.0,
        "forward_latency_ms": 15.0,
        "forward_backward_latency_ms": 80.0,
    }


def test_p3_allowlist_and_candidate_lock_paths_are_exact() -> None:
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    planned = [row for row in lock["planned_variant_allowlist"] if row["stage"] == "p3"]

    assert P3_VARIANTS == ("crd_tfw_v2_d4_full_12v_film",)
    assert tuple(CRD_TF_W_V2_VARIANTS) == TF_W_V2_VARIANTS == (*P1_VARIANTS, *P3_VARIANTS)
    assert tuple(lock["stage_allowlist"]["p3"]) == P3_VARIANTS
    assert len(planned) == 1 and planned[0]["id"] == P3_VARIANT
    assert Path(planned[0]["stress_config"]) == STRESS_CONFIG
    assert Path(planned[0]["formal_config"]) == FORMAL_CONFIG
    assert STRESS_CONFIG.is_file() and FORMAL_CONFIG.is_file()


def test_p3_contract_is_full_12v_d4_without_composite_view() -> None:
    contract = p3_variant_contract(P3_VARIANT)

    assert contract["w_view"] == "full_12v"
    assert contract["source_shape_per_sample"] == contract["model_input_shape_per_sample"] == [97, 360]
    assert contract["source_tensor_elements_per_sample"] == contract["model_input_tensor_elements_per_sample"]
    assert contract["active_scale_count"] == 97
    assert contract["view_index_sha256"] == FULL_12V_INDICES_SHA256
    assert hashlib.sha256(np.arange(97, dtype="<i8").tobytes()).hexdigest() == FULL_12V_INDICES_SHA256
    assert contract["local_bimamba2_blocks"] == 4
    assert contract["trainable_parameters"] == 902_722
    assert contract["source_cache_modified"] is False
    assert contract["target_read"] is False
    assert contract["research_test_used"] is False


@pytest.mark.parametrize("seed", (20260811, 20260812, 20260813))
def test_p3_model_removes_only_last_two_local_blocks_with_shared_initialization(seed: int) -> None:
    anchor = CRDTfV1Model("crd_tf102_w", seed)
    candidate = CRDTfV1Model(P3_VARIANT, seed)

    assert trainable_parameter_count(anchor) == 1_219_850
    assert trainable_parameter_count(candidate) == P3_TRAINABLE_PARAMETERS
    assert trainable_parameter_count(anchor) - trainable_parameter_count(candidate) == 317_128
    assert len(anchor.base.local_blocks) == 6
    assert len(candidate.base.local_blocks) == P3_LOCAL_BLOCK_COUNT
    assert candidate.branches["w"].scale_count == 97
    assert trainable_parameter_count(anchor.base.local_blocks[4]) == 158_564
    assert trainable_parameter_count(anchor.base.local_blocks[5]) == 158_564

    anchor_state = anchor.state_dict()
    candidate_state = candidate.state_dict()
    removed = set(anchor_state) - set(candidate_state)
    assert removed
    assert all(name.startswith(("base.local_blocks.4.", "base.local_blocks.5.")) for name in removed)
    assert not (set(candidate_state) - set(anchor_state))
    for name, value in candidate_state.items():
        torch.testing.assert_close(value, anchor_state[name], rtol=0.0, atol=0.0)


def test_four_block_override_is_not_a_general_or_d8_interface() -> None:
    with pytest.raises(ValueError, match="只允许协议注册的 4 或 6"):
        CRDCoarseModel("crd_c201_decoder_10hz_cap", 20260811, local_block_count=8)
    with pytest.raises(ValueError, match="只允许 CRD-TF-W v2 D4"):
        CRDCoarseModel("crd_102_b0_local_mamba", 20260811, local_block_count=4)


class _CaptureBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observed: torch.Tensor | None = None

    def forward(self, tf: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        self.observed = tf["w"]
        batch = int(self.observed.shape[0])
        zero = torch.zeros(batch, 96, 1800, dtype=self.observed.dtype)
        return zero, zero


class _FastBase(nn.Module):
    def encode_local(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 96, 1800, dtype=x.dtype)

    def decode_local(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"waveform": latent}


def test_p3_forward_passes_unmodified_full_12v_immediately_to_w_branch() -> None:
    model = CRDTfV1Model(P3_VARIANT, 20260811)
    capture = _CaptureBranch()
    model.base = _FastBase()
    model.branches["w"] = capture
    source = torch.arange(97 * 360, dtype=torch.float32).reshape(1, 97, 360)
    frozen = source.clone()

    model(torch.zeros(1, 1, 18000), tf={"w": source})

    assert capture.observed is source
    assert capture.observed.shape == (1, 97, 360)
    torch.testing.assert_close(source, frozen, rtol=0.0, atol=0.0)


def test_p3_builder_and_configs_are_frozen() -> None:
    stress = load_crd_config(STRESS_CONFIG)
    formal = load_crd_config(FORMAL_CONFIG)
    candidate = build_crd_model(formal)

    assert isinstance(candidate, CRDTfV1Model)
    assert trainable_parameter_count(candidate) == P3_TRAINABLE_PARAMETERS
    assert str(stress.protocol.name) == str(formal.protocol.name) == CRD_TF_W_V2_PROTOCOL_VERSION
    assert str(stress.protocol.execution_gate) == "p3_cuda_stress"
    assert str(formal.protocol.execution_gate) == "p3_formal"
    assert (stress.training.epochs, stress.training.batch_size, stress.training.gradient_accumulation_steps) == (
        5,
        128,
        1,
    )
    assert (formal.training.epochs, formal.training.batch_size, formal.training.gradient_accumulation_steps) == (
        80,
        128,
        1,
    )
    for cfg in (stress, formal):
        assert str(cfg.model.variant) == P3_VARIANT
        assert list(cfg.model.tf_representations) == ["w"]
        assert str(cfg.data.tf_cache_path) == str(SOURCE_CACHE_ROOT)
        assert cfg.training.early_stopping_enabled is False
        assert cfg.data.max_train_windows is cfg.data.max_val_windows is cfg.data.max_test_windows is None


def test_p3_formal_fixed_seeds_and_gate_role_pairs() -> None:
    for seed in (20260811, 20260812, 20260813):
        cfg = load_crd_config(FORMAL_CONFIG, overrides=[f"training.seed={seed}"])
        assert int(cfg.model.initialization_seed) == seed
        assert f"seed_{seed}" in str(cfg.outputs.run_root)

    with pytest.raises(ValueError, match="formal seed"):
        load_crd_config(FORMAL_CONFIG, overrides=["training.seed=20260814"])
    bad = deepcopy(load_crd_config(FORMAL_CONFIG))
    bad.protocol.execution_gate = "p3_cuda_stress"
    with pytest.raises(ValueError, match="p3_cuda_stress"):
        _validate_crd_config(bad)


def test_p3_stress_receipt_contract_and_memory_status() -> None:
    receipt = _valid_receipt()
    _validate_p3_stress_receipt(receipt)

    with pytest.raises(RuntimeError, match="恰为 400"):
        _validate_p3_stress_receipt(dict(receipt, optimizer_updates=399))
    with pytest.raises(RuntimeError, match="超过 90%"):
        _validate_p3_stress_receipt(dict(receipt, peak_reserved_fraction=0.91))
    with pytest.raises(RuntimeError, match="memory_status"):
        _validate_p3_stress_receipt(dict(receipt, peak_reserved_fraction=0.86, memory_status="passed"))


def test_p3_formal_preflight_requires_unique_same_commit_receipt(tmp_path: Path, monkeypatch) -> None:
    import resp_train.crd.tf_w_v2_p3 as p3

    engineering = tmp_path / "engineering"
    run_dir = engineering / P3_VARIANT / "20260818_120000_000000"
    run_dir.mkdir(parents=True)
    (run_dir / p3.P3_STRESS_RECEIPT).write_text(
        json.dumps(_valid_receipt()) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(p3, "P3_ENGINEERING_ROOT", engineering)
    monkeypatch.setattr(p3, "verify_p1_source_identity", lambda: {})
    monkeypatch.setattr(p3, "_git_commit", lambda: "0123456789abcdef0123456789abcdef01234567")

    assert validate_p3_formal_preflight()["status"] == "passed"
    monkeypatch.setattr(p3, "_git_commit", lambda: "different-commit")
    with pytest.raises(RuntimeError, match="不是同一 commit"):
        validate_p3_formal_preflight()
