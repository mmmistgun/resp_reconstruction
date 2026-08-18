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
from resp_train.crd.tf_v1_model import CRDTfV1Model, CwtBranch, trainable_parameter_count
from resp_train.crd.tf_w_v2 import (
    CANDIDATE_LOCK,
    CANDIDATE_LOCK_SHA256,
    FULL_6V_INDICES,
    FULL_6V_MAPPED_FREQUENCY_SHA256,
    P1_VARIANT_W_VIEW,
    P1_VARIANTS,
    TF_W_V2_VARIANTS,
    SOURCE_CACHE_ROOT,
    VIEW_INDEX_SHA256,
    apply_p1_w_view,
    p1_variant_contract,
    p1_w_scale_count,
    verify_p1_source_identity,
)
from resp_train.crd.tf_w_v2_p1 import _validate_p1_stress_receipt


FORMAL_CONFIGS = {
    variant: Path(f"configs/crd_tf_w_v2/{variant}_formal.yaml") for variant in P1_VARIANTS
}
STRESS_CONFIGS = {
    variant: Path(f"configs/crd_tf_w_v2/{variant}_stress.yaml") for variant in P1_VARIANTS
}


def test_p1_allowlist_and_candidate_lock_config_paths_are_exact() -> None:
    assert tuple(CRD_TF_W_V2_VARIANTS) == TF_W_V2_VARIANTS
    assert P1_VARIANT_W_VIEW == {
        "crd_tfw_v2_w1_resp_12v_film_d6": "resp",
        "crd_tfw_v2_w2_carrier_12v_film_d6": "carrier",
        "crd_tfw_v2_w3_full_6v_film_d6": "full_6v",
    }
    assert hashlib.sha256(CANDIDATE_LOCK.read_bytes()).hexdigest() == CANDIDATE_LOCK_SHA256
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    planned = {row["id"]: row for row in lock["planned_variant_allowlist"] if row["stage"] == "p1"}

    assert tuple(lock["stage_allowlist"]["p1"]) == P1_VARIANTS
    assert set(planned) == set(P1_VARIANTS)
    for variant in P1_VARIANTS:
        assert Path(planned[variant]["formal_config"]) == FORMAL_CONFIGS[variant]
        assert Path(planned[variant]["stress_config"]) == STRESS_CONFIGS[variant]
        assert FORMAL_CONFIGS[variant].is_file()
        assert STRESS_CONFIGS[variant].is_file()


def test_p1_masks_and_6v_view_are_exact_without_source_mutation() -> None:
    source = torch.arange(2 * 97 * 360, dtype=torch.float32).reshape(2, 97, 360)
    frozen = source.clone()
    w1 = apply_p1_w_view(source, P1_VARIANTS[0])
    w2 = apply_p1_w_view(source, P1_VARIANTS[1])
    w3 = apply_p1_w_view(source, P1_VARIANTS[2])

    assert w1.shape == w2.shape == (2, 97, 360)
    torch.testing.assert_close(w1[:, :56], source[:, :56], rtol=0.0, atol=0.0)
    assert torch.count_nonzero(w1[:, 56:]) == 0
    assert torch.count_nonzero(w2[:, :56]) == 0
    torch.testing.assert_close(w2[:, 56:], source[:, 56:], rtol=0.0, atol=0.0)
    assert w3.shape == (2, 49, 360)
    torch.testing.assert_close(w3, source[:, ::2], rtol=0.0, atol=0.0)
    assert w3.data_ptr() == source.data_ptr()
    torch.testing.assert_close(source, frozen, rtol=0.0, atol=0.0)


def test_p1_view_rejects_unknown_variant_and_wrong_source_shape() -> None:
    with pytest.raises(ValueError, match="未知"):
        apply_p1_w_view(torch.zeros(1, 97, 360), "crd_tfw_v2_unknown")
    with pytest.raises(ValueError, match="97,360"):
        apply_p1_w_view(torch.zeros(1, 49, 360), P1_VARIANTS[0])


def test_p1_source_cache_frequency_and_index_identity() -> None:
    verify_p1_source_identity.cache_clear()
    receipt = verify_p1_source_identity(SOURCE_CACHE_ROOT)

    assert receipt["candidate_lock_sha256"] == CANDIDATE_LOCK_SHA256
    assert receipt["view_index_sha256"] == VIEW_INDEX_SHA256
    assert receipt["full_6v_mapped_frequency_sha256"] == FULL_6V_MAPPED_FREQUENCY_SHA256
    assert receipt["full_6v_mapped_frequency_hz"] == [0.03662109375, 7.99560546875]
    assert receipt["full_6v_nominal_grid_max_abs_delta_hz"] <= 2e-15
    assert receipt["source_train_shape"] == [10141, 97, 360]
    assert receipt["source_val_shape"] == [2675, 97, 360]
    assert receipt["source_train_w_sha256"] == "9ed2657cb92ba8e73ce8199dd8b19834ac3b964df10ef615ebb771e9a57e09a3"
    assert receipt["source_val_w_sha256"] == "fb8b44f174f078f93ae6f58083f2ba8c089a011457329e361a9965e409df3cc2"
    assert receipt["source_cache_modified"] is False
    assert receipt["target_read"] is False
    assert receipt["research_test_used"] is False
    assert hashlib.sha256(FULL_6V_INDICES.tobytes()).hexdigest() == VIEW_INDEX_SHA256["full_6v"]

    w3 = p1_variant_contract(P1_VARIANTS[2])
    assert w3["model_input_shape_per_sample"] == [49, 360]
    assert w3["source_tensor_elements_per_sample"] == 97 * 360
    assert w3["model_input_tensor_elements_per_sample"] == 49 * 360
    assert w3["active_scale_count"] == 49
    assert w3["materialization_strategy"] == "strided input-only slice [:,::2,:]"


def test_p1_models_preserve_w0_parameter_and_initialization_identity() -> None:
    seed = 20260811
    anchor = CRDTfV1Model("crd_tf102_w", seed)
    anchor_state = anchor.state_dict()
    assert trainable_parameter_count(anchor) == 1_219_850

    for variant in P1_VARIANTS:
        candidate = CRDTfV1Model(variant, seed)
        assert trainable_parameter_count(candidate) == 1_219_850
        assert p1_w_scale_count(variant) == (49 if variant == P1_VARIANTS[2] else 97)
        assert candidate.branches["w"].scale_count == p1_w_scale_count(variant)
        assert len(candidate.base.local_blocks) == 6
        state = candidate.state_dict()
        assert state.keys() == anchor_state.keys()
        for name in state:
            torch.testing.assert_close(state[name], anchor_state[name], rtol=0.0, atol=0.0)


def test_w3_cwt_branch_accepts_only_the_locked_49_scale_input() -> None:
    branch = CwtBranch(scale_count=49).eval()
    gamma, beta = branch({"w": torch.randn(1, 49, 360)})

    assert trainable_parameter_count(branch) == 150_048
    assert gamma.shape == beta.shape == (1, 96, 1800)
    torch.testing.assert_close(gamma, torch.zeros_like(gamma), rtol=0.0, atol=0.0)
    torch.testing.assert_close(beta, torch.zeros_like(beta), rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="49,360"):
        branch({"w": torch.randn(1, 97, 360)})


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


@pytest.mark.parametrize("variant", P1_VARIANTS)
def test_p1_model_applies_view_immediately_before_w_branch(variant: str) -> None:
    model = CRDTfV1Model(variant, 20260811)
    capture = _CaptureBranch()
    model.base = _FastBase()
    model.branches["w"] = capture
    signal = torch.zeros(1, 1, 18000)
    source = torch.arange(97 * 360, dtype=torch.float32).reshape(1, 97, 360)
    frozen = source.clone()

    model(signal, tf={"w": source})

    assert capture.observed is not None
    torch.testing.assert_close(capture.observed, apply_p1_w_view(source, variant), rtol=0.0, atol=0.0)
    torch.testing.assert_close(source, frozen, rtol=0.0, atol=0.0)


def test_p1_builder_and_same_seed_base_match_c201() -> None:
    seed = 20260811
    cfg = load_crd_config(FORMAL_CONFIGS[P1_VARIANTS[0]])
    candidate = build_crd_model(cfg)
    c201 = CRDCoarseModel("crd_c201_decoder_10hz_cap", seed)

    assert isinstance(candidate, CRDTfV1Model)
    candidate_base = candidate.base.state_dict()
    c201_state = c201.state_dict()
    assert candidate_base.keys() == c201_state.keys()
    for name in candidate_base:
        torch.testing.assert_close(candidate_base[name], c201_state[name], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("variant", P1_VARIANTS)
def test_p1_stress_and_formal_configs_are_frozen(variant: str) -> None:
    stress = load_crd_config(STRESS_CONFIGS[variant])
    formal = load_crd_config(FORMAL_CONFIGS[variant])

    assert str(stress.protocol.name) == str(formal.protocol.name) == CRD_TF_W_V2_PROTOCOL_VERSION
    assert str(stress.protocol.stage) == str(formal.protocol.stage) == "tf_w_v2"
    assert str(stress.protocol.run_role) == "stress"
    assert str(stress.protocol.execution_gate) == "p1_cuda_stress"
    assert (stress.training.epochs, stress.training.batch_size, stress.training.gradient_accumulation_steps) == (
        5,
        128,
        1,
    )
    assert 5 * int(np.ceil(10141 / 128)) == 400
    assert int(stress.training.seed) == 20260811
    assert str(stress.outputs.run_root).endswith(f"runs/crd_tf_w_v2/engineering/{variant}")
    assert str(formal.protocol.run_role) == "formal"
    assert str(formal.protocol.execution_gate) == "p1_formal"
    assert (formal.training.epochs, formal.training.batch_size, formal.training.gradient_accumulation_steps) == (
        80,
        128,
        1,
    )
    assert str(formal.outputs.run_root).endswith(f"runs/crd_tf_w_v2/formal/{variant}/seed_20260811")
    for cfg in (stress, formal):
        assert str(cfg.model.variant) == variant
        assert list(cfg.model.tf_representations) == ["w"]
        assert str(cfg.data.tf_cache_path) == str(SOURCE_CACHE_ROOT)
        assert cfg.training.early_stopping_enabled is False
        assert cfg.data.max_train_windows is None
        assert cfg.data.max_val_windows is None
        assert cfg.data.max_test_windows is None


def test_p1_formal_allows_only_fixed_seeds_and_gate_role_pairs() -> None:
    path = FORMAL_CONFIGS[P1_VARIANTS[0]]
    for seed in (20260811, 20260812, 20260813):
        cfg = load_crd_config(path, overrides=[f"training.seed={seed}"])
        assert int(cfg.model.initialization_seed) == seed
        assert f"seed_{seed}" in str(cfg.outputs.run_root)

    with pytest.raises(ValueError, match="formal seed"):
        load_crd_config(path, overrides=["training.seed=20260814"])
    bad = deepcopy(load_crd_config(path))
    bad.protocol.execution_gate = "p1_cuda_stress"
    with pytest.raises(ValueError, match="p1_cuda_stress"):
        _validate_crd_config(bad)


def test_p1_formal_preflight_receipt_contract_blocks_incomplete_stress() -> None:
    variant = P1_VARIANTS[0]
    receipt = {
        "protocol": CRD_TF_W_V2_PROTOCOL_VERSION,
        "phase": "p1_isolation_stress",
        "variant": variant,
        "status": "passed",
        "complete": True,
        "run_role": "stress",
        "git_dirty": False,
        "candidate_lock_sha256": CANDIDATE_LOCK_SHA256,
        "source_cache_manifest_sha256": "6fb44aad2689d9426ad78dc1f054db5aaac698792af5818bc01a54563cb9f0b8",
        "physical_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "use_amp": True,
        "amp_dtype": "bfloat16",
        "benchmark_batch_size": 1,
        "all_benchmark_input_gradients_finite": True,
        "all_benchmark_parameter_gradients_finite": True,
        "optimizer_updates": 400,
        "full_validation_count": 5,
        "all_history_finite": True,
        "all_checkpoint_finite": True,
        "all_primary_finite": True,
        "prediction_degenerate_fraction": 0.0,
        "validation_peak_prediction_degenerate_fraction": 0.0,
        "oom": False,
        "peak_reserved_fraction": 0.84,
        "warm_train_samples_per_second": 100.0,
        "forward_latency_ms": 10.0,
        "forward_backward_latency_ms": 30.0,
        "validation_peak_allocated_mib": 1000.0,
        "validation_peak_reserved_mib": 1200.0,
    }
    _validate_p1_stress_receipt(receipt, variant)

    bad = dict(receipt, optimizer_updates=399)
    with pytest.raises(RuntimeError, match="恰为 400"):
        _validate_p1_stress_receipt(bad, variant)
    bad = dict(receipt, peak_reserved_fraction=0.91)
    with pytest.raises(RuntimeError, match="超过 90%"):
        _validate_p1_stress_receipt(bad, variant)
