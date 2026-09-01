from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import resp_train.paper_evidence.center_context_acceptance as acceptance
from resp_train.paper_evidence.center_context_acceptance import (
    P2_ACCEPTANCE_ARMS,
    P2_BATCH_ACCEPTANCE_ARM,
    _is_cuda_oom,
    exercise_synthetic_arm,
    make_synthetic_center_batch,
    run_p2_gpu_acceptance,
)
from resp_train.paper_evidence.center_context_config import load_center_context_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/paper_evidence_v1/center_context_v1.yaml"


class _IdentityMamba(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


@pytest.mark.parametrize("samples,context", [(6000, 120), (9000, 180), (18000, 360)])
def test_p2_synthetic_batch_is_deterministic_finite_and_length_specific(
    samples: int,
    context: int,
) -> None:
    left = make_synthetic_center_batch(2, samples, device="cpu")
    right = make_synthetic_center_batch(2, samples, device="cpu")
    assert left[0].shape == (2, 1, samples)
    assert left[1].shape == (2, 1, 6000)
    assert left[2]["w"].shape == (2, 49, context)
    for lhs, rhs in zip((left[0], left[1], left[2]["w"]), (right[0], right[1], right[2]["w"])):
        torch.testing.assert_close(lhs, rhs, rtol=0.0, atol=0.0)
        assert torch.isfinite(lhs).all()


def test_p2_plan_has_six_batch1_arms_and_one_maximum_batch_arm() -> None:
    assert P2_ACCEPTANCE_ARMS == (
        ("c201_center60", 6000),
        ("c201_center60", 9000),
        ("c201_center60", 18000),
        ("w_reduced_center60", 6000),
        ("w_reduced_center60", 9000),
        ("w_reduced_center60", 18000),
    )
    assert P2_BATCH_ACCEPTANCE_ARM == ("w_reduced_center60", 18000)


def test_p2_cpu_unit_exercises_model_loss_backward_and_optimizer_step() -> None:
    cfg = load_center_context_config(CONFIG)
    result = exercise_synthetic_arm(
        cfg,
        variant="w_reduced_center60",
        input_samples=6000,
        batch_size=1,
        device="cpu",
        optimizer_step=True,
        mamba_factory=_IdentityMamba,
    )
    assert result["waveform_shape"] == [1, 1, 6000]
    assert result["waveform_10hz_shape"] == [1, 1, 600]
    assert result["optimizer_step"] is True
    assert result["finite"] is True
    assert result["peak_allocated_mib"] is None


def test_p2_oom_classifier_is_cuda_specific() -> None:
    assert _is_cuda_oom(RuntimeError("CUDA out of memory")) is True
    assert _is_cuda_oom(RuntimeError("CPU out of memory")) is False
    assert _is_cuda_oom(RuntimeError("unrelated CUDA failure")) is False


def _mock_gpu_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acceptance, "check_crd_dependencies", lambda: [])
    monkeypatch.setattr(acceptance, "_assert_clean_git", lambda: "a" * 40)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        acceptance,
        "_cuda_identity",
        lambda _: {"logical_device": "cuda:0", "name": "fixture"},
    )
    monkeypatch.setattr(acceptance, "_release_cuda", lambda: None)


def test_p2_output_is_hashed_access_closed_and_nonoverwriting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_gpu_preflight(monkeypatch)

    def exercise(_cfg, **kwargs):
        return {
            "variant": kwargs["variant"],
            "input_samples": kwargs["input_samples"],
            "batch_size": kwargs["batch_size"],
            "optimizer_step": kwargs["optimizer_step"],
            "finite": True,
        }

    monkeypatch.setattr(acceptance, "exercise_synthetic_arm", exercise)
    receipt_path = run_p2_gpu_acceptance(
        config_path=CONFIG,
        output_root=tmp_path,
        device="cuda:0",
        command=["fixture"],
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["decision"] == "batch128_accepted"
    assert receipt["batch1_completed_arms"] == 6
    assert receipt["batch128_completed"] is True
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
    assert (receipt_path.parent / "artifact_manifest.json").is_file()
    assert json.loads(
        (receipt_path.parent / "lifecycle.json").read_text(encoding="utf-8")
    )["status"] == "complete"
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        run_p2_gpu_acceptance(
            config_path=CONFIG,
            output_root=tmp_path,
            device="cuda:0",
            command=["fixture"],
        )


def test_p2_unexpected_failure_preserves_failed_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_gpu_preflight(monkeypatch)
    monkeypatch.setattr(
        acceptance,
        "exercise_synthetic_arm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )
    with pytest.raises(RuntimeError, match="fixture failure"):
        run_p2_gpu_acceptance(
            config_path=CONFIG,
            output_root=tmp_path,
            device="cuda:0",
            command=["fixture"],
        )
    output = tmp_path / ("a" * 12)
    lifecycle = json.loads((output / "lifecycle.json").read_text(encoding="utf-8"))
    assert lifecycle["status"] == "failed"
    assert lifecycle["error_type"] == "RuntimeError"
    assert (output / "artifact_manifest.json").is_file()


def test_p2_batch128_oom_is_a_complete_fallback_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_gpu_preflight(monkeypatch)
    calls = 0

    def exercise(_cfg, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("CUDA out of memory")
        return {
            "variant": kwargs["variant"],
            "input_samples": kwargs["input_samples"],
            "batch_size": kwargs["batch_size"],
            "optimizer_step": kwargs["optimizer_step"],
            "finite": True,
        }

    monkeypatch.setattr(acceptance, "exercise_synthetic_arm", exercise)
    receipt_path = run_p2_gpu_acceptance(
        config_path=CONFIG,
        output_root=tmp_path,
        device="cuda:0",
        command=["fixture"],
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["decision"] == "batch128_fallback_required"
    assert receipt["batch1_completed_arms"] == 6
    assert receipt["batch128_completed"] is False
    assert receipt["access"]["optimizer_step_used"] is False
    assert receipt["failure"]["stage"] == "w_reduced_180_batch128"
