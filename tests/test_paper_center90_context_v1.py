from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from resp_train.paper_evidence.center90_acceptance import (
    exercise_synthetic_center90_max_arm,
    make_synthetic_center90_batch,
)
from resp_train.paper_evidence.center90_cache import (
    Center90WCacheReader,
    build_center90_w_cache_from_rows,
    center90_cwt_features,
    fixed_center90_w_transform_spec,
    open_center90_w_cache,
)
from resp_train.paper_evidence.center90_config import (
    CENTER90_FORMAL_SEEDS,
    CENTER90_W_CACHE_MANIFEST_SHA256,
    CENTER90_W_CACHE_PATHS,
    load_center90_config,
)
from resp_train.paper_evidence.center90_data import (
    Center90ContextDataset,
    audit_center90_nested_views,
    center90_latent_bounds,
    crop_center90_tensors,
)
from resp_train.paper_evidence.center90_experiment import Center90Experiment, center90_experiment_id
from resp_train.paper_evidence.center90_loss import Center90ContextLoss, pi90_numpy, pi90_torch
from resp_train.paper_evidence.center90_metrics import (
    center90_rr_strictly_improved,
    evaluate_center90_predictions,
    summarize_center90_metrics,
    validation_center90_rr_mean,
)
from resp_train.paper_evidence.center90_model import (
    Center90ContextModel,
    center90_shared_state_identity,
)
from resp_train.paper_evidence.center_context_cache import fixed_center_w_transform_spec


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/paper_evidence_v1/center90_context_v1.yaml"


class _IdentityMamba(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _Parent(Dataset):
    def __init__(self, split: str = "train") -> None:
        self.rows = pd.DataFrame({"dataset_row_id": [7], "split": [split]})
        self.item = {
            "x": torch.arange(18000, dtype=torch.float32).view(1, -1),
            "target": (torch.arange(18000, dtype=torch.float32) * 2).view(1, -1),
            "meta": {"dataset_row_id": 7, "split": split, "samp_id": 1, "coupling_state_id": 2},
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int):
        return self.item


def _signal(*, frequency_hz: float = 0.2, modulation: float = 0.25) -> np.ndarray:
    time = np.arange(9000, dtype=np.float64) / 100.0
    envelope = 1.0 + modulation * np.sin(2.0 * np.pi * time / 25.0)
    return envelope * np.sin(2.0 * np.pi * frequency_hz * time)


def _predictions(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "r_tho_hat": pred[None, None].astype(np.float32),
        "tho_ref": target[None, None].astype(np.float32),
        "dataset_row_id": np.asarray([1]),
        "split": np.asarray(["val"]),
        "input_set": np.asarray(["research_v2_waveform"]),
        "samp_id": np.asarray([2]),
        "coupling_state_id": np.asarray([3]),
    }


def test_center90_config_freezes_task_and_full_three_seed_matrix() -> None:
    cfg = load_center90_config(CONFIG)
    assert cfg.protocol.name == "paper-center90-context-v1-20260904"
    assert cfg.window.output_samples == 9000
    assert cfg.data.access_splits == ["train", "val"]
    identities = set()
    for variant in ("c201_center90", "w_reduced_center90"):
        for input_sec in (90, 135, 180):
            for seed in CENTER90_FORMAL_SEEDS:
                identities.add((variant, input_sec, seed))
    assert len(identities) == 18
    for variant, input_sec, seed in identities:
        assert center90_experiment_id(variant, input_sec * 100).startswith("C90V1_")
    with pytest.raises(ValueError, match="test windows"):
        load_center90_config(CONFIG, overrides=["data.max_test_windows=1"])
    formal = load_center90_config(CONFIG.parent / "c90v1_wr_135_seed20260811.yaml")
    assert formal.model.variant == "w_reduced_center90"
    assert formal.window.input_sec == 135


def test_three_inputs_share_exact_center90_target_and_latent_bounds() -> None:
    parent = _Parent().item
    expected = {9000: (0, 900), 13500: (225, 1125), 18000: (450, 1350)}
    targets = []
    for length, bounds in expected.items():
        x, target = crop_center90_tensors(parent["x"], parent["target"], input_samples=length)
        assert x.shape == (1, length)
        assert target.shape == (1, 9000)
        assert center90_latent_bounds(length) == bounds
        targets.append(target)
    assert all(torch.equal(targets[0], value) for value in targets[1:])
    assert audit_center90_nested_views([parent])["center90_targets_pointwise_identical"] is True
    with pytest.raises(ValueError, match="train/val"):
        Center90ContextDataset(_Parent("test"), input_samples=9000)


@pytest.mark.parametrize("length,context", [(9000, 180), (13500, 270), (18000, 360)])
def test_center90_models_emit_fixed_90s_output(length: int, context: int) -> None:
    c201 = Center90ContextModel("c201_center90", 20260811, mamba_factory=_IdentityMamba).eval()
    w_reduced = Center90ContextModel(
        "w_reduced_center90", 20260811, mamba_factory=_IdentityMamba
    ).eval()
    x = torch.randn(1, 1, length)
    with torch.no_grad():
        c201_output = c201(x)
        w_output = w_reduced(x, tf={"w": torch.randn(1, 49, context)})
    assert c201_output["waveform"].shape == w_output["waveform"].shape == (1, 1, 9000)
    assert c201_output["waveform_10hz"].shape == w_output["waveform_10hz"].shape == (1, 1, 900)
    assert c201.trainable_parameter_count() > 0
    assert w_reduced.trainable_parameter_count() - c201.trainable_parameter_count() == 150048


def test_center90_shared_initialization_and_tf_contract() -> None:
    c201 = Center90ContextModel("c201_center90", 1, mamba_factory=_IdentityMamba)
    w_reduced = Center90ContextModel("w_reduced_center90", 1, mamba_factory=_IdentityMamba)
    assert center90_shared_state_identity(c201, w_reduced)["shared_tensors_identical"] is True
    x = torch.zeros(1, 1, 9000)
    with pytest.raises(ValueError, match="TF feature"):
        c201(x, tf={"w": torch.zeros(1, 49, 180)})
    with pytest.raises(ValueError, match="需要"):
        w_reduced(x)


def test_center90_real_mamba_parameter_contract() -> None:
    c201 = Center90ContextModel("c201_center90", 1)
    w_reduced = Center90ContextModel("w_reduced_center90", 1)
    assert c201.trainable_parameter_count() == 1069802
    assert w_reduced.trainable_parameter_count() == 1219850


def test_pi90_loss_metrics_selector_and_17_point_envelope() -> None:
    cfg = load_center90_config(CONFIG)
    signal = _signal().astype(np.float32)
    tensor = torch.tensor(signal).view(1, -1).requires_grad_()
    np.testing.assert_allclose(pi90_torch(tensor).detach(), pi90_numpy(signal[None]), rtol=0.0, atol=2e-3)
    loss_fn = Center90ContextLoss(cfg)
    target = tensor.detach().view(1, 1, -1)
    identity_loss, _ = loss_fn(target.clone(), target)
    assert float(identity_loss) < 1e-5
    metrics = evaluate_center90_predictions(_predictions(signal, signal), cfg)
    wrong = evaluate_center90_predictions(_predictions(_signal(frequency_hz=0.4), signal), cfg)
    assert metrics.loc[0, "center90_rr_mae_bpm"] < wrong.loc[0, "center90_rr_mae_bpm"]
    assert summarize_center90_metrics(metrics).loc[0, "n_samples"] == 1
    assert validation_center90_rr_mean(_predictions(signal, signal), cfg) == pytest.approx(
        metrics.loc[0, "center90_rr_mae_bpm"]
    )
    assert center90_rr_strictly_improved(0.9, 1.0)
    assert metrics.loc[0, "center90_envelope_trajectory_mae"] == pytest.approx(0.0, abs=1e-12)


def test_actual_center90_135s_cwt_has_49_scales_and_270_context_points() -> None:
    time = np.arange(13500, dtype=np.float32) / 100.0
    feature, frequencies = center90_cwt_features(np.sin(2.0 * np.pi * 0.2 * time))
    assert feature.shape == (49, 270)
    assert frequencies.shape == (49,)
    assert np.isfinite(feature).all() and np.isfinite(frequencies).all()


def _cache_rows(split: str, row_id: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_row_id": [row_id],
            "split": [split],
            "source_npz": [f"{split}.npz"],
            "bcg_signal_key": ["bcg"],
            "window_start_sample": [0],
            "window_end_sample": [18000],
        }
    )


def test_center90_cache_is_input_only_hashed_and_reuses_identical_specs(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeWholeNightCache:
        def __init__(self, _: Path) -> None:
            pass

        def get_arrays(self, _: str, keys: list[str]):
            return {keys[0]: np.zeros(18000, dtype=np.float32)}

    monkeypatch.setattr("resp_train.data.cache.WholeNightCache", FakeWholeNightCache)

    from resp_train.paper_evidence.center_context_cache import reduced_target_frequencies

    def extract(view: np.ndarray):
        assert view.shape == (13500,)
        return np.zeros((49, 270), dtype=np.float32), reduced_target_frequencies()

    output = tmp_path / "cache135"
    manifest_path = build_center90_w_cache_from_rows(
        output_dir=output,
        index_path=tmp_path / "index.csv",
        rows_by_split={"train": _cache_rows("train", 1), "val": _cache_rows("val", 2)},
        input_samples=13500,
        dataset_index_sha256="fixture",
        feature_extractor=extract,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_only"] is True
    assert manifest["target_read"] is manifest["test_read"] is False
    assert Center90WCacheReader(output, split="val", input_samples=13500).get(2).shape == (49, 270)
    assert fixed_center90_w_transform_spec(9000) == fixed_center_w_transform_spec(9000)
    assert fixed_center90_w_transform_spec(18000) == fixed_center_w_transform_spec(18000)


@pytest.mark.parametrize("samples", [9000, 18000])
def test_center90_opens_frozen_shared_input_cache_by_hash(samples: int) -> None:
    reader = open_center90_w_cache(
        CENTER90_W_CACHE_PATHS[samples],
        split="val",
        input_samples=samples,
        expected_manifest_sha256=CENTER90_W_CACHE_MANIFEST_SHA256[samples],
    )
    assert reader.features.shape == (2675, 49, samples // 50)


def test_center90_synthetic_max_arm_and_cli_confirmation_gates() -> None:
    cfg = load_center90_config(CONFIG)
    sensor, target, w = make_synthetic_center90_batch(2, 18000, device="cpu")
    assert sensor.shape == (2, 1, 18000)
    assert target.shape == (2, 1, 9000)
    assert w["w"].shape == (2, 49, 360)
    result = exercise_synthetic_center90_max_arm(
        cfg,
        batch_size=1,
        device="cpu",
        optimizer_step=False,
        mamba_factory=_IdentityMamba,
    )
    assert result["waveform_shape"] == [1, 1, 9000]
    assert result["finite"] is True
    with pytest.raises(RuntimeError, match="不创建真实 lifecycle"):
        Center90Experiment(cfg).train()
    assert center90_experiment_id("c201_center90", 13500) == "C90V1_C201_135"
    for script, marker in (
        ("train_paper_center90_context_v1.py", "formal config"),
        ("accept_paper_center90_context_v1.py", "confirm-gpu-acceptance"),
        ("build_paper_center90_context_w_cache_v1.py", "confirm-cache-build"),
    ):
        command = [sys.executable, str(ROOT / "scripts" / script)]
        if script.startswith("train") or script.startswith("accept"):
            command.extend(["--config", str(CONFIG)])
        else:
            command.extend(["--config", str(CONFIG), "--input-sec", "135"])
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        assert completed.returncode != 0
        assert marker in completed.stderr
