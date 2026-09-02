from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from resp_train.paper_evidence.center30_cache import (
    Center30WCacheReader,
    build_center30_w_cache_from_rows,
    center30_cwt_features,
    fixed_center30_w_transform_spec,
)
from resp_train.paper_evidence.center30_config import load_center30_config
from resp_train.paper_evidence.center30_data import (
    Center30ContextDataset,
    audit_center30_nested_views,
    center30_latent_bounds,
    crop_center30_tensors,
)
from resp_train.paper_evidence.center30_loss import Center30ContextLoss, pi30_numpy, pi30_torch
from resp_train.paper_evidence.center30_metrics import (
    center30_rr_strictly_improved,
    evaluate_center30_predictions,
    summarize_center30_metrics,
    validation_center30_rr_mean,
)
from resp_train.paper_evidence.center30_model import (
    Center30ContextModel,
    center30_shared_state_identity,
)
from resp_train.paper_evidence.center_context_cache import fixed_center_w_transform_spec


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/paper_evidence_v1/center30_context_v1.yaml"


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

    def __getitem__(self, index: int):
        return self.item


def _signal(*, frequency_hz: float = 0.2, modulation: float = 0.25) -> np.ndarray:
    time = np.arange(3000, dtype=np.float64) / 100.0
    envelope = 1.0 + modulation * np.sin(2.0 * np.pi * time / 25.0)
    return envelope * np.sin(2.0 * np.pi * frequency_hz * time)


def _predictions(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "r_tho_hat": pred[None, None].astype(np.float32),
        "tho_ref": target[None, None].astype(np.float32),
        "dataset_row_id": np.asarray([1]),
        "split": np.asarray(["val"]),
        "samp_id": np.asarray([2]),
        "coupling_state_id": np.asarray([3]),
    }


def test_center30_config_freezes_new_task_and_rejects_formal_or_test_drift() -> None:
    cfg = load_center30_config(CONFIG)
    assert cfg.protocol.name == "paper-center30-context-v1-20260902"
    assert cfg.window.output_samples == 3000
    assert cfg.data.access_splits == ["train", "val"]
    with pytest.raises(ValueError, match="test windows"):
        load_center30_config(CONFIG, overrides=["data.max_test_windows=1"])
    with pytest.raises(ValueError, match="formal gate"):
        load_center30_config(
            CONFIG,
            overrides=["protocol.run_role=formal", "protocol.stage=p4s_single_seed", "training.device=cuda:0"],
        )


def test_four_inputs_share_exact_center30_target_and_expected_latent_bounds() -> None:
    parent = _Parent().item
    expected = {3000: (0, 300), 4500: (75, 375), 6000: (150, 450), 9000: (300, 600)}
    targets = []
    for length, bounds in expected.items():
        x, target = crop_center30_tensors(parent["x"], parent["target"], input_samples=length)
        assert x.shape == (1, length)
        assert target.shape == (1, 3000)
        assert center30_latent_bounds(length) == bounds
        targets.append(target)
    assert all(torch.equal(targets[0], value) for value in targets[1:])
    assert audit_center30_nested_views([parent])["center30_targets_pointwise_identical"] is True
    with pytest.raises(ValueError, match="train/val"):
        Center30ContextDataset(_Parent("test"), input_samples=3000)


@pytest.mark.parametrize("length,context", [(3000, 60), (4500, 90), (6000, 120), (9000, 180)])
def test_center30_models_emit_fixed_30s_output(length: int, context: int) -> None:
    c201 = Center30ContextModel("c201_center30", 20260811, mamba_factory=_IdentityMamba).eval()
    w_reduced = Center30ContextModel("w_reduced_center30", 20260811, mamba_factory=_IdentityMamba).eval()
    x = torch.randn(1, 1, length)
    with torch.no_grad():
        c201_output = c201(x)
        w_output = w_reduced(x, tf={"w": torch.randn(1, 49, context)})
    assert c201_output["waveform"].shape == w_output["waveform"].shape == (1, 1, 3000)
    assert c201_output["waveform_10hz"].shape == w_output["waveform_10hz"].shape == (1, 1, 300)
    assert c201.trainable_parameter_count() > 0
    assert w_reduced.trainable_parameter_count() - c201.trainable_parameter_count() == 150048


def test_center30_shared_initialization_and_tf_contract() -> None:
    c201 = Center30ContextModel("c201_center30", 1, mamba_factory=_IdentityMamba)
    w_reduced = Center30ContextModel("w_reduced_center30", 1, mamba_factory=_IdentityMamba)
    assert center30_shared_state_identity(c201, w_reduced)["shared_tensors_identical"] is True
    x = torch.zeros(1, 1, 3000)
    with pytest.raises(ValueError, match="不得读取"):
        c201(x, tf={"w": torch.zeros(1, 49, 60)})
    with pytest.raises(ValueError, match="缺少"):
        w_reduced(x)


def test_pi30_loss_and_metrics_preserve_task_specific_contract() -> None:
    cfg = load_center30_config(CONFIG)
    signal = _signal().astype(np.float32)
    tensor = torch.tensor(signal).view(1, -1).requires_grad_()
    np.testing.assert_allclose(pi30_torch(tensor).detach(), pi30_numpy(signal[None]), rtol=0.0, atol=2e-3)
    loss_fn = Center30ContextLoss(cfg)
    target = tensor.detach().view(1, 1, -1)
    identity_loss, _ = loss_fn(target.clone(), target)
    assert float(identity_loss) < 1e-5
    metrics = evaluate_center30_predictions(_predictions(signal, signal), cfg)
    wrong = evaluate_center30_predictions(_predictions(_signal(frequency_hz=0.4), signal), cfg)
    assert metrics.loc[0, "center30_rr_mae_bpm"] < wrong.loc[0, "center30_rr_mae_bpm"]
    assert summarize_center30_metrics(metrics).loc[0, "n_samples"] == 1
    assert validation_center30_rr_mean(_predictions(signal, signal), cfg) == pytest.approx(
        metrics.loc[0, "center30_rr_mae_bpm"]
    )
    assert center30_rr_strictly_improved(0.9, 1.0)


@pytest.mark.parametrize("length,context", [(3000, 60), (4500, 90)])
def test_actual_center30_cwt_has_49_scales_and_expected_context(length: int, context: int) -> None:
    time = np.arange(length, dtype=np.float32) / 100.0
    feature, frequencies = center30_cwt_features(np.sin(2.0 * np.pi * 0.2 * time))
    assert feature.shape == (49, context)
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


def test_center30_cache_is_input_only_hashed_and_legacy_60_90_spec_compatible(monkeypatch, tmp_path: Path) -> None:
    class FakeWholeNightCache:
        def __init__(self, _: Path) -> None:
            pass

        def get_arrays(self, _: str, keys: list[str]):
            return {keys[0]: np.zeros(18000, dtype=np.float32)}

    monkeypatch.setattr("resp_train.data.cache.WholeNightCache", FakeWholeNightCache)

    from resp_train.paper_evidence.center_context_cache import reduced_target_frequencies

    def extract(view: np.ndarray):
        assert view.shape == (3000,)
        return np.zeros((49, 60), dtype=np.float32), reduced_target_frequencies()

    output = tmp_path / "cache30"
    manifest_path = build_center30_w_cache_from_rows(
        output_dir=output,
        index_path=tmp_path / "index.csv",
        rows_by_split={"train": _cache_rows("train", 1), "val": _cache_rows("val", 2)},
        input_samples=3000,
        dataset_index_sha256="fixture",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_only"] is True
    assert manifest["target_read"] is manifest["test_read"] is False
    assert Center30WCacheReader(output, split="val", input_samples=3000).get(2).shape == (49, 60)
    with pytest.raises(ValueError, match="train/val"):
        Center30WCacheReader(output, split="test", input_samples=3000)
    with pytest.raises(FileExistsError):
        build_center30_w_cache_from_rows(
            output_dir=output,
            index_path=tmp_path / "index.csv",
            rows_by_split={"train": _cache_rows("train", 1), "val": _cache_rows("val", 2)},
            input_samples=3000,
            dataset_index_sha256="fixture",
            feature_extractor=extract,
        )
    assert fixed_center30_w_transform_spec(6000) == fixed_center_w_transform_spec(6000)
    assert fixed_center30_w_transform_spec(9000) == fixed_center_w_transform_spec(9000)
