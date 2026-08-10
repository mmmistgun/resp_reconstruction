from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from resp_train.crd.config import (
    CRD_DIAGNOSTIC_PROTOCOL_VERSION,
    CRD_EXPLORATORY_PROTOCOL_VERSION,
    CRD_PROTOCOL_VERSION,
    CRD_S1F_PROTOCOL_VERSION,
    CRD_S2A_PROTOCOL_VERSION,
    CRD_S2BR_PROTOCOL_VERSION,
    load_crd_config,
)
from resp_train.crd.experiment import CRDExperiment
from resp_train.engine import collect_predictions


class _IdentityDataset(Dataset):
    def __init__(self) -> None:
        time = torch.arange(18000, dtype=torch.float32) / 100.0
        envelope = 1.0 + 0.3 * torch.sin(2.0 * torch.pi * 0.01 * time)
        self.waveform = envelope * torch.sin(2.0 * torch.pi * 0.2 * time)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        del index
        return {
            "x": self.waveform[None, :],
            "target": self.waveform[None, :],
            "meta": {
                "dataset_row_id": 1,
                "split": "val",
                "input_set": "test",
                "samp_id": 7,
                "coupling_state_id": 3,
            },
        }


class _ScaledIdentity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, signal: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        return {"waveform": signal * self.scale}


@pytest.mark.parametrize(
    ("config_path", "expected_protocol"),
    [
        ("configs/crd_v1/crd_101_b0_coarse.yaml", CRD_PROTOCOL_VERSION),
        ("configs/crd_v1/crd_105_direct_coarse.yaml", CRD_DIAGNOSTIC_PROTOCOL_VERSION),
        ("configs/crd_v1/crd_102_b0_local_mamba.yaml", CRD_EXPLORATORY_PROTOCOL_VERSION),
        ("configs/crd_v1/crd_106_b0_hier_mamba.yaml", CRD_S1F_PROTOCOL_VERSION),
        ("configs/crd_v1/crd_202_base_legacy_energy.yaml", CRD_S2A_PROTOCOL_VERSION),
        ("configs/crd_v1/crd_205_base_em_static.yaml", CRD_S2BR_PROTOCOL_VERSION),
        ("configs/crd_v1/crd_207_base_cap_em.yaml", CRD_S2BR_PROTOCOL_VERSION),
    ],
)
def test_crd_experiment_writes_complete_nonresumable_lifecycle(
    monkeypatch,
    tmp_path,
    config_path: str,
    expected_protocol: str,
) -> None:
    cfg = load_crd_config(
        config_path,
        overrides=[
            "protocol.run_role=smoke",
            "data.max_train_windows=1",
            "data.max_val_windows=1",
            "training.epochs=1",
            "training.batch_size=1",
            "training.gradient_accumulation_steps=1",
            "training.device=cpu",
            "training.show_progress=false",
            f"outputs.run_root={tmp_path.as_posix()}",
        ],
    )
    dataset = _IdentityDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    bundle = SimpleNamespace(loader=loader, dataset=dataset)
    data = SimpleNamespace(
        train=bundle,
        val=bundle,
        audit_summary=pd.DataFrame([{"split": "val", "n_windows": 1}]),
    )
    monkeypatch.setattr("resp_train.crd.experiment.build_tho_data", lambda _cfg: data)
    monkeypatch.setattr("resp_train.crd.experiment.build_crd_model", lambda _cfg: _ScaledIdentity())

    run_dir = CRDExperiment(cfg).train()
    history = pd.read_csv(run_dir / "train_history.csv")
    assert list(history.columns) == [
        "epoch",
        "optimizer_update",
        "train_loss_total",
        "train_loss_sync",
        "train_loss_effort",
        "first_learning_rate",
        "last_learning_rate",
        "val_core_loss",
        "val_local_rr_mae",
    ]
    assert history.loc[0, "optimizer_update"] == 1
    checkpoint = torch.load(run_dir / "checkpoint_best_local_rr.pt", map_location="cpu")
    assert checkpoint["extra_state"]["protocol"] == expected_protocol
    assert checkpoint["extra_state"]["resume_supported"] is False
    assert checkpoint["extra_state"]["update_index"] == 1
    for filename in (
        "checkpoint_final.pt",
        "config.yaml",
        "run_manifest.json",
        "audit.csv",
        "optimizer_parameter_groups.json",
        "metrics.csv",
        "metrics_summary.csv",
        "runtime_summary.json",
    ):
        assert (run_dir / filename).exists()
    runtime = pd.read_json(run_dir / "runtime_summary.json", typ="series")
    assert runtime["device"] == "cpu"
    assert pd.isna(runtime["peak_allocated_mib"])


class _RegularizedScaledIdentity(_ScaledIdentity):
    def __init__(self) -> None:
        super().__init__()
        self.P = torch.nn.Parameter(torch.tensor([1.0, 0.0]))

    def regularization_terms(self):
        return {"loss_proto": self.P.square().sum()}


def test_morphology_lifecycle_records_frozen_prototype_regularizer(monkeypatch, tmp_path) -> None:
    cfg = load_crd_config(
        "configs/crd_v1/crd_204_base_morphology.yaml",
        overrides=[
            "protocol.run_role=smoke",
            "data.max_train_windows=1",
            "data.max_val_windows=1",
            "training.epochs=1",
            "training.batch_size=1",
            "training.device=cpu",
            "training.show_progress=false",
            f"outputs.run_root={tmp_path.as_posix()}",
        ],
    )
    dataset = _IdentityDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    bundle = SimpleNamespace(loader=loader, dataset=dataset)
    data = SimpleNamespace(
        train=bundle,
        val=bundle,
        audit_summary=pd.DataFrame([{"split": "val", "n_windows": 1}]),
    )
    monkeypatch.setattr("resp_train.crd.experiment.build_tho_data", lambda _cfg: data)
    monkeypatch.setattr("resp_train.crd.experiment.build_crd_model", lambda _cfg: _RegularizedScaledIdentity())

    run_dir = CRDExperiment(cfg).train()
    history = pd.read_csv(run_dir / "train_history.csv")
    assert history.loc[0, "train_loss_proto"] == pytest.approx(1.0)
    assert history.loc[0, "train_loss_proto_weighted"] == pytest.approx(0.0)
    assert history.loc[0, "regularizer_ramp_first"] == pytest.approx(0.0)
    checkpoint = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu")
    assert checkpoint["extra_state"]["structural_regularizer"] == {
        "name": "prototype_orthogonality",
        "weight": 1e-3,
        "ramp": "optimizer_update_5S_to_15S",
        "updates_per_epoch": 1,
    }


class _BFloatOutput(torch.nn.Module):
    def forward(self, signal: torch.Tensor, **_: object) -> torch.Tensor:
        return signal.to(torch.bfloat16)


def test_prediction_collection_converts_bfloat16_before_numpy() -> None:
    dataset = _IdentityDataset()
    predictions = collect_predictions(
        _BFloatOutput(),
        DataLoader(dataset, batch_size=1),
        device="cpu",
        max_windows=1,
    )
    assert predictions["r_tho_hat"].dtype.name == "float32"


def test_internal_crd_checkpoint_evaluation_uses_frozen_test_split(monkeypatch, tmp_path) -> None:
    cfg = load_crd_config(
        "configs/crd_v1/crd_101_b0_coarse.yaml",
        overrides=["training.device=cpu", "training.show_progress=false"],
    )
    model = _ScaledIdentity()
    checkpoint_path = tmp_path / "checkpoint_best_local_rr.pt"
    torch.save(
        {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    calls: dict[str, object] = {}

    def fake_build_window_data(_cfg, **kwargs):
        calls["window"] = kwargs
        return SimpleNamespace(loader="test-loader")

    def fake_evaluate_model(_model, loader, **kwargs):
        calls["evaluation"] = {"loader": loader, **kwargs}
        return pd.DataFrame([{"evaluation_split": "test"}])

    monkeypatch.setattr("resp_train.crd.experiment.build_crd_model", lambda _cfg: _ScaledIdentity())
    monkeypatch.setattr("resp_train.crd.experiment.build_window_data", fake_build_window_data)
    experiment = CRDExperiment(cfg)
    monkeypatch.setattr(experiment, "_evaluate_model", fake_evaluate_model)

    result = experiment.evaluate_checkpoint(checkpoint_path, split="test")

    assert result.loc[0, "evaluation_split"] == "test"
    assert calls["window"] == {
        "split": "test",
        "max_windows": None,
        "sample_strategy": "stratified_random",
        "sample_seed": 20260612,
        "shuffle": False,
    }
    assert calls["evaluation"] == {
        "loader": "test-loader",
        "evaluation_split": "test",
        "include_test_only": True,
    }
