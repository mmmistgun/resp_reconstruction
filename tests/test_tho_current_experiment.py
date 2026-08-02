from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from resp_train.config import load_config
from resp_train.experiments.tho import ThoExperiment, evaluate_tho_checkpoint


class _IdentityDataset(Dataset):
    def __init__(self) -> None:
        time = torch.arange(18000, dtype=torch.float32) / 100.0
        amplitude = 1.0 + 0.3 * torch.sin(2.0 * torch.pi * 0.01 * time)
        self.waveform = amplitude * torch.sin(2.0 * torch.pi * 0.2 * time)

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

    def forward(self, signal: torch.Tensor, **_: object) -> torch.Tensor:
        return signal * self.scale


def test_current_experiment_keeps_earliest_local_rr_tie(monkeypatch, tmp_path) -> None:
    cfg = load_config(
        "configs/tho_research_v2.yaml",
        overrides=[
            "training.epochs=2",
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
    monkeypatch.setattr("resp_train.experiments.tho.build_tho_data", lambda _cfg: data)
    monkeypatch.setattr("resp_train.experiments.tho.build_model", lambda _cfg: _ScaledIdentity())

    run_dir = ThoExperiment(cfg).train()
    history = pd.read_csv(run_dir / "train_history.csv")
    assert list(history.columns) == [
        "epoch",
        "train_loss_total",
        "train_loss_sync",
        "train_loss_rhythm",
        "train_loss_effort",
        "train_loss_pol",
        "val_core_loss",
        "val_local_rr_mae",
    ]
    assert np.allclose(history["val_local_rr_mae"], 0.0)
    best = torch.load(run_dir / "checkpoint_best_local_rr.pt", map_location="cpu")
    assert best["epoch"] == 1
    assert (run_dir / "checkpoint_final.pt").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "metrics_summary.csv").exists()
    assert (run_dir / "run_manifest.json").exists()


def test_public_evaluation_api_requires_designated_test_confirmation(tmp_path) -> None:
    with pytest.raises(ValueError, match="显式确认"):
        evaluate_tho_checkpoint(
            checkpoint_path=tmp_path / "missing.pt",
            config_path=None,
            metrics_output_path=None,
            split="test",
        )
