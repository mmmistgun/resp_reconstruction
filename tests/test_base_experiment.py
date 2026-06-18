from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from resp_train.experiments.base import BaseExperiment, ExperimentData


class TinyDataset(Dataset):
    def __init__(self, length: int = 4):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.tensor([[float(idx), float(idx + 1)]])
        target = x * 0.0
        return {"x": x.float(), "target": target.float(), "meta": {"dataset_row_id": idx}}


class ConstantLoss(torch.nn.Module):
    def forward(self, pred, target):
        loss = pred.sum() * 0.0 + 1.0
        return loss, {"constant": loss.detach()}


class EpochAwareLoss(ConstantLoss):
    def __init__(self):
        super().__init__()
        self.epochs = []

    def set_epoch(self, epoch: int):
        self.epochs.append(int(epoch))


class ToyExperiment(BaseExperiment):
    task_name = "toy"

    def build_data(self):
        loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)
        return ExperimentData(
            train_loader=loader,
            val_loader=loader,
            audit_frame=pd.DataFrame({"split": ["train", "val"]}),
            audit_summary=pd.DataFrame({"n_windows": [4]}),
            extras={},
        )

    def build_model(self):
        return torch.nn.Conv1d(1, 1, kernel_size=1)

    def build_loss(self):
        return ConstantLoss()

    def run_baseline(self, data, run_dir):
        pd.DataFrame({"baseline": [1.0]}).to_csv(run_dir / "baseline_metrics.csv", index=False)

    def evaluate_best(self, model, data, run_dir):
        pd.DataFrame({"metric": [1.0]}).to_csv(run_dir / "metrics.csv", index=False)
        torch.save({"ok": True}, run_dir / "predictions_marker.pt")


class EpochAwareExperiment(ToyExperiment):
    def build_loss(self):
        self.loss = EpochAwareLoss()
        return self.loss


def _cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "outputs": {"run_root": str(tmp_path / "runs")},
            "training": {
                "seed": 1,
                "device": "cpu",
                "epochs": 2,
                "learning_rate": 0.01,
                "patience": 1,
                "min_delta": 0.0,
                "lr_scheduler": "none",
                "grad_clip_norm": None,
                "use_amp": False,
            },
        }
    )


def test_base_experiment_runs_lifecycle_and_writes_outputs(tmp_path: Path):
    run_dir = ToyExperiment(_cfg(tmp_path)).train()

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "train_history.csv").exists()
    assert (run_dir / "baseline_metrics.csv").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "predictions_marker.pt").exists()


def test_base_experiment_early_stopping_records_reason(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.training.epochs = 5
    cfg.training.patience = 1

    run_dir = ToyExperiment(cfg).train()
    history = pd.read_csv(run_dir / "train_history.csv")

    assert history["epoch"].max() < 5
    assert (run_dir / "train.log").read_text(encoding="utf-8").find("early_stop") >= 0


def test_base_experiment_epoch_log_includes_train_and_val_metric_parts(tmp_path: Path):
    run_dir = ToyExperiment(_cfg(tmp_path)).train()
    log_text = (run_dir / "train.log").read_text(encoding="utf-8")

    assert "epoch=1 | train: loss=1.000000 constant=1.000000 | val: loss=1.000000 constant=1.000000" in log_text


def test_base_experiment_notifies_loss_about_current_epoch(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.training.epochs = 3
    cfg.training.patience = 0
    experiment = EpochAwareExperiment(cfg)

    experiment.train()

    assert experiment.loss.epochs == [1, 2, 3]


def test_checkpoint_gate_accepts_direction_metric_range(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.training.checkpoint_gate = {
        "metric": "val_signed_cosine",
        "max": 0.5,
    }
    experiment = ToyExperiment(cfg)

    assert experiment._checkpoint_gate_allows({"val_loss": 1.0, "val_signed_cosine": 0.2})
    assert not experiment._checkpoint_gate_allows({"val_loss": 0.8, "val_signed_cosine": 1.7})


def test_checkpoint_gate_auto_direction_uses_active_signed_corr(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.loss = {
        "signed_corr_weight": 0.1,
        "signed_cosine_weight": 0.0,
    }
    cfg.training.checkpoint_gate = {
        "metric": "auto_direction",
        "max": 0.5,
    }
    experiment = ToyExperiment(cfg)

    assert experiment._checkpoint_gate_allows(
        {"val_loss": 1.0, "val_signed_corr": 0.2, "val_signed_cosine": 0.0}
    )
    assert not experiment._checkpoint_gate_allows(
        {"val_loss": 0.8, "val_signed_corr": 1.7, "val_signed_cosine": 0.0}
    )


def test_checkpoint_gate_rejects_inactive_signed_metric(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.loss = {
        "signed_corr_weight": 0.1,
        "signed_cosine_weight": 0.0,
    }
    cfg.training.checkpoint_gate = {
        "metric": "val_signed_cosine",
        "max": 0.5,
    }
    experiment = ToyExperiment(cfg)

    with pytest.raises(ValueError, match="未启用"):
        experiment._checkpoint_gate_allows({"val_loss": 1.0, "val_signed_cosine": 0.0})


def test_training_with_checkpoint_gate_requires_passing_checkpoint(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.training.epochs = 2
    cfg.training.patience = 0
    cfg.training.checkpoint_gate = {
        "metric": "val_constant",
        "max": 0.5,
    }

    with pytest.raises(ValueError, match="没有 epoch 满足 checkpoint_gate"):
        ToyExperiment(cfg).train()
