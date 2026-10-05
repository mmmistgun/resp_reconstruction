"""正式入口的配置、分片、身份与 epoch 恢复测试，全部使用合成 fixture。"""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from resp_train import patch_aligned_tf_training as run
from resp_train.crd.config import load_crd_config
from resp_train.crd.training import build_crd_optimizer, optimizer_updates_per_epoch
from resp_train.losses.task import RespirationTaskLoss
from resp_train.models.patch_aligned_tf_mamba import PatchTFConfig


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def test_matrix_shards_cover_exactly_once():
    full = run.plan()
    a, b = run.plan(0, 2), run.plan(1, 2)
    assert len(full) == 9 and (len(a), len(b)) == (5, 4)
    assert {c["cell"] for c in a}.isdisjoint(c["cell"] for c in b)
    assert {c["cell"] for c in a + b} == {c["cell"] for c in full}
    with pytest.raises(ValueError):
        run.plan(2, 2)


def test_all_configs_preserve_scientific_contract_and_update_budget(tmp_path):
    base = load_crd_config(run.ROOT / "configs/crd_tf_v1/crd_tf102_w_formal.yaml")
    for c in run.plan():
        cfg = run.config(c["patch_seconds"], c["seed"], tmp_path / c["cell"])
        for section in ("loss", "evaluation", "window"):
            assert cfg[section] == base[section]
        assert cfg.training.batch_size == 64 and cfg.training.gradient_accumulation_steps == 2
        assert optimizer_updates_per_epoch((10141 + 63) // 64, 2) == 80
        assert cfg.training.epochs == 80
        assert cfg.training.seed == cfg.model.initialization_seed == c["seed"]
        assert cfg.model.patch_aligned_tf_mamba.patch_samples == c["patch_seconds"] * 100
        assert cfg.data.train_sample_seed == base.data.train_sample_seed
        assert cfg.data.val_sample_seed == base.data.val_sample_seed
        assert cfg.data.tf_cache_path == base.data.tf_cache_path
        assert list(cfg.model.tf_representations) == ["w"]
        assert cfg.data.max_train_windows is None and cfg.data.max_val_windows is None
        assert cfg.data.drop_nonfinite_windows is False


def test_session_snapshot_idempotence_and_source_change(tmp_path, monkeypatch):
    # prepare 只读取代码与配置，不允许触碰数据工厂。
    monkeypatch.setattr(run, "build_tho_data", lambda cfg: pytest.fail("prepare 不应读取数据"))
    session = run.prepare(tmp_path / "session")
    identity = (session / "session.json").read_bytes()
    assert run.prepare(session) == session
    assert (session / "session.json").read_bytes() == identity
    payload = run.verify_session(session)
    assert run.sha(session / "source_snapshot.tar.gz") == payload["snapshot_sha256"]
    monkeypatch.setattr(run, "source_identity", lambda: {})
    with pytest.raises(ValueError):
        run.verify_session(session)


def test_cell_mutex_and_exclusive_writes(tmp_path):
    with run.mutex(tmp_path / "lock"):
        with pytest.raises(RuntimeError):
            with run.mutex(tmp_path / "lock"):
                pass
    run.write_json(tmp_path / "a.json", {"value": 1})
    with pytest.raises(FileExistsError):
        run.write_json(tmp_path / "a.json", {"value": 2})


class SyntheticDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        t = torch.arange(18000).float() / 100
        y = torch.sin(2 * torch.pi * (.2 + i * .02) * t)
        return {"x": (y + .01 * torch.cos(2 * torch.pi * t))[None], "target": y[None],
            "meta": {"dataset_row_id": i, "samp_id": 1, "split": "val"}}


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1)
        self.drop = nn.Dropout(.1)

    def forward(self, x):
        return {"waveform": self.conv(self.drop(x))}


def fixture_training(cell):
    cell.mkdir()
    cfg = run.config(2, run.SEEDS[0], cell)
    cfg.training.epochs = 3
    cfg.training.batch_size = 2
    cfg.training.early_stopping_min_epoch = 3
    cfg.training.use_amp = False
    run.set_seed(run.SEEDS[0])
    model = SmallModel()
    optimizer, _ = build_crd_optimizer(model, cfg)
    data = SimpleNamespace(train=SimpleNamespace(loader=DataLoader(SyntheticDataset(), batch_size=2, shuffle=True)),
        val=SimpleNamespace(loader=DataLoader(SyntheticDataset(), batch_size=2)))
    return model, data, RespirationTaskLoss(cfg), optimizer, cfg, cell


def test_epoch_resume_matches_uninterrupted_parameters_and_optimizer(tmp_path, monkeypatch):
    full = fixture_training(tmp_path / "full")
    full_result = run.run_epochs(*full)
    partial = fixture_training(tmp_path / "partial")
    actual = run._train_epoch
    def interrupt(model, loader, loss, optimizer, cfg, update, total, epoch):
        if epoch == 2:
            raise RuntimeError("synthetic interruption")
        return actual(model, loader, loss, optimizer, cfg, update, total, epoch)
    monkeypatch.setattr(run, "_train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run.run_epochs(*partial)
    assert len(run.completed_epochs(partial[-1])) == 1
    # 构造新对象模拟新进程；恢复必须覆盖所有随机状态和 optimizer 状态。
    model = SmallModel()
    optimizer, _ = build_crd_optimizer(model, partial[-2])
    resumed = (model, partial[1], partial[2], optimizer, partial[-2], partial[-1])
    monkeypatch.setattr(run, "_train_epoch", actual)
    result = run.run_epochs(*resumed, resume=True)
    assert result["best_epoch"] == full_result["best_epoch"]
    assert result["optimizer_updates"] == 3
    for key, value in full[0].state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    for k, value in full[3].state_dict()["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(tensor, optimizer.state_dict()["state"][k][name], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        run.run_epochs(*resumed)
    checkpoint = partial[-1] / result["final_checkpoint"]
    with checkpoint.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="身份"):
        run.completed_epochs(partial[-1])


def test_real_loss_smoke_cpu_uses_accumulation(tmp_path, monkeypatch):
    class FixtureMamba(nn.Module):
        def __init__(self, d_model, **kwargs):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.linear(x)
    monkeypatch.setattr(run, "verify_session", lambda session: {})
    # 定向验证完整模型接口，使用小容量以限制 CPU 测试成本。
    actual = run.config
    def small_config(*args, **kwargs):
        cfg = actual(*args, **kwargs)
        cfg.model.patch_aligned_tf_mamba = asdict(PatchTFConfig(dimension=32, waveform_channels=16,
            condition_channels=8, layers=1, patch_samples=400))
        return cfg
    monkeypatch.setattr(run, "config", small_config)
    session = tmp_path / "smoke"
    session.mkdir()
    output = run.smoke(session, 4, "cpu", mamba_factory=FixtureMamba, batch_size=1)
    report = run.read_json(output / "receipt.json")
    assert report["updates"] == 2 and report["accumulation"] == 2
    assert report["passed"] and not report["official_mamba"]


def test_early_stopping_keeps_earliest_tie(tmp_path, monkeypatch):
    fixture = fixture_training(tmp_path / "early_stop")
    cfg = fixture[-2]
    cfg.training.epochs = 6
    cfg.training.early_stopping_min_epoch = 3
    cfg.training.early_stopping_patience = 2
    scores = iter([2., 1., 1., 1.])
    monkeypatch.setattr(run, "validation_local_rr_mean", lambda predictions, cfg: next(scores))
    result = run.run_epochs(*fixture)
    assert result["completed_epochs"] == 4
    assert result["best_epoch"] == 2 and result["early_stopped"]


def test_summary_requires_complete_matrix_and_reuses_verified_output(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "verify_session", lambda path: {})
    monkeypatch.setattr(run, "spec", lambda: {"data": {"val_windows": 2}})
    monkeypatch.setattr(run, "summarize_task_metrics", lambda frame: pd.DataFrame({"score_mean": [frame.score.mean()]}))
    with pytest.raises(FileNotFoundError):
        run.summarize(tmp_path)
    for cell in run.plan():
        directory = tmp_path / "cells" / cell["cell"]
        directory.mkdir(parents=True)
        frame = pd.DataFrame({"dataset_row_id": [10, 20], "samp_id": [1, 2], "split": ["val"] * 2,
            "score": [1., 2.], "seed": [cell["seed"]] * 2, "patch_seconds": [cell["patch_seconds"]] * 2})
        frame.to_csv(directory / "metrics.csv", index=False)
        run.write_json(directory / "completed.json", {"metrics": "metrics.csv",
            "artifacts": {"metrics.csv": run.sha(directory / "metrics.csv")}})
    output = run.summarize(tmp_path)
    assert len(pd.read_csv(output / "per_seed.csv")) == 9
    assert run.summarize(tmp_path) == output
    (output / "per_seed.csv").write_text("tampered")
    with pytest.raises(ValueError, match="身份"):
        run.summarize(tmp_path)
