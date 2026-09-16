from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from resp_train.crd.config import load_crd_config
from resp_train.crd import experiment as native
from resp_train.paper_evidence import e2_effort_ablation as e2


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(old)


def baseline(seed=e2.SEEDS[0]):
    root = Path(__file__).resolve().parents[1]
    return load_crd_config(root / "configs/crd_tf_v1/crd_tf102_w_formal.yaml", overrides=[
        f"training.seed={seed}", f"model.initialization_seed={seed}"])


def waveform():
    t = torch.arange(18000) / 100
    target = (1 + 0.4 * torch.sin(2 * torch.pi * 0.025 * t)) * torch.sin(2 * torch.pi * 0.23 * t)
    prediction = (1 + 0.15 * torch.cos(2 * torch.pi * 0.035 * t)) * torch.sin(2 * torch.pi * 0.235 * t + 0.1)
    return prediction[None, None], target[None, None]


def test_configuration_has_one_scientific_change_and_preserves_baseline(tmp_path):
    original = baseline()
    saved = OmegaConf.to_container(original, resolve=True)
    cfg = e2.derived_config(original, output_root=tmp_path, device="cpu")
    e2.validate_config(cfg, original, output_root=tmp_path, device="cpu")
    assert OmegaConf.to_container(original, resolve=True) == saved
    assert cfg.loss.effort_weight == 0 and original.loss.effort_weight == 0.25
    assert cfg.model == original.model and cfg.data == original.data and cfg.evaluation == original.evaluation
    restored = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    restored.loss.effort_weight = 0.25
    restored.protocol = original.protocol
    restored.training.device = original.training.device
    restored.training.show_progress = original.training.show_progress
    restored.outputs = original.outputs
    assert OmegaConf.to_container(restored, resolve=True) == saved


@pytest.mark.parametrize("key,value", [("training.epochs", 79), ("training.max_learning_rate", 0.001),
    ("model.initialization_seed", 3), ("loss.effort_weight", 0.1), ("data.train_split", "test"),
    ("evaluation.local_rr_window_sec", 30), ("training.batch_size", 64), ("protocol.name", "legacy")])
def test_configuration_rejects_additional_changes(tmp_path, key, value):
    original = baseline()
    cfg = e2.derived_config(original, output_root=tmp_path, device="cpu")
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError):
        e2.validate_config(cfg, original, output_root=tmp_path, device="cpu")


def test_generic_crd_loader_remains_frozen(tmp_path):
    cfg = e2.derived_config(baseline(), output_root=tmp_path, device="cpu")
    path = tmp_path / "e2.yaml"
    OmegaConf.save(cfg, path)
    with pytest.raises(ValueError, match="冻结要求"):
        load_crd_config(path)


def test_zero_effort_has_exact_sync_loss_and_gradient(tmp_path):
    cfg = e2.derived_config(baseline(), output_root=tmp_path, device="cpu")
    pred, target = waveform()
    pred = pred.requires_grad_()
    loss = e2.RespirationTaskLoss(cfg)
    total, parts = loss(pred, target)
    torch.testing.assert_close(total, parts["loss_sync"], atol=0, rtol=0)
    actual = torch.autograd.grad(total, pred, retain_graph=True)[0]
    expected = torch.autograd.grad(parts["loss_sync"], pred)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.isfinite(actual).all() and actual.abs().max() > 0
    assert torch.isfinite(parts["loss_effort"])


def test_accumulation_numerator_matches_sync_objective(tmp_path):
    cfg = e2.derived_config(baseline(), output_root=tmp_path, device="cpu")
    pred, target = waveform()
    pred = pred.repeat(2, 1, 1).requires_grad_()
    target = target.repeat(2, 1, 1)
    loss = e2.RespirationTaskLoss(cfg)
    sums = loss.differentiable_component_sums(pred, target)
    objective = sums["loss_sync_sum"] / sums["loss_sync_count"] + loss.effort_weight * sums["loss_effort_sum"] / sums["loss_effort_count"]
    direct, _ = loss(pred, target)
    torch.testing.assert_close(objective, direct, atol=0, rtol=0)
    a = torch.autograd.grad(objective, pred, retain_graph=True)[0]
    b = torch.autograd.grad(direct, pred)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_nonfinite_is_rejected_even_with_effort_weight_zero(tmp_path):
    cfg = e2.derived_config(baseline(), output_root=tmp_path, device="cpu")
    pred, target = waveform()
    pred[0, 0, 5] = float("nan")
    with pytest.raises(FloatingPointError):
        e2.RespirationTaskLoss(cfg)(pred, target)


@pytest.mark.parametrize("seed", e2.SEEDS)
def test_same_seed_native_w0_initialization_is_identical(tmp_path, seed):
    cfg = baseline(seed)
    candidate = e2.derived_config(cfg, output_root=tmp_path, device="cpu")
    full_model, ablated_model = e2.build_crd_model(cfg), e2.build_crd_model(candidate)
    assert sum(p.numel() for p in full_model.parameters() if p.requires_grad) == 1219850
    assert full_model.state_dict().keys() == ablated_model.state_dict().keys()
    assert all(torch.equal(value, ablated_model.state_dict()[name]) for name, value in full_model.state_dict().items())


def comparison_frame():
    records = []
    for arm in ("W0_FULL", e2.ARM):
        for i, seed in enumerate(e2.SEEDS):
            value = [1., 2., 4.][i]
            change = 0 if arm == "W0_FULL" else [0.1, -0.2, 1.2][i]
            records.append({"arm": arm, "seed": seed, **{k + "_mean": value + change for k in e2.ERRORS},
                            e2.PCC + "_mean": 0.8 if arm == "W0_FULL" else [0.7, 0.9, 0.6][i]})
    return pd.DataFrame(records)


def test_paired_statistics_keep_direction_and_aggregation_distinct():
    paired, aggregate = e2.paired_tables(comparison_frame().sample(frac=1, random_state=8))
    assert len(paired) == 15 and len(aggregate) == 5
    row = aggregate.iloc[0]
    assert row.paired_delta_mean == pytest.approx(10.)
    assert row.paired_delta_sample_sd == pytest.approx(20.)
    assert row.delta_of_seed_means == pytest.approx(100 * 1.1 / 7)
    assert row.full_better_seeds == 2 and row.sync_only_better_seeds == 1
    assert aggregate.iloc[-1].paired_delta_mean == pytest.approx((0.1 - 0.1 + 0.2) / 3)


@pytest.mark.parametrize("failure", ["partial", "duplicate", "nan", "zero"])
def test_paired_summary_rejects_invalid_matrix(failure):
    frame = comparison_frame()
    if failure == "partial": frame = frame.iloc[:-1]
    elif failure == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    elif failure == "nan": frame.loc[0, e2.PCC + "_mean"] = np.nan
    else: frame.loc[0, e2.ERRORS[0] + "_mean"] = 0
    with pytest.raises((ValueError, FloatingPointError)):
        e2.paired_tables(frame)


def test_failure_attempt_preserves_partial_and_has_no_success_receipt(tmp_path):
    with pytest.raises(RuntimeError, match="fixture failure"):
        with e2.attempt(tmp_path, "a" * 64, "formal", e2.SEEDS[0]) as path:
            (path / "partial.txt").write_text("keep")
            raise RuntimeError("fixture failure")
    assert (path / "partial.txt").read_text() == "keep"
    assert (path / "lifecycle_failed.json").is_file()
    assert not (path / "freeze_receipt.json").exists()


class SmallFixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 5, padding=2)

    def forward(self, x, **kwargs):
        return {"waveform": x + self.conv(x)}


@pytest.fixture
def native_training_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(e2, "ROOT", tmp_path)
    monkeypatch.setattr(e2, "EPOCHS", 2)
    monkeypatch.setattr(e2, "UPDATES_PER_EPOCH", 2)
    monkeypatch.setattr(e2, "COUNTS", {"train": 4, "val": 2})
    monkeypatch.setattr(e2, "runtime_preflight", lambda device: {"fixture": "CPU", "git": {"commit": "synthetic"}})
    pred, target = waveform()
    def samples(count, split):
        return [{"x": pred[0] + i * 0.001 * torch.cos(torch.arange(18000) * 0.019)[None], "target": target[0],
                 "tf": {"w": torch.zeros(97, 360)}, "meta": {"dataset_row_id": i, "samp_id": i, "split": split}}
                for i in range(count)]
    train, val = samples(4, "train"), samples(2, "val")
    rows = pd.DataFrame({"dataset_row_id": [0, 1], "samp_id": [0, 1], "split": "val"})
    data = SimpleNamespace(train=SimpleNamespace(loader=DataLoader(train, batch_size=2)),
                           val=SimpleNamespace(loader=DataLoader(val, batch_size=2)), audit_summary=pd.DataFrame({"fixture": [True]}))
    monkeypatch.setattr(native, "build_tho_data", lambda cfg: data)
    monkeypatch.setattr(native, "build_crd_model", lambda cfg: SmallFixtureModel())
    baselines = {}
    entries = []
    for seed in e2.SEEDS:
        cfg = baseline(seed)
        cfg.training.epochs = 2
        baselines[str(seed)] = OmegaConf.to_container(cfg, resolve=True)
        source = tmp_path / f"full_{seed}"
        source.mkdir()
        reference = {k + "_mean": [0.5 if k == e2.PCC else 1.] for k in e2.PRIMARY}
        reference.update({k + "_n": [2] for k in e2.PRIMARY})
        pd.DataFrame(reference).to_csv(source / "metrics_summary.csv", index=False)
        entries.append({"seed": seed, "run_dir": source.name, "validation_summary": e2.identity(source / "metrics_summary.csv")})
    lock = {"baselines": baselines, "w0_entries": entries}
    lock_hash = "a" * 64
    monkeypatch.setattr(e2, "load_lock", lambda: (lock, lock_hash))
    def audit(lock, cfg, output):
        rows.to_csv(output / "val_rows.csv", index=False)
        return {"val": rows}
    monkeypatch.setattr(e2, "audit_sources", audit)
    with e2.attempt(tmp_path / "gpu", lock_hash, "gpu_smoke") as gpu:
        e2.write_json(gpu / "environment.json", {"fixture": "CPU"})
        e2.write_json(gpu / "gpu_acceptance.json", {"passed": True, "effort_weight": 0.0})
    return tmp_path, gpu


def test_native_training_three_seed_lifecycle_and_immutable_summary(native_training_fixture):
    root, gpu = native_training_fixture
    completed = []
    for seed in e2.SEEDS:
        path = e2.run_formal(seed, gpu_receipt=gpu, device="cpu")
        receipt = json.loads((path / "formal_receipt.json").read_text())
        assert receipt["epochs"] == 2 and receipt["updates"] == 4
        run = path / receipt["run_dir"]
        history = pd.read_csv(run / "train_history.csv")
        assert receipt["selected_epoch"] == int(history.loc[history.val_local_rr_mae.idxmin(), "epoch"])
        assert (run / "checkpoint_best_local_rr.pt").is_file() and (run / "checkpoint_final.pt").is_file()
        completed.append(path)
    with pytest.raises(FileExistsError):
        e2.run_formal(e2.SEEDS[0], gpu_receipt=gpu, device="cpu")
    output = e2.summarize(completed)
    assert len(pd.read_csv(output / "seed_metrics.csv")) == 6
    assert len(pd.read_csv(output / "paired_seed_delta.csv")) == 15
    assert len(pd.read_csv(output / "three_seed_comparison.csv")) == 5
    with pytest.raises(FileExistsError):
        e2.summarize(completed)
    with pytest.raises(ValueError):
        e2.summarize(completed[:2])


def test_selector_and_history_fail_explicitly(native_training_fixture):
    root, gpu = native_training_fixture
    completed = e2.run_formal(e2.SEEDS[0], gpu_receipt=gpu, device="cpu")
    receipt = json.loads((completed / "formal_receipt.json").read_text())
    run = completed / receipt["run_dir"]
    history = pd.read_csv(run / "train_history.csv")
    cfg = OmegaConf.load(run / "config.yaml")
    history.loc[:, "val_local_rr_mae"] = 1.
    assert e2.validate_history(history, cfg) == 1
    with pytest.raises(ValueError, match="80 epochs"):
        e2.validate_history(history.iloc[:1], cfg)
    history.loc[0, "last_learning_rate"] *= 2
    with pytest.raises(ValueError, match="schedule"):
        e2.validate_history(history, cfg)
    checkpoint_path = run / "checkpoint_best_local_rr.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["metrics"]["train_loss_total"] += 0.01
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="对应 history"):
        e2.validate_run(run, cfg, pd.read_csv(completed / "val_rows.csv"))
