"""双选点的synthetic CPU检查：loss独立保存且不影响RR早停或训练路径。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from scripts import m4_dual_selector_training as train
from scripts import m4_dual_selector_runtime as run
from scripts import m4_single_projection_runtime as old


def fixture_loop(monkeypatch, rr, losses, *, fail_epoch=None):
    def epoch(model, loader, loss, optimizer, cfg, update, total, index):
        if index == fail_epoch:
            raise RuntimeError("synthetic interruption")
        optimizer.zero_grad(set_to_none=True)
        value = model(torch.ones(1, 2)).square().mean()
        value.backward(); optimizer.step()
        model.fixture_epoch = index
        return {"loss": float(value.detach())}, update + 1
    def validate(model, *args, **kwargs):
        return {"loss": losses[model.fixture_epoch - 1]}, model.fixture_epoch
    monkeypatch.setattr(train.legacy, "_train_epoch", epoch)
    monkeypatch.setattr(train, "validate", validate)
    monkeypatch.setattr(train.legacy, "validate", validate)
    monkeypatch.setattr(train.legacy, "validation_local_rr_mean", lambda i, cfg: rr[i - 1])


def cfg(epochs=6):
    return OmegaConf.create({"training": {"device": "cpu", "epochs": epochs,
        "gradient_accumulation_steps": 1, "early_stopping_min_epoch": 3,
        "early_stopping_patience": 2, "early_stopping_min_delta": 0., "use_amp": False}})


def invoke(path, config, *, resume=False, original=False):
    path.mkdir(exist_ok=True)
    torch.manual_seed(37)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    data = SimpleNamespace(train=SimpleNamespace(loader=[None]), val=SimpleNamespace(loader=[None]))
    runner = train.legacy.run_epochs if original else train.run_epochs
    result = runner(model, data, None, optimizer, config, path, resume=resume)
    return result, model.state_dict()


def test_loss_improving_does_not_reset_rr_patience_or_change_training(tmp_path, monkeypatch):
    fixture_loop(monkeypatch, [.5, .4, .42, .43, .44, .45], [6, 5, 4, 3, 2, 1])
    result, state = invoke(tmp_path / "dual", cfg())
    original, old_state = invoke(tmp_path / "old", cfg(), original=True)
    assert result["completed_epochs"] == original["completed_epochs"] == 4
    assert result["best_epoch"] == 2 and result["best_loss_epoch"] == 4
    assert result["early_stopped"] and result["best_validation_loss"] == 3
    for name in state:
        assert torch.equal(state[name], old_state[name])
    for key in original:
        if "checkpoint" not in key:
            assert result[key] == original[key]
    checkpoint = torch.load(tmp_path / "dual" / result["best_loss_checkpoint"], weights_only=False)
    assert checkpoint["epoch"] == 4 and checkpoint["wait"] == 2


def test_loss_ties_keep_first_and_can_precede_rr_winner(tmp_path, monkeypatch):
    fixture_loop(monkeypatch, [.6, .5, .4, .3, .31, .32], [4, 1, 1, 2, 3, 4])
    result, _ = invoke(tmp_path / "cell", cfg())
    assert result["best_epoch"] == 4 and result["best_loss_epoch"] == 2
    assert result["completed_epochs"] == 6


def test_complete_epoch_resume_preserves_both_selectors(tmp_path, monkeypatch):
    rr, losses = [.6, .5, .4, .3, .31, .32], [4, 1, 1, 2, 3, 4]
    fixture_loop(monkeypatch, rr, losses, fail_epoch=3)
    with pytest.raises(RuntimeError, match="interruption"):
        invoke(tmp_path / "resume", cfg())
    assert len(train.legacy.completed_epochs(tmp_path / "resume")) == 2
    fixture_loop(monkeypatch, rr, losses)
    resumed, state = invoke(tmp_path / "resume", cfg(), resume=True)
    full, full_state = invoke(tmp_path / "full", cfg())
    for key in full:
        if "checkpoint" not in key:
            assert resumed[key] == full[key]
    for name in state:
        assert torch.equal(state[name], full_state[name])


def test_nonfinite_validation_loss_fails_before_saving_bad_epoch(tmp_path, monkeypatch):
    fixture_loop(monkeypatch, [.5, .4], [1., float("nan")])
    with pytest.raises((ValueError, FloatingPointError)):
        invoke(tmp_path / "bad", cfg(2))
    assert len(train.legacy.completed_epochs(tmp_path / "bad")) == 1


def test_six_fresh_cells_preserve_original_training_contract_and_snapshot(tmp_path):
    assert len(run.plan()) == 6
    assert run.spec()["early_stopping_monitor"] == "local_rr_only"
    for cell in run.plan():
        new = run.config(cell["arm"], cell["seed"], tmp_path)
        baseline = old.config(cell["arm"], cell["seed"], tmp_path)
        for section in ("model", "data", "window", "training", "loss", "evaluation"):
            assert new[section] == baseline[section]
    session = run.prepare(tmp_path / "session")
    lock = run.load_session(session)
    assert "scripts/m4_dual_selector_training.py" in lock["code_sha256"]
    assert lock["research_test"] is False
    cell = session / "cells/M4_seed20260811"
    cell.mkdir(parents=True)
    (cell / "historical_reference.json").write_text("{}")
    with pytest.raises(ValueError, match="全新"):
        run.cell_directory(session, cell.name)


def test_complete_two_selector_summary(tmp_path, monkeypatch):
    session = tmp_path / "summary_fixture"
    session.mkdir()
    monkeypatch.setattr(run, "load_session", lambda path: {})
    spec = run.spec()
    monkeypatch.setattr(run, "spec", lambda: {**spec, "data": {**spec["data"], "val_windows": 2, "val_subjects": 2}})
    environment = {"torch": "fixture", "cuda": "fixture", "packages": {}, "gpu": "fixture",
        "LD_LIBRARY_PATH_unset": True, "LD_PRELOAD_unset": True}
    for cell in run.plan():
        directory = session / "cells" / cell["cell"]
        attempt = directory / "attempts/fixture"
        attempt.mkdir(parents=True)
        run.write_json(directory / "environment.json", environment)
        er = attempt / "receipt.json"
        run.write_json(er, {"passed": True, "arm": cell["arm"], "environment": environment,
            "batch_size": 32, "accumulation": 4, "updates": 2, "mamba": "official_mamba2",
            "timing_scope": "two_update_engineering_check_including_initial_kernel_cost",
            "train_elapsed_seconds": 1., "peak_allocated_bytes": 10, "peak_reserved_bytes": 20})
        run.write_json(attempt / "engineering.json", {"path": str(attempt), "sha256": run.sha(er)})
        base = 1. if cell["arm"] == "M4" else .9
        pd.DataFrame([{"epoch": 1, "val_local_rr_mae": base, "validation.loss": 3., "elapsed_seconds": 1.},
            {"epoch": 2, "val_local_rr_mae": base + .1, "validation.loss": 2., "elapsed_seconds": 1.}]).to_csv(attempt / "train_history.csv", index=False)
        selectors = {}
        for selector, epoch, offset in (("rr", 1, 0.), ("loss", 2, .1)):
            rows = []
            for row, subject in ((1, 10), (2, 20)):
                record = {"arm": cell["arm"], "seed": cell["seed"], "dataset_row_id": row,
                    "samp_id": subject, "split": "val", "whole_rr_target_eligible": True,
                    "local_rr_target_eligible": True, "local_rr_target_eligible_windows": 9,
                    "joint_target_eligible": True, "envelope_spearman_target_eligible": True,
                    "joint_prediction_degenerate": False, "envelope_spearman_prediction_degenerate": False}
                record.update({metric: .8 if metric == run.sf.PCC else base + offset for metric in run.sf.PRIMARY})
                rows.append(record)
            metric = attempt / f"{selector}.csv"
            pd.DataFrame(rows).to_csv(metric, index=False)
            checkpoint = directory / f"{selector}.pt"
            checkpoint.write_bytes(b"synthetic checkpoint; not loaded")
            selectors[selector] = {"epoch": epoch, "checkpoint": checkpoint.name,
                "metrics": str(metric.relative_to(directory))}
        files = [p for p in directory.rglob("*") if p.is_file()]
        run.write_json(directory / "completed.json", {"arm": cell["arm"], "seed": cell["seed"],
            "best_epoch": 1, "best_checkpoint": "rr.pt", "best_local_rr": base,
            "best_loss_epoch": 2, "best_loss_checkpoint": "loss.pt", "best_validation_loss": 2.,
            "completed_epochs": 2, "optimizer_updates": 160, "early_stopped": False,
            "parameters": 100, "selectors": selectors,
            "artifacts": {str(p.relative_to(directory)): run.sha(p) for p in files}})
    output = run.summarize(session)
    assert len(pd.read_csv(output / "selection.csv")) == 12
    assert len(pd.read_csv(output / "rr_per_seed.csv")) == 30
    assert len(pd.read_csv(output / "loss_per_seed.csv")) == 30
    delta = pd.read_csv(output / "selector_delta.csv")
    np.testing.assert_allclose(delta.loc[delta.metric != run.sf.PCC, "loss_selected_minus_rr_selected"], .1)
    assert run.summarize(session) == output
