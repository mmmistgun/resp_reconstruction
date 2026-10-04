from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from scripts import apor_grid_decoder_v2_runtime as exp
from scripts.apor_grid_decoder_v2_model import (
    ARMS, TRAIN_ARMS, REFERENCE_ARMS, ARM_SPECS, SEEDS,
    GridDecoderModel, PostFilmTemporal, model_contract,
)
from resp_train.paper_evidence.patch_apor_v1_model import PatchAporModel
from resp_train.losses.task import RespirationTaskLoss
from resp_train.crd.training import build_crd_optimizer


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def replace_trunk(model):
    model.base.local_blocks = nn.ModuleList([nn.Identity() for _ in model.base.local_blocks])
    return model


def test_matrix_contract_and_balanced_new_training():
    exp.load_spec()
    assert len(exp.scientific_plan()) == 12 and len(exp.plan()) == 6
    layout = exp.parallel_layout(["cuda:0", "cuda:1"])
    first, second = [w["cells"] for w in layout["workers"]]
    ids = lambda cells: {(c["arm"], c["seed"]) for c in cells}
    assert len(first) == len(second) == 3 and not ids(first) & ids(second)
    assert ids(first) | ids(second) == ids(exp.plan())
    assert {c["arm"] for c in first} == set(TRAIN_ARMS)
    for arm in ARMS:
        model = GridDecoderModel(arm, SEEDS[0])
        assert sum(p.numel() for p in model.parameters()) == model_contract(arm)["parameters"]
        cfg = exp.config(arm, SEEDS[0], Path("SYNTHETIC"))
        old = exp.legacy.config(ARM_SPECS[arm].source_arm, SEEDS[0], Path("OLD"))
        assert exp.scientific_config(cfg) == exp.scientific_config(old)
    assert sum(p.numel() for p in PostFilmTemporal().parameters()) == 37440
    with pytest.raises(ValueError):
        exp.parallel_layout(["cuda:0", "cuda:0"])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("arm", REFERENCE_ARMS)
def test_original_reference_and_initial_identity_forward_backward(seed, arm):
    old = replace_trunk(PatchAporModel(ARM_SPECS[arm].source_arm, seed)).eval()
    reference = replace_trunk(GridDecoderModel(arm, seed)).eval()
    added = replace_trunk(GridDecoderModel("N1" if arm == "N0" else "D1", seed)).eval()
    assert set(old.state_dict()) == set(reference.state_dict())
    for name, value in old.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name])
        assert torch.equal(value, added.state_dict()[name])
    with torch.no_grad():
        old.branches["w"].final_projection.weight.normal_(std=.02)
    reference.load_state_dict(old.state_dict())
    added.load_state_dict(old.state_dict(), strict=False)
    batch = exp.synthetic_batch(1, 832)
    outputs, gradients = [], []
    for model in (old, reference, added):
        out = model(batch["x"], tf=batch["tf"])["waveform"]
        out.square().mean().backward()
        outputs.append(out)
        gradients.append({n: p.grad for n, p in model.named_parameters() if not n.startswith("postfilm.")})
    for i in (1, 2):
        torch.testing.assert_close(outputs[0], outputs[i], rtol=0, atol=0)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[i][name], rtol=0, atol=0)


@pytest.mark.parametrize("arm", TRAIN_ARMS)
def test_three_updates_activate_temporal_block_and_roundtrip(arm):
    cfg = exp.config(arm, SEEDS[0], Path("SYNTHETIC"))
    model = replace_trunk(GridDecoderModel(arm, SEEDS[0]))
    optimizer, partition = build_crd_optimizer(model, cfg)
    assert set(partition.decay_names) | set(partition.no_decay_names) == set(dict(model.named_parameters()))
    initial = {n: p.detach().clone() for n, p in model.postfilm.named_parameters()}
    batch = exp.synthetic_batch(1, 12)
    loss_fn = RespirationTaskLoss(cfg)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_fn(model(batch["x"], tf=batch["tf"]), batch["target"])
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    for name, value in model.postfilm.named_parameters():
        assert value.grad.ne(0).any() and not torch.equal(initial[name], value)
    clone = replace_trunk(GridDecoderModel(arm, SEEDS[0])).eval()
    clone.load_state_dict(model.state_dict())
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch["x"], tf=batch["tf"])["waveform"], clone(batch["x"], tf=batch["tf"])["waveform"], rtol=0, atol=0)
    batch["x"][..., 4] = float("nan")
    with pytest.raises(FloatingPointError):
        model(batch["x"], tf=batch["tf"])


def metric_fixture():
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            for row in range(4):
                item = {"arm": arm, "seed": seed, "dataset_row_id": row, "samp_id": row % 2, "split": "val",
                        "whole_rr_target_eligible": True, "local_rr_target_eligible": True, "joint_target_eligible": True,
                        "joint_prediction_degenerate": False, "envelope_spearman_prediction_degenerate": False}
                for metric in exp.sf.PRIMARY:
                    item[metric] = .8 if metric == exp.sf.PCC else {"N0": 1., "N1": .9, "D0": 1.2, "D1": .8}[arm]
                rows.append(item)
    return pd.DataFrame(rows)


def test_complete_factorial_interaction_and_validation_selection(monkeypatch):
    monkeypatch.setitem(exp.COUNTS, "val", 4)
    frame = metric_fixture()
    tables = exp.summary_tables(frame)
    assert len(tables["per_seed"]) == 60
    assert len(tables["paired_delta"]) == 60
    assert len(tables["interaction_by_seed"]) == 30
    inter = tables["interaction_by_seed"]
    assert np.allclose(inter.loc[inter.metric.ne(exp.sf.PCC), "interaction"], -.3)
    decision = exp.validation_decision(tables)
    assert decision["candidate"] == "D1" and all(c["eligible"] for c in decision["checks"])
    with pytest.raises(ValueError, match="12-cell"):
        exp.summary_tables(frame[frame.arm != "N0"])
    bad = frame.copy()
    bad.loc[0, exp.sf.PRIMARY[0]] = np.nan
    with pytest.raises(FloatingPointError):
        exp.summary_tables(bad)
    # 更低RR但PCC损失超限不能获得新增臂资格。
    frame.loc[frame.arm == "D1", exp.sf.PCC] = .7
    assert exp.validation_decision(exp.summary_tables(frame))["candidate"] == "N1"


def test_failed_attempt_preserved_and_references_cannot_train(tmp_path):
    exp.write_json(tmp_path / "session.json", {"fixture": True})
    parent = tmp_path / "formal/N1/seed_1"
    with pytest.raises(RuntimeError, match="fixture"):
        with exp.attempt(parent, tmp_path, "formal", arm="N1", seed=1) as failed:
            raise RuntimeError("fixture")
    assert (failed / "failed.json").exists()
    with pytest.raises(RuntimeError, match="retry-failed"):
        exp.completed(parent, tmp_path, "formal", arm="N1", seed=1)
    with pytest.raises(ValueError, match="禁止重训"):
        exp.run_formal(tmp_path, "N0", SEEDS[0], "cpu", tmp_path)


@pytest.mark.parametrize("arm", TRAIN_ARMS)
def test_native_trainer_selected_validation_export(tmp_path, monkeypatch, arm):
    cfg = exp.config(arm, SEEDS[0], tmp_path / arm)
    cfg.training.epochs = 2
    cfg.training.early_stopping_min_epoch = 1
    cfg.training.batch_size = 2
    cfg.training.use_amp = False

    class Loader(list):
        dataset = range(2)

    builder = exp.build_model
    monkeypatch.setattr(exp, "build_model", lambda cfg: replace_trunk(builder(cfg)))

    class SyntheticExperiment(exp.ModuleExperiment):
        def _build_data(self):
            train = exp.synthetic_batch(2, 241, split="train")
            val = exp.synthetic_batch(2, 242, split="val", row_offset=100)
            return SimpleNamespace(train=SimpleNamespace(loader=Loader([train])), val=SimpleNamespace(loader=Loader([val])),
                                   audit_summary=pd.DataFrame({"synthetic": [True]}))

    initialization = tmp_path / "initialization.json"
    run = SyntheticExperiment(cfg, initialization_path=initialization).train()
    frame = pd.read_csv(run / "metrics.csv")
    assert len(frame) == 2 and frame.arm.eq(arm).all()
    assert np.isfinite(frame.overlap_relative_mse).all()
    assert np.load(run / "validation_prediction.npy").shape == (2, 1, 18000)
    assert np.load(run / "validation_reference.npy").shape == (2, 1, 18000)
    monkeypatch.setattr(exp, "UPDATES_PER_EPOCH", 1)
    monkeypatch.setattr(exp, "PLANNED_UPDATES", 2)
    monkeypatch.setattr(exp.sf, "UPDATES_PER_EPOCH", 1)
    monkeypatch.setattr(exp.sf, "PLANNED_UPDATES", 2)
    monkeypatch.setattr(exp.sf, "EPOCHS", 2)
    monkeypatch.setattr(exp.sf, "EARLY_STOP_MIN_EPOCH", 1)
    receipt = exp.validate_run(run, cfg, frame[["dataset_row_id", "samp_id", "split"]], initialization)
    assert receipt["validation_rows"] == 2


@pytest.mark.parametrize("exit_codes", [(0, 0), (0, 1)])
def test_parallel_all_gpu_acceptance_precedes_training_and_summary(tmp_path, monkeypatch, exit_codes):
    exp.write_json(tmp_path / "session.json", {"fixture": True})
    monkeypatch.setattr(exp, "load_session", lambda session: {})
    monkeypatch.setattr(exp, "cuda_environment", lambda device: {"device_uuid": device})
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    accepted, calls, summaries = [], [], []
    monkeypatch.setattr(exp, "run_gpu", lambda session, device, retry: accepted.append(device) or tmp_path / device)
    monkeypatch.setattr(exp, "verify_gpu", lambda *args: None)

    class Process:
        def __init__(self, command, **kwargs):
            assert accepted == ["cuda:0", "cuda:1"]
            self.returncode = exit_codes[len(calls)]
            self.pid = 12340 + len(calls)
            calls.append(command)

        def poll(self):
            return self.returncode

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(exp.subprocess, "Popen", Process)
    monkeypatch.setattr(exp, "summarize", lambda *args: summaries.append(args) or tmp_path / "summary")
    if exit_codes == (0, 0):
        exp.run_parallel(tmp_path, ["cuda:0", "cuda:1"])
        assert len(summaries) == 1
    else:
        with pytest.raises(RuntimeError, match="worker失败"):
            exp.run_parallel(tmp_path, ["cuda:0", "cuda:1"])
        assert not summaries and list((tmp_path / "dispatch").rglob("failed.json"))
    assert len(calls) == 2


def test_equivalence_failure_never_launches_workers(tmp_path, monkeypatch):
    exp.write_json(tmp_path / "session.json", {"fixture": True})
    monkeypatch.setattr(exp, "load_session", lambda session: {})
    monkeypatch.setattr(exp, "cuda_environment", lambda device: {"device_uuid": device})

    def reject(*args):
        raise RuntimeError("equivalence failed")

    monkeypatch.setattr(exp, "run_gpu", reject)
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("不应启动训练"))
    with pytest.raises(RuntimeError, match="equivalence failed"):
        exp.run_parallel(tmp_path, ["cuda:0", "cuda:1"])


def test_deterministic_equivalence_isolated_from_training_environment(tmp_path, monkeypatch):
    exp.write_json(tmp_path / "session.json", {"fixture": True})
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    environment = {"device_name": "fixture", "dependencies": {}, "deterministic_algorithms": False}
    references = {f"{a}/{s}": {"environment": environment, "initialization": {}}
                  for a in REFERENCE_ARMS for s in SEEDS}
    monkeypatch.setattr(exp, "load_session", lambda _: {"references": references})
    monkeypatch.setattr(exp, "cuda_environment", lambda _: environment)
    calls = []

    def isolated(command, *, cwd, env, check):
        assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert "CUBLAS_WORKSPACE_CONFIG" not in exp.os.environ
        assert check
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        exp.write_json(output, {"passed": True, "mode": "isolated_deterministic_backward", "cells": [
            {"arm": a, "seed": s, "initialization": {}, "fp32": "exact", "bf16": "exact"}
            for a in REFERENCE_ARMS for s in SEEDS]})

    def training(arm, seed, batch_size, device):
        assert "CUBLAS_WORKSPACE_CONFIG" not in exp.os.environ
        return {"arm": arm, "seed": seed, "batch_size": batch_size, "updates": 3,
                "modules_with_gradient_and_update": 1, "peak_reserved_fraction": .1, "eval_loss": 1.0}

    monkeypatch.setattr(exp.subprocess, "run", isolated)
    monkeypatch.setattr(exp, "engineering_cell", training)
    accepted = exp.run_gpu(tmp_path, "cuda:0")
    exp.verify_gpu(accepted, tmp_path)
    assert len(calls) == 1 and "CUBLAS_WORKSPACE_CONFIG" not in exp.os.environ
