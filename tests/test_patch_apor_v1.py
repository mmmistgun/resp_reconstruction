from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from resp_train.crd.training import build_crd_optimizer
from resp_train.losses.task import RespirationTaskLoss
from resp_train.paper_evidence import patch_apor_v1 as exp
from resp_train.paper_evidence.patch_apor_v1_model import (
    ARMS, ARM_SPECS, SEEDS, CoordinateSample, OverlapAdd, PatchAporModel,
)
from resp_train.paper_evidence.w0_structural_factorial_v1_model import W0StructuralFactorialModel


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def replace_trunk(model):
    # 本fixture只验证新模块；原生Mamba前后向由独立GPU验收覆盖。
    model.base.local_blocks = nn.ModuleList([nn.Identity() for _ in model.base.local_blocks])
    return model


def test_matrix_and_configs_preserve_scientific_contract():
    assert len(exp.plan()) == 39
    assert len({(x["arm"], x["seed"]) for x in exp.plan()}) == 39
    exp.load_spec()
    reference = exp.config("M0", SEEDS[0], Path("/tmp/patch_apor_fixture"))
    for arm in ARMS:
        for seed in SEEDS:
            cfg = exp.config(arm, seed, Path("/tmp/patch_apor_fixture"))
            for key in ("loss", "evaluation", "window"):
                assert cfg[key] == reference[key]
            assert cfg.training.epochs == 80 and cfg.training.early_stopping_min_epoch == 30
            assert cfg.training.batch_size == 128
            assert cfg.training.seed == cfg.model.initialization_seed == seed
            assert list(cfg.model.tf_representations) == ([] if arm == "M1" else ["w"])
    with pytest.raises(ValueError):
        exp.config("M0", 1, Path("/tmp/unused"))


@pytest.mark.parametrize("seed", SEEDS)
def test_m0_matches_frozen_patch_tm0_ref0(seed):
    reference = replace_trunk(W0StructuralFactorialModel("sfv1_patch_tm0_ref0", seed)).eval()
    model = replace_trunk(PatchAporModel("M0", seed)).eval()
    assert reference.state_dict().keys() == model.state_dict().keys()
    for name, value in reference.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    # 非零FiLM也必须等价，避免零初始化掩盖条件路径接错。
    with torch.no_grad():
        reference.branches["w"].final_projection.weight.normal_(std=0.01)
        reference.branches["w"].final_projection.bias.normal_(std=0.01)
        model.load_state_dict(reference.state_dict())
        batch = exp.synthetic_batch(1, 5)
        expected = reference(batch["x"], tf=batch["tf"])["waveform"]
        actual = model(batch["x"], tf=batch["tf"])["waveform"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_native_parameter_counts_and_module_removals():
    base = PatchAporModel("M0", SEEDS[0])
    assert sum(p.numel() for p in base.parameters()) == 1031690
    state = base.state_dict()
    for arm in ARMS:
        model = PatchAporModel(arm, SEEDS[0])
        optimizer, partition = build_crd_optimizer(model, exp.config(arm, SEEDS[0], Path("/tmp/unused")))
        assert set(partition.decay_names) | set(partition.no_decay_names) == set(dict(model.named_parameters()))
        for name, value in model.state_dict().items():
            if name in state and state[name].shape == value.shape:
                assert torch.equal(state[name], value), (arm, name)
        assert not any("temporal" in name or "refinement" in name for name in model.state_dict())
        if arm == "M1":
            assert not model.branches
        if arm == "M2":
            assert not model.base.local_blocks
        if arm == "M3":
            assert model.branches["w"].conv_in.kernel_size == (1, 3)
            assert model.branches["w"].depthwise.kernel_size == (1, 3)
        if arm == "M4":
            assert isinstance(model.branches["w"].parameter_fill, nn.Identity)
        if arm == "M5" or arm.startswith("A"):
            assert model.base.decoder_residual is None
        if arm in {"M6", "M7"}:
            assert model.branches["w"].final_projection.out_channels == 96
        del optimizer


@pytest.mark.parametrize("arm", ARMS)
def test_synthetic_forward_training_and_roundtrip(arm):
    cfg = exp.config(arm, SEEDS[0], Path("/tmp/unused"))
    model = replace_trunk(PatchAporModel(arm, SEEDS[0]))
    batch = exp.synthetic_batch(1, 14)
    tf = batch["tf"] if arm != "M1" else None
    optimizer, _ = build_crd_optimizer(model, cfg)
    loss_fn = RespirationTaskLoss(cfg)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["x"], tf=tf)
        assert output["waveform"].shape == (1, 1, 18000)
        loss, _ = loss_fn(output, batch["target"])
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    groups = {name.rsplit(".", 1)[0] for name in initial}
    for group in groups:
        assert any(name.rsplit(".", 1)[0] == group and p.grad.ne(0).any() and not torch.equal(p, initial[name])
                   for name, p in model.named_parameters()), group
    clone = replace_trunk(PatchAporModel(arm, SEEDS[0])).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch["x"], tf=tf)["waveform"], clone(batch["x"], tf=tf)["waveform"], rtol=0, atol=0)


def test_coordinates_affine_interpolation_clamping_and_gradient():
    positions = torch.tensor([[0., 24.5, 49.5], [74.5, 125., 18000.]])
    sampler = CoordinateSample(360, 24.5, 50., positions)
    value = (torch.arange(360) * 50 + 24.5).view(1, 1, -1).requires_grad_()
    result = sampler(value)
    torch.testing.assert_close(result[0, 0], positions.clamp(24.5, 17974.5))
    result.sum().backward()
    assert torch.isfinite(value.grad).all() and value.grad.sum() == 6


@pytest.mark.parametrize("window", ["hann", "uniform"])
def test_overlap_reconstructs_waveforms_including_boundaries(window):
    t = torch.arange(18000) / 100
    signals = torch.stack([
        torch.ones_like(t), torch.sin(2 * torch.pi * .7 * t),
        (1 + .7 * torch.sin(.03 * t)) * torch.sin(2 * torch.pi * (.05 * t + .0018 * t.square())),
    ])
    patches = F_pad(signals, 48).unfold(-1, 256, 128).contiguous().requires_grad_()
    reconstructed = OverlapAdd(window)(patches)[:, 0]
    torch.testing.assert_close(reconstructed, signals, atol=3e-7, rtol=1e-6)
    reconstructed.sum().backward()
    assert torch.isfinite(patches.grad).all() and patches.grad.abs().sum() > 0


def F_pad(value, right):
    return torch.nn.functional.pad(value, (0, right))


def test_overlap_matches_loop_reference():
    patches = torch.randn(2, 140, 256, requires_grad=True)
    module = OverlapAdd("hann")
    numerator, denominator = torch.zeros(2, 18048), torch.zeros(18048)
    for j in range(140):
        numerator[:, j * 128:j * 128 + 256] += patches[:, j] * module.weight
        denominator[j * 128:j * 128 + 256] += module.weight
    torch.testing.assert_close(module(patches)[:, 0], (numerator / denominator)[:, :18000])


@pytest.mark.parametrize("arm", ["M0", "M1", "A0"])
def test_bad_inputs_fail_explicitly(arm):
    model = replace_trunk(PatchAporModel(arm, SEEDS[0]))
    batch = exp.synthetic_batch(1, 15)
    tf = None if arm == "M1" else batch["tf"]
    with pytest.raises(ValueError):
        model(batch["x"][..., :-1], tf=tf)
    batch["x"][..., 10] = float("nan")
    with pytest.raises(FloatingPointError):
        model(batch["x"], tf=tf)
    batch = exp.synthetic_batch(1, 15)
    if arm == "M1":
        with pytest.raises(ValueError):
            model(batch["x"], tf=batch["tf"])
    else:
        batch["tf"]["w"][..., 0] = float("inf")
        with pytest.raises(FloatingPointError):
            model(batch["x"], tf=batch["tf"])


def test_artifact_lifecycle_failure_and_tamper(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    exp.write_json(session / "session.json", {"fixture": True})
    parent = session / "formal/M0/seed_1"
    with pytest.raises(RuntimeError, match="fixture failure"):
        with exp.attempt(parent, session, "formal", arm="M0", seed=1) as failed:
            raise RuntimeError("fixture failure")
    assert (failed / "failed.json").exists()
    with pytest.raises(RuntimeError, match="retry-failed"):
        exp.completed(parent, session, "formal", arm="M0", seed=1)
    assert exp.completed(parent, session, "formal", retry_failed=True, arm="M0", seed=1) is None
    with exp.attempt(parent, session, "formal", arm="M0", seed=1) as successful:
        exp.write_json(successful / "result.json", {"value": 3})
    assert exp.completed(parent, session, "formal", arm="M0", seed=1) == successful
    with pytest.raises(FileExistsError):
        exp.write_json(successful / "result.json", {})
    (successful / "result.json").write_text("tampered")
    with pytest.raises(RuntimeError):
        exp.completed(parent, session, "formal", arm="M0", seed=1)


def metric_fixture():
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            for row in range(4):
                item = {"arm": arm, "seed": seed, "dataset_row_id": row, "samp_id": row % 2, "split": "val",
                        "whole_rr_target_eligible": True, "local_rr_target_eligible": True, "joint_target_eligible": True,
                        "joint_prediction_degenerate": False, "envelope_spearman_prediction_degenerate": False}
                for metric in exp.sf.PRIMARY:
                    item[metric] = 1.0 if arm == "M0" else 0.5
                rows.append(item)
    return pd.DataFrame(rows)


def test_summary_complete_pairing_subjects_and_failures(monkeypatch):
    monkeypatch.setitem(exp.COUNTS, "val", 4)
    frame = metric_fixture()
    tables = exp.summary_tables(frame)
    assert len(tables["per_seed"]) == 39 * 5
    assert len(tables["paired_delta"]) == 12 * 3 * 5
    assert len(tables["subject_macro"]) == 39 * 5
    delta = tables["paired_delta"]
    assert delta.loc[delta.arm.eq("A1"), "delta"].eq(0).all()
    assert delta.loc[delta.arm.eq("A0"), "delta"].eq(-.5).all()
    with pytest.raises(ValueError):
        exp.summary_tables(frame.loc[~frame.arm.eq("A4")])
    broken = frame.copy()
    broken.loc[0, exp.sf.PRIMARY[0]] = np.nan
    with pytest.raises(FloatingPointError):
        exp.summary_tables(broken)
    changed_eligibility = frame.copy()
    changed_eligibility.loc[4, "whole_rr_target_eligible"] = False
    changed_eligibility.loc[4, "whole_rr_abs_error_bpm"] = np.nan
    with pytest.raises(ValueError, match="target资格"):
        exp.summary_tables(changed_eligibility)


@pytest.mark.parametrize("arm", ["M0", "M1", "A0"])
def test_native_trainer_synthetic_lifecycle(tmp_path, arm, monkeypatch):
    cfg = exp.config(arm, SEEDS[0], tmp_path / arm)
    cfg.training.epochs = 2
    cfg.training.early_stopping_min_epoch = 1
    cfg.training.batch_size = 2
    cfg.training.use_amp = False

    class SyntheticLoader(list):
        dataset = range(2)

    native_builder = exp.build_model
    monkeypatch.setattr(exp, "build_model", lambda config: replace_trunk(native_builder(config)))

    class SyntheticExperiment(exp.ModuleExperiment):
        def _build_data(self):
            train = exp.synthetic_batch(2, 241, split="train")
            val = exp.synthetic_batch(2, 242, split="val", row_offset=100)
            if arm == "M1":
                train.pop("tf")
                val.pop("tf")
            return SimpleNamespace(train=SimpleNamespace(loader=SyntheticLoader([train])), val=SimpleNamespace(loader=SyntheticLoader([val])),
                                   audit_summary=pd.DataFrame({"synthetic": [True]}))

    initialization = tmp_path / "initialization.json"
    run = SyntheticExperiment(cfg, initialization_path=initialization).train()
    history = pd.read_csv(run / "train_history.csv")
    assert history.epoch.tolist() == [1, 2]
    assert history.optimizer_update.tolist() == [1, 2]
    best = int(history.loc[history.val_local_rr_mae.idxmin(), "epoch"])
    ckpt = torch.load(run / "checkpoint_best_local_rr.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == best
    assert ckpt["extra_state"]["total_updates"] == 2
    metrics = pd.read_csv(run / "metrics.csv")
    assert metrics.arm.eq(arm).all() and len(metrics) == 2
    assert metrics[list(exp.sf.PRIMARY)].notna().all().all()
    assert set(metrics.split) == {"val"}
    # 用原生两epoch产物验证完整收尾器；仅fixture预算常量缩小。
    monkeypatch.setattr(exp, "UPDATES_PER_EPOCH", 1)
    monkeypatch.setattr(exp, "PLANNED_UPDATES", 2)
    monkeypatch.setattr(exp.sf, "UPDATES_PER_EPOCH", 1)
    monkeypatch.setattr(exp.sf, "PLANNED_UPDATES", 2)
    monkeypatch.setattr(exp.sf, "EPOCHS", 2)
    monkeypatch.setattr(exp.sf, "EARLY_STOP_MIN_EPOCH", 1)
    receipt = exp.validate_run(run, cfg, metrics[["dataset_row_id", "samp_id", "split"]], initialization)
    assert receipt["best_epoch"] == best and receipt["validation_rows"] == 2


def test_session_snapshot_and_source_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "OUTPUT_ROOT", tmp_path)
    session = exp.prepare()
    exp.load_session(session)
    payload = exp.read_json(session / "session.json")
    assert len(payload["templates"]) == 39
    snapshot = session / "source_snapshot" / exp.SPEC
    snapshot.write_text(snapshot.read_text() + "# changed\n")
    with pytest.raises(RuntimeError):
        exp.load_session(session)


def test_gpu_receipt_requires_full_matrix(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    exp.write_json(session / "session.json", {"fixture": True})
    with exp.attempt(session / "engineering", session, "engineering") as output:
        exp.write_json(output / "acceptance.json", {"passed": True, "cells": []})
    with pytest.raises(ValueError, match="GPU验收"):
        exp.verify_gpu(output, session)


def test_execution_stops_before_training_on_engineering_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "load_session", lambda session: {})
    calls = []

    def failed_gpu(*args):
        raise RuntimeError("synthetic GPU failure")

    monkeypatch.setattr(exp, "run_gpu", failed_gpu)
    monkeypatch.setattr(exp, "run_formal", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="synthetic GPU failure"):
        exp.execute(tmp_path, "cuda:0", phase="run-all")
    assert calls == []


def test_parallel_partition_is_complete_disjoint_and_balanced():
    layout = exp.parallel_layout(["cuda:0", "cuda:1"])
    first, second = [worker["cells"] for worker in layout["workers"]]
    assert (len(first), len(second)) == (20, 19)
    ids = lambda cells: {(cell["arm"], cell["seed"]) for cell in cells}
    assert not ids(first) & ids(second)
    assert ids(first) | ids(second) == ids(exp.plan())
    for arm in ARMS:
        assert sum(cell["arm"] == arm for cell in first) in (1, 2)
    for devices in (["cuda:0", "cuda:0"], ["cuda:0"], ["cpu", "cuda:0"]):
        with pytest.raises(ValueError):
            exp.parallel_layout(devices)
    with pytest.raises(ValueError):
        exp.shard_plan(2, 2)


def test_session_shared_locks_exclude_serial_and_duplicate_workers(tmp_path):
    with exp.session_mutex(tmp_path, shared=True):
        with exp.session_mutex(tmp_path, shared=True):
            with pytest.raises(RuntimeError, match="执行锁"):
                with exp.session_mutex(tmp_path):
                    pytest.fail("共享worker期间不应启动串行调度")
        with exp.file_mutex(tmp_path / ".shard_0.lock"):
            with pytest.raises(RuntimeError, match="执行锁"):
                with exp.file_mutex(tmp_path / ".shard_0.lock"):
                    pytest.fail("重复worker获得执行权")
    with exp.session_mutex(tmp_path):
        with pytest.raises(RuntimeError):
            with exp.session_mutex(tmp_path, shared=True):
                pytest.fail("汇总期间不应启动worker")


def test_worker_executes_only_registered_cells(tmp_path, monkeypatch):
    exp.write_json(tmp_path / "session.json", {"fixture": True})
    layout = exp.parallel_layout(["cuda:0", "cuda:1"])
    exp.write_json(tmp_path / "parallel_plan.json", {**layout, "session_sha256": exp.sf.sha256_file(tmp_path / "session.json")})
    monkeypatch.setattr(exp, "load_session", lambda session: {})
    monkeypatch.setattr(exp, "run_gpu", lambda *args: tmp_path / "gpu")
    cells = []
    monkeypatch.setattr(exp, "run_formal", lambda session, arm, seed, device, *args: cells.append((arm, seed, device)))
    result = exp.execute_shard(tmp_path, "cuda:1", 1, 2)
    assert result["completed_cells"] == 19
    assert cells == [(c["arm"], c["seed"], "cuda:1") for c in exp.shard_plan(1, 2)]
    with pytest.raises(ValueError, match="worker分片"):
        exp.execute_shard(tmp_path, "cuda:0", 1, 2)


@pytest.mark.parametrize("exit_codes", [(0, 0), (0, 1)])
def test_parallel_controller_waits_for_both_and_gates_summary(tmp_path, monkeypatch, exit_codes):
    exp.write_json(tmp_path / "session.json", {"fixture": True})
    monkeypatch.setattr(exp, "load_session", lambda session: {})
    monkeypatch.setattr(exp, "cuda_environment", lambda device: {"device_uuid": device})
    calls, summaries = [], []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.returncode = exit_codes[len(calls)]
            self.pid = 10000 + len(calls)
            calls.append((command, kwargs))

        def poll(self):
            return self.returncode

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(exp.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(exp, "summarize", lambda *args: summaries.append(args) or tmp_path / "summary")
    if exit_codes == (0, 0):
        assert exp.run_parallel(tmp_path, ["cuda:0", "cuda:1"]) == tmp_path / "summary"
        assert len(summaries) == 1
    else:
        with pytest.raises(RuntimeError, match="worker失败"):
            exp.run_parallel(tmp_path, ["cuda:0", "cuda:1"])
        assert summaries == []
        assert list((tmp_path / "dispatch").rglob("failed.json"))
    assert len(calls) == 2
    for index, (command, kwargs) in enumerate(calls):
        assert command[command.index("--shard-index") + 1] == str(index)
        assert command[command.index("--device") + 1] == f"cuda:{index}"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "4"
