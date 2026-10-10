"""APOR新对照：只使用CPU合成输入，Mamba算子由GPU验收单独覆盖。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from scripts import apor_gelu_refiner_v1_runtime as exp
from scripts.apor_gelu_refiner_v1_model import ARMS, TRAIN_ARMS, SEEDS, PARAMETERS, GeluRefinerModel
from resp_train.paper_evidence.patch_apor_v1_model import PatchAporModel
from resp_train.crd.training import build_crd_optimizer
from resp_train.losses.task import RespirationTaskLoss


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


@pytest.mark.parametrize("seed", SEEDS)
def test_parameter_changes_and_exact_common_initialization(seed):
    original = PatchAporModel("A0", seed).state_dict()
    prefix = "branches.w.parameter_fill."
    for arm in ARMS:
        model = GeluRefinerModel(arm, seed)
        current = model.state_dict()
        assert sum(p.numel() for p in model.parameters()) == PARAMETERS[arm]
        expected_keys = {k for k in original if arm != "DIRECT" or not k.startswith(prefix)}
        assert set(current) == expected_keys
        for name, value in current.items():
            expected = original[name]
            if arm == "H64" and name == prefix + "expand.weight":
                expected = expected[:64]
            elif arm == "H64" and name == prefix + "project.weight":
                expected = expected[:, :64]
            assert torch.equal(value, expected), (arm, name)
        assert model.branches["w"].local_projection.in_channels == 480
        assert model.branches["w"].sample.output_shape == (140, 5)


def test_baseline_exact_forward_and_backward_with_nonzero_condition():
    original = replace_trunk(PatchAporModel("A0", SEEDS[0])).eval()
    baseline = replace_trunk(GeluRefinerModel("B0", SEEDS[0])).eval()
    with torch.no_grad():
        original.branches["w"].final_projection.weight.normal_(std=.01)
    baseline.load_state_dict(original.state_dict())
    batch = exp.synthetic_batch(1, 32)
    outputs, grads = [], []
    for model in (original, baseline):
        output = model(batch["x"], tf=batch["tf"])["waveform"]
        output.square().mean().backward()
        outputs.append(output)
        grads.append({n: p.grad for n, p in model.named_parameters()})
    assert torch.equal(*outputs)
    assert all(torch.equal(grads[0][n], grads[1][n]) for n in grads[0])


def test_all_five_external_silu_sites_are_changed(monkeypatch):
    model = replace_trunk(GeluRefinerModel("GELU", SEEDS[0])).eval()
    batch = exp.synthetic_batch(1, 33)
    calls = []
    original_gelu = F.gelu

    def gelu(value, *args, **kwargs):
        calls.append(tuple(value.shape))
        return original_gelu(value, *args, **kwargs)

    def silu(*args, **kwargs):
        raise AssertionError("普通隐藏层中仍执行SiLU")

    monkeypatch.setattr(F, "gelu", gelu)
    monkeypatch.setattr(F, "silu", silu)
    with torch.no_grad():
        output = model(batch["x"], tf=batch["tf"])["waveform"]
    assert len(calls) == 9  # PatchMixer原4处，加上本轮替换的5处。
    assert output.shape == (1, 1, 18000) and torch.isfinite(output).all()


@pytest.mark.parametrize("arm", TRAIN_ARMS)
def test_candidate_active_gradients_and_checkpoint_roundtrip(arm):
    cfg = exp.config(arm, SEEDS[0], Path("SYNTHETIC"))
    model = replace_trunk(GeluRefinerModel(arm, SEEDS[0]))
    optimizer, partition = build_crd_optimizer(model, cfg)
    assert set(partition.decay_names) | set(partition.no_decay_names) == set(dict(model.named_parameters()))
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    batch = exp.synthetic_batch(1, 34)
    loss_fn = RespirationTaskLoss(cfg)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_fn(model(batch["x"], tf=batch["tf"]), batch["target"])
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    named = dict(model.named_parameters())
    for group in {n.rsplit(".", 1)[0] for n in named}:
        assert any(n.rsplit(".", 1)[0] == group and p.grad.ne(0).any() and not torch.equal(initial[n], p)
                   for n, p in named.items()), group
    clone = replace_trunk(GeluRefinerModel(arm, SEEDS[0])).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(batch["x"], tf=batch["tf"])["waveform"], clone(batch["x"], tf=batch["tf"])["waveform"])


def test_contract_matrix_partition_and_source_paths():
    assert exp.load_spec()["research_test"] == "not_enabled"
    assert len(exp.plan()) == 9 and len(exp.scientific_plan()) == 12
    layout = exp.parallel_layout(["cuda:0", "cuda:1"])
    assert [len(w["cells"]) for w in layout["workers"]] == [5, 4]
    cells = [(c["arm"], c["seed"]) for w in layout["workers"] for c in w["cells"]]
    assert len(set(cells)) == 9 and set(cells) == {(a, s) for a in TRAIN_ARMS for s in SEEDS}
    assert all(path.is_file() for path in exp.source_files())
    for arm in ARMS:
        cfg = exp.config(arm, SEEDS[0], Path("SYNTHETIC"))
        old = exp.legacy.config("A0", SEEDS[0], Path("OLD"))
        assert exp.scientific_config(cfg) == exp.scientific_config(old)


def metric_fixture():
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            for index in range(4):
                row = {"arm": arm, "seed": seed, "dataset_row_id": index, "samp_id": 1 if index < 3 else 2,
                       "split": "val", "whole_rr_target_eligible": True, "local_rr_target_eligible": True,
                       "joint_target_eligible": True, "joint_prediction_degenerate": False,
                       "envelope_spearman_prediction_degenerate": False}
                row.update({m: .8 if m == exp.sf.PCC else 1. for m in exp.sf.PRIMARY})
                rows.append(row)
    return pd.DataFrame(rows)


def test_independent_quality_checks_and_complete_matrix(monkeypatch):
    monkeypatch.setitem(exp.COUNTS, "val", 4)
    frame = metric_fixture()
    tables = exp.summary_tables(frame)
    assert len(tables["per_seed"]) == 60 and len(tables["paired_delta"]) == 45
    assert exp.validation_decision(tables)["eligible_arms"] == list(ARMS)
    frame.loc[frame.arm.eq("GELU"), "local_rr_mae_bpm"] = 1.006
    decision = exp.validation_decision(exp.summary_tables(frame))
    assert decision["eligible_arms"] == ["B0", "H64", "DIRECT"]
    assert not decision["test_used"] and not decision["replaces_main_model"]
    frame.loc[frame.arm.eq("H64") & frame.samp_id.eq(1), "local_rr_mae_bpm"] = .97
    frame.loc[frame.arm.eq("H64") & frame.samp_id.eq(2), "local_rr_mae_bpm"] = 1.06
    assert exp.validation_decision(exp.summary_tables(frame))["eligible_arms"] == ["B0", "DIRECT"]
    with pytest.raises(ValueError, match="12-cell"):
        exp.summary_tables(frame[~(frame.arm.eq("DIRECT") & frame.seed.eq(SEEDS[0]))])
    frame.loc[0, "local_rr_mae_bpm"] = np.nan
    with pytest.raises((ValueError, FloatingPointError)):
        exp.summary_tables(frame)


def test_reference_writers_wait_and_preserve_identity(tmp_path, monkeypatch):
    target = np.arange(40, dtype=np.float32).reshape(2, 1, 20)
    contended = Event()
    original_flock = exp.fcntl.flock

    def observe_contention(*args):
        try:
            return original_flock(*args)
        except BlockingIOError:
            contended.set()
            raise

    monkeypatch.setattr(exp.fcntl, "flock", observe_contention)

    def save():
        return exp.save_validation_reference(tmp_path, target)

    with ThreadPoolExecutor(max_workers=2) as pool:
        with exp.file_mutex(tmp_path / ".validation_reference.lock"):
            future = pool.submit(save)
            assert contended.wait(2)  # 确保确实发生过锁冲突，防止调度时序使测试空过。
            assert not future.done()
            other = pool.submit(save)
        first, second = future.result(timeout=3), other.result(timeout=3)
    assert first == second
    np.testing.assert_array_equal(np.load(first), target)
    original = first.read_bytes()
    with pytest.raises(ValueError, match="不一致"):
        exp.save_validation_reference(tmp_path, target + 1)
    assert first.read_bytes() == original
    with exp.file_mutex(tmp_path / ".validation_reference.lock"):
        with pytest.raises(RuntimeError, match="超时"):
            with exp.file_mutex(tmp_path / ".validation_reference.lock", wait_timeout=.01):
                raise AssertionError("同一排他锁被重复获得")


def test_partial_reference_is_not_published(tmp_path, monkeypatch):
    def fail(stream, *args, **kwargs):
        stream.write(b"partial")
        raise OSError("synthetic write failure")

    monkeypatch.setattr(exp.np, "save", fail)
    with pytest.raises(OSError, match="synthetic"):
        exp.save_validation_reference(tmp_path, np.zeros((1, 1, 20), np.float32))
    assert not (tmp_path / "validation_reference.npy").exists()
    assert not list(tmp_path.glob(".validation_reference_*.npy"))


def test_native_trainer_lifecycle(tmp_path, monkeypatch):
    cfg = exp.config("H64", SEEDS[0], tmp_path / "training")
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

    initial = tmp_path / "initialization.json"
    run = SyntheticExperiment(cfg, initialization_path=initial).train()
    frame = pd.read_csv(run / "metrics.csv")
    assert len(frame) == 2 and frame.arm.eq("H64").all()
    assert np.load(run / "validation_prediction.npy").shape == (2, 1, 18000)
    for module in (exp, exp.sf):
        monkeypatch.setattr(module, "UPDATES_PER_EPOCH", 1)
        monkeypatch.setattr(module, "PLANNED_UPDATES", 2)
    monkeypatch.setattr(exp.sf, "EPOCHS", 2)
    monkeypatch.setattr(exp.sf, "EARLY_STOP_MIN_EPOCH", 1)
    assert exp.validate_run(run, cfg, frame[["dataset_row_id", "samp_id", "split"]], initial)["validation_rows"] == 2
    with pytest.raises(ValueError, match="只允许validation"):
        SyntheticExperiment(cfg)._evaluate_model(None, None, include_test_only=True)


@pytest.mark.parametrize("exit_codes", [(0, 0), (0, 1)])
def test_parallel_acceptance_barrier_and_failure_gate(tmp_path, monkeypatch, exit_codes):
    exp.write_json(tmp_path / "session.json", {"fixture": True})
    monkeypatch.setattr(exp, "load_session", lambda p: {})
    monkeypatch.setattr(exp, "cuda_environment", lambda d: {"device_uuid": d})
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    accepted, commands, summaries = [], [], []
    monkeypatch.setattr(exp, "run_gpu", lambda s, d, r: accepted.append(d) or tmp_path / d)
    monkeypatch.setattr(exp, "verify_gpu", lambda *a: None)

    class Process:
        def __init__(self, command, **kwargs):
            assert accepted == ["cuda:0", "cuda:1"]
            self.returncode = exit_codes[len(commands)]
            self.pid = 23400 + len(commands)
            commands.append(command)

        def poll(self):
            return self.returncode

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(exp.subprocess, "Popen", Process)
    monkeypatch.setattr(exp, "summarize", lambda *a: summaries.append(a) or tmp_path / "summary")
    if exit_codes == (0, 0):
        exp.run_parallel(tmp_path, ["cuda:0", "cuda:1"])
        assert len(summaries) == 1
    else:
        with pytest.raises(RuntimeError):
            exp.run_parallel(tmp_path, ["cuda:0", "cuda:1"])
        assert not summaries
