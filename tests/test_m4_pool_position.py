"""相对位置消融的公式、初始化、梯度及六 cell 汇总定向验收。"""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import fcntl
from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from scripts import m4_pool_position_model as mod
from scripts import m4_pool_position_runtime as run
from scripts.p1_components_model import P1ComponentsModel
from test_p1_components import FixtureMamba, frequencies, cpu_only
from test_m4_components import metrics_fixture


def small():
    return replace(mod.architecture(), dimension=32, waveform_channels=16, condition_channels=8, layers=1)


def model(arm, seed=20260811):
    return mod.PoolPositionModel(arm, seed, frequencies(), cfg=small(), mamba_factory=FixtureMamba)


@pytest.mark.parametrize("seed", run.SEEDS)
def test_original_m4_and_pair_exact(seed):
    old = P1ComponentsModel("M4", seed, frequencies(), cfg=small(), mamba_factory=FixtureMamba).eval()
    a, b = model("A", seed).eval(), model("B", seed)
    assert old.state_dict().keys() == a.state_dict().keys()
    for name, value in old.state_dict().items():
        assert torch.equal(value, a.state_dict()[name]), name
    for name, value in b.state_dict().items():
        assert torch.equal(value, a.state_dict()[name]), name
    assert not hasattr(b.waveform_encoder, "position")
    assert torch.count_nonzero(a.fusion.projection.weight) == 0
    assert torch.count_nonzero(a.fusion.projection.bias) == 0
    assert a.condition.output is not a.fusion.projection
    x, tf = torch.randn(1, 1, 18000), {"w": torch.randn(1, 97, 360)}
    # 打开融合后验证完整时频路径，避免零初始化掩盖结构差异。
    with torch.no_grad():
        old.fusion.projection.weight.normal_(std=.02)
        a.load_state_dict(old.state_dict())
        torch.testing.assert_close(a(x, tf=tf)["waveform"], old(x, tf=tf)["waveform"], rtol=0, atol=0)


@pytest.mark.parametrize("seed", run.SEEDS)
def test_constructor_preserves_original_device_seed_calls(seed, monkeypatch):
    # 追踪真实 torch.manual_seed 的设备分发；CPU 环境即可检查 CUDA seed 副作用。
    device_seeds = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", device_seeds.append)
    cpu_before = torch.get_rng_state().clone()
    P1ComponentsModel("M4", seed, frequencies(), cfg=small(), mamba_factory=FixtureMamba)
    expected_calls = list(device_seeds)
    assert expected_calls
    assert torch.equal(torch.get_rng_state(), cpu_before)
    for arm in ("A", "B"):
        device_seeds.clear()
        model(arm, seed)
        assert device_seeds == expected_calls
        assert torch.equal(torch.get_rng_state(), cpu_before)


def test_environment_registration_waits_for_peer(tmp_path):
    environments = [{"torch": "fixture", "cuda": "fixture", "packages": {},
                     "gpu": "fixture", "device": f"cuda:{i}"} for i in (0, 1)]
    started = [Event(), Event()]

    def register(index):
        started[index].set()
        run.register_training_environment(tmp_path, environments[index])

    # 每个调用独立打开文件，实际 flock 与跨进程使用相同的互斥语义。
    with (tmp_path / ".environment.lock").open("a") as owner, ThreadPoolExecutor(max_workers=2) as workers:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX)
        futures = [workers.submit(register, i) for i in (0, 1)]
        try:
            assert all(event.wait(timeout=5) for event in started)
            for future in futures:
                with pytest.raises(TimeoutError):
                    future.result(timeout=.1)
        finally:
            fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
        for future in futures:
            future.result(timeout=5)
    path = tmp_path / "training_environment.json"
    assert run.read_json(path) in environments
    original_bytes = path.read_bytes()
    run.register_training_environment(tmp_path, environments[0])
    assert path.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="GPU 型号"):
        run.register_training_environment(tmp_path, {**environments[0], "gpu": "other"})
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("failure", ["runtime", "session_environment", "cell_environment"])
def test_environment_failures_are_recorded_in_attempt(tmp_path, monkeypatch, failure):
    original_spec = run.spec
    def confirmed_spec():
        value = original_spec()
        value["decision"]["confirmed"] = True
        value["decision"]["local_rr_margin_bpm"] = .01
        return value
    monkeypatch.setattr(run, "spec", confirmed_spec)
    monkeypatch.setattr(run, "load_session", lambda _: {})
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(run, "ensure_smoke", lambda *args: pytest.fail("环境检查失败须立即退出"))
    environment = {"torch": "fixture", "cuda": "fixture", "packages": {}, "gpu": "fixture"}
    monkeypatch.setattr(run.legacy, "runtime", lambda _: environment)
    cell = tmp_path / "cells" / f"A_seed{run.SEEDS[0]}"
    cell.mkdir(parents=True)
    if failure == "runtime":
        def unavailable(_):
            raise RuntimeError("environment unavailable")
        monkeypatch.setattr(run.legacy, "runtime", unavailable)
    else:
        path = (tmp_path / "training_environment.json" if failure == "session_environment"
                else cell / "environment.json")
        run.write_json(path, {**environment, "gpu": "other"})
        before = path.read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        run.run_cell(tmp_path, "A", run.SEEDS[0], "cuda:0")
    markers = list(cell.glob("attempts/*/failed.json"))
    assert len(markers) == 1
    assert run.read_json(markers[0])["type"] == ("RuntimeError" if failure == "runtime" else "ValueError")
    assert not (cell / "completed.json").exists()
    if failure != "runtime":
        assert path.read_bytes() == before


@pytest.mark.parametrize("arm", list(mod.ALL_ARMS))
def test_formula_and_gradients_against_explicit_positioned_tensor(arm):
    pool = model(arm).waveform_encoder
    x = torch.randn(1, 1, 18000)
    features = pool.stem(x).unfold(-1, 100, 50).permute(0, 2, 3, 1)
    p = pool.position(pool.relative_seconds[:, None]) if hasattr(pool, "position") else 0
    spec = mod.ALL_ARMS[arm]
    weights = pool.score(features + spec.score_position * p).squeeze(-1).softmax(-1)
    expected = pool.project((weights[..., None] * (features + spec.value_position * p)).sum(2))
    actual = pool(x)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    parameters = tuple(pool.parameters())
    actual_grad = torch.autograd.grad(actual.square().mean(), parameters)
    expected_grad = torch.autograd.grad(expected.square().mean(), parameters)
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, rtol=1e-3, atol=2e-6)


@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize("bf16", [False, True])
def test_finite_complete_backward(arm, bf16):
    net = model(arm)
    with torch.no_grad():
        net.fusion.projection.weight.normal_(std=.02)
    tf = torch.randn(1, 97, 360, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        output = net(torch.randn(1, 1, 18000), tf={"w": tf})["waveform"]
    output.square().mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert tf.grad.abs().sum() > 0
    with pytest.raises(FloatingPointError):
        net(torch.full((1, 1, 18000), float("nan")), tf={"w": tf})


def test_matrix_contract_and_run_gate(tmp_path, monkeypatch):
    assert len(run.plan()) == 6
    assert set(c["arm"] for c in run.plan()) == {"A", "B"}
    assert len(run.plan(0, 2)) == len(run.plan(1, 2)) == 3
    for c in run.plan():
        cfg = run.config(c["arm"], c["seed"], tmp_path)
        old = run.p1.config("M4", c["seed"], tmp_path)
        for key in ("data", "window", "training", "loss", "evaluation"):
            assert cfg[key] == old[key]
    # 描述性 session 可进入运行入口；CUDA 与 session 完整性检查继续生效。
    monkeypatch.setattr(run, "load_session", lambda _: {})
    with pytest.raises(ValueError, match="CUDA"):
        run.run_cell(tmp_path, "A", run.SEEDS[0], "cpu")
    with pytest.raises(ValueError):
        run.config("C", run.SEEDS[0], tmp_path)
    before = torch.get_rng_state().clone()
    report = run.audit_initialization(cfg=small(), mamba_factory=FixtureMamba)
    assert torch.equal(before, torch.get_rng_state())
    for value in report.values():
        assert value["original_m4_exact"] and value["common_exact"]
        assert value["A"]["parameters"] - value["B"]["parameters"] == value["A"]["position_parameters"]
        assert value["B"]["position_parameters"] == 0


def fixture_frame():
    frame = metrics_fixture().query("arm in ['B0', 'B1']").copy()
    frame["arm"] = frame.arm.map({"B0": "A", "B1": "B"})
    return frame


def test_summary_pairs_denominators_and_failure_states():
    frame = fixture_frame()
    tables = run.summary_tables(frame)
    assert len(tables["per_seed"]) == 30 and len(tables["paired_delta"]) == 15
    assert tables["paired_summary"].worsened_seeds.eq(3).all()
    subject = tables["subject_paired_delta"]
    assert subject.loc[subject.metric != run.sf.PCC, "delta_B_minus_A"].eq(3).all()
    with pytest.raises(ValueError, match="完整"):
        run.summary_tables(frame.iloc[:-2])
    bad = frame.copy()
    bad.loc[bad.index[0], run.sf.PRIMARY[0]] = np.inf
    with pytest.raises((ValueError, FloatingPointError)):
        run.summary_tables(bad)
    bad = frame.copy()
    bad.loc[bad.index[0], "dataset_row_id"] = 99
    with pytest.raises(ValueError):
        run.summary_tables(bad)


@pytest.mark.parametrize("arm", ["A", "B"])
def test_two_updates_with_formal_task_loss(arm, tmp_path, monkeypatch):
    run.write_json(tmp_path / "session.json", {"fixture": True})
    monkeypatch.setattr(run, "load_session", lambda _: {})
    monkeypatch.setattr(run, "architecture", small)
    monkeypatch.setattr(run, "cwt_magnitude_features", lambda _: (np.ones((97, 360), dtype=np.float32), frequencies()))
    output = run.smoke(tmp_path, arm, "cpu", mamba_factory=FixtureMamba, batch_size=1)
    receipt = run.read_json(output / "receipt.json")
    assert receipt["passed"] and receipt["updates"] == 2 and receipt["mamba"] == "fixture"


def test_prepare_freezes_configs_and_refuses_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "audit_initialization", lambda: {"fixture": True})
    monkeypatch.setattr(run, "build_tho_data", lambda _: pytest.fail("prepare 不得读取真实数据"))
    session = run.prepare(tmp_path / "session")
    assert run.prepare(session) == session
    manifest = run.load_session(session)
    assert len(manifest["resolved_configs"]) == 6
    assert all(c["status"] == "pending" for c in run.status(session))
    config_path = session / next(iter(manifest["resolved_configs"]))
    config_path.write_text("changed fixture")
    with pytest.raises(ValueError, match="config"):
        run.load_session(session)


def test_fixed_case_rule_and_full_summary(tmp_path, monkeypatch):
    frame = fixture_frame()
    rows = pd.DataFrame({"dataset_row_id": [20, 10, 30], "samp_id": [102, 101, 101]})
    waves = np.sin(np.arange(18000, dtype=np.float32) / 100)[None, None].repeat(3, axis=0)
    bundle = SimpleNamespace(rows=rows, dataset=[{"x": torch.from_numpy(wave)} for wave in waves])
    predictions = {"r_tho_hat": waves, "tho_ref": waves, "dataset_row_id": rows.dataset_row_id.to_numpy(),
                   "samp_id": rows.samp_id.to_numpy()}
    sample = tmp_path / "cases.npz"
    run.save_cases(sample, bundle, predictions)
    with np.load(sample) as archive:
        assert archive["dataset_row_id"].tolist() == [10, 20]
    original = run.spec
    monkeypatch.setattr(run, "load_session", lambda _: {})
    def small_spec():
        spec = original()
        spec["data"].update(val_windows=2, val_subjects=2)
        return spec
    monkeypatch.setattr(run, "spec", small_spec)
    for cell in run.plan():
        directory = tmp_path / "cells" / cell["cell"]
        directory.mkdir(parents=True)
        cfg = run.config(cell["arm"], cell["seed"], directory)
        OmegaConf.save(cfg, directory / "config.yaml")
        (directory / "checkpoint.pt").write_bytes(b"synthetic checkpoint")
        (directory / "cases.npz").write_bytes(sample.read_bytes())
        frame.loc[frame.arm.eq(cell["arm"]) & frame.seed.eq(cell["seed"])].to_csv(directory / "validation_metrics.csv", index=False)
        run.write_json(directory / "completed.json", {"arm": cell["arm"], "seed": cell["seed"], "parameters": 1,
            "best_epoch": 2, "metrics": "validation_metrics.csv", "best_checkpoint": "checkpoint.pt", "final_checkpoint": "checkpoint.pt",
            "artifacts": {name: run.sha(directory / name) for name in ("config.yaml", "checkpoint.pt", "cases.npz", "validation_metrics.csv")}})
    output = run.summarize(tmp_path)
    assert len(list(output.glob("*.png"))) == 6
    decision = pd.read_csv(output / "decision.csv")
    assert decision.stage2_discussion_gate.isna().all()
    assert decision.margin_bpm.isna().all()
    assert not decision.threshold_confirmed.any()
    assert len(pd.read_csv(output / "selection.csv")) == 6
    assert run.summarize(tmp_path) == output
    (output / "selection.csv").write_text("changed")
    with pytest.raises(ValueError, match="身份变化"):
        run.summarize(tmp_path)


@pytest.mark.parametrize("margin,confirmed,expected", [(None, False, None), (.01, False, None), (.01, True, True), (4., True, False)])
def test_decision_reports_only_confirmed_thresholds(margin, confirmed, expected):
    paired = run.summary_tables(fixture_frame())["paired_delta"]
    decision = run.decision_summary(paired, {"local_rr_margin_bpm": margin, "confirmed": confirmed})
    assert decision.gain_A_over_B_bpm.iloc[0] == 3.
    if expected is None:
        assert decision.stage2_discussion_gate.isna().all()
    else:
        assert bool(decision.stage2_discussion_gate.iloc[0]) == expected
