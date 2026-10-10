"""原始 M4 消融的合成 CPU 合同、梯度、来源与汇总测试。"""
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from scripts import m4_components_model as mod
from scripts import m4_components_runtime as run
from scripts import p1_components_model as p1mod
from test_p1_components import cache_fixture, FixtureMamba, frequencies, cpu_only
from resp_train.crd.training import build_crd_optimizer
from resp_train.data.factory import ThoDataBundle, WindowDataBundle
from resp_train.data.research_v2 import ResearchV2WindowDataset
from resp_train.losses.task import RespirationTaskLoss
from torch.utils.data import DataLoader


def small_architecture():
    return replace(mod.architecture(), dimension=32, waveform_channels=16, condition_channels=8, layers=1)


def model(arm, seed=run.SEEDS[0]):
    return mod.M4ComponentsModel(arm, seed, None if arm == "B1" else frequencies(),
        cfg=small_architecture(), mamba_factory=FixtureMamba)


def inputs(arm):
    x = torch.randn(1, 1, 18000)
    tf = None if arm == "B1" else {"w": torch.rand(1, len(mod.band_indices(frequencies(), mod.ARMS[arm].band)), 360)}
    return x, tf


def open_fusion(net):
    if net.fusion is not None:
        with torch.no_grad():
            net.fusion.projection.weight.normal_(std=.02)


def test_matrix_and_original_scientific_contract(tmp_path):
    assert len(run.plan()) == 18 and len(run.plan(include_references=True)) == 24
    left, right = run.plan(0, 2), run.plan(1, 2)
    assert len(left) == len(right) == 9
    assert {c["cell"] for c in left}.isdisjoint(c["cell"] for c in right)
    assert {c["cell"] for c in left + right} == {c["cell"] for c in run.plan()}
    for c in run.plan(include_references=True):
        cfg = run.config(c["arm"], c["seed"], tmp_path)
        old = run.p1.config("M1" if c["arm"] == "B1" else "M4", c["seed"], tmp_path)
        for key in ("data", "window", "training", "loss", "evaluation"):
            assert cfg[key] == old[key]
        assert cfg.model.architecture == old.model.architecture
        assert dict(cfg.model.m4_components) == mod.contract(c["arm"])
    for arm in mod.REFERENCES:
        with pytest.raises(ValueError, match="冻结参照"):
            run.run_cell(tmp_path, arm, run.SEEDS[0], "cpu")


@pytest.mark.parametrize("seed", run.SEEDS)
def test_original_references_and_common_initialization_are_exact(seed):
    for arm, old_arm in mod.REFERENCES.items():
        old = p1mod.P1ComponentsModel(old_arm, seed, None if arm == "B1" else frequencies(),
            cfg=small_architecture(), mamba_factory=FixtureMamba)
        candidate = model(arm, seed)
        assert old.state_dict().keys() == candidate.state_dict().keys()
        for key, value in old.state_dict().items():
            torch.testing.assert_close(candidate.state_dict()[key], value, rtol=0, atol=0)
        x, tf = inputs(arm)
        old.eval(); candidate.eval()
        open_fusion(old)
        candidate.load_state_dict(old.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(old(x, tf=tf)["waveform"], candidate(x, tf=tf)["waveform"], rtol=0, atol=0)
    full = dict(model("B0", seed).named_parameters())
    for arm in mod.ARMS:
        before = torch.get_rng_state().clone()
        net = model(arm, seed)
        assert torch.equal(before, torch.get_rng_state())
        for name, parameter in net.named_parameters():
            if name in full:
                torch.testing.assert_close(parameter, full[name], rtol=0, atol=0, msg=f"{arm}:{name}")


def test_initial_condition_outputs_equal_no_tf():
    x, _ = inputs("B1")
    with torch.no_grad():
        expected = model("B1").eval()(x)["waveform"]
        for arm in ("B0", "B2", "B3", "B4", "B5"):
            _, tf = inputs(arm)
            torch.testing.assert_close(model(arm).eval()(x, tf=tf)["waveform"], expected, rtol=0, atol=0)


def test_content_bias_removal_keeps_geometry_qkv_and_two_projections():
    full, content = model("B0").condition, model("B3").condition
    assert not list(content.frequency_bias.parameters()) and not list(content.time_bias.parameters())
    with torch.no_grad():
        for block in (full.frequency_bias, full.time_bias):
            block[-1].weight.zero_(); block[-1].bias.zero_()
        tokens, tf = torch.randn(1, 359, 32), torch.rand(1, 97, 360)
        torch.testing.assert_close(full(tokens, tf), content(tokens, tf), rtol=0, atol=0)
    assert isinstance(content.output, nn.Linear)
    torch.testing.assert_close(content.relative_seconds, torch.tensor([-.25, .25]))


def test_mean_tf_and_patch_pool_match_explicit_local_averages():
    net = model("B2")
    assert isinstance(net.condition.projection, nn.Linear) and isinstance(net.fusion.projection, nn.Linear)
    z, tf = torch.randn(1, 359, 32), torch.rand(1, 97, 360)
    with torch.no_grad():
        features = net.condition.encoder(tf[:, None])
        actual = net.condition(z, tf)
        for j in (0, 128, 358):
            expected = net.condition.projection(features[:, :, :, j:j+2].mean((2, 3)))
            torch.testing.assert_close(actual[:, j], expected)
        pool = model("B7").waveform_encoder
        x, _ = inputs("B7")
        windows = pool.stem(x).unfold(-1, 100, 50).permute(0, 2, 3, 1)
        position = pool.position(pool.relative_seconds[:, None])
        torch.testing.assert_close(pool(x), pool.project((windows + position).mean(2)))


def test_inactive_modules_are_not_built(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "frequency_metadata", lambda _: pytest.fail("NoTF 不读取 cache"))
    cfg = run.config("B1", run.SEEDS[0], tmp_path)
    cfg.model.architecture = asdict(small_architecture())
    net = mod.build_model(cfg, mamba_factory=FixtureMamba)
    assert net.condition is None and net.fusion is None
    def forbidden(**kwargs):
        pytest.fail("NoMamba 不构造主干")
    net = mod.M4ComponentsModel("B6", run.SEEDS[0], frequencies(), cfg=small_architecture(), mamba_factory=forbidden)
    assert isinstance(net.trunk, nn.Identity)


@pytest.mark.parametrize("arm", list(mod.ARMS))
@pytest.mark.parametrize("bf16", [False, True])
def test_complete_finite_gradients(arm, bf16):
    net = model(arm)
    open_fusion(net)
    x, tf = inputs(arm)
    if tf is not None:
        tf["w"].requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        output = net(x, tf=tf)["waveform"]
    assert output.shape == (1, 1, 18000) and torch.isfinite(output).all()
    output.square().mean().backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    if tf is not None:
        assert tf["w"].grad.abs().sum() > 0


@pytest.mark.parametrize("arm", list(mod.TRAIN_ARMS))
def test_two_update_task_loss_smoke_opens_condition_branch(arm, tmp_path, monkeypatch):
    session = tmp_path / arm
    session.mkdir()
    run.write_json(session / "session.json", {"fixture": True})
    monkeypatch.setattr(run, "load_session", lambda _: {})
    monkeypatch.setattr(run, "architecture", small_architecture)
    monkeypatch.setattr(run, "cwt_magnitude_features", lambda _: (np.random.default_rng(42).random((97, 360), dtype=np.float32), frequencies()))
    output = run.smoke(session, arm, "cpu", mamba_factory=FixtureMamba, batch_size=1)
    receipt = run.read_json(output / "receipt.json")
    assert receipt["passed"] and receipt["updates"] == 2 and receipt["accumulation"] == 4
    assert receipt["mamba"] == ("identity" if arm == "B6" else "fixture")


@pytest.mark.parametrize("arm", ["B1", "B2", "B4", "B5"])
def test_disposable_cache_to_task_loss(arm, cache_fixture, tmp_path, monkeypatch):
    cfg = run.config(arm, run.SEEDS[0], tmp_path / "out")
    cfg.model.architecture = asdict(small_architecture())
    cfg.training.batch_size = 1
    if arm != "B1":
        cfg.data.tf_cache_path = str(cache_fixture)
    else:
        monkeypatch.setattr(run, "frequency_metadata", lambda _: pytest.fail("NoTF 不读取元数据"))
    t = np.arange(36000, dtype=np.float32) / 100
    np.savez(tmp_path / "signals.npz", bcg=np.sin(2*np.pi*.25*t), tho=np.sin(2*np.pi*.25*t+.1))
    rows = pd.DataFrame([{"dataset_row_id": rid, "split": "train", "samp_id": 1, "coupling_state_id": i,
        "window_start_sample": i*18000, "window_end_sample": (i+1)*18000, "bcg_signal_key": "bcg",
        "target_signal_key": "tho", "source_npz": "signals.npz", "target_source_npz": "signals.npz",
        "allowed_losses": "waveform"} for i, rid in enumerate([11, 29])])
    dataset = ResearchV2WindowDataset(tmp_path / "index.csv", rows, cfg)
    bundle = WindowDataBundle(tmp_path / "index.csv", rows, dataset, DataLoader(dataset, batch_size=1), rows, pd.DataFrame())
    data = run.condition_data(ThoDataBundle(bundle, bundle, rows, pd.DataFrame()), cfg)
    batch = next(iter(data.train.loader))
    assert ("tf" in batch) == (arm != "B1")
    if arm != "B1":
        assert batch["tf"]["w"].shape[1] == {"B2": 97, "B4": 41, "B5": 56}[arm]
    net = mod.build_model(cfg, mamba_factory=FixtureMamba)
    optimizer, _ = build_crd_optimizer(net, cfg)
    summary, updates = run.legacy._train_epoch(net, list(data.train.loader)*4, RespirationTaskLoss(cfg), optimizer, cfg, 0, 6400, 1)
    assert updates == 2 and np.isfinite(summary["loss"])


def metrics_fixture():
    records = []
    errors = {"B0": 1., "B3": 2., "B2": 3., "B1": 4.}
    for cell in run.plan(include_references=True):
        for row_id, subject in [(10, 101), (20, 102)]:
            r = {"arm": cell["arm"], "seed": cell["seed"], "dataset_row_id": row_id, "samp_id": subject,
                "split": "val", "whole_rr_target_eligible": True, "local_rr_target_eligible": True,
                "local_rr_target_eligible_windows": 9, "joint_target_eligible": True,
                "envelope_spearman_target_eligible": True, "joint_prediction_degenerate": False,
                "envelope_spearman_prediction_degenerate": False}
            error = errors.get(cell["arm"], 5.)
            r.update({m: 1-.1*error if m == run.sf.PCC else error for m in run.sf.PRIMARY})
            records.append(r)
    return pd.DataFrame(records)


def test_summary_complete_matrix_increment_sign_and_finite_checks():
    frame = metrics_fixture()
    tables = run.summary_tables(frame)
    assert len(tables["per_seed"]) == 120 and len(tables["paired_delta"]) == 105
    inc = tables["incremental_delta"]
    assert len(inc) == 45
    assert tables["incremental_summary"].improved_seeds.eq(3).all()
    assert tables["incremental_summary"].delta_sd.eq(0).all()
    np.testing.assert_allclose(inc.loc[inc.metric != run.sf.PCC, "improvement"], 1.)
    np.testing.assert_allclose(inc.loc[inc.metric == run.sf.PCC, "improvement"], .1)
    with pytest.raises(ValueError, match="完整"):
        run.summary_tables(frame.iloc[:-2])
    broken = frame.copy()
    broken.loc[0, run.sf.PRIMARY[0]] = np.nan
    with pytest.raises((ValueError, FloatingPointError)):
        run.summary_tables(broken)
    broken = frame.copy()
    broken.loc[0, "dataset_row_id"] = 999
    with pytest.raises(ValueError):
        run.summary_tables(broken)


def test_band_adapter_checks_full_source_before_cropping():
    full = torch.ones(97, 360)
    full[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        run.BandDataset([{"tf": {"w": full}}], mod.band_indices(frequencies(), "H"))[0]


def make_cell(directory, arm, seed, frame, cfg, parameters=1):
    directory.mkdir(parents=True)
    OmegaConf.save(cfg, directory / "config.yaml")
    (directory / "checkpoint.pt").write_bytes(b"synthetic checkpoint, never deserialized")
    frame.to_csv(directory / "validation_metrics.csv", index=False)
    run.write_json(directory / "environment.json", {"torch": "fixture", "cuda": None, "packages": {}, "gpu": None})
    run.write_json(directory / "completed.json", {"arm": arm, "seed": seed, "parameters": parameters,
        "metrics": "validation_metrics.csv", "best_checkpoint": "checkpoint.pt", "final_checkpoint": "checkpoint.pt",
        "artifacts": {n: run.sha(directory / n) for n in ("config.yaml", "checkpoint.pt", "validation_metrics.csv")}})


@pytest.fixture
def references_fixture(tmp_path, monkeypatch):
    source = tmp_path / "p1"
    source.mkdir()
    run.write_json(source / "session.json", {"synthetic": True})
    monkeypatch.setattr(run, "REFERENCE_SESSION", source)
    monkeypatch.setattr(run, "REFERENCE_SHA", run.sha(source / "session.json"))
    spec = OmegaConf.load(run.SPEC)
    spec.reference_session_sha256 = run.REFERENCE_SHA
    fixture_spec = tmp_path / "spec.yaml"
    OmegaConf.save(spec, fixture_spec)
    monkeypatch.setattr(run, "SPEC", fixture_spec)
    monkeypatch.setattr(run.p1, "load_session", lambda _: {})
    # 快照用真实模型源码的身份；测试依赖均为临时产物。
    source_file = "scripts/m4_components_model.py"
    monkeypatch.setattr(run, "code_identity", lambda: {source_file: run.sha(run.ROOT / source_file)})
    frames = metrics_fixture()
    for arm, old_arm in mod.REFERENCES.items():
        for seed in run.SEEDS:
            directory = source / "cells" / f"{old_arm}_seed{seed}"
            cfg = run.p1.config(old_arm, seed, directory)
            frame = frames.query("arm == @arm and seed == @seed").copy()
            frame["arm"] = old_arm
            make_cell(directory, old_arm, seed, frame, cfg, {"B0": 1056357, "B1": 1020125}[arm])
    return source


def test_references_audit_and_tamper_rejection(references_fixture):
    result = run.audit_references()
    assert len(result["cells"]) == 6
    path = Path(result["cells"][f"B0_seed{run.SEEDS[0]}"]["path"]) / "checkpoint.pt"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="身份变化"):
        run.audit_references()


def test_reference_scientific_config_rejected_even_with_new_hash(references_fixture):
    directory = references_fixture / "cells" / f"M4_seed{run.SEEDS[0]}"
    cfg = OmegaConf.load(directory / "config.yaml")
    cfg.training.max_learning_rate *= 2
    OmegaConf.save(cfg, directory / "config.yaml")
    receipt = run.read_json(directory / "completed.json")
    receipt["artifacts"]["config.yaml"] = run.sha(directory / "config.yaml")
    (directory / "completed.json").unlink()
    run.write_json(directory / "completed.json", receipt)
    with pytest.raises(ValueError, match="科学配置"):
        run.audit_references()


def test_prepare_status_and_summary_preserve_reference_bytes(references_fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "build_tho_data", lambda _: pytest.fail("准备和汇总不访问真实数据"))
    before = {str(p): run.sha(p) for p in references_fixture.rglob("*") if p.is_file()}
    session = run.prepare(tmp_path / "new")
    assert run.prepare(session) == session
    states = run.status(session)
    assert sum(c["status"] == "reference_verified" for c in states) == 6
    assert len(list((session / "cells").glob("*"))) == 0
    frames = metrics_fixture()
    for c in run.plan():
        arm, seed = c["arm"], c["seed"]
        directory = session / "cells" / c["cell"]
        make_cell(directory, arm, seed, frames.query("arm == @arm and seed == @seed"), run.config(arm, seed, directory))
    original_spec = run.spec
    def small_spec():
        result = original_spec()
        result["data"] = {**result["data"], "val_windows": 2, "val_subjects": 2}
        return result
    # 只缩小汇总分母；session 核验仍使用准备时的完整合同。
    manifest = run.load_session(session)
    monkeypatch.setattr(run, "load_session", lambda _: manifest)
    monkeypatch.setattr(run, "spec", small_spec)
    output = run.summarize(session)
    assert run.summarize(session) == output
    assert len(pd.read_csv(output / "per_seed.csv")) == 120
    assert run.read_json(output / "receipt.json")["sources"][f"B0_seed{run.SEEDS[0]}"]["source_arm"] == "M4"
    assert before == {str(p): run.sha(p) for p in references_fixture.rglob("*") if p.is_file()}


def test_receipt_rejects_escape_paths(tmp_path):
    with pytest.raises(ValueError, match="越过"):
        run.artifact_path(tmp_path, "../foreign")


def test_completed_acceptance_reused_only_for_matching_environment(tmp_path, monkeypatch):
    session = tmp_path / "new"
    session.mkdir()
    run.write_json(session / "session.json", {"fixture": True})
    monkeypatch.setattr(run, "load_session", lambda _: {})
    monkeypatch.setattr(run.legacy, "runtime", lambda _: {"fixture": True})
    output = session / "engineering/B3_cuda_0_fixture"
    output.mkdir(parents=True)
    run.write_json(output / "receipt.json", {"passed": True, "arm": "B3", "contract": mod.contract("B3"),
        "batch_size": 32, "accumulation": 4, "updates": 2, "mamba": "official_mamba2", "environment": {"fixture": True}})
    run.write_json(output / "completed.json", {"receipt_sha256": run.sha(output / "receipt.json"), "session_sha256": run.sha(session / "session.json")})
    monkeypatch.setattr(run, "smoke", lambda *args: pytest.fail("同环境已完成验收应复用"))
    assert run.ensure_smoke(session, "B3", "cuda:0") == output
    (output / "receipt.json").write_text("{}")
    with pytest.raises(ValueError, match="身份变化"):
        run.ensure_smoke(session, "B3", "cuda:0")
