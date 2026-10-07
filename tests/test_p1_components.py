"""P1 三十组矩阵的合成 CPU 结构、数据接口与合同检查。"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from resp_train.crd.config import load_crd_config
from resp_train.crd.training import build_crd_optimizer, optimizer_updates_per_epoch, train_crd_one_epoch
from resp_train.data.factory import ThoDataBundle, WindowDataBundle
from resp_train.data.research_v2 import ResearchV2WindowDataset
from resp_train.losses.task import RespirationTaskLoss
from scripts import p1_components_model as mod
from scripts import p1_components_runtime as run


class FixtureMamba(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.project = nn.Linear(d_model, d_model)

    def forward(self, value):
        return self.project(value)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(before)


def frequencies():
    # 仅用于合成接口测试；真实训练从冻结 cache 元数据读取坐标。
    return np.concatenate((np.geomspace(.03, .8, 56), np.geomspace(.81, 8, 41)))


def small_architecture():
    return replace(mod.architecture(), dimension=32, waveform_channels=16, condition_channels=8, layers=1)


def model(arm):
    return mod.P1ComponentsModel(arm, run.SEEDS[0], None if arm == "M1" else frequencies(),
        cfg=small_architecture(), mamba_factory=FixtureMamba)


def inputs(arm, *, requires_grad=False):
    x = torch.randn(1, 1, 18000)
    if mod.ARMS[arm].band == "none":
        return x, None
    f = len(mod.band_indices(frequencies(), mod.ARMS[arm].band))
    return x, {"w": torch.rand(1, f, 360, requires_grad=requires_grad)}


def open_fusion(net):
    if net.fusion is None:
        return
    projection = net.fusion.projection if net.spec.fusion == "add" else net.fusion.projection[-1]
    with torch.no_grad():
        projection.weight.normal_(std=.02)


def test_matrix_and_config_preserve_fixed_p1_and_training_contract(tmp_path):
    baseline = load_crd_config(run.ROOT / "configs/crd_tf_v1/crd_tf102_w_formal.yaml")
    assert len(run.plan()) == 30
    left, right = run.plan(0, 2), run.plan(1, 2)
    assert len(left) == len(right) == 15
    assert {c["cell"] for c in left}.isdisjoint(c["cell"] for c in right)
    assert {c["cell"] for c in left + right} == {c["cell"] for c in run.plan()}
    assert set(mod.ARMS) == {"M0", "M1", "M2", "M3", "M4", "M5", "M6", "S1", "S2", "S3"}
    for c in run.plan():
        cfg = run.config(c["arm"], c["seed"], tmp_path / c["cell"])
        assert cfg.training.seed == cfg.model.initialization_seed == c["seed"]
        for section in ("loss", "evaluation", "window"):
            assert cfg[section] == baseline[section]
        assert cfg.model.architecture.patch_samples == 100
        assert cfg.training.batch_size == 32 and cfg.training.gradient_accumulation_steps == 4
        assert optimizer_updates_per_epoch((10141 + 31) // 32, 4) == 80
        assert cfg.training.epochs == 80 and cfg.training.early_stopping_min_epoch == 30
        assert cfg.data.train_sample_seed == baseline.data.train_sample_seed
        assert cfg.data.val_sample_seed == baseline.data.val_sample_seed
        assert dict(cfg.model.p1_components) == mod.contract(c["arm"])
        if c["arm"] == "M1":
            assert list(cfg.model.tf_representations) == [] and cfg.data.tf_cache_path is None
        else:
            assert list(cfg.model.tf_representations) == ["w"] and cfg.data.tf_cache_path == baseline.data.tf_cache_path


def test_common_parameters_are_identical_for_all_arms():
    full = model("M0")
    for arm in mod.ARMS:
        candidate = model(arm)
        reference = dict(full.named_parameters())
        for name, p in candidate.named_parameters():
            common = name.startswith(("waveform_encoder.", "trunk.", "decoder.", "condition.encoder."))
            common |= name.startswith("condition.") and candidate.spec.readout == "attention"
            common |= name.startswith("fusion.") and candidate.spec.fusion == "film"
            if common:
                assert name in reference
                torch.testing.assert_close(p, reference[name], rtol=0, atol=0, msg=f"{arm}:{name}")
    assert sum(p.numel() for p in model("M2").parameters()) == sum(p.numel() for p in full.parameters())


def test_zero_init_identity_across_condition_variants():
    x = torch.randn(1, 1, 18000)
    with torch.no_grad():
        expected = model("M1").eval()(x)["waveform"]
        for arm in ("M0", "M2", "M3", "M4", "M5", "S1", "S2"):
            net = model(arm).eval()
            _, tf = inputs(arm)
            torch.testing.assert_close(net(x, tf=tf)["waveform"], expected, rtol=0, atol=0, msg=arm)


def test_no_tf_and_no_mamba_construct_only_active_modules(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "frequency_metadata", lambda path: pytest.fail("NoTF 不应读取频率/cache"))
    cfg = run.config("M1", run.SEEDS[0], tmp_path)
    cfg.model.architecture = asdict(small_architecture())
    net = mod.build_model(cfg, mamba_factory=FixtureMamba)
    assert net.condition is None and net.fusion is None
    assert not any(name.startswith(("condition.", "fusion.")) for name, _ in net.named_parameters())
    x = torch.randn(1, 1, 18000)
    with pytest.raises(ValueError):
        net(x, tf={"w": torch.ones(1, 97, 360)})
    def forbidden_mamba(**kwargs):
        pytest.fail("NoMamba 不应构造时序模块")
    no_mamba = mod.P1ComponentsModel("M6", run.SEEDS[0], frequencies(), cfg=small_architecture(), mamba_factory=forbidden_mamba)
    assert isinstance(no_mamba.trunk, nn.Identity)
    assert not any(name.startswith("trunk.") for name, _ in no_mamba.named_parameters())


@pytest.mark.parametrize("arm", list(mod.ARMS))
def test_all_arms_have_complete_finite_gradients(arm):
    net = model(arm)
    open_fusion(net)
    x, tf = inputs(arm, requires_grad=True)
    output = net(x, tf=tf)["waveform"]
    assert output.shape == (1, 1, 18000) and torch.isfinite(output).all()
    output.square().mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"{arm}:{name}"
    if tf is not None:
        assert tf["w"].grad is not None and tf["w"].grad.abs().sum() > 0


def test_post_film_uses_original_z_query_and_only_changes_placement():
    pre, post = model("M0").eval(), model("M5").eval()
    open_fusion(pre)
    post.load_state_dict(pre.state_dict())
    x, tf = inputs("M0")
    seen = {}
    handle = post.condition.register_forward_pre_hook(lambda module, args: seen.update(query=args[0].detach().clone()))
    with torch.no_grad():
        z = post.waveform_encoder(x)
        context = post.condition(z, tf["w"])
        expected = post.overlap(post.decoder(post.fusion(post.trunk(z), context)))
        actual = post(x, tf=tf)["waveform"]
        torch.testing.assert_close(seen["query"], z, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert (actual - pre(x, tf=tf)["waveform"]).abs().max() > 1e-6
    handle.remove()


@pytest.mark.parametrize("arm", ["M3", "S2"])
def test_mean_tf_is_uniform_over_same_physical_region(arm):
    net = model(arm).condition.eval()
    _, tf = inputs(arm)
    z = torch.randn(1, 359, 32)
    with torch.no_grad():
        actual = net(z, tf["w"])
        features = net.encoder(tf["w"][:, None])
        for j in (0, 128, 358):
            expected = net.projection(features[:, :, :, j:j + 2].mean((2, 3)))
            torch.testing.assert_close(actual[:, j], expected)
        torch.testing.assert_close(net(z * 10, tf["w"]), actual, rtol=0, atol=0)


def test_uniform_patch_pool_retains_position_and_matches_reference():
    net = model("S3").waveform_encoder.eval()
    assert not hasattr(net, "score")
    x = torch.randn(1, 1, 18000)
    with torch.no_grad():
        features = net.stem(x).unfold(-1, 100, 50).permute(0, 2, 3, 1)
        position = net.position(net.relative_seconds[:, None])
        expected = net.project((features + position).mean(2))
        torch.testing.assert_close(net(x), expected)


@pytest.mark.parametrize("arm", ["M0", "M4", "M5", "S3"])
def test_bfloat16_variants_have_finite_gradients(arm):
    net = model(arm)
    open_fusion(net)
    x, tf = inputs(arm)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = net(x, tf=tf)["waveform"]
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


@pytest.fixture
def cache_fixture(tmp_path, monkeypatch):
    from resp_train.crd import tf_v1_data
    from resp_train.crd.tf_v1_features import PROTOCOL, fixed_transform_spec
    root = tmp_path / "cache"
    root.mkdir()
    ids = np.array([11, 29], dtype=np.int64)
    np.save(root / "train_row_ids.npy", ids)
    np.save(root / "train_w.npy", np.random.default_rng(8).random((2, 97, 360), dtype=np.float32))
    np.save(root / "w_frequencies_hz.npy", frequencies())
    inventory = {}
    for path in root.iterdir():
        values = np.load(path)
        inventory[path.name] = {"sha256": run.sha(path), "size_bytes": path.stat().st_size,
            "shape": list(values.shape), "dtype": str(values.dtype), "finite": True}
    run.write_json(root / "cache_manifest.json", {"protocol": PROTOCOL, "fixed_transform_spec": fixed_transform_spec(),
        "transform_sha256": tf_v1_data.FROZEN_CACHE_TRANSFORM_SHA256, "complete": True,
        "research_test_used": False, "test_cache_created": False, "target_read": False,
        "files": inventory, "splits": {"train": {"split": "train", "count": 2,
            "row_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}}})
    monkeypatch.setattr(tf_v1_data, "FROZEN_CACHE_MANIFEST_SHA256", run.sha(root / "cache_manifest.json"))
    return root


@pytest.mark.parametrize("arm", ["M0", "M1", "M2", "S1"])
def test_dataset_adapter_and_model_training_step(arm, cache_fixture, tmp_path, monkeypatch):
    cfg = run.config(arm, run.SEEDS[0], tmp_path / "out")
    cfg.model.architecture = asdict(small_architecture())
    cfg.training.batch_size = 1
    if arm != "M1":
        cfg.data.tf_cache_path = str(cache_fixture)
    else:
        monkeypatch.setattr(run, "frequency_metadata", lambda path: pytest.fail("NoTF 不应读取元数据"))
        monkeypatch.setattr(mod, "frequency_metadata", lambda path: pytest.fail("NoTF 不应读取元数据"))
    t = np.arange(36000, dtype=np.float32) / 100
    np.savez(tmp_path / "signals.npz", bcg=np.sin(2 * np.pi * .25 * t), tho=np.sin(2 * np.pi * .25 * t + .1))
    rows = pd.DataFrame([{"dataset_row_id": row_id, "split": "train", "samp_id": 1,
        "coupling_state_id": i, "window_start_sample": i * 18000, "window_end_sample": (i + 1) * 18000,
        "bcg_signal_key": "bcg", "target_signal_key": "tho", "source_npz": "signals.npz",
        "target_source_npz": "signals.npz", "allowed_losses": "waveform"} for i, row_id in enumerate([11, 29])])
    dataset = ResearchV2WindowDataset(tmp_path / "index.csv", rows, cfg)
    bundle = WindowDataBundle(tmp_path / "index.csv", rows, dataset, DataLoader(dataset, batch_size=1), rows, pd.DataFrame())
    data = run.condition_data(ThoDataBundle(bundle, bundle, rows, pd.DataFrame()), cfg)
    batch = next(iter(data.train.loader))
    if arm == "M1":
        assert "tf" not in batch
    else:
        assert batch["tf"]["w"].shape[1] == {"M0": 97, "M2": 41, "S1": 56}[arm]
    net = mod.build_model(cfg, mamba_factory=FixtureMamba)
    loss = RespirationTaskLoss(cfg)
    optimizer, _ = build_crd_optimizer(net, cfg)
    summary, update = run.legacy._train_epoch(net, list(data.train.loader) * 4, loss, optimizer, cfg, 0, 6400, 1)
    assert update == 2 and np.isfinite(summary["loss"])


def test_snapshot_prepare_and_completed_acceptance_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "build_tho_data", lambda cfg: pytest.fail("prepare 不应访问真实数据"))
    session = run.prepare(tmp_path / "session")
    assert run.prepare(session) == session
    run.load_session(session)
    environment = {"fixture": True}
    output = session / "engineering/M0_cuda_0_fixture"
    output.mkdir(parents=True)
    run.write_json(output / "receipt.json", {"passed": True, "arm": "M0", "contract": mod.contract("M0"),
        "batch_size": 32, "accumulation": 4, "updates": 2, "mamba": "official_mamba2", "environment": environment})
    run.write_json(output / "completed.json", {"receipt_sha256": run.sha(output / "receipt.json"),
        "session_sha256": run.sha(session / "session.json")})
    monkeypatch.setattr(run.legacy, "runtime", lambda device: environment)
    monkeypatch.setattr(run, "smoke", lambda *args: pytest.fail("已通过验收应复用"))
    assert run.ensure_smoke(session, "M0", "cuda:0") == output
    (output / "receipt.json").write_text("{}")
    with pytest.raises(ValueError, match="身份"):
        run.ensure_smoke(session, "M0", "cuda:0")


def test_summary_complete_matrix_and_interaction_sign():
    records = []
    for cell in run.plan():
        for row_id, subject in [(10, 101), (20, 102)]:
            record = {"arm": cell["arm"], "seed": cell["seed"], "dataset_row_id": row_id, "samp_id": subject,
                "split": "val", "whole_rr_target_eligible": True, "local_rr_target_eligible": True,
                "local_rr_target_eligible_windows": 9, "joint_target_eligible": True,
                "envelope_spearman_target_eligible": True, "joint_prediction_degenerate": False,
                "envelope_spearman_prediction_degenerate": False}
            for metric in run.sf.PRIMARY:
                record[metric] = ({"M0": .9, "M2": .8, "M3": .6, "S2": .6}.get(cell["arm"], .9)
                                  if metric == run.sf.PCC else {"M0": 1., "M2": 2., "M3": 3., "S2": 3.}.get(cell["arm"], 1.))
            records.append(record)
    frame = pd.DataFrame(records)
    tables = run.summary_tables(frame)
    assert len(tables["per_seed"]) == 150 and len(tables["across_seed"]) == 50
    assert len(tables["paired_delta"]) == 135
    interaction = tables["representation_attention_interaction"]
    assert len(interaction) == 15
    np.testing.assert_allclose(interaction.loc[interaction.metric != run.sf.PCC, "interaction"], 1.)
    np.testing.assert_allclose(interaction.loc[interaction.metric == run.sf.PCC, "interaction"], .1)
    with pytest.raises(ValueError, match="完整"):
        run.summary_tables(frame.iloc[:-2])


def test_band_adapter_rejects_nonfinite_source_before_selection():
    full = torch.ones(97, 360)
    full[0, 0] = float("nan")
    base = [{"tf": {"w": full}}]
    dataset = run.BandDataset(base, mod.band_indices(frequencies(), "H"))
    with pytest.raises(FloatingPointError):
        dataset[0]


@pytest.mark.parametrize("arm", ["M1", "M3", "M4", "M5", "M6", "S3"])
def test_public_smoke_checks_real_loss_and_records_resources(arm, tmp_path, monkeypatch):
    session = tmp_path / arm
    session.mkdir()
    run.write_json(session / "session.json", {"fixture": True})
    monkeypatch.setattr(run, "load_session", lambda path: {})
    monkeypatch.setattr(run, "architecture", small_architecture)
    def synthetic_cwt(signal):
        if arm == "M1":
            pytest.fail("NoTF 验收不应计算 CWT")
        return np.random.default_rng(42).random((97, 360), dtype=np.float32), frequencies()
    monkeypatch.setattr(run, "cwt_magnitude_features", synthetic_cwt)
    output = run.smoke(session, arm, "cpu", mamba_factory=FixtureMamba, batch_size=1)
    receipt = run.read_json(output / "receipt.json")
    assert receipt["passed"] and receipt["accumulation"] == 4 and receipt["updates"] == 2
    assert receipt["parameters"] > 0 and receipt["train_elapsed_seconds"] > 0
    assert receipt["peak_allocated_bytes"] is None
    assert receipt["mamba"] == ("identity" if arm == "M6" else "fixture")
