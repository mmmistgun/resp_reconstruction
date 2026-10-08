"""M4-v2 的合成 CPU 几何、梯度、初始化和生命周期合同验证。"""
from dataclasses import asdict, replace
import hashlib

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from resp_train.crd.config import load_crd_config
from resp_train.crd.training import build_crd_optimizer, optimizer_updates_per_epoch
from resp_train.data.factory import ThoDataBundle, WindowDataBundle
from resp_train.data.research_v2 import ResearchV2WindowDataset
from resp_train.losses.task import RespirationTaskLoss
from scripts import m4_residual_model as mod
from scripts import m4_residual_runtime as run


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
    return np.concatenate((np.geomspace(.03, .8, 56), np.geomspace(.81, 8, 41)))


def small_architecture():
    return replace(mod.architecture(), dimension=32, waveform_channels=16, condition_channels=8, layers=1)


def model(arm):
    return mod.M4ResidualModel(arm, run.SEEDS[0], None if arm == "A1" else frequencies(),
        cfg=small_architecture(), mamba_factory=FixtureMamba)


def inputs(arm, *, requires_grad=False):
    return torch.randn(1, 1, 18000), (None if arm == "A1" else {"w": torch.rand(1, 97, 360, requires_grad=requires_grad)})


def open_branch(net):
    if net.condition is not None:
        with torch.no_grad():
            net.condition.output.weight.normal_(std=.02)


def test_fixed_matrix_data_loss_and_selector_contract(tmp_path):
    baseline = load_crd_config(run.ROOT / "configs/crd_tf_v1/crd_tf102_w_formal.yaml")
    assert len(run.plan()) == 18
    shards = [run.plan(i, 2) for i in (0, 1)]
    assert [len(x) for x in shards] == [9, 9]
    assert len({c["cell"] for shard in shards for c in shard}) == 18
    assert set(mod.ARMS) == {"A0", "A1", "A2", "A3", "S1", "S2"}
    for cell in run.plan():
        cfg = run.config(cell["arm"], cell["seed"], tmp_path / cell["cell"])
        assert cfg.training.seed == cfg.model.initialization_seed == cell["seed"]
        for section in ("loss", "evaluation", "window"):
            assert cfg[section] == baseline[section]
        assert cfg.model.architecture.patch_samples == 100
        assert cfg.training.batch_size == 32 and cfg.training.gradient_accumulation_steps == 4
        assert optimizer_updates_per_epoch((10141 + 31) // 32, 4) == 80
        assert cfg.training.epochs == 80 and cfg.training.early_stopping_min_epoch == 30
        assert cfg.data.train_sample_seed == baseline.data.train_sample_seed
        assert cfg.data.val_sample_seed == baseline.data.val_sample_seed
        assert dict(cfg.model.m4_residual) == mod.contract(cell["arm"])
        assert list(cfg.model.tf_representations) == ([] if cell["arm"] == "A1" else ["w"])
    assert run.config("A1", run.SEEDS[0], tmp_path).data.tf_cache_path is None


def test_common_initialization_independent_of_optional_modules():
    full = dict(model("A0").named_parameters())
    for arm in mod.ARMS:
        for name, parameter in model(arm).named_parameters():
            if name in full and parameter.shape == full[name].shape:
                torch.testing.assert_close(parameter, full[name], rtol=0, atol=0, msg=f"{arm}:{name}")


def test_zero_output_keeps_initial_waveform_equal_to_notf():
    x = torch.randn(1, 1, 18000)
    with torch.no_grad():
        expected = model("A1").eval()(x)["waveform"]
        for arm in ("A0", "A2", "S1", "S2"):
            net = model(arm).eval()
            assert net.condition.output.weight.count_nonzero() == 0
            assert net.condition.output.bias.count_nonzero() == 0
            assert not hasattr(net, "fusion")
            torch.testing.assert_close(net(x, tf={"w": torch.rand(1, 97, 360)})["waveform"], expected, rtol=0, atol=0)


def test_one_affine_output_matches_collapsed_two_affines():
    # 检验函数等价性，不把它等同于从头训练的优化等价性。
    c = torch.randn(2, 5, 32)
    first, second = nn.Linear(32, 32), nn.Linear(32, 32)
    collapsed = nn.Linear(32, 32)
    with torch.no_grad():
        collapsed.weight.copy_(second.weight @ first.weight)
        collapsed.bias.copy_(second.weight @ first.bias + second.bias)
    torch.testing.assert_close(collapsed(c), second(first(c)))


@pytest.mark.parametrize("arm", ["A0", "A2", "A3", "S1", "S2"])
def test_zero_projection_opens_before_upstream_gradient(arm):
    net = model(arm).eval()
    x, tf = inputs(arm)
    optimizer = torch.optim.SGD(net.parameters(), lr=.1)
    output = net(x, tf=tf)["waveform"]
    target = torch.randn_like(output)
    (output - target).square().mean().backward()
    assert net.condition.output.weight.grad.abs().sum() > 0
    assert net.condition.encoder[1].weight.grad.count_nonzero() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (net(x, tf=tf)["waveform"] - target).square().mean().backward()
    assert net.condition.encoder[1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("arm", list(mod.ARMS))
def test_all_arms_have_complete_finite_gradients(arm):
    net = model(arm)
    open_branch(net)
    x, tf = inputs(arm, requires_grad=True)
    output = net(x, tf=tf)["waveform"]
    assert output.shape == (1, 1, 18000) and torch.isfinite(output).all()
    output.square().mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"{arm}:{name}"
    if tf is not None:
        assert tf["w"].grad.abs().sum() > 0


def test_query_norm_only_changes_query_input():
    net = model("S1").eval()
    open_branch(net)
    x, tf = inputs("S1")
    seen = {}
    handle = net.condition.query.register_forward_pre_hook(lambda _, args: seen.update(query=args[0].detach().clone()))
    with torch.no_grad():
        z = net.waveform_encoder(x)
        delta = net.condition(z, tf["w"])
        torch.testing.assert_close(seen["query"], net.condition.query_norm(z), rtol=0, atol=0)
        expected = net.overlap(net.decoder(net.trunk(z + delta)))
        torch.testing.assert_close(net(x, tf=tf)["waveform"], expected, rtol=0, atol=0)
    handle.remove()


def test_coordinate_ablation_retains_geometry_and_content_attention():
    full, content = model("A0").eval(), model("S2").eval()
    assert content.condition.frequency_bias is None and content.condition.time_bias is None
    assert not any("frequency_bias" in n or "time_bias" in n for n, _ in content.named_parameters())
    open_branch(full)
    state = {k: v for k, v in full.state_dict().items() if k in content.state_dict()}
    content.load_state_dict(state)
    with torch.no_grad():
        for module in (full.condition.frequency_bias, full.condition.time_bias):
            for p in module.parameters():
                p.zero_()
        x, tf = inputs("S2")
        torch.testing.assert_close(full(x, tf=tf)["waveform"], content(x, tf=tf)["waveform"], rtol=0, atol=0)
        torch.testing.assert_close(content.condition.relative_seconds, torch.tensor([-.25, .25]))


def test_mean_residual_uses_same_aligned_cells_and_single_c_to_d_projection():
    net = model("A2").condition.eval()
    assert net.output.in_features == 8 and net.output.out_features == 32
    with torch.no_grad():
        net.output.weight.normal_(std=.02)
        w, z = torch.rand(1, 97, 360), torch.randn(1, 359, 32)
        actual = net(z, w)
        features = net.encoder(w[:, None])
        for j in (0, 128, 358):
            torch.testing.assert_close(actual[:, j], net.output(features[:, :, :, j:j + 2].mean((2, 3))))
        torch.testing.assert_close(net(z * 10, w), actual, rtol=0, atol=0)


def test_inactive_modules_do_not_access_metadata_or_construct_mamba(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "frequency_metadata", lambda _: pytest.fail("NoTF 不应读取 cache"))
    cfg = run.config("A1", run.SEEDS[0], tmp_path)
    cfg.model.architecture = asdict(small_architecture())
    net = mod.build_model(cfg, mamba_factory=FixtureMamba)
    assert net.condition is None
    with pytest.raises(ValueError):
        net(torch.randn(1, 1, 18000), tf={"w": torch.rand(1, 97, 360)})
    def forbidden(**kwargs):
        pytest.fail("NoMamba 不应构造时序模块")
    net = mod.M4ResidualModel("A3", run.SEEDS[0], frequencies(), cfg=small_architecture(), mamba_factory=forbidden)
    assert isinstance(net.trunk, nn.Identity)


@pytest.mark.parametrize("arm", list(mod.ARMS))
def test_bfloat16_complete_gradients(arm):
    net = model(arm)
    open_branch(net)
    x, tf = inputs(arm)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = net(x, tf=tf)["waveform"]
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


@pytest.mark.parametrize("arm", list(mod.ARMS))
def test_public_smoke_uses_task_loss_and_two_accumulated_updates(arm, tmp_path, monkeypatch):
    session = tmp_path / arm
    session.mkdir()
    run.write_json(session / "session.json", {"fixture": True})
    monkeypatch.setattr(run, "load_session", lambda _: {})
    monkeypatch.setattr(run, "architecture", small_architecture)
    def synthetic_cwt(signal):
        if arm == "A1":
            pytest.fail("NoTF 验收不应计算 CWT")
        return np.random.default_rng(42).random((97, 360), dtype=np.float32), frequencies()
    monkeypatch.setattr(run, "cwt_magnitude_features", synthetic_cwt)
    output = run.smoke(session, arm, "cpu", mamba_factory=FixtureMamba, batch_size=1)
    receipt = run.read_json(output / "receipt.json")
    assert receipt["passed"] and receipt["accumulation"] == 4 and receipt["updates"] == 2
    assert receipt["parameters"] > 0 and receipt["train_elapsed_seconds"] > 0
    assert receipt["mamba"] == ("identity" if arm == "A3" else "fixture")


def test_snapshot_and_acceptance_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "build_tho_data", lambda _: pytest.fail("prepare 不应读取真实数据"))
    session = run.prepare(tmp_path / "session")
    assert run.prepare(session) == session
    environment = {"fixture": True}
    output = session / "engineering/A0_cuda_0_fixture"
    output.mkdir(parents=True)
    run.write_json(output / "receipt.json", {"passed": True, "arm": "A0", "contract": mod.contract("A0"),
        "batch_size": 32, "accumulation": 4, "updates": 2, "mamba": "official_mamba2", "environment": environment})
    run.write_json(output / "completed.json", {"receipt_sha256": run.sha(output / "receipt.json"),
        "session_sha256": run.sha(session / "session.json")})
    monkeypatch.setattr(run.legacy, "runtime", lambda _: environment)
    monkeypatch.setattr(run, "smoke", lambda *a: pytest.fail("应复用已通过验收"))
    assert run.ensure_smoke(session, "A0", "cuda:0") == output
    (output / "receipt.json").write_text("{}")
    with pytest.raises(ValueError, match="身份"):
        run.ensure_smoke(session, "A0", "cuda:0")


def test_summary_complete_matrix_and_increment_directions():
    records = []
    errors = {"A1": 4., "A2": 3., "S2": 2., "A0": 1., "A3": 5., "S1": 1.}
    for cell in run.plan():
        for row_id, subject in ((10, 101), (20, 102)):
            r = {"arm": cell["arm"], "seed": cell["seed"], "dataset_row_id": row_id, "samp_id": subject,
                "split": "val", "whole_rr_target_eligible": True, "local_rr_target_eligible": True,
                "local_rr_target_eligible_windows": 9, "joint_target_eligible": True,
                "envelope_spearman_target_eligible": True, "joint_prediction_degenerate": False,
                "envelope_spearman_prediction_degenerate": False}
            for metric in run.sf.PRIMARY:
                r[metric] = 1 - .1 * errors[cell["arm"]] if metric == run.sf.PCC else errors[cell["arm"]]
            records.append(r)
    frame = pd.DataFrame(records)
    tables = run.summary_tables(frame)
    assert len(tables["per_seed"]) == 90 and len(tables["across_seed"]) == 30
    assert len(tables["paired_delta"]) == 75 and len(tables["incremental_delta"]) == 45
    assert set(tables["incremental_delta"].stage) == set(run.INCREMENTS)
    for metric, expected in ((run.sf.PCC, .1), (run.sf.PRIMARY[0], 1.)):
        np.testing.assert_allclose(tables["incremental_delta"].query("metric == @metric").improvement, expected)
    with pytest.raises(ValueError, match="完整"):
        run.summary_tables(frame.iloc[:-2])


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


@pytest.mark.parametrize("arm", ["A0", "A1", "A2"])
def test_dataset_to_optimizer_step(arm, cache_fixture, tmp_path, monkeypatch):
    cfg = run.config(arm, run.SEEDS[0], tmp_path / "out")
    cfg.model.architecture = asdict(small_architecture())
    cfg.training.batch_size = 1
    if arm != "A1":
        cfg.data.tf_cache_path = str(cache_fixture)
    else:
        monkeypatch.setattr(run, "frequency_metadata", lambda _: pytest.fail("NoTF 不应读取元数据"))
        monkeypatch.setattr(mod, "frequency_metadata", lambda _: pytest.fail("NoTF 不应读取元数据"))
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
    assert ("tf" in batch) == (arm != "A1")
    net = mod.build_model(cfg, mamba_factory=FixtureMamba)
    loss = RespirationTaskLoss(cfg)
    optimizer, _ = build_crd_optimizer(net, cfg)
    summary, updates = run.legacy._train_epoch(net, list(data.train.loader) * 4, loss, optimizer, cfg, 0, 6400, 1)
    assert updates == 2 and np.isfinite(summary["loss"])
