"""RespDiff源码一致性与合成适配测试；不读取真实波形或独立测试集。"""

import ast
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.signal import resample
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from resp_train.respdiff import RespDiff, RespDiffSpec
from resp_train.respdiff.data import (
    ChunkDataset, assert_split_independence, chunk_manifest, interval_counts, lowpass,
    prepare_condition, prepare_target, restore_parent,
)
from resp_train.respdiff.diffusion import keyed_noise
from resp_train.respdiff.experiment import (
    evaluate_chunk_predictions, load_checkpoint, save_checkpoint, train_step,
)
from resp_train.respdiff.provenance import verify_source

SOURCE = Path("/mnt/disk_code/marques/reference_repos/RespDiff")


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(37)
    yield
    torch.set_num_threads(previous)


def source_class(profile):
    verify_source(SOURCE)
    path = SOURCE / ("model_fft.py" if profile == "source_fft" else "model.py")
    # 只加载已核验源码的类/函数定义，跳过不需要的第三方import与训练脚本。
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    scope = {"torch": torch, "nn": nn, "F": F, "np": np}
    exec(compile(tree, str(path), "exec"), scope)
    return scope["diffusion_pipeline"]


def pair(profile="source_fft"):
    torch.manual_seed(13)
    legacy = source_class(profile)(384, 8, 1, 4, torch.device("cpu"))
    torch.manual_seed(13)
    migrated = RespDiff(RespDiffSpec(8, 1, 4, profile))
    return legacy, migrated


@pytest.mark.parametrize("profile", ["source_fft", "source_plain"])
def test_source_initialization_forward_loss_grad_and_adam(profile):
    legacy, model = pair(profile)
    assert legacy.state_dict().keys() == model.state_dict().keys()
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    x, y, noise = (torch.randn(2, 1, 17) for _ in range(3))
    step = torch.tensor([0, 49])
    actual = model.diffusion_model(x, y, step)
    expected = legacy.diffusion_model(x, y, step).permute(0, 2, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with patch("torch.randint", return_value=step), patch("torch.randn", return_value=noise):
        expected_loss = legacy.train_process(x, y)
    losses = model.training_loss(x, y, step=step, noise=noise)
    torch.testing.assert_close(losses["loss"], expected_loss, rtol=2e-6, atol=2e-6)
    expected_loss.backward()
    losses["loss"].backward()
    for (name1, p1), (name2, p2) in zip(legacy.named_parameters(), model.named_parameters(), strict=True):
        assert name1 == name2
        assert p1.grad is not None and p2.grad is not None
        torch.testing.assert_close(p1.grad, p2.grad, rtol=2e-5, atol=1e-5)
    torch.optim.Adam(legacy.parameters(), lr=1e-4).step()
    torch.optim.Adam(model.parameters(), lr=1e-4).step()
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=2e-5, atol=1e-6)


def test_original_size_parameter_contract_on_meta():
    legacy_type = source_class("source_fft")
    with torch.device("meta"):
        original = legacy_type(384, 1024, 6, 128, torch.device("meta"))
        migrated = RespDiff()
    assert {key: tuple(value.shape) for key, value in original.state_dict().items()} == {
        key: tuple(value.shape) for key, value in migrated.state_dict().items()}
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in migrated.parameters())


def test_loss_batch_scaling_preserves_source_contract():
    _, model = pair()
    model.eval()
    x, target, noise = (torch.randn(1, 1, 17) for _ in range(3))
    a = model.training_loss(x, target, step=torch.tensor([23]), noise=noise)
    b = model.training_loss(x.repeat(3, 1, 1), target.repeat(3, 1, 1),
                            step=torch.tensor([23, 23, 23]), noise=noise.repeat(3, 1, 1))
    torch.testing.assert_close(b["loss_noise"], 3 * a["loss_noise"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(b["loss_fft"], a["loss_fft"], rtol=1e-5, atol=1e-6)


def test_full_ddpm_matches_source_with_identical_per_step_noise():
    legacy, model = pair()
    legacy.eval()
    model.eval()
    x = torch.randn(2, 1, 9)
    keys = ["val:1:0", "val:1:1"]
    noises = {step: keyed_noise(x.shape, keys=keys, seed=7, trajectory=0, step=step,
                                device="cpu") for step in range(51)}
    with torch.no_grad(), patch("torch.randn", return_value=noises[50]), patch(
        "torch.randn_like", side_effect=[noises[step] for step in range(49, -1, -1)]
    ):
        expected = legacy.imputation(x, 1)[:, 0]
    actual = model.sample_mean(x, keys=keys, seed=7, n_samples=1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)
    again = model.sample_mean(x, keys=keys, seed=7, n_samples=1)
    torch.testing.assert_close(again, actual, rtol=0, atol=0)


def test_source_batch_one_scalar_index_bug_and_portable_sampler():
    legacy, model = pair()
    legacy.eval()
    model.eval()
    x = torch.randn(1, 1, 9)
    # 本环境中NumPy用单元素torch索引产生标量；保留来源失败证据。
    with torch.no_grad(), pytest.raises(IndexError, match="0-dim"):
        legacy.imputation(x, 1)
    assert torch.isfinite(model.sample_mean(x, keys=["val:1:0"], seed=7, n_samples=1)).all()


def test_sampling_mean_keys_and_rng_are_separate():
    _, model = pair()
    model.eval()
    x = torch.randn(1, 1, 7)
    state = torch.get_rng_state().clone()
    actual = model.sample_mean(x, keys=["val:1:0"], seed=9, n_samples=2)
    assert torch.equal(state, torch.get_rng_state())
    samples = []
    with torch.no_grad():
        for trajectory in range(2):
            current = keyed_noise(x.shape, keys=["val:1:0"], seed=9, trajectory=trajectory, step=50, device="cpu")
            for step in range(49, -1, -1):
                prediction = model.diffusion_model(x, current, torch.tensor([step]))
                noise = keyed_noise(x.shape, keys=["val:1:0"], seed=9, trajectory=trajectory, step=step, device="cpu")
                current = model.reverse_step(current, prediction, step=step, noise=noise)
            samples.append(current)
    torch.testing.assert_close(actual, torch.stack(samples).mean(0))
    joined = keyed_noise((2, 1, 7), keys=["a", "b"], seed=9, trajectory=0, step=2, device="cpu")
    single = keyed_noise((1, 1, 7), keys=["b"], seed=9, trajectory=0, step=2, device="cpu")
    assert torch.equal(joined[1:], single)


@pytest.mark.parametrize("failure", ["nonfinite", "shape", "step", "mode", "samples"])
def test_explicit_model_failures(failure):
    _, model = pair()
    x = torch.ones(1, 1, 9)
    with pytest.raises(ValueError):
        if failure == "nonfinite":
            model.training_loss(x * float("nan"), x)
        elif failure == "shape":
            model.training_loss(x, torch.ones(1, 2, 9))
        elif failure == "step":
            model.training_loss(x, x, step=torch.tensor([-1]))
        elif failure == "mode":
            model.sample_mean(x, keys=["a"], seed=1, n_samples=1)
        else:
            model.eval().sample_mean(x, keys=["a"], seed=1, n_samples=0)


def wave():
    t = np.arange(18000) / 100
    return (1 + .3 * np.cos(2 * np.pi * .02 * t)) * np.sin(2 * np.pi * .2 * t)


def rows(split="val", subject=1, row_id=1):
    return pd.DataFrame([{"dataset_row_id": row_id, "split": split, "samp_id": subject,
                          "source_npz": f"/synthetic/bcg_{subject}.npz",
                          "target_source_npz": f"/synthetic/tho_{subject}.npz",
                          "window_start_sample": 0, "window_end_sample": 18000}])


def test_adapter_length_phase_amplitude_and_source_minmax():
    signal = wave()
    x, y = prepare_condition(signal), prepare_target(signal)
    assert x.shape == y.shape == (36, 1, 150)
    np.testing.assert_allclose(x.reshape(-1), resample(signal, 5400), atol=1e-7)
    restored = restore_parent(x, chunk_indices=range(36))
    np.testing.assert_allclose(restored[1000:-1000], signal[1000:-1000], atol=.001)
    rms = np.sqrt(np.mean(x[:, 0] ** 2, axis=1))
    assert rms.max() / rms.min() > 1.5
    source = prepare_condition(signal, profile="source_minmax")
    np.testing.assert_allclose(source.min(axis=-1), -1)
    np.testing.assert_allclose(source.max(axis=-1), 1)
    target = prepare_target(signal, profile="source_minmax")
    np.testing.assert_allclose(target.min(axis=-1), 0)
    np.testing.assert_allclose(target.max(axis=-1), 1)


def test_adapter_impulse_frequency_and_failures():
    impulse = np.zeros(18000)
    impulse[9000] = 1
    down = prepare_condition(impulse)
    assert np.argmax(down) == 2700
    assert np.argmax(restore_parent(down, chunk_indices=range(36))) == 9000
    t = np.arange(5400) / 30
    high = np.sin(2 * np.pi * 3 * t)
    assert np.sqrt(np.mean(lowpass(high)[300:-300] ** 2)) < 1e-4
    for indices in (range(35), list(reversed(range(36))), [0] * 36):
        with pytest.raises(ValueError):
            restore_parent(down, chunk_indices=indices)
    with pytest.raises(ValueError, match="常量"):
        prepare_condition(np.ones(18000), profile="source_minmax")
    with pytest.raises(ValueError):
        prepare_target(np.full(18000, np.nan))


def test_chunk_identity_split_and_parent_dataset():
    train, val = rows("train", 2, 2), rows()
    assert_split_independence(train, val)
    with pytest.raises(ValueError, match="主体重叠"):
        assert_split_independence(rows("train", 1, 2), val)
    with pytest.raises(ValueError, match="test"):
        chunk_manifest(rows("test"))
    manifest = chunk_manifest(val)
    assert manifest.iloc[-1].chunk_end_sample == 18000
    parent = {"x": torch.from_numpy(wave()[None]), "target": torch.from_numpy(wave()[None]),
              "meta": {"dataset_row_id": 1}}
    dataset = ChunkDataset([parent], val)
    assert len(dataset) == 36
    assert dataset[35]["meta"]["chunk_start_sample"] == 17500
    assert dataset[0]["x"].shape == (1, 150)
    shifted = val.copy()
    shifted["dataset_row_id"] = 2
    shifted["window_start_sample"] = 500
    shifted["window_end_sample"] = 18500
    counts = interval_counts(chunk_manifest(pd.concat([val, shifted], ignore_index=True)))
    assert counts == {"chunks": 72, "unique_source_intervals": 37, "max_multiplicity": 2}


def test_complete_validation():
    metadata = rows()
    manifest = chunk_manifest(metadata)
    predictions = prepare_condition(wave())
    evaluated = evaluate_chunk_predictions(predictions, manifest, metadata, wave()[None])
    assert evaluated["waveform"].shape == (1, 18000)
    assert evaluated["summary"]["lag_signed_pcc"] > .999
    assert evaluated["summary"]["rr180_mae_bpm"] < .01
    with pytest.raises(ValueError, match="身份"):
        evaluate_chunk_predictions(predictions, manifest.iloc[::-1], metadata, wave()[None])


def test_update_checkpoint_roundtrip_and_nonfinite(tmp_path):
    _, model = pair()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    x = torch.randn(2, 1, 17)
    summary = train_step(model, torch.optim.Adam(model.parameters(), lr=1e-4), x, x,
                         generator=torch.Generator().manual_seed(4))
    assert np.isfinite(list(summary.values())).all()
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters())
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, update=1, metadata={"synthetic": True})
    restored, payload = load_checkpoint(checkpoint)
    assert payload["update"] == 1
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, model, update=2, metadata={})
    corrupt = tmp_path / "corrupt.pt"
    payload["state_dict"]["alpha_torch"][0] = float("nan")
    torch.save(payload, corrupt)
    with pytest.raises(ValueError, match="非有限"):
        load_checkpoint(corrupt)


def test_source_identity_rejects_modified_copy(tmp_path):
    for path in SOURCE.glob("*.py"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    (tmp_path / "model.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="哈希"):
        verify_source(tmp_path)


def test_research_v2_adapter_uses_disposable_npz_and_exact_rows(tmp_path):
    from resp_train.data.research_v2 import ResearchV2WindowDataset

    source_path = tmp_path / "bcg.npz"
    target_path = tmp_path / "tho.npz"
    np.savez(source_path, bcg=wave())
    np.savez(target_path, tho=wave())
    metadata = rows()
    metadata["source_npz"], metadata["target_source_npz"] = str(source_path), str(target_path)
    metadata["bcg_signal_key"], metadata["target_signal_key"] = "bcg", "tho"
    metadata["coupling_state_id"], metadata["allowed_losses"] = 0, "waveform"
    index_path = tmp_path / "index.csv"
    metadata.to_csv(index_path, index=False)
    config = OmegaConf.create({"window": {"duration_samples": 18000, "target_fs": 100},
                              "data": {"drop_nonfinite_windows": False, "tf_cache_path": None}})
    parents = ResearchV2WindowDataset(index_path, metadata, config)
    dataset = ChunkDataset(parents, metadata)
    assert dataset[0]["x"].shape == (1, 150)
    np.testing.assert_allclose(dataset[0]["target"], prepare_target(wave())[0], atol=1e-6)
    changed = metadata.copy()
    changed["dataset_row_id"] = 2
    with pytest.raises(ValueError, match="实际rows"):
        ChunkDataset(parents, changed)


def test_cpu_cli_completion_and_failure_receipts(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/run_respdiff_tho_v1.py"
    spec = importlib.util.spec_from_file_location("respdiff_cli_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "complete"
    module.synthetic_smoke(output, SOURCE)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["chunks"] == 36
    assert not (output / "failure.json").exists()
    assert np.load(output / "prediction.npy").shape == (1, 18000)
    metrics = json.loads((output / "metrics.json").read_text())
    assert set(metrics) == {"training", "validation"}
    assert np.isfinite(metrics["validation"]["rr180_mae_bpm"])
    with pytest.raises(FileExistsError):
        module.synthetic_smoke(output, SOURCE)
    def fail(*args, **kwargs):
        raise ValueError("synthetic injected failure")
    monkeypatch.setattr(module, "train_step", fail)
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="injected"):
        module.synthetic_smoke(failed, SOURCE)
    assert not (failed / "receipt.json").exists()
    assert "injected" in json.loads((failed / "failure.json").read_text())["error"]


def test_overflow_and_optimizer_coverage_fail_explicitly():
    with pytest.raises((FloatingPointError, ValueError)):
        prepare_condition(wave() * 1e100)
    _, model = pair()
    optimizer = torch.optim.Adam([next(model.parameters())])
    x = torch.randn(2, 1, 9)
    with pytest.raises(ValueError, match="精确覆盖"):
        train_step(model, optimizer, x, x, generator=torch.Generator().manual_seed(1))
