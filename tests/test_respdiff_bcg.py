"""BCG 适配合同的合成 CPU 定向验证；不访问真实数据。"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.config import ROOT, load_config
from resp_train.respdiff_bcg.model import RespDiffBCG, TIMESTEPS
from resp_train.respdiff_bcg.runtime import check_primary_metrics, train_updates
from resp_train.respdiff_bcg.signal import (
    chunks_from_parent, parent_noise, prepare_parent, restore_parent,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def cli():
    spec = importlib.util.spec_from_file_location("bcg_cli", ROOT / "scripts/run_respdiff_bcg_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KnownNoise(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.17))

    def forward(self, condition, current, steps):
        return condition * self.scale + steps[:, None, None] * 0.001


def oracle_model():
    model = RespDiffBCG(RespDiffSpec(8, 1, 4))
    model.diffusion_model = KnownNoise()
    return model


def test_loss_matches_mean_fft_and_gradient_and_batch_invariance():
    model = oracle_model()
    x, target, noise = (torch.randn(2, 1, 600) for _ in range(3))
    step = torch.tensor([2, 40])
    losses = model.training_loss(x, target, step=step, noise=noise)
    alpha = model.alpha_torch[step]
    noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
    predicted = model.diffusion_model(x, noisy, step)
    reconstructed = (noisy - (1 - alpha).sqrt() * predicted) / alpha.sqrt()
    expected = (noise - predicted).square().mean() + 0.01 * (
        torch.fft.fft(reconstructed, norm="ortho").abs()
        - torch.fft.fft(target, norm="ortho").abs()).square().mean()
    torch.testing.assert_close(losses["loss"], expected)
    actual_grad = torch.autograd.grad(losses["loss"], model.diffusion_model.scale, retain_graph=True)[0]
    torch.testing.assert_close(actual_grad, torch.autograd.grad(expected, model.diffusion_model.scale)[0])
    doubled = model.training_loss(x.repeat(2, 1, 1), target.repeat(2, 1, 1),
                                  step=step.repeat(2), noise=noise.repeat(2, 1, 1))
    for key in losses:
        torch.testing.assert_close(losses[key], doubled[key])


def test_ddim_matches_independent_oracle_and_keeps_amplitude():
    model = oracle_model().eval()
    x, initial = torch.randn(2, 1, 600), torch.randn(2, 1, 600)
    actual = model.sample_chunks(x, initial)
    current = initial.double()
    alpha = np.cumprod(1 - np.linspace(0.0001, 0.5, 50))
    for index, step in enumerate(TIMESTEPS):
        eps = x.double() * float(model.diffusion_model.scale.detach()) + step * 0.001
        a, b = alpha[step], alpha[TIMESTEPS[index + 1]] if index < 5 else 1.0
        current = np.sqrt(b / a) * current + (np.sqrt(1 - b) - np.sqrt(b / a * (1 - a))) * eps
    torch.testing.assert_close(actual.double(), current, rtol=2e-6, atol=0.001)
    assert actual.abs().max() > 2
    torch.testing.assert_close(model.sample_chunks(x, initial), actual, rtol=0, atol=0)


def test_shared_noise_overlap_identity_and_rng_isolation():
    state = torch.get_rng_state()
    noise = parent_noise(split="val", row_id=4, seed=7)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(noise[:-1, :, 300:], noise[1:, :, :300])
    assert torch.equal(noise, parent_noise(split="val", row_id=4, seed=7))
    assert not torch.equal(noise, parent_noise(split="val", row_id=5, seed=7))
    assert not torch.equal(noise[0, 0, :300].flip(0), noise[0, 0, 300:])


@pytest.mark.parametrize("kind", ["random", "edges", "constant"])
def test_ola_reconstructs_all_retained_samples(kind):
    parent = np.random.default_rng(2).normal(size=3600).astype(np.float32)
    if kind == "edges":
        parent[:] = 0
        parent[[0, 299, 300, 3299, 3599]] = [1, 2, 3, 4, 5]
    if kind == "constant":
        parent[:] = 4.5
    low, high = restore_parent(chunks_from_parent(parent), chunk_indices=range(13))
    np.testing.assert_allclose(low, parent, atol=1e-6)
    assert high.shape == (18000,) and np.isfinite(high).all()
    if kind == "constant":
        np.testing.assert_allclose(high, 4.5, atol=1e-6)
    with pytest.raises(ValueError, match="完整"):
        restore_parent(chunks_from_parent(parent), chunk_indices=list(range(12)) + [11])


def test_antialias_alignment_gain_and_no_chunk_normalization():
    time = np.arange(18000) / 100
    low = prepare_parent(3 * np.sin(2 * np.pi * 0.2 * time) + 2)
    np.testing.assert_allclose(low[60:-60], (3 * np.sin(2 * np.pi * 0.2 * time[::5]) + 2)[60:-60], atol=0.0002)
    rejected = prepare_parent(np.sin(2 * np.pi * 15 * time))
    assert np.max(np.abs(rejected[60:-60])) < 0.001
    impulse = np.zeros(18000)
    impulse[9000] = 1
    assert np.argmax(prepare_parent(impulse)) == 1800
    chunks = chunks_from_parent(low)
    np.testing.assert_array_equal(chunks[1, 0], low[:600])


def test_exact_6400_budget_and_update_lr_boundaries(tmp_path):
    class Scalar(torch.nn.Module):
        spec = RespDiffSpec(8, 1, 4)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def training_loss(self, x, y):
            loss = (self.weight * x - y).square().mean()
            return {"loss": loss}

    dataset = [{"x": torch.ones(1, 1), "target": torch.zeros(1, 1), "index": i} for i in range(3)]
    train_updates(Scalar(), dataset, load_config(), tmp_path, batch_size=2)
    history = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert len(history) == 6400 and history[-1]["update"] == 6400
    assert [len(row["chunk_indices"]) for row in history[:4]] == [2, 1, 2, 1]
    for update, expected in ((1, 1e-4), (4480, 1e-4), (4481, 1e-5), (6336, 1e-5), (6337, 1e-6)):
        assert history[update - 1]["lr"] == pytest.approx(expected)
    checkpoint = torch.load(tmp_path / "final.pt", weights_only=False)
    assert checkpoint["update"] == 6400 and checkpoint["selector"] == "final_update"


def test_synthetic_npz_training_validation_receipt_and_failure(tmp_path, monkeypatch, capsys):
    import pandas as pd
    from resp_train.metrics.task import evaluate_task_predictions
    from resp_train.respdiff_bcg.runtime import sha256

    module = cli()
    output = tmp_path / "complete"
    module.run("synthetic-smoke", output, load_config())
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete" and not receipt["formal_evidence"]
    waves = np.load(output / "validation_waveforms.npz")
    assert waves["r_tho_hat"].shape == (2, 18000)
    assert waves["prediction_20hz"].shape == (2, 3600)
    assert len((output / "history.jsonl").read_text().splitlines()) == 2
    terminal = capsys.readouterr().err
    logfile = (output / "run.log").read_text()
    for message in ("train_batch=2 inference_batch=8", "训练 update 1/2", "训练 update 2/2",
                    "loss_noise=", "loss_fft=", "ETA=", "验证采样 batch 4/4",
                    "父窗口重建 2/2", "指标父窗口 2/2", "Validation 汇总"):
        assert message in terminal and message in logfile
    assert receipt["artifacts"]["run.log"] == sha256(output / "run.log")
    direct = evaluate_task_predictions(dict(waves), load_config(), method="RespDiff-BCG-v1")
    pd.testing.assert_frame_equal(pd.read_csv(output / "validation_per_parent.csv"), direct,
                                  check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    with pytest.raises(FileExistsError):
        module.run("synthetic-smoke", output, load_config())
    def fail(*args, **kwargs):
        raise ValueError("注入失败")
    monkeypatch.setattr(module, "train_updates", fail)
    with pytest.raises(ValueError, match="注入失败"):
        module.run("synthetic-smoke", tmp_path / "failed", load_config())
    assert json.loads((tmp_path / "failed/failure.json").read_text())["stage"] == "training"
    failed_log = (tmp_path / "failed/run.log").read_text()
    assert "运行终止：stage=training" in failed_log and "Traceback" in failed_log
    assert not (tmp_path / "failed/receipt.json").exists()
    with pytest.raises(ValueError, match="confirm-training"):
        module.run("train", tmp_path / "forbidden", load_config())
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize("path,value", [("training.max_updates", 10), ("data.train_split", "test"),
                                      ("inference.n_samples", 100), ("training.proxy", True)])
def test_contract_rejects_changes(tmp_path, path, value):
    cfg = load_config()
    OmegaConf.update(cfg, path, value)
    filename = tmp_path / "changed.yaml"
    OmegaConf.save(cfg, filename)
    with pytest.raises(ValueError, match="合同"):
        load_config(filename)


def test_nonfinite_is_not_silently_dropped():
    import pandas as pd
    with pytest.raises(ValueError, match="有限"):
        prepare_parent(np.full(18000, np.nan))
    metrics = pd.DataFrame({"whole_rr_abs_error_bpm": [np.nan], "whole_rr_target_eligible": [True]})
    with pytest.raises(FloatingPointError, match="关键指标"):
        check_primary_metrics(metrics)


def test_factory_full_rows_relative_paths_and_split_leakage(tmp_path):
    import pandas as pd
    from resp_train.data.factory import build_tho_data
    from resp_train.respdiff.data import assert_split_independence
    from resp_train.respdiff_bcg.data import ChunkDataset

    cfg = load_config()
    cfg.data.dataset_root, cfg.data.index_csv = str(tmp_path), "index.csv"
    records = []
    time = np.arange(18000) / 100
    for row_id, split in enumerate(("train", "val", "test")):
        # test 路径故意不存在：整个 train/val 数据工厂不应打开它。
        filename = f"{split}.npz"
        if split != "test":
            np.savez(tmp_path / filename, bcg=np.sin(time), tho=np.cos(time))
        records.append({"dataset_row_id": row_id, "samp_id": row_id, "split": split,
                        "source_npz": filename, "target_source_npz": filename,
                        "coupling_state_id": 0, "window_start_s": 0, "window_end_s": 180,
                        "bcg_rawish_segment_soft_z_key": "bcg",
                        "target_waveform_segment_soft_z_key": "tho",
                        "hard_valid_ratio": 1, "state_alignment_valid_ratio": 1,
                        "allowed_losses": "waveform", "reason": "",
                        "state_alignment_method": "synthetic", "supervision_confidence_level": "high"})
    pd.DataFrame(records).to_csv(tmp_path / "index.csv", index=False)
    bundle = build_tho_data(cfg)
    train, val = [ChunkDataset(part.dataset, part.rows) for part in (bundle.train, bundle.val)]
    assert_split_independence(train.rows, val.rows)
    assert train[0]["x"].shape == val[12]["target"].shape == (1, 600)
    assert train.rows.source_npz.iloc[0] == str(tmp_path / "train.npz")
    changed = val.rows.copy()
    changed["samp_id"] = train.rows.samp_id.iloc[0]
    with pytest.raises(ValueError, match="主体重叠"):
        assert_split_independence(train.rows, changed)
