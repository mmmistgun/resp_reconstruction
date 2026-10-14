"""临时合成 cache/NPZ 验证数据、任务循环与运行产物。"""
import hashlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from resp_train import bcg_cwt_training as run
from resp_train.crd import tf_v1_data
from resp_train.crd.tf_v1_features import PROTOCOL, fixed_transform_spec
from resp_train.data.bcg_cwt import FrozenWCWTReader, frozen_w_metadata
from resp_train.data.research_v2 import ResearchV2WindowDataset
from resp_train.models import bcg_cwt_patch_mamba as mod
from scripts.check_bcg_cwt_patch_mamba import SyntheticMamba


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def fixture_data(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    root.mkdir()
    frequencies = np.geomspace(.03, 8, 97)
    np.save(root / "w_frequencies_hz.npy", frequencies)
    splits = {}
    for split, row_id in (("train", 11), ("val", 29)):
        ids = np.array([row_id], dtype=np.int64)
        np.save(root / f"{split}_row_ids.npy", ids)
        np.save(root / f"{split}_w.npy", np.random.default_rng(row_id).random((1, 97, 360), dtype=np.float32))
        splits[split] = {"split": split, "count": 1, "row_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
    inventory = {}
    for path in root.iterdir():
        value = np.load(path)
        inventory[path.name] = {"size_bytes": path.stat().st_size, "sha256": run.sha256(path),
                                "shape": list(value.shape), "dtype": str(value.dtype), "finite": True}
    manifest = {"protocol": PROTOCOL, "fixed_transform_spec": fixed_transform_spec(),
                "transform_sha256": tf_v1_data.FROZEN_CACHE_TRANSFORM_SHA256,
                "complete": True, "research_test_used": False, "test_cache_created": False,
                "target_read": False, "files": inventory, "splits": splits}
    run.write_json(root / "cache_manifest.json", manifest)
    # 只在 fixture 中替换 manifest 哈希；读取和校验代码均走生产实现。
    monkeypatch.setattr(tf_v1_data, "FROZEN_CACHE_MANIFEST_SHA256", run.sha256(root / "cache_manifest.json"))
    cfg = run.load_training_config(run.ROOT / "configs/bcg_cwt_patch_mamba_v1/train.yaml",
        [f"data.tf_cache_path={root}", f"data.dataset_root={tmp_path}", "data.index_csv=index.csv",
         "training.device=cpu", "training.epochs=1", "training.batch_size=1", "training.use_amp=false",
         f"outputs.run_root={tmp_path / 'run'}"])
    t = np.arange(18000, dtype=np.float32) / 100
    np.savez(tmp_path / "signals.npz", bcg=np.cos(2 * np.pi * 2 * t).astype(np.float32),
             tho=((1 + .3 * np.sin(2 * np.pi * .015 * t)) * np.sin(2 * np.pi * .25 * t)).astype(np.float32))
    bundles = {}
    for split, row_id, subject in (("train", 11, 1), ("val", 29, 2)):
        rows = pd.DataFrame([{"dataset_row_id": row_id, "split": split, "samp_id": subject,
            "coupling_state_id": 1, "window_start_sample": 0, "window_end_sample": 18000,
            "bcg_signal_key": "bcg", "target_signal_key": "tho", "source_npz": "signals.npz",
            "target_source_npz": "signals.npz", "allowed_losses": "waveform"}])
        dataset = ResearchV2WindowDataset(tmp_path / "index.csv", rows, cfg)
        bundles[split] = SimpleNamespace(rows=rows, dataset=dataset, loader=DataLoader(dataset, batch_size=1))
    pd.concat([b.rows for b in bundles.values()]).to_csv(tmp_path / "index.csv", index=False)
    return cfg, root, SimpleNamespace(**bundles)


def test_default_training_config_preserves_task_semantics():
    from omegaconf import OmegaConf
    cfg = run.load_training_config(run.ROOT / "configs/bcg_cwt_patch_mamba_v1/train.yaml")
    base = OmegaConf.load(run.ROOT / "configs/crd_tf_v1/crd_tf101_m_smoke.yaml")
    for section in ("window", "loss", "evaluation"):
        assert cfg[section] == base[section]
    assert cfg.data.bcg_input_key == base.data.bcg_input_key
    assert cfg.data.target_key == base.data.target_key
    assert cfg.model.initialization_seed == 20261013
    assert cfg.training.early_stopping_enabled is True
    assert cfg.training.early_stopping_min_epoch == 30
    assert cfg.training.early_stopping_patience == 15
    assert cfg.training.early_stopping_min_delta == 0.
    assert (cfg.training.batch_size, cfg.training.gradient_accumulation_steps) == (2, 64)
    assert run.optimizer_updates_per_epoch((10141 + 1) // 2, 64) == 80
    assert cfg.data.max_train_windows is None and cfg.data.max_val_windows is None
    for key, value in (("data.train_split", "test"), ("data.drop_nonfinite_windows", "true"),
                       ("training.amp_dtype", "float16"), ("training.resume", "1"),
                       ("training.early_stopping_enabled", "1"),
                       ("training.early_stopping_min_epoch", "0"),
                       ("training.early_stopping_patience", "0"),
                       ("training.early_stopping_patience", "1.5"),
                       ("training.early_stopping_min_delta", "-0.1"),
                       ("training.early_stopping_min_delta", ".nan"),
                       ("training.early_stopping_min_delta", ".inf")):
        with pytest.raises(ValueError):
            run.load_training_config(run.ROOT / "configs/bcg_cwt_patch_mamba_v1/train.yaml", [f"{key}={value}"])


def test_full_frequency_adapter_and_metadata_identity(fixture_data):
    cfg, root, data = fixture_data
    item = data.train.dataset[0]
    np.testing.assert_array_equal(item["tf"]["w"], np.load(root / "train_w.npy")[0])
    net = mod.build_bcg_cwt_patch_mamba(cfg, mamba_factory=SyntheticMamba)
    np.testing.assert_array_equal(net.frequencies_hz, frozen_w_metadata(root))
    cfg.model.frequencies_hz = (net.frequencies_hz * 1.01).tolist()
    with pytest.raises(ValueError, match="频率身份冲突"):
        mod.build_bcg_cwt_patch_mamba(cfg, mamba_factory=SyntheticMamba)
    with pytest.raises(ValueError, match="train/val"):
        FrozenWCWTReader(root, split="test")
    rows = data.train.rows.copy()
    rows["split"] = "test"
    with pytest.raises(ValueError, match="train/val"):
        ResearchV2WindowDataset(root / "index.csv", rows, cfg)
    cfg.data.drop_nonfinite_windows = True
    with pytest.raises(ValueError, match="drop_nonfinite_windows"):
        ResearchV2WindowDataset(root / "index.csv", data.train.rows, cfg)
    reader = FrozenWCWTReader(root, split="train")
    with pytest.raises(KeyError): reader.get(999)
    corrupted = np.load(root / "train_w.npy", mmap_mode="r+")
    corrupted[0, 0, 0] = np.nan
    corrupted.flush()
    with pytest.raises(FloatingPointError): reader.get(11)
    freq = np.load(root / "w_frequencies_hz.npy")
    freq[0] *= 1.01
    np.save(root / "w_frequencies_hz.npy", freq)
    with pytest.raises(RuntimeError, match="频率文件身份"):
        frozen_w_metadata(root)


@pytest.mark.parametrize("fail", [False, True])
def test_training_lifecycle_with_synthetic_dataset(fixture_data, tmp_path, monkeypatch, fail):
    cfg, root, data = fixture_data
    monkeypatch.setattr(run, "STORAGE", tmp_path)
    monkeypatch.setattr(run, "build_tho_data", lambda cfg: data)
    monkeypatch.setattr(run, "save_sources", fixture_sources)
    original = mod.build_bcg_cwt_patch_mamba
    monkeypatch.setattr(mod, "build_bcg_cwt_patch_mamba", lambda cfg: original(cfg, mamba_factory=SyntheticMamba))
    if fail:
        def interrupted(*args):
            raise RuntimeError("合成中断")
        monkeypatch.setattr(run, "train_epoch", interrupted)
        with pytest.raises(RuntimeError, match="合成中断"):
            run.train(cfg)
        assert (tmp_path / "run/failed.json").is_file()
        assert not (tmp_path / "run/completed.json").exists()
    else:
        result = run.train(cfg)
        assert result["best_epoch"] == 1 and result["status"] == "completed"
        assert result["best_loss_epoch"] == 1
        assert run.sha256(tmp_path / "run/best_loss.pt") == run.sha256(tmp_path / "run/epoch_0001/checkpoint.pt")
        checkpoint = torch.load(tmp_path / "run/epoch_0001/checkpoint.pt", weights_only=False)
        assert checkpoint["history"][0]["optimizer_updates"] == 1
        assert checkpoint["model_state_dict"]["_extra_state"]["architecture"] == mod.DEFAULT_ARCHITECTURE
        frame = pd.read_csv(tmp_path / "run/epoch_0001/validation_metrics.csv")
        assert frame.dataset_row_id.tolist() == [29]
        assert frame.local_rr_mae_bpm.notna().all()
        assert json.loads((tmp_path / "run/model_metadata.json").read_text())["official_mamba"] is False
    with pytest.raises(FileExistsError):
        run.train(cfg)


def scripted_training(fixture_data, tmp_path, monkeypatch, losses, scores):
    """用 epoch 编号权重和给定指标序列隔离验证真实训练入口的选择与停止逻辑。"""
    cfg, _, data = fixture_data
    cfg.training.epochs = len(losses)
    monkeypatch.setattr(run, "STORAGE", tmp_path)
    monkeypatch.setattr(run, "build_tho_data", lambda cfg: data)
    monkeypatch.setattr(run, "save_sources", fixture_sources)

    class EpochMarker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.epoch_marker = torch.nn.Parameter(torch.zeros(()))

        def get_extra_state(self):
            return {"fixture": True}

        def set_extra_state(self, state):
            assert state == {"fixture": True}

    net = EpochMarker()
    monkeypatch.setattr(run, "build_model", lambda cfg: net)

    def advance(model, loader, loss, optimizer, cfg, update, total, epoch):
        with torch.no_grad():
            model.epoch_marker.fill_(epoch)
        return {"loss": .5}, update + 1

    def validation(*args, **kwargs):
        epoch = int(net.epoch_marker.item())
        return {"loss": losses[epoch - 1]}, {"epoch": epoch}

    monkeypatch.setattr(run, "train_epoch", advance)
    monkeypatch.setattr(run, "validate", validation)
    monkeypatch.setattr(run, "evaluate_task_predictions", lambda pred, *args, **kwargs: pd.DataFrame({
        "local_rr_target_eligible": [True], "local_rr_mae_bpm": [scores[pred["epoch"] - 1]]}))
    return cfg


@pytest.mark.parametrize("losses", [[.9, .8, .6, .6], [.9, float("nan")], [.9, float("inf")]])
def test_loss_checkpoint_selection_is_independent(fixture_data, tmp_path, monkeypatch, losses):
    cfg = scripted_training(fixture_data, tmp_path, monkeypatch, losses, [3., 1., 2., 4.])
    cfg.training.early_stopping_enabled = False
    output = tmp_path / "run"
    if not np.isfinite(losses).all():
        with pytest.raises(FloatingPointError, match="validation loss selector"):
            run.train(cfg)
        assert (output / "failed.json").is_file()
        assert not (output / "completed.json").exists()
        assert not (output / "best_loss.pt").exists()
        return
    result = run.train(cfg)
    assert result["epochs"] == 4 and cfg.training.early_stopping_enabled is False
    assert result["best_epoch"] == 2 and result["best_checkpoint"] == "epoch_0002/checkpoint.pt"
    assert result["best_loss_epoch"] == 3 and result["best_val_loss"] == .6
    assert result["best_loss_source_checkpoint"] == "epoch_0003/checkpoint.pt"
    selected = torch.load(output / result["best_loss_checkpoint"], weights_only=False)
    assert selected["model_state_dict"]["epoch_marker"].item() == 3
    assert selected["best_loss_epoch"] == 3 and selected["best_val_loss"] == .6
    assert run.sha256(output / "best_loss.pt") == result["best_loss_checkpoint_sha256"] == run.sha256(output / "epoch_0003/checkpoint.pt")
    last = json.loads((output / "epoch_0004/record.json").read_text())
    assert last["best_epoch"] == 2 and last["best_loss_epoch"] == 3


@pytest.mark.parametrize("scores,min_epoch,patience,delta,enabled,epochs,best_epoch,wait,stopped", [
    # loss 每轮下降而 RR 持平，仍在 RR 耐心耗尽时停止。
    ([3., 1., 1., 1., .5], 1, 2, 0., True, 4, 2, 2, True),
    # RR 改善重置等待；最小轮数之前累计的等待在门槛轮生效。
    ([3., 3., 2., 2., 2., 1.], 1, 2, 0., True, 5, 3, 2, True),
    ([1., 1., 1., 1., 1., .5], 5, 2, 0., True, 5, 1, 4, True),
    # min_delta 仅影响早停参考；RR checkpoint 仍保存严格最小值。
    ([3., 2.875, 2.75, 2.5], 1, 2, .5, True, 3, 3, 2, True),
    ([1., 1., 1., 1.], 1, 2, 0., False, 4, 1, 3, False),
    ([4., 3., 2., 1.], 1, 2, 0., True, 4, 4, 0, False),
])
def test_rr_early_stopping_and_independent_loss_save(
        fixture_data, tmp_path, monkeypatch, scores, min_epoch, patience, delta,
        enabled, epochs, best_epoch, wait, stopped):
    losses = [1. / (i + 1) for i in range(len(scores))]
    cfg = scripted_training(fixture_data, tmp_path, monkeypatch, losses, scores)
    cfg.training.early_stopping_enabled = enabled
    cfg.training.early_stopping_min_epoch = min_epoch
    cfg.training.early_stopping_patience = patience
    cfg.training.early_stopping_min_delta = delta
    result = run.train(cfg)
    output = tmp_path / "run"
    assert result["epochs"] == epochs and result["early_stopped"] is stopped
    assert result["best_epoch"] == best_epoch and result["wait"] == wait
    assert result["best_loss_epoch"] == epochs and result["best_val_loss"] == losses[epochs - 1]
    selected = torch.load(output / result["best_loss_checkpoint"], weights_only=False)
    assert selected["model_state_dict"]["epoch_marker"].item() == epochs
    assert selected["wait"] == wait and selected["stopped"] is stopped
    assert selected["early_stopping_best"] == result["early_stopping_best"]
    assert len(selected["history"]) == epochs
    assert len(list(output.glob("epoch_*"))) == epochs
    assert not (output / f"epoch_{epochs + 1:04d}").exists()
    assert run.sha256(output / "best_loss.pt") == run.sha256(output / f"epoch_{epochs:04d}/checkpoint.pt")
    if delta:
        assert result["early_stopping_best"] == 3.
        assert result["best_local_rr_mae"] == 2.75


def fixture_sources(output):
    name = "resp_train/bcg_cwt_training.py"
    run.write_json(output / "source_manifest.json", {"files": {name: run.sha256(run.ROOT / name)}})


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


class SmallStochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Conv1d(1, 4, 1), torch.nn.GELU(),
                                       torch.nn.Dropout(.2), torch.nn.Conv1d(4, 1, 1))

    def forward(self, x, *, tf):
        return {"waveform": self.net(x)}

    def get_extra_state(self):
        return {"fixture": True}

    def set_extra_state(self, state):
        assert state == {"fixture": True}


@pytest.mark.parametrize("device,interruption", [("cpu", 1), ("cpu", 3), ("cuda:0", 3)])
def test_resume_matches_uninterrupted_optimizer_rng_and_stopping(fixture_data, tmp_path, monkeypatch, device, interruption):
    from omegaconf import OmegaConf
    if device.startswith("cuda"):
        if os.environ.get("BCG_CWT_RUN_CUDA") != "1" or not torch.cuda.is_available():
            pytest.skip("CUDA 恢复检查要求 BCG_CWT_RUN_CUDA=1 且 CUDA 可用")
        if not torch.cuda.is_bf16_supported():
            pytest.skip("CUDA 恢复检查要求 bfloat16 支持")
    cfg, _, data = fixture_data
    cfg.training.device = device
    cfg.training.use_amp = device.startswith("cuda")
    cfg.training.epochs = 5
    cfg.training.early_stopping_min_epoch = 1
    cfg.training.early_stopping_patience = 2
    monkeypatch.setattr(run, "STORAGE", tmp_path)
    monkeypatch.setattr(run, "build_tho_data", lambda cfg: data)
    monkeypatch.setattr(run, "build_model", lambda cfg: SmallStochasticModel())
    monkeypatch.setattr(run, "save_sources", fixture_sources)
    monkeypatch.setattr(run, "evaluate_task_predictions", lambda *args, **kwargs: pd.DataFrame({
        "local_rr_target_eligible": [True], "local_rr_mae_bpm": [1.]}))
    actual_train_epoch = run.train_epoch

    def random_epoch(*args):
        # 同时覆盖三类 RNG；真实任务循环使用 dropout 和 DataLoader 的 RNG。
        random.random()
        np.random.random()
        return actual_train_epoch(*args)

    monkeypatch.setattr(run, "train_epoch", random_epoch)
    cfg.outputs.run_root = str(tmp_path / "full")
    full_result = run.train(cfg)
    full_state = torch.load(tmp_path / "full/epoch_0003/checkpoint.pt", weights_only=False)
    partial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    partial_cfg.outputs.run_root = str(tmp_path / "partial")

    def interrupt(*args):
        result = random_epoch(*args)
        if args[-1] == interruption:
            raise RuntimeError("epoch 内中断")
        return result

    monkeypatch.setattr(run, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="epoch 内中断"):
        run.train(partial_cfg)
    old_failure = (tmp_path / "partial/failed.json").read_bytes()
    old_checkpoints = {p: run.sha256(p / "checkpoint.pt") for p in run.completed_epochs(tmp_path / "partial").values()}
    partial_cfg = run.load_training_config(tmp_path / "partial/resolved_config.yaml", ["training.resume=true"])
    monkeypatch.setattr(run, "train_epoch", random_epoch)
    resumed = run.train(partial_cfg)
    completed = run.completed_epochs(tmp_path / "partial")
    last = completed[3]
    assert "retry" in completed[interruption].name
    resumed_state = torch.load(last / "checkpoint.pt", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "rng", "update_index", "total_updates",
                "best_epoch", "best_local_rr", "best_val_loss", "best_loss_epoch", "wait", "stopped", "early_stopping_best"):
        assert_tree_equal(full_state[key], resumed_state[key])
    assert full_result["epochs"] == resumed["epochs"] == 3
    assert full_result["early_stopped"] is resumed["early_stopped"] is True
    assert full_result["best_loss_epoch"] == resumed["best_loss_epoch"]
    assert (tmp_path / "partial/failed.json").read_bytes() == old_failure
    assert all(run.sha256(p / "checkpoint.pt") == digest for p, digest in old_checkpoints.items())
    assert (tmp_path / f"partial/epoch_{interruption:04d}").is_dir()
    assert len(list((tmp_path / "partial/attempts").glob("*/started.json"))) == 1
    monkeypatch.setattr(run, "build_tho_data", lambda cfg: pytest.fail("已完成运行不能重新读取数据或训练"))
    assert run.train(partial_cfg) == resumed


@pytest.mark.parametrize("change", ["config", "checkpoint", "source", "rows"])
def test_resume_rejects_changed_identity(fixture_data, tmp_path, monkeypatch, change):
    cfg = scripted_training(fixture_data, tmp_path, monkeypatch, [.8, .7, .6], [2., 1., 1.])
    original = run.train_epoch
    def interrupt(*args):
        if args[-1] == 2:
            raise RuntimeError("合成中断")
        return original(*args)
    monkeypatch.setattr(run, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="合成中断"):
        run.train(cfg)
    output = tmp_path / "run"
    cfg.training.resume = True
    expected = "配置"
    if change == "config":
        cfg.training.gradient_accumulation_steps = 32
    elif change == "checkpoint":
        with (output / "epoch_0001/checkpoint.pt").open("ab") as stream:
            stream.write(b"fixture corruption")
        expected = "SHA-256"
    elif change == "source":
        manifest = json.loads((output / "source_manifest.json").read_text())
        manifest["files"]["resp_train/bcg_cwt_training.py"] = "bad"
        (output / "source_manifest.json").write_text(json.dumps(manifest))
        expected = "源码身份"
    else:
        fixture_data[2].train.rows.loc[0, "coupling_state_id"] = 2
        expected = "行身份"
    with pytest.raises(ValueError, match=expected):
        run.train(cfg)
    assert len(list((output / "attempts").glob("*/failed.json"))) == 1
    assert (output / "failed.json").is_file()


def test_resume_finalization_after_early_stop_does_not_train(fixture_data, tmp_path, monkeypatch):
    cfg = scripted_training(fixture_data, tmp_path, monkeypatch, [.9, .8, .7, .6], [1., 1., 1., 1.])
    cfg.training.early_stopping_min_epoch = 1
    cfg.training.early_stopping_patience = 2
    original_copy = run.shutil.copyfileobj
    def interrupted_copy(*args):
        raise RuntimeError("导出阶段中断")
    monkeypatch.setattr(run.shutil, "copyfileobj", interrupted_copy)
    with pytest.raises(RuntimeError, match="导出阶段中断"):
        run.train(cfg)
    output = tmp_path / "run"
    assert len(run.completed_epochs(output)) == 3
    cfg.training.resume = True
    monkeypatch.setattr(run.shutil, "copyfileobj", original_copy)
    monkeypatch.setattr(run, "train_epoch", lambda *args: pytest.fail("已早停不能继续训练"))
    result = run.train(cfg)
    assert result["early_stopped"] and result["epochs"] == 3
    assert result["best_loss_epoch"] == 3
    assert run.sha256(output / "best_loss.pt") == run.sha256(output / "epoch_0003/checkpoint.pt")


def test_metric_reuse_matches_original_selector_and_checks_eligibility(fixture_data, monkeypatch):
    from resp_train.metrics import task
    cfg, _, data = fixture_data
    target = data.val.dataset[0]["target"].numpy()
    t = np.arange(18000, dtype=np.float32) / 100
    other_target = ((1 + .3 * np.sin(2 * np.pi * .015 * t)) * np.sin(2 * np.pi * .35 * t))[None]
    predictions = {"r_tho_hat": np.stack([target * .8, target, target]),
                   "tho_ref": np.stack([target, other_target, np.zeros_like(target)])}
    calls = []
    original = task._local_rr
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(task, "_local_rr", counted)
    metrics = task.evaluate_task_predictions(predictions, cfg)
    assert len(calls) == 3
    score = run.local_rr_from_metrics(metrics)
    assert len(calls) == 3  # 聚合只读表格，不再运行逐窗口信号处理。
    assert score == task.validation_local_rr_mean(predictions, cfg)
    assert metrics.local_rr_target_eligible.tolist() == [True, True, False]
    assert score > 0.
    metrics.loc[0, "local_rr_mae_bpm"] = np.nan
    with pytest.raises(FloatingPointError): run.local_rr_from_metrics(metrics)
    with pytest.raises(ValueError): run.local_rr_from_metrics(metrics.iloc[2:])


def test_accumulation_2_by_64_performs_one_correct_update(fixture_data):
    cfg, _, data = fixture_data
    cfg.training.gradient_accumulation_steps = 64
    batch = next(iter(data.train.loader))
    batch["x"] = batch["x"].repeat(2, 1, 1)
    batch["target"] = batch["target"].repeat(2, 1, 1)
    batch["tf"]["w"] = batch["tf"]["w"].repeat(2, 1, 1)
    model = SmallStochasticModel()
    model.net[2].p = 0.
    from copy import deepcopy
    reference = deepcopy(model)
    loss = run.RespirationTaskLoss(cfg)
    optimizer, _ = run.build_crd_optimizer(model, cfg)
    other_optimizer, _ = run.build_crd_optimizer(reference, cfg)
    _, update = run.train_epoch(model, [batch] * 64, loss, optimizer, cfg, 0, 6400, 1)
    cfg.training.gradient_accumulation_steps = 1
    _, other_update = run.train_epoch(reference, [batch], loss, other_optimizer, cfg, 0, 6400, 1)
    assert update == other_update == 1
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=1e-6)
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
    assert all(int(state["step"]) == 1 for state in optimizer.state.values())


def test_run_mutex_rejects_concurrent_resume(tmp_path):
    with run.run_mutex(tmp_path):
        with pytest.raises(RuntimeError, match="已有训练进程"):
            with run.run_mutex(tmp_path):
                pytest.fail("第二个运行不应获得锁")
    with run.run_mutex(tmp_path):
        pass
