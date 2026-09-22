from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import resp_train.aligned_dual_view.data as data_module
import resp_train.aligned_dual_view.experiment as experiment
from resp_train.aligned_dual_view.artifacts import (
    artifact_directory, read_json, sha256_file, verified_manifest, write_json,
)
from resp_train.aligned_dual_view.config import CORE_VIEWS, PROTOCOL, SEEDS, load_experiment_config, model_config
from resp_train.aligned_dual_view.data import CacheReader, build_cache, build_loaders, sample_identity, select_rows
from resp_train.aligned_dual_view.experiment import PRIMARY, RuntimeModel, evaluate_validation, train
from resp_train.aligned_dual_view.summary import summarize_runs
from resp_train.metrics.task import summarize_task_metrics


class ContractMamba(nn.Module):
    """管线接线替身；原生 Mamba GPU 验收通过独立入口执行。"""
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(kwargs["d_model"]))

    def forward(self, value):
        return value * self.scale


def model_factory(config):
    return RuntimeModel(config, mamba_factory=ContractMamba)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def cfg(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "training").mkdir(parents=True)
    records = []
    time = np.arange(19000, dtype=np.float64) / 100
    for split, subject in (("train", 11), ("val", 22)):
        waveform = (0.2 * np.sin(2 * np.pi * 0.2 * time)
                    + (1 + 0.3 * np.sin(2 * np.pi * 0.2 * time)) * np.sin(2 * np.pi * 4 * time)).astype(np.float32)
        target = ((1 + 0.3 * np.sin(2 * np.pi * 0.015 * time))
                  * np.sin(2 * np.pi * 0.2 * time + 0.1)).astype(np.float32)
        np.savez(dataset / f"{split}_input.npz", bcg=waveform)
        np.savez(dataset / f"{split}_target.npz", target=target, tho_bad_sec=np.zeros(190, dtype=np.uint8))
        for start in (0, 10):
            records.append({
                "dataset_row_id": len(records) + 1, "split": split, "samp_id": subject,
                "coupling_state_id": 1, "window_start_s": start, "window_end_s": start + 180,
                "source_npz": f"../{split}_input.npz", "target_source_npz": f"../{split}_target.npz",
                "bcg_rawish_segment_soft_z_key": "bcg", "target_waveform_segment_soft_z_key": "target",
                "hard_valid_ratio": 1.0, "state_alignment_valid_ratio": 1.0,
                "allowed_losses": "waveform", "state_alignment_method": "constant_shift", "reason": "",
            })
    records.append({**records[0], "dataset_row_id": 99, "split": "test", "samp_id": 33,
                    "source_npz": "../must_not_open_test.npz", "target_source_npz": "../must_not_open_test_target.npz"})
    pd.DataFrame(records).to_csv(dataset / "training/dataset_index.csv", index=False)
    return load_experiment_config(overrides=[
        "protocol.run_role=smoke", f"data.dataset_root={dataset}",
        "data.max_train_windows=2", "data.max_val_windows=2", "training.epochs=1",
        "training.batch_size=1", "training.gradient_accumulation_steps=2", "training.device=cpu",
        "training.use_amp=false", "training.show_progress=false",
    ])


@pytest.fixture
def cheap_transform(monkeypatch):
    # 数值 CWT 已在模型测试中验证；这里隔离哈希、I/O、生命周期和对齐逻辑。
    monkeypatch.setattr(data_module, "extract_cwt_10hz", lambda x: np.repeat(x[None, ::10], 97, axis=0).copy())


def test_config_rejects_test_loss_and_partial_formal_before_data_access(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: pytest.fail("配置拒绝前不应读数据"))
    for override in ("data.val_split=test", "loss.effort_weight=0.5", "data.max_val_windows=2",
                     "training.seed=7", "training.epochs=1", "training.gradient_accumulation_steps=1"):
        with pytest.raises(ValueError):
            load_experiment_config(overrides=[override])
    with pytest.raises(Exception, match="test_split"):
        load_experiment_config(overrides=["data.test_split=test"])


def test_cache_build_reads_only_input_and_cannot_overwrite(cfg, tmp_path, cheap_transform, monkeypatch):
    original = np.load
    accessed = []
    def checked_load(path, *args, **kwargs):
        accessed.append(str(path))
        assert "target.npz" not in str(path) and "must_not_open" not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", checked_load)
    output = build_cache(cfg, tmp_path / "cache")
    manifest = verified_manifest(output, "cache")
    assert manifest["target_read"] is False and manifest["test_waveform_read"] is False
    assert len(accessed) == 2
    before = sha256_file(output / "manifest.json")
    with pytest.raises(FileExistsError):
        build_cache(cfg, output)
    assert sha256_file(output / "manifest.json") == before


def test_cache_pairs_row_input_and_feature_content(cfg, tmp_path, cheap_transform):
    root = build_cache(cfg, tmp_path / "cache")
    selection, loaders = build_loaders(cfg, root)
    dataset = loaders["train"].dataset
    item = dataset[0]
    assert item["tf"]["adv_cwt"].shape == (97, 1800)
    reader = dataset.reader
    identity = sample_identity(selection.rows["train"].iloc[0])
    with pytest.raises(ValueError, match="输入内容错配"):
        reader.get(0, identity=identity, waveform=item["x"] + 1, include_features=True)
    with pytest.raises(ValueError, match="identity"):
        reader.get(1, identity=identity, waveform=item["x"], include_features=True)
    values = np.load(root / "train_cwt.npy", mmap_mode="r+")
    values[0, 0, 0] += 0.25
    values.flush()
    with pytest.raises(ValueError, match="特征内容损坏"):
        dataset[0]
    with pytest.raises(ValueError, match="train/val"):
        CacheReader(root, selection, split="test")


def test_cache_failure_and_incomplete_artifacts_stay_failed(cfg, tmp_path, monkeypatch):
    def broken(_):
        raise FloatingPointError("fixture transform failure")
    monkeypatch.setattr(data_module, "extract_cwt_10hz", broken)
    output = tmp_path / "failed_cache"
    with pytest.raises(FloatingPointError):
        build_cache(cfg, output)
    assert (output / "started.json").exists() and (output / "failed.json").exists()
    assert not (output / "completed.json").exists()
    with pytest.raises(ValueError, match="失败"):
        verified_manifest(output, "cache")
    with pytest.raises(FileExistsError):
        build_cache(cfg, output)


def test_subject_overlap_and_index_drift_are_rejected(cfg, tmp_path, cheap_transform):
    root = build_cache(cfg, tmp_path / "cache")
    path = Path(cfg.data.dataset_root) / cfg.data.index_csv
    frame = pd.read_csv(path)
    frame.loc[frame.split == "val", "samp_id"] = 11
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="隔离"):
        select_rows(cfg)
    frame.loc[frame.split == "val", "samp_id"] = 23
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="身份不匹配"):
        CacheReader(root, select_rows(cfg), split="train")


def test_actual_cwt_cache_train_select_reload_and_validation(cfg, tmp_path):
    cache = build_cache(cfg, tmp_path / "cache")
    cfg.training.epochs = 2
    run = train(cfg, cache_root=cache, output=tmp_path / "train", model_factory=model_factory)
    manifest = verified_manifest(run, "train")
    assert manifest["epochs_completed"] == 2
    assert manifest["test_waveform_read"] is False
    history = pd.read_csv(run / "history.csv")
    assert history.optimizer_update.tolist() == [1, 2]
    assert manifest["selected_epoch"] == int(np.argmin(history.val_local_rr_mae.to_numpy())) + 1
    assert set(read_json(run / "samples.json")) == {"train", "val"}
    assert (run / manifest["selected_checkpoint"]).is_file()
    output = evaluate_validation(run_root=run, cache_root=cache, output=tmp_path / "evaluation", model_factory=model_factory)
    verified_manifest(output, "validation")
    assert set(read_json(output / "samples.json")) == {"val"}
    pd.testing.assert_frame_equal(pd.read_csv(run / "metrics.csv"), pd.read_csv(output / "metrics.csv"))
    with pytest.raises(FileExistsError):
        train(cfg, cache_root=cache, output=run, model_factory=model_factory)
    # 新进程式重建 dataset 后，目标内容变化必须被训练时 provenance 拦住。
    target_path = Path(cfg.data.dataset_root) / "val_target.npz"
    with np.load(target_path) as content:
        target, mask = content["target"].copy(), content["tho_bad_sec"].copy()
    np.savez(target_path, target=target + 0.1, tho_bad_sec=mask)
    with pytest.raises(ValueError, match="target/mask"):
        evaluate_validation(run_root=run, cache_root=cache, output=tmp_path / "drift", model_factory=model_factory)
    assert (tmp_path / "drift/failed.json").is_file()


def test_waveform_control_loader_has_no_tf_features(cfg, tmp_path, cheap_transform):
    cache = build_cache(cfg, tmp_path / "cache")
    cfg.model.input_view = "waveform"
    _, loaders = build_loaders(cfg, cache)
    batch = next(iter(loaders["train"]))
    assert "tf" not in batch
    model = model_factory(model_config(cfg))
    output = model(batch["x"])
    assert output["waveform"].shape == (1, 1, 18000)


def test_training_failure_preserves_lifecycle(cfg, tmp_path, cheap_transform, monkeypatch):
    cache = build_cache(cfg, tmp_path / "cache")
    def broken(*args, **kwargs):
        raise FloatingPointError("fixture gradient failure")
    monkeypatch.setattr(experiment, "train_crd_one_epoch", broken)
    with pytest.raises(FloatingPointError):
        train(cfg, cache_root=cache, output=tmp_path / "failed_train", model_factory=model_factory)
    assert (tmp_path / "failed_train/failed.json").is_file()
    assert not (tmp_path / "failed_train/completed.json").exists()


def test_checkpoint_nonfinite_is_rejected_before_loading(cfg, tmp_path):
    model = model_factory(model_config(cfg))
    state = copy.deepcopy(model.state_dict())
    state["readout.weight"].fill_(float("nan"))
    path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": state, "identity": {}, "resume_supported": False}, path)
    with pytest.raises(FloatingPointError, match="checkpoint"):
        experiment._load_checkpoint(path, model=model, identity={})


def test_output_paths_protect_dataset_and_existing_artifacts(cfg, tmp_path, cheap_transform):
    with pytest.raises(ValueError, match="数据源"):
        build_cache(cfg, Path(cfg.data.dataset_root) / "output")
    cache = build_cache(cfg, tmp_path / "cache")
    with pytest.raises(ValueError, match="补写"):
        build_cache(cfg, cache / "another_run")


def test_transform_drift_and_runtime_wrong_feature_kind_fail(cfg, tmp_path, cheap_transform, monkeypatch):
    cache = build_cache(cfg, tmp_path / "cache")
    selection = select_rows(cfg)
    monkeypatch.setattr(data_module, "transform_identity", lambda: {"changed": True})
    with pytest.raises(ValueError, match="前处理实现"):
        CacheReader(cache, selection, split="val")
    model = model_factory(model_config(cfg))
    with pytest.raises(ValueError, match="adv_cwt"):
        model(torch.zeros(1, 1, 18000), tf={"w": torch.zeros(1, 97, 360)})


def _summary_fixture(root, view, seed, comparison="same"):
    cfg = load_experiment_config(overrides=[f"model.input_view={view}", f"training.seed={seed}"])
    with artifact_directory(root, kind="train", cfg=cfg) as (root, _):
        write_json(root / "samples.json", {"fixture": "synthetic summary structure"})
        (root / "checkpoints").mkdir()
        for epoch in (1, 80):
            (root / f"checkpoints/epoch_{epoch:03d}.pt").write_bytes(b"disposable checksum fixture")
        history = pd.DataFrame({"epoch": range(1, 81), "val_local_rr_mae": [1.] * 80})
        history.to_csv(root / "history.csv", index=False)
        metrics = pd.DataFrame({key: [1., 2.] for key in PRIMARY})
        metrics["lag_aware_signed_pcc"] = [0.2, 0.4]
        for key in ("whole_rr_target_eligible", "local_rr_target_eligible", "joint_target_eligible"):
            metrics[key] = True
        metrics.to_csv(root / "metrics.csv", index=False)
        summarize_task_metrics(metrics).to_csv(root / "metrics_summary.csv", index=False)
        write_json(root / "manifest.json", {
            "protocol": PROTOCOL, "kind": "train", "view": view, "seed": seed, "run_role": "formal",
            "epochs_completed": 80, "selected_epoch": 1, "comparison_id": comparison, "test_waveform_read": False,
            "metrics_sha256": sha256_file(root / "metrics.csv"),
            "summary_sha256": sha256_file(root / "metrics_summary.csv"),
            "history_sha256": sha256_file(root / "history.csv"), "config_sha256": sha256_file(root / "config.yaml"),
            "samples_sha256": sha256_file(root / "samples.json"),
            "selected_checkpoint": "checkpoints/epoch_001.pt", "selected_sha256": sha256_file(root / "checkpoints/epoch_001.pt"),
            "final_checkpoint": "checkpoints/epoch_080.pt", "final_sha256": sha256_file(root / "checkpoints/epoch_080.pt"),
        })
    return root


def test_summary_requires_complete_matrix_and_preserves_five_metrics(tmp_path):
    roots = [_summary_fixture(tmp_path / f"{view}_{seed}", view, seed) for view in CORE_VIEWS for seed in SEEDS]
    with pytest.raises(ValueError, match="完整"):
        summarize_runs(roots[:-1], output=tmp_path / "partial")
    with pytest.raises(ValueError, match="不同"):
        summarize_runs([roots[0]] * 9, output=tmp_path / "duplicates")
    output = summarize_runs(roots, output=tmp_path / "summary")
    manifest = verified_manifest(output, "summary")
    assert manifest["views"] == list(CORE_VIEWS)
    scores = pd.read_csv(output / "per_seed.csv")
    assert len(scores) == 9 and set(PRIMARY).issubset(scores.columns)
    aggregate = pd.read_csv(output / "across_seed.csv")
    assert len(aggregate) == 15 and (aggregate.n_seeds == 3).all()
    mismatched = _summary_fixture(tmp_path / "different", "joint", SEEDS[0], comparison="other")
    with pytest.raises(ValueError, match="不一致"):
        summarize_runs([mismatched, *roots[1:]], output=tmp_path / "mixed")
