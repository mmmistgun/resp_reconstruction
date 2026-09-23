from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from resp_eval.fusion_test import artifacts, contract, data, evaluate, summary
from resp_fusion.artifacts import code_identity, read_json, sha256_file, write_json
from resp_fusion.config import DEFAULT_CONFIG, model_config
from resp_fusion.experiment import RuntimeModel
from resp_train.aligned_dual_view.features import spec_digest
from resp_train.data.cache import WholeNightCache


class ContractMamba(nn.Module):
    """CPU 接线替身，不模拟原生 CUDA 状态空间计算。"""
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(kwargs["d_model"]))

    def forward(self, value):
        return value * self.scale


def fixture_model(cfg):
    return RuntimeModel(cfg, mamba_factory=ContractMamba)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "training").mkdir(parents=True)
    time = np.arange(19000) / 100
    wave = (np.sin(2 * np.pi * .2 * time) + .3 * np.sin(2 * np.pi * 4 * time)).astype(np.float32)
    target = ((1 + .3 * np.sin(2 * np.pi * .015 * time))
              * np.sin(2 * np.pi * .2 * time + .1)).astype(np.float32)
    np.savez(dataset / "test_input.npz", bcg=wave)
    np.savez(dataset / "test_target.npz", target=target, tho_bad_sec=np.zeros(190, dtype=np.uint8))
    rows = []
    for split, subject, start in (("train", 11, 0), ("val", 22, 0), ("test", 33, 0), ("test", 33, 10)):
        rows.append({
            "dataset_row_id": len(rows) + 1, "split": split, "samp_id": subject,
            "coupling_state_id": 1, "window_start_s": start, "window_end_s": start + 180,
            "source_npz": f"../{split}_input.npz", "target_source_npz": f"../{split}_target.npz",
            "bcg_rawish_segment_soft_z_key": "bcg", "target_waveform_segment_soft_z_key": "target",
            "hard_valid_ratio": 1., "state_alignment_valid_ratio": 1., "allowed_losses": "waveform",
            "state_alignment_method": "constant_shift", "reason": "",
        })
    index = dataset / "training/dataset_index.csv"
    pd.DataFrame(rows).to_csv(index, index=False)
    # 生产锁仅作为结构模板；所有可访问数据和权重都替换为 disposable fixture。
    lock = copy.deepcopy(read_json(contract.LOCK_PATH))
    lock["dataset_index_sha256"] = sha256_file(index)
    lock["test_count"], lock["test_subject_count"] = 2, 1
    lock["test_row_ids_sha256"] = hashlib.sha256(np.asarray([3, 4], dtype="<i8").tobytes()).hexdigest()
    configs, checkpoints = {}, tmp_path / "checkpoints"
    checkpoints.mkdir()
    for name, entry in lock["entries"].items():
        cfg = OmegaConf.load(DEFAULT_CONFIG)
        cfg.training.seed = entry["seed"]
        OmegaConf.resolve(cfg)
        cfg.model.arm = entry["arm"]
        cfg.data.dataset_root = str(dataset)
        cfg.training.device = "cpu"
        cfg.training.batch_size = 1
        cfg.training.use_amp = False
        cfg.training.show_progress = False
        configs[name] = cfg
        model = fixture_model(model_config(cfg))
        checkpoint = checkpoints / f"{name}.pt"
        torch.save({"model_state_dict": model.state_dict(), "epoch": entry["selected_epoch"],
                    "identity": entry["training_identity"], "resume_supported": False}, checkpoint)
        entry["checkpoint"] = checkpoint.name
        entry["checkpoint_sha256"] = sha256_file(checkpoint)
    lock["lock_id"] = spec_digest({key: value for key, value in lock.items() if key != "lock_id"})
    lock_path = tmp_path / "fixture_lock.json"
    write_json(lock_path, lock)
    monkeypatch.setattr(contract, "LOCK_SHA", sha256_file(lock_path))
    def source_config(_lock, name):
        if name not in configs:
            raise ValueError("checkpoint 不在冻结清单内")
        return copy.deepcopy(configs[name])
    for module in (data, evaluate, summary):
        monkeypatch.setattr(module, "source_config", source_config)
    monkeypatch.setattr(evaluate, "repo_path", lambda name: checkpoints / name)
    # 此处隔离 I/O 契约；实际变换在单独集成用例中恢复原实现。
    actual_transform = data.extract_cwt_10hz
    monkeypatch.setattr(data, "extract_cwt_10hz", lambda x: np.repeat(x[None, ::10], 97, axis=0).copy())
    return SimpleNamespace(root=tmp_path, dataset=dataset, index=index, lock=lock, lock_path=lock_path,
                           configs=configs, checkpoints=checkpoints, actual_transform=actual_transform)


def build_cache(fixture, name="cache"):
    return data.build_test_cache(fixture.root / name, lock_path=fixture.lock_path, confirm_research_test=True)


def run_evaluation(fixture, cache, name="A_seed20260811", **kwargs):
    return evaluate.evaluate_checkpoint(name, cache_root=cache, lock_path=fixture.lock_path, device="cpu",
                                       confirm_research_test=True, show_progress=False,
                                       model_factory=kwargs.pop("model_factory", fixture_model), **kwargs)


def test_real_candidate_lock_and_training_identity_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("候选准备不得读数组"))
    monkeypatch.setattr(data, "read_research_v2_index", lambda *a, **k: pytest.fail("不得读数据集 index"))
    lock = contract.load_lock()
    assert len(lock["entries"]) == 18
    assert code_identity()["sha256"] == contract.TRAINING_CODE_SHA
    recreated = contract.prepare_lock(tmp_path / "candidate_copy.json")
    assert sha256_file(recreated) == contract.LOCK_SHA
    with pytest.raises(FileExistsError):
        contract.prepare_lock(recreated)


def test_confirmation_precedes_all_data_access(tmp_path, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("缺少授权时不得读数组"))
    monkeypatch.setattr(data, "read_research_v2_index", lambda *a, **k: pytest.fail("缺少授权时不得读 index"))
    for call in (
        lambda: data.build_test_cache(tmp_path / "cache"),
        lambda: evaluate.evaluate_checkpoint("A_seed20260811", cache_root=tmp_path / "cache"),
        lambda: summary.summarize_test(tmp_path / "cache", tmp_path / "summary"),
    ):
        with pytest.raises(PermissionError, match="confirm-research-test"):
            call()
    assert not list(tmp_path.iterdir())


def test_lock_tampering_and_nonallowlisted_entry_fail(fixture):
    payload = copy.deepcopy(fixture.lock)
    payload["entries"].pop(next(iter(payload["entries"])))
    path = fixture.root / "tampered.json"
    write_json(path, payload)
    with pytest.raises(ValueError, match="哈希"):
        contract.load_lock(path)
    with pytest.raises(ValueError, match="清单"):
        run_evaluation(fixture, fixture.root / "absent", "A_seed999")
    with pytest.raises(ValueError, match="工作树"):
        contract.repo_path("../../outside")


def test_input_only_cache_and_target_after_complete_inference(fixture, monkeypatch):
    monkeypatch.setattr(data, "extract_cwt_10hz", fixture.actual_transform)
    events = []
    original = WholeNightCache.get_arrays
    def watched(self, source, keys):
        if "target" in keys or any(key.endswith("_sec") for key in keys):
            events.append("target")
        return original(self, source, keys)
    monkeypatch.setattr(WholeNightCache, "get_arrays", watched)
    cache = build_cache(fixture)
    assert events == []
    manifest = artifacts.verified_artifact(cache, kind="test_cache", lock=fixture.lock)
    assert manifest["target_read"] is False and manifest["model_inference_used"] is False
    def counted(config):
        model = fixture_model(config)
        model.register_forward_hook(lambda *args: events.append("forward"))
        return model
    output = run_evaluation(fixture, cache, model_factory=counted)
    assert events[:2] == ["forward", "forward"] and events[2] == "target"
    manifest = artifacts.verified_artifact(output, kind="test_evaluation", lock=fixture.lock)
    assert manifest["selected_epoch"] == fixture.lock["entries"]["A_seed20260811"]["selected_epoch"]
    assert manifest["inference_completed_before_target_read"] is True
    assert manifest["checkpoint_reselected"] is False
    metrics = pd.read_csv(output / "metrics.csv")
    assert metrics.dataset_row_id.tolist() == [3, 4] and set(metrics.split) == {"test"}
    assert np.isfinite(metrics.respiratory_band_coherence).all()
    assert np.isfinite(metrics.constrained_ndtw).all()
    with pytest.raises(FileExistsError, match="成功评价"):
        run_evaluation(fixture, cache, attempt="r2")


def test_post_attention_and_partial_matrix_guard(fixture):
    cache = build_cache(fixture)
    output = run_evaluation(fixture, cache, "F_seed20260811")
    assert artifacts.verified_artifact(output, kind="test_evaluation", lock=fixture.lock)["arm"] == "F"
    with pytest.raises(ValueError, match="成功评价"):
        summary.summarize_test(cache, fixture.root / "partial", lock_path=fixture.lock_path, confirm_research_test=True)
    assert (fixture.root / "partial/failed.json").is_file()


def test_cache_corruption_and_input_drift_fail(fixture):
    cache = build_cache(fixture)
    cfg = fixture.configs["A_seed20260811"]
    _, _, identities = data.select_test_rows(cfg, fixture.lock, confirmed=True)
    reader = data.TestCache(cache, fixture.lock, identities)
    source = WholeNightCache(fixture.index)
    waveform = data.read_input(source, identities[0])
    with pytest.raises(ValueError, match="波形内容"):
        reader.get(0, waveform + .2, use_cwt=True)
    values = np.load(cache / "test_cwt.npy", mmap_mode="r+")
    values[0, 0, 0] += .3
    values.flush()
    with pytest.raises(ValueError, match="哈希"):
        data.TestCache(cache, fixture.lock, identities)
    with pytest.raises(FileExistsError):
        build_cache(fixture)


def test_split_leakage_and_row_hash_mismatch_fail(fixture):
    rows = pd.read_csv(fixture.index)
    rows.loc[rows.split == "test", "samp_id"] = 11
    rows.to_csv(fixture.index, index=False)
    lock = copy.deepcopy(fixture.lock)
    lock["dataset_index_sha256"] = sha256_file(fixture.index)
    with pytest.raises(ValueError, match="隔离"):
        data.select_test_rows(fixture.configs["A_seed20260811"], lock, confirmed=True)
    rows.loc[rows.split == "test", "samp_id"] = 33
    rows.loc[rows.dataset_row_id == 4, "dataset_row_id"] = 10
    rows.to_csv(fixture.index, index=False)
    lock["dataset_index_sha256"] = sha256_file(fixture.index)
    with pytest.raises(ValueError, match="row-ID"):
        data.select_test_rows(fixture.configs["A_seed20260811"], lock, confirmed=True)


def test_checkpoint_drift_fails_before_test_index_access(fixture, monkeypatch):
    cache = build_cache(fixture)
    path = fixture.checkpoints / "A_seed20260811.pt"
    with path.open("ab") as handle:
        handle.write(b"corrupted disposable fixture")
    monkeypatch.setattr(evaluate, "select_test_rows", lambda *a, **k: pytest.fail("坏 checkpoint 不得打开 test index"))
    with pytest.raises(ValueError, match="checkpoint 哈希"):
        run_evaluation(fixture, cache)
    failed = evaluate.evaluation_root(cache) / "A_seed20260811/attempt_r1/failed.json"
    assert failed.is_file()


def test_nonfinite_prediction_stops_before_targets_and_keeps_retry(fixture, monkeypatch):
    cache = build_cache(fixture)
    original = WholeNightCache.get_arrays
    target_calls = []
    def watched(self, source, keys):
        if "target" in keys:
            target_calls.append(source)
        return original(self, source, keys)
    monkeypatch.setattr(WholeNightCache, "get_arrays", watched)
    def broken(config):
        model = fixture_model(config)
        model.register_forward_hook(lambda module, args, output: {"waveform": output["waveform"] * float("nan")})
        return model
    with pytest.raises(FloatingPointError, match="prediction"):
        run_evaluation(fixture, cache, model_factory=broken)
    assert not target_calls
    parent = evaluate.evaluation_root(cache) / "A_seed20260811"
    assert (parent / "attempt_r1/failed.json").is_file()
    output = run_evaluation(fixture, cache, attempt="r2")
    assert (output / "completed.json").is_file() and (parent / "attempt_r1/failed.json").is_file()


def test_complete_test_matrix_summary(fixture):
    cache = build_cache(fixture)
    for name in fixture.lock["entries"]:
        run_evaluation(fixture, cache, name)
    output = summary.summarize_test(cache, fixture.root / "summary", lock_path=fixture.lock_path,
                                    confirm_research_test=True)
    manifest = artifacts.verified_artifact(output, kind="test_summary", lock=fixture.lock)
    assert len(manifest["sources"]) == 18
    scores = pd.read_csv(output / "per_seed.csv")
    assert len(scores) == 18 and (scores.n_samples == 2).all()
    across = pd.read_csv(output / "across_seed.csv")
    assert (across.n_seeds == 3).all()
    assert set(("respiratory_band_coherence", "constrained_ndtw")).issubset(across.metric)
    assert len(pd.read_csv(output / "contrasts_per_seed.csv")) == 240
    assert len(pd.read_csv(output / "contrasts_across_seed.csv")) == 80
    from resp_fusion.summary import factorial_contrasts
    from resp_fusion.experiment import PRIMARY
    primary = scores[["arm", "seed", *[f"{metric}_mean" for metric in PRIMARY]]].rename(
        columns={f"{metric}_mean": metric for metric in PRIMARY})
    pd.testing.assert_frame_equal(pd.read_csv(output / "contrasts_per_seed.csv"), factorial_contrasts(primary),
                                  check_dtype=False, rtol=1e-12, atol=1e-12)
    source = Path(manifest["sources"][0]["root"])
    original_manifest = (source / "manifest.json").read_bytes()
    original_receipt = (source / "completed.json").read_bytes()
    changed = read_json(source / "manifest.json")
    changed["method"] = "film"  # A 的方式应为concat；重算文件哈希也不能绕过语义核对。
    (source / "manifest.json").write_text(json.dumps(changed))
    receipt = read_json(source / "completed.json")
    receipt["manifest_sha256"] = sha256_file(source / "manifest.json")
    (source / "completed.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="身份"):
        summary.summarize_test(cache, fixture.root / "wrong_factor", lock_path=fixture.lock_path,
                              confirm_research_test=True)
    (source / "manifest.json").write_bytes(original_manifest)
    (source / "completed.json").write_bytes(original_receipt)
    with (source / "metrics.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="哈希漂移"):
        summary.summarize_test(cache, fixture.root / "corrupted_summary", lock_path=fixture.lock_path,
                              confirm_research_test=True)


def test_concurrent_and_unclosed_attempts_are_blocked(fixture):
    cache = fixture.root / "cache"
    name = "A_seed20260811"
    with evaluate.claim_attempt(cache, name, "r1"):
        with pytest.raises(RuntimeError, match="正在执行"):
            with evaluate.claim_attempt(cache, name, "r2"):
                pytest.fail("并发尝试不得进入")
    unfinished = evaluate.evaluation_root(cache) / name / "attempt_interrupted"
    unfinished.mkdir()
    with pytest.raises(RuntimeError, match="未闭合"):
        with evaluate.claim_attempt(cache, name, "r2"):
            pytest.fail("未闭合尝试不能静默跳过")


def test_nonfinite_checkpoint_is_rejected_before_test_access(fixture, monkeypatch):
    name = "A_seed20260811"
    checkpoint = fixture.checkpoints / f"{name}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_state_dict"]["readout.weight"].fill_(float("nan"))
    torch.save(payload, checkpoint)  # 仅修改disposable checkpoint及其fixture来源锁。
    fixture.lock["entries"][name]["checkpoint_sha256"] = sha256_file(checkpoint)
    fixture.lock["lock_id"] = spec_digest({k:v for k,v in fixture.lock.items() if k != "lock_id"})
    fixture.lock_path.write_text(json.dumps(fixture.lock))
    monkeypatch.setattr(contract, "LOCK_SHA", sha256_file(fixture.lock_path))
    cache = build_cache(fixture)
    monkeypatch.setattr(evaluate, "select_test_rows", lambda *a, **k: pytest.fail("坏状态不得访问test index"))
    with pytest.raises(FloatingPointError, match="checkpoint"):
        run_evaluation(fixture, cache)
    assert (evaluate.evaluation_root(cache) / name / "attempt_r1/failed.json").exists()
