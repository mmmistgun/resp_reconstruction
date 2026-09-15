from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from resp_train.paper_evidence import e1_scale_topology as core
from resp_train.paper_evidence import e1_scale_topology_runtime as runtime


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FakeBranch(nn.Module):
    def forward(self, tf):
        # 使用不对称尺度读取，使每种操作的轴错误能在测试中暴露。
        base = tf["w"][:, 0, :].repeat_interleave(5, dim=-1).unsqueeze(1)
        gamma = base.expand(-1, 96, -1)
        beta = -gamma * 2
        return gamma, beta


class FakeW0(nn.Module):
    tf_variant = "crd_tf102_w"

    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleDict({"w": FakeBranch()})
        self.controls = nn.ModuleList()
        self.fusion_gate = None
        self.finished = False

    def forward(self, x, *, tf):
        gamma, beta = self.branches["w"](tf)
        result = {"waveform": x + (0.5 * torch.tanh(gamma) + 0.5 * torch.tanh(beta)).mean((1, 2))[:, None, None]}
        self.finished = True
        return result


def test_indices_are_locked_exact_inverse_and_do_not_change_global_rng():
    np.random.seed(77)
    before = np.random.get_state()
    lock = core.make_index_lock()
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert all(lock["self_checks"].values())
    assert lock == core.make_index_lock()
    value = torch.arange(2 * 97 * 360, dtype=torch.float32).reshape(2, 97, 360)
    original = value.clone()
    for name, entry in lock["conditions"].items():
        result = core.apply_scale(value, torch.tensor(entry["index"]))
        assert torch.equal(result[:, entry["inverse"]], original)
        assert torch.equal(result.sort(dim=1).values, original.sort(dim=1).values)
        for slot, source in enumerate(entry["index"]):
            assert torch.equal(result[:, slot], original[:, source])
        if name == "SCALE_SHIFT_12":
            assert torch.equal(result, torch.roll(original, 12, 1))
        elif name == "SCALE_REVERSE":
            assert torch.equal(result, original.flip(1))
        assert torch.equal(value, original)


@pytest.mark.parametrize("mutation", ["duplicate", "inverse", "hash", "shift", "condition"])
def test_reject_bad_index_lock(mutation):
    lock = core.make_index_lock()
    entry = lock["conditions"]["SCALE_PERMUTE_FIXED"]
    if mutation == "duplicate":
        entry["index"][0] = entry["index"][1]
    elif mutation == "inverse":
        entry["inverse"][0], entry["inverse"][1] = entry["inverse"][1], entry["inverse"][0]
    elif mutation == "hash":
        entry["index_sha256"] = "0" * 64
    elif mutation == "shift":
        lock["conditions"]["SCALE_SHIFT_12"] = copy.deepcopy(lock["conditions"]["FULL"])
    else:
        del lock["conditions"]["SCALE_REVERSE"]
    with pytest.raises(ValueError):
        core.validate_index_lock(lock)


def test_wrapper_calls_native_forward_then_callback_and_preserves_state():
    model = FakeW0().eval()
    x, w = torch.zeros(2, 1, 18000), torch.arange(2 * 97 * 360).float().reshape(2, 97, 360) / 10000
    original = w.clone()
    lock = core.make_index_lock()
    for name in core.CONDITIONS:
        captured = []

        def callback(raw):
            assert model.finished
            captured.append(raw)

        model.finished = False
        result = core.ScaleAuditModel(model, name, lock, callback)(x, tf={"w": w})
        expected_w = w if name == "FULL" else w[:, lock["conditions"][name]["index"], :]
        expected = model(x, tf={"w": expected_w})
        assert torch.equal(result["waveform"], expected["waveform"])
        assert len(captured) == 1 and captured[0][0].shape == (2, 96, 1800)
        assert torch.equal(w, original)
        assert not model.branches["w"]._forward_hooks


def test_hook_removed_if_native_forward_fails():
    model = FakeW0()

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic native failure")

    model.forward = fail
    wrapper = core.ScaleAuditModel(model, "FULL", core.make_index_lock(), lambda raw: None)
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        wrapper(torch.zeros(1), tf={"w": torch.zeros(1, 97, 360)})
    assert not model.branches["w"]._forward_hooks


def test_wrapper_exact_with_real_cwt_and_decoder_on_cpu():
    from resp_train.crd.tf_v1_model import CRDTfV1Model

    torch.manual_seed(81)
    model = CRDTfV1Model("crd_tf102_w", core.SEEDS[0]).eval()
    # CPU fixture 以 identity 替换 CUDA Mamba；保留真实 CWT、FiLM、前后端。
    model.base.local_blocks = nn.ModuleList(nn.Identity() for _ in model.base.local_blocks)
    with torch.no_grad():
        model.branches["w"].final_projection.weight.normal_(std=1e-4)
        model.branches["w"].final_projection.bias.normal_(std=1e-4)
        x, w = torch.randn(1, 1, 18000), torch.randn(1, 97, 360)
        native = model(x, tf={"w": w})
        captured = []
        result = core.ScaleAuditModel(model, "FULL", core.make_index_lock(), captured.append)(x, tf={"w": w})
    assert len(captured) == 1
    for key in native:
        torch.testing.assert_close(result[key], native[key], atol=0, rtol=0)


def test_raw_mae_uses_pointwise_pairing_not_difference_of_amplitudes():
    anchor = np.array([[[1., -1.], [2., -2.]]], dtype=np.float32)
    current = -anchor
    assert np.abs(anchor).mean() == np.abs(current).mean()
    assert core.raw_pair_mae(current, anchor).tolist() == [3.]
    assert core.raw_pair_mae(anchor, anchor).tolist() == [0.]
    with pytest.raises(FloatingPointError):
        core.raw_pair_mae(current * np.nan, anchor)
    with pytest.raises(ValueError):
        core.raw_pair_mae(current[:, :1], anchor)


def test_film_store_pairs_batches_and_retains_full_files(tmp_path):
    torch.manual_seed(19)
    gamma = torch.randn(3, 96, 1800).to(torch.bfloat16)
    beta = -gamma
    full = runtime.FilmStore(tmp_path, 3, full=True)
    full((gamma[:2], beta[:2]))
    full((gamma[2:], beta[2:]))
    assert np.all(full.finish() == 0)
    raw_before = runtime.identity(tmp_path / "full_gamma_raw.npy")
    paired = runtime.FilmStore(tmp_path, 3, full=False)
    paired((-gamma[:2], -beta[:2]))
    paired((-gamma[2:], -beta[2:]))
    expected = 2 * np.abs(gamma.float().numpy().astype(np.float64)).mean((1, 2))
    np.testing.assert_array_equal(paired.finish()[:, 0], expected)
    assert runtime.identity(tmp_path / "full_gamma_raw.npy") == raw_before
    with pytest.raises(FileExistsError):
        runtime.FilmStore(tmp_path, 3, full=True)
    short = runtime.FilmStore(tmp_path, 3, full=False)
    short((gamma[:1], beta[:1]))
    with pytest.raises(ValueError, match="不完整"):
        short.finish()


def seed_summary():
    records = []
    for name in core.CONDITIONS:
        for position, seed in enumerate(core.SEEDS):
            anchor = [1., 2., 4.][position]
            difference = 0 if name == "FULL" else [0.1, -0.2, 1.2][position]
            row = {"condition": name, "seed": seed, **{key + "_mean": anchor + difference for key in core.ERRORS},
                   core.PCC + "_mean": 0.8 if name == "FULL" else [0.7, 0.9, 0.6][position]}
            row.update({key: 0. if name == "FULL" else 1. for key in core.FILM_COLUMNS})
            records.append(row)
    return pd.DataFrame(records)


def test_paired_summary_sign_sd_and_ratio_of_means_are_distinct():
    raw, seed, aggregate = core.summarize_pairs(seed_summary().sample(frac=1, random_state=17))
    row = aggregate.query("condition == 'SCALE_SHIFT_12' and metric == 'whole_rr_abs_error_bpm'").iloc[0]
    assert row.paired_delta_mean == pytest.approx(10.)
    assert row.paired_delta_sample_sd == pytest.approx(20.)
    assert row.delta_of_seed_means == pytest.approx(100 * 1.1 / 7)
    assert (row.worse_seeds, row.better_seeds, row.equal_seeds) == (2, 1, 0)
    pcc = seed.query("condition == 'SCALE_SHIFT_12' and metric == 'lag_aware_signed_pcc'")
    np.testing.assert_allclose(pcc.delta, [0.1, -0.1, 0.2])
    assert len(raw) == 4 * 7 and len(seed) == 4 * 3 * 5


@pytest.mark.parametrize("failure", ["missing", "duplicate", "nan", "zero_anchor", "negative_film"])
def test_summary_rejects_incomplete_or_invalid_data(failure):
    frame = seed_summary()
    if failure == "missing":
        frame = frame.iloc[:-1]
    elif failure == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif failure == "nan":
        frame.loc[5, core.PCC + "_mean"] = np.nan
    elif failure == "zero_anchor":
        frame.loc[0, core.ERRORS[0] + "_mean"] = 0
    else:
        frame.loc[5, core.FILM_COLUMNS[0]] = -1
    with pytest.raises((ValueError, FloatingPointError)):
        core.summarize_pairs(frame)


def metrics_fixture(rows):
    frame = rows[["dataset_row_id", "split", "samp_id"]].copy()
    for key in core.PRIMARY:
        frame[key] = 0.5 if key == core.PCC else 1.
    for key in ("whole_rr_target_eligible", "local_rr_target_eligible", "joint_target_eligible", "envelope_spearman_target_eligible"):
        frame[key] = True
    frame["local_rr_target_eligible_windows"] = 25
    frame["joint_prediction_degenerate"] = False
    frame["envelope_spearman_prediction_degenerate"] = False
    return frame


def test_metrics_respect_target_eligibility_and_reject_prediction_failure():
    rows = pd.DataFrame({"dataset_row_id": [1, 2], "samp_id": [3, 4], "split": "val"})
    frame = metrics_fixture(rows)
    core.validate_metrics(frame, rows)
    frame.loc[0, "whole_rr_abs_error_bpm"] = np.nan
    with pytest.raises(FloatingPointError):
        core.validate_metrics(frame, rows)
    frame.loc[0, "whole_rr_target_eligible"] = False
    core.validate_metrics(frame, rows)
    frame.loc[1, "joint_prediction_degenerate"] = True
    with pytest.raises(RuntimeError, match="degeneracy"):
        core.validate_metrics(frame, rows)
    frame.loc[1, "joint_prediction_degenerate"] = False
    with pytest.raises(ValueError, match="identity/order"):
        core.validate_metrics(frame.iloc[::-1], rows)


def test_full_anchor_checks_each_metric_and_denominator():
    columns = {key + "_mean": [1.] for key in core.PRIMARY}
    columns.update({key + "_n": [7] for key in core.PRIMARY})
    reference = pd.DataFrame(columns)
    observed = reference.copy()
    observed.loc[0, core.PCC + "_mean"] += 5e-7
    assert core.check_full_anchor(observed, reference, core.SEEDS[0])["passed"]
    observed.loc[0, core.PCC + "_mean"] += 1e-6
    with pytest.raises(RuntimeError, match="超差"):
        core.check_full_anchor(observed, reference, core.SEEDS[0])
    observed = reference.copy()
    observed.loc[0, core.PCC + "_n"] = 6
    with pytest.raises(ValueError, match="分母"):
        core.check_full_anchor(observed, reference, core.SEEDS[0])


def test_failed_attempt_is_preserved_and_new_attempt_is_independent(tmp_path):
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with runtime.attempt("validation", "a" * 64, tmp_path) as failed:
            (failed / "partial.csv").write_text("synthetic\n", encoding="utf-8")
            raise RuntimeError("synthetic failure")
    assert (failed / "partial.csv").is_file()
    assert json.loads((failed / "lifecycle_failed.json").read_text())["status"] == "failed"
    assert not (failed / "manifest.json").exists()
    with runtime.attempt("validation", "a" * 64, tmp_path) as passed:
        runtime.write_json(passed / "payload.json", {"fixture": True})
    assert passed != failed
    freeze = json.loads((passed / "freeze_receipt.json").read_text())
    runtime.verify_file(passed / "manifest.json", freeze["manifest"])
    assert (failed / "partial.csv").read_text() == "synthetic\n"


def test_manifest_failure_also_records_failed_lifecycle(tmp_path, monkeypatch):
    original = runtime.identity

    def fail_payload(path):
        if path.name == "payload.json":
            raise OSError("synthetic hash failure")
        return original(path)

    monkeypatch.setattr(runtime, "identity", fail_payload)
    with pytest.raises(OSError, match="synthetic hash failure"):
        with runtime.attempt("validation", "e" * 64, tmp_path) as output:
            runtime.write_json(output / "payload.json", {"fixture": True})
    assert (output / "lifecycle_failed.json").is_file()
    assert not (output / "freeze_receipt.json").exists()


def test_prepare_locks_only_reads_allowlisted_sources_and_detects_drift(tmp_path, monkeypatch):
    root = tmp_path
    (root / runtime.DOCS).mkdir(parents=True)
    monkeypatch.setattr(runtime, "git_state", lambda *args, **kwargs: {"commit": "synthetic"})
    entries = []
    for seed, epoch in zip(core.SEEDS, [13, 15, 14], strict=True):
        directory = root / f"source_{seed}"
        directory.mkdir()
        entry = {"seed": seed, "selected_epoch": epoch, "run_dir": directory.name}
        for key, name in (("checkpoint", "checkpoint_best_local_rr.pt"), ("config", "config.yaml"),
                          ("manifest", "run_manifest.json"), ("validation_summary", "metrics_summary.csv")):
            path = directory / name
            path.write_bytes(b"synthetic identity only; not a real checkpoint")
            entry[key] = runtime.identity(path)
        entries.append(entry)
    cache_root = root / "synthetic_cache"
    cache_root.mkdir()
    cache = {"root": cache_root.name}
    for key, name in (("manifest", "cache_manifest.json"), ("val_w", "val_w.npy"), ("frequency_file", "w_frequencies_hz.npy")):
        path = cache_root / name
        if key == "manifest":
            runtime.write_json(path, {"dataset_index": "synthetic_index.csv", "dataset_index_sha256": "synthetic"})
        else:
            path.write_bytes(b"synthetic bytes; not a numpy payload")
        cache[key] = {"path": str(path.relative_to(root)), **runtime.identity(path)}
    (cache_root / "val_row_ids.npy").write_bytes(b"synthetic row identity bytes")
    cache["row_identity"] = {"val_row_file_sha256": runtime.sha256_file(cache_root / "val_row_ids.npy")}
    source = {"anchors": [{"role": "w0_anchor", "variant": "crd_tf102_w", "checkpoints": entries}], "cache_lock": cache}
    runtime.write_json(root / runtime.SOURCE_LOCK, source)
    monkeypatch.setattr(runtime, "SOURCE_LOCK_SHA256", runtime.sha256_file(root / runtime.SOURCE_LOCK))
    for relative in (runtime.SCRIPT, runtime.TEST, runtime.PROTOCOL_PATH, Path("resp_train/synthetic.py")):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("synthetic fixture\n")
    before = {path: runtime.identity(path) for path in root.rglob("*") if path.is_file()}
    index_path, lock_path = runtime.prepare_locks(root)
    assert index_path.is_file() and lock_path.is_file()
    lock, _, _ = runtime.load_locks(root)
    assert len(lock["w0_entries"]) == 3
    for path, expected in before.items():
        assert runtime.identity(path) == expected
    with pytest.raises(FileExistsError):
        runtime.prepare_locks(root)
    (root / runtime.SCRIPT).write_text("changed fixture\n")
    with pytest.raises(RuntimeError, match="身份漂移"):
        runtime.load_locks(root)


def test_gpu_receipt_rejects_changed_payload(tmp_path):
    lock_hash = "b" * 64
    with runtime.attempt("gpu_smoke", lock_hash, tmp_path) as output:
        runtime.write_json(output / "gpu_acceptance.json", {"passed": True, "conditions": list(core.CONDITIONS), "batch_size": 1})
    assert runtime.validate_gpu_receipt(output, lock_hash)["receipt"]["passed"]
    with pytest.raises(ValueError, match="身份不匹配"):
        runtime.validate_gpu_receipt(output, "c" * 64)
    (output / "gpu_acceptance.json").write_text("{}")
    with pytest.raises(RuntimeError, match="身份漂移"):
        runtime.validate_gpu_receipt(output, lock_hash)


def test_checked_batches_rejects_row_swap_and_nonfinite():
    rows = pd.DataFrame({"dataset_row_id": [1, 2], "samp_id": [3, 4], "split": "val"})
    batch = {"x": torch.zeros(2, 1, 18000), "target": torch.zeros(2, 1, 18000),
             "tf": {"w": torch.zeros(2, 97, 360)},
             "meta": {"dataset_row_id": torch.tensor([1, 2]), "samp_id": torch.tensor([3, 4]), "split": ["val", "val"]}}
    assert len(list(runtime.checked_batches([batch], rows))) == 1
    batch["meta"]["dataset_row_id"] = torch.tensor([2, 1])
    with pytest.raises(ValueError, match="identity/order"):
        list(runtime.checked_batches([batch], rows))
    batch["meta"]["dataset_row_id"] = torch.tensor([1, 2])
    batch["target"][0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        list(runtime.checked_batches([batch], rows))


@pytest.fixture
def synthetic_runtime(tmp_path, monkeypatch):
    """完整生命周期走真实 wrapper/collector/汇总，全部数据与模型均为 disposable fixture。"""
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    original_attempt = runtime.attempt
    monkeypatch.setattr(runtime, "attempt", lambda phase, lock_hash: original_attempt(phase, lock_hash, tmp_path))
    monkeypatch.setattr(runtime, "WINDOW_COUNT", 7)
    monkeypatch.setattr(runtime, "validate_rows", lambda rows: core.validate_rows(rows, expected_count=7))
    monkeypatch.setattr(runtime, "git_state", lambda **kwargs: {"commit": "synthetic", "status_porcelain": ""})
    monkeypatch.setattr(runtime, "require_gpu", lambda device: None)
    monkeypatch.setattr(runtime, "environment", lambda device: {"fixture": True})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runtime, "validate_gpu_receipt", lambda *args: {"fixture": True})
    rows = pd.DataFrame({"dataset_row_id": np.arange(7), "samp_id": np.arange(7), "split": "val"})
    samples = [{"x": torch.zeros(1, 18000), "target": torch.zeros(1, 18000),
                "tf": {"w": torch.arange(97 * 360).float().reshape(97, 360) / 10000},
                "meta": {"dataset_row_id": i, "samp_id": i, "split": "val"}} for i in range(7)]
    data = SimpleNamespace(rows=rows, dataset=samples, loader=DataLoader(samples, batch_size=3))
    monkeypatch.setattr(runtime, "build_window_data", lambda *args, **kwargs: data)
    cfg = OmegaConf.create({"data": {"val_sample_strategy": "stratified_random", "val_sample_seed": 20260611}})
    monkeypatch.setattr(runtime, "_load_audit_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(runtime, "load_model", lambda *args: (cfg, FakeW0()))
    monkeypatch.setattr(runtime, "evaluate_task_predictions", lambda predictions, *args, **kwargs: metrics_fixture(rows))
    index_path = tmp_path / "synthetic_index.csv"
    rows.to_csv(index_path, index=False)
    np.save(tmp_path / "frequency.npy", np.arange(97).astype(float), allow_pickle=False)
    entries = []
    for seed, epoch in zip(core.SEEDS, [13, 15, 14], strict=True):
        relative = f"synthetic_source_{seed}"
        (tmp_path / relative).mkdir()
        runtime.summarize_task_metrics(metrics_fixture(rows)).to_csv(tmp_path / relative / "metrics_summary.csv", index=False)
        entries.append({"seed": seed, "selected_epoch": epoch, "run_dir": relative, "checkpoint": {"sha256": "synthetic"}})
    lock = {"source_files": {}, "dataset_index": {"path": str(index_path), "sha256": runtime.sha256_file(index_path)},
            "w0_entries": entries, "cache_lock": {"row_identity": {"val_row_content_sha256": core.array_hash(np.arange(7))},
                                                "frequency_file": {"path": "frequency.npy"}}}
    monkeypatch.setattr(runtime, "load_locks", lambda: (lock, core.make_index_lock(), "d" * 64))
    calls = []
    original_evaluate = runtime.evaluate_condition

    def evaluate(*args, **kwargs):
        calls.append((args[2]["seed"], args[3]))
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(runtime, "evaluate_condition", evaluate)
    return tmp_path, calls


def test_complete_synthetic_runtime_full_gate_outputs_and_manifest(synthetic_runtime):
    root, calls = synthetic_runtime
    output = runtime.run_validation(device="cpu", gpu_receipt=root / "synthetic_receipt")
    assert calls[:3] == [(seed, "FULL") for seed in core.SEEDS]
    assert len(calls) == 12
    assert (output / "full_anchor_receipt.json").is_file()
    summary = pd.read_csv(output / "seed_summary.csv")
    assert len(summary) == 12
    metrics = pd.concat(pd.read_csv(path) for path in output.glob("seed_*/*/metrics.csv"))
    assert len(metrics) == 84
    assert not metrics[["condition", "seed", "split", "dataset_row_id"]].duplicated().any()
    manifest = json.loads((output / "manifest.json").read_text())
    assert len([key for key in manifest["files"] if key.endswith("_raw.npy")]) == 6
    for relative, expected in manifest["files"].items():
        runtime.verify_file(output / relative, expected)
    assert json.loads((output / "completion_receipt.json").read_text())["metric_rows"] == 84


def test_failed_third_full_prevents_every_intervention(synthetic_runtime):
    root, calls = synthetic_runtime
    reference = root / f"synthetic_source_{core.SEEDS[-1]}" / "metrics_summary.csv"
    frame = pd.read_csv(reference)
    frame.loc[0, core.PCC + "_mean"] += 0.01
    frame.to_csv(reference, index=False)
    with pytest.raises(RuntimeError, match="超差"):
        runtime.run_validation(device="cpu", gpu_receipt=root / "synthetic_receipt")
    assert calls == [(seed, "FULL") for seed in core.SEEDS]
    attempts = list((root / runtime.OUTPUT / "validation").iterdir())
    assert len(attempts) == 1 and (attempts[0] / "lifecycle_failed.json").exists()
    assert not (attempts[0] / "full_anchor_receipt.json").exists()
    assert not (attempts[0] / "manifest.json").exists()
