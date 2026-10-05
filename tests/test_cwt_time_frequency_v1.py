"""仅 synthetic/disposable fixture。此实现提交阶段按用户要求未执行。"""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from resp_train.crd.initialization import module_seed
from resp_train.crd.tf_v1_model import CwtBranch
from resp_train.paper_evidence.cwt_time_frequency_v1 import artifacts as io
from resp_train.paper_evidence.cwt_time_frequency_v1.spec import ARMS, SEEDS, plan, load_spec, config
from resp_train.paper_evidence.cwt_time_frequency_v1.model import NativeCwtBranch
from resp_train.paper_evidence.cwt_time_frequency_v1.data import CacheReader
from resp_train.paper_evidence.cwt_time_frequency_v1.interventions import transform_condition, CONDITIONS
from resp_train.paper_evidence.cwt_time_frequency_v1.research_test import guard
from resp_train.paper_evidence.cwt_time_frequency_v1.signals import association
from resp_train.paper_evidence.cwt_time_frequency_v1.summary import paired_table


def test_matrix_and_training_contract(tmp_path):
    spec = load_spec()
    cells = plan()
    assert len(ARMS) == 20 and len(cells) == 60
    assert len({(r["arm"], r["seed"]) for r in cells}) == 60
    assert spec["early_stopping"] is False
    for arm in ARMS:
        cfg = config(arm, SEEDS[0], tmp_path / arm)
        assert cfg.training.epochs == 80 and cfg.training.batch_size == 128
        assert not cfg.training.early_stopping_enabled
        assert Path(cfg.outputs.run_root).is_relative_to(tmp_path)


def test_w0_branch_state_and_nonzero_film_equivalence():
    with module_seed(SEEDS[0], "tf_branch_w"):
        native = CwtBranch().eval()
    with module_seed(SEEDS[0], "tf_branch_w"):
        candidate = NativeCwtBranch(97, 360).eval()
    for key, value in native.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    generator = torch.Generator().manual_seed(29)
    with torch.no_grad():
        native.final_projection.weight.copy_(torch.randn(native.final_projection.weight.shape, generator=generator)*.01)
        candidate.load_state_dict(native.state_dict())
        inputs = {"w": torch.rand(1, 97, 360, generator=generator)}
        for left, right in zip(native(inputs), candidate(inputs)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("scales,frames", [(33, 360), (193, 360), (97, 180), (97, 720), (41, 360)])
def test_native_geometry_has_live_gradient(scales, frames):
    branch = NativeCwtBranch(scales, frames)
    # 使输入导数可观测，不把 zero-init 屏蔽误判为 shape 接口正确。
    with torch.no_grad():
        branch.final_projection.weight.fill_(.001)
    value = torch.rand(1, scales, frames, requires_grad=True)
    gamma, beta = branch({"w": value})
    assert gamma.shape == beta.shape == (1, 96, 1800)
    (gamma.square().mean()+beta.square().mean()).backward()
    assert torch.isfinite(value.grad).all() and value.grad.ne(0).any()
    with pytest.raises((ValueError, RuntimeError)):
        branch({"w": torch.rand(1, scales, frames+1)})
    with pytest.raises(FloatingPointError):
        branch({"w": torch.full((1, scales, frames), float("nan"))})


def test_exact_frequency_masks_and_common_shifts():
    frequencies = [.5, .8, .842, 2., 2.19, 7.995]
    w = torch.arange(2*6*360).reshape(2, 6, 360).float()
    shifts = torch.tensor([[60, 100, 200], [70, 120, 250]])
    original = w.clone()
    changed = transform_condition(w, frequencies, "H2_SHIFT1__NAT", shifts)
    torch.testing.assert_close(changed[:, :4], w[:, :4], rtol=0, atol=0)
    for i in range(2):
        torch.testing.assert_close(changed[i, 4:], torch.roll(w[i, 4:], int(shifts[i, 0]), -1), rtol=0, atol=0)
    flat = transform_condition(w, frequencies, "H_MEAN__FIXED", shifts)
    torch.testing.assert_close(flat[:, :2], w[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(flat[:, 2:], w[:, 2:].mean(-1, keepdim=True).expand_as(w[:, 2:]))
    assert torch.equal(original, w) and len(CONDITIONS) == 18


def test_cache_refuses_wrong_rows_and_nonfinite(tmp_path):
    rep = {"shape": [3, 180]}
    io.write_json(tmp_path / "cache.json", {"representation": rep, "splits": ["val"], "finite": True})
    np.save(tmp_path / "val_row_ids.npy", np.array([10, 30], np.int64))
    value = np.zeros((2, 3, 180), np.float32)
    value[1, 1, 1] = np.nan
    np.save(tmp_path / "val_w.npy", value)
    with pytest.raises(ValueError):
        CacheReader(tmp_path, "val", rep, np.array([30, 10], np.int64))
    reader = CacheReader(tmp_path, "val", rep, np.array([10, 30], np.int64))
    with pytest.raises(ValueError):
        reader.get(0, 30)
    with pytest.raises(FloatingPointError):
        reader.get(1, 30)


def test_lifecycle_keeps_failure_and_rejects_overwrite(tmp_path):
    key = {"phase": "fixture"}
    with pytest.raises(RuntimeError):
        with io.attempt(tmp_path, key) as output:
            io.write_json(output / "partial.json", {"kept": True})
            raise RuntimeError("synthetic failure")
    assert (output / "failed.json").exists() and (output / "partial.json").exists()
    with pytest.raises(RuntimeError):
        with io.attempt(tmp_path, key):
            pass
    with io.attempt(tmp_path, key, retry=True) as success:
        io.write_json(success / "result.json", {"ok": True})
    assert io.completed(tmp_path, key) == success
    with pytest.raises(FileExistsError):
        with io.attempt(tmp_path, key, retry=True):
            pass
    (success / "result.json").write_text("{}")
    with pytest.raises(ValueError):
        io.completed(tmp_path, key)


def test_scientific_missingness_and_delta_direction():
    value, defined = association(np.ones(30), np.arange(30))
    assert np.isnan(value) and not defined
    with pytest.raises(FloatingPointError):
        association(np.full(30, np.inf), np.arange(30))
    with pytest.raises(PermissionError):
        guard(False)
    data = pd.DataFrame([{"arm": arm, "seed": 1, "metric": metric, "mean": value}
                         for arm, metric, value in [("B", "local_rr_mae_bpm", 1.), ("X", "local_rr_mae_bpm", 1.2),
                                                     ("B", "lag_aware_signed_pcc", .8), ("X", "lag_aware_signed_pcc", .7)]])
    paired = paired_table(data, [("X", "B")], ["seed", "metric"])
    indexed = paired.set_index("metric")
    assert indexed.loc["local_rr_mae_bpm", "degradation"] == pytest.approx(.2)
    assert indexed.loc["lag_aware_signed_pcc", "degradation"] == pytest.approx(.1)


def test_synthetic_scope_blocks_real_data_and_training(monkeypatch, tmp_path):
    from resp_train.paper_evidence.cwt_time_frequency_v1 import data, runtime, research_test
    monkeypatch.setattr(io, "load_session", lambda _: {"execution_scope": "synthetic_only"})
    actions = (lambda: data.prepare_data(tmp_path), lambda: data.data_receipt(tmp_path),
               lambda: runtime.run_formal(tmp_path, "B", SEEDS[0], "cuda:0"),
               lambda: runtime.run_parallel(tmp_path, ["cuda:0"]),
               lambda: research_test.prepare_allowlist(tmp_path), lambda: research_test.allowlist(tmp_path))
    for action in actions:
        with pytest.raises(PermissionError, match="仅供合成"):
            action()
    monkeypatch.setattr(io, "load_session", lambda _: {"execution_scope": "train_validation"})
    assert io.require_data_scope(tmp_path)["execution_scope"] == "train_validation"


def test_cwt_baseline_actual_transform():
    from resp_train.paper_evidence.cwt_time_frequency_v1.features import representation, transform
    from resp_train.crd.tf_v1_features import cwt_magnitude_features
    t = np.arange(18000) / 100
    x = (np.sin(2*np.pi*.23*t) + (1+.3*np.sin(2*np.pi*.23*t))*np.cos(2*np.pi*4*t)).astype(np.float32)
    rep = representation(ARMS["B"])
    old, frequencies = cwt_magnitude_features(x)
    np.testing.assert_array_equal(rep["frequencies_hz"], frequencies)
    np.testing.assert_allclose(transform(x, rep), old, rtol=1e-6, atol=1e-7)
    for name, mask in (("C_0p8", frequencies <= .8), ("H", (frequencies > .8) & (frequencies <= 8))):
        candidate = representation(ARMS[name])
        np.testing.assert_array_equal(candidate["frequencies_hz"], frequencies[mask])
        np.testing.assert_allclose(transform(x, candidate), old[mask], rtol=1e-6, atol=1e-7)


def test_calibration_truth_export(tmp_path):
    from resp_train.paper_evidence.cwt_time_frequency_v1.calibration import synthetic_signals
    signals, truth = synthetic_signals()
    assert set(signals).isdisjoint(truth)
    np.savez_compressed(tmp_path / "truth.npz", **signals, **truth)
    with np.load(tmp_path / "truth.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["transient"], signals["transient"])
        np.testing.assert_array_equal(saved["transient_component"], truth["transient_component"])


@pytest.mark.parametrize("reserved,passed,elevated", [(85, True, False), (90, True, False), (94, True, True), (95, True, True), (96, False, True)])
def test_memory_policy_boundary_and_failure_record(reserved, passed, elevated):
    from resp_train.paper_evidence.cwt_time_frequency_v1.engineering import memory_report, require_memory_passed
    record = memory_report(reserved, 80, 100)
    assert record["memory_passed"] is passed
    assert record["elevated_memory"] is elevated
    assert record["peak_allocated_fraction"] == .8
    if passed:
        require_memory_passed(record)
    else:
        with pytest.raises(RuntimeError, match="96.00%.*95%"):
            require_memory_passed(record)
    record["peak_reserved_fraction"] = .01
    with pytest.raises(ValueError, match="显存记录"):
        require_memory_passed(record)


def test_calibration_reuse_rejects_transform_changes():
    from resp_train.paper_evidence.cwt_time_frequency_v1.engineering import audit_calibration_reuse
    engineering = "resp_train/paper_evidence/cwt_time_frequency_v1/engineering.py"
    features = "resp_train/paper_evidence/cwt_time_frequency_v1/features.py"
    old = {engineering: {"sha256": "old", "size_bytes": 1}, features: {"sha256": "same", "size_bytes": 1}}
    current = {**old, engineering: {"sha256": "new", "size_bytes": 2}}
    payload = {"spec": load_spec(), "provenance": {"dependencies": io.dependencies(), "files": old},
               "numerical_checks_passed": True, "representations": dict.fromkeys(ARMS)}
    assert set(audit_calibration_reuse(payload, current)) == {engineering}
    with pytest.raises(ValueError, match="实际依赖"):
        audit_calibration_reuse(payload, {**current, features: {"sha256": "changed", "size_bytes": 2}})
    with pytest.raises(ValueError, match="集合"):
        audit_calibration_reuse(payload, {**current, "new.py": {"sha256": "new", "size_bytes": 1}})


def test_formal_handoff_keeps_scientific_dependencies():
    from resp_train.paper_evidence.cwt_time_frequency_v1.formal_launch import audit_reuse
    model = "resp_train/paper_evidence/cwt_time_frequency_v1/model.py"
    engineering = "resp_train/paper_evidence/cwt_time_frequency_v1/engineering.py"
    driver = "resp_train/paper_evidence/cwt_time_frequency_v1/formal_launch.py"
    old = {model: {"sha256": "model", "size_bytes": 1}, engineering: {"sha256": "old", "size_bytes": 1}}
    current = {**old, engineering: {"sha256": "new", "size_bytes": 2}, driver: {"sha256": "driver", "size_bytes": 3}}
    source = {"spec": load_spec(), "provenance": {"files": old, "dependencies": io.dependencies()}}
    assert set(audit_reuse(source, current)["added"]) == {driver}
    with pytest.raises(ValueError, match="实际依赖"):
        audit_reuse(source, {**current, model: {"sha256": "changed", "size_bytes": 2}})


def test_formal_gpu_gate_reuses_receipt_without_rerun(monkeypatch, tmp_path):
    from resp_train.paper_evidence.cwt_time_frequency_v1 import engineering
    monkeypatch.setattr(io, "load_session", lambda _: {"engineering_reference": {}})
    monkeypatch.setattr(engineering, "require_gpu", lambda *_: tmp_path / "receipt")
    monkeypatch.setattr(engineering, "environment", lambda *_: pytest.fail("不应重跑GPU验收"))
    assert engineering.run_gpu(tmp_path, "cuda:0") == tmp_path / "receipt"


def test_formal_pipeline_orders_preparation_and_training(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from resp_train.paper_evidence.cwt_time_frequency_v1 import formal_launch, data, signals, runtime
    observed = []
    monkeypatch.setattr(io, "binding", lambda *_, **__: {})
    @contextmanager
    def disposable_attempt(*args, **kwargs):
        yield tmp_path
    monkeypatch.setattr(io, "attempt", disposable_attempt)
    monkeypatch.setattr(data, "prepare_data", lambda _: observed.append("data"))
    monkeypatch.setattr(data, "build_cache", lambda _, arm: observed.append("cache_"+arm))
    monkeypatch.setattr(signals, "run_signals", lambda _: observed.append("signals"))
    monkeypatch.setattr(runtime, "run_parallel", lambda *_: observed.append("formal"))
    formal_launch.execute(tmp_path, ["cuda:0", "cuda:1"])
    assert observed[:3] == ["data", "cache_C_20", "signals"]
    assert observed[-1] == "formal"
    assert {v for v in observed if v.startswith("cache_")} == {"cache_"+arm for arm in ARMS}
