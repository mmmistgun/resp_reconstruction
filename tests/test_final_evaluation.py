"""最终评价定向测试：仅使用合成信号及临时文件。"""

import json

import numpy as np
import pandas as pd
import pytest

from resp_train.metrics.final_evaluation import (
    EPS, METRICS, EvaluationFailure, envelope, estimate_rr, evaluate_window,
    lag_correlation, project, rr_from_power, select_nonoverlap_centers,
    summarize_seeds, summarize_windows,
)
from resp_train.paper_evidence.w0_final_evaluation import check_coverage, write_json


def wave(hz=.2, n=18000):
    return np.sin(2 * np.pi * hz * np.arange(n) / 100)


def test_projection_matches_direct_dft_and_separate_scale():
    raw = 3 + 2 * wave(.2) + wave(2)
    band, normalized = project(raw)
    np.testing.assert_allclose(band, 2 * wave(.2), atol=2e-13)
    np.testing.assert_allclose(normalized, band / np.sqrt(np.mean(band**2) + EPS), atol=1e-13)
    result = evaluate_window(raw, wave(.2), center_selected=True)
    assert result[METRICS[0]] < 1e-10
    assert result[METRICS[1]] < 1e-10
    assert result[METRICS[2]] < 2e-8
    assert result[METRICS[3]] < 1e-10
    assert result[METRICS[4]] > .999999999
    assert result["best_lag_samples"] == 0


def test_rr_interpolation_median_and_band_edges():
    assert estimate_rr(wave(.237, n=6000)) == pytest.approx(.237 * 60, abs=.02)
    assert estimate_rr(wave(.237)) == pytest.approx(.237 * 60, abs=.02)
    assert estimate_rr(wave(.05, n=6000)) == pytest.approx(3)
    assert estimate_rr(wave(.7, n=6000)) == pytest.approx(42)
    # 对数谱构造精确抛物线，峰值偏移 +0.25 bin。
    bins = np.arange(3001)
    power = np.exp(-.5 * (bins - 14.25)**2)
    assert rr_from_power(power) == pytest.approx(14.25, abs=1e-10)
    corrupted = wave(.2)
    corrupted[:3000] += 100 * wave(.4, n=3000)
    assert estimate_rr(corrupted) == pytest.approx(12, abs=.02)


@pytest.mark.parametrize("delay", [-30, -17, 0, 21, 30])
def test_pcc_direction_common_interval_and_mae_reuses_lag(delay):
    target = wave(.2) + .4 * wave(.35)
    prediction = np.roll(target, delay)
    result = evaluate_window(prediction, target, center_selected=False)
    assert result["best_lag_samples"] == delay
    assert result[METRICS[2]] < 1e-12
    assert result[METRICS[4]] > .999999999
    assert result[METRICS[1]] is None
    _, px = project(prediction)
    _, tx = project(target)
    # 独立公式核验所有 lag 使用相同长度，均值按配对区间重算。
    scores = []
    for lag in range(-30, 31):
        a, b = px[30+lag:17970+lag].copy(), tx[30:17970].copy()
        assert len(a) == len(b) == 17940
        a -= a.mean()
        b -= b.mean()
        scores.append(np.sum(a*b) / (np.sqrt(np.sum(a*a)+EPS) * np.sqrt(np.sum(b*b)+EPS)))
    assert result[METRICS[4]] == pytest.approx(max(scores), abs=1e-14)


def test_pcc_signed_and_exact_tie_priority():
    result = evaluate_window(-wave(), wave(), center_selected=True)
    assert result[METRICS[4]] < -.9
    assert result[METRICS[2]] > 1
    assert lag_correlation(np.zeros(18000), np.ones(18000)) == (0, 0)
    alternating = np.tile([1., -1.], 9000)
    assert lag_correlation(-alternating, alternating)[1] == -1


def test_center_projection_uses_raw_crop_not_full_projection():
    # 外侧脉冲经 180 s 投影泄入中央，但原始中央仅含独立的 0.3 Hz 正弦。
    reference = wave(.2)
    prediction = wave(.3)
    prediction[100] = 1e4
    result = evaluate_window(prediction, reference, center_selected=True)
    _, crop = project(prediction[6000:12000])
    _, full = project(prediction)
    assert result["rr60_pred_bpm"] == pytest.approx(estimate_rr(crop), abs=1e-12)
    assert not np.allclose(crop, full[6000:12000])
    assert result[METRICS[1]] == pytest.approx(6, abs=.001)


def test_envelope_35_points_median_centered_and_not_lag_shifted():
    time = np.arange(18000) / 100
    target = wave()
    prediction = (1 + .6 * np.sin(2*np.pi*time/90)) * wave()
    _, px = project(prediction)
    _, tx = project(target)
    def literal(v):
        q = np.array([.5*np.log(np.mean(v[500*j:500*j+1000]**2)+EPS) for j in range(35)])
        return q - np.median(q)
    assert envelope(px).shape == (35,)
    np.testing.assert_allclose(envelope(px), literal(px))
    result = evaluate_window(prediction, target, center_selected=False)
    assert result[METRICS[3]] == pytest.approx(np.mean(np.abs(literal(px)-literal(tx))))
    assert result[METRICS[3]] > .1


def rows_fixture():
    return pd.DataFrame({
        "dataset_row_id": np.arange(9), "samp_id": [1]*7 + [2]*2,
        "target_source_npz": ["night_a"]*5 + ["night_b"]*2 + ["night_a"]*2,
        "window_start_sample": [0, 3000, 6000, 15000, 18000, 0, 3000, 0, 6000],
    }).assign(window_end_sample=lambda x: x.window_start_sample + 18000)


def test_nonoverlap_absolute_time_records_gaps_order_invariant():
    rows = rows_fixture()
    selected = select_nonoverlap_centers(rows)
    assert rows.loc[selected, "dataset_row_id"].tolist() == [0, 2, 3, 5, 7, 8]
    shuffled = rows.sample(frac=1, random_state=5)
    assert set(shuffled.loc[select_nonoverlap_centers(shuffled), "dataset_row_id"]) == {0, 2, 3, 5, 7, 8}
    # 状态变化不能重启同一整晚记录的选窗。
    rows["coupling_state_id"] = np.arange(9)
    np.testing.assert_array_equal(select_nonoverlap_centers(rows), selected)
    with pytest.raises(EvaluationFailure, match="重复"):
        select_nonoverlap_centers(pd.concat([rows, rows.iloc[:1]]))
    rows.loc[0, "window_start_sample"] = np.nan
    with pytest.raises(EvaluationFailure):
        select_nonoverlap_centers(rows)


@pytest.mark.parametrize("bad", [np.zeros(18000), wave()*1e-5, np.full(18000, np.nan), np.full(18000, np.inf)])
def test_bad_prediction_fails_even_if_reference_ineligible(bad):
    with pytest.raises((EvaluationFailure, FloatingPointError)):
        evaluate_window(bad, np.zeros(18000), center_selected=True)


def test_center_degeneracy_and_nonfinite_reference():
    prediction = wave()
    prediction[6000:12000] = 0
    with pytest.raises(EvaluationFailure, match="中央"):
        evaluate_window(prediction, wave(), center_selected=True)
    assert evaluate_window(prediction, wave(), center_selected=False)[METRICS[0]] is not None
    target = wave()
    target[0] = np.inf
    with pytest.raises(EvaluationFailure):
        evaluate_window(wave(), target, center_selected=True)


def test_explicit_reference_eligibility_window_means_and_seed_sd():
    valid = evaluate_window(wave(.2), wave(.25), center_selected=True)
    ineligible = evaluate_window(wave(.2), np.zeros(18000), center_selected=True)
    assert all(ineligible[key] is None for key in METRICS)
    result = summarize_windows([valid, ineligible])
    assert all(result[key + "_n"] == 1 for key in METRICS)
    assert result[METRICS[0]] == pytest.approx(valid[METRICS[0]])
    summaries = [{"seed": i, **{key: float(i) for key in METRICS}, **{key+"_n": 1 for key in METRICS}} for i in (1, 2, 3)]
    combined = summarize_seeds(summaries, (1, 2, 3))
    assert combined[METRICS[0]] == {"mean": 2., "sample_sd": 1., "n_per_seed": 1}
    assert summarize_seeds(summaries[:1], (1,))[METRICS[0]]["sample_sd"] is None
    with pytest.raises(EvaluationFailure):
        summarize_seeds(summaries[:2], (1, 2, 3))
    valid[METRICS[0]] = float("nan")
    with pytest.raises(EvaluationFailure, match="有限"):
        summarize_windows([valid])
    with pytest.raises(EvaluationFailure, match="零"):
        summarize_windows([ineligible])


def test_coverage_and_no_overwrite(tmp_path):
    check_coverage([4, 7, 9], [4, 7, 9])
    for observed in ([4, 7], [4, 7, 7], [7, 4, 9]):
        with pytest.raises(EvaluationFailure):
            check_coverage(observed, [4, 7, 9])
    path = tmp_path / "receipt.json"
    write_json(path, {"status": "failed", "dataset_row_id": 7})
    with pytest.raises(FileExistsError):
        write_json(path, {"status": "complete"})
    assert json.loads(path.read_text())["status"] == "failed"


@pytest.mark.parametrize("fail_second_seed", [False, True])
def test_synthetic_runtime_lifecycle(tmp_path, monkeypatch, fail_second_seed):
    """替换数据装载和模型为 CPU 合成 fixture，走真实产物/汇总/失败路径。"""
    from contextlib import nullcontext
    from types import SimpleNamespace
    import hashlib
    import torch
    from omegaconf import OmegaConf
    from resp_train.paper_evidence import w0_final_evaluation as runtime
    import resp_train.crd.experiment as experiment
    import resp_train.crd.model as models
    import resp_train.crd.tf_v1_data as tf_data
    import resp_train.data.factory as factory
    import resp_train.data.research_v2 as research
    import resp_train.paper_evidence.w0_cwt_film_behavior_runtime as env

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(runtime, "ROOT", root)
    monkeypatch.setattr(runtime, "COUNT", 2)
    monkeypatch.setattr(runtime, "SUBJECTS", 1)
    paths = ["resp_train/fixture.py", "scripts/eval_w0_final_metrics.py", runtime.PROTOCOL_PATH,
             "tests/test_final_evaluation.py", *["docs/experiments/" + name for name in runtime.SOURCE_LOCKS]]
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source")
    source = root / "signal.npz"
    source.write_bytes(b"synthetic identity; never loaded")
    rows = rows_fixture().iloc[:2].copy()
    rows["target_source_npz"] = str(source)
    rows["source_npz"] = str(source)
    rows["split"], rows["input_set"], rows["coupling_state_id"] = "test", "research_v2_waveform", 0
    rows["window_start_s"] = rows.window_start_sample / 100
    rows["window_end_s"] = rows.window_end_sample / 100
    index = root / "index.csv"
    rows.to_csv(index, index=False)
    reference_file = runtime.record(index)
    cache = root / "cache"
    cache.mkdir()
    (cache / "cache_manifest.json").write_text("{}")
    (cache / "test_w.npy").write_bytes(b"synthetic cache identity")
    baselines, configs = [], []
    for seed, epoch in zip(runtime.SEEDS, runtime.EPOCHS, strict=True):
        run = root / f"run_{seed}"
        run.mkdir()
        cfg = OmegaConf.create({
            "data": {"dataset_root": str(root), "index_csv": "index.csv", "input_set": "research_v2_waveform",
                     "max_test_windows": None, "test_sample_seed": 20260612, "drop_nonfinite_windows": False,
                     "test_sample_strategy": "stratified_random", "preload_windows": True},
            "model": {"variant": "crd_tf102_w", "initialization_seed": seed},
            "training": {"seed": seed, "batch_size": 128, "use_amp": True, "amp_dtype": "bfloat16",
                         "device": "cuda:0", "show_progress": False},
            "window": {"target_fs": 100, "duration_samples": 18000},
        })
        configs.append(cfg)
        OmegaConf.save(cfg, run / "config.yaml")
        (run / "checkpoint_best_local_rr.pt").write_bytes(b"fake checkpoint")
        (run / "train_history.csv").write_text("synthetic history")
        (run / "run_manifest.json").write_text("{}")
        baselines.append({"seed": seed, "selected_epoch": epoch, "run_dir": str(run), **{
            key: runtime.record(run / name) for key, name in (("config", "config.yaml"),
            ("checkpoint", "checkpoint_best_local_rr.pt"), ("history", "train_history.csv"), ("manifest", "run_manifest.json"))}})
    test = {"entries": [{"seed": seed, "baseline_selected_epoch": epoch,
              "baseline_test": {"research_test_metrics.csv": reference_file}, "development_samp_ids": [999]}
             for seed, epoch in zip(runtime.SEEDS, runtime.EPOCHS, strict=True)],
            "cache_root": str(cache), "cache_manifest_sha256": runtime.sha256(cache / "cache_manifest.json"),
            "cache_files": {"test_w.npy": runtime.record(cache / "test_w.npy")}, "dataset_index": reference_file,
            "row_order_sha256": hashlib.sha256(rows.dataset_row_id.to_numpy(np.int64).tobytes()).hexdigest()}
    monkeypatch.setattr(runtime, "load_sources", lambda: (baselines, test))
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *args, **kwargs: "synthetic git")
    monkeypatch.setattr(env, "environment", lambda device: {"fixture": "CPU synthetic"})
    monkeypatch.setattr(torch, "device", lambda device: SimpleNamespace(type="cuda", index=0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.amp, "autocast", lambda *args, **kwargs: nullcontext())
    def fake_load(path, **kwargs):
        seed = next(s for s in runtime.SEEDS if str(s) in str(path))
        i = runtime.SEEDS.index(seed)
        return {"epoch": runtime.EPOCHS[i], "config": configs[i], "model_state_dict": {"a": torch.ones(1)}}
    monkeypatch.setattr(torch, "load", fake_load)
    monkeypatch.setattr(experiment, "_validate_checkpoint_config", lambda *args: None)
    monkeypatch.setattr(research, "read_research_v2_index", lambda *args: rows.copy())
    values = torch.from_numpy(np.stack([wave(), wave(.25)]).astype(np.float32))[:, None, :]
    batch = {"x": values, "target": values, "tf": {"w": torch.ones(2, 97, 360)},
             "meta": {"dataset_row_id": torch.tensor(rows.dataset_row_id.to_numpy())}}
    monkeypatch.setattr(factory, "build_window_data", lambda *args, **kwargs: SimpleNamespace(rows=rows, dataset=[0, 1], loader=[batch]))
    monkeypatch.setattr(tf_data, "batch_tf_to_device", lambda *args, **kwargs: {})
    class Model:
        def __init__(self, cfg):
            self.fail = fail_second_seed and cfg.training.seed == runtime.SEEDS[1]
        def load_state_dict(self, *args, **kwargs):
            pass
        def to(self, *args):
            return self
        def eval(self):
            return self
        def __call__(self, x, **kwargs):
            output = x.clone()
            if self.fail:
                output[1, 0, 5] = float("nan")
            return output
    monkeypatch.setattr(models, "build_crd_model", Model)
    output = root / "evaluation"
    if fail_second_seed:
        with pytest.raises(EvaluationFailure, match="NaN"):
            runtime.evaluate(output, device="cuda:0", command="synthetic test")
        failure = json.loads((output / "failure.json").read_text())
        assert failure["seed"] == runtime.SEEDS[1]
        assert failure["dataset_row_id"] == 1
        assert failure["method"] == "W0" and failure["checkpoint"]
        assert not (output / "summary.json").exists()
        assert not (output / "receipt.json").exists()
        assert not (output / f"seed_{runtime.SEEDS[2]}").exists()
    else:
        runtime.evaluate(output, device="cuda:0", command="synthetic test")
        summary = json.loads((output / "summary.json").read_text())
        assert summary["seeds"] == list(runtime.SEEDS)
        assert summary["metrics"][METRICS[0]]["n_per_seed"] == 2
        assert summary["metrics"][METRICS[1]]["n_per_seed"] == 1
        assert summary["metrics"][METRICS[2]]["mean"] == 0
        assert json.loads((output / "receipt.json").read_text())["status"] == "complete"
        manifest = json.loads((output / "artifact_manifest.json").read_text())
        for relative, identity in manifest["files"].items():
            runtime.verify(output / relative, identity)
        assert not (output / "failure.json").exists()
    with pytest.raises(FileExistsError):
        runtime.evaluate(output, device="cuda:0", command="synthetic duplicate")
