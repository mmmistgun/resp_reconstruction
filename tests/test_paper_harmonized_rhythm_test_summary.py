import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resp_train.paper_evidence import harmonized_rhythm_test_summary as summary


def sample_frame():
    return pd.DataFrame({
        "dataset_row_id": [1, 2, 3, 4], "whole_rr_abs_error_bpm": [1., 2., 39., np.nan],
        "whole_rr_target_eligible": [True, True, True, False],
        "ibi_medae_sec": [0.1, 0.3, np.nan, np.nan], "ibi_coverage": [1., 0.8, 0., np.nan],
        "ibi_interpretable": [True, True, False, False], "ibi_target_eligible": [True, True, True, False],
    })


def test_sample_direct_mean_and_eligible_denominator():
    row, support = summary.summarize_samples(sample_frame(), "")
    assert row["native_whole_rr_mae_bpm"] == 14
    assert row["ibi_medae_sec"] == pytest.approx(0.2)
    assert row["ibi_coverage"] == pytest.approx(0.6)
    assert row["ibi_interpretable_fraction"] == pytest.approx(2 / 3)
    assert row["ibi_valid_n"] == 2 and row["ibi_target_eligible_n"] == 3
    assert support.eligible.tolist() == [True, True, True, False]


@pytest.mark.parametrize("column,value", [
    ("ibi_medae_sec", np.inf), ("ibi_medae_sec", np.nan), ("ibi_coverage", -1.),
    ("ibi_coverage", 1.1), ("ibi_coverage", 0.7), ("ibi_interpretable", "unknown"),
    ("whole_rr_abs_error_bpm", np.nan), ("whole_rr_abs_error_bpm", -1.),
])
def test_invalid_metrics_fail_closed(column, value):
    frame = sample_frame().astype({column: object})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        summary.summarize_samples(frame, "")


def test_all_ineligible_remains_explicit_nan():
    frame = sample_frame().iloc[[3]]
    row, _ = summary.summarize_samples(frame, "")
    assert np.isnan(row["ibi_medae_sec"]) and row["ibi_valid_n"] == 0
    assert np.isnan(row["ibi_interpretable_fraction"])


def seed_rows():
    rows = []
    for seed, metric, count in zip(summary.SEEDS, [1., 2., 6.], [10, 100, 1000]):
        rows.append({"task": "center30", "model": "C201-center30", "method_id": "c201_center30",
            "input_sec": 30, "output_sec": 30, "seed": seed, "deterministic": False,
            "seed_semantics": "three_training_seeds", "conclusion_lock_status": "confirmed",
            "checkpoint_selector": "validation_rr", "native_rr_source_column": "center30_rr_mae_bpm", "fft_rr_spacing_bpm": 2.,
            **{m: metric for m in summary.VALUES}, **{c: count for c in summary.COUNTS}})
    return pd.DataFrame(rows)


def test_arithmetic_seed_mean_ddof1_and_no_pooling():
    row = summary.aggregate_arms(seed_rows()).iloc[0]
    assert row.ibi_medae_sec_mean == 3
    assert row.ibi_medae_sec_sample_sd == pytest.approx(np.sqrt(7))
    frame = seed_rows()
    frame.loc[0, "ibi_medae_sec"] = np.nan
    row = summary.aggregate_arms(frame).iloc[0]
    assert np.isnan(row.ibi_medae_sec_mean) and row.ibi_medae_sec_finite_records == 2


def test_missing_duplicate_seed_and_deterministic_semantics():
    frame = seed_rows()
    with pytest.raises(ValueError, match="三个"):
        summary.aggregate_arms(frame.iloc[:2])
    frame.loc[0, "seed"] = summary.SEEDS[1]
    with pytest.raises(ValueError, match="三个"):
        summary.aggregate_arms(frame)
    frame = seed_rows().iloc[[0]].copy()
    frame["deterministic"] = True
    frame["seed"] = np.nan
    frame["seed_semantics"] = "deterministic-no-seed"
    row = summary.aggregate_arms(frame).iloc[0]
    assert row.seed_count == 0 and np.isnan(row.ibi_medae_sec_sample_sd)


def test_frozen_hash_gate_and_no_checkpoint_read(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("value\n1\n")
    reader = summary.FrozenReader(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        reader.read(path, "0" * 64)
    with pytest.raises(ValueError, match="禁止读取"):
        reader.read(tmp_path / "checkpoint.pt")
    reader.read(path)
    path.write_text("value\n2\n")
    with pytest.raises(ValueError, match="输入发生变化"):
        reader.verify_unchanged()


def test_no_overwrite_before_any_audit(tmp_path, monkeypatch):
    output = tmp_path / summary.OUTPUT_DIR
    output.mkdir(parents=True)
    sentinel = output / "user.txt"
    sentinel.write_text("保留")
    monkeypatch.setattr(summary, "audit_inputs", lambda _: pytest.fail("不应读取输入"))
    with pytest.raises(FileExistsError):
        summary.run_summary(repo_root=tmp_path, command="test")
    assert sentinel.read_text() == "保留"


def test_dirty_git_rejected_before_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(summary.subprocess, "check_output", lambda args, **kwargs: "abc\n" if args[1] == "rev-parse" else "?? user.txt\n")
    monkeypatch.setattr(summary, "audit_inputs", lambda _: pytest.fail("不应读取输入"))
    with pytest.raises(RuntimeError, match="干净"):
        summary.run_summary(repo_root=tmp_path, command="test")


def test_identity_rejects_non_test_or_duplicates():
    frame = pd.DataFrame({"dataset_row_id": [1] * 2310, "split": ["test"] * 2310,
        "input_set": ["x"] * 2310, "samp_id": ["a"] * 2310, "coupling_state_id": ["b"] * 2310})
    with pytest.raises(ValueError, match="重复"):
        summary.validate_identity(frame, None)
    frame["split"] = "val"
    with pytest.raises(ValueError, match="split"):
        summary.validate_identity(frame, None)


def test_stored_summary_disagreement_is_error():
    row, _ = summary.summarize_samples(sample_frame(), "")
    stored = pd.Series({"sample_rows": 4, "whole_rr_abs_error_bpm": 999,
                        "whole_rr_abs_error_bpm_eligible_count": 3})
    with pytest.raises(ValueError, match="summary 数值不一致"):
        summary.validate_stored(row, stored, "", rtm=True)


def test_metric_config_parameter_drift():
    cfg = {"window": {"target_fs": 100}, "loss": {"max_lag_sec": 0.3},
           "evaluation": {**summary.IBI_CONFIG, "local_rr_window_sec": 60, "local_rr_step_sec": 15}}
    summary.validate_metric_config(cfg, historical=True)
    cfg["evaluation"]["ibi_match_tolerance_sec"] = 1
    with pytest.raises(ValueError, match="参数漂移"):
        summary.validate_metric_config(cfg, historical=True)


def test_output_manifest_and_exclusive_directory(tmp_path, monkeypatch):
    reader = summary.FrozenReader(tmp_path)
    monkeypatch.setattr(summary, "audit_inputs", lambda _: (seed_rows(), pd.DataFrame({"seed": summary.SEEDS}), reader))
    monkeypatch.setattr(summary, "build_comparability", lambda *_: {"evidence_scope": "synthetic_test"})
    monkeypatch.setattr(summary.subprocess, "check_output", lambda args, **kwargs: "abc\n" if args[1] == "rev-parse" else "")
    output = summary.run_summary(repo_root=tmp_path, command="synthetic-test")
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert len(manifest["files"]) == 6
    for record in manifest["files"]:
        data = (output / record["filename"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        assert len(data) == record["size_bytes"]
    receipt = json.loads((output / "summary_receipt.json").read_text())
    assert receipt["execution"]["git_dirty"] is False
    assert receipt["access"]["model_inference_used"] is False
    with pytest.raises(FileExistsError):
        summary.run_summary(repo_root=tmp_path, command="synthetic-test")


def test_frozen_inputs_cpu_read_only():
    root = Path(__file__).resolve().parents[1]
    if not (root / summary.BASE / "p0_comparison_audit/manifest.json").is_file():
        pytest.skip("冻结运行产物不随 Git 分发")
    seeds, supports, reader = summary.audit_inputs(root)
    arms = summary.aggregate_arms(seeds)
    ladder = summary.rhythm_ladder(arms)
    assert len(seeds) == 101 and len(arms) == 35 and len(ladder) == 9
    assert seeds.n_samples.sum() == 233310
    assert len(supports) == 21
    for model, expected, sd in [("C201", 0.09743666367316874, 0.003859544159955016),
                               ("W0", 0.09987037375788027, 0.0037629907776207333),
                               ("W3", 0.098556473246063, 0.0004722010547319398)]:
        row = arms.loc[arms.model.eq(model)].iloc[0]
        assert row.ibi_medae_sec_mean == pytest.approx(expected, abs=1e-12)
        assert row.ibi_medae_sec_sample_sd == pytest.approx(sd, abs=1e-12)
    assert len(arms.loc[arms.conclusion_lock_status.eq("pending_user_confirmation_audit_only")]) == 5
    assert arms.loc[arms.deterministic, "seed_count"].eq(0).all()
    assert ladder.fft_rr_spacing_bpm.tolist() == [2., 2., 1., 1., 2 / 3, 2 / 3, 1 / 3, 1 / 3, 1 / 3]
    assert not any("/metrics.csv" in key or key.endswith(".pt") for key in reader.records)
    comparability = summary.build_comparability(arms, supports)
    json.dumps(comparability, allow_nan=False)
    assert comparability["ibi"]["eligible_changed_rows_max"] == 0
    assert comparability["local_rr"]["cross_task_comparable_column_generated"] is False
    assert len(comparability["ibi"]["adjacent_seed_directions"]) == 7
    reader.verify_unchanged()
