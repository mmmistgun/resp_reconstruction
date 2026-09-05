import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resp_train.paper_evidence import p6_rr_strata as p6


def dataset_contract(row_ids=(1, 2, 3, 4)):
    ids = np.asarray(row_ids, dtype="<i8")
    return {
        "test_count": len(ids),
        "test_samp_id_count": 2,
        "test_row_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "target_key": "target",
        "target_signal_key": "target",
        "window_samples": 18000,
    }


def identity_frame():
    return pd.DataFrame(
        {
            "dataset_row_id": [1, 2, 3, 4],
            "split": "test",
            "input_set": "research_v2_waveform",
            "samp_id": [10, 10, 20, 20],
            "coupling_state_id": [1, 1, 2, 2],
        }
    )


def target_attributes():
    return identity_frame().assign(
        target_sha256=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(4)],
        target_rr_bpm=[10.0, 14.0, 16.0, 20.0],
        target_rr_eligible=True,
        target_envelope_modulation=[0.1, 0.2, 0.3, 0.4],
        rr_stratum=["low", "low", "medium", "high"],
    )


def metric_frame(offset=0.0):
    frame = identity_frame().assign(
        method="source_method",
        target_envelope_modulation=[0.1, 0.2, 0.3, 0.4],
    )
    for metric in p6.PRIMARY_METRICS[:-1]:
        frame[metric] = np.arange(1.0, 5.0) + offset
    frame[p6.PRIMARY_METRICS[-1]] = np.asarray([0.1, 0.2, 0.3, 0.4]) + offset / 100.0
    return frame


def test_contract_identity_and_frozen_matrix():
    contract = p6.load_contract()
    assert contract["dataset"]["test_count"] == 2310
    assert contract["dataset"]["test_samp_id_count"] == 8
    assert len(contract["method_sources"]["method_allowlist"]) == 10
    assert contract["rr_strata"]["cutpoints_bpm"] == pytest.approx(
        [14.327967747931218, 17.188694745285627]
    )


def test_test_row_identity_fail_closed():
    rows = identity_frame().assign(
        target_source_npz="target.npz",
        target_signal_key="target",
        window_start_sample=0,
        window_end_sample=18000,
    )
    p6.validate_test_rows(rows, dataset_contract())
    duplicate = rows.copy()
    duplicate.loc[1, "dataset_row_id"] = 1
    with pytest.raises(ValueError, match="递增"):
        p6.validate_test_rows(duplicate, dataset_contract())
    wrong_split = rows.copy()
    wrong_split.loc[0, "split"] = "val"
    with pytest.raises(ValueError, match="split"):
        p6.validate_test_rows(wrong_split, dataset_contract())


def test_train_frozen_boundary_assignment_and_counts():
    attrs = target_attributes()
    attrs["rr_stratum"] = np.where(
        attrs["target_rr_bpm"] <= 14.0,
        "low",
        np.where(attrs["target_rr_bpm"] <= 17.0, "medium", "high"),
    )
    counts = p6.rr_stratum_counts(attrs, ["low", "medium", "high"])
    assert counts["window_count"].tolist() == [2, 1, 1]
    assert counts["samp_id_count"].tolist() == [1, 1, 1]
    assert counts["window_count"].sum() == 4


def test_metric_join_and_modulation_anchor():
    validated = p6.validate_metric_frame(
        metric_frame(), target_attributes(), dataset_contract(), modulation_atol=1e-10
    )
    assert validated["rr_stratum"].tolist() == ["low", "low", "medium", "high"]
    bad = metric_frame()
    bad.loc[0, "target_envelope_modulation"] += 1e-3
    with pytest.raises(RuntimeError, match="modulation"):
        p6.validate_metric_frame(
            bad, target_attributes(), dataset_contract(), modulation_atol=1e-10
        )
    bad = metric_frame()
    bad.loc[0, p6.PRIMARY_METRICS[0]] = np.nan
    with pytest.raises(FloatingPointError, match="finite"):
        p6.validate_metric_frame(
            bad, target_attributes(), dataset_contract(), modulation_atol=1e-10
        )


def test_sample_direct_then_seed_mean_and_sd():
    attrs = target_attributes()
    record_rows = []
    for method_order, (method_id, deterministic) in enumerate(
        [("det", True), ("learned", False)]
    ):
        seeds = [None] if deterministic else [11, 12, 13]
        for seed_index, seed in enumerate(seeds):
            frame = metric_frame(float(seed_index)).assign(rr_stratum=attrs["rr_stratum"])
            record_rows.extend(
                p6.summarize_metric_record(
                    frame,
                    method_order=method_order,
                    method_id=method_id,
                    display_name=method_id,
                    deterministic=deterministic,
                    seed=seed,
                    source_method=method_id,
                    source_path=f"{method_id}.csv",
                    strata_order=["low", "medium", "high"],
                )
            )
    summary = p6.aggregate_method_strata(
        pd.DataFrame(record_rows),
        method_allowlist=["det", "learned"],
        learned_seeds=[11, 12, 13],
        strata_order=["low", "medium", "high"],
    )
    assert len(summary) == 6
    learned_low = summary.loc[
        summary["method_id"].eq("learned") & summary["rr_stratum"].eq("low")
    ].iloc[0]
    assert learned_low[f"{p6.PRIMARY_METRICS[0]}_mean"] == pytest.approx(2.5)
    assert learned_low[f"{p6.PRIMARY_METRICS[0]}_sample_sd"] == pytest.approx(1.0)
    deterministic = summary.loc[summary["method_id"].eq("det")]
    assert deterministic["seed_count"].eq(0).all()
    assert deterministic[f"{p6.PRIMARY_METRICS[0]}_sample_sd"].isna().all()


def test_output_rejected_before_contract_or_source_access(tmp_path, monkeypatch):
    output = tmp_path / p6.TEST_TARGET_OUTPUT
    output.mkdir(parents=True)
    sentinel = output / "user.txt"
    sentinel.write_text("保留", encoding="utf-8")
    monkeypatch.setattr(p6, "load_contract", lambda _: pytest.fail("不应读取合同"))
    with pytest.raises(FileExistsError):
        p6.build_test_target_attributes(repo_root=tmp_path, command="test")
    assert sentinel.read_text(encoding="utf-8") == "保留"


def test_cli_confirmation_and_closed_matrix():
    root = Path(__file__).resolve().parents[1]
    build = (root / "scripts/build_paper_p6_test_target_attributes_v1.py").read_text()
    summary = (root / "scripts/summarize_paper_p6_test_rr_strata_v1.py").read_text()
    assert "--confirm-test-target-attribute-build" in build
    assert "--split" not in build + summary
    assert "--checkpoint" not in build + summary
    assert "--method" not in build + summary
    assert "--output" not in build + summary
    json.dumps(p6.load_contract(), allow_nan=False)
