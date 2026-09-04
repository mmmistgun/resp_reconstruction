import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resp_train.paper_evidence import p5_functional_evidence as p5


def _synthetic_summary() -> pd.DataFrame:
    rows = []
    for seed_index, seed in enumerate(p5.SEEDS):
        for intervention_index, intervention in enumerate(p5.INTERVENTIONS):
            row = {
                "summary_level": "seed",
                "seed": seed,
                "intervention": intervention,
                "selected_epoch": seed_index + 1,
                "checkpoint_sha256": str(seed),
                "n_samples": 2675,
            }
            for metric_index, metric in enumerate(p5.ERROR_METRICS):
                full = 1.0 + seed_index + metric_index
                row[metric] = full * (1.0 + 0.01 * intervention_index)
            row[p5.PCC_METRIC] = 0.90 - 0.01 * seed_index - 0.002 * intervention_index
            rows.append(row)

    seed_frame = pd.DataFrame(rows)
    aggregate_rows = []
    for intervention in p5.INTERVENTIONS:
        group = seed_frame.loc[seed_frame["intervention"].eq(intervention)]
        row = {
            "summary_level": "three_seed_mean",
            "seed": np.nan,
            "intervention": intervention,
            "selected_epoch": np.nan,
            "checkpoint_sha256": np.nan,
            "n_samples": float(group["n_samples"].mean()),
        }
        for metric in p5.METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            row[metric] = float(np.mean(values))
            row[f"{metric}_seed_sd"] = float(np.std(values, ddof=1))
        aggregate_rows.append(row)
    return pd.concat([seed_frame, pd.DataFrame(aggregate_rows)], ignore_index=True)


def test_delta_orientation_pairing_and_paper_subset():
    summary = _synthetic_summary()
    p5._validate_intervention_summary(summary)
    seed = p5.build_seed_deltas(summary)
    aggregate = p5.build_aggregate_deltas(summary, seed)
    paper = p5.build_paper_table(aggregate)

    assert len(seed) == 9 * 3 * 5
    assert len(aggregate) == 9 * 5
    assert paper["intervention"].tolist() == list(p5.PAPER_INTERVENTIONS)
    assert "whole_rr_aggregate_degradation" in paper
    assert "whole_rr_paired_seed_degradation_mean" in paper
    assert "whole_rr_paired_seed_degradation_sample_sd" in paper
    condition_whole = aggregate.loc[
        aggregate["intervention"].eq("CONDITION_OFF")
        & aggregate["metric"].eq("whole_rr_abs_error_bpm_mean")
    ].iloc[0]
    assert condition_whole["aggregate_mean_oriented_degradation"] == pytest.approx(0.03)
    assert condition_whole["paired_seed_oriented_degradation_mean"] == pytest.approx(0.03)
    assert condition_whole["worsened_seed_count"] == 3
    condition_pcc = aggregate.loc[
        aggregate["intervention"].eq("CONDITION_OFF")
        & aggregate["metric"].eq(p5.PCC_METRIC)
    ].iloc[0]
    assert condition_pcc["aggregate_mean_oriented_degradation"] == pytest.approx(0.006)


def test_validator_rejects_duplicate_and_nonfinite_seed_matrix():
    summary = _synthetic_summary()
    summary.loc[1, "intervention"] = "FULL"
    with pytest.raises(RuntimeError, match="matrix"):
        p5._validate_intervention_summary(summary)

    summary = _synthetic_summary()
    summary.loc[0, p5.ERROR_METRICS[0]] = np.inf
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        p5._validate_intervention_summary(summary)


def test_film_summary_uses_sample_direct_seed_means_and_ddof1():
    rows = []
    for seed_index, seed in enumerate(p5.SEEDS):
        for sample_value in (1.0, 3.0):
            row = {"seed": seed, "intervention": "FULL", "dataset_row_id": sample_value}
            row.update({column: sample_value + seed_index for column in p5.FILM_COLUMNS})
            rows.append(row)
    seed_rows, aggregate = p5.build_film_summary(pd.DataFrame(rows))
    assert seed_rows[p5.FILM_COLUMNS[0]].tolist() == [2.0, 3.0, 4.0]
    row = aggregate.loc[aggregate["statistic"].eq(p5.FILM_COLUMNS[0])].iloc[0]
    assert row["three_seed_mean"] == 3.0
    assert row["three_seed_sample_sd"] == 1.0


def test_no_overwrite_precedes_git_or_source_access(tmp_path, monkeypatch):
    output = tmp_path / p5.OUTPUT_DIR
    output.mkdir(parents=True)
    sentinel = output / "user.txt"
    sentinel.write_text("保留", encoding="utf-8")
    monkeypatch.setattr(p5, "_require_clean_git", lambda _: pytest.fail("不应检查 Git"))
    monkeypatch.setattr(p5, "load_frozen_sources", lambda: pytest.fail("不应读取来源"))
    with pytest.raises(FileExistsError):
        p5.run_p5_summary(repo_root=tmp_path, command="test")
    assert sentinel.read_text(encoding="utf-8") == "保留"


def test_frozen_sources_and_expected_primary_results(tmp_path):
    if not (p5.SOURCE_ROOT / "p_minus_1_manifest.json").is_file():
        pytest.skip("冻结运行产物不随 Git 分发")
    summary, film, correction, records = p5.load_frozen_sources()
    seed = p5.build_seed_deltas(summary)
    aggregate = p5.build_aggregate_deltas(summary, seed)
    paper = p5.build_paper_table(aggregate)
    film_seed, film_aggregate = p5.build_film_summary(film)
    p5.render_functional_panel(aggregate, tmp_path / "panel.png")

    assert len(summary) == 40 and len(film) == 8025
    assert len(seed) == 135 and len(aggregate) == 45 and len(paper) == 7
    assert len(film_seed) == 3 and len(film_aggregate) == 8
    assert correction["source_decision_unchanged"] is True
    assert len(records) == 5 and (tmp_path / "panel.png").stat().st_size > 0

    indexed = aggregate.set_index(["intervention", "metric"])
    assert indexed.loc[("CONDITION_OFF", "global_envelope_modulation_error_mean"),
                       "aggregate_mean_oriented_degradation"] == pytest.approx(0.2040, abs=5e-5)
    assert indexed.loc[("TIME_MEAN", "global_envelope_modulation_error_mean"),
                       "aggregate_mean_oriented_degradation"] == pytest.approx(0.2597, abs=5e-5)
    assert indexed.loc[("TIME_SHIFT_30S", p5.PCC_METRIC),
                       "aggregate_mean_oriented_degradation"] == pytest.approx(0.02382, abs=5e-5)
    film_values = film_aggregate.set_index("statistic")["three_seed_mean"]
    assert film_values["mean_abs_gamma"] == pytest.approx(0.26685546, abs=5e-9)
    assert film_values["gamma_time_mean_abs_difference"] == pytest.approx(0.02261490, abs=5e-9)


def test_cli_has_no_source_or_output_override():
    script = Path(__file__).resolve().parents[1] / "scripts/summarize_paper_p5_functional_evidence_v1.py"
    text = script.read_text(encoding="utf-8")
    assert "--source" not in text and "--output" not in text
    assert "run_p5_summary" in text


def test_receipt_contract_is_json_finite(monkeypatch, tmp_path):
    summary = _synthetic_summary()
    film_rows = []
    for seed in p5.SEEDS:
        for row_id in range(2):
            row = {"seed": seed, "intervention": "FULL", "dataset_row_id": row_id}
            row.update({column: 0.1 for column in p5.FILM_COLUMNS})
            film_rows.append(row)
    source = [{"role": "synthetic", "path": str(tmp_path / "source"), "sha256": "0" * 64, "size_bytes": 0}]
    (tmp_path / "source").write_bytes(b"")
    source[0]["sha256"] = p5.sha256_file(tmp_path / "source")
    monkeypatch.setattr(p5, "_require_clean_git", lambda _: "abc")
    monkeypatch.setattr(
        p5,
        "load_frozen_sources",
        lambda: (summary, pd.DataFrame(film_rows), {"correction_scope": "synthetic"}, source),
    )
    output = p5.run_p5_summary(repo_root=tmp_path, command="synthetic")
    receipt = json.loads((output / "summary_receipt.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert receipt["access"]["research_test_read"] is False
    assert receipt["counts"]["aggregate_delta_rows"] == 45
    assert {item["filename"] for item in manifest["files"]} >= {
        "paper_functional_table.csv", "summary_receipt.json"
    }
