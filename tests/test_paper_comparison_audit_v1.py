from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from resp_train.paper_evidence.comparison_audit import (
    COMMON_CONTRACT,
    P0_SCHEMA_VERSION,
    PAPER_EVIDENCE_PROTOCOL_ID,
    PRIMARY_METRICS,
    load_p0_source_manifest,
    run_p0_comparison_audit,
)


def _contracts(*, deterministic: bool) -> dict:
    return {
        **COMMON_CONTRACT,
        "input_carrier": "fixture_bcg",
        "checkpoint_selector": (
            "not_applicable_deterministic"
            if deterministic
            else "full_validation_local_rr_strict_lower_tie_earlier"
        ),
        "prediction_uses_target": False,
    }


def _record(root: Path, name: str, *, seed: int | None, offset: float) -> dict:
    row_ids = np.arange(2310, dtype=np.int64)
    sample = pd.DataFrame(
        {
            "dataset_row_id": row_ids,
            "split": ["test"] * row_ids.size,
            **{
                metric: row_ids.astype(np.float64) + 1.0 + offset + index
                for index, metric in enumerate(PRIMARY_METRICS)
            },
        }
    )
    sample_path = root / f"{name}_sample.csv"
    sample.to_csv(sample_path, index=False)
    summary = {
        f"{metric}_mean": float(sample[metric].mean())
        for metric in PRIMARY_METRICS
    }
    summary_path = root / f"{name}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return {
        "seed": seed,
        "split": "test",
        "sample_metrics": sample_path.name,
        "summary": summary_path.name,
    }


def test_p0_audit_is_read_only_non_overwriting_and_uses_ddof1(tmp_path: Path) -> None:
    learned_records = [
        _record(tmp_path, f"learned_{seed}", seed=seed, offset=float(index))
        for index, seed in enumerate((20260811, 20260812, 20260813))
    ]
    deterministic_record = _record(tmp_path, "deterministic", seed=None, offset=0.5)
    source = {
        "protocol_id": PAPER_EVIDENCE_PROTOCOL_ID,
        "schema_version": P0_SCHEMA_VERSION,
        "representative_method_rule": {
            "primary_table_method_ids": ["learned", "deterministic"],
            "rule": "synthetic_predeclared_fixture",
        },
        "methods": [
            {
                "method_id": "learned",
                "paper_role": "primary proposed model",
                "quality_scope": "independent_test",
                "deterministic": False,
                "expected_seed_count": 3,
                "conclusion_lock_confirmed": True,
                "include_in_primary_table": True,
                "contracts": _contracts(deterministic=False),
                "records": learned_records,
            },
            {
                "method_id": "deterministic",
                "paper_role": "traditional baseline",
                "quality_scope": "independent_test",
                "deterministic": True,
                "conclusion_lock_confirmed": True,
                "include_in_primary_table": True,
                "contracts": _contracts(deterministic=True),
                "records": [deterministic_record],
            },
        ],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "audit"
    manifest = run_p0_comparison_audit(
        source_manifest_path=source_path,
        output_dir=output,
        repo_root=tmp_path,
        require_clean_git=False,
    )
    assert manifest == output / "manifest.json"
    compatibility = pd.read_csv(output / "protocol_compatibility.csv")
    assert set(compatibility["compatibility"]) == {"compatible"}
    table = pd.read_csv(output / "paper_primary_metrics_table.csv").set_index("method_id")
    assert table.loc["learned", "whole_rr_abs_error_bpm_mean"] == 1156.5
    assert table.loc["learned", "whole_rr_abs_error_bpm_sample_sd"] == 1.0
    assert table.loc["deterministic", "seed_semantics"] == "deterministic_no_seed"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dataset_signal_or_target_read"] is False
    assert payload["prediction_generation_used"] is False
    try:
        run_p0_comparison_audit(
            source_manifest_path=source_path,
            output_dir=output,
            repo_root=tmp_path,
            require_clean_git=False,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("P0 audit 必须拒绝覆盖")


def test_p0_unconfirmed_lock_stays_out_of_primary_table(tmp_path: Path) -> None:
    record = _record(tmp_path, "rtm", seed=None, offset=0.0)
    source = {
        "protocol_id": PAPER_EVIDENCE_PROTOCOL_ID,
        "schema_version": P0_SCHEMA_VERSION,
        "representative_method_rule": {"primary_table_method_ids": ["locked"]},
        "methods": [
            {
                "method_id": "locked",
                "paper_role": "primary proposed model",
                "quality_scope": "independent_test",
                "deterministic": True,
                "conclusion_lock_confirmed": True,
                "include_in_primary_table": True,
                "contracts": _contracts(deterministic=True),
                "records": [record],
            },
            {
                "method_id": "rtm_unconfirmed",
                "paper_role": "independent task candidate",
                "quality_scope": "independent_test",
                "deterministic": True,
                "conclusion_lock_confirmed": False,
                "include_in_primary_table": False,
                "contracts": _contracts(deterministic=True),
                "records": [record],
            },
        ],
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    run_p0_comparison_audit(
        source_manifest_path=path,
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        require_clean_git=False,
    )
    table = pd.read_csv(tmp_path / "out/paper_primary_metrics_table.csv")
    assert table["method_id"].tolist() == ["locked"]


def test_repository_p0_source_manifest_covers_frozen_candidate_families() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = load_p0_source_manifest(root / "configs/paper_evidence_v1/p0_comparison_sources.json")
    ids = {method["method_id"] for method in payload["methods"]}
    assert {
        "F0_fixed_band_bcg",
        "IEWT",
        "B0_final_loss_patchmixer",
        "T2_g3c_wide_native",
        "T4_g3c_bandenergy_native",
        "crd_c201_decoder_10hz_cap",
        "crd_tf101_m",
        "crd_tf102_w",
        "crd_tf203_ms",
        "crd_tfw_v2_w3_full_6v_film_d6",
        "crd_tfw_v2_d4_full_12v_film",
        "rtm_v1_all_five_unconfirmed",
    }.issubset(ids)
    rtm = next(method for method in payload["methods"] if method["method_id"] == "rtm_v1_all_five_unconfirmed")
    assert len(rtm["record_matrix"]["candidate_ids"]) == 5
    assert rtm["conclusion_lock_confirmed"] is False
