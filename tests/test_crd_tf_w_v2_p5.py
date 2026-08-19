from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from resp_train.crd.config import FORMAL_SEEDS
from resp_train.crd.tf_v1_research_test_summary import SECONDARY_METRICS
from resp_train.crd.tf_v1_selection import PRIMARY_METRICS
from resp_train.crd.tf_w_v2 import CANDIDATE_LOCK
from resp_train.crd.tf_w_v2_p5 import (
    ALLOWLIST,
    ALLOWLIST_SHA256,
    CANDIDATE_ID,
    CANDIDATE_VARIANT,
    EVALUATION_FILES,
    EVIDENCE_ROLE,
    EXPECTED_TEST_SAMPLES,
    REUSED_REFERENCE_VARIANTS,
    SUMMARY_FILES,
    _aggregate,
    audit_p5_inputs,
    paired_vs_references,
)


@pytest.fixture(scope="module")
def audited_inputs() -> dict:
    return audit_p5_inputs()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_seed_metrics() -> pd.DataFrame:
    variants = (*REUSED_REFERENCE_VARIANTS, CANDIDATE_VARIANT)
    base = {
        "crd_c201_decoder_10hz_cap": (0.70, 0.66, 0.145, 0.170, 0.875),
        "crd_tf102_w": (0.62, 0.61, 0.140, 0.173, 0.876),
        CANDIDATE_VARIANT: (0.60, 0.59, 0.138, 0.171, 0.877),
    }
    rows = []
    for variant in variants:
        for index, seed in enumerate(FORMAL_SEEDS):
            primary = [value + index * 0.001 for value in base[variant]]
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    **dict(zip(PRIMARY_METRICS, primary)),
                    **{name: 0.5 + index * 0.01 for name in SECONDARY_METRICS},
                }
            )
    return pd.DataFrame(rows)


def test_p5_allowlist_is_frozen_to_w3_three_seed_checkpoints(audited_inputs: dict) -> None:
    allowlist = audited_inputs["allowlist"]
    identities = audited_inputs["checkpoint_identities"]

    assert _sha256(ALLOWLIST) == ALLOWLIST_SHA256
    assert allowlist["evidence_role"] == EVIDENCE_ROLE
    assert allowlist["candidate_id"] == CANDIDATE_ID
    assert allowlist["model_variant"] == CANDIDATE_VARIANT
    assert [item["seed"] for item in identities] == list(FORMAL_SEEDS)
    assert [item["selected_epoch"] for item in identities] == [9, 12, 12]
    assert len({item["checkpoint_sha256"] for item in identities}) == 3
    assert allowlist["selection_source"]["strict_efficiency_candidate"] is None


def test_p5_allowlist_retains_d4_tradeoff_but_forbids_its_test_evaluation(audited_inputs: dict) -> None:
    allowlist = audited_inputs["allowlist"]

    assert allowlist["selection_source"]["descriptive_quality_efficiency_pareto"] == [
        CANDIDATE_ID,
        "D4_W0",
    ]
    assert "descriptive non-catastrophic efficiency trade-off" in allowlist["selection_source"][
        "d4_exclusion_reason"
    ]
    assert allowlist["constraints"]["d4_evaluation_allowed"] is False
    assert allowlist["constraints"]["w1_w2_evaluation_allowed"] is False


def test_p5_test_cache_contract_is_input_only_full_6v(audited_inputs: dict) -> None:
    cache = audited_inputs["allowlist"]["test_w_cache"]

    assert cache["target_read"] is False
    assert cache["view"] == "full_6v"
    assert cache["source_shape_per_sample"] == [97, 360]
    assert cache["model_input_shape_per_sample"] == [49, 360]
    assert cache["active_scale_count"] == 49
    assert cache["view_index_sha256"] == "3734c4183c17744eb4779ac65ac8521de1897f6a12f364e913aeb02528e250d6"


def test_p5_candidate_lock_routes_only_to_dedicated_entry() -> None:
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    command = lock["command_templates"]["p5"]

    assert "scripts/eval_crd_tf_w_v2_research_test.py" in command
    assert "--confirm-research-test" in command
    assert lock["forbidden"]["ordinary_eval_crd_test"] is True
    assert lock["summary_schema"]["samp_id_analysis_allowed"] is False


def test_p5_paired_comparison_is_descriptive_and_keeps_seed_direction() -> None:
    paired = paired_vs_references(_synthetic_seed_metrics())

    assert len(paired) == len(REUSED_REFERENCE_VARIANTS) * len(PRIMARY_METRICS)
    assert paired["comparison_role"].eq("descriptive_only_not_selection").all()
    local_vs_w0 = paired.loc[
        (paired["reference"] == "crd_tf102_w") & (paired["metric"] == "local_rr_mae")
    ].iloc[0]
    assert float(local_vs_w0["relative_delta"]) == pytest.approx(0.591 / 0.611 - 1.0)
    assert int(local_vs_w0["better_seed_count"]) == 3
    pcc_vs_w0 = paired.loc[
        (paired["reference"] == "crd_tf102_w") & (paired["metric"] == "signed_pcc")
    ].iloc[0]
    assert pd.isna(pcc_vs_w0["relative_delta"])
    assert float(pcc_vs_w0["absolute_delta"]) == pytest.approx(0.001)


def test_p5_aggregate_requires_all_three_fixed_seeds() -> None:
    frame = _synthetic_seed_metrics()
    aggregate = _aggregate(frame, PRIMARY_METRICS, (*REUSED_REFERENCE_VARIANTS, CANDIDATE_VARIANT))

    assert len(aggregate) == 3 * len(PRIMARY_METRICS)
    assert set(aggregate.columns) >= {f"seed_{seed}" for seed in FORMAL_SEEDS}
    incomplete = frame.loc[
        ~((frame["variant"] == CANDIDATE_VARIANT) & (frame["seed"] == FORMAL_SEEDS[-1]))
    ]
    with pytest.raises(RuntimeError, match="seeds 不完整"):
        _aggregate(incomplete, PRIMARY_METRICS, (*REUSED_REFERENCE_VARIANTS, CANDIDATE_VARIANT))


def test_p5_output_contract_is_isolated_and_has_no_selection_score(audited_inputs: dict) -> None:
    allowlist = audited_inputs["allowlist"]

    assert allowlist["output_contract"]["root"] == "runs/crd_tf_w_v2/research_test"
    assert allowlist["output_contract"]["overwrite_allowed"] is False
    assert allowlist["output_contract"]["resume_by_validated_skip_allowed"] is True
    assert set(EVALUATION_FILES) == {
        "research_test_metrics.csv",
        "research_test_metrics_summary.csv",
        "research_test_metrics_manifest.json",
    }
    assert set(SUMMARY_FILES) == {
        "evaluation_audit.csv",
        "seed_metrics.csv",
        "aggregate_primary.csv",
        "aggregate_secondary.csv",
        "paired_vs_references.csv",
        "p5_summary.json",
        "p5_summary_manifest.json",
    }
    assert allowlist["constraints"]["samp_id_analysis_allowed"] is False
    assert allowlist["constraints"]["total_score_allowed"] is False
    assert allowlist["constraints"]["new_training_runs"] == 0
    assert EXPECTED_TEST_SAMPLES == 2310
