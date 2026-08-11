from __future__ import annotations

import pandas as pd

from resp_train.crd.c1_selection import SEED_SUMMARY_METRICS
from resp_train.crd.c2_selection import C201_VARIANT, C202_VARIANT, apply_c2_decision
from resp_train.crd.config import FORMAL_SEEDS
from resp_train.crd.s2_selection import BASE_VARIANT


def _summary(
    *,
    c201_local: tuple[float, float, float] = (0.54, 0.55, 0.56),
    c202_local: tuple[float, float, float] = (0.53, 0.54, 0.55),
    c201_trajectory: float = 0.151,
    c202_trajectory: float = 0.151,
) -> pd.DataFrame:
    rows = []
    for position, seed in enumerate(FORMAL_SEEDS):
        for variant, local, trajectory, pcc in (
            (BASE_VARIANT, (0.55, 0.56, 0.57)[position], 0.15, 0.865),
            (C201_VARIANT, c201_local[position], c201_trajectory, 0.864),
            (C202_VARIANT, c202_local[position], c202_trajectory, 0.864),
        ):
            values = {
                "whole_rr_abs_error_bpm": 0.5,
                "local_rr_mae_bpm": local,
                "envelope_trajectory_mae": trajectory,
                "global_envelope_modulation_error": 0.19,
                "lag_aware_signed_pcc": pcc,
                "ibi_medae_sec": 0.08,
                "ibi_coverage": 0.84,
            }
            rows.append(
                {"variant": variant, "seed": seed, **{f"{key}_mean": values[key] for key in SEED_SUMMARY_METRICS}}
            )
    return pd.DataFrame(rows)


def test_c2_selects_100hz_only_when_basic_and_placement_both_pass() -> None:
    decision = apply_c2_decision(_summary())
    assert decision["basic_eligibility"][C201_VARIANT]["passed"] is True
    assert decision["basic_eligibility"][C202_VARIANT]["passed"] is True
    assert decision["c202_vs_c201_placement"]["passed"] is True
    assert decision["selected_variant"] == C202_VARIANT
    assert decision["outcome"] == "100hz_nonlinear_placement_supported"


def test_c2_selects_10hz_capacity_when_100hz_has_no_incremental_gain() -> None:
    decision = apply_c2_decision(
        _summary(c201_local=(0.54, 0.55, 0.56), c202_local=(0.541, 0.551, 0.561))
    )
    assert decision["basic_eligibility"][C201_VARIANT]["passed"] is True
    assert decision["c202_vs_c201_placement"]["passed"] is False
    assert decision["selected_variant"] == C201_VARIANT
    assert decision["outcome"] == "decoder_capacity_supported_100hz_placement_not_supported"


def test_c2_retains_base_when_both_basic_gates_fail() -> None:
    decision = apply_c2_decision(
        _summary(
            c201_local=(0.56, 0.57, 0.58),
            c202_local=(0.56, 0.57, 0.58),
            c201_trajectory=0.154,
            c202_trajectory=0.154,
        )
    )
    assert decision["basic_eligibility"][C201_VARIANT]["passed"] is False
    assert decision["basic_eligibility"][C202_VARIANT]["passed"] is False
    assert decision["selected_variant"] == BASE_VARIANT
    assert decision["outcome"] == "retain_original_10hz_fourier_decoder"

