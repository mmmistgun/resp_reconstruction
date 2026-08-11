from __future__ import annotations

import numpy as np
import pandas as pd

from resp_train.crd.matched_observability import (
    FIXED_BAND_METHOD,
    RAWISH_METHOD,
    apply_observability_decision,
    select_matched_windows,
)


def _metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_id = 1
    for samp_id in (1, 2):
        for state in (1, 2):
            starts_and_failure: list[tuple[float, bool]] = []
            for episode in range(4):
                base = state * 10000.0 + episode * 1200.0
                starts_and_failure.extend((base + offset, True) for offset in (0.0, 30.0, 60.0))
                starts_and_failure.extend((base + offset, False) for offset in (300.0, 600.0, 900.0))
            for position, (start, failure) in enumerate(starts_and_failure):
                rows.append(
                    {
                        "dataset_row_id": row_id,
                        "samp_id": samp_id,
                        "coupling_state_id": state,
                        "window_start_s": start,
                        "window_end_s": start + 180.0,
                        "envelope_target_stratum": "high",
                        "target_envelope_modulation": 1.0 + 0.01 * position,
                        "waveform_confidence_score": 0.6 + 0.001 * position,
                        "transient_motion_ratio": 0.7 + 0.001 * position,
                        "local_rr_mae_bpm_persistent_failure": failure,
                        "persistent_core_failure_count": 2 if failure else 0,
                    }
                )
                row_id += 1
    return pd.DataFrame(rows)


def test_select_matched_windows_is_deterministic_and_nonoverlapping() -> None:
    first, audit = select_matched_windows(_metadata())
    second, _ = select_matched_windows(_metadata())

    pd.testing.assert_frame_equal(first, second)
    assert audit["primary_exact_state_pair_count"] >= 12
    primary = first.loc[first["match_scheme"].eq("exact_state_primary")]
    assert (primary["case_samp_id"] == primary["control_samp_id"]).all()
    assert (primary["case_coupling_state_id"] == primary["control_coupling_state_id"]).all()
    assert (primary["case_window_start_s"].sub(primary["control_window_start_s"]).abs() >= 180.0).all()
    assert primary["normalized_l1_cost"].le(2.0).all()


def test_observability_decision_uses_both_proxy_primary_metrics() -> None:
    rows = []
    for method, local, pcc in (
        (RAWISH_METHOD, 0.8, 0.7),
        (FIXED_BAND_METHOD, 0.9, 0.8),
    ):
        rows.extend(
            [
                {
                    "match_scheme": "exact_state_primary",
                    "method": method,
                    "metric": "local_rr_mae_bpm",
                    "pair_n": 21,
                    "case_worse_fraction": local,
                },
                {
                    "match_scheme": "exact_state_primary",
                    "method": method,
                    "metric": "lag_aware_signed_pcc",
                    "pair_n": 21,
                    "case_worse_fraction": pcc,
                },
            ]
        )
    decision = apply_observability_decision(pd.DataFrame(rows))
    assert decision["diagnostic_outcome"] == "input_observability_associated"
    assert decision["primary_pair_count"] == 21
