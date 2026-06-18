import numpy as np

from resp_train.metrics.signal import motion_dominance_metrics, robust_zscore
from scripts.audit_motion_dominance import _recommended_scope


def test_robust_zscore_preserves_regular_region_when_one_spike_inflates_std():
    t = np.linspace(0, 6 * np.pi, 1800)
    clean = np.sin(t)
    spiked = clean.copy()
    spiked[900] = 80.0
    regular = np.ones_like(spiked, dtype=bool)
    regular[880:920] = False

    standard = (spiked - float(np.mean(spiked))) / float(np.std(spiked))
    robust = robust_zscore(spiked)

    assert float(np.std(robust[regular])) > float(np.std(standard[regular])) * 5.0


def test_motion_dominance_metrics_flags_short_extreme_artifact():
    t = np.linspace(0, 6 * np.pi, 1800)
    clean = np.sin(t)
    spiked = clean.copy()
    spiked[850:870] += 40.0

    clean_metrics = motion_dominance_metrics(clean)
    spiked_metrics = motion_dominance_metrics(spiked)

    assert clean_metrics["motion_dominated"] is False
    assert spiked_metrics["motion_dominated"] is True
    assert spiked_metrics["std_to_robust_scale"] > clean_metrics["std_to_robust_scale"]
    assert spiked_metrics["top1pct_energy_fraction"] > clean_metrics["top1pct_energy_fraction"]


def test_recommended_scope_distinguishes_input_and_target_artifacts():
    assert _recommended_scope(input_dominated=False, target_dominated=False) == "task_regular"
    assert _recommended_scope(input_dominated=True, target_dominated=False) == "task_input_robustness"
    assert _recommended_scope(input_dominated=False, target_dominated=True) == "dataset_label_review"
    assert _recommended_scope(input_dominated=True, target_dominated=True) == "dataset_quality_flag"
