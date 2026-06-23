from pathlib import Path

import pandas as pd
import yaml

from scripts.peak_band_misclass_rate import collect, grouped_summary


def _write_peak_run(root: Path, name: str, *, fusion_mode: str, encoder_type: str, seed: int) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    pd.DataFrame({"rr_peak_band_abs_error": [0.1, 0.2, 1.4, 2.2]}).to_csv(run_dir / "metrics.csv", index=False)
    config = {
        "model": {
            "name": "time_stft_dual1d",
            "time_backbone": "patch_mixer1d",
            "branch_mode": "dual",
            "fusion_mode": fusion_mode,
            "stft_encoder_type": encoder_type,
            "stft_high_hz": 8.0,
            "stft_norm": "n0",
        },
        "training": {"seed": seed},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_peak_band_summary_groups_by_fusion_and_encoder(tmp_path):
    root = tmp_path / "runs"
    _write_peak_run(root, "run_a", fusion_mode="concat_generic", encoder_type="conv2d", seed=1)
    _write_peak_run(root, "run_b", fusion_mode="token_context_inject", encoder_type="bandgroup", seed=2)

    frame = collect([root], thresholds=[1.0], trim_frac=0.05)
    summary = grouped_summary(frame, thresholds=[1.0])

    assert {"fusion_mode", "stft_encoder_type"} <= set(frame.columns)
    assert {"fusion_mode", "stft_encoder_type"} <= set(summary.columns)
    assert set(summary["fusion_mode"]) == {"concat_generic", "token_context_inject"}
    assert set(summary["stft_encoder_type"]) == {"conv2d", "bandgroup"}
