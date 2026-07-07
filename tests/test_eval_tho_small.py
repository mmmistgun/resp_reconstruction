import sys
from pathlib import Path

import scripts.eval_tho_small as eval_small


def test_eval_tho_small_cli_passes_chunk_and_cache_args(monkeypatch, tmp_path: Path, capsys):
    seen = {}

    def fake_evaluate_tho_checkpoint(**kwargs):
        seen.update(kwargs)
        return Path(kwargs["metrics_output_path"])

    monkeypatch.setattr(eval_small, "evaluate_tho_checkpoint", fake_evaluate_tho_checkpoint)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_tho_small.py",
            "--checkpoint",
            "run/checkpoint_top1.pt",
            "--config",
            "run/config.yaml",
            "--metrics-output",
            str(tmp_path / "metrics_top1.csv"),
            "--metrics-workers",
            "4",
            "--metrics-chunk-size",
            "64",
            "--target-cache-dir",
            str(tmp_path / "target_cache"),
            "--set",
            "training.device=cuda:1",
        ],
    )

    eval_small.main()

    assert seen == {
        "checkpoint_path": "run/checkpoint_top1.pt",
        "config_path": "run/config.yaml",
        "metrics_output_path": str(tmp_path / "metrics_top1.csv"),
        "overrides": ["training.device=cuda:1"],
        "metrics_workers": 4,
        "metrics_chunk_size": 64,
        "target_cache_dir": tmp_path / "target_cache",
    }
    assert "写出指标" in capsys.readouterr().out
