import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from resp_train.experiments.selection import TASK_SELECTION_COLUMNS
import scripts.eval_topk_checkpoints as topk


def _make_run(root: Path, arm: str = "f_d0_high_stft_anchor", branch: str = "dual", run_id: str = "run_a") -> Path:
    run_dir = root / arm / branch / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        "training:\n  seed: 20260700\nmodel:\n  name: time_stft_dual1d\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "rank": [1, 2, 3],
            "epoch": [10, 9, 8],
            "val_loss": [0.1, 0.2, 0.3],
            "checkpoint": ["checkpoint_top1.pt", "checkpoint_top2.pt", "checkpoint_top3.pt"],
        }
    ).to_csv(run_dir / "checkpoint_topk.csv", index=False)
    for rank in (1, 2, 3):
        (run_dir / f"checkpoint_top{rank}.pt").write_bytes(b"ckpt")
    return run_dir


def _write_metrics(
    path: Path,
    peak: list[float],
    spec: list[float] | None = None,
    *,
    robust: list[float] | None = None,
    count: list[float] | None = None,
) -> None:
    if spec is None:
        spec = [0.2 for _ in peak]
    if robust is None:
        robust = peak
    if count is None:
        count = [1.0 for _ in peak]
    pd.DataFrame(
        {
            "rr_peak_band_abs_error": peak,
            "rr_peak_band_robust_abs_error": robust,
            "rr_spec_abs_error": spec,
            "breath_count_zero_cross_abs_error": count,
            "relative_envelope_mae": [0.3 for _ in peak],
            "relative_envelope_corr": [0.5 for _ in peak],
            "spectrum_similarity": [0.9 for _ in peak],
            "band_limited_corr": [0.7 for _ in peak],
            "best_lag_corr": [0.8 for _ in peak],
            "best_lag_sec": [0.0 for _ in peak],
        }
    ).to_csv(path, index=False)


def test_discover_eval_specs_finds_generic_topk_checkpoints(tmp_path):
    run_dir = _make_run(tmp_path)

    specs = topk.discover_eval_specs(tmp_path, top_k=3, force=False)

    assert [spec.rank for spec in specs] == [1, 2, 3]
    assert {spec.run_dir for spec in specs} == {run_dir}
    assert specs[0].checkpoint_path == run_dir / "checkpoint_top1.pt"
    assert specs[0].metrics_output == run_dir / "metrics_top1.csv"
    assert specs[0].tag == "f_d0_high_stft_anchor_dual_run_a_top1"


def test_discover_eval_specs_skips_existing_outputs_unless_force(tmp_path):
    run_dir = _make_run(tmp_path)
    (run_dir / "metrics_top2.csv").write_text("x\n1\n", encoding="utf-8")

    specs = topk.discover_eval_specs(tmp_path, top_k=3, force=False)
    forced = topk.discover_eval_specs(tmp_path, top_k=3, force=True)

    assert [spec.rank for spec in specs] == [1, 3]
    assert [spec.rank for spec in forced] == [1, 2, 3]


def test_command_for_spec_uses_eval_script_config_output_and_device(tmp_path):
    run_dir = _make_run(tmp_path)
    spec = topk.discover_eval_specs(tmp_path, top_k=3, force=False)[0]

    cmd = topk.command_for_spec(spec, "cuda:1", python="python")

    assert cmd == [
        "python",
        "scripts/eval_tho_small.py",
        "--checkpoint",
        str(run_dir / "checkpoint_top1.pt"),
        "--config",
        str(run_dir / "config.yaml"),
        "--metrics-output",
        str(run_dir / "metrics_top1.csv"),
        "--set",
        "training.device=cuda:1",
    ]


def test_command_for_spec_can_pass_metric_workers(tmp_path):
    run_dir = _make_run(tmp_path)
    spec = topk.discover_eval_specs(tmp_path, top_k=3, force=False)[0]

    cmd = topk.command_for_spec(
        spec,
        "cuda:1",
        python="python",
        metric_workers=4,
        metrics_chunk_size=64,
        target_cache_dir=tmp_path / "topk_target_cache",
    )

    assert cmd == [
        "python",
        "scripts/eval_tho_small.py",
        "--checkpoint",
        str(run_dir / "checkpoint_top1.pt"),
        "--config",
        str(run_dir / "config.yaml"),
        "--metrics-output",
        str(run_dir / "metrics_top1.csv"),
        "--metrics-workers",
        "4",
        "--metrics-chunk-size",
        "64",
        "--target-workers",
        "4",
        "--target-chunk-size",
        "64",
        "--target-cache-dir",
        str(tmp_path / "topk_target_cache"),
        "--set",
        "training.device=cuda:1",
    ]


def test_build_launch_plan_staggers_by_parallel_slot(tmp_path):
    _make_run(tmp_path, arm="f_d0_high_stft_anchor", branch="dual", run_id="run_a")
    _make_run(tmp_path, arm="f_d0_high_stft_anchor", branch="dual", run_id="run_b")
    specs = topk.discover_eval_specs(tmp_path, top_k=2, force=False)
    assignments = topk.assign_devices(specs, ["cuda:0", "cuda:1"])

    plan = topk.build_launch_plan(assignments, max_parallel=3, start_stagger_sec=20.0)

    assert [(spec.rank, device, delay) for spec, device, delay in plan] == [
        (1, "cuda:0", 0.0),
        (2, "cuda:1", 20.0),
        (1, "cuda:0", 40.0),
        (2, "cuda:1", 0.0),
    ]


def test_run_one_sleeps_before_eval_when_launch_delay_is_set(tmp_path, monkeypatch):
    _make_run(tmp_path)
    spec = topk.discover_eval_specs(tmp_path, top_k=1, force=False)[0]
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(topk.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))
    monkeypatch.setattr(
        topk.subprocess,
        "run",
        lambda command, check: calls.append(("run", command)),
    )

    result = topk._run_one(spec, "cuda:0", metric_workers=1, launch_delay_sec=12.5)

    assert result == spec.tag
    assert calls[0] == ("sleep", 12.5)
    assert calls[1][0] == "run"


def test_summarize_topk_results_selects_best_rank_by_task_metrics(tmp_path):
    run_dir = _make_run(tmp_path)
    _write_metrics(
        run_dir / "metrics_top1.csv",
        peak=[0.1, 1.5, 2.5],
        spec=[0.1, 0.1, 0.1],
        robust=[0.5, 0.5, 0.5],
        count=[0.0, 0.0, 0.0],
    )
    _write_metrics(
        run_dir / "metrics_top2.csv",
        peak=[0.2, 0.3, 0.4],
        spec=[0.3, 0.3, 0.3],
        robust=[0.2, 0.2, 0.2],
        count=[2.0, 2.0, 2.0],
    )
    _write_metrics(
        run_dir / "metrics_top3.csv",
        peak=[0.2, 0.3, 1.4],
        spec=[0.2, 0.2, 0.2],
        robust=[0.2, 0.2, 0.2],
        count=[1.0, 1.0, 1.0],
    )

    all_frame, best_frame = topk.summarize_topk_results(tmp_path, top_k=3)

    assert len(all_frame) == 3
    assert best_frame[["run_dir", "rank"]].to_dict("records") == [
        {"run_dir": str(run_dir), "rank": 3}
    ]
    assert best_frame.iloc[0]["rr_peak_band_robust_abs_error_mean"] == pytest.approx(0.2)
    assert best_frame.iloc[0]["breath_count_zero_cross_abs_error_mean"] == pytest.approx(1.0)
    assert best_frame.iloc[0]["seed"] == 20260700


def test_topk_selection_columns_reuse_training_task_selection_contract():
    assert topk.SELECTION_COLUMNS == list(TASK_SELECTION_COLUMNS)


def test_summarize_topk_results_rejects_metrics_without_current_guards(tmp_path):
    run_dir = _make_run(tmp_path)
    for rank in (1, 2, 3):
        _write_metrics(run_dir / f"metrics_top{rank}.csv", peak=[0.1 * rank])
        frame = pd.read_csv(run_dir / f"metrics_top{rank}.csv").drop(columns="rr_peak_band_robust_abs_error")
        frame.to_csv(run_dir / f"metrics_top{rank}.csv", index=False)

    with pytest.raises(ValueError, match="缺少当前主护栏"):
        topk.summarize_topk_results(tmp_path, top_k=3)


def test_summarize_topk_results_requires_both_guards_on_same_checkpoint(tmp_path):
    run_dir = _make_run(tmp_path)
    for rank in (1, 2, 3):
        _write_metrics(run_dir / f"metrics_top{rank}.csv", peak=[0.1 * rank])
    first = pd.read_csv(run_dir / "metrics_top1.csv")
    first["breath_count_zero_cross_abs_error"] = float("nan")
    first.to_csv(run_dir / "metrics_top1.csv", index=False)
    for rank in (2, 3):
        frame = pd.read_csv(run_dir / f"metrics_top{rank}.csv")
        frame["rr_peak_band_robust_abs_error"] = float("nan")
        frame.to_csv(run_dir / f"metrics_top{rank}.csv", index=False)

    with pytest.raises(ValueError, match="同时包含两个主护栏"):
        topk.summarize_topk_results(tmp_path, top_k=3)


def test_eval_topk_script_help_works_outside_repo_root(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(topk.__file__).resolve()), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "checkpoint_topN" in result.stdout


def test_output_paths_default_to_runs_root_name(tmp_path):
    paths = topk.output_paths(tmp_path / "runs" / "f_d_highfreq", output_prefix=None)

    assert paths.manifest == tmp_path / "runs" / "f_d_highfreq_topk_eval_manifest.csv"
    assert paths.all_metrics == tmp_path / "runs" / "f_d_highfreq_topk_all_metrics.csv"
    assert paths.best_by_rr == tmp_path / "runs" / "f_d_highfreq_topk_best_by_rr.csv"
