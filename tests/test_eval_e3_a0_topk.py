from pathlib import Path

import pandas as pd

import scripts.eval_e3_a0_topk as topk


def _make_run(root: Path, arm: str = "e3_a0_0_concat_fullband", branch: str = "dual", run_id: str = "run_a") -> Path:
    run_dir = root / arm / branch / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("training:\n  seed: 20260700\nmodel:\n  name: time_stft_dual1d\n", encoding="utf-8")
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


def test_discover_eval_specs_finds_three_topk_checkpoints(tmp_path):
    run_dir = _make_run(tmp_path)

    specs = topk.discover_eval_specs(tmp_path, top_k=3, force=False)

    assert [spec.rank for spec in specs] == [1, 2, 3]
    assert {spec.run_dir for spec in specs} == {run_dir}
    assert specs[0].checkpoint_path == run_dir / "checkpoint_top1.pt"
    assert specs[0].metrics_output == run_dir / "metrics_top1.csv"
    assert specs[0].tag == "e3_a0_0_concat_fullband_dual_run_a_top1"


def test_discover_eval_specs_skips_existing_outputs_unless_force(tmp_path):
    run_dir = _make_run(tmp_path)
    (run_dir / "metrics_top2.csv").write_text("x\n1\n", encoding="utf-8")

    specs = topk.discover_eval_specs(tmp_path, top_k=3, force=False)
    forced = topk.discover_eval_specs(tmp_path, top_k=3, force=True)

    assert [spec.rank for spec in specs] == [1, 3]
    assert [spec.rank for spec in forced] == [1, 2, 3]


def test_command_for_spec_uses_eval_script_config_output_and_device(tmp_path):
    spec = topk.discover_eval_specs(tmp_path, top_k=3, force=False)
    assert spec == []

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


def test_assign_devices_round_robin(tmp_path):
    _make_run(tmp_path)
    specs = topk.discover_eval_specs(tmp_path, top_k=3, force=False)

    assignments = topk.assign_devices(specs, ["cuda:0", "cuda:1"])

    assert [device for _, device in assignments] == ["cuda:0", "cuda:1", "cuda:0"]
