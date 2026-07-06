from pathlib import Path

import scripts.run_g_series_test_eval as runner


def test_load_specs_reads_external_csv_manifest(tmp_path: Path):
    manifest = tmp_path / "specs.csv"
    manifest.write_text(
        "label,seed,checkpoint\n"
        "g0_time_only,20260700,runs/example_a/checkpoint_top3.pt\n"
        "g3_c_wide_8p0,20260837,runs/example_b/checkpoint_top1.pt\n",
        encoding="utf-8",
    )

    specs = runner.load_specs(manifest)

    assert specs == [
        runner.TestEvalSpec(
            label="g0_time_only",
            seed=20260700,
            checkpoint=Path("runs/example_a/checkpoint_top3.pt"),
        ),
        runner.TestEvalSpec(
            label="g3_c_wide_8p0",
            seed=20260837,
            checkpoint=Path("runs/example_b/checkpoint_top1.pt"),
        ),
    ]


def test_assign_devices_round_robin_across_two_gpus():
    specs = [
        runner.TestEvalSpec(label=f"arm{i}", seed=20260700 + i, checkpoint=Path(f"run{i}/checkpoint.pt"))
        for i in range(5)
    ]

    assignments = runner.assign_devices(specs, ["cuda:0", "cuda:1"])

    assert [device for _, device in assignments] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"]


def test_command_for_spec_uses_test_eval_outputs_and_device(tmp_path: Path):
    spec = runner.TestEvalSpec(
        label="g0_time_only",
        seed=20260700,
        checkpoint=Path("runs/example/checkpoint_top3.pt"),
    )

    command = runner.command_for_spec(
        spec,
        "cuda:1",
        output_dir=tmp_path,
        python="python",
        metric_workers=4,
    )

    assert command == [
        "python",
        "scripts/eval_tho_test.py",
        "--checkpoint",
        "runs/example/checkpoint_top3.pt",
        "--metrics-output",
        str(tmp_path / "g0_time_only_20260700_test_metrics.csv"),
        "--summary-output",
        str(tmp_path / "g0_time_only_20260700_test_summary.csv"),
        "--manifest-output",
        str(tmp_path / "g0_time_only_20260700_test_manifest.csv"),
        "--set",
        "training.device=cuda:1",
        "--set",
        "evaluation.metric_workers=4",
    ]


def test_pending_specs_skip_existing_outputs_unless_force(tmp_path: Path):
    done = runner.TestEvalSpec("done", 20260700, Path("done/checkpoint.pt"))
    pending = runner.TestEvalSpec("pending", 20260700, Path("pending/checkpoint.pt"))
    (tmp_path / "done_20260700_test_metrics.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "done_20260700_test_summary.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "done_20260700_test_manifest.csv").write_text("x\n1\n", encoding="utf-8")

    filtered = runner.pending_specs([done, pending], output_dir=tmp_path, force=False)
    forced = runner.pending_specs([done, pending], output_dir=tmp_path, force=True)

    assert filtered == [pending]
    assert forced == [done, pending]


def test_build_launch_plan_uses_parallel_slots_for_stagger():
    specs = [
        runner.TestEvalSpec(label=f"arm{i}", seed=20260700 + i, checkpoint=Path(f"run{i}/checkpoint.pt"))
        for i in range(4)
    ]
    assignments = runner.assign_devices(specs, ["cuda:0", "cuda:1"])

    plan = runner.build_launch_plan(assignments, max_parallel=3, start_stagger_sec=10.0)

    assert [(spec.label, device, delay) for spec, device, delay in plan] == [
        ("arm0", "cuda:0", 0.0),
        ("arm1", "cuda:1", 10.0),
        ("arm2", "cuda:0", 20.0),
        ("arm3", "cuda:1", 0.0),
    ]
