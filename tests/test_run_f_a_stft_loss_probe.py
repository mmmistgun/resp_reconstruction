import scripts.run_f_a_stft_loss_probe as fa


def test_f_a_specs_cover_f0_and_three_loss_candidates():
    specs = fa.build_run_specs()

    assert len(specs) == 15
    assert {s["label"] for s in specs} == {
        "F0_native_time_only",
        "F0_native_stft_pre_mixer",
        "F-A0_dist",
        "F-A1_bandE",
        "F-A2_dist_bandE",
    }
    assert len([s for s in specs if s["branch_mode"] == "time_only"]) == 3
    assert len([s for s in specs if s["branch_mode"] == "dual"]) == 12


def test_f_a_specs_use_three_pilot_seeds():
    assert {s["seed"] for s in fa.build_run_specs()} == {20260700, 20260837, 20260901}


def test_f_a_candidates_only_change_target_stft_loss_weights():
    specs = {s["label"]: s for s in fa.build_run_specs() if s["seed"] == 20260700}

    assert "loss.stft_dist_weight=0.0" in specs["F0_native_stft_pre_mixer"]["overrides"]
    assert "loss.stft_band_energy_weight=0.0" in specs["F0_native_stft_pre_mixer"]["overrides"]
    assert "loss.stft_dist_weight=0.02" in specs["F-A0_dist"]["overrides"]
    assert "loss.stft_band_energy_weight=0.0" in specs["F-A0_dist"]["overrides"]
    assert "loss.stft_dist_weight=0.0" in specs["F-A1_bandE"]["overrides"]
    assert "loss.stft_band_energy_weight=0.01" in specs["F-A1_bandE"]["overrides"]
    assert "loss.stft_dist_weight=0.02" in specs["F-A2_dist_bandE"]["overrides"]
    assert "loss.stft_band_energy_weight=0.01" in specs["F-A2_dist_bandE"]["overrides"]


def test_f_a_manifest_rows_are_pairable():
    rows = [fa.manifest_row(s) for s in fa.build_run_specs()]
    candidate_rows = [r for r in rows if r["label"].startswith("F-A")]

    assert {
        "tag",
        "label",
        "branch_mode",
        "seed",
        "paired_f0_label",
        "paired_time_only_label",
        "stft_dist_weight",
        "stft_band_energy_weight",
        "overrides",
    } <= set(rows[0])
    assert {r["paired_f0_label"] for r in candidate_rows} == {"F0_native_stft_pre_mixer"}
    assert {r["paired_time_only_label"] for r in candidate_rows} == {"F0_native_time_only"}


def test_f_a_command_uses_full_windows_gate_and_grad_logging():
    spec = [s for s in fa.build_run_specs() if s["label"] == "F-A2_dist_bandE"][0]
    joined = " ".join(fa._command_for_spec(spec, "cuda:0"))

    assert "data.max_train_windows=null" in joined
    assert "data.max_val_windows=null" in joined
    assert "training.batch_size=128" in joined
    assert "training.checkpoint_gate.metric=auto_direction" in joined
    assert "loss.log_component_grad_norms=true" in joined
    assert "training.device=cuda:0" in joined


def test_f_a_launch_plan_staggers_parallel_slots():
    launches = fa._build_launch_plan(
        fa.build_run_specs()[:4],
        devices=["cuda:0", "cuda:1"],
        max_parallel=2,
        start_stagger_sec=30.0,
    )

    assert [(device, delay) for _, device, delay in launches] == [
        ("cuda:0", 0.0),
        ("cuda:1", 30.0),
        ("cuda:0", 0.0),
        ("cuda:1", 30.0),
    ]
