import scripts.run_e5_a0_gated_fusion_probe as e5a0


def test_e5_a0_specs_cover_ungated_and_gated_native_arms():
    specs = e5a0.build_run_specs()

    assert len(specs) == 9
    assert {s["label"] for s in specs} == {
        "E5-A0T_native_time_only",
        "E5-A0.0_native_pre_mixer_ungated",
        "E5-A0.1_gated_native_pre_mixer",
    }
    assert len([s for s in specs if s["branch_mode"] == "time_only"]) == 3
    assert len([s for s in specs if s["branch_mode"] == "dual"]) == 6


def test_e5_a0_defaults_to_c1_probe_seed_set():
    assert {s["seed"] for s in e5a0.build_run_specs()} == {20260700, 20260837, 20260901}


def test_e5_a0_changes_only_gating_for_dual_arms():
    dual = [s for s in e5a0.build_run_specs() if s["branch_mode"] == "dual"]
    expected_modes = {
        "E5-A0.0_native_pre_mixer_ungated": "native_inject",
        "E5-A0.1_gated_native_pre_mixer": "gated_native_inject",
    }

    for label, fusion_mode in expected_modes.items():
        matching = [s for s in dual if s["label"] == label]
        assert len(matching) == 3
        for spec in matching:
            joined = " ".join(spec["overrides"])
            assert f"model.fusion_mode={fusion_mode}" in joined
            assert "model.stft_inject_position=pre_mixer" in joined
            assert "model.stft_encoder_type=conv2d" in joined
            assert "model.stft_high_hz=8.0" in joined
            assert "model.branch_mode=dual" in joined
            assert spec["paired_time_only_label"] == "E5-A0T_native_time_only"


def test_e5_a0_commands_use_full_data_and_direction_gate():
    spec = e5a0.build_run_specs()[0]
    joined = " ".join(e5a0._command_for_spec(spec, "cuda:0"))

    assert "data.max_train_windows=null" in joined
    assert "data.max_val_windows=null" in joined
    assert "training.checkpoint_gate.metric=auto_direction" in joined
    assert "training.checkpoint_gate.max=0.5" in joined
    assert "training.device=cuda:0" in joined
    assert "data.preload_windows=true" in joined


def test_e5_a0_manifest_row_records_gating_fields():
    row = e5a0.manifest_row(e5a0.build_run_specs()[-1])

    assert {
        "tag",
        "label",
        "branch_mode",
        "seed",
        "fusion_mode",
        "stft_inject_position",
        "stft_encoder_type",
        "paired_time_only_label",
        "overrides",
    } <= set(row)
