import scripts.run_e5_a1_cross_attention_probe as e5a1


def test_e5_a1_specs_cover_ungated_and_cross_attention_arms():
    specs = e5a1.build_run_specs()

    assert len(specs) == 9
    assert {s["label"] for s in specs} == {
        "E5-A1T_native_time_only",
        "E5-A1.0_native_pre_mixer_ungated",
        "E5-A1.1_cross_attention_pre_mixer",
    }
    assert len([s for s in specs if s["branch_mode"] == "time_only"]) == 3
    assert len([s for s in specs if s["branch_mode"] == "dual"]) == 6


def test_e5_a1_cross_attention_changes_only_fusion_mechanism():
    cross = [s for s in e5a1.build_run_specs() if s["label"] == "E5-A1.1_cross_attention_pre_mixer"]

    assert len(cross) == 3
    for spec in cross:
        joined = " ".join(spec["overrides"])
        assert "model.fusion_mode=cross_attention_inject" in joined
        assert "model.stft_inject_position=pre_mixer" in joined
        assert "model.cross_attention_heads=2" in joined
        assert "model.stft_encoder_type=conv2d" in joined
        assert "model.stft_high_hz=8.0" in joined
        assert "model.branch_mode=dual" in joined
        assert spec["paired_time_only_label"] == "E5-A1T_native_time_only"


def test_e5_a1_commands_use_full_data_and_direction_gate():
    spec = e5a1.build_run_specs()[0]
    joined = " ".join(e5a1._command_for_spec(spec, "cuda:0"))

    assert "data.max_train_windows=null" in joined
    assert "data.max_val_windows=null" in joined
    assert "training.checkpoint_gate.metric=auto_direction" in joined
    assert "training.checkpoint_gate.max=0.5" in joined
    assert "training.device=cuda:0" in joined


def test_e5_a1_manifest_row_records_attention_fields():
    row = e5a1.manifest_row(e5a1.build_run_specs()[-1])

    assert {
        "tag",
        "label",
        "branch_mode",
        "seed",
        "fusion_mode",
        "stft_inject_position",
        "cross_attention_heads",
        "stft_encoder_type",
        "paired_time_only_label",
        "overrides",
    } <= set(row)
