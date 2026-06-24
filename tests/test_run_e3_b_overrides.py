import scripts.run_e3_b_probe as e3b


def test_e3_b_specs_cover_three_arms_with_time_only_and_dual():
    specs = e3b.build_run_specs()

    assert len(specs) == 18
    assert {s["label"] for s in specs} == {
        "E3-B0_concat_fullband_ref",
        "E3-B1_freq_mlp_fullband",
        "E3-B2_soft_band_concat",
    }
    assert {s["branch_mode"] for s in specs} == {"time_only", "dual"}
    for label in {s["label"] for s in specs}:
        assert len([s for s in specs if s["label"] == label and s["branch_mode"] == "time_only"]) == 3
        assert len([s for s in specs if s["label"] == label and s["branch_mode"] == "dual"]) == 3


def test_e3_b_defaults_to_small_probe_seed_set():
    assert {s["seed"] for s in e3b.build_run_specs()} == {20260700, 20260837, 20260901}


def test_e3_b_encoder_types_match_arm_design():
    specs = e3b.build_run_specs()
    expected = {
        "E3-B0_concat_fullband_ref": "conv2d",
        "E3-B1_freq_mlp_fullband": "freq_mlp",
        "E3-B2_soft_band_concat": "soft_band",
    }

    for spec in specs:
        joined = " ".join(spec["overrides"])
        assert f"model.stft_encoder_type={expected[spec['label']]}" in joined
        assert "model.fusion_mode=concat_generic" in joined
        assert "model.fuse_len=600" in joined
        assert "model.fusion_decoder=deep" in joined


def test_e3_b_manifest_row_has_pairing_fields():
    row = e3b.manifest_row(e3b.build_run_specs()[0])

    assert {
        "tag",
        "label",
        "branch_mode",
        "seed",
        "fusion_mode",
        "stft_encoder_type",
        "paired_time_only_label",
        "overrides",
    } <= set(row)


def test_e3_b_assigns_devices_round_robin():
    specs = e3b.build_run_specs()[:5]
    assignments = e3b._assign_devices(specs, ["cuda:0", "cuda:1"])

    assert [device for _, device in assignments] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"]
