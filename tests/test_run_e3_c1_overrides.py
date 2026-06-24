import scripts.run_e3_c1_injection_probe as e3c1


def test_e3_c1_specs_cover_position_arms_and_two_substrates():
    specs = e3c1.build_run_specs()

    assert len(specs) == 18
    assert {s["label"] for s in specs} == {
        "E3-C1A_concat_post_fusion",
        "E3-C1B_token_pre_mixer",
        "E3-C1C_token_mid_mixer",
        "E3-C1D_token_post_mixer",
        "E3-C1S_concat_time_only",
        "E3-C1T_native_time_only",
    }
    assert len([s for s in specs if s["branch_mode"] == "dual"]) == 12
    assert len([s for s in specs if s["branch_mode"] == "time_only"]) == 6


def test_e3_c1_defaults_to_three_probe_seeds():
    assert {s["seed"] for s in e3c1.build_run_specs()} == {20260700, 20260837, 20260901}


def test_e3_c1_commands_default_to_preloaded_windows():
    spec = e3c1.build_run_specs()[0]
    joined = " ".join(e3c1._command_for_spec(spec, "cuda:0"))

    assert "data.preload_windows=true" in joined
    assert "training.num_workers=0" in joined
    assert "training.persistent_workers=true" not in joined
    assert "training.prefetch_factor=2" not in joined


def test_e3_c1_changes_only_fusion_position_for_dual_arms():
    specs = [s for s in e3c1.build_run_specs() if s["branch_mode"] == "dual"]

    expected = {
        "E3-C1A_concat_post_fusion": ("concat_generic", "post_fusion"),
        "E3-C1B_token_pre_mixer": ("native_inject", "pre_mixer"),
        "E3-C1C_token_mid_mixer": ("native_inject", "mid_mixer"),
        "E3-C1D_token_post_mixer": ("native_inject", "post_mixer"),
    }

    for label, (fusion_mode, position) in expected.items():
        matching = [s for s in specs if s["label"] == label]
        assert len(matching) == 3
        for spec in matching:
            joined = " ".join(spec["overrides"])
            assert f"model.fusion_mode={fusion_mode}" in joined
            assert f"model.stft_inject_position={position}" in joined
            assert "model.stft_encoder_type=conv2d" in joined
            assert "model.stft_high_hz=8.0" in joined
            assert "model.branch_mode=dual" in joined


def test_e3_c1_time_only_substrates_are_pairable():
    rows = [e3c1.manifest_row(s) for s in e3c1.build_run_specs()]
    dual_rows = [r for r in rows if r["branch_mode"] == "dual"]

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
    } <= set(rows[0])
    assert {r["paired_time_only_label"] for r in dual_rows} == {
        "E3-C1S_concat_time_only",
        "E3-C1T_native_time_only",
    }


def test_e3_c1_assigns_devices_round_robin():
    specs = e3c1.build_run_specs()[:5]
    assignments = e3c1._assign_devices(specs, ["cuda:0", "cuda:1"])

    assert [device for _, device in assignments] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"]


def test_e3_c1_launch_plan_staggers_parallel_slots():
    specs = e3c1.build_run_specs()[:5]
    launches = e3c1._build_launch_plan(
        specs,
        devices=["cuda:0", "cuda:1"],
        max_parallel=2,
        start_stagger_sec=30.0,
    )

    assert [(device, delay) for _, device, delay in launches] == [
        ("cuda:0", 0.0),
        ("cuda:1", 30.0),
        ("cuda:0", 0.0),
        ("cuda:1", 30.0),
        ("cuda:0", 0.0),
    ]
