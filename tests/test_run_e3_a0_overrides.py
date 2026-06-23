import scripts.run_e3_a0_probe as e3a0


def test_e3_a0_has_expected_dual_arms_and_deduplicated_time_only():
    specs = e3a0.build_run_specs()
    labels = {s["label"] for s in specs}

    assert {
        "E3-A0.0_concat_fullband",
        "E3-A0.1_token_context_fullband",
        "E3-A0.2_concat_bandgroup",
        "E3-A0.3_token_context_bandgroup",
    } <= labels
    assert len(specs) == 18

    dual_specs = [s for s in specs if s["branch_mode"] == "dual"]
    time_specs = [s for s in specs if s["branch_mode"] == "time_only"]
    assert len(dual_specs) == 12
    assert len(time_specs) == 6
    assert {s["label"] for s in time_specs} == {
        "E3-A0.0_concat_fullband",
        "E3-A0.1_token_context_fullband",
    }
    assert {s["label"] for s in dual_specs} == {
        "E3-A0.0_concat_fullband",
        "E3-A0.1_token_context_fullband",
        "E3-A0.2_concat_bandgroup",
        "E3-A0.3_token_context_bandgroup",
    }


def test_e3_a0_defaults_to_small_probe_seed_set():
    seeds = {s["seed"] for s in e3a0.build_run_specs()}
    assert seeds == {20260700, 20260837, 20260901}


def test_e3_a0_token_context_arms_use_context_fusion():
    specs = [s for s in e3a0.build_run_specs() if "token_context" in s["label"]]
    assert specs
    for spec in specs:
        joined = " ".join(spec["overrides"])
        assert "model.fusion_mode=token_context_inject" in joined
        assert "model.time_backbone=patch_mixer1d" in joined


def test_e3_a0_frontend_arms_use_bandgroup_only_on_dual():
    specs = [s for s in e3a0.build_run_specs() if "bandgroup" in s["label"]]
    assert specs
    assert {s["branch_mode"] for s in specs} == {"dual"}
    for spec in specs:
        joined = " ".join(spec["overrides"])
        assert "model.stft_encoder_type=bandgroup" in joined


def test_e3_a0_dual_specs_record_paired_time_only_label():
    specs = [s for s in e3a0.build_run_specs() if s["branch_mode"] == "dual"]
    paired = {(s["label"], s["paired_time_only_label"]) for s in specs}

    assert ("E3-A0.0_concat_fullband", "E3-A0.0_concat_fullband") in paired
    assert ("E3-A0.1_token_context_fullband", "E3-A0.1_token_context_fullband") in paired
    assert ("E3-A0.2_concat_bandgroup", "E3-A0.0_concat_fullband") in paired
    assert ("E3-A0.3_token_context_bandgroup", "E3-A0.1_token_context_fullband") in paired


def test_e3_a0_manifest_row_has_factors():
    row = e3a0.manifest_row(e3a0.build_run_specs()[0])
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


def test_e3_a0_resolves_devices_without_default_pollution():
    assert e3a0._resolve_devices(None) == ["cuda:0"]
    assert e3a0._resolve_devices(["cuda:1"]) == ["cuda:1"]
    assert e3a0._resolve_devices(["cuda:0", "cuda:1"]) == ["cuda:0", "cuda:1"]


def test_e3_a0_assigns_devices_round_robin():
    specs = e3a0.build_run_specs()[:5]
    assignments = e3a0._assign_devices(specs, ["cuda:0", "cuda:1"])

    assert [device for _, device in assignments] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"]
