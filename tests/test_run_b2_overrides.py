import scripts.run_b2_native_decode as b2


def test_specs_cover_two_arms_per_seed():
    specs = b2.build_run_specs()
    arms = {s["branch_mode"] for s in specs}
    assert arms == {"time_only", "dual"}
    assert len(specs) == 2 * len(b2.SEEDS)


def test_all_specs_are_native_inject_patch_8hz_n0():
    for s in b2.build_run_specs():
        joined = " ".join(s["overrides"])
        assert "model.name=time_stft_dual1d" in joined
        assert "model.fusion_mode=native_inject" in joined
        assert "model.time_backbone=patch_mixer1d" in joined
        assert "model.stft_high_hz=8.0" in joined
        assert "model.stft_norm=n0" in joined
        assert "model.stft_encoder_type=conv2d" in joined


def test_time_only_and_dual_use_separate_run_roots():
    roots = {s["branch_mode"]: None for s in b2.build_run_specs()}
    for s in b2.build_run_specs():
        root = next(o.split("=", 1)[1] for o in s["overrides"] if o.startswith("outputs.run_root="))
        roots[s["branch_mode"]] = root
    assert roots["time_only"] != roots["dual"]


def test_manifest_row_preserves_factors():
    row = b2.manifest_row(b2.build_run_specs()[0])
    assert {"tag", "branch_mode", "seed", "overrides"} <= set(row)
