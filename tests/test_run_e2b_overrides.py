import scripts.run_e2b_band_group as e2b


def test_dual_arm_is_bandgroup_4_seeds():
    specs = [s for s in e2b.build_run_specs() if s["kind"] == "dual"]
    assert len(specs) == 4
    for s in specs:
        joined = " ".join(s["overrides"])
        assert "model.stft_encoder_type=bandgroup" in joined
        assert "model.fusion_mode=native_inject" in joined
        assert "model.branch_mode=dual" in joined


def test_has_one_fullband_sanity_run():
    sanity = [s for s in e2b.build_run_specs() if s["kind"] == "sanity"]
    assert len(sanity) == 1
    assert "model.stft_encoder_type=conv2d" in " ".join(sanity[0]["overrides"])


def test_seeds_subset_of_e1d_for_pairing():
    seeds = {s["seed"] for s in e2b.build_run_specs() if s["kind"] == "dual"}
    assert seeds <= {20260700, 20260710, 20260837, 20260901, 20260911, 20260920}


def test_manifest_row_has_factors():
    row = e2b.manifest_row(e2b.build_run_specs()[0])
    assert {"tag", "kind", "seed", "overrides"} <= set(row)
