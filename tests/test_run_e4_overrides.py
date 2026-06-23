import scripts.run_e4_sst_probe as e4


def test_dual_arm_is_sst_cached_4_seeds():
    specs = [s for s in e4.build_run_specs() if s["kind"] == "dual"]
    assert len(specs) == 4
    for s in specs:
        joined = " ".join(s["overrides"])
        assert "model.stft_encoder_type=sst_cached" in joined
        assert "model.fusion_mode=native_inject" in joined
        assert "model.branch_mode=dual" in joined
        assert "data.sst_cache_path=" in joined


def test_has_one_fullband_sanity_run():
    sanity = [s for s in e4.build_run_specs() if s["kind"] == "sanity"]
    assert len(sanity) == 1
    joined = " ".join(sanity[0]["overrides"])
    assert "model.stft_encoder_type=conv2d" in joined
    assert "data.sst_cache_path=" not in joined  # sanity 不读 SST 缓存


def test_seeds_subset_of_e1d_for_pairing():
    seeds = {s["seed"] for s in e4.build_run_specs() if s["kind"] == "dual"}
    assert seeds <= {20260700, 20260710, 20260837, 20260901, 20260911, 20260920}


def test_sst_cache_override_propagates():
    specs = e4.build_run_specs(sst_cache="runs/custom/sst.npz")
    dual = next(s for s in specs if s["kind"] == "dual")
    assert "data.sst_cache_path=runs/custom/sst.npz" in " ".join(dual["overrides"])


def test_manifest_row_has_factors():
    row = e4.manifest_row(e4.build_run_specs()[0])
    assert {"tag", "kind", "seed", "overrides"} <= set(row)
