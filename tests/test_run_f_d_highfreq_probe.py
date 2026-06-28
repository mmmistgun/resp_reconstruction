import scripts.run_f_d_highfreq_probe as fd


def test_f_d_highfreq_probe_builds_f0_and_three_candidates():
    labels = {spec["label"] for spec in fd.build_run_specs()}

    assert labels == {
        "F0_native_stft_pre_mixer",
        "F-D0_high_stft_anchor",
        "F-D1_high_cwt",
        "F-D2_high_cwt_modulation",
    }


def test_f_d0_uses_short_high_stft_without_cache():
    spec = next(spec for spec in fd.build_run_specs() if spec["label"] == "F-D0_high_stft_anchor")
    joined = " ".join(spec["overrides"])

    assert spec["train"] is True
    assert "model.stft_encoder_type=conv2d" in joined
    assert "model.stft_win=800" in joined
    assert "model.stft_hop=100" in joined
    assert "model.stft_low_hz=1.0" in joined
    assert "model.stft_high_hz=8.0" in joined
    assert "data.sst_cache_path=" not in joined


def test_f_d1_uses_cached_tf_context_path():
    spec = next(
        spec
        for spec in fd.build_run_specs(cwt_cache="runs/custom/high_cwt.npz")
        if spec["label"] == "F-D1_high_cwt"
    )
    joined = " ".join(spec["overrides"])

    assert "model.stft_encoder_type=cached_tf" in joined
    assert "model.stft_cached_in_freq=36" in joined
    assert "data.sst_cache_path=runs/custom/high_cwt.npz" in joined
    assert spec["cache_path"] == "runs/custom/high_cwt.npz"


def test_f_d2_uses_cached_sequence_context_path():
    spec = next(
        spec
        for spec in fd.build_run_specs(modulation_cache="runs/custom/high_mod.npz")
        if spec["label"] == "F-D2_high_cwt_modulation"
    )
    joined = " ".join(spec["overrides"])

    assert "model.stft_encoder_type=cached_sequence" in joined
    assert "model.stft_cached_in_freq=8" in joined
    assert "data.sst_cache_path=runs/custom/high_mod.npz" in joined
    assert spec["cache_path"] == "runs/custom/high_mod.npz"


def test_f_d_manifest_rows_include_representation_and_cache():
    spec = next(spec for spec in fd.build_run_specs() if spec["label"] == "F-D1_high_cwt")
    row = fd.manifest_row(spec)

    assert row["representation"] == "high_cwt"
    assert row["cache_path"] == fd.DEFAULT_CWT_CACHE
    assert row["train"] == "true"
