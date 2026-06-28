import scripts.run_f_d_feature_extractor_probe as fdfe


def test_f_d_feature_extractor_probe_reuses_baselines_and_trains_two_candidates():
    specs = fdfe.build_run_specs()
    labels = {spec["label"] for spec in specs}
    train_labels = {spec["label"] for spec in specs if spec["train"]}

    assert labels == {
        "F0_native_stft_pre_mixer",
        "F-D0_high_stft_anchor",
        "F-D1_high_cwt",
        "F-D1b_high_cwt_cnn_tcn",
        "F-D2_high_cwt_modulation",
        "F-D2b_high_cwt_modulation_res_tcn",
    }
    assert train_labels == {
        "F-D1b_high_cwt_cnn_tcn",
        "F-D2b_high_cwt_modulation_res_tcn",
    }
    assert len(specs) == 18


def test_f_d_feature_extractor_probe_manifest_rows_include_encoder_overrides():
    cwt = next(spec for spec in fdfe.build_run_specs(cwt_cache="runs/custom/cwt.npz") if spec["label"] == "F-D1b_high_cwt_cnn_tcn")
    mod = next(
        spec
        for spec in fdfe.build_run_specs(modulation_cache="runs/custom/mod.npz")
        if spec["label"] == "F-D2b_high_cwt_modulation_res_tcn"
    )

    cwt_row = fdfe.manifest_row(cwt)
    mod_row = fdfe.manifest_row(mod)

    assert cwt_row["train"] == "true"
    assert cwt_row["feature_extractor"] == "cached_tf_tcn"
    assert "model.stft_encoder_type=cached_tf_tcn" in cwt_row["overrides"]
    assert "model.stft_cached_in_freq=36" in cwt_row["overrides"]
    assert "model.stft_cached_hidden_channels=32" in cwt_row["overrides"]
    assert "model.stft_cached_pooled_freq=6" in cwt_row["overrides"]
    assert "data.sst_cache_path=runs/custom/cwt.npz" in cwt_row["overrides"]

    assert mod_row["train"] == "true"
    assert mod_row["feature_extractor"] == "cached_sequence_res_tcn"
    assert "model.stft_encoder_type=cached_sequence_res_tcn" in mod_row["overrides"]
    assert "model.stft_cached_in_freq=8" in mod_row["overrides"]
    assert "model.stft_cached_hidden_channels=32" in mod_row["overrides"]
    assert "data.sst_cache_path=runs/custom/mod.npz" in mod_row["overrides"]
    assert mod_row["paired_f0_label"] == "F0_native_stft_pre_mixer"
