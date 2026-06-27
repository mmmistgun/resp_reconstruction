import scripts.run_f_b_feature_extractor_probe as fbfe


def test_f_b_feature_extractor_probe_builds_enc1_reuse_and_enc2_candidate():
    specs = fbfe.build_run_specs()
    labels = {spec["label"] for spec in specs}

    assert labels == {
        "F0_native_stft_pre_mixer",
        "F-B1_aux_consistency_detach",
        "F-B1b_aux_enc2_band_aware_consistency",
    }
    assert len(specs) == 9


def test_f_b_feature_extractor_probe_manifest_rows_include_enc2_overrides():
    enc1 = next(spec for spec in fbfe.build_run_specs() if spec["label"] == "F-B1_aux_consistency_detach")
    enc2 = next(
        spec for spec in fbfe.build_run_specs() if spec["label"] == "F-B1b_aux_enc2_band_aware_consistency"
    )

    row1 = fbfe.manifest_row(enc1)
    row2 = fbfe.manifest_row(enc2)

    assert row1["train"] == "false"
    assert row2["train"] == "true"
    assert "model.fb_aux_head=enc2_band_aware_aux" in row2["overrides"]
    assert "loss.fb_aux_weight=0.01" in row2["overrides"]
    assert "loss.fb_consistency_weight=0.005" in row2["overrides"]
    assert "loss.fb_consistency_start_epoch=7" in row2["overrides"]
    assert row2["paired_f0_label"] == "F0_native_stft_pre_mixer"
