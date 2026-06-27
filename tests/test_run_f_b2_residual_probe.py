import scripts.run_f_b2_residual_probe as fb2


def test_f_b2_residual_probe_builds_reuse_and_residual_arms():
    specs = fb2.build_run_specs()
    labels = {spec["label"] for spec in specs}

    assert labels == {
        "F0_native_stft_pre_mixer",
        "F-B1_aux_consistency_detach",
        "F-B2_low_complex_residual",
    }
    assert len(specs) == 9


def test_f_b2_residual_probe_manifest_rows_include_residual_overrides():
    fb1 = next(spec for spec in fb2.build_run_specs() if spec["label"] == "F-B1_aux_consistency_detach")
    fb2_spec = next(spec for spec in fb2.build_run_specs() if spec["label"] == "F-B2_low_complex_residual")

    row1 = fb2.manifest_row(fb1)
    row2 = fb2.manifest_row(fb2_spec)

    assert row1["train"] == "false"
    assert row2["train"] == "true"
    assert "model.fb_residual_head=low_complex_residual" in row2["overrides"]
    assert "model.fb_residual_scale=0.03" in row2["overrides"]
    assert "model.fb_aux_head=enc1_min_aux" in row2["overrides"]
    assert row2["paired_f0_label"] == "F0_native_stft_pre_mixer"
