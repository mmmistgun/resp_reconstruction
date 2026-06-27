import scripts.run_f_b_enc3_residual_probe as fb3


def test_f_b_enc3_residual_probe_builds_reuse_enc3_and_cap_arms():
    specs = fb3.build_run_specs()
    labels = {spec["label"] for spec in specs}

    assert labels == {
        "F0_native_stft_pre_mixer",
        "F-B2_low_complex_residual",
        "F-B3_enc3_tfgrid_residual",
        "F-B3b_enc3_tfgrid_residual_cap",
    }
    assert len(specs) == 12


def test_f_b_enc3_residual_probe_manifest_rows_include_enc3_and_cap_overrides():
    fb2 = next(spec for spec in fb3.build_run_specs() if spec["label"] == "F-B2_low_complex_residual")
    enc3 = next(spec for spec in fb3.build_run_specs() if spec["label"] == "F-B3_enc3_tfgrid_residual")
    cap = next(spec for spec in fb3.build_run_specs() if spec["label"] == "F-B3b_enc3_tfgrid_residual_cap")

    row_fb2 = fb3.manifest_row(fb2)
    row_enc3 = fb3.manifest_row(enc3)
    row_cap = fb3.manifest_row(cap)

    assert row_fb2["train"] == "false"
    assert row_enc3["train"] == "true"
    assert row_cap["train"] == "true"
    assert "model.fb_residual_head=enc3_tfgrid_residual" in row_enc3["overrides"]
    assert "model.fb_residual_head=enc3_tfgrid_residual" in row_cap["overrides"]
    assert "model.fb_residual_energy_cap=0.0" in row_enc3["overrides"]
    assert "model.fb_residual_energy_cap=0.05" in row_cap["overrides"]
    assert "model.fb_aux_head=enc1_min_aux" in row_cap["overrides"]
    assert row_cap["paired_f0_label"] == "F0_native_stft_pre_mixer"
