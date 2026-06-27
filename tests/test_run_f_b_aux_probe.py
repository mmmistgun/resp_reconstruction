import scripts.run_f_b_aux_probe as fb


def test_f_b_aux_probe_builds_f0_and_aux_consistency_arms():
    specs = fb.build_run_specs()
    labels = {spec["label"] for spec in specs}

    assert labels == {
        "F0_native_stft_pre_mixer",
        "F-B0_aux_enc1",
        "F-B1_aux_consistency_detach",
    }
    assert len(specs) == 9


def test_f_b_aux_probe_manifest_rows_include_aux_overrides():
    fb0 = next(spec for spec in fb.build_run_specs() if spec["label"] == "F-B0_aux_enc1")
    fb1 = next(spec for spec in fb.build_run_specs() if spec["label"] == "F-B1_aux_consistency_detach")

    row0 = fb.manifest_row(fb0)
    row1 = fb.manifest_row(fb1)

    assert "model.fb_aux_head=enc1_min_aux" in row0["overrides"]
    assert "loss.fb_aux_weight=0.01" in row0["overrides"]
    assert "loss.fb_consistency_weight=0.0" in row0["overrides"]
    assert "loss.fb_consistency_weight=0.005" in row1["overrides"]
    assert "loss.fb_consistency_start_epoch=7" in row1["overrides"]
    assert row1["paired_f0_label"] == "F0_native_stft_pre_mixer"
