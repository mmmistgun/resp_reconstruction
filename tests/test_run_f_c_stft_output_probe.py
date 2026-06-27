import scripts.run_f_c_stft_output_probe as fc


def test_f_c_stft_output_probe_builds_reuse_and_low_complex_output_arms():
    specs = fc.build_run_specs()
    labels = {spec["label"] for spec in specs}

    assert labels == {"F0_native_stft_pre_mixer", "F-C0_low_complex_stft_output"}
    assert len(specs) == 6


def test_f_c_stft_output_probe_manifest_rows_include_output_stft_overrides():
    f0 = next(spec for spec in fc.build_run_specs() if spec["label"] == "F0_native_stft_pre_mixer")
    fc0 = next(spec for spec in fc.build_run_specs() if spec["label"] == "F-C0_low_complex_stft_output")

    row_f0 = fc.manifest_row(f0)
    row_fc0 = fc.manifest_row(fc0)

    assert row_f0["train"] == "false"
    assert row_fc0["train"] == "true"
    assert row_fc0["output_stft_low_hz"] == 0.0
    assert row_fc0["output_stft_high_hz"] == 3.0
    assert "model.name=time_stft_low_complex_output1d" in row_fc0["overrides"]
    assert "model.output_stft_low_hz=0.0" in row_fc0["overrides"]
    assert "model.output_stft_high_hz=3.0" in row_fc0["overrides"]
    assert row_fc0["paired_f0_label"] == "F0_native_stft_pre_mixer"
