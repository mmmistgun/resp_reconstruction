import scripts.run_f_a2_confidence_guard_probe as fa2c


def test_f_a2_confidence_guard_specs_reuse_prior_anchors_and_train_new_candidates():
    specs = fa2c.build_run_specs()

    assert len(specs) == 15
    reused = [s for s in specs if s["reuse_existing"]]
    runnable = [s for s in specs if not s["reuse_existing"]]
    assert {s["label"] for s in reused} == {
        "F0_native_stft_pre_mixer",
        "F-A2_dist_bandE",
        "F-A2b_dist_bandE_w005",
    }
    assert {s["label"] for s in runnable} == {
        "F-A2d_confScoreInv_w005",
        "F-A2e_confLevelMedLow_w005",
    }
    assert len(runnable) == 6


def test_f_a2_confidence_guard_candidates_keep_w005_and_only_change_sample_weighting():
    specs = {s["label"]: s for s in fa2c.build_run_specs() if s["seed"] == 20260700}

    assert specs["F-A2b_dist_bandE_w005"]["stft_sample_weight_mode"] == "none"
    assert specs["F-A2d_confScoreInv_w005"]["stft_dist_weight"] == 0.02
    assert specs["F-A2d_confScoreInv_w005"]["stft_band_energy_weight"] == 0.005
    assert specs["F-A2d_confScoreInv_w005"]["stft_sample_weight_mode"] == "waveform_confidence_score_inverse"
    assert specs["F-A2d_confScoreInv_w005"]["stft_sample_weight_min"] == 0.05
    assert specs["F-A2e_confLevelMedLow_w005"]["stft_dist_weight"] == 0.02
    assert specs["F-A2e_confLevelMedLow_w005"]["stft_band_energy_weight"] == 0.005
    assert specs["F-A2e_confLevelMedLow_w005"]["stft_sample_weight_mode"] == "waveform_confidence_level_medlow"


def test_f_a2_confidence_guard_manifest_rows_record_weighting_mode_and_reuse():
    rows = [fa2c.manifest_row(s) for s in fa2c.build_run_specs()]
    candidate_rows = [r for r in rows if r["label"].startswith("F-A")]

    assert {
        "tag",
        "label",
        "seed",
        "paired_f0_label",
        "stft_dist_weight",
        "stft_band_energy_weight",
        "stft_sample_weight_mode",
        "stft_sample_weight_min",
        "reuse_existing",
        "overrides",
    } <= set(rows[0])
    assert {r["paired_f0_label"] for r in candidate_rows} == {"F0_native_stft_pre_mixer"}
    assert {r["reuse_existing"] for r in rows if r["label"] == "F-A2b_dist_bandE_w005"} == {"true"}
    assert {r["reuse_existing"] for r in rows if r["label"].startswith("F-A2d")} == {"false"}


def test_f_a2_confidence_guard_command_uses_new_run_root_and_weighting_override():
    spec = [s for s in fa2c.build_run_specs() if s["label"] == "F-A2d_confScoreInv_w005"][0]
    joined = " ".join(fa2c._command_for_spec(spec, "cuda:0"))

    assert "outputs.run_root=runs/f_a2_confidence_guard/f_a2d_confscoreinv_w005/dual" in joined
    assert "loss.stft_dist_weight=0.02" in joined
    assert "loss.stft_band_energy_weight=0.005" in joined
    assert "loss.stft_sample_weight_mode=waveform_confidence_score_inverse" in joined
    assert "loss.stft_sample_weight_min=0.05" in joined
    assert "loss.log_component_grad_norms=true" in joined
    assert "training.device=cuda:0" in joined
