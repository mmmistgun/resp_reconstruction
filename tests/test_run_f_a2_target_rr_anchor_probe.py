import scripts.run_f_a2_target_rr_anchor_probe as fa2g


def test_f_a2_target_rr_anchor_specs_reuse_prior_anchors_and_train_only_f_a2g():
    specs = fa2g.build_run_specs()

    assert len(specs) == 21
    reused = [s for s in specs if s["reuse_existing"]]
    runnable = [s for s in specs if not s["reuse_existing"]]
    assert {s["label"] for s in reused} == {
        "F0_native_stft_pre_mixer",
        "F-A2_dist_bandE",
        "F-A2b_dist_bandE_w005",
        "F-A2d_confScoreInv_w005",
        "F-A2e_confLevelMedLow_w005",
        "F-A2f_peak_anchor_w005",
    }
    assert {s["label"] for s in runnable} == {"F-A2g_targetRRAware_w005"}
    assert len(runnable) == 3


def test_f_a2_target_rr_anchor_candidate_keeps_w005_and_uses_guarded_peak_anchor():
    specs = {s["label"]: s for s in fa2g.build_run_specs() if s["seed"] == 20260700}

    assert specs["F-A2b_dist_bandE_w005"]["stft_peak_anchor_weight"] == 0.0
    assert specs["F-A2f_peak_anchor_w005"]["reuse_existing"] is True
    candidate = specs["F-A2g_targetRRAware_w005"]
    assert candidate["stft_dist_weight"] == 0.02
    assert candidate["stft_band_energy_weight"] == 0.005
    assert candidate["stft_peak_anchor_weight"] == 0.005
    assert candidate["stft_peak_anchor_sigma_bins"] == 1.0
    assert candidate["stft_peak_anchor_mode"] == "target_rr_guard"
    assert candidate["stft_peak_anchor_guard_tolerance_bpm"] == 2.0
    assert candidate["stft_peak_anchor_target_quantile"] == 0.5
    assert candidate["stft_sample_weight_mode"] == "none"


def test_f_a2_target_rr_anchor_manifest_rows_record_guarded_peak_anchor_and_reuse():
    rows = [fa2g.manifest_row(s) for s in fa2g.build_run_specs()]

    assert {
        "tag",
        "label",
        "seed",
        "paired_f0_label",
        "stft_dist_weight",
        "stft_band_energy_weight",
        "stft_peak_anchor_weight",
        "stft_peak_anchor_sigma_bins",
        "stft_peak_anchor_mode",
        "stft_peak_anchor_guard_tolerance_bpm",
        "stft_peak_anchor_target_quantile",
        "stft_sample_weight_mode",
        "reuse_existing",
        "overrides",
    } <= set(rows[0])
    assert {r["reuse_existing"] for r in rows if r["label"] == "F-A2f_peak_anchor_w005"} == {"true"}
    assert {r["reuse_existing"] for r in rows if r["label"] == "F-A2g_targetRRAware_w005"} == {"false"}


def test_f_a2_target_rr_anchor_command_uses_new_run_root_and_guard_overrides():
    spec = [s for s in fa2g.build_run_specs() if s["label"] == "F-A2g_targetRRAware_w005"][0]
    joined = " ".join(fa2g._command_for_spec(spec, "cuda:1"))

    assert "outputs.run_root=runs/f_a2_target_rr_anchor/f_a2g_targetrraware_w005/dual" in joined
    assert "loss.stft_dist_weight=0.02" in joined
    assert "loss.stft_band_energy_weight=0.005" in joined
    assert "loss.stft_peak_anchor_weight=0.005" in joined
    assert "loss.stft_peak_anchor_sigma_bins=1.0" in joined
    assert "loss.stft_peak_anchor_mode=target_rr_guard" in joined
    assert "loss.stft_peak_anchor_guard_tolerance_bpm=2.0" in joined
    assert "loss.stft_peak_anchor_target_quantile=0.5" in joined
    assert "loss.log_component_grad_norms=true" in joined
    assert "training.device=cuda:1" in joined
