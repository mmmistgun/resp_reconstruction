import scripts.run_f_a2_guard_probe as fa2


def test_f_a2_guard_specs_reuse_f0_and_original_f_a2():
    specs = fa2.build_run_specs()

    assert len(specs) == 12
    reused = [s for s in specs if s["reuse_existing"]]
    runnable = [s for s in specs if not s["reuse_existing"]]
    assert {s["label"] for s in reused} == {"F0_native_stft_pre_mixer", "F-A2_dist_bandE"}
    assert {s["label"] for s in runnable} == {"F-A2b_dist_bandE_w005", "F-A2c_dist_bandE_w003"}
    assert len(runnable) == 6


def test_f_a2_guard_candidates_only_lower_band_energy_weight():
    specs = {s["label"]: s for s in fa2.build_run_specs() if s["seed"] == 20260700}

    assert specs["F-A2_dist_bandE"]["stft_dist_weight"] == 0.02
    assert specs["F-A2_dist_bandE"]["stft_band_energy_weight"] == 0.01
    assert specs["F-A2b_dist_bandE_w005"]["stft_dist_weight"] == 0.02
    assert specs["F-A2b_dist_bandE_w005"]["stft_band_energy_weight"] == 0.005
    assert specs["F-A2c_dist_bandE_w003"]["stft_dist_weight"] == 0.02
    assert specs["F-A2c_dist_bandE_w003"]["stft_band_energy_weight"] == 0.003


def test_f_a2_guard_manifest_rows_are_pairable_and_mark_reuse():
    rows = [fa2.manifest_row(s) for s in fa2.build_run_specs()]
    candidate_rows = [r for r in rows if r["label"].startswith("F-A")]

    assert {
        "tag",
        "label",
        "seed",
        "paired_f0_label",
        "stft_dist_weight",
        "stft_band_energy_weight",
        "reuse_existing",
        "overrides",
    } <= set(rows[0])
    assert {r["paired_f0_label"] for r in candidate_rows} == {"F0_native_stft_pre_mixer"}
    assert {r["reuse_existing"] for r in rows if r["label"] == "F0_native_stft_pre_mixer"} == {"true"}
    assert {r["reuse_existing"] for r in rows if r["label"].startswith("F-A2b")} == {"false"}


def test_f_a2_guard_command_uses_new_run_root_and_grad_logging():
    spec = [s for s in fa2.build_run_specs() if s["label"] == "F-A2b_dist_bandE_w005"][0]
    joined = " ".join(fa2._command_for_spec(spec, "cuda:0"))

    assert "outputs.run_root=runs/f_a2_guard/f_a2b_dist_bande_w005/dual" in joined
    assert "loss.stft_dist_weight=0.02" in joined
    assert "loss.stft_band_energy_weight=0.005" in joined
    assert "loss.log_component_grad_norms=true" in joined
    assert "training.device=cuda:0" in joined
