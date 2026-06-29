import scripts.run_g_series_stft_input_probe as g


def test_g_series_cli_seed_override_runs_only_requested_seed(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "g_seed_manifest.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_g_series_stft_input_probe.py",
            "--stage",
            "g0-g1",
            "--seed",
            "20260901",
            "--dry-run",
            "--manifest",
            str(manifest),
        ],
    )

    g.main()

    out = capsys.readouterr().out
    assert "manifest=" in out
    assert "runs=10 runnable=0" in out

    rows = manifest.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 11
    assert all("20260901" in row for row in rows[1:])


def test_g_series_default_specs_cover_g0_and_g1_pilot_matrix():
    specs = g.build_run_specs()

    assert len(specs) == 20
    assert {spec["stage"] for spec in specs} == {"G0", "G1"}
    assert {spec["seed"] for spec in specs} == {20260700, 20260837}
    assert {spec["label"] for spec in specs if spec["stage"] == "G0"} == {
        "G0_time_only",
        "G0_f0_native_stft_pre_mixer",
    }
    assert {spec["label"] for spec in specs if spec["stage"] == "G1"} == {
        "G1A_wide",
        "G1B_wide",
        "G1C_wide",
        "G1D_wide",
        "G1A_resp_mid",
        "G1B_resp_mid",
        "G1C_resp_mid",
        "G1D_resp_mid",
    }


def test_g1_time_resolution_specs_change_only_declared_stft_parameters():
    specs = {spec["label"]: spec for spec in g.build_run_specs(stage="g1") if spec["seed"] == 20260700}

    assert _override_set(specs["G1A_wide"]) >= {
        "model.stft_win=3000",
        "model.stft_hop=500",
        "model.stft_low_hz=0.05",
        "model.stft_high_hz=8.0",
    }
    assert _override_set(specs["G1B_wide"]) >= {
        "model.stft_win=3000",
        "model.stft_hop=250",
        "model.stft_low_hz=0.05",
        "model.stft_high_hz=8.0",
    }
    assert _override_set(specs["G1C_resp_mid"]) >= {
        "model.stft_win=2000",
        "model.stft_hop=250",
        "model.stft_low_hz=0.05",
        "model.stft_high_hz=3.0",
    }
    assert _override_set(specs["G1D_resp_mid"]) >= {
        "model.stft_win=1500",
        "model.stft_hop=128",
        "model.stft_low_hz=0.05",
        "model.stft_high_hz=3.0",
    }


def test_g2_specs_use_c_time_parameter_and_band_matrix():
    specs = {spec["label"]: spec for spec in g.build_run_specs(stage="g2", seeds=[20260901])}

    assert set(specs) == {
        "G2_R0_wide_8p0",
        "G2_R1_resp_1p2",
        "G2_R2_resp_3p0",
        "G2_R3_strict_resp",
        "G2_R4_bandgroup",
        "G2_R5_bandenergy",
        "G2_R6_high_1p2_8p0",
        "G2_R6_high_3p0_8p0",
    }

    for spec in specs.values():
        assert spec["stage"] == "G2"
        assert spec["seed"] == 20260901
        assert spec["paired_f0_label"] == "G2_R0_wide_8p0"
        assert _override_set(spec) >= {
            "model.stft_win=2000",
            "model.stft_hop=250",
            "model.stft_inject_position=pre_mixer",
        }

    assert _override_set(specs["G2_R1_resp_1p2"]) >= {
        "model.stft_low_hz=0.05",
        "model.stft_high_hz=1.2",
        "model.stft_encoder_type=conv2d",
    }
    assert _override_set(specs["G2_R3_strict_resp"]) >= {
        "model.stft_low_hz=0.067",
        "model.stft_high_hz=1.2",
    }
    assert _override_set(specs["G2_R4_bandgroup"]) >= {"model.stft_encoder_type=bandgroup"}
    assert _override_set(specs["G2_R5_bandenergy"]) >= {"model.stft_encoder_type=bandenergy"}
    assert _override_set(specs["G2_R6_high_3p0_8p0"]) >= {
        "model.stft_low_hz=3.0",
        "model.stft_high_hz=8.0",
    }


def test_g_series_specs_keep_waveform_loss_and_native_pre_mixer_contract():
    dual_specs = [spec for spec in g.build_run_specs() if spec["branch_mode"] == "dual"]

    assert dual_specs
    for spec in dual_specs:
        overrides = _override_set(spec)
        assert "model.name=time_stft_dual1d" in overrides
        assert "model.time_backbone=patch_mixer1d" in overrides
        assert "model.fusion_mode=native_inject" in overrides
        assert "model.stft_inject_position=pre_mixer" in overrides
        assert "model.stft_encoder_type=conv2d" in overrides
        assert "loss.stft_dist_weight=0.0" in overrides
        assert "loss.stft_band_energy_weight=0.0" in overrides
        assert "loss.log_component_grad_norms=false" in overrides


def test_g_series_time_only_anchor_keeps_paired_labels_and_no_stft_injection():
    spec = next(spec for spec in g.build_run_specs(stage="g0") if spec["label"] == "G0_time_only")
    overrides = _override_set(spec)

    assert spec["branch_mode"] == "time_only"
    assert spec["paired_anchor_label"] == "G0_time_only"
    assert spec["paired_f0_label"] == "G0_time_only"
    assert spec["paired_time_only_label"] == "G0_time_only"
    assert "model.branch_mode=time_only" in overrides
    assert "model.stft_inject_position=post_mixer" in overrides


def test_g_series_manifest_rows_include_resolution_and_pairing_metadata():
    spec = next(spec for spec in g.build_run_specs(stage="g1") if spec["label"] == "G1B_resp_mid")
    row = g.manifest_row(spec)

    assert {
        "tag",
        "label",
        "stage",
        "branch_mode",
        "seed",
        "stft_win",
        "stft_hop",
        "stft_low_hz",
        "stft_high_hz",
        "stft_encoder_type",
        "paired_anchor_label",
        "paired_f0_label",
        "paired_time_only_label",
        "expected_stft_frames",
        "token_interp_ratio",
        "overrides",
    } <= set(row)
    assert row["stft_win"] == 3000
    assert row["stft_hop"] == 250
    assert row["stft_low_hz"] == 0.05
    assert row["stft_high_hz"] == 3.0
    assert row["expected_stft_frames"] == 73
    assert row["token_interp_ratio"] == 140 / 73


def test_g_series_command_uses_full_window_training_contract():
    spec = next(spec for spec in g.build_run_specs(stage="g1") if spec["label"] == "G1B_wide")
    joined = " ".join(g._command_for_spec(spec, "cuda:1"))

    assert "data.max_train_windows=null" in joined
    assert "data.max_val_windows=null" in joined
    assert "training.batch_size=128" in joined
    assert "training.checkpoint_gate.metric=auto_direction" in joined
    assert "training.device=cuda:1" in joined
    assert "outputs.run_root=runs/g_series_stft_input/g1b_wide/dual" in joined


def _override_set(spec: dict) -> set[str]:
    return set(spec["overrides"])
