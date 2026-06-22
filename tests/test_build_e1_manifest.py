import sys

import scripts.build_e1_stft_info_gain_manifest as e1


def test_build_overrides_covers_four_labels():
    labels = {spec["label"] for spec in e1.build_run_specs()}

    assert {"E1a", "E1a_prime", "E1b", "E1c"} <= labels


def test_e1a_is_plain_backbone_not_wrapper():
    spec = next(
        s for s in e1.build_run_specs() if s["label"] == "E1a" and s["time_backbone"] == "patch_mixer1d"
    )
    joined = " ".join(spec["overrides"])

    # E1a 直接用现有模型，不经双分支包装器。
    assert "model.name=patch_mixer1d" in joined
    assert "time_stft_dual1d" not in joined
    assert "model.overlap_window=hann" in joined


def test_e1b_dual_sweeps_three_bands():
    bands = sorted({
        float(override.split("=")[1])
        for spec in e1.build_run_specs()
        if spec["label"] == "E1b" and spec["time_backbone"] == "patch_mixer1d"
        for override in spec["overrides"]
        if override.startswith("model.stft_high_hz=")
    })

    assert bands == [3.0, 8.0, 12.0]


def test_e1a_prime_is_wrapper_time_only():
    spec = next(
        s for s in e1.build_run_specs() if s["label"] == "E1a_prime" and s["time_backbone"] == "patch_mixer1d"
    )
    joined = " ".join(spec["overrides"])

    assert "model.name=time_stft_dual1d" in joined
    assert "model.branch_mode=time_only" in joined


def test_n1_phase_builds_three_representative_specs():
    specs = e1.build_n1_specs(encoder="conv1d", band_scale_path="runs/stft_band_scale/band_scale_3hz.npy")

    assert len(specs) == 3
    assert all(spec["label"] == "E1n1" for spec in specs)
    assert all("model.stft_norm=n1" in spec["overrides"] for spec in specs)
    assert all("model.stft_norm=n0" not in spec["overrides"] for spec in specs)
    assert all(
        "model.stft_band_scale_path=runs/stft_band_scale/band_scale_3hz.npy" in spec["overrides"] for spec in specs
    )


def test_phase_spec_counts_match_plan():
    assert len(e1.build_zero_ablation_specs()) == 6
    assert len(e1.build_run_specs()) == 48
    assert len(e1.build_n1_specs()) == 3


def test_manifest_rows_preserve_run_factors():
    spec = e1.build_run_specs()[0]
    row = e1.manifest_row(spec)

    assert {"tag", "label", "time_backbone", "high_hz", "encoder_type", "seed", "overrides"} <= set(row)


def test_write_manifest_writes_csv(tmp_path):
    manifest = tmp_path / "e1_manifest.csv"

    e1.write_manifest(e1.build_n1_specs(), manifest)

    text = manifest.read_text(encoding="utf-8")
    assert text.startswith("tag,label,time_backbone,high_hz,encoder_type,seed,overrides")
    assert text.count("\n") == 4


def test_default_manifest_path_is_defined_for_manifest_generation():
    assert e1.DEFAULT_MANIFEST_PATH.endswith("_manifest.csv")


def test_common_overrides_enable_e1_startup_acceleration():
    assert "data.drop_nonfinite_windows=false" in e1.COMMON_OVERRIDES
    assert "baseline.enabled=false" in e1.COMMON_OVERRIDES
    assert not any(override.startswith("baseline.metrics_cache_path=") for override in e1.COMMON_OVERRIDES)


def test_main_writes_manifest_only(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_e1_stft_info_gain_manifest.py",
            "--phase",
            "zero",
            "--manifest",
            str(manifest),
        ],
    )

    e1.main()

    assert manifest.exists()
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 7


def test_script_does_not_manage_parallel_training():
    assert not hasattr(e1, "_run_one")
    assert not hasattr(e1, "_run_many")
    assert not hasattr(e1, "_command_for_spec")
    assert not hasattr(e1, "write_commands")
