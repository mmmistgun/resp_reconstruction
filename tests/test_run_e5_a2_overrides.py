from pathlib import Path

import pytest

import scripts.run_e5_a2_cross_attention_warm_start_probe as e5a2


def _write_time_only_run(root: Path, seed: int) -> Path:
    run_dir = root / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.pt").write_bytes(b"placeholder")
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "training:",
                f"  seed: {seed}",
                "model:",
                "  branch_mode: time_only",
            ]
        ),
        encoding="utf-8",
    )
    return run_dir / "checkpoint.pt"


def test_e5_a2_specs_require_seed_matched_warm_start_checkpoints(tmp_path):
    warm_root = tmp_path / "time_only"
    ckpts = {seed: _write_time_only_run(warm_root, seed) for seed in e5a2.SEEDS}

    specs = e5a2.build_run_specs(warm_start_root=warm_root)

    assert len(specs) == 3
    assert {s["label"] for s in specs} == {"E5-A2.0_cross_attention_warm_start"}
    assert {s["seed"] for s in specs} == set(e5a2.SEEDS)
    for spec in specs:
        joined = " ".join(spec["overrides"])
        assert f"training.warm_start_checkpoint={ckpts[spec['seed']]}" in joined
        assert "training.time_backbone_learning_rate=0.0001" in joined
        assert "training.learning_rate=0.001" in joined
        assert "model.fusion_mode=cross_attention_inject" in joined
        assert "model.cross_attention_heads=2" in joined


def test_e5_a2_missing_warm_start_seed_raises(tmp_path):
    _write_time_only_run(tmp_path / "time_only", e5a2.SEEDS[0])

    with pytest.raises(FileNotFoundError, match="warm-start checkpoint"):
        e5a2.build_run_specs(warm_start_root=tmp_path / "time_only")


def test_e5_a2_command_uses_dedicated_train_entry(tmp_path):
    warm_root = tmp_path / "time_only"
    for seed in e5a2.SEEDS:
        _write_time_only_run(warm_root, seed)
    spec = e5a2.build_run_specs(warm_start_root=warm_root)[0]

    joined = " ".join(e5a2._command_for_spec(spec, "cuda:1"))

    assert "scripts/train_e5_a2_tho.py" in joined
    assert "training.device=cuda:1" in joined
    assert "data.max_train_windows=null" in joined
    assert "training.checkpoint_gate.metric=auto_direction" in joined
