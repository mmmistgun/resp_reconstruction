from __future__ import annotations

from pathlib import Path

import pytest

from resp_train.crd.config import FORMAL_SEEDS, load_crd_config
from resp_train.crd.tf_v1_model import TF_VARIANT_REPRESENTATIONS


FORMAL_CONFIGS = tuple(sorted(Path("configs/crd_tf_v1").glob("*_formal.yaml")))


def test_every_tf_arm_has_one_independent_formal_config() -> None:
    assert len(FORMAL_CONFIGS) == 15
    observed = set()
    for path in FORMAL_CONFIGS:
        cfg = load_crd_config(path)
        observed.add(str(cfg.model.variant))
    assert observed == set(TF_VARIANT_REPRESENTATIONS)


@pytest.mark.parametrize("path", FORMAL_CONFIGS, ids=lambda path: path.stem)
def test_independent_formal_config_freezes_p4_contract(path: Path) -> None:
    cfg = load_crd_config(path)
    variant = str(cfg.model.variant)

    assert str(cfg.protocol.run_role) == "formal"
    assert str(cfg.protocol.execution_gate) == "p4_formal"
    assert list(cfg.model.tf_representations) == list(TF_VARIANT_REPRESENTATIONS[variant])
    assert int(cfg.model.initialization_seed) == int(cfg.training.seed) == FORMAL_SEEDS[0]
    assert (cfg.training.epochs, cfg.training.batch_size, cfg.training.gradient_accumulation_steps) == (80, 128, 1)
    assert cfg.training.early_stopping_enabled is False
    assert str(cfg.training.device) == "cuda:0"
    assert (cfg.data.max_train_windows, cfg.data.max_val_windows, cfg.data.max_test_windows) == (None, None, None)
    assert str(cfg.outputs.run_root).endswith(f"/{variant}/seed_{FORMAL_SEEDS[0]}")


@pytest.mark.parametrize("seed", FORMAL_SEEDS)
def test_seed_override_updates_initialization_and_output_identity(seed: int) -> None:
    cfg = load_crd_config(
        "configs/crd_tf_v1/crd_tf302_wls_formal.yaml",
        overrides=[f"training.seed={seed}"],
    )

    assert int(cfg.model.initialization_seed) == seed
    assert str(cfg.outputs.run_root).endswith(f"/crd_tf302_wls/seed_{seed}")


def test_config_inheritance_is_single_level_and_same_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside.yaml"
    outside.write_text("value: 1\n", encoding="utf-8")
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = child_dir / "config.yaml"
    child.write_text("_base_: ../outside.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="同目录"):
        load_crd_config(child)
