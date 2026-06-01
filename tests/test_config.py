from pathlib import Path

from resp_train.config import REQUIRED_PACKAGES, load_config, check_required_packages


def test_load_default_config_has_expected_values():
    cfg = load_config("configs/tho_small.yaml")

    assert Path(cfg.data.dataset_root).name == "20260530_tho_ramp5_stage2_1"
    assert cfg.data.input_set == "mixed_zscore"
    assert cfg.data.max_train_windows == 1024
    assert cfg.data.max_val_windows == 256
    assert bool(cfg.data.filter_unusable) is True
    assert cfg.training.epochs == 3
    assert cfg.model.name == "unet1d_tiny"
    assert cfg.outputs.max_prediction_windows == 32


def test_required_packages_list_is_explicit():
    assert REQUIRED_PACKAGES == [
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
        "omegaconf",
    ]


def test_check_required_packages_returns_missing_names(monkeypatch):
    import importlib

    real_import = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "omegaconf":
            raise ImportError("missing omegaconf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    assert check_required_packages() == ["omegaconf"]


def test_load_config_applies_dotlist_overrides():
    cfg = load_config(
        "configs/tho_small.yaml",
        overrides=[
            "data.max_train_windows=16",
            "data.max_val_windows=8",
            "training.epochs=1",
        ],
    )

    assert cfg.data.max_train_windows == 16
    assert cfg.data.max_val_windows == 8
    assert cfg.training.epochs == 1
