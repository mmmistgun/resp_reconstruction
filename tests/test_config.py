from pathlib import Path

from resp_train.config import REQUIRED_PACKAGES, load_config, check_required_packages


def test_load_default_config_has_expected_values():
    cfg = load_config("configs/tho_small.yaml")

    assert Path(cfg.data.dataset_root).name == "20260530_tho_ramp5_stage2_1"
    assert cfg.data.input_set == "mixed_zscore"
    assert cfg.data.max_train_windows == 1024
    assert cfg.data.max_val_windows == 256
    assert cfg.data.train_sample_strategy == "stratified_random"
    assert cfg.data.val_sample_strategy == "stratified_random"
    assert cfg.data.train_sample_seed == 20260601
    assert cfg.data.val_sample_seed == 20260602
    assert cfg.data.stratify_column == "residual_quality_class"
    assert bool(cfg.data.filter_unusable) is True
    assert cfg.training.epochs == 3
    assert cfg.training.patience == 5
    assert cfg.training.min_delta == 0.0
    assert cfg.training.lr_scheduler == "none"
    assert cfg.training.grad_clip_norm is None
    assert bool(cfg.training.use_amp) is False
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


def test_load_config_applies_sampling_overrides():
    cfg = load_config(
        "configs/tho_small.yaml",
        overrides=[
            "data.train_sample_strategy=random",
            "data.val_sample_strategy=head",
            "data.train_sample_seed=7",
            "data.val_sample_seed=8",
            "training.patience=2",
        ],
    )

    assert cfg.data.train_sample_strategy == "random"
    assert cfg.data.val_sample_strategy == "head"
    assert cfg.data.train_sample_seed == 7
    assert cfg.data.val_sample_seed == 8
    assert cfg.training.patience == 2


def test_load_research_v2_config_has_expected_format():
    cfg = load_config("configs/tho_research_v2.yaml")

    assert cfg.data.format == "research_v2"
    assert cfg.data.target_task == "waveform"
    assert cfg.data.bcg_input_key == "bcg_input_aligned_key"
    assert cfg.data.target_key == "target_waveform_key"
    assert cfg.data.stratify_column == "allowed_losses"
    assert cfg.loss.smooth_weight == 0.10
