from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "figure"
    / "F0_native_stft_pre_mixer"
    / "instantiate_f0_native_stft_pre_mixer.py"
)


def _load_instantiation_module():
    spec = importlib.util.spec_from_file_location("instantiate_f0_native_stft_pre_mixer", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_f0_instantiation_file_does_not_use_config_or_registry() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "from resp_train." not in source
    assert "from resp_train.config import load_config" not in source
    assert "from resp_train.models.registry import build_model" not in source
    assert "from resp_train.models.stft_branch import TimeStftDual1D" not in source
    assert "load_config(" not in source
    assert "build_model(" not in source
    assert "torch.load(" not in source
    assert "load_f0_checkpoint" not in source
    assert "checkpoint" not in source.lower()


def test_f0_instantiation_uses_explicit_model_parameters() -> None:
    module = _load_instantiation_module()

    model = module.instantiate_f0_model(device="cpu")

    assert isinstance(model, module.TimeStftDual1D)
    assert model.branch_mode == "dual"
    assert model.fusion_mode == "native_inject"
    assert model.stft_inject_position == "pre_mixer"
    assert model.fuse_len == 600
    assert model.time_backbone.patch_len == 256
    assert model.time_backbone.patch_stride == 128
    assert len(model.time_backbone.blocks) == 2
    assert model.stft_encoder.stft_win == 3000
    assert model.stft_encoder.stft_hop == 500
    assert model.stft_encoder.low_hz == 0.05
    assert model.stft_encoder.high_hz == 8.0
    assert model.stft_encoder.out_channels == 16
    assert model.stft_encoder.norm == "n0"
    assert not model.training
    assert next(model.parameters()).device.type == "cpu"
    assert module.count_parameters(model) == (14192, 14192)


def test_f0_instantiation_file_contains_runnable_model_structure() -> None:
    module = _load_instantiation_module()
    model = module.instantiate_f0_model(device="cpu")

    with torch.no_grad():
        output = model(torch.zeros(1, 1, 18000))

    assert output.shape == (1, 1, 18000)
