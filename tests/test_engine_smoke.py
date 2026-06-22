import logging

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from resp_train.engine import collect_predictions, save_checkpoint, train_one_epoch, validate
from resp_train.engine.train import _move_batch
from resp_train.losses.weak import WeakSyncLoss
from resp_train.models.registry import build_model
from resp_train.utils.run import create_run_dir, setup_logger


class DictDataset(Dataset):
    """返回训练引擎期望的最小 dict batch。"""

    def __init__(self, duration_samples: int = 512) -> None:
        t = torch.linspace(0, 8 * torch.pi, int(duration_samples))
        self.samples = []
        for phase in (0.0, 0.3, 0.6, 0.9):
            sensor = torch.sin(t + phase).unsqueeze(0)
            target = torch.cos(t + phase).unsqueeze(0)
            self.samples.append({"sensor": sensor.float(), "target": target.float()})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {"x": sample["sensor"], "target": sample["target"], "meta": {"dataset_row_id": idx}}


class _FakeTensor:
    def __init__(self) -> None:
        self.to_calls = []

    def to(self, device, *, non_blocking=False):
        self.to_calls.append({"device": device, "non_blocking": non_blocking})
        return self


def _cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "unet1d_tiny",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 4,
            },
            "window": {"target_fs": 100},
            "loss": {
                "envelope_weight": 1.0,
                "spectrum_weight": 0.2,
                "smooth_weight": 0.01,
                "envelope_window_sec": 0.2,
                "spectrum_low_hz": 0.1,
                "spectrum_high_hz": 5.0,
            },
        }
    )


def _time_stft_dual1d_cfg():
    return OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 1024},
            "model": {
                "name": "time_stft_dual1d",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 4,
                "branch_mode": "dual",
                "time_backbone": "patch_mixer1d",
                "patch_len": 64,
                "patch_stride": 32,
                "mixer_layers": 1,
                "stft_win": 256,
                "stft_hop": 128,
                "stft_low_hz": 0.05,
                "stft_high_hz": 3.0,
                "stft_out_channels": 4,
                "stft_norm": "n0",
                "fuse_len": 64,
            },
            "loss": {
                "envelope_weight": 1.0,
                "spectrum_weight": 0.2,
                "smooth_weight": 0.01,
                "envelope_window_sec": 0.2,
                "spectrum_low_hz": 0.1,
                "spectrum_high_hz": 5.0,
            },
        }
    )


def test_train_one_epoch_returns_positive_average_loss():
    cfg = _cfg()
    loader = DataLoader(DictDataset(), batch_size=2, shuffle=False)
    model = build_model(cfg)
    loss_fn = WeakSyncLoss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    summary = train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))

    assert summary["loss"] > 0


def test_time_stft_dual1d_engine_smoke_trains_and_reloads_checkpoint(tmp_path):
    cfg = _time_stft_dual1d_cfg()
    loader = DataLoader(DictDataset(duration_samples=cfg.window.duration_samples), batch_size=2, shuffle=False)
    model = build_model(cfg)
    loss_fn = WeakSyncLoss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before_train = {name: param.detach().clone() for name, param in model.named_parameters()}

    summary = train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
    assert summary["loss"] > 0
    assert optimizer.state
    assert any(not torch.equal(before_train[name], param) for name, param in model.named_parameters())

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, epoch=1, metrics=summary, cfg=cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    reloaded = build_model(cfg)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    reloaded_optimizer = torch.optim.Adam(reloaded.parameters(), lr=1e-3)
    reloaded_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    reloaded.eval()

    batch = next(iter(loader))
    with torch.no_grad():
        pred = reloaded(batch["x"])

    assert pred.shape == batch["x"].shape


def test_move_batch_passes_non_blocking_to_tensor_to():
    sensor = _FakeTensor()
    target = _FakeTensor()
    device = torch.device("cuda:0")

    moved_sensor, moved_target = _move_batch(
        {"x": sensor, "target": target},
        device,
        non_blocking=True,
    )

    assert moved_sensor is sensor
    assert moved_target is target
    assert sensor.to_calls == [{"device": device, "non_blocking": True}]
    assert target.to_calls == [{"device": device, "non_blocking": True}]


def test_train_one_epoch_accepts_grad_clip_norm():
    cfg = _cfg()
    loader = DataLoader(DictDataset(), batch_size=2, shuffle=False)
    model = build_model(cfg)
    loss_fn = WeakSyncLoss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    summary = train_one_epoch(
        model,
        loader,
        loss_fn,
        optimizer,
        torch.device("cpu"),
        grad_clip_norm=0.1,
        use_amp=False,
    )

    assert summary["loss"] > 0


def test_train_one_epoch_rejects_empty_loader():
    cfg = _cfg()
    loader = DataLoader(Subset(DictDataset(), []), batch_size=2, shuffle=False)
    model = build_model(cfg)
    loss_fn = WeakSyncLoss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="没有可用 batch"):
        train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))


def test_validate_rejects_empty_loader():
    cfg = _cfg()
    loader = DataLoader(Subset(DictDataset(), []), batch_size=2, shuffle=False)
    model = build_model(cfg)
    loss_fn = WeakSyncLoss(cfg)

    with pytest.raises(ValueError, match="没有可用 batch"):
        validate(model, loader, loss_fn, torch.device("cpu"))


def test_collect_predictions_accepts_custom_output_keys():
    loader = DataLoader(DictDataset(), batch_size=2, shuffle=False)
    model = torch.nn.Identity()

    preds = collect_predictions(
        model,
        loader,
        device=torch.device("cpu"),
        max_windows=3,
        pred_key="custom_pred",
        target_key="custom_target",
    )

    assert preds["custom_pred"].shape[0] == 3
    assert preds["custom_target"].shape[0] == 3
    assert preds["dataset_row_id"].tolist() == [0, 1, 2]


def test_engine_public_api_exports_checkpoint_saver():
    assert callable(save_checkpoint)


def test_create_run_dir_allows_multiple_runs_in_same_second(tmp_path):
    first = create_run_dir(tmp_path)
    second = create_run_dir(tmp_path)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_setup_logger_closes_replaced_file_handlers(tmp_path):
    logger = logging.getLogger("resp_train")
    logger.handlers.clear()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    setup_logger(first)
    old_file_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
    assert old_file_handlers

    setup_logger(second)

    assert all(handler.stream is None for handler in old_file_handlers)


def test_time_stft_dual1d_native_inject_forward_backward_and_state_roundtrip():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "conv2d",
                "fusion_mode": "native_inject",
            },
        }
    )
    model = build_model(cfg)
    x = torch.randn(2, 1, 18000)
    target = torch.randn(2, 1, 18000)

    pred = model(x)
    (pred - target).square().mean().backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert pred.shape == (2, 1, 18000)
    assert grad_norm > 0.0

    state = model.state_dict()
    fresh = build_model(cfg)
    fresh.load_state_dict(state)
    model.eval()
    fresh.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), fresh(x), atol=1e-5)
