import logging

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from resp_train.engine import collect_predictions, save_checkpoint, train_one_epoch, validate
from resp_train.losses.weak import WeakSyncLoss
from resp_train.models.registry import build_model
from resp_train.utils.run import create_run_dir, setup_logger


class DictDataset(Dataset):
    """返回训练引擎期望的最小 dict batch。"""

    def __init__(self) -> None:
        t = torch.linspace(0, 8 * torch.pi, 512)
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


def test_train_one_epoch_returns_positive_average_loss():
    cfg = _cfg()
    loader = DataLoader(DictDataset(), batch_size=2, shuffle=False)
    model = build_model(cfg)
    loss_fn = WeakSyncLoss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    summary = train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))

    assert summary["loss"] > 0


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
