from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages, load_config
from resp_train.data.audit import add_usable_flag, summarize_audit
from resp_train.data.dataset import RespWindowDataset
from resp_train.data.index import filter_index, read_index
from resp_train.engine.train import collect_predictions, save_checkpoint, train_one_epoch, validate
from resp_train.losses.weak import WeakSyncLoss
from resp_train.metrics.baseline import run_baseline
from resp_train.metrics.evaluate import evaluate_prediction_dict
from resp_train.models.registry import build_model
from resp_train.utils.run import create_run_dir, resolve_device, save_config, set_seed, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="训练胸带参考小规模模型")
    parser.add_argument("--config", default="configs/tho_small.yaml", help="配置文件路径")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dotlist 覆盖，例如 data.max_train_windows=16，可重复传入",
    )
    args = parser.parse_args()

    missing = check_required_packages()
    if missing:
        raise SystemExit(f"缺少依赖: {missing}; 请先确认是否安装。")

    cfg = load_config(args.config, overrides=args.overrides)
    run_dir = create_run_dir(cfg.outputs.run_root)
    save_config(cfg, run_dir)
    logger = setup_logger(run_dir)
    set_seed(int(cfg.training.seed))
    device = resolve_device(str(cfg.training.device))
    logger.info("device=%s", device)

    df = read_index(cfg.data.dataset_root, cfg.data.index_csv)
    audited = add_usable_flag(df, cfg)
    summarize_audit(audited).to_csv(run_dir / "audit.csv", index=False)
    baseline_frame = run_baseline(cfg, run_dir / "baseline_metrics.csv")
    logger.info("baseline_windows=%s", len(baseline_frame))

    train_rows = filter_index(audited, cfg, split=str(cfg.data.train_split), max_windows=cfg.data.max_train_windows)
    val_rows = filter_index(audited, cfg, split=str(cfg.data.val_split), max_windows=cfg.data.max_val_windows)
    index_path = Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)
    train_ds = RespWindowDataset(index_path, train_rows, cfg, preload_windows=bool(cfg.data.preload_windows))
    val_ds = RespWindowDataset(index_path, val_rows, cfg, preload_windows=bool(cfg.data.preload_windows))
    logger.info("train_windows=%s val_windows=%s", len(train_ds), len(val_ds))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("可训练或可验证窗口为空，请检查 filter_unusable、input_set 和 split 配置。")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
    )

    model = build_model(cfg).to(device)
    loss_fn = WeakSyncLoss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.learning_rate))

    history_records: list[dict[str, float | int]] = []
    best_loss = float("inf")
    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device=device)
        val_metrics = validate(model, val_loader, loss_fn, device=device)
        record = {"epoch": epoch, "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"]}
        history_records.append(record)
        logger.info("epoch=%s train_loss=%.6f val_loss=%.6f", epoch, record["train_loss"], record["val_loss"])
        if record["val_loss"] < best_loss:
            best_loss = record["val_loss"]
            save_checkpoint(run_dir / "checkpoint.pt", model=model, optimizer=optimizer, epoch=epoch, metrics=record, cfg=cfg)

    pd.DataFrame(history_records).to_csv(run_dir / "train_history.csv", index=False)

    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("loaded_best_checkpoint_epoch=%s", checkpoint["epoch"])

    eval_preds = collect_predictions(model, val_loader, device=device, max_windows=len(val_ds))
    evaluate_prediction_dict(eval_preds, cfg, method=str(cfg.model.name)).to_csv(run_dir / "metrics.csv", index=False)

    # predictions.npz 只保存少量窗口，完整逐窗口指标保存在 metrics.csv。
    diag_preds = collect_predictions(model, val_loader, device=device, max_windows=int(cfg.outputs.max_prediction_windows))
    np.savez(run_dir / "predictions.npz", **diag_preds)
    print(run_dir)


if __name__ == "__main__":
    main()
