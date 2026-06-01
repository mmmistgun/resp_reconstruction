from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import load_config
from resp_train.data.audit import add_usable_flag
from resp_train.data.dataset import RespWindowDataset
from resp_train.data.index import filter_index, read_index
from resp_train.engine.train import collect_predictions
from resp_train.metrics.evaluate import evaluate_prediction_dict
from resp_train.models.registry import build_model
from resp_train.utils.run import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="用 checkpoint 生成验证预测和指标")
    parser.add_argument("--config", default="", help="配置文件路径；为空时优先使用 checkpoint 同目录 config.yaml")
    parser.add_argument("--checkpoint", required=True, help="训练产生的 checkpoint.pt")
    parser.add_argument("--output", required=True, help="预测 NPZ 输出路径")
    parser.add_argument("--metrics-output", default="", help="可选指标 CSV 输出路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = _resolve_config_path(args.config, checkpoint_path)
    cfg = load_config(config_path, overrides=args.overrides)
    device = resolve_device(str(cfg.training.device))
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _validate_checkpoint_config(checkpoint.get("config"), cfg)
    model.load_state_dict(checkpoint["model_state_dict"])

    df = read_index(cfg.data.dataset_root, cfg.data.index_csv)
    audited = add_usable_flag(df, cfg)
    val_rows = filter_index(audited, cfg, split=str(cfg.data.val_split), max_windows=cfg.data.max_val_windows)
    if len(val_rows) == 0:
        raise RuntimeError("可评价窗口为空，请检查 input_set、split 和可用性过滤配置。")

    dataset = RespWindowDataset(
        Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv),
        val_rows,
        cfg,
        preload_windows=False,
    )
    loader = DataLoader(dataset, batch_size=int(cfg.training.batch_size), shuffle=False, num_workers=0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    diag_preds = collect_predictions(model, loader, device=device, max_windows=int(cfg.outputs.max_prediction_windows))
    np.savez(output, **diag_preds)
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        eval_preds = collect_predictions(model, loader, device=device, max_windows=len(dataset))
        evaluate_prediction_dict(eval_preds, cfg, method=str(cfg.model.name)).to_csv(metrics_path, index=False)
    print(f"写出预测: {output}")


def _resolve_config_path(config_arg: str, checkpoint_path: Path) -> Path:
    """解析评价配置；默认复用训练 run 目录中的配置快照。"""
    if config_arg:
        return Path(config_arg)
    sidecar = checkpoint_path.parent / "config.yaml"
    if sidecar.exists():
        return sidecar
    raise FileNotFoundError("未指定 --config，且 checkpoint 同目录不存在 config.yaml")


def _validate_checkpoint_config(
    checkpoint_config: dict | None,
    cfg: DictConfig,
    *,
    keys: tuple[str, ...] | list[str] = (
        "data.dataset_root",
        "data.index_csv",
        "data.input_set",
        "data.val_split",
        "data.max_val_windows",
        "data.filter_unusable",
        "data.valid_ratio_min",
        "data.input_finite_ratio_min",
        "data.target_finite_ratio_min",
        "data.unusable_residual_classes",
        "window.target_fs",
        "window.duration_samples",
        "model.name",
        "model.in_channels",
        "model.out_channels",
        "model.base_channels",
        "loss.envelope_window_sec",
        "loss.spectrum_low_hz",
        "loss.spectrum_high_hz",
    ),
) -> None:
    """校验评价配置和 checkpoint 记录的训练配置是否一致。"""
    if checkpoint_config is None:
        raise ValueError("checkpoint 缺少训练配置，无法校验评价配置一致性")
    checkpoint_cfg = OmegaConf.create(checkpoint_config)
    mismatched: list[str] = []
    for key in keys:
        checkpoint_value = _plain_value(OmegaConf.select(checkpoint_cfg, key))
        current_value = _plain_value(OmegaConf.select(cfg, key))
        if checkpoint_value != current_value:
            mismatched.append(f"{key}: checkpoint={checkpoint_value!r} current={current_value!r}")
    if mismatched:
        details = "; ".join(mismatched)
        raise ValueError(f"checkpoint 配置与当前配置不一致: {details}")


def _plain_value(value):
    """将 OmegaConf 容器转成普通 Python 值，便于稳定比较。"""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


if __name__ == "__main__":
    main()
