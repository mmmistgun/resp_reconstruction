from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.baselines.fixed_band import (  # noqa: E402
    FIXED_BAND_EXPECTED_SIGNAL_KEY,
    FIXED_BAND_METHOD,
    add_baseline_summary_metadata,
    attach_alignment_metadata,
    evaluate_fixed_band_loader,
    prepare_fixed_band_config,
)
from resp_train.config import load_config  # noqa: E402
from resp_train.data.factory import build_window_data  # noqa: E402
from resp_train.utils.run import create_run_dir, save_execution_manifest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评价当前数据集的固定呼吸带 BCG 基线")
    parser.add_argument("--config", default="configs/tho_research_v2.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--run-root", default="runs/tho_fixed_band_baseline")
    parser.add_argument("--max-windows", type=int, default=None, help="仅用于实现 smoke；正式统计保持为空")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--confirm-designated-test",
        action="store_true",
        help="明确确认全部方案已冻结；test 评价必须显式提供",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.split == "test" and not args.confirm_designated_test:
        raise SystemExit("test 评价需要显式添加 --confirm-designated-test")

    cfg = prepare_fixed_band_config(load_config(args.config, overrides=args.overrides))
    split_name = str(cfg.data.val_split if args.split == "val" else cfg.data.test_split)
    strategy = str(cfg.data.val_sample_strategy if args.split == "val" else cfg.data.test_sample_strategy)
    sample_seed = int(cfg.data.val_sample_seed if args.split == "val" else cfg.data.test_sample_seed)
    configured_max = cfg.data.get("max_val_windows" if args.split == "val" else "max_test_windows")
    max_windows = args.max_windows if args.max_windows is not None else configured_max

    bundle = build_window_data(
        cfg,
        split=split_name,
        max_windows=max_windows,
        sample_strategy=strategy,
        sample_seed=sample_seed,
        shuffle=False,
    )
    metrics, summary = evaluate_fixed_band_loader(
        bundle.loader,
        cfg,
        include_test_only=args.split == "test",
    )
    metrics = attach_alignment_metadata(metrics, bundle.rows)
    summary = add_baseline_summary_metadata(summary, metrics, split=args.split)

    run_dir = create_run_dir(args.run_root)
    metrics.to_csv(run_dir / "sample_metrics.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    OmegaConf.save(config=cfg, f=run_dir / "resolved_config.yaml")
    save_execution_manifest(
        run_dir / "run_manifest.json",
        experiment_id=FIXED_BAND_METHOD,
        split=args.split,
        n_samples=len(metrics),
        max_windows=max_windows,
        source_signal_key=FIXED_BAND_EXPECTED_SIGNAL_KEY,
        include_test_only=args.split == "test",
    )
    print(summary.to_string(index=False))
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
