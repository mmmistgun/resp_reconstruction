from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.baselines.fixed_band import attach_alignment_metadata  # noqa: E402
from resp_train.baselines.iewt import (  # noqa: E402
    IEWT_EXPECTED_SIGNAL_KEY,
    IEWT_METHOD,
    add_iewt_summary_metadata,
    evaluate_iewt_loader,
    prepare_iewt_config,
)
from resp_train.config import load_config  # noqa: E402
from resp_train.data.factory import build_window_data  # noqa: E402
from resp_train.utils.run import create_run_dir, save_execution_manifest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评价协议化 Python IEWT 呼吸提取基线")
    parser.add_argument("--config", default="configs/tho_research_v2.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--run-root", default="runs/tho_iewt_baseline")
    parser.add_argument("--max-windows", type=int, default=None, help="仅用于实现 smoke；正式统计保持为空")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--confirm-research-test",
        action="store_true",
        help="明确确认本次将读取可重复观察的 research-test；test 评价必须显式提供",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.split == "test" and not args.confirm_research_test:
        raise SystemExit("test 评价需要显式添加 --confirm-research-test")

    cfg = prepare_iewt_config(load_config(args.config, overrides=args.overrides))
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
    metrics, summary = evaluate_iewt_loader(
        bundle.loader,
        cfg,
        include_test_only=args.split == "test",
    )
    metrics = attach_alignment_metadata(metrics, bundle.rows)
    summary = add_iewt_summary_metadata(summary, metrics, split=args.split)

    run_dir = create_run_dir(args.run_root)
    metrics.to_csv(run_dir / "sample_metrics.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    OmegaConf.save(config=cfg, f=run_dir / "resolved_config.yaml")
    save_execution_manifest(
        run_dir / "run_manifest.json",
        experiment_id=IEWT_METHOD,
        split=args.split,
        n_samples=len(metrics),
        max_windows=max_windows,
        source_signal_key=IEWT_EXPECTED_SIGNAL_KEY,
        algorithm="protocolized_python_iewt",
        filter_phase="zero_phase",
        matlab_pointwise_parity=False,
        include_test_only=args.split == "test",
        evaluation_role="research_test" if args.split == "test" else "validation",
    )
    print(summary.to_string(index=False))
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
