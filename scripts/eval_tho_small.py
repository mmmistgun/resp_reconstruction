from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.experiments.tho import _resolve_config_path, _validate_checkpoint_config, evaluate_tho_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="用 checkpoint 生成 THO 验证指标")
    parser.add_argument("--config", default="", help="配置文件路径；为空时优先使用 checkpoint 同目录 config.yaml")
    parser.add_argument("--checkpoint", required=True, help="训练产生的 checkpoint.pt")
    parser.add_argument("--metrics-output", default="", help="指标 CSV 输出路径；为空时仅校验 checkpoint 可加载")
    parser.add_argument("--metrics-workers", type=int, default=1, help="metrics chunk 进程数；1 表示当前进程串行计算")
    parser.add_argument("--metrics-chunk-size", type=int, default=128, help="每个 metrics 进程任务处理的窗口数")
    parser.add_argument("--target-cache-dir", default="", help="target-side feature cache 目录；为空时只使用本进程内缓存")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()
    if args.metrics_workers < 1:
        raise SystemExit("--metrics-workers 必须 >= 1")
    if args.metrics_chunk_size < 1:
        raise SystemExit("--metrics-chunk-size 必须 >= 1")

    output = evaluate_tho_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config or None,
        metrics_output_path=args.metrics_output or None,
        overrides=args.overrides,
        metrics_workers=int(args.metrics_workers),
        metrics_chunk_size=int(args.metrics_chunk_size),
        target_cache_dir=Path(args.target_cache_dir) if args.target_cache_dir else None,
    )
    if args.metrics_output:
        print(f"写出指标: {output}")
    else:
        print("checkpoint 评价加载完成，未写出指标")


if __name__ == "__main__":
    main()
