#!/usr/bin/env python3
"""P5 固定扩展配置的三模型全量 validation。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resp_train.respdiff_bcg.study_full_validation import freeze_full, run_full, summarize_full


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'run', 'summarize'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-runs', type=Path, nargs=3)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--grid-run', type=Path)
    parser.add_argument('--selection-run', type=Path)
    parser.add_argument('--manifest-run', type=Path)
    parser.add_argument('--runs', type=Path, nargs=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--confirm-data-access', action='store_true')
    args = parser.parse_args(argv)
    if args.phase == 'freeze':
        if any(v is None for v in (args.source_runs, args.grid_run, args.selection_run)):
            parser.error('freeze 需要三个来源模型、原 grid 和扩展选择')
        freeze_full(args.output, source_runs=args.source_runs, grid_run=args.grid_run, selection_run=args.selection_run)
    elif args.phase == 'run':
        if not args.confirm_data_access or args.source_run is None or args.manifest_run is None:
            parser.error('run 需要来源、冻结清单和 --confirm-data-access')
        if not args.device.startswith('cuda:') or not args.device[5:].isdigit():
            parser.error('真实全量评价必须指定 cuda:N')
        run_full(args.output, source_run=args.source_run, manifest_run=args.manifest_run, device=args.device)
    else:
        if args.manifest_run is None or args.runs is None:
            parser.error('summarize 需要冻结清单和三个完成的全量运行')
        summarize_full(args.output, manifest_run=args.manifest_run, runs=args.runs)


if __name__ == '__main__':
    main()
