#!/usr/bin/env python3
"""显式授权的 RespDiff-BCG P6 独立 research-test。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resp_train.respdiff_bcg.research_test import CHECKPOINTS, freeze, preflight, gpu_check, run_test, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'preflight', 'gpu-check', 'run', 'summarize'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest-run', type=Path)
    parser.add_argument('--preflight-run', type=Path)
    parser.add_argument('--training-seed', type=int, choices=sorted(CHECKPOINTS))
    parser.add_argument('--runs', nargs=3, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--confirm-research-test', action='store_true')
    parser.add_argument('--confirm-gpu-check', action='store_true')
    args = parser.parse_args(argv)
    if args.phase in ('freeze', 'preflight', 'run') and not args.confirm_research_test:
        parser.error('此阶段需要 --confirm-research-test')
    if args.phase in ('preflight', 'run', 'summarize') and args.manifest_run is None:
        parser.error('需要 --manifest-run')
    if args.phase in ('run', 'gpu-check') and (not args.device.startswith('cuda:') or not args.device[5:].isdigit()):
        parser.error('必须显式指定 cuda:N')
    if args.phase == 'freeze': freeze(args.output)
    elif args.phase == 'preflight': preflight(args.output, args.manifest_run)
    elif args.phase == 'gpu-check':
        if not args.confirm_gpu_check: parser.error('需要 --confirm-gpu-check')
        gpu_check(args.output, device=args.device)
    elif args.phase == 'run':
        if args.training_seed is None or args.preflight_run is None: parser.error('需要 seed 和 preflight')
        run_test(args.output, seed=args.training_seed, manifest_run=args.manifest_run,
                 preflight_run=args.preflight_run, device=args.device)
    else:
        if args.runs is None: parser.error('需要三个完整 --runs')
        summarize(args.output, runs=args.runs, manifest_run=args.manifest_run)


if __name__ == '__main__': main()
