#!/usr/bin/env python3
"""P2-A2：固定 1600→3200 的 lr=1e-5 诊断分支。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resp_train.respdiff_bcg.study_cooldown import run_cooldown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-run', 'p1-run', 'p2-run', 'audit-run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--confirm-data-access', action='store_true')
    parser.add_argument('--confirm-training', action='store_true')
    args = parser.parse_args(argv)
    if not args.confirm_data_access or not args.confirm_training:
        parser.error('真实数据与降档训练需要两项显式确认')
    if not args.device.startswith('cuda:') or not args.device[5:].isdigit():
        parser.error('需要显式 cuda:N')
    run_cooldown(args.output, args.baseline_run, args.p1_run, args.p2_run, args.audit_run, device=args.device)


if __name__ == '__main__':
    main()
