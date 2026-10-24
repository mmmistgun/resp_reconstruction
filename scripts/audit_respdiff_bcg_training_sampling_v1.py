#!/usr/bin/env python3
"""P2-A1：固定三个 checkpoint、八个训练 parent 的训练／采样与梯度审计。"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from resp_train.respdiff_bcg.study_audit import run_audit


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run',type=Path,required=True)
    parser.add_argument('--p2-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--confirm-data-access',action='store_true')
    parser.add_argument('--confirm-gpu-audit',action='store_true')
    args=parser.parse_args(argv)
    if not args.confirm_data_access or not args.confirm_gpu_audit:
        parser.error('真实 train 输入与 GPU 梯度诊断需要两项显式确认')
    if not args.device.startswith('cuda:') or not args.device[5:].isdigit():
        parser.error('GPU 审计需要显式 cuda:N')
    run_audit(args.output,args.baseline_run,args.p2_run,device=args.device)


if __name__=='__main__':
    main()
