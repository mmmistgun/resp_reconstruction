from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from resp_train.paper_evidence import e4_r3_norm_runtime as experiment


def main():
    parser=argparse.ArgumentParser(description='W0 R3 幅度时序×GN 统计量 validation 诊断')
    phases=parser.add_subparsers(dest='phase',required=True)
    phases.add_parser('prepare-lock'); phases.add_parser('signals')
    phases.add_parser('gpu-smoke').add_argument('--device',default='cuda:0')
    evaluate=phases.add_parser('evaluate')
    evaluate.add_argument('--seed',type=int,choices=experiment.SEEDS,required=True)
    evaluate.add_argument('--device',default='cuda:0')
    evaluate.add_argument('--signals',type=Path,required=True)
    evaluate.add_argument('--gpu-receipt',type=Path,required=True)
    phases.add_parser('summarize')
    args=parser.parse_args()
    if args.phase=='prepare-lock': out=experiment.prepare_lock()
    elif args.phase=='signals': out=experiment.signal_entry()
    elif args.phase=='gpu-smoke': out=experiment.gpu_smoke(args.device)
    elif args.phase=='evaluate': out=experiment.evaluate(args.seed,args.device,args.signals.resolve(),args.gpu_receipt.resolve())
    else: out=experiment.summarize()
    print(out)


if __name__=='__main__': main()
