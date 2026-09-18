from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from resp_train.paper_evidence import e4_band_audit_runtime as audit


def main():
    parser=argparse.ArgumentParser(description="E4 固定 validation 的编码/聚合/入口频带诊断")
    phases=parser.add_subparsers(dest="phase",required=True)
    phases.add_parser("prepare-lock").add_argument("--validation-summary",type=Path,required=True)
    phases.add_parser("prepare-reference")
    phases.add_parser("gpu-smoke").add_argument("--device",default="cuda:0")
    evaluate=phases.add_parser("evaluate")
    evaluate.add_argument("--arm",choices=audit.ARMS,required=True)
    evaluate.add_argument("--seed",type=int,choices=audit.SEEDS,required=True)
    evaluate.add_argument("--device",default="cuda:0")
    evaluate.add_argument("--reference",type=Path,required=True)
    evaluate.add_argument("--gpu-receipt",type=Path,required=True)
    phases.add_parser("summarize")
    args=parser.parse_args()
    if args.phase=="prepare-lock":
        out=audit.prepare_lock(args.validation_summary)
    elif args.phase=="prepare-reference":
        out=audit.prepare_reference()
    elif args.phase=="gpu-smoke":
        out=audit.gpu_smoke(args.device)
    elif args.phase=="evaluate":
        out=audit.evaluate(args.arm,args.seed,args.device,args.reference.resolve(),args.gpu_receipt.resolve())
    else:
        out=audit.summarize()
    print(out)


if __name__=="__main__":
    main()
