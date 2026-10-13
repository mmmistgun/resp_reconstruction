#!/usr/bin/env python3
"""固定 P4-N2 扩展与只读汇总入口。"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from resp_train.respdiff_bcg.study_extension import run_extension,summarize_extensions


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['run','summarize'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grid-run',type=Path,required=True)
    parser.add_argument('--source-run',type=Path)
    parser.add_argument('--sampler',choices=['ddim','ddpm'])
    parser.add_argument('--ddim-run',type=Path)
    parser.add_argument('--ddpm-run',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--confirm-data-access',action='store_true')
    args=parser.parse_args(argv)
    if args.phase=='summarize':
        if args.ddim_run is None or args.ddpm_run is None:parser.error('汇总需要两条完整扩展运行')
        summarize_extensions(args.output,grid_run=args.grid_run,ddim_run=args.ddim_run,ddpm_run=args.ddpm_run)
    else:
        if not args.confirm_data_access or args.source_run is None or args.sampler is None:
            parser.error('运行需要显式真实数据授权、source-run 和 sampler')
        if not args.device.startswith('cuda:') or not args.device[5:].isdigit():parser.error('需要显式 cuda:N')
        run_extension(args.output,sampler=args.sampler,source_run=args.source_run,grid_run=args.grid_run,device=args.device)


if __name__=='__main__':main()
