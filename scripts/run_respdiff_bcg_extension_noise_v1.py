#!/usr/bin/env python3
"""P4-N2 固定配置的 sampling seeds20261013/20261014 核验。"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from resp_train.respdiff_bcg.study_extension_noise import run_noise,summarize_noise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['run','summarize'])
    for name in ('output','grid-run','reference-noise-run','selection-run'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('source-run','main-run','ddim-run','ddpm-run'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--sampler',choices=['ddim','ddpm'])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--confirm-data-access',action='store_true')
    args=parser.parse_args(argv)
    common={'grid_run':args.grid_run,'reference_noise_run':args.reference_noise_run,'selection_run':args.selection_run}
    if args.phase=='run':
        if not args.confirm_data_access or args.sampler is None or args.source_run is None or args.main_run is None:
            parser.error('运行需要显式数据授权、sampler、source-run 与 main-run')
        if not args.device.startswith('cuda:') or not args.device[5:].isdigit():parser.error('需要显式 cuda:N')
        run_noise(args.output,sampler=args.sampler,source_run=args.source_run,main_run=args.main_run,device=args.device,**common)
    else:
        if args.ddim_run is None or args.ddpm_run is None:parser.error('汇总需要两条完整核验运行')
        summarize_noise(args.output,ddim_run=args.ddim_run,ddpm_run=args.ddpm_run,**common)


if __name__=='__main__':main()
