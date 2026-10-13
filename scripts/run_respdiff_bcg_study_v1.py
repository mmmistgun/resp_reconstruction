#!/usr/bin/env python3
"""RespDiff 训练诊断与最终 DDPM/DDIM/N 评定计划 v1。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from resp_train.respdiff_bcg.study_runtime import (
    DEFAULT_SOURCE, freeze_final_evaluation, gpu_check, run_diagnosis, run_grid, run_training, select_training_runs,
)
from resp_train.respdiff_bcg.study_io import study_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("synthetic-smoke", "gpu-check", "diagnose", "overfit",
                                         "train", "select-training", "grid", "noise-check", "freeze-final", "full-validation"))
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-data-access", action="store_true")
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--confirm-gpu-check", action="store_true")
    parser.add_argument("--diagnostic-run", type=Path)
    parser.add_argument("--overfit-run", type=Path)
    parser.add_argument("--grid-run", type=Path)
    parser.add_argument("--source-runs", type=Path, nargs=3)
    parser.add_argument("--final-manifest-run", type=Path)
    parser.add_argument("--training-selection-run", type=Path)
    parser.add_argument("--baseline-inference-run", type=Path)
    parser.add_argument("--absolute-run", type=Path)
    parser.add_argument("--stretched-run", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--regime", choices=("baseline", "long_absolute", "long_stretched"))
    parser.add_argument("--training-seed", type=int, default=20260811)
    args = parser.parse_args(argv)
    if args.phase == "freeze-final":
        if args.grid_run is None or args.source_runs is None:
            parser.error("冻结清单需要 --grid-run 和 --source-runs 的三个模型目录")
        freeze_final_evaluation(args.output,grid_run=args.grid_run,source_runs=args.source_runs)
        return
    if args.phase == "select-training":
        if any(value is None for value in (args.baseline_inference_run, args.absolute_run, args.stretched_run)):
            parser.error("训练选择须提供 --baseline-inference-run、--absolute-run、--stretched-run")
        select_training_runs(args.output, baseline_inference_run=args.baseline_inference_run,
                             absolute_run=args.absolute_run, stretched_run=args.stretched_run)
        return
    if args.phase == "synthetic-smoke":
        torch.set_num_threads(1)
        with study_run(args.output, "synthetic-smoke", synthetic=True) as (out, receipt):
            run_diagnosis(out / "diagnose", tiny=True)
            run_training(out / "train", regime="long_absolute", tiny=True)
            run_grid(out / "grid", tiny=True)
            receipt["coverage"] = "tiny CPU diagnosis, snapshot training and all five sampler step grids"
        return
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        parser.error("真实运行或 GPU 合成检查必须使用显式 cuda:N")
    if args.phase == "gpu-check":
        if not args.confirm_gpu_check:
            parser.error("GPU 合成检查需要 --confirm-gpu-check")
        gpu_check(args.output, device=args.device)
        return
    if not args.confirm_data_access:
        parser.error("真实 train/validation 访问需要 --confirm-data-access；test 未开放")
    if args.phase in ("overfit", "train"):
        if not args.confirm_training:
            parser.error("训练需要 --confirm-training")
        if args.phase == "train" and args.regime is None:
            parser.error("正式训练须指定 --regime")
        run_training(args.output, regime="overfit" if args.phase == "overfit" else args.regime,
            seed=args.training_seed, source_run=args.source_run, diagnostic_run=args.diagnostic_run,
            overfit_run=args.overfit_run, device=args.device, resume=args.resume_checkpoint)
    elif args.phase == "diagnose":
        run_diagnosis(args.output, source_run=args.source_run, device=args.device,
                      expected_sha256=args.checkpoint_sha256)
    else:
        if args.phase in ("full-validation", "noise-check") and args.grid_run is None:
            parser.error("最终 validation 须指定 --grid-run，复用冻结选择")
        if args.phase == "grid" and args.grid_run is not None:
            parser.error("grid 阶段不能传 --grid-run")
        run_grid(args.output, source_run=args.source_run, device=args.device, grid_run=args.grid_run,
                 expected_sha256=args.checkpoint_sha256, training_selection_run=args.training_selection_run,
                 noise_check=args.phase == "noise-check", final_manifest_run=args.final_manifest_run)


if __name__ == "__main__":
    main()
