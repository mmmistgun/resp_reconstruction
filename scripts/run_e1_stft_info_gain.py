from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


COMMON_OVERRIDES = [
    "data.max_train_windows=null",
    "data.max_val_windows=null",
    "data.train_sample_seed=20260610",
    "data.val_sample_seed=20260611",
    "loss.phase_alignment_weight=0.0",
    "loss.signed_corr_weight=0.2",
    "training.epochs=50",
    "training.batch_size=128",
    "training.patience=8",
    "training.min_delta=0.001",
    "training.use_amp=false",
    "training.show_progress=false",
    "training.checkpoint_gate.metric=auto_direction",
    "training.checkpoint_gate.max=0.5",
    "outputs.run_root=runs/tho_research_v2_20260620_e1_stft_info_gain",
]

TIME_BACKBONES = ["patch_mixer1d", "multiscale_decomp_mixer1d"]
HIGH_HZ_BANDS = [3.0, 8.0, 12.0]
SEEDS = [20260700, 20260710, 20260837]

STFT_BASE = [
    "model.name=time_stft_dual1d",
    "model.base_channels=16",
    "model.mixer_layers=2",
    "model.patch_len=256",
    "model.patch_stride=128",
    "model.overlap_window=hann",
    "model.output_smoothing_kernel=1",
    "model.stft_win=3000",
    "model.stft_hop=500",
    "model.stft_low_hz=0.05",
    "model.stft_out_channels=16",
    "model.stft_norm=n0",
    "model.fuse_len=600",
]

MANIFEST_FIELDS = ["tag", "label", "time_backbone", "high_hz", "encoder_type", "seed", "overrides"]
DEFAULT_MANIFEST_PATH = "runs/tho_research_v2_20260620_e1_stft_info_gain_manifest.csv"


RunSpec = dict[str, Any]


def _plain_backbone_overrides(backbone: str) -> list[str]:
    """E1a 使用原始时序模型路径，不进入 time_stft_dual1d 包装器。"""

    base = [f"model.name={backbone}", "model.base_channels=16", "model.mixer_layers=2"]
    if backbone == "patch_mixer1d":
        return [
            *base,
            "model.patch_len=256",
            "model.patch_stride=128",
            "model.overlap_window=hann",
            "model.output_smoothing_kernel=1",
        ]
    if backbone == "multiscale_decomp_mixer1d":
        return [*base, "model.downsample_factors=[1,4,16]"]
    raise ValueError(f"未知 time backbone: {backbone}")


def build_zero_ablation_specs() -> list[RunSpec]:
    """零号消融：代表档 E1b/patch_mixer1d/high_hz=8 跑 conv1d 与 conv2d。"""

    specs: list[RunSpec] = []
    for encoder_type in ("conv1d", "conv2d"):
        for seed in SEEDS:
            specs.append(
                {
                    "label": "E1z",
                    "time_backbone": "patch_mixer1d",
                    "high_hz": 8.0,
                    "encoder_type": encoder_type,
                    "seed": seed,
                    "overrides": [
                        *STFT_BASE,
                        "model.time_backbone=patch_mixer1d",
                        "model.branch_mode=dual",
                        "model.stft_high_hz=8.0",
                        f"model.stft_encoder_type={encoder_type}",
                        f"training.seed={seed}",
                    ],
                }
            )
    return specs


def build_run_specs(encoder: str = "conv1d") -> list[RunSpec]:
    """生成 E1 主线 48 个 run 规格。"""

    specs: list[RunSpec] = []
    stft_encoder = [f"model.stft_encoder_type={encoder}"]
    for backbone in TIME_BACKBONES:
        for seed in SEEDS:
            specs.append(
                {
                    "label": "E1a",
                    "time_backbone": backbone,
                    "high_hz": None,
                    "encoder_type": encoder,
                    "seed": seed,
                    "overrides": [*_plain_backbone_overrides(backbone), f"training.seed={seed}"],
                }
            )
            specs.append(
                {
                    "label": "E1a_prime",
                    "time_backbone": backbone,
                    "high_hz": None,
                    "encoder_type": encoder,
                    "seed": seed,
                    "overrides": [
                        *STFT_BASE,
                        *stft_encoder,
                        f"model.time_backbone={backbone}",
                        "model.branch_mode=time_only",
                        "model.stft_high_hz=3.0",
                        f"training.seed={seed}",
                    ],
                }
            )
            for high_hz in HIGH_HZ_BANDS:
                specs.append(
                    {
                        "label": "E1b",
                        "time_backbone": backbone,
                        "high_hz": high_hz,
                        "encoder_type": encoder,
                        "seed": seed,
                        "overrides": [
                            *STFT_BASE,
                            *stft_encoder,
                            f"model.time_backbone={backbone}",
                            "model.branch_mode=dual",
                            f"model.stft_high_hz={high_hz}",
                            f"training.seed={seed}",
                        ],
                    }
                )
                specs.append(
                    {
                        "label": "E1c",
                        "time_backbone": backbone,
                        "high_hz": high_hz,
                        "encoder_type": encoder,
                        "seed": seed,
                        "overrides": [
                            *STFT_BASE,
                            *stft_encoder,
                            f"model.time_backbone={backbone}",
                            "model.branch_mode=stft_only",
                            f"model.stft_high_hz={high_hz}",
                            f"training.seed={seed}",
                        ],
                    }
                )
    return specs


def build_n1_specs(
    encoder: str = "conv1d",
    band_scale_path: str = "runs/stft_band_scale/band_scale_3hz.npy",
) -> list[RunSpec]:
    """N1 对照：代表档 E1b/patch_mixer1d/high_hz=3 加挂 3 seed。"""

    specs: list[RunSpec] = []
    stft_base = [override for override in STFT_BASE if not override.startswith("model.stft_norm=")]
    for seed in SEEDS:
        specs.append(
            {
                "label": "E1n1",
                "time_backbone": "patch_mixer1d",
                "high_hz": 3.0,
                "encoder_type": encoder,
                "seed": seed,
                "overrides": [
                    *stft_base,
                    f"model.stft_encoder_type={encoder}",
                    "model.time_backbone=patch_mixer1d",
                    "model.branch_mode=dual",
                    "model.stft_high_hz=3.0",
                    "model.stft_norm=n1",
                    f"model.stft_band_scale_path={band_scale_path}",
                    f"training.seed={seed}",
                ],
            }
        )
    return specs


def _tag(spec: RunSpec) -> str:
    band = "na" if spec["high_hz"] is None else f"{spec['high_hz']:g}hz"
    return f"{spec['label']}_{spec['time_backbone']}_{spec['encoder_type']}_{band}_{spec['seed']}"


def manifest_row(spec: RunSpec) -> dict[str, str]:
    """生成 manifest 行，保留实验因子和完整 override 便于回溯。"""

    high_hz = "" if spec["high_hz"] is None else f"{spec['high_hz']:g}"
    return {
        "tag": _tag(spec),
        "label": str(spec["label"]),
        "time_backbone": str(spec["time_backbone"]),
        "high_hz": high_hz,
        "encoder_type": str(spec["encoder_type"]),
        "seed": str(spec["seed"]),
        "overrides": " ".join(str(override) for override in spec["overrides"]),
    }


def write_manifest(specs: list[RunSpec], path: str | Path) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_row(spec) for spec in specs)


def _command_for_spec(spec: RunSpec, device: str) -> list[str]:
    cmd = [sys.executable, "scripts/train_tho_small.py", "--config", "configs/tho_research_v2.yaml"]
    for override in [*COMMON_OVERRIDES, f"training.device={device}", *spec["overrides"]]:
        cmd.extend(["--set", override])
    return cmd


def _run_one(spec: RunSpec, device: str) -> str:
    tag = _tag(spec)
    print(f"start {tag} device={device}", flush=True)
    subprocess.run(_command_for_spec(spec, device), check=True)
    print(f"done {tag}", flush=True)
    return tag


def _run_many(specs: list[RunSpec], devices: list[str], max_parallel: int) -> None:
    """滚动提交训练任务，首个失败后停止继续启动后续实验。"""

    assignments = [(spec, devices[idx % len(devices)]) for idx, spec in enumerate(specs)]
    next_index = 0
    pending: set[Future[str]] = set()

    def submit_next(pool: ThreadPoolExecutor) -> None:
        nonlocal next_index
        if next_index >= len(assignments):
            return
        spec, device = assignments[next_index]
        next_index += 1
        pending.add(pool.submit(_run_one, spec, device))

    with ThreadPoolExecutor(max_workers=min(max_parallel, len(assignments))) as pool:
        for _ in range(min(max_parallel, len(assignments))):
            submit_next(pool)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except Exception:
                    for pending_future in pending:
                        pending_future.cancel()
                    raise
                submit_next(pool)


def _specs_for_phase(phase: str, encoder: str, band_scale_path: str) -> list[RunSpec]:
    if phase == "zero":
        return build_zero_ablation_specs()
    if phase == "main":
        return build_run_specs(encoder)
    if phase == "n1":
        return build_n1_specs(encoder=encoder, band_scale_path=band_scale_path)
    raise ValueError(f"未知 phase: {phase}")


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 STFT 信息增益批次编排")
    parser.add_argument(
        "--phase",
        choices=["zero", "main", "n1"],
        default="main",
        help="zero=零号消融；main=主线 48 run；n1=代表档 N1 对照",
    )
    parser.add_argument("--encoder", choices=["conv1d", "conv2d"], default="conv1d", help="主线 STFT 编码器")
    parser.add_argument(
        "--band-scale-path",
        default="runs/stft_band_scale/band_scale_3hz.npy",
        help="N1 阶段使用的 per-freq-bin IQR 文件",
    )
    parser.add_argument("--skip", action="append", default=[], help="跳过指定 run tag，可重复传入")
    parser.add_argument("--dry-run", action="store_true", help="只打印 plan tag，不实际训练")
    parser.add_argument("--device", action="append", default=None, help="训练设备，可重复传入；默认 cuda:0")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="并发训练进程数；同 GPU 多实验由该参数显式控制",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH, help="写出 manifest CSV 路径")
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")

    specs = _specs_for_phase(args.phase, args.encoder, args.band_scale_path)
    skipped = set(args.skip)
    known_tags = {_tag(spec) for spec in specs}
    unknown_skips = sorted(skipped - known_tags)
    if unknown_skips:
        raise SystemExit(f"未知 skip tag: {unknown_skips}")

    write_manifest(specs, args.manifest)

    runnable: list[RunSpec] = []
    for spec in specs:
        tag = _tag(spec)
        if tag in skipped:
            print(f"skip {tag}", flush=True)
            continue
        if args.dry_run:
            print(f"plan {tag}", flush=True)
            continue
        runnable.append(spec)

    if args.dry_run or not runnable:
        return

    devices = args.device or ["cuda:0"]
    _run_many(runnable, devices, args.max_parallel)


if __name__ == "__main__":
    main()
