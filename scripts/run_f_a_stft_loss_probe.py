from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from scripts.batch_utils import (
        DATALOADER_WORKER_OVERRIDES,
        assign_devices,
        build_launch_plan,
        resolve_devices,
        run_command_with_delay,
    )
except ModuleNotFoundError:
    from batch_utils import (
        DATALOADER_WORKER_OVERRIDES,
        assign_devices,
        build_launch_plan,
        resolve_devices,
        run_command_with_delay,
    )


COMMON_OVERRIDES = [
    "data.max_train_windows=null",
    "data.max_val_windows=null",
    "data.train_sample_seed=20260610",
    "data.val_sample_seed=20260611",
    "loss.phase_alignment_weight=0.0",
    "training.epochs=50",
    "training.batch_size=128",
    *DATALOADER_WORKER_OVERRIDES,
    "training.patience=8",
    "training.min_delta=0.001",
    "training.show_progress=false",
    "training.checkpoint_gate.metric=auto_direction",
    "training.checkpoint_gate.max=0.5",
]

PATCH_NATIVE_BASE = [
    "model.name=time_stft_dual1d",
    "model.time_backbone=patch_mixer1d",
    "model.base_channels=16",
    "model.mixer_layers=2",
    "model.patch_len=256",
    "model.patch_stride=128",
    "model.overlap_window=hann",
    "model.output_smoothing_kernel=1",
    "model.stft_win=3000",
    "model.stft_hop=500",
    "model.stft_low_hz=0.05",
    "model.stft_high_hz=8.0",
    "model.stft_out_channels=16",
    "model.stft_norm=n0",
    "model.stft_encoder_type=conv2d",
    "model.fusion_mode=native_inject",
    "model.fuse_len=600",
    "model.fusion_decoder=deep",
    "loss.stft_win_length=3000",
    "loss.stft_hop_length=500",
    "loss.stft_n_fft=3000",
    "loss.stft_center=false",
    "loss.stft_dist_low_hz=0.067",
    "loss.stft_dist_high_hz=1.2",
    "loss.stft_dist_beta=3.0",
    "loss.stft_frame_weight_mode=target_band_energy",
]

SEEDS = [20260700, 20260837, 20260901]

DIST_WEIGHT = 0.02
BAND_ENERGY_WEIGHT = 0.01

ARMS = [
    {
        "label": "F0_native_time_only",
        "branch_mode": "time_only",
        "stft_inject_position": "post_mixer",
        "stft_dist_weight": 0.0,
        "stft_band_energy_weight": 0.0,
        "paired_f0_label": "F0_native_time_only",
        "paired_time_only_label": "F0_native_time_only",
    },
    {
        "label": "F0_native_stft_pre_mixer",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "stft_dist_weight": 0.0,
        "stft_band_energy_weight": 0.0,
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
    },
    {
        "label": "F-A0_dist",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "stft_dist_weight": DIST_WEIGHT,
        "stft_band_energy_weight": 0.0,
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
    },
    {
        "label": "F-A1_bandE",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "stft_dist_weight": 0.0,
        "stft_band_energy_weight": BAND_ENERGY_WEIGHT,
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
    },
    {
        "label": "F-A2_dist_bandE",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "stft_dist_weight": DIST_WEIGHT,
        "stft_band_energy_weight": BAND_ENERGY_WEIGHT,
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
    },
]


def _slug(label: str) -> str:
    return label.lower().replace(".", "_").replace("-", "_")


def _run_root(label: str, branch_mode: str) -> str:
    return f"runs/f_a_stft_loss/{_slug(label)}/{branch_mode}"


def _base_spec(arm: dict, seed: int) -> dict:
    stft_loss_enabled = float(arm["stft_dist_weight"]) > 0 or float(arm["stft_band_energy_weight"]) > 0
    return {
        **arm,
        "seed": seed,
        "overrides": [
            *PATCH_NATIVE_BASE,
            f"model.branch_mode={arm['branch_mode']}",
            f"model.stft_inject_position={arm['stft_inject_position']}",
            f"loss.stft_dist_weight={arm['stft_dist_weight']}",
            f"loss.stft_band_energy_weight={arm['stft_band_energy_weight']}",
            f"loss.log_component_grad_norms={str(stft_loss_enabled).lower()}",
            f"outputs.run_root={_run_root(arm['label'], arm['branch_mode'])}",
            f"training.seed={seed}",
        ],
    }


def build_run_specs() -> list[dict]:
    """生成 F-A：2 个 F0 anchor + 3 个 STFT loss 候选，同 3 seed 配对。"""

    specs: list[dict] = []
    for arm in ARMS:
        for seed in SEEDS:
            specs.append(_base_spec(arm, seed))
    return specs


def _tag(spec: dict) -> str:
    return f"{_slug(spec['label'])}_{spec['branch_mode']}_{spec['seed']}"


def manifest_row(spec: dict) -> dict:
    return {
        "tag": _tag(spec),
        "label": spec["label"],
        "branch_mode": spec["branch_mode"],
        "seed": spec["seed"],
        "stft_inject_position": spec["stft_inject_position"],
        "paired_f0_label": spec["paired_f0_label"],
        "paired_time_only_label": spec["paired_time_only_label"],
        "stft_dist_weight": spec["stft_dist_weight"],
        "stft_band_energy_weight": spec["stft_band_energy_weight"],
        "overrides": " ".join(spec["overrides"]),
    }


def _command_for_spec(spec: dict, device: str) -> list[str]:
    cmd = [sys.executable, "scripts/train_tho_small.py", "--config", "configs/tho_research_v2.yaml"]
    for override in [*COMMON_OVERRIDES, f"training.device={device}", *spec["overrides"]]:
        cmd.extend(["--set", override])
    return cmd


def _run_one(spec: dict, device: str, launch_delay_sec: float = 0.0) -> str:
    tag = _tag(spec)
    return run_command_with_delay(tag, _command_for_spec(spec, device), device, launch_delay_sec)


def _assign_devices(specs: list[dict], devices: list[str]) -> list[tuple[dict, str]]:
    return assign_devices(specs, devices)


def _build_launch_plan(
    specs: list[dict],
    devices: list[str],
    max_parallel: int,
    start_stagger_sec: float,
) -> list[tuple[dict, str, float]]:
    return build_launch_plan(specs, devices, max_parallel, start_stagger_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="F-A target STFT loss pilot 编排")
    parser.add_argument("--skip", action="append", default=[], help="跳过的 run tag")
    parser.add_argument("--dry-run", action="store_true", help="只打印将运行的 tag，不实际训练")
    parser.add_argument("--device", action="append", default=None, help="训练设备，可重复传入；默认 cuda:0")
    parser.add_argument("--max-parallel", type=int, default=1, help="并发训练进程数；默认 1")
    parser.add_argument(
        "--start-stagger-sec",
        type=float,
        default=30.0,
        help="同一批并发 run 的槽位启动间隔秒数；0 表示不延迟，默认 30",
    )
    parser.add_argument("--manifest", default="runs/f_a_stft_loss_manifest.csv")
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")
    if args.start_stagger_sec < 0:
        raise SystemExit("--start-stagger-sec 必须 >= 0")

    specs = build_run_specs()
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "tag",
                "label",
                "branch_mode",
                "seed",
                "stft_inject_position",
                "paired_f0_label",
                "paired_time_only_label",
                "stft_dist_weight",
                "stft_band_energy_weight",
                "overrides",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_row(spec) for spec in specs)

    skipped = set(args.skip)
    runnable: list[dict] = []
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

    devices = resolve_devices(args.device)
    workers = min(args.max_parallel, len(runnable))
    launch_plan = _build_launch_plan(runnable, devices, workers, args.start_stagger_sec)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, spec, device, delay) for spec, device, delay in launch_plan]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
