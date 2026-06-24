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

# E5-A0：在 E3-C1B token_pre_mixer 主线之上，只改变 STFT token delta 的使用方式。
# A0.0 复现 ungated native pre-mixer，A0.1 加轻量逐 token/channel gate；
# 二者共享同 seed 的 native time-only substrate，避免把 native 解码器强度误解释为 gate 收益。
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

PATCH_STFT_BASE = [
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
    "model.fuse_len=600",
    "model.fusion_decoder=deep",
]

SEEDS = [20260700, 20260837, 20260901]

TIME_ONLY_SUBSTRATE = {
    "label": "E5-A0T_native_time_only",
    "fusion_mode": "native_inject",
    "stft_inject_position": "pre_mixer",
    "paired_time_only_label": "E5-A0T_native_time_only",
}

DUAL_ARMS = [
    {
        "label": "E5-A0.0_native_pre_mixer_ungated",
        "fusion_mode": "native_inject",
        "stft_inject_position": "pre_mixer",
        "paired_time_only_label": "E5-A0T_native_time_only",
    },
    {
        "label": "E5-A0.1_gated_native_pre_mixer",
        "fusion_mode": "gated_native_inject",
        "stft_inject_position": "pre_mixer",
        "paired_time_only_label": "E5-A0T_native_time_only",
    },
]


def _slug(label: str) -> str:
    return label.lower().replace(".", "_").replace("-", "_")


def _run_root(label: str, branch_mode: str) -> str:
    return f"runs/e5_a0/{_slug(label)}/{branch_mode}"


def _base_spec(arm: dict, branch_mode: str, seed: int) -> dict:
    return {
        "label": arm["label"],
        "branch_mode": branch_mode,
        "seed": seed,
        "fusion_mode": arm["fusion_mode"],
        "stft_inject_position": arm["stft_inject_position"],
        "stft_encoder_type": "conv2d",
        "paired_time_only_label": arm["paired_time_only_label"],
        "overrides": [
            *PATCH_STFT_BASE,
            f"model.fusion_mode={arm['fusion_mode']}",
            f"model.stft_inject_position={arm['stft_inject_position']}",
            f"model.branch_mode={branch_mode}",
            f"outputs.run_root={_run_root(arm['label'], branch_mode)}",
            f"training.seed={seed}",
        ],
    }


def build_run_specs() -> list[dict]:
    """生成 E5-A0：1 个 native time-only substrate + 2 个 pre-mixer dual arm，同 3 seed 配对。"""

    specs: list[dict] = []
    for seed in SEEDS:
        specs.append(_base_spec(TIME_ONLY_SUBSTRATE, "time_only", seed))
    for arm in DUAL_ARMS:
        for seed in SEEDS:
            specs.append(_base_spec(arm, "dual", seed))
    return specs


def _tag(spec: dict) -> str:
    return f"{_slug(spec['label'])}_{spec['branch_mode']}_{spec['seed']}"


def manifest_row(spec: dict) -> dict:
    return {
        "tag": _tag(spec),
        "label": spec["label"],
        "branch_mode": spec["branch_mode"],
        "seed": spec["seed"],
        "fusion_mode": spec["fusion_mode"],
        "stft_inject_position": spec["stft_inject_position"],
        "stft_encoder_type": spec["stft_encoder_type"],
        "paired_time_only_label": spec["paired_time_only_label"],
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


def _resolve_devices(devices: list[str] | None) -> list[str]:
    return resolve_devices(devices)


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
    parser = argparse.ArgumentParser(description="E5-A0 gated/conditional STFT 融合探针编排")
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
    parser.add_argument("--manifest", default="runs/e5_a0_manifest.csv")
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
                "fusion_mode",
                "stft_inject_position",
                "stft_encoder_type",
                "paired_time_only_label",
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

    devices = _resolve_devices(args.device)
    workers = min(args.max_parallel, len(runnable))
    launch_plan = _build_launch_plan(runnable, devices, workers, args.start_stagger_sec)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, spec, device, delay) for spec, device, delay in launch_plan]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
