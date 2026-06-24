from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import OmegaConf

try:
    from scripts.batch_utils import (
        DATALOADER_WORKER_OVERRIDES,
        build_launch_plan,
        resolve_devices,
        run_command_with_delay,
    )
except ModuleNotFoundError:
    from batch_utils import (
        DATALOADER_WORKER_OVERRIDES,
        build_launch_plan,
        resolve_devices,
        run_command_with_delay,
    )

# E5-A2：在同 seed time-only checkpoint 上启动 cross-attention，time backbone 使用较低学习率。
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
    "model.branch_mode=dual",
    "model.fusion_mode=cross_attention_inject",
    "model.stft_inject_position=pre_mixer",
    "model.cross_attention_heads=2",
]

SEEDS = [20260700, 20260837, 20260901]
TIME_BACKBONE_LR = 0.0001
NEW_MODULE_LR = 0.001


def _slug(label: str) -> str:
    return label.lower().replace(".", "_").replace("-", "_")


def _run_root(label: str) -> str:
    return f"runs/e5_a2/{_slug(label)}/dual"


def _checkpoint_seed(config_path: Path) -> int | None:
    try:
        cfg = OmegaConf.load(config_path)
    except Exception:
        return None
    seed = OmegaConf.select(cfg, "training.seed")
    return None if seed is None else int(seed)


def _discover_warm_start_checkpoints(root: Path) -> dict[int, Path]:
    discovered: dict[int, Path] = {}
    for config_path in root.rglob("config.yaml"):
        seed = _checkpoint_seed(config_path)
        if seed is None:
            continue
        checkpoint_path = config_path.parent / "checkpoint.pt"
        if checkpoint_path.exists():
            discovered[seed] = checkpoint_path
    return discovered


def build_run_specs(warm_start_root: str | Path = "runs/e5_a1/e5_a1t_native_time_only/time_only") -> list[dict]:
    root = Path(warm_start_root)
    checkpoints = _discover_warm_start_checkpoints(root)
    missing = [seed for seed in SEEDS if seed not in checkpoints]
    if missing:
        raise FileNotFoundError(f"缺少 seed={missing} 对应的 warm-start checkpoint，root={root}")

    label = "E5-A2.0_cross_attention_warm_start"
    specs: list[dict] = []
    for seed in SEEDS:
        checkpoint = checkpoints[seed]
        specs.append(
            {
                "label": label,
                "branch_mode": "dual",
                "seed": seed,
                "fusion_mode": "cross_attention_inject",
                "stft_inject_position": "pre_mixer",
                "cross_attention_heads": 2,
                "stft_encoder_type": "conv2d",
                "warm_start_checkpoint": str(checkpoint),
                "time_backbone_learning_rate": TIME_BACKBONE_LR,
                "learning_rate": NEW_MODULE_LR,
                "overrides": [
                    *PATCH_STFT_BASE,
                    f"outputs.run_root={_run_root(label)}",
                    f"training.seed={seed}",
                    f"training.warm_start_checkpoint={checkpoint}",
                    "training.warm_start_prefixes=[time_backbone.]",
                    f"training.time_backbone_learning_rate={TIME_BACKBONE_LR}",
                    f"training.learning_rate={NEW_MODULE_LR}",
                ],
            }
        )
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
        "cross_attention_heads": spec["cross_attention_heads"],
        "stft_encoder_type": spec["stft_encoder_type"],
        "warm_start_checkpoint": spec["warm_start_checkpoint"],
        "time_backbone_learning_rate": spec["time_backbone_learning_rate"],
        "learning_rate": spec["learning_rate"],
        "overrides": " ".join(spec["overrides"]),
    }


def _command_for_spec(spec: dict, device: str) -> list[str]:
    cmd = [sys.executable, "scripts/train_e5_a2_tho.py", "--config", "configs/tho_research_v2.yaml"]
    for override in [*COMMON_OVERRIDES, f"training.device={device}", *spec["overrides"]]:
        cmd.extend(["--set", override])
    return cmd


def _run_one(spec: dict, device: str, launch_delay_sec: float = 0.0) -> str:
    tag = _tag(spec)
    return run_command_with_delay(tag, _command_for_spec(spec, device), device, launch_delay_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="E5-A2 cross-attention warm-start 融合探针编排")
    parser.add_argument("--warm-start-root", default="runs/e5_a1/e5_a1t_native_time_only/time_only")
    parser.add_argument("--skip", action="append", default=[], help="跳过的 run tag")
    parser.add_argument("--dry-run", action="store_true", help="只打印将运行的 tag，不实际训练")
    parser.add_argument("--device", action="append", default=None, help="训练设备，可重复传入；默认 cuda:0")
    parser.add_argument("--max-parallel", type=int, default=1, help="并发训练进程数；默认 1")
    parser.add_argument("--start-stagger-sec", type=float, default=30.0, help="同一批并发 run 的槽位启动间隔秒数")
    parser.add_argument("--manifest", default="runs/e5_a2_manifest.csv")
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")
    if args.start_stagger_sec < 0:
        raise SystemExit("--start-stagger-sec 必须 >= 0")

    specs = build_run_specs(warm_start_root=args.warm_start_root)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(manifest_row(specs[0]).keys()))
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
    launch_plan = build_launch_plan(runnable, devices, workers, args.start_stagger_sec)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, spec, device, delay) for spec, device, delay in launch_plan]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
