from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# E1-D 实验口径：仅覆盖与默认 yaml 不同的项；极性 warmup 已固化进
# configs/tho_research_v2.yaml 默认 loss 段，这里不再覆盖（让默认生效）。
COMMON_OVERRIDES = [
    "data.max_train_windows=null",
    "data.max_val_windows=null",
    "data.train_sample_seed=20260610",
    "data.val_sample_seed=20260611",
    "loss.phase_alignment_weight=0.0",
    "training.epochs=50",
    "training.batch_size=128",
    "training.patience=8",
    "training.min_delta=0.001",
    "training.show_progress=false",
    "training.checkpoint_gate.metric=auto_direction",
    "training.checkpoint_gate.max=0.5",
]

# native_inject 固定口径：patch + 8Hz + N0 + conv2d。
# fuse_len / fusion_decoder 在 native_inject 下不参与，故不设置。
STFT_BASE = [
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
]

# 复用 E1-D 同款 6 个代表性 seed。
SEEDS = [20260700, 20260710, 20260837, 20260901, 20260911, 20260920]

ARMS = ["time_only", "dual"]

# time_only / dual 分开 run root，便于 peak_band_misclass_rate.py 按臂回连配对。
RUN_ROOT = {
    "time_only": "runs/b2_native_time_only",
    "dual": "runs/b2_native_dual_8hz",
}


def build_run_specs() -> list[dict]:
    """生成 B2-0 全部 run 规格：time_only / dual × SEEDS，全部 native_inject。"""
    specs: list[dict] = []
    for branch_mode in ARMS:
        for seed in SEEDS:
            specs.append(
                {
                    "branch_mode": branch_mode,
                    "seed": seed,
                    "overrides": [
                        *STFT_BASE,
                        f"model.branch_mode={branch_mode}",
                        f"outputs.run_root={RUN_ROOT[branch_mode]}",
                        f"training.seed={seed}",
                    ],
                }
            )
    return specs


def _tag(spec: dict) -> str:
    return f"b2_native_{spec['branch_mode']}_{spec['seed']}"


def manifest_row(spec: dict) -> dict:
    return {
        "tag": _tag(spec),
        "branch_mode": spec["branch_mode"],
        "seed": spec["seed"],
        "overrides": " ".join(spec["overrides"]),
    }


def _command_for_spec(spec: dict, device: str) -> list[str]:
    cmd = [sys.executable, "scripts/train_tho_small.py", "--config", "configs/tho_research_v2.yaml"]
    for override in [*COMMON_OVERRIDES, f"training.device={device}", *spec["overrides"]]:
        cmd.extend(["--set", override])
    return cmd


def _run_one(spec: dict, device: str) -> str:
    tag = _tag(spec)
    print(f"start {tag} device={device}", flush=True)
    subprocess.run(_command_for_spec(spec, device), check=True)
    print(f"done {tag}", flush=True)
    return tag


def _resolve_devices(devices: list[str] | None) -> list[str]:
    """显式传入设备时不混入默认 cuda:0，避免多卡调度偏置。"""

    return devices or ["cuda:0"]


def _assign_devices(specs: list[dict], devices: list[str]) -> list[tuple[dict, str]]:
    """按 run 顺序轮转分配设备；并发数由 --max-parallel 独立控制。"""

    return [(spec, devices[idx % len(devices)]) for idx, spec in enumerate(specs)]


def main() -> None:
    parser = argparse.ArgumentParser(description="B2-0 native_inject 原生解码融合批次编排（time_only/dual × 6 seed）")
    parser.add_argument("--skip", action="append", default=[], help="跳过的 run tag")
    parser.add_argument("--dry-run", action="store_true", help="只打印将运行的 tag，不实际训练")
    parser.add_argument("--device", action="append", default=None, help="训练设备，可重复传入；默认 cuda:0")
    parser.add_argument("--max-parallel", type=int, default=1, help="并发训练进程数；默认 1")
    parser.add_argument("--manifest", default="runs/b2_native_decode_manifest.csv")
    args = parser.parse_args()
    skipped = set(args.skip)

    specs = build_run_specs()

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["tag", "branch_mode", "seed", "overrides"])
        writer.writeheader()
        writer.writerows(manifest_row(spec) for spec in specs)

    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")

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
    assignments = _assign_devices(runnable, devices)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, spec, device) for spec, device in assignments]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
