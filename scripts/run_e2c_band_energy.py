from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# E1-D 实验口径：仅覆盖与默认 yaml 不同的项；极性 warmup 已固化进
# configs/tho_research_v2.yaml 默认 loss 段，这里不再覆盖。
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

# native_inject 固定口径：patch + 8Hz + N0（与 B2-0 对齐，使全频带 sanity 可复现 B2-0 dual）。
# stft_encoder_type 是唯一变量：dual 臂用 bandenergy，sanity 用 conv2d（全频带图）。
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
    "model.branch_mode=dual",
    "model.fusion_mode=native_inject",
]

# E1-D 前 4 seed，保证与 runs/e1_d_warmup_time_only 同 seed 可配对（探针快速否决力度）。
SEEDS = [20260700, 20260710, 20260837, 20260901]

# sanity 用其中一个 seed 复现 B2-0 dual 全频带数，守住「复用旧 time_only 桩」前提。
SANITY_SEED = 20260700

DUAL_RUN_ROOT = "runs/e2c_band_energy_dual"
SANITY_RUN_ROOT = "runs/e2c_fullband_sanity"


def build_run_specs() -> list[dict]:
    """4 个 bandenergy dual + 1 个全频带 conv2d sanity。"""
    specs: list[dict] = []
    for seed in SEEDS:
        specs.append(
            {
                "kind": "dual",
                "seed": seed,
                "overrides": [
                    *STFT_BASE,
                    "model.stft_encoder_type=bandenergy",
                    f"outputs.run_root={DUAL_RUN_ROOT}",
                    f"training.seed={seed}",
                ],
            }
        )
    # 全频带 sanity：encoder_type=conv2d，应复现 B2-0 dual 同 seed（runs/b2_native_dual_8hz）。
    specs.append(
        {
            "kind": "sanity",
            "seed": SANITY_SEED,
            "overrides": [
                *STFT_BASE,
                "model.stft_encoder_type=conv2d",
                f"outputs.run_root={SANITY_RUN_ROOT}",
                f"training.seed={SANITY_SEED}",
            ],
        }
    )
    return specs


def _tag(spec: dict) -> str:
    return f"e2c_{spec['kind']}_{spec['seed']}"


def manifest_row(spec: dict) -> dict:
    return {
        "tag": _tag(spec),
        "kind": spec["kind"],
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
    parser = argparse.ArgumentParser(description="E2c 分频带能量探针编排（4 bandenergy dual + 1 全频带 sanity）")
    parser.add_argument("--skip", action="append", default=[], help="跳过的 run tag")
    parser.add_argument("--dry-run", action="store_true", help="只打印将运行的 tag，不实际训练")
    parser.add_argument("--device", action="append", default=None, help="训练设备，可重复传入；默认 cuda:0")
    parser.add_argument("--max-parallel", type=int, default=1, help="并发训练进程数；默认 1")
    parser.add_argument("--manifest", default="runs/e2c_band_energy_manifest.csv")
    args = parser.parse_args()
    skipped = set(args.skip)

    specs = build_run_specs()

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["tag", "kind", "seed", "overrides"])
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
