from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# E3-B：可学习频带前端验证。主问题是“STFT 频率表达是否需要可学习对齐/聚合”。
# B0 保留 A0.0 fullband concat 作为参考；B1/B2 只替换 STFT encoder，融合头和训练口径保持一致。
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
]

CONCAT_HEAD = [
    "model.fusion_mode=concat_generic",
    "model.fuse_len=600",
    "model.fusion_decoder=deep",
]

# 探索期只跑 3 个代表性 seed；若 B1/B2 出现正信号，再单独扩 seed 收口。
SEEDS = [20260700, 20260837, 20260901]

ARMS = [
    {
        "label": "E3-B0_concat_fullband_ref",
        "fusion_mode": "concat_generic",
        "stft_encoder_type": "conv2d",
    },
    {
        "label": "E3-B1_freq_mlp_fullband",
        "fusion_mode": "concat_generic",
        "stft_encoder_type": "freq_mlp",
    },
    {
        "label": "E3-B2_soft_band_concat",
        "fusion_mode": "concat_generic",
        "stft_encoder_type": "soft_band",
    },
]


def _slug(label: str) -> str:
    return label.lower().replace(".", "_").replace("-", "_")


def _run_root(label: str, branch_mode: str) -> str:
    return f"runs/e3_b/{_slug(label)}/{branch_mode}"


def build_run_specs() -> list[dict]:
    """生成 B0/B1/B2：每个 arm 都有 time_only/dual 同 seed 配对。"""

    specs: list[dict] = []
    for arm in ARMS:
        for branch_mode in ["time_only", "dual"]:
            for seed in SEEDS:
                encoder_type = arm["stft_encoder_type"]
                specs.append(
                    {
                        "label": arm["label"],
                        "branch_mode": branch_mode,
                        "seed": seed,
                        "fusion_mode": arm["fusion_mode"],
                        "stft_encoder_type": encoder_type,
                        "paired_time_only_label": arm["label"],
                        "overrides": [
                            *PATCH_STFT_BASE,
                            *CONCAT_HEAD,
                            f"model.stft_encoder_type={encoder_type}",
                            f"model.branch_mode={branch_mode}",
                            f"outputs.run_root={_run_root(arm['label'], branch_mode)}",
                            f"training.seed={seed}",
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
        "stft_encoder_type": spec["stft_encoder_type"],
        "paired_time_only_label": spec["paired_time_only_label"],
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
    parser = argparse.ArgumentParser(description="E3-B 可学习频带前端探针编排")
    parser.add_argument("--skip", action="append", default=[], help="跳过的 run tag")
    parser.add_argument("--dry-run", action="store_true", help="只打印将运行的 tag，不实际训练")
    parser.add_argument("--device", action="append", default=None, help="训练设备，可重复传入；默认 cuda:0")
    parser.add_argument("--max-parallel", type=int, default=1, help="并发训练进程数；默认 1")
    parser.add_argument("--manifest", default="runs/e3_b_manifest.csv")
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")

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
    assignments = _assign_devices(runnable, devices)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, spec, device) for spec, device in assignments]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
