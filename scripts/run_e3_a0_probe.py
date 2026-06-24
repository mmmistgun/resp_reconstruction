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

# E3-A0：同时拆分「融合对齐」和「频带前端」两条怀疑。
# 仅覆盖与 configs/tho_research_v2.yaml 默认值不同的项；极性 warmup 采用默认定稿口径。
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
]

CONCAT_HEAD = [
    "model.fusion_mode=concat_generic",
    "model.fuse_len=600",
    "model.fusion_decoder=deep",
]

# E3-A0 只是机制探针，默认少量代表性 seed：一个常规锚点 + 两个历史高风险 seed。
# 若出现明确方向，再单独扩 seed，避免探索期 GPU 冗余。
SEEDS = [20260700, 20260837, 20260901]

ARMS = [
    {
        "label": "E3-A0.0_concat_fullband",
        "fusion_mode": "concat_generic",
        "dual_encoder": "conv2d",
        "overrides": [*CONCAT_HEAD],
        "include_time_only": True,
        "paired_time_only_label": "E3-A0.0_concat_fullband",
    },
    {
        "label": "E3-A0.1_token_context_fullband",
        "fusion_mode": "token_context_inject",
        "dual_encoder": "conv2d",
        "overrides": ["model.fusion_mode=token_context_inject"],
        "include_time_only": True,
        "paired_time_only_label": "E3-A0.1_token_context_fullband",
    },
    {
        "label": "E3-A0.2_concat_bandgroup",
        "fusion_mode": "concat_generic",
        "dual_encoder": "bandgroup",
        "overrides": [*CONCAT_HEAD],
        "include_time_only": False,
        "paired_time_only_label": "E3-A0.0_concat_fullband",
    },
    {
        "label": "E3-A0.3_token_context_bandgroup",
        "fusion_mode": "token_context_inject",
        "dual_encoder": "bandgroup",
        "overrides": ["model.fusion_mode=token_context_inject"],
        "include_time_only": False,
        "paired_time_only_label": "E3-A0.1_token_context_fullband",
    },
]


def _slug(label: str) -> str:
    return label.lower().replace(".", "_").replace("-", "_")


def _run_root(label: str, branch_mode: str) -> str:
    return f"runs/e3_a0/{_slug(label)}/{branch_mode}"


def build_run_specs() -> list[dict]:
    """生成 E3-A0 候选：4 个 dual arm + 2 个去重 time_only substrate。"""

    specs: list[dict] = []
    for arm in ARMS:
        branch_modes = ["dual"]
        if arm["include_time_only"]:
            branch_modes.insert(0, "time_only")
        for branch_mode in branch_modes:
            # time_only 不使用 STFT 分支；固定 conv2d 只是保持 manifest 因子可读。
            # bandgroup arms 复用同 fusion substrate 的 fullband time_only。
            encoder = arm["dual_encoder"] if branch_mode == "dual" else "conv2d"
            for seed in SEEDS:
                specs.append(
                    {
                        "label": arm["label"],
                        "branch_mode": branch_mode,
                        "seed": seed,
                        "fusion_mode": arm["fusion_mode"],
                        "stft_encoder_type": encoder,
                        "paired_time_only_label": arm["paired_time_only_label"],
                        "overrides": [
                            *PATCH_STFT_BASE,
                            *arm["overrides"],
                            f"model.stft_encoder_type={encoder}",
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


def _run_one(spec: dict, device: str, launch_delay_sec: float = 0.0) -> str:
    tag = _tag(spec)
    return run_command_with_delay(tag, _command_for_spec(spec, device), device, launch_delay_sec)


def _resolve_devices(devices: list[str] | None) -> list[str]:
    """显式传入设备时不混入默认 cuda:0，避免多卡调度偏置。"""

    return resolve_devices(devices)


def _assign_devices(specs: list[dict], devices: list[str]) -> list[tuple[dict, str]]:
    """按 run 顺序轮转分配设备；并发数由 --max-parallel 独立控制。"""

    return assign_devices(specs, devices)


def _build_launch_plan(
    specs: list[dict],
    devices: list[str],
    max_parallel: int,
    start_stagger_sec: float,
) -> list[tuple[dict, str, float]]:
    return build_launch_plan(specs, devices, max_parallel, start_stagger_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="E3-A0 融合对齐 × 频带前端联合探针编排")
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
    parser.add_argument("--manifest", default="runs/e3_a0_manifest.csv")
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
            future.result()


if __name__ == "__main__":
    main()
