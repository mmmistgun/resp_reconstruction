from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from scripts.batch_utils import assign_devices, build_launch_plan, resolve_devices, run_command_with_delay
    from scripts.run_f_a_stft_loss_probe import COMMON_OVERRIDES, DIST_WEIGHT, PATCH_NATIVE_BASE, SEEDS, _slug
except ModuleNotFoundError:
    from batch_utils import assign_devices, build_launch_plan, resolve_devices, run_command_with_delay
    from run_f_a_stft_loss_probe import COMMON_OVERRIDES, DIST_WEIGHT, PATCH_NATIVE_BASE, SEEDS, _slug


ORIGINAL_BAND_ENERGY_WEIGHT = 0.01

ARMS = [
    {
        "label": "F0_native_stft_pre_mixer",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "stft_dist_weight": 0.0,
        "stft_band_energy_weight": 0.0,
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "reuse_existing": True,
        "run_root": "runs/f_a_stft_loss/f0_native_stft_pre_mixer/dual",
    },
    {
        "label": "F-A2_dist_bandE",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "stft_dist_weight": DIST_WEIGHT,
        "stft_band_energy_weight": ORIGINAL_BAND_ENERGY_WEIGHT,
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "reuse_existing": True,
        "run_root": "runs/f_a_stft_loss/f_a2_dist_bande/dual",
    },
    {
        "label": "F-A2b_dist_bandE_w005",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "stft_dist_weight": DIST_WEIGHT,
        "stft_band_energy_weight": 0.005,
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "reuse_existing": False,
    },
    {
        "label": "F-A2c_dist_bandE_w003",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "stft_dist_weight": DIST_WEIGHT,
        "stft_band_energy_weight": 0.003,
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "reuse_existing": False,
    },
]


def _run_root(label: str, branch_mode: str) -> str:
    return f"runs/f_a2_guard/{_slug(label)}/{branch_mode}"


def _base_spec(arm: dict, seed: int) -> dict:
    stft_loss_enabled = float(arm["stft_dist_weight"]) > 0 or float(arm["stft_band_energy_weight"]) > 0
    run_root = arm.get("run_root") or _run_root(arm["label"], arm["branch_mode"])
    return {
        **arm,
        "seed": seed,
        "run_root": run_root,
        "overrides": [
            *PATCH_NATIVE_BASE,
            f"model.branch_mode={arm['branch_mode']}",
            f"model.stft_inject_position={arm['stft_inject_position']}",
            f"loss.stft_dist_weight={arm['stft_dist_weight']}",
            f"loss.stft_band_energy_weight={arm['stft_band_energy_weight']}",
            f"loss.log_component_grad_norms={str(stft_loss_enabled).lower()}",
            f"outputs.run_root={run_root}",
            f"training.seed={seed}",
        ],
    }


def build_run_specs() -> list[dict]:
    """生成 F-A2 护栏 probe：复用旧 F0/F-A2，对比低 band-energy 权重候选。"""

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
        "reuse_existing": str(bool(spec["reuse_existing"])).lower(),
        "overrides": " ".join(spec["overrides"]),
    }


def _command_for_spec(spec: dict, device: str) -> list[str]:
    if spec["reuse_existing"]:
        raise ValueError(f"复用 run 不应重新训练: {_tag(spec)}")
    cmd = [sys.executable, "scripts/train_tho_small.py", "--config", "configs/tho_research_v2.yaml"]
    for override in [*COMMON_OVERRIDES, f"training.device={device}", *spec["overrides"]]:
        cmd.extend(["--set", override])
    return cmd


def _run_one(spec: dict, device: str, launch_delay_sec: float = 0.0) -> str:
    tag = _tag(spec)
    return run_command_with_delay(tag, _command_for_spec(spec, device), device, launch_delay_sec)


def _build_launch_plan(
    specs: list[dict],
    devices: list[str],
    max_parallel: int,
    start_stagger_sec: float,
) -> list[tuple[dict, str, float]]:
    return build_launch_plan(specs, devices, max_parallel, start_stagger_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="F-A2 guard probe 编排")
    parser.add_argument("--skip", action="append", default=[], help="跳过的 run tag")
    parser.add_argument("--dry-run", action="store_true", help="只打印将运行或复用的 tag，不实际训练")
    parser.add_argument("--device", action="append", default=None, help="训练设备，可重复传入；默认 cuda:0")
    parser.add_argument("--max-parallel", type=int, default=1, help="并发训练进程数；默认 1")
    parser.add_argument(
        "--start-stagger-sec",
        type=float,
        default=30.0,
        help="同一批并发 run 的槽位启动间隔秒数；0 表示不延迟，默认 30",
    )
    parser.add_argument("--manifest", default="runs/f_a2_guard_manifest.csv")
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
                "reuse_existing",
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
        if spec["reuse_existing"]:
            print(f"reuse {tag}", flush=True)
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
