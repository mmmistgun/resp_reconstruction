from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from scripts.batch_utils import assign_devices, build_launch_plan, resolve_devices, run_command_with_delay
    from scripts.run_f_a_stft_loss_probe import COMMON_OVERRIDES, PATCH_NATIVE_BASE, SEEDS, _slug
    from scripts.run_f_d_highfreq_probe import (
        DEFAULT_CWT_CACHE,
        DEFAULT_MODULATION_CACHE,
        F_D_COMMON_LOSS_OVERRIDES,
    )
except ModuleNotFoundError:
    from batch_utils import assign_devices, build_launch_plan, resolve_devices, run_command_with_delay
    from run_f_a_stft_loss_probe import COMMON_OVERRIDES, PATCH_NATIVE_BASE, SEEDS, _slug
    from run_f_d_highfreq_probe import DEFAULT_CWT_CACHE, DEFAULT_MODULATION_CACHE, F_D_COMMON_LOSS_OVERRIDES


ARMS = [
    {
        "label": "F0_native_stft_pre_mixer",
        "representation": "low_stft_anchor",
        "feature_extractor": "conv2d",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "run_root": "runs/f_a_stft_loss/f0_native_stft_pre_mixer/dual",
        "cache_kind": "",
        "cache_path": "",
        "train": False,
        "extra_overrides": [],
    },
    {
        "label": "F-D0_high_stft_anchor",
        "representation": "high_stft",
        "feature_extractor": "conv2d",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "run_root": "runs/f_d_highfreq/f_d0_high_stft_anchor/dual",
        "cache_kind": "",
        "cache_path": "",
        "train": False,
        "extra_overrides": [
            "model.stft_encoder_type=conv2d",
            "model.stft_win=800",
            "model.stft_hop=100",
            "model.stft_low_hz=1.0",
            "model.stft_high_hz=8.0",
        ],
    },
    {
        "label": "F-D1_high_cwt",
        "representation": "high_cwt",
        "feature_extractor": "cached_tf",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "run_root": "runs/f_d_highfreq/f_d1_high_cwt/dual",
        "cache_kind": "cwt",
        "cache_path": DEFAULT_CWT_CACHE,
        "train": False,
        "extra_overrides": [
            "model.stft_encoder_type=cached_tf",
            "model.stft_cached_in_freq=36",
        ],
    },
    {
        "label": "F-D1b_high_cwt_cnn_tcn",
        "representation": "high_cwt",
        "feature_extractor": "cached_tf_tcn",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "run_root": "runs/f_d_feature_extractor/f_d1b_high_cwt_cnn_tcn/dual",
        "cache_kind": "cwt",
        "cache_path": DEFAULT_CWT_CACHE,
        "train": True,
        "extra_overrides": [
            "model.stft_encoder_type=cached_tf_tcn",
            "model.stft_cached_in_freq=36",
            "model.stft_cached_hidden_channels=32",
            "model.stft_cached_pooled_freq=6",
        ],
    },
    {
        "label": "F-D2_high_cwt_modulation",
        "representation": "high_cwt_modulation",
        "feature_extractor": "cached_sequence",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "run_root": "runs/f_d_highfreq/f_d2_high_cwt_modulation/dual",
        "cache_kind": "modulation",
        "cache_path": DEFAULT_MODULATION_CACHE,
        "train": False,
        "extra_overrides": [
            "model.stft_encoder_type=cached_sequence",
            "model.stft_cached_in_freq=8",
        ],
    },
    {
        "label": "F-D2b_high_cwt_modulation_res_tcn",
        "representation": "high_cwt_modulation",
        "feature_extractor": "cached_sequence_res_tcn",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "run_root": "runs/f_d_feature_extractor/f_d2b_high_cwt_modulation_res_tcn/dual",
        "cache_kind": "modulation",
        "cache_path": DEFAULT_MODULATION_CACHE,
        "train": True,
        "extra_overrides": [
            "model.stft_encoder_type=cached_sequence_res_tcn",
            "model.stft_cached_in_freq=8",
            "model.stft_cached_hidden_channels=32",
        ],
    },
]


def _tag(spec: dict) -> str:
    return f"{_slug(spec['label'])}_{spec['branch_mode']}_{spec['seed']}"


def _arm_with_cache_overrides(arm: dict, *, cwt_cache: str, modulation_cache: str) -> dict:
    arm = dict(arm)
    if arm["cache_kind"] == "cwt":
        arm["cache_path"] = cwt_cache
    elif arm["cache_kind"] == "modulation":
        arm["cache_path"] = modulation_cache
    return arm


def _base_spec(arm: dict, seed: int) -> dict:
    overrides = [
        *PATCH_NATIVE_BASE,
        *arm["extra_overrides"],
        f"model.branch_mode={arm['branch_mode']}",
        f"model.stft_inject_position={arm['stft_inject_position']}",
        *F_D_COMMON_LOSS_OVERRIDES,
        f"outputs.run_root={arm['run_root']}",
        f"training.seed={seed}",
    ]
    if arm["cache_path"]:
        overrides.append(f"data.sst_cache_path={arm['cache_path']}")
    return {**arm, "seed": seed, "overrides": overrides}


def build_run_specs(
    *,
    cwt_cache: str = DEFAULT_CWT_CACHE,
    modulation_cache: str = DEFAULT_MODULATION_CACHE,
) -> list[dict]:
    """生成 F-D feature extractor probe manifest；F0/F-D0/F-D1/F-D2 默认复用。"""

    specs = []
    for arm in ARMS:
        resolved_arm = _arm_with_cache_overrides(arm, cwt_cache=cwt_cache, modulation_cache=modulation_cache)
        specs.extend(_base_spec(resolved_arm, seed) for seed in SEEDS)
    return specs


def manifest_row(spec: dict) -> dict:
    return {
        "tag": _tag(spec),
        "label": spec["label"],
        "representation": spec["representation"],
        "feature_extractor": spec["feature_extractor"],
        "branch_mode": spec["branch_mode"],
        "seed": spec["seed"],
        "stft_inject_position": spec["stft_inject_position"],
        "paired_f0_label": spec["paired_f0_label"],
        "paired_time_only_label": spec["paired_time_only_label"],
        "cache_path": spec["cache_path"],
        "train": str(bool(spec["train"])).lower(),
        "overrides": " ".join(spec["overrides"]),
    }


def _command_for_spec(spec: dict, device: str) -> list[str]:
    cmd = [sys.executable, "scripts/train_tho_small.py", "--config", "configs/tho_research_v2.yaml"]
    for override in [*COMMON_OVERRIDES, f"training.device={device}", *spec["overrides"]]:
        cmd.extend(["--set", override])
    return cmd


def _run_one(spec: dict, device: str, launch_delay_sec: float = 0.0) -> str:
    return run_command_with_delay(_tag(spec), _command_for_spec(spec, device), device, launch_delay_sec)


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
    parser = argparse.ArgumentParser(description="F-D feature extractor probe 编排")
    parser.add_argument("--cwt-cache", default=DEFAULT_CWT_CACHE, help="F-D1/F-D1b high-CWT 缓存 npz 路径")
    parser.add_argument("--modulation-cache", default=DEFAULT_MODULATION_CACHE, help="F-D2/F-D2b modulation 缓存 npz 路径")
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
    parser.add_argument("--manifest", default="runs/f_d_feature_extractor_manifest.csv")
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")
    if args.start_stagger_sec < 0:
        raise SystemExit("--start-stagger-sec 必须 >= 0")

    specs = build_run_specs(cwt_cache=args.cwt_cache, modulation_cache=args.modulation_cache)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "tag",
                "label",
                "representation",
                "feature_extractor",
                "branch_mode",
                "seed",
                "stft_inject_position",
                "paired_f0_label",
                "paired_time_only_label",
                "cache_path",
                "train",
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
        if not spec["train"]:
            print(f"reuse {tag}", flush=True)
            continue
        if args.dry_run:
            print(f"plan {tag}", flush=True)
            continue
        runnable.append(spec)

    if args.dry_run or not runnable:
        print(f"manifest={manifest_path} runs={len(specs)} runnable={len(runnable)}", flush=True)
        return

    devices = resolve_devices(args.device)
    plan = _build_launch_plan(runnable, devices, args.max_parallel, args.start_stagger_sec)
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(_run_one, spec, device, delay): _tag(spec)
            for spec, device, delay in plan
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
