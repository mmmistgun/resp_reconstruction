from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from scripts.batch_utils import assign_devices, build_launch_plan, resolve_devices, run_command_with_delay
    from scripts.run_f_a_stft_loss_probe import COMMON_OVERRIDES, PATCH_NATIVE_BASE, SEEDS, _slug
    from scripts.run_f_b_aux_probe import FB_AUX_LOSS_OVERRIDES, FB_AUX_MODEL_OVERRIDES
except ModuleNotFoundError:
    from batch_utils import assign_devices, build_launch_plan, resolve_devices, run_command_with_delay
    from run_f_a_stft_loss_probe import COMMON_OVERRIDES, PATCH_NATIVE_BASE, SEEDS, _slug
    from run_f_b_aux_probe import FB_AUX_LOSS_OVERRIDES, FB_AUX_MODEL_OVERRIDES


FB_ENC2_AUX_MODEL_OVERRIDES = [
    "model.fb_aux_head=enc2_band_aware_aux",
    "model.fb_aux_stft_win_length=3000",
    "model.fb_aux_stft_hop_length=500",
    "model.fb_aux_stft_n_fft=3000",
    "model.fb_aux_stft_center=false",
    "model.fb_aux_stft_low_hz=0.033",
    "model.fb_aux_stft_high_hz=3.0",
]

F_B1_LOSS = {
    "fb_aux_weight": 0.01,
    "fb_consistency_weight": 0.005,
    "fb_consistency_start_epoch": 7,
}

ARMS = [
    {
        "label": "F0_native_stft_pre_mixer",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "fb_aux_head": "none",
        "run_root": "runs/f_a_stft_loss/f0_native_stft_pre_mixer/dual",
        "train": False,
        **{"fb_aux_weight": 0.0, "fb_consistency_weight": 0.0, "fb_consistency_start_epoch": 1},
    },
    {
        "label": "F-B1_aux_consistency_detach",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "fb_aux_head": "enc1_min_aux",
        "run_root": "runs/f_b_aux/f_b1_aux_consistency_detach/dual",
        "train": False,
        **F_B1_LOSS,
    },
    {
        "label": "F-B1b_aux_enc2_band_aware_consistency",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "paired_f0_label": "F0_native_stft_pre_mixer",
        "paired_time_only_label": "F0_native_time_only",
        "fb_aux_head": "enc2_band_aware_aux",
        "run_root": "runs/f_b_feature_extractor/f_b1b_aux_enc2_band_aware_consistency/dual",
        "train": True,
        **F_B1_LOSS,
    },
]


def _tag(spec: dict) -> str:
    return f"{_slug(spec['label'])}_{spec['branch_mode']}_{spec['seed']}"


def _base_spec(arm: dict, seed: int) -> dict:
    overrides = [
        *PATCH_NATIVE_BASE,
        f"model.branch_mode={arm['branch_mode']}",
        f"model.stft_inject_position={arm['stft_inject_position']}",
        f"loss.fb_aux_weight={arm['fb_aux_weight']}",
        f"loss.fb_consistency_weight={arm['fb_consistency_weight']}",
        f"loss.fb_consistency_start_epoch={arm['fb_consistency_start_epoch']}",
        f"outputs.run_root={arm['run_root']}",
        f"training.seed={seed}",
    ]
    if arm["fb_aux_head"] == "enc1_min_aux":
        overrides.extend([*FB_AUX_MODEL_OVERRIDES, *FB_AUX_LOSS_OVERRIDES, "loss.log_component_grad_norms=true"])
    elif arm["fb_aux_head"] == "enc2_band_aware_aux":
        overrides.extend([*FB_ENC2_AUX_MODEL_OVERRIDES, *FB_AUX_LOSS_OVERRIDES, "loss.log_component_grad_norms=true"])
    return {**arm, "seed": seed, "overrides": overrides}


def build_run_specs() -> list[dict]:
    """生成 F-B 3.5 feature extractor probe manifest；F0 与 Enc1 F-B1 默认复用。"""
    return [_base_spec(arm, seed) for arm in ARMS for seed in SEEDS]


def manifest_row(spec: dict) -> dict:
    return {
        "tag": _tag(spec),
        "label": spec["label"],
        "branch_mode": spec["branch_mode"],
        "seed": spec["seed"],
        "stft_inject_position": spec["stft_inject_position"],
        "paired_f0_label": spec["paired_f0_label"],
        "paired_time_only_label": spec["paired_time_only_label"],
        "fb_aux_head": spec["fb_aux_head"],
        "fb_aux_weight": spec["fb_aux_weight"],
        "fb_consistency_weight": spec["fb_consistency_weight"],
        "fb_consistency_start_epoch": spec["fb_consistency_start_epoch"],
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
    parser = argparse.ArgumentParser(description="F-B 3.5 feature extractor probe 编排")
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
    parser.add_argument("--manifest", default="runs/f_b_feature_extractor_manifest.csv")
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
                "fb_aux_head",
                "fb_aux_weight",
                "fb_consistency_weight",
                "fb_consistency_start_epoch",
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
