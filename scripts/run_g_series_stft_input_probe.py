from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from scripts.batch_utils import assign_devices, build_launch_plan, resolve_devices, run_command_with_delay
    from scripts.run_f_a_stft_loss_probe import COMMON_OVERRIDES, PATCH_NATIVE_BASE, _slug
except ModuleNotFoundError:
    from batch_utils import assign_devices, build_launch_plan, resolve_devices, run_command_with_delay
    from run_f_a_stft_loss_probe import COMMON_OVERRIDES, PATCH_NATIVE_BASE, _slug


PILOT_SEEDS = [20260700, 20260837]
NATIVE_TOKEN_COUNT = 140
WINDOW_SAMPLES = 18000

G_COMMON_LOSS_OVERRIDES = [
    "loss.stft_dist_weight=0.0",
    "loss.stft_band_energy_weight=0.0",
    "loss.stft_peak_anchor_weight=0.0",
    "loss.log_component_grad_norms=false",
]

G0_ARMS = [
    {
        "label": "G0_time_only",
        "stage": "G0",
        "branch_mode": "time_only",
        "stft_inject_position": "post_mixer",
        "stft_win": None,
        "stft_hop": None,
        "stft_low_hz": None,
        "stft_high_hz": None,
        "stft_encoder_type": "",
        "paired_anchor_label": "G0_time_only",
        "paired_f0_label": "G0_time_only",
        "paired_time_only_label": "G0_time_only",
    },
    {
        "label": "G0_f0_native_stft_pre_mixer",
        "stage": "G0",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "stft_win": 3000,
        "stft_hop": 500,
        "stft_low_hz": 0.05,
        "stft_high_hz": 8.0,
        "stft_encoder_type": "conv2d",
        "paired_anchor_label": "G0_f0_native_stft_pre_mixer",
        "paired_f0_label": "G0_f0_native_stft_pre_mixer",
        "paired_time_only_label": "G0_time_only",
    },
]

G1_ARMS = [
    ("G1A_wide", 3000, 500, 0.05, 8.0),
    ("G1B_wide", 3000, 250, 0.05, 8.0),
    ("G1C_wide", 2000, 250, 0.05, 8.0),
    ("G1D_wide", 1500, 128, 0.05, 8.0),
    ("G1A_resp_mid", 3000, 500, 0.05, 3.0),
    ("G1B_resp_mid", 3000, 250, 0.05, 3.0),
    ("G1C_resp_mid", 2000, 250, 0.05, 3.0),
    ("G1D_resp_mid", 1500, 128, 0.05, 3.0),
]


def _g1_arm(label: str, stft_win: int, stft_hop: int, low_hz: float, high_hz: float) -> dict:
    return {
        "label": label,
        "stage": "G1",
        "branch_mode": "dual",
        "stft_inject_position": "pre_mixer",
        "stft_win": stft_win,
        "stft_hop": stft_hop,
        "stft_low_hz": low_hz,
        "stft_high_hz": high_hz,
        "stft_encoder_type": "conv2d",
        "paired_anchor_label": "G1A_wide",
        "paired_f0_label": "G1A_wide",
        "paired_time_only_label": "G0_time_only",
    }


def _arms_for_stage(stage: str) -> list[dict]:
    if stage == "g0":
        return [dict(arm) for arm in G0_ARMS]
    if stage == "g1":
        return [_g1_arm(*arm) for arm in G1_ARMS]
    if stage == "g0-g1":
        return [dict(arm) for arm in G0_ARMS] + [_g1_arm(*arm) for arm in G1_ARMS]
    raise ValueError(f"未知 stage={stage!r}")


def _run_root(label: str, branch_mode: str) -> str:
    return f"runs/g_series_stft_input/{_slug(label)}/{branch_mode}"


def _expected_stft_frames(stft_hop: int | None) -> int | None:
    if stft_hop is None:
        return None
    return WINDOW_SAMPLES // int(stft_hop) + 1


def _token_interp_ratio(stft_hop: int | None) -> float | None:
    frames = _expected_stft_frames(stft_hop)
    if frames is None:
        return None
    return NATIVE_TOKEN_COUNT / float(frames)


def _base_spec(arm: dict, seed: int) -> dict:
    overrides = [
        *PATCH_NATIVE_BASE,
        f"model.branch_mode={arm['branch_mode']}",
        f"model.stft_inject_position={arm['stft_inject_position']}",
        *G_COMMON_LOSS_OVERRIDES,
        f"outputs.run_root={_run_root(arm['label'], arm['branch_mode'])}",
        f"training.seed={seed}",
    ]
    if arm["stft_win"] is not None:
        overrides.extend(
            [
                f"model.stft_win={arm['stft_win']}",
                f"model.stft_hop={arm['stft_hop']}",
                f"model.stft_low_hz={arm['stft_low_hz']}",
                f"model.stft_high_hz={arm['stft_high_hz']}",
                f"model.stft_encoder_type={arm['stft_encoder_type']}",
            ]
        )
    return {**arm, "seed": seed, "overrides": overrides}


def build_run_specs(*, stage: str = "g0-g1", seeds: list[int] | None = None) -> list[dict]:
    """生成 G0/G1 STFT 输入分辨率 pilot；只定义矩阵，不隐式启动训练。"""

    selected_seeds = seeds or PILOT_SEEDS
    specs: list[dict] = []
    for arm in _arms_for_stage(stage):
        specs.extend(_base_spec(arm, seed) for seed in selected_seeds)
    return specs


def _tag(spec: dict) -> str:
    return f"{_slug(spec['label'])}_{spec['branch_mode']}_{spec['seed']}"


def _manifest_value(value):
    return "" if value is None else value


def manifest_row(spec: dict) -> dict:
    frames = _expected_stft_frames(spec["stft_hop"])
    ratio = _token_interp_ratio(spec["stft_hop"])
    return {
        "tag": _tag(spec),
        "label": spec["label"],
        "stage": spec["stage"],
        "branch_mode": spec["branch_mode"],
        "seed": spec["seed"],
        "stft_win": _manifest_value(spec["stft_win"]),
        "stft_hop": _manifest_value(spec["stft_hop"]),
        "stft_low_hz": _manifest_value(spec["stft_low_hz"]),
        "stft_high_hz": _manifest_value(spec["stft_high_hz"]),
        "stft_encoder_type": spec["stft_encoder_type"],
        "paired_anchor_label": spec["paired_anchor_label"],
        "paired_f0_label": spec["paired_f0_label"],
        "paired_time_only_label": spec["paired_time_only_label"],
        "expected_stft_frames": _manifest_value(frames),
        "token_interp_ratio": _manifest_value(ratio),
        "overrides": " ".join(spec["overrides"]),
    }


def _command_for_spec(spec: dict, device: str) -> list[str]:
    cmd = [sys.executable, "scripts/train_tho_small.py", "--config", "configs/tho_research_v2.yaml"]
    for override in [*COMMON_OVERRIDES, f"training.device={device}", *spec["overrides"]]:
        cmd.extend(["--set", override])
    return cmd


def _run_one(spec: dict, device: str, launch_delay_sec: float = 0.0) -> str:
    return run_command_with_delay(_tag(spec), _command_for_spec(spec, device), device, launch_delay_sec)


def _build_launch_plan(
    specs: list[dict],
    devices: list[str],
    max_parallel: int,
    start_stagger_sec: float,
) -> list[tuple[dict, str, float]]:
    return build_launch_plan(specs, devices, max_parallel, start_stagger_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="G 系列 STFT 输入时间分辨率与频带范围 pilot 编排")
    parser.add_argument("--stage", choices=["g0", "g1", "g0-g1"], default="g0-g1", help="生成/运行的实验阶段")
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="覆盖默认 pilot seed，可重复传入；未传时使用默认两 seed",
    )
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
    parser.add_argument("--manifest", default="runs/g_series_stft_input_manifest.csv")
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")
    if args.start_stagger_sec < 0:
        raise SystemExit("--start-stagger-sec 必须 >= 0")

    specs = build_run_specs(stage=args.stage, seeds=args.seed)
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
        print(f"manifest={manifest_path} runs={len(specs)} runnable={len(runnable)}", flush=True)
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
