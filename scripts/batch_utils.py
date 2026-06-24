from __future__ import annotations

import subprocess
import time


DATALOADER_WORKER_OVERRIDES = [
    # 完整多 epoch 训练优先复用内存窗口；streaming 多 worker 只用于内存受限诊断。
    "data.preload_windows=true",
    "training.num_workers=0",
]


def resolve_devices(devices: list[str] | None) -> list[str]:
    """显式传入设备时不混入默认 cuda:0，避免多卡调度偏置。"""

    return devices or ["cuda:0"]


def assign_devices(specs: list[dict], devices: list[str]) -> list[tuple[dict, str]]:
    """按 run 顺序轮转分配设备；并发数由 --max-parallel 独立控制。"""

    return [(spec, devices[idx % len(devices)]) for idx, spec in enumerate(specs)]


def build_launch_plan(
    specs: list[dict],
    devices: list[str],
    max_parallel: int,
    start_stagger_sec: float,
) -> list[tuple[dict, str, float]]:
    """为每个并发槽位添加固定启动延迟，错开数据读取和 GPU 初始化峰值。"""

    if not specs:
        return []
    workers = min(max(1, int(max_parallel)), len(specs))
    stagger = max(0.0, float(start_stagger_sec))
    plan: list[tuple[dict, str, float]] = []
    for idx, (spec, device) in enumerate(assign_devices(specs, devices)):
        delay = float(idx % workers) * stagger
        plan.append((spec, device, delay))
    return plan


def run_command_with_delay(tag: str, command: list[str], device: str, launch_delay_sec: float = 0.0) -> str:
    """可选延迟后启动一个训练子进程，降低并发启动阶段的 I/O 峰值。"""

    if float(launch_delay_sec) > 0.0:
        print(f"delay {tag} device={device} sleep={float(launch_delay_sec):.1f}s", flush=True)
        time.sleep(float(launch_delay_sec))
    print(f"start {tag} device={device}", flush=True)
    subprocess.run(command, check=True)
    print(f"done {tag}", flush=True)
    return tag
