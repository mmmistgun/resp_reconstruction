from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("runs/test_eval_g_series_20260705")


@dataclass(frozen=True)
class TestEvalSpec:
    label: str
    seed: int
    checkpoint: Path

    @property
    def tag(self) -> str:
        return f"{self.label}_{self.seed}"


def load_specs(path: str | Path) -> list[TestEvalSpec]:
    spec_path = Path(path)
    if not spec_path.exists():
        raise FileNotFoundError(f"specs 文件不存在: {spec_path}")
    with spec_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        fieldnames = set(reader.fieldnames or [])
        required = {"label", "seed", "checkpoint"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"specs 缺少必需列: {missing}")
        specs = [
            TestEvalSpec(
                label=str(row["label"]),
                seed=int(row["seed"]),
                checkpoint=Path(str(row["checkpoint"])),
            )
            for row in reader
            if str(row.get("checkpoint", "")).strip()
        ]
    if not specs:
        raise ValueError(f"specs 中没有可运行任务: {spec_path}")
    return specs


def resolve_devices(devices: list[str] | None) -> list[str]:
    return devices or ["cuda:0", "cuda:1"]


def output_paths(spec: TestEvalSpec, output_dir: Path) -> tuple[Path, Path, Path]:
    stem = f"{spec.tag}_test"
    return (
        output_dir / f"{stem}_metrics.csv",
        output_dir / f"{stem}_summary.csv",
        output_dir / f"{stem}_manifest.csv",
    )


def outputs_complete(spec: TestEvalSpec, output_dir: Path) -> bool:
    return all(path.exists() for path in output_paths(spec, output_dir))


def pending_specs(specs: list[TestEvalSpec], *, output_dir: Path, force: bool) -> list[TestEvalSpec]:
    if force:
        return list(specs)
    return [spec for spec in specs if not outputs_complete(spec, output_dir)]


def assign_devices(specs: list[TestEvalSpec], devices: list[str]) -> list[tuple[TestEvalSpec, str]]:
    return [(spec, devices[idx % len(devices)]) for idx, spec in enumerate(specs)]


def build_launch_plan(
    assignments: list[tuple[TestEvalSpec, str]],
    *,
    max_parallel: int,
    start_stagger_sec: float,
) -> list[tuple[TestEvalSpec, str, float]]:
    if not assignments:
        return []
    workers = min(max(1, int(max_parallel)), len(assignments))
    stagger = max(0.0, float(start_stagger_sec))
    return [(spec, device, float(idx % workers) * stagger) for idx, (spec, device) in enumerate(assignments)]


def command_for_spec(
    spec: TestEvalSpec,
    device: str,
    *,
    output_dir: Path,
    python: str = sys.executable,
    metric_workers: int = 1,
) -> list[str]:
    metrics_output, summary_output, manifest_output = output_paths(spec, output_dir)
    command = [
        python,
        "scripts/eval_tho_test.py",
        "--checkpoint",
        str(spec.checkpoint),
        "--metrics-output",
        str(metrics_output),
        "--summary-output",
        str(summary_output),
        "--manifest-output",
        str(manifest_output),
        "--set",
        f"training.device={device}",
    ]
    if int(metric_workers) > 1:
        command.extend(["--set", f"evaluation.metric_workers={int(metric_workers)}"])
    return command


def manifest_row(spec: TestEvalSpec, device: str, output_dir: Path, launch_delay_sec: float) -> dict[str, str | int | float]:
    metrics_output, summary_output, manifest_output = output_paths(spec, output_dir)
    return {
        "tag": spec.tag,
        "label": spec.label,
        "seed": int(spec.seed),
        "device": device,
        "launch_delay_sec": float(launch_delay_sec),
        "checkpoint": str(spec.checkpoint),
        "metrics_output": str(metrics_output),
        "summary_output": str(summary_output),
        "manifest_output": str(manifest_output),
    }


def write_manifest(path: Path, launch_plan: list[tuple[TestEvalSpec, str, float]], output_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "tag",
                "label",
                "seed",
                "device",
                "launch_delay_sec",
                "checkpoint",
                "metrics_output",
                "summary_output",
                "manifest_output",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_row(spec, device, output_dir, delay) for spec, device, delay in launch_plan)


def run_one(
    spec: TestEvalSpec,
    device: str,
    *,
    output_dir: Path,
    metric_workers: int,
    launch_delay_sec: float,
) -> str:
    if float(launch_delay_sec) > 0.0:
        print(f"delay {spec.tag} device={device} sleep={float(launch_delay_sec):.1f}s", flush=True)
        time.sleep(float(launch_delay_sec))
    print(f"start {spec.tag} device={device}", flush=True)
    subprocess.run(
        command_for_spec(spec, device, output_dir=output_dir, metric_workers=metric_workers),
        check=True,
    )
    print(f"done {spec.tag}", flush=True)
    return spec.tag


def run_eval(
    launch_plan: list[tuple[TestEvalSpec, str, float]],
    *,
    output_dir: Path,
    max_parallel: int,
    metric_workers: int,
) -> None:
    if not launch_plan:
        print("no pending test eval tasks", flush=True)
        return
    workers = min(int(max_parallel), len(launch_plan))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_one,
                spec,
                device,
                output_dir=output_dir,
                metric_workers=int(metric_workers),
                launch_delay_sec=delay,
            )
            for spec, device, delay in launch_plan
        ]
        for future in as_completed(futures):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description="并发运行 G 系列代表 checkpoint 的 held-out test 评价")
    parser.add_argument("--specs", required=True, help="待评价 checkpoint 清单 CSV，必需列: label,seed,checkpoint")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="test 输出目录")
    parser.add_argument("--manifest", default="", help="调度 manifest；默认写入 output-dir/g_series_test_eval_manifest.csv")
    parser.add_argument("--device", action="append", default=None, help="评价设备，可重复传入；默认 cuda:0 和 cuda:1")
    parser.add_argument("--max-parallel", type=int, default=2, help="总并发 test 进程数；两卡各一个时设为 2")
    parser.add_argument("--metric-workers", type=int, default=4, help="每个 test 进程的指标线程数")
    parser.add_argument("--start-stagger-sec", type=float, default=0.0, help="按并发槽位错开启动秒数")
    parser.add_argument("--force", action="store_true", help="重新运行已存在完整输出的任务")
    parser.add_argument("--dry-run", action="store_true", help="只写调度 manifest 并打印计划")
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")
    if args.metric_workers < 1:
        raise SystemExit("--metric-workers 必须 >= 1")
    if args.start_stagger_sec < 0:
        raise SystemExit("--start-stagger-sec 必须 >= 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    devices = resolve_devices(args.device)
    specs = pending_specs(load_specs(args.specs), output_dir=output_dir, force=bool(args.force))
    assignments = assign_devices(specs, devices)
    launch_plan = build_launch_plan(
        assignments,
        max_parallel=int(args.max_parallel),
        start_stagger_sec=float(args.start_stagger_sec),
    )
    manifest = Path(args.manifest) if args.manifest else output_dir / "g_series_test_eval_manifest.csv"
    write_manifest(manifest, launch_plan, output_dir)

    if args.dry_run:
        for spec, device, delay in launch_plan:
            print(f"plan {spec.tag} device={device} delay={delay:.1f}s", flush=True)
        print(f"manifest: {manifest} rows={len(launch_plan)}", flush=True)
        return

    run_eval(
        launch_plan,
        output_dir=output_dir,
        max_parallel=int(args.max_parallel),
        metric_workers=int(args.metric_workers),
    )
    print(f"manifest: {manifest} rows={len(launch_plan)}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
