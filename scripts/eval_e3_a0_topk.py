from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalSpec:
    run_dir: Path
    arm_slug: str
    branch_mode: str
    run_id: str
    rank: int
    checkpoint_path: Path
    config_path: Path
    metrics_output: Path

    @property
    def tag(self) -> str:
        return f"{self.arm_slug}_{self.branch_mode}_{self.run_id}_top{self.rank}"


def discover_eval_specs(runs_root: Path, *, top_k: int = 3, force: bool = False) -> list[EvalSpec]:
    """发现 E3-A0 run 目录中的 topK checkpoint 评价任务。"""

    if top_k < 1:
        raise ValueError("top_k 必须 >= 1")
    if not runs_root.exists():
        raise FileNotFoundError(f"runs 根目录不存在: {runs_root}")

    specs: list[EvalSpec] = []
    for run_dir in sorted(path for path in runs_root.glob("*/*/*") if path.is_dir()):
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue
        arm_slug = run_dir.parents[1].name
        branch_mode = run_dir.parent.name
        run_id = run_dir.name
        for rank in range(1, top_k + 1):
            checkpoint_path = run_dir / f"checkpoint_top{rank}.pt"
            metrics_output = run_dir / f"metrics_top{rank}.csv"
            if not checkpoint_path.exists():
                continue
            if metrics_output.exists() and not force:
                continue
            specs.append(
                EvalSpec(
                    run_dir=run_dir,
                    arm_slug=arm_slug,
                    branch_mode=branch_mode,
                    run_id=run_id,
                    rank=rank,
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    metrics_output=metrics_output,
                )
            )
    return specs


def resolve_devices(devices: list[str] | None) -> list[str]:
    """显式传入设备时不混入默认 cuda:0，避免多卡调度偏置。"""

    return devices or ["cuda:0"]


def assign_devices(specs: list[EvalSpec], devices: list[str]) -> list[tuple[EvalSpec, str]]:
    """按任务顺序轮转分配设备；并发数由 --max-parallel 独立控制。"""

    return [(spec, devices[idx % len(devices)]) for idx, spec in enumerate(specs)]


def command_for_spec(spec: EvalSpec, device: str, *, python: str = sys.executable) -> list[str]:
    """生成单个 checkpoint 评价命令。"""

    return [
        python,
        "scripts/eval_tho_small.py",
        "--checkpoint",
        str(spec.checkpoint_path),
        "--config",
        str(spec.config_path),
        "--metrics-output",
        str(spec.metrics_output),
        "--set",
        f"training.device={device}",
    ]


def manifest_row(spec: EvalSpec, device: str) -> dict[str, str | int]:
    return {
        "tag": spec.tag,
        "arm_slug": spec.arm_slug,
        "branch_mode": spec.branch_mode,
        "run_id": spec.run_id,
        "rank": spec.rank,
        "device": device,
        "checkpoint": str(spec.checkpoint_path),
        "config": str(spec.config_path),
        "metrics_output": str(spec.metrics_output),
    }


def _run_one(spec: EvalSpec, device: str) -> str:
    print(f"start {spec.tag} device={device}", flush=True)
    subprocess.run(command_for_spec(spec, device), check=True)
    print(f"done {spec.tag}", flush=True)
    return spec.tag


def _write_manifest(path: Path, assignments: list[tuple[EvalSpec, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "tag",
                "arm_slug",
                "branch_mode",
                "run_id",
                "rank",
                "device",
                "checkpoint",
                "config",
                "metrics_output",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_row(spec, device) for spec, device in assignments)


def main() -> None:
    parser = argparse.ArgumentParser(description="临时脚本：重评 E3-A0 每个 run 的 checkpoint_top1/2/3")
    parser.add_argument("--runs-root", default="runs/e3_a0", help="E3-A0 runs 根目录")
    parser.add_argument("--top-k", type=int, default=3, help="重评 checkpoint_topN 的 N 上限，默认 3")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的 metrics_topN.csv")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行评价")
    parser.add_argument("--device", action="append", default=None, help="评价设备，可重复传入；默认 cuda:0")
    parser.add_argument("--max-parallel", type=int, default=1, help="并发评价进程数；默认 1")
    parser.add_argument("--manifest", default="runs/e3_a0_topk_eval_manifest.csv", help="评价任务 manifest 输出路径")
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")

    specs = discover_eval_specs(Path(args.runs_root), top_k=args.top_k, force=args.force)
    assignments = assign_devices(specs, resolve_devices(args.device))
    _write_manifest(Path(args.manifest), assignments)

    if not assignments:
        print("no pending topK eval tasks", flush=True)
        return

    if args.dry_run:
        for spec, device in assignments:
            print(f"plan {spec.tag} device={device}", flush=True)
        return

    workers = min(args.max_parallel, len(assignments))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, spec, device) for spec, device in assignments]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
