from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "docs/experiments/crd_v1_candidate_lock_20260809.json"
ASSOCIATED_ARTIFACTS = {
    "resolved_config_sha256": "config.yaml",
    "run_manifest_sha256": "run_manifest.json",
    "validation_metrics_summary_sha256": "metrics_summary.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 CRD candidate-lock checkpoint 与关联产物")
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    args = parser.parse_args()

    lock_path = Path(args.lock)
    if not lock_path.is_absolute():
        lock_path = REPO_ROOT / lock_path
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked = 0

    for variant in payload["variants"]:
        for item in variant["checkpoints"]:
            checkpoint = REPO_ROOT / item["checkpoint_path"]
            artifacts = {
                "checkpoint_sha256": checkpoint,
                **{key: checkpoint.parent / filename for key, filename in ASSOCIATED_ARTIFACTS.items()},
            }
            for hash_key, path in artifacts.items():
                if not path.exists():
                    failures.append(f"缺失: {path}")
                    continue
                actual = _sha256(path)
                if actual != item[hash_key]:
                    failures.append(f"SHA-256 不一致: {path}: {actual} != {item[hash_key]}")
            if checkpoint.exists() and checkpoint.stat().st_size != int(item["checkpoint_size_bytes"]):
                failures.append(
                    f"大小不一致: {checkpoint}: {checkpoint.stat().st_size} != {item['checkpoint_size_bytes']}"
                )
            checked += 1

    if failures:
        raise SystemExit("candidate lock 验证失败:\n" + "\n".join(failures))
    print(f"candidate lock 验证通过: {checked} checkpoints, {checked * 4} artifacts")


if __name__ == "__main__":
    main()
