from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_LOCK = REPO_ROOT / "docs/experiments/crd_v1_candidate_lock_20260809.json"
ASSOCIATED_ARTIFACTS = {
    "resolved_config_sha256": "config.yaml",
    "run_manifest_sha256": "run_manifest.json",
    "validation_metrics_summary_sha256": "metrics_summary.csv",
}


class CandidateLockError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateLockVerification:
    lock_path: Path
    lock_sha256: str
    payload: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    matched_record: dict[str, Any] | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise CandidateLockError(f"candidate lock 路径越出仓库: {path}") from exc
    return resolved


def verify_candidate_lock(
    lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    *,
    checkpoint_path: str | Path | None = None,
) -> CandidateLockVerification:
    resolved_lock = resolve_repo_path(lock_path)
    payload = json.loads(resolved_lock.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise CandidateLockError("candidate lock schema_version 必须为 1")

    target = resolve_repo_path(checkpoint_path) if checkpoint_path is not None else None
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    matched: dict[str, Any] | None = None
    seen_paths: set[Path] = set()

    for variant in payload.get("variants", []):
        for item in variant.get("checkpoints", []):
            checkpoint = resolve_repo_path(item["checkpoint_path"])
            if checkpoint in seen_paths:
                failures.append(f"candidate lock checkpoint 重复: {checkpoint}")
            seen_paths.add(checkpoint)
            record = {
                **item,
                "role": variant["role"],
                "variant": variant["variant"],
                "training_commit": variant["training_commit"],
                "training_protocol": variant["training_protocol"],
            }
            records.append(record)
            if checkpoint == target:
                matched = record

            artifacts = {
                "checkpoint_sha256": checkpoint,
                **{key: checkpoint.parent / filename for key, filename in ASSOCIATED_ARTIFACTS.items()},
            }
            for hash_key, artifact in artifacts.items():
                if not artifact.exists():
                    failures.append(f"缺失: {artifact}")
                    continue
                actual = sha256_file(artifact)
                if actual != item[hash_key]:
                    failures.append(f"SHA-256 不一致: {artifact}: {actual} != {item[hash_key]}")
            if checkpoint.exists() and checkpoint.stat().st_size != int(item["checkpoint_size_bytes"]):
                failures.append(
                    f"大小不一致: {checkpoint}: {checkpoint.stat().st_size} != {item['checkpoint_size_bytes']}"
                )

    if len(records) != 12:
        failures.append(f"candidate lock 必须恰含 12 checkpoints，实际 {len(records)}")
    if target is not None and matched is None:
        failures.append(f"checkpoint 不在 candidate lock 中: {target}")
    if failures:
        raise CandidateLockError("candidate lock 验证失败:\n" + "\n".join(failures))
    return CandidateLockVerification(
        lock_path=resolved_lock,
        lock_sha256=sha256_file(resolved_lock),
        payload=payload,
        records=tuple(records),
        matched_record=matched,
    )
