from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
P3_CODE_COMMIT = "56cabf1d37fa01104902b6aeaef3d256bee6b2a1"
P3_RECEIPT = REPO_ROOT / (
    "runs/crd_tf_v1/p3_acceptance_audit/"
    "0b4af9bd0c1c6441460702ad893cc5713f8ce3578cc055e51e86e60ad285234e/"
    "p3_acceptance.json"
)
P3_RECEIPT_SHA256 = "d68790e5db45ce65f60a17badab535196a913466176b7ac52a5cac4910f49f14"
P3_CRITICAL_PATHS = (
    "resp_train/crd/model.py",
    "resp_train/crd/training.py",
    "resp_train/crd/experiment.py",
    "resp_train/crd/tf_v1_data.py",
    "resp_train/crd/tf_v1_features.py",
    "resp_train/crd/tf_v1_model.py",
    "resp_train/data/research_v2.py",
    "resp_train/engine/train.py",
)


def validate_p4_formal_preflight() -> str:
    """每个独立 formal run 共用的最小 P3/clean-commit 闸门。"""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("CRD-TF P4 formal 必须从干净 Git commit 启动")
    if _sha256_file(P3_RECEIPT) != P3_RECEIPT_SHA256:
        raise RuntimeError("CRD-TF P3 frozen receipt SHA-256 不一致")
    receipt: dict[str, Any] = json.loads(P3_RECEIPT.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "passed"
        or receipt.get("complete") is not True
        or receipt.get("batch_decision") != "128x1"
        or receipt.get("git_commit") != P3_CODE_COMMIT
        or receipt.get("research_test_used") is not False
    ):
        raise RuntimeError("CRD-TF P3 frozen receipt identity 不合格")
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", P3_CODE_COMMIT, "--", *P3_CRITICAL_PATHS],
        cwd=REPO_ROOT,
        check=False,
    )
    if unchanged.returncode != 0:
        raise RuntimeError("P3 后模型/数据/训练关键实现发生变化，P4 不得启动")
    return commit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
