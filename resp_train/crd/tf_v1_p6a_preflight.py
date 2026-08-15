from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
P6A_ENGINEERING_COMMIT = "38024237059a4c2f2a22b502415502427dc1ada2"
P6A_AUDIT_COMMIT = "5829d63af9e91ead64dccca2e8ab3b0d0e947504"
P6A_ACCEPTANCE_RECEIPT = REPO_ROOT / (
    "runs/crd_tf_v1/p6a_acceptance_audit/"
    "9304ae7abd2056c9c28b09702d8fcfc88672a4b53ea412e2922d8d7c7ee821b1/"
    "p6a_acceptance.json"
)
P6A_ACCEPTANCE_RECEIPT_SHA256 = "a23c1dd9aca724ecae3d867429911043a0da1843ede61a3784578abbf99040a9"
P6A_CRITICAL_PATHS = (
    "configs/crd_tf_v1/crd_tf401_mws_add_smoke.yaml",
    "configs/crd_tf_v1/crd_tf401_mws_add_formal.yaml",
    "configs/crd_tf_v1/crd_tf402_mws_gate_formal.yaml",
    "configs/crd_tf_v1/crd_tf403_ctrl_gate_formal.yaml",
    "resp_train/crd/config.py",
    "resp_train/crd/experiment.py",
    "resp_train/crd/model.py",
    "resp_train/crd/tf_v1_data.py",
    "resp_train/crd/tf_v1_features.py",
    "resp_train/crd/tf_v1_model.py",
    "resp_train/crd/training.py",
    "resp_train/data/research_v2.py",
    "resp_train/engine/train.py",
)


def validate_p6a_formal_preflight() -> str:
    """验证冻结 acceptance、干净工作树与工程验收后的关键实现 identity。"""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("CRD-TF P6a formal 必须从干净 Git commit 启动")
    if _sha256_file(P6A_ACCEPTANCE_RECEIPT) != P6A_ACCEPTANCE_RECEIPT_SHA256:
        raise RuntimeError("CRD-TF P6a frozen acceptance receipt SHA-256 不一致")
    receipt: dict[str, Any] = json.loads(P6A_ACCEPTANCE_RECEIPT.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "passed"
        or receipt.get("complete") is not True
        or receipt.get("batch_decision") != "128x1"
        or receipt.get("engineering_commit") != P6A_ENGINEERING_COMMIT
        or receipt.get("audit_commit") != P6A_AUDIT_COMMIT
        or receipt.get("research_test_used") is not False
        or float(receipt.get("maximum_peak_reserved_fraction", 1.0)) > 0.80
    ):
        raise RuntimeError("CRD-TF P6a frozen acceptance receipt identity 不合格")
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", P6A_ENGINEERING_COMMIT, "--", *P6A_CRITICAL_PATHS],
        cwd=REPO_ROOT,
        check=False,
    )
    if unchanged.returncode != 0:
        raise RuntimeError("P6a engineering acceptance 后模型/数据/训练关键实现发生变化，formal 不得启动")
    return commit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

