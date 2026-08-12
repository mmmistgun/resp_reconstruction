from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from resp_train.crd.config import CRD_TF_PROTOCOL_VERSION, FORMAL_SEEDS, load_crd_config
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.tf_v1_model import TF_VARIANT_REPRESENTATIONS
from resp_train.crd.tf_v1_p3 import require_clean_git


REPO_ROOT = Path(__file__).resolve().parents[2]
P3_CODE_COMMIT = "56cabf1d37fa01104902b6aeaef3d256bee6b2a1"
P3_RECEIPT = REPO_ROOT / (
    "runs/crd_tf_v1/p3_acceptance_audit/"
    "0b4af9bd0c1c6441460702ad893cc5713f8ce3578cc055e51e86e60ad285234e/"
    "p3_acceptance.json"
)
P3_RECEIPT_SHA256 = "d68790e5db45ce65f60a17badab535196a913466176b7ac52a5cac4910f49f14"
BASE_CONFIG = REPO_ROOT / "configs/crd_tf_v1/crd_tf101_m_smoke.yaml"
FORMAL_OUTPUT_ROOT = REPO_ROOT / "runs/crd_tf_v1/formal"
MATRIX_STATE_ROOT = REPO_ROOT / "runs/crd_tf_v1/formal_matrix"
P4_DEVICE = "cuda:0"

P4_ARM_ORDER = (
    "crd_tf_ctrl1",
    "crd_tf101_m",
    "crd_tf102_w",
    "crd_tf103_l",
    "crd_tf104_s",
    "crd_tf_ctrl2",
    "crd_tf201_mw",
    "crd_tf202_ml",
    "crd_tf203_ms",
    "crd_tf204_wl",
    "crd_tf205_ws",
    "crd_tf206_ls",
    "crd_tf_ctrl3",
    "crd_tf301_mls",
    "crd_tf302_wls",
)

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


@dataclass(frozen=True)
class FormalSpec:
    index: int
    seed: int
    variant: str
    representations: tuple[str, ...]
    run_root: str

    @property
    def key(self) -> str:
        return f"seed_{self.seed}:{self.variant}"


def formal_plan() -> tuple[FormalSpec, ...]:
    specs: list[FormalSpec] = []
    for seed in FORMAL_SEEDS:
        for variant in P4_ARM_ORDER:
            specs.append(
                FormalSpec(
                    index=len(specs) + 1,
                    seed=int(seed),
                    variant=variant,
                    representations=TF_VARIANT_REPRESENTATIONS[variant],
                    run_root=str((FORMAL_OUTPUT_ROOT / variant / f"seed_{seed}").resolve()),
                )
            )
    if len(specs) != 45 or len({spec.key for spec in specs}) != 45:
        raise RuntimeError("CRD-TF P4 formal plan 必须严格包含 45 个唯一 run")
    return tuple(specs)


def formal_overrides(spec: FormalSpec) -> tuple[str, ...]:
    reps = ",".join(spec.representations)
    return (
        "protocol.run_role=formal",
        "protocol.execution_gate=p4_formal",
        f"model.variant={spec.variant}",
        f"model.tf_representations=[{reps}]",
        f"model.initialization_seed={spec.seed}",
        "data.max_train_windows=null",
        "data.max_val_windows=null",
        "data.max_test_windows=null",
        "training.epochs=80",
        "training.batch_size=128",
        "training.gradient_accumulation_steps=1",
        f"training.seed={spec.seed}",
        f"training.device={P4_DEVICE}",
        "training.show_progress=false",
        f"outputs.run_root={spec.run_root}",
    )


def preflight_p4() -> str:
    commit = require_clean_git()
    if _sha256_file(P3_RECEIPT) != P3_RECEIPT_SHA256:
        raise RuntimeError("CRD-TF P3 frozen receipt SHA-256 不一致")
    receipt = json.loads(P3_RECEIPT.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "passed"
        or receipt.get("complete") is not True
        or receipt.get("batch_decision") != "128x1"
        or receipt.get("git_commit") != P3_CODE_COMMIT
        or receipt.get("research_test_used") is not False
    ):
        raise RuntimeError("CRD-TF P3 frozen receipt identity 不合格")
    result = subprocess.run(
        ["git", "diff", "--quiet", P3_CODE_COMMIT, "--", *P3_CRITICAL_PATHS],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("P3 后模型/数据/训练关键实现发生变化，P4 不得启动")
    for spec in formal_plan():
        load_crd_config(BASE_CONFIG, overrides=formal_overrides(spec))
    return commit


def matrix_identity(commit: str) -> dict[str, Any]:
    return {
        "protocol": CRD_TF_PROTOCOL_VERSION,
        "phase": "p4_formal_matrix",
        "git_commit": commit,
        "p3_receipt_sha256": P3_RECEIPT_SHA256,
        "device": P4_DEVICE,
        "epochs": 80,
        "physical_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "early_stopping_enabled": False,
        "seeds": list(FORMAL_SEEDS),
        "arm_order": list(P4_ARM_ORDER),
        "plan": [asdict(spec) | {"key": spec.key} for spec in formal_plan()],
    }


def matrix_id(commit: str) -> str:
    return _sha256_json(matrix_identity(commit))


def current_state_path() -> Path:
    commit = preflight_p4()
    return MATRIX_STATE_ROOT / matrix_id(commit) / "matrix_state.json"


def state_summary(state: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in state["items"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    completed = [item for item in state["items"] if item["status"] == "completed"]
    return {
        "matrix_id": state["matrix_id"],
        "status": state["status"],
        "complete": state["complete"],
        "counts": counts,
        "active_key": state.get("active_key"),
        "last_completed_key": completed[-1]["key"] if completed else None,
        "last_completed_run_dir": completed[-1]["run_dir"] if completed else None,
    }


def run_formal_matrix(
    *,
    retry_failed: bool = False,
    retry_interrupted: bool = False,
) -> Path:
    commit = preflight_p4()
    identity = matrix_identity(commit)
    identifier = matrix_id(commit)
    matrix_dir = MATRIX_STATE_ROOT / identifier
    matrix_dir.mkdir(parents=True, exist_ok=True)
    state_path = matrix_dir / "matrix_state.json"
    lock_path = matrix_dir / "matrix.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("同一 CRD-TF P4 matrix 已有进程在运行") from exc
        state = _load_or_create_state(state_path, identity, identifier)
        _validate_restart_policy(state, retry_failed=retry_failed, retry_interrupted=retry_interrupted)
        _prepare_retries(state, retry_failed=retry_failed, retry_interrupted=retry_interrupted)
        state["status"] = "running"
        _save_state(state_path, state)
        by_key = {item["key"]: item for item in state["items"]}
        for spec in formal_plan():
            item = by_key[spec.key]
            if item["status"] == "completed":
                _verify_completed_run(Path(item["run_dir"]))
                print(f"P4 [{spec.index:02d}/45] skip completed {spec.key}", flush=True)
                continue
            attempt = {
                "attempt": len(item["attempts"]) + 1,
                "started_at": _now(),
                "status": "running",
                "run_dir": None,
                "error": None,
            }
            item["attempts"].append(attempt)
            item["status"] = "running"
            state["active_key"] = spec.key
            _save_state(state_path, state)
            print(f"P4 [{spec.index:02d}/45] start {spec.key}", flush=True)
            cfg = load_crd_config(BASE_CONFIG, overrides=formal_overrides(spec))
            experiment = CRDExperiment(cfg)
            try:
                run_dir = experiment.train()
            except BaseException as exc:
                attempt["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                attempt["ended_at"] = _now()
                attempt["run_dir"] = str(experiment.run_dir.resolve()) if experiment.run_dir else None
                item["status"] = attempt["status"]
                state["status"] = attempt["status"]
                state["active_key"] = None
                _save_state(state_path, state)
                raise
            attempt.update({"status": "completed", "ended_at": _now(), "run_dir": str(run_dir.resolve())})
            item["status"] = "completed"
            item["run_dir"] = str(run_dir.resolve())
            state["active_key"] = None
            _save_state(state_path, state)
            print(f"P4 [{spec.index:02d}/45] completed {run_dir}", flush=True)
        state["status"] = "completed"
        state["complete"] = True
        state["completed_at"] = _now()
        _save_state(state_path, state)
    return state_path


def _load_or_create_state(path: Path, identity: dict[str, Any], identifier: str) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("matrix_id") != identifier or state.get("identity") != identity:
            raise RuntimeError("现有 P4 matrix state identity 不一致")
        return state
    return {
        "protocol": CRD_TF_PROTOCOL_VERSION,
        "phase": "p4_formal_matrix",
        "matrix_id": identifier,
        "identity": identity,
        "status": "pending",
        "complete": False,
        "created_at": _now(),
        "completed_at": None,
        "active_key": None,
        "items": [
            {
                "key": spec.key,
                "index": spec.index,
                "seed": spec.seed,
                "variant": spec.variant,
                "representations": list(spec.representations),
                "run_root": spec.run_root,
                "status": "pending",
                "run_dir": None,
                "attempts": [],
            }
            for spec in formal_plan()
        ],
    }


def _validate_restart_policy(state: dict[str, Any], *, retry_failed: bool, retry_interrupted: bool) -> None:
    failed = [item["key"] for item in state["items"] if item["status"] == "failed"]
    interrupted = [item["key"] for item in state["items"] if item["status"] in {"running", "interrupted"}]
    if failed and not retry_failed:
        raise RuntimeError(f"matrix 含失败 run，确认从头重跑需传 --retry-failed: {failed}")
    if interrupted and not retry_interrupted:
        raise RuntimeError(f"matrix 含中断 run，确认从头重跑需传 --retry-interrupted: {interrupted}")


def _prepare_retries(state: dict[str, Any], *, retry_failed: bool, retry_interrupted: bool) -> None:
    for item in state["items"]:
        if item["status"] == "failed" and retry_failed:
            item["status"] = "pending"
        elif item["status"] in {"running", "interrupted"} and retry_interrupted:
            if item["attempts"] and item["attempts"][-1]["status"] == "running":
                item["attempts"][-1].update(
                    {
                        "status": "interrupted",
                        "ended_at": _now(),
                        "error": "runner restart recovered stale running state",
                    }
                )
            item["status"] = "pending"


def _verify_completed_run(run_dir: Path) -> None:
    required = (
        "checkpoint_best_local_rr.pt",
        "checkpoint_final.pt",
        "config.yaml",
        "metrics.csv",
        "metrics_summary.csv",
        "run_manifest.json",
        "runtime_summary.json",
        "train_history.csv",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"已标记 completed 的 run 产物缺失，不得跳过: {run_dir}: {missing}")


def _save_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
