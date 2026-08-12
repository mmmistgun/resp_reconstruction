from __future__ import annotations

from pathlib import Path

import pytest

from resp_train.crd.config import FORMAL_SEEDS, load_crd_config
from resp_train.crd.tf_v1_p4 import (
    P4_ARM_ORDER,
    _load_or_create_state,
    _prepare_retries,
    _validate_restart_policy,
    formal_overrides,
    formal_plan,
    matrix_id,
    matrix_identity,
    state_summary,
)


def test_formal_plan_is_seed_outer_arm_inner_complete_matrix() -> None:
    plan = formal_plan()

    assert len(plan) == 45
    assert [spec.index for spec in plan] == list(range(1, 46))
    assert [spec.variant for spec in plan[:15]] == list(P4_ARM_ORDER)
    assert [spec.seed for spec in plan] == [seed for seed in FORMAL_SEEDS for _ in P4_ARM_ORDER]
    assert len({spec.key for spec in plan}) == 45
    assert plan[0].variant == "crd_tf_ctrl1"
    assert plan[-1].variant == "crd_tf302_wls"


@pytest.mark.parametrize("spec_index", (0, 1, 5, 12, 14, 15, 44))
def test_every_formal_group_resolves_strict_p4_config(spec_index: int) -> None:
    spec = formal_plan()[spec_index]
    cfg = load_crd_config("configs/crd_tf_v1/crd_tf101_m_smoke.yaml", overrides=formal_overrides(spec))

    assert str(cfg.protocol.execution_gate) == "p4_formal"
    assert str(cfg.protocol.run_role) == "formal"
    assert str(cfg.model.variant) == spec.variant
    assert list(cfg.model.tf_representations) == list(spec.representations)
    assert int(cfg.model.initialization_seed) == int(cfg.training.seed) == spec.seed
    assert (cfg.training.epochs, cfg.training.batch_size, cfg.training.gradient_accumulation_steps) == (80, 128, 1)
    assert cfg.training.early_stopping_enabled is False
    assert (cfg.data.max_train_windows, cfg.data.max_val_windows, cfg.data.max_test_windows) == (None, None, None)
    assert str(cfg.outputs.run_root) == spec.run_root


def test_matrix_identity_and_id_are_deterministic() -> None:
    commit = "b" * 40
    left = matrix_identity(commit)
    right = matrix_identity(commit)

    assert left == right
    assert matrix_id(commit) == matrix_id(commit)
    assert left["early_stopping_enabled"] is False
    assert len(left["plan"]) == 45


def test_matrix_state_refuses_implicit_failed_or_interrupted_retry(tmp_path: Path) -> None:
    commit = "c" * 40
    identity = matrix_identity(commit)
    state = _load_or_create_state(tmp_path / "missing.json", identity, matrix_id(commit))
    state["items"][0]["status"] = "failed"
    state["items"][0]["attempts"] = [{"status": "failed"}]
    state["items"][1]["status"] = "running"
    state["items"][1]["attempts"] = [{"status": "running"}]

    with pytest.raises(RuntimeError, match="--retry-failed"):
        _validate_restart_policy(state, retry_failed=False, retry_interrupted=False)
    with pytest.raises(RuntimeError, match="--retry-interrupted"):
        _validate_restart_policy(state, retry_failed=True, retry_interrupted=False)

    _validate_restart_policy(state, retry_failed=True, retry_interrupted=True)
    _prepare_retries(state, retry_failed=True, retry_interrupted=True)
    assert state["items"][0]["status"] == "pending"
    assert state["items"][1]["status"] == "pending"
    assert state["items"][1]["attempts"][-1]["status"] == "interrupted"

    state["items"][2]["status"] = "completed"
    state["items"][2]["run_dir"] = "/tmp/completed"
    summary = state_summary(state)
    assert summary["counts"] == {"pending": 44, "completed": 1}
    assert summary["last_completed_key"] == state["items"][2]["key"]
