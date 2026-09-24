from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from resp_train.crd.training import crd_learning_rate
from resp_train.paper_evidence import e7_scale_encoding_aggregation_engineering as engineering


class FixtureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        branch = nn.Module()
        branch.scale_encoder = nn.ParameterDict({"weight": nn.Parameter(torch.tensor([0.1]))})
        branch.aggregation = nn.Module()
        branch.final_projection = nn.Linear(1, 1, bias=False)
        self.branches = nn.ModuleDict({"w": branch})

    def forward(self, value):
        branch = self.branches["w"]
        encoded = branch.scale_encoder["weight"] * value
        return branch.final_projection(encoded.reshape(-1, 1)).sum()


@pytest.fixture
def case(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        training=SimpleNamespace(seed=20260813),
        model=SimpleNamespace(e7_factorial=SimpleNamespace(arm="s1_deep_local__mean")),
    )
    model = FixtureModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, eps=1e-8, weight_decay=0)
    state = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        gradient=1e-3,
        steps=0,
        cfg=cfg,
        path=tmp_path / "diagnostics",
    )

    def update(model, batch, loss, optimizer, cfg, index):
        state.steps += 1
        optimizer.zero_grad(set_to_none=True)
        lr = crd_learning_rate(
            index,
            total_updates=6400,
            max_learning_rate=3e-4,
            min_learning_rate=3e-5,
            warmup_fraction=0.05,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        if state.gradient is not None:
            model(torch.tensor(state.gradient)).sum().backward()
        optimizer.step()
        return {"last_learning_rate": lr}

    def run_case(cfg, batch_size, check_initial, record, diagnostic_dir):
        engineering.acceptance_updates(
            model, None, None, optimizer, cfg, record, diagnostic_dir
        )

    monkeypatch.setattr(engineering, "update", update)
    monkeypatch.setattr(engineering, "_acceptance_case", run_case)
    return state


def run(case):
    return engineering.acceptance_case(
        case.cfg, 1, False, diagnostic_dir=case.path
    )


def test_p3_contract_matrix_and_benchmark_order():
    contract = engineering.p3_contract()
    assert contract["acceptance"]["batch1_cells"] == 18
    assert contract["acceptance"]["batch128_cells"] == 6
    assert contract["acceptance"]["branch_checkpoint_batch_chunk"] == 8
    assert contract["benchmark"]["processes"] == 36
    assert engineering.benchmark_order(0) == engineering.ARMS
    assert engineering.benchmark_order(1) == tuple(reversed(engineering.ARMS))
    assert engineering.benchmark_order(2) == engineering.ARMS[2:] + engineering.ARMS[:2]
    with pytest.raises(ValueError):
        engineering.benchmark_order(3)


def test_p1_lock_and_p3_critical_paths_are_available():
    _lock, digest = engineering.e7.load_implementation_lock()
    assert digest == engineering.P1_LOCK_SHA256
    assert all((engineering.e7.ROOT / path).is_file() for path in engineering.p3_critical_paths())


def test_minimum_five_updates_and_diagnostics(case):
    record = run(case)
    assert record["updates"] == case.steps == 5
    assert record["status"] == "passed"
    assert record["tracked_parameters_updated"] == {
        "branches.w.scale_encoder.weight": True
    }
    assert len(list(case.path.glob("step_*.json"))) == 5
    assert json.loads((case.path / "case.json").read_text()) == record
    assert not case.model._forward_hooks


@pytest.mark.parametrize("gradient,reason", [(0.0, "零梯度参数"), (1e-20, "未更新参数")])
def test_bounded_stall_preserves_failure_evidence(case, gradient, reason):
    case.gradient = gradient
    with pytest.raises(RuntimeError, match=reason):
        run(case)
    record = json.loads((case.path / "case.json").read_text())
    assert record["updates"] == 20
    assert record["status"] == "failed"
    assert len(list(case.path.glob("step_*.json"))) == 20
    assert not case.model._forward_hooks


@pytest.mark.parametrize("gradient,error", [(None, RuntimeError), (float("nan"), FloatingPointError)])
def test_missing_or_nonfinite_gradient_fails(case, gradient, error):
    case.gradient = gradient
    with pytest.raises(error):
        run(case)
    record = json.loads((case.path / "case.json").read_text())
    assert record["status"] == "failed"
    assert record["stage"] == "update_1"
    assert not case.model._forward_hooks


def test_attempt_lifecycle_is_non_overwriting(tmp_path):
    digest = "a" * 64
    parent = tmp_path / "gpu"
    with engineering.phase_guard(parent, digest, "gpu_acceptance"):
        with engineering.attempt(parent, digest, "gpu_acceptance") as output:
            engineering.e7.write_json(output / "environment.json", {"fixture": True})
            engineering.e7.write_json(output / "access_receipt.json", {"fixture": True})
            engineering.e7.write_json(output / "gpu_acceptance.json", {"passed": True})
            engineering.e7.write_json(output / "parameter_compute_report.json", {})
    engineering.verify_attempt(output, phase="gpu_acceptance", lock_hash=digest)
    with pytest.raises(FileExistsError):
        with engineering.phase_guard(parent, digest, "gpu_acceptance"):
            pass
