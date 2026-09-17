"""以短 CPU fixture 验证工程门控、FP32 更新舍入及失败现场持久化。"""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from resp_train.crd.training import crd_learning_rate
from resp_train.paper_evidence import e4_aggregation_v2_engineering as engineering


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        branch = nn.Module()
        branch.aggregation = nn.ParameterDict({"bias": nn.Parameter(torch.tensor([.1]))})
        self.branches = nn.ModuleDict({"w": branch})

    def forward(self, value):
        return self.branches["w"].aggregation["bias"] * value


@pytest.fixture
def case(tmp_path, monkeypatch):
    cfg = SimpleNamespace(training=SimpleNamespace(seed=20260813),
                          model=SimpleNamespace(aggregation_v2=SimpleNamespace(arm="scale_attention")))
    model = FixtureModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, eps=1e-8, weight_decay=0)
    state = SimpleNamespace(model=model, optimizer=optimizer, gradient=5e-12, steps=0,
                            cfg=cfg, path=tmp_path / "diagnostics")

    def update(model, batch, loss, optimizer, cfg, index):
        state.steps += 1
        optimizer.zero_grad(set_to_none=True)
        lr = crd_learning_rate(index, total_updates=6400, max_learning_rate=3e-4,
                               min_learning_rate=3e-5, warmup_fraction=.05)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if state.gradient is not None:
            model(torch.tensor(state.gradient)).sum().backward()
        optimizer.step()
        return {"last_learning_rate": lr}

    def run_case(cfg, batch_size, check_initial, record, diagnostic_dir):
        engineering.acceptance_updates(model, None, None, optimizer, cfg, record, diagnostic_dir)

    monkeypatch.setattr(engineering, "update", update)
    monkeypatch.setattr(engineering, "_acceptance_case", run_case)
    return state


def run(case):
    return engineering.acceptance_case(case.cfg, 1, False, diagnostic_dir=case.path)


def test_native_warmup_positive_gradient_can_round_to_no_update_at_step_five(case):
    record = run(case)
    fifth = json.loads((case.path / "step_005.json").read_text())
    assert fifth["gradient_norms"]["bias"] > 0
    assert fifth["max_abs_changes"]["bias"] == 0
    assert not fifth["parameters_updated"]["bias"]
    assert fifth["summary"]["last_learning_rate"] == 4.6875e-6
    assert 5 < record["updates"] <= 20
    assert record["status"] == "passed"
    assert record["new_parameters_updated"] == {"bias": True}
    assert record["new_parameter_max_abs_changes"][-1]["bias"] > 0
    assert len(list(case.path.glob("step_*.json"))) == record["updates"]
    assert json.loads((case.path / "case.json").read_text()) == record


def test_minimum_five_updates_and_gradient_requirement(case):
    case.gradient = 1e-3
    record = run(case)
    assert record["updates"] == case.steps == 5
    assert not case.model._forward_hooks


@pytest.mark.parametrize("gradient, reason", [(0., "零梯度参数"), (1e-20, "未更新参数")])
def test_bounded_failure_keeps_named_parameter_evidence(case, gradient, reason):
    case.gradient = gradient
    with pytest.raises(RuntimeError, match=reason) as error:
        run(case)
    record = json.loads((case.path / "case.json").read_text())
    assert case.steps == record["updates"] == 20
    assert record["status"] == "failed"
    assert "bias" in record["error"]
    assert any("seed=20260813" in note for note in error.value.__notes__)
    assert len(list(case.path.glob("step_*.json"))) == 20
    assert not case.model._forward_hooks


@pytest.mark.parametrize("gradient, error_type, message", [
    (None, RuntimeError, "gradient 通路缺失"),
    (float("nan"), FloatingPointError, None),
    (float("inf"), FloatingPointError, None),
])
def test_missing_and_nonfinite_gradients_fail_immediately(case, gradient, error_type, message):
    case.gradient = gradient
    with pytest.raises(error_type, match=message):
        run(case)
    record = json.loads((case.path / "case.json").read_text())
    assert record["status"] == "failed" and record["stage"] == "update_1"
    assert case.steps == 1
    assert not case.model._forward_hooks


def test_tiny_nonzero_gradient_norm_does_not_underflow(case):
    case.gradient = 1e-30
    with torch.no_grad():
        case.model.branches["w"].aggregation["bias"].zero_()
    record = run(case)
    assert record["updates"] == 5
    assert all(row["bias"] > 0 for row in record["new_parameter_gradient_norms"])


def test_prior_parameter_change_does_not_excuse_zero_final_gradient(case, monkeypatch):
    original = engineering.update

    def interrupted_gradient(*args):
        case.gradient = 1e-3 if args[-1] == 0 else 0.
        return original(*args)

    monkeypatch.setattr(engineering, "update", interrupted_gradient)
    with pytest.raises(RuntimeError, match="零梯度参数"):
        run(case)
    record = json.loads((case.path / "case.json").read_text())
    assert record["updates"] == 20
    assert record["new_parameters_updated"] == {"bias": True}
    assert record["new_parameter_gradient_norms"][-1] == {"bias": 0.}


def test_failure_after_partial_progress_retains_steps_and_exception(case, monkeypatch):
    original = engineering.update

    def interrupted(*args):
        if args[-1] == 2:
            raise RuntimeError("synthetic CUDA failure")
        return original(*args)

    monkeypatch.setattr(engineering, "update", interrupted)
    with pytest.raises(RuntimeError, match="synthetic CUDA failure"):
        run(case)
    record = json.loads((case.path / "case.json").read_text())
    assert record["updates"] == 2 and record["stage"] == "update_3"
    assert len(list(case.path.glob("step_*.json"))) == 2
    assert not case.model._forward_hooks
    with pytest.raises(FileExistsError):
        run(case)
