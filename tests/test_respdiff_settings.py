"""论文/源码设置核对，仅解析参考脚本并使用标量CPU optimizer。"""

import ast
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from resp_train.respdiff.provenance import verify_source
from resp_train.respdiff.settings import (
    build_source_optimizer, build_source_scheduler, dataset_dependent_budget, validate_source_settings,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/mnt/disk_code/marques/reference_repos/RespDiff")


def load_config():
    spec = importlib.util.spec_from_file_location("respdiff_settings_cli", ROOT / "scripts/run_respdiff_tho_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_development_config(module.DEFAULT_CONFIG)


def source_optimization(model):
    verify_source(SOURCE)
    tree = ast.parse((SOURCE / "breathing_bidmc_fft.py").read_text())
    names = {"optimizer", "num_epochs", "p1", "p2", "lr_scheduler"}
    statements = sorted((node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                         and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                         and node.targets[0].id in names), key=lambda node: node.lineno)
    assert len(statements) == len(names)
    scope = {"torch": torch, "Adam": torch.optim.Adam, "model": model}
    # 提取的只有优化器/轮数/调度赋值；不执行数据加载、cuda或训练循环。
    exec(compile(ast.Module(body=statements, type_ignores=[]), "source_optimizer_only", "exec"), scope)
    return scope


def test_optimizer_epochs_and_scheduler_match_source_at_all_boundaries():
    config = load_config()
    reference_model = torch.nn.Linear(1, 1)
    model = copy.deepcopy(reference_model)
    original = source_optimization(reference_model)
    optimizer = build_source_optimizer(model, config)
    scheduler = build_source_scheduler(optimizer, config)
    assert config["training"]["epochs"] == original["num_epochs"] == 400
    assert optimizer.defaults == original["optimizer"].defaults
    rates = []
    for epoch in range(1, 401):
        rates.append(optimizer.param_groups[0]["lr"])
        assert rates[-1] == original["optimizer"].param_groups[0]["lr"]
        for opt, net in ((optimizer, model), (original["optimizer"], reference_model)):
            opt.zero_grad(set_to_none=True)
            net(torch.ones(1, 1)).square().sum().backward()
            opt.step()
        scheduler.step()
        original["lr_scheduler"].step()
    # step在epoch末尾：第280轮仍用1e-4，第281轮才用1e-5。
    for epoch, expected in ((1, 1e-4), (280, 1e-4), (281, 1e-5), (396, 1e-5), (397, 1e-6), (400, 1e-6)):
        assert rates[epoch - 1] == pytest.approx(expected)
    for reference, actual in zip(reference_model.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(reference, actual, rtol=0, atol=0)


def test_declared_batches_and_sampling_count_match_source_ast():
    config = load_config()
    tree = ast.parse((SOURCE / "breathing_bidmc_fft.py").read_text())
    loaders = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"train_loader", "val_loader"}):
            loaders[node.targets[0].id] = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
    assert config["training"]["batch_size"] == loaders["train_loader"]["batch_size"] == 128
    assert config["inference"]["batch_size"] == loaders["val_loader"]["batch_size"] == 64
    samples = [ast.literal_eval(kw.value) for node in ast.walk(tree) if isinstance(node, ast.Call)
               for kw in node.keywords if kw.arg == "n_samples"]
    assert samples == [config["inference"]["n_samples"]] == [100]


@pytest.mark.parametrize("section,key,value", [
    ("training", "epochs", 80), ("training", "batch_size", 64),
    ("training", "checkpoint_policy", "best_validation_local_rr"),
    ("training", "early_stopping", True), ("training", "learning_rate", 3e-4),
    ("inference", "n_samples", 1), ("inference", "batch_size", 128),
    ("model", "hidden_dim", 8),
])
def test_reference_settings_reject_silent_method_changes(section, key, value):
    config = load_config()
    config[section][key] = value
    with pytest.raises(ValueError, match="偏离"):
        validate_source_settings(config)


def test_dataset_size_derives_updates_without_changing_method_settings():
    config = load_config()
    result = dataset_dependent_budget(config, train_chunks=10141 * 36, validation_chunks=2675 * 36)
    assert result["updates_per_epoch"] == 2853
    assert result["total_updates_per_seed"] == 1141200
    assert result["validation_batches"] == 1505
    assert result["denoiser_calls_per_validation"] == 7525000
    smaller = dataset_dependent_budget(config, train_chunks=129, validation_chunks=65)
    assert smaller["total_updates_per_seed"] == 800
    assert smaller["denoiser_calls_per_validation"] == 10000
    assert config["signal"]["formal_normalization"] is None
    assert config["protocol"]["formal_enabled"] is False
    with pytest.raises(ValueError):
        dataset_dependent_budget(config, train_chunks=0, validation_chunks=65)


@pytest.mark.parametrize("key,value", [
    ("proxy", {"enabled": True}), ("top_k", 5),
    ("validation_interval_updates", 80), ("max_updates", 6400),
])
def test_source_training_rejects_unregistered_selection_and_budget_fields(key, value):
    config = load_config()
    config["training"][key] = value
    with pytest.raises(ValueError, match="字段与来源设置不一致"):
        validate_source_settings(config)
