from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from resp_train.crd.config import CRD_TF_P6A_PROTOCOL_VERSION, FORMAL_SEEDS, load_crd_config
from resp_train.crd.experiment import CRDExperiment, _early_stopping_step
from resp_train.crd.tf_v1_model import (
    P6_GATE_PARAMETER_COUNT,
    TF_P6_VARIANT_REPRESENTATIONS,
    TF_P6_VARIANTS,
    TF_VARIANTS,
    CRDTfV1Model,
    TemporalFusionGate,
    trainable_parameter_count,
)
from resp_train.crd.tf_v1_p6a import (
    P6A_ACCEPTANCE_VARIANTS,
    P6A_EXPECTED_INCREMENTAL_PARAMETERS,
    p6a_acceptance_overrides,
)


def test_p6a_variants_do_not_mutate_frozen_p4_matrix() -> None:
    assert len(TF_VARIANTS) == 15
    assert TF_P6_VARIANTS == (
        "crd_tf401_mws_add",
        "crd_tf402_mws_gate",
        "crd_tf403_ctrl_gate",
    )
    assert set(TF_VARIANTS).isdisjoint(TF_P6_VARIANTS)
    assert TF_P6_VARIANT_REPRESENTATIONS["crd_tf401_mws_add"] == ("m", "w", "s")
    assert TF_P6_VARIANT_REPRESENTATIONS["crd_tf403_ctrl_gate"] == ()


def test_p6a_gate_is_identity_at_initialization_and_bounded_after_update() -> None:
    gate = TemporalFusionGate(3)
    latent = torch.randn(2, 96, 1800)

    initial = gate(latent)
    torch.testing.assert_close(initial, torch.ones_like(initial), rtol=0.0, atol=0.0)
    assert trainable_parameter_count(gate) == P6_GATE_PARAMETER_COUNT

    with torch.no_grad():
        gate.final_projection.weight.normal_()
        gate.final_projection.bias.normal_()
    updated = gate(latent)
    assert bool((updated >= 0.5).all() and (updated <= 1.5).all())


def test_p6a_representation_and_control_gates_are_tightly_capacity_matched() -> None:
    models = {variant: CRDTfV1Model(variant, 20260811) for variant in TF_P6_VARIANTS}
    base_count = trainable_parameter_count(models["crd_tf401_mws_add"].base)
    increments = {
        variant: trainable_parameter_count(model) - base_count for variant, model in models.items()
    }

    assert increments == P6A_EXPECTED_INCREMENTAL_PARAMETERS
    assert increments["crd_tf402_mws_gate"] - increments["crd_tf403_ctrl_gate"] == 224
    assert models["crd_tf401_mws_add"].fusion_gate is None
    assert models["crd_tf402_mws_gate"].branch_parameter_counts()["fusion_gate"] == 13_347
    assert models["crd_tf403_ctrl_gate"].branch_parameter_counts()["fusion_gate"] == 13_347


def test_p6a_gate_does_not_perturb_paired_base_or_branch_initialization() -> None:
    additive = CRDTfV1Model("crd_tf401_mws_add", 20260811)
    gated = CRDTfV1Model("crd_tf402_mws_gate", 20260811)

    for left, right in (
        (additive.base.state_dict(), gated.base.state_dict()),
        (additive.branches.state_dict(), gated.branches.state_dict()),
    ):
        assert left.keys() == right.keys()
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("variant", TF_P6_VARIANTS)
def test_p6a_formal_configs_are_independent_and_strict(variant: str) -> None:
    cfg = load_crd_config(f"configs/crd_tf_v1/{variant}_formal.yaml")

    assert str(cfg.protocol.name) == CRD_TF_P6A_PROTOCOL_VERSION
    assert str(cfg.protocol.stage) == "tf_p6a"
    assert str(cfg.protocol.execution_gate) == "p6a_formal"
    assert str(cfg.protocol.run_role) == "formal"
    assert list(cfg.model.tf_representations) == list(TF_P6_VARIANT_REPRESENTATIONS[variant])
    assert int(cfg.model.initialization_seed) == int(cfg.training.seed) == FORMAL_SEEDS[0]
    assert (cfg.training.epochs, cfg.training.batch_size, cfg.training.gradient_accumulation_steps) == (
        80,
        128,
        1,
    )
    assert cfg.training.early_stopping_enabled is True
    assert cfg.training.early_stopping_patience == 30
    assert cfg.training.early_stopping_min_delta == 0.0
    assert str(cfg.outputs.run_root).endswith(f"/{variant}/seed_{FORMAL_SEEDS[0]}")


@pytest.mark.parametrize("variant", P6A_ACCEPTANCE_VARIANTS)
def test_p6a_acceptance_overrides_resolve_strict_config(tmp_path, variant: str) -> None:
    cfg = load_crd_config(
        "configs/crd_tf_v1/crd_tf401_mws_add_smoke.yaml",
        overrides=p6a_acceptance_overrides(variant, device="cuda:0", output_root=tmp_path),
    )

    assert str(cfg.protocol.execution_gate) == "p6a_cuda_acceptance"
    assert str(cfg.protocol.run_role) == "acceptance"
    assert (cfg.training.epochs, cfg.training.batch_size, cfg.training.gradient_accumulation_steps) == (
        1,
        128,
        1,
    )
    assert (cfg.data.max_train_windows, cfg.data.max_val_windows, cfg.data.max_test_windows) == (
        128,
        32,
        None,
    )


def test_p6a_early_stopping_replay_semantics_wait_for_thirty_misses() -> None:
    best = float("inf")
    wait = 0
    improved, wait = _early_stopping_step(
        value=0.5, best=best, epochs_without_improvement=wait, min_delta=0.0
    )
    assert improved and wait == 0
    best = 0.5

    for miss in range(1, 31):
        improved, wait = _early_stopping_step(
            value=0.5, best=best, epochs_without_improvement=wait, min_delta=0.0
        )
        assert not improved
        assert wait == miss
        assert (wait >= 30) is (miss == 30)


def test_p6a_training_lifecycle_stops_after_thirty_full_validation_misses(
    monkeypatch, tmp_path
) -> None:
    cfg = load_crd_config("configs/crd_tf_v1/crd_tf401_mws_add_formal.yaml")
    # 配置已经过 formal strict validation；此处只把设备替换为 CPU 来隔离测试 lifecycle。
    cfg.training.device = "cpu"
    cfg.outputs.run_root = str(tmp_path)
    class _OneBatchLoader:
        dataset = [0]

        def __len__(self) -> int:
            return 1

    data = SimpleNamespace(
        train=SimpleNamespace(loader=_OneBatchLoader()),
        val=SimpleNamespace(loader=_OneBatchLoader()),
        audit_summary=pd.DataFrame([{"split": "val", "n_windows": 1}]),
    )
    model = torch.nn.Linear(1, 1)
    partition = SimpleNamespace(decay_names=("weight",), no_decay_names=("bias",))

    def fake_train(*_args, update_index: int, **_kwargs):
        summary = {
            "loss": 1.0,
            "loss_sync": 1.0,
            "loss_effort": 0.0,
            "first_learning_rate": 3e-4,
            "last_learning_rate": 3e-4,
        }
        return summary, update_index + 1

    local_rr_values = iter([0.5] * 31)
    monkeypatch.setattr("resp_train.crd.experiment.build_tho_data", lambda _cfg: data)
    monkeypatch.setattr("resp_train.crd.experiment.build_crd_model", lambda _cfg: model)
    monkeypatch.setattr(
        "resp_train.crd.experiment.build_crd_optimizer",
        lambda _model, _cfg: (torch.optim.SGD(_model.parameters(), lr=1e-3), partition),
    )
    monkeypatch.setattr("resp_train.crd.experiment.train_crd_one_epoch", fake_train)
    monkeypatch.setattr(
        "resp_train.crd.experiment.validate",
        lambda *_args, **_kwargs: ({"loss_sync": 1.0, "loss_effort": 0.0}, {}),
    )
    monkeypatch.setattr(
        "resp_train.crd.experiment.validation_local_rr_mean",
        lambda *_args, **_kwargs: next(local_rr_values),
    )
    monkeypatch.setattr(
        "resp_train.crd.experiment.summarize_task_metrics",
        lambda _frame: pd.DataFrame([{"n_samples": 1}]),
    )
    experiment = CRDExperiment(cfg)
    monkeypatch.setattr(
        experiment,
        "_evaluate_model",
        lambda *_args, **_kwargs: pd.DataFrame([{"evaluation_split": "validation"}]),
    )

    run_dir = experiment.train()
    history = pd.read_csv(run_dir / "train_history.csv")
    final_checkpoint = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu")

    assert len(history) == 31
    assert history.iloc[-1]["early_stopping_wait"] == 30
    assert history.iloc[-1]["early_stopping_triggered"] == 1
    assert final_checkpoint["epoch"] == 31
    assert final_checkpoint["extra_state"]["update_index"] == 31
    assert final_checkpoint["extra_state"]["total_updates"] == 80
    assert final_checkpoint["extra_state"]["early_stopping"]["triggered"] is True
