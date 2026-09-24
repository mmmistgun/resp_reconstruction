from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from resp_train.crd.tf_v1_model import CRDTfV1Model, trainable_parameter_count
from resp_train.crd.training import crd_learning_rate, partition_weight_decay_parameters
from resp_train.paper_evidence import e7_scale_encoding_aggregation as e7
from resp_train.paper_evidence.e7_scale_encoding_aggregation_model import (
    AGGREGATIONS,
    ARMS,
    BLOCK_PARAMETERS,
    DILATIONS,
    ENCODER_PARAMETERS,
    ENCODERS,
    THEORETICAL_RECEPTIVE_FIELDS,
    ChannelLayerNorm2D,
    E7ScaleFactorialModel,
    FrequencyAttentionAggregation,
    ScaleEncoder,
    ScaleResidualBlock,
    arm_contract,
    build_e7_model,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def frequencies() -> np.ndarray:
    return e7.load_frequencies()


def baseline(seed: int = e7.SEEDS[0]):
    return e7.baseline_config(seed)


def derived(tmp_path: Path, frequencies: np.ndarray, arm: str = ARMS[0], seed: int = e7.SEEDS[0]):
    reference = baseline(seed)
    cfg = e7.derived_config(
        reference,
        arm,
        frequencies,
        output_root=tmp_path,
        device="cpu",
    )
    return reference, cfg


def test_spec_matrix_and_all_resolved_configs_are_strict(tmp_path, frequencies):
    spec = e7.load_experiment_spec()
    assert tuple(spec.matrix.arms) == ARMS
    assert tuple(spec.matrix.seeds) == e7.SEEDS
    assert spec.matrix.max_epochs == 80
    assert spec.matrix.early_stopping.min_epoch == 30
    assert spec.matrix.early_stopping.patience == 15
    assert spec.matrix.early_stopping.min_delta == 0.0
    assert spec.tolerances.error_relative_percent == 0.5
    assert spec.tolerances.pcc_absolute == 0.002
    for arm in ARMS:
        for seed in e7.SEEDS:
            reference, cfg = derived(tmp_path / arm / str(seed), frequencies, arm, seed)
            e7.validate_config(
                cfg,
                reference,
                arm,
                frequencies,
                output_root=tmp_path / arm / str(seed),
                device="cpu",
            )
            assert cfg.training.epochs == 80
            assert cfg.training.early_stopping_enabled is True
            assert cfg.training.early_stopping_min_epoch == 30
            assert cfg.training.early_stopping_patience == 15
            assert cfg.training.early_stopping_min_delta == 0.0
    assert e7.check_p1()["resolved_config_count"] == 18


@pytest.mark.parametrize(
    "key,value",
    [
        ("model.e7_factorial.encoder_kernel", [7, 1]),
        ("model.e7_factorial.arm", "other"),
        ("model.e7_factorial.fill_hidden_channels", 64),
        ("training.epochs", 79),
        ("training.early_stopping_enabled", False),
        ("training.early_stopping_min_epoch", 29),
        ("training.early_stopping_patience", 14),
        ("training.early_stopping_min_delta", 0.1),
        ("training.batch_size", 64),
        ("loss.effort_weight", 0.0),
        ("model.initialization_seed", 1),
        ("data.train_split", "test"),
        ("protocol.name", "other"),
    ],
)
def test_scientific_config_drift_is_rejected(tmp_path, frequencies, key, value):
    reference, cfg = derived(tmp_path, frequencies)
    OmegaConf.update(cfg, key, value)
    with pytest.raises((ValueError, TypeError)):
        e7.validate_config(
            cfg,
            reference,
            ARMS[0],
            frequencies,
            output_root=tmp_path,
            device="cpu",
        )


def test_channel_layer_norm_and_residual_block_contract():
    norm = ChannelLayerNorm2D(96)
    value = torch.randn(2, 96, 13, 7, generator=torch.Generator().manual_seed(1))
    normalized = norm(value)
    assert normalized.shape == value.shape
    torch.testing.assert_close(normalized.mean(dim=1), torch.zeros(2, 13, 7), atol=2e-6, rtol=0)
    torch.testing.assert_close(
        normalized.var(dim=1, unbiased=False),
        torch.ones(2, 13, 7),
        atol=2e-4,
        rtol=0,
    )
    for dilation in (1, 2, 4, 8):
        block = ScaleResidualBlock(dilation)
        assert trainable_parameter_count(block) == BLOCK_PARAMETERS
        assert block.depthwise.padding == (4 * dilation, 0)
        assert block.depthwise.dilation == (dilation, 1)
        assert block.depthwise.groups == 96
        output = block(value)
        assert output.shape == value.shape and torch.isfinite(output).all()
        assert block.projection.weight.ne(0).any()


def test_receptive_field_parameter_and_encoder_initialization_pairing():
    assert THEORETICAL_RECEPTIVE_FIELDS == {
        "s0_shallow": 7,
        "s1_deep_local": 39,
        "s2_axis_spanning": 127,
    }
    assert ENCODER_PARAMETERS == {
        "s0_shallow": 0,
        "s1_deep_local": 4 * BLOCK_PARAMETERS,
        "s2_axis_spanning": 4 * BLOCK_PARAMETERS,
    }
    local = ScaleEncoder("s1_deep_local", 17)
    spanning = ScaleEncoder("s2_axis_spanning", 17)
    assert DILATIONS[local.encoder] == (1, 1, 1, 1)
    assert DILATIONS[spanning.encoder] == (1, 2, 4, 8)
    assert local.state_dict().keys() == spanning.state_dict().keys()
    assert all(
        torch.equal(value, spanning.state_dict()[name])
        for name, value in local.state_dict().items()
    )
    value = torch.randn(1, 96, 17, 5, generator=torch.Generator().manual_seed(2))
    bypassed = local(value, enabled=False)
    torch.testing.assert_close(bypassed, value, atol=0, rtol=0)
    output, records = local(value, return_blocks=True)
    assert output.shape == value.shape and len(records) == 4
    output.square().mean().backward()
    for parameter in local.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.ne(0).any()


@pytest.mark.parametrize("seed", e7.SEEDS)
def test_six_arm_model_state_rng_parameters_and_optimizer(seed, frequencies):
    torch.manual_seed(913)
    start = torch.get_rng_state()
    reference = CRDTfV1Model("crd_tf102_w", seed)
    expected_rng = torch.get_rng_state()
    models = {}
    for arm in ARMS:
        torch.set_rng_state(start)
        model = E7ScaleFactorialModel(arm, seed, frequencies)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        contract = arm_contract(arm)
        assert trainable_parameter_count(model) == contract["model_parameters"]
        assert trainable_parameter_count(model.e7_branch) == contract["branch_parameters"]
        assert model.e7_branch.parameter_fill.expand.out_channels == 65
        models[arm] = model

        shared_reference = {
            key: value
            for key, value in reference.state_dict().items()
            if not key.startswith("branches.w.scale_encoder.")
            and not key.startswith("branches.w.aggregation.")
        }
        shared_candidate = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("branches.w.scale_encoder.")
            and not key.startswith("branches.w.aggregation.")
        }
        assert shared_reference.keys() == shared_candidate.keys()
        assert all(
            torch.equal(value, shared_candidate[name])
            for name, value in shared_reference.items()
        )

    for aggregation in AGGREGATIONS:
        local = models[f"s1_deep_local__{aggregation}"].e7_branch.scale_encoder.state_dict()
        spanning = models[f"s2_axis_spanning__{aggregation}"].e7_branch.scale_encoder.state_dict()
        assert local.keys() == spanning.keys()
        assert all(torch.equal(value, spanning[name]) for name, value in local.items())
    attention_states = [
        models[f"{encoder}__frequency_attention"].e7_branch.aggregation.state_dict()
        for encoder in ENCODERS
    ]
    assert all(
        all(torch.equal(first[name], other[name]) for name in first)
        for first, other in [(attention_states[0], state) for state in attention_states[1:]]
    )
    score = models["s0_shallow__frequency_attention"].e7_branch.aggregation.score.weight
    assert score.ne(0).any()

    reference_partition = partition_weight_decay_parameters(reference)
    candidate_partition = partition_weight_decay_parameters(
        models["s2_axis_spanning__frequency_attention"]
    )
    assert set(reference_partition.decay_names).issubset(candidate_partition.decay_names)
    assert {
        name
        for name in candidate_partition.no_decay_names
        if ".scale_encoder.blocks." in name and ".norm." in name
    }


def test_frequency_attention_standard_forward_uniform_intervention_and_gradient(frequencies):
    aggregation = FrequencyAttentionAggregation(23, frequencies)
    value = torch.randn(1, 96, 97, 360, generator=torch.Generator().manual_seed(3), requires_grad=True)
    output, details = aggregation(value, return_details=True)
    assert output.shape == (1, 96, 360)
    assert details["weights"].shape == (1, 1, 97, 360)
    torch.testing.assert_close(
        details["weights"].sum(dim=2),
        torch.ones(1, 1, 360),
        atol=1e-6,
        rtol=0,
    )
    assert details["correction"].abs().max() > 0
    uniform, uniform_details = aggregation(value, uniform=True, return_details=True)
    torch.testing.assert_close(uniform, value.mean(dim=2), atol=0, rtol=0)
    torch.testing.assert_close(uniform_details["correction"], torch.zeros_like(uniform), atol=0, rtol=0)
    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    for parameter in aggregation.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.ne(0).any()


def test_builder_branch_features_and_interventions(tmp_path, frequencies):
    _, cfg = derived(tmp_path, frequencies, "s0_shallow__frequency_attention")
    model = build_e7_model(cfg)
    tf = {"w": torch.randn(1, 97, 360, generator=torch.Generator().manual_seed(4))}
    details = model.e7_branch.forward_features(tf)
    assert details["x0"].shape == (1, 96, 97, 360)
    assert details["xe"].shape == details["x0"].shape
    assert details["aggregated"].shape == (1, 96, 360)
    assert details["weights"].shape == (1, 1, 97, 360)
    with model.intervention(residual_enabled=False, uniform_attention=True):
        intervened = model.e7_branch.forward_features(tf)
    torch.testing.assert_close(intervened["xe"], intervened["x0"], atol=0, rtol=0)
    torch.testing.assert_close(intervened["aggregated"], intervened["mean"], atol=0, rtol=0)
    assert model.e7_branch._residual_enabled is True
    assert model.e7_branch._uniform_attention is False
    with pytest.raises((ValueError, FloatingPointError)):
        model.e7_branch.forward_features({"w": torch.full((1, 97, 360), float("nan"))})


def _history(cfg, epochs: int = 30) -> pd.DataFrame:
    rows = []
    for epoch in range(1, epochs + 1):
        rows.append(
            {
                "epoch": epoch,
                "optimizer_update": epoch * e7.UPDATES_PER_EPOCH,
                "train_loss_total": 1.25,
                "train_loss_sync": 1.0,
                "train_loss_effort": 1.0,
                "first_learning_rate": crd_learning_rate(
                    (epoch - 1) * e7.UPDATES_PER_EPOCH,
                    total_updates=e7.EPOCHS * e7.UPDATES_PER_EPOCH,
                    max_learning_rate=float(cfg.training.max_learning_rate),
                    min_learning_rate=float(cfg.training.min_learning_rate),
                    warmup_fraction=float(cfg.training.warmup_fraction),
                ),
                "last_learning_rate": crd_learning_rate(
                    epoch * e7.UPDATES_PER_EPOCH - 1,
                    total_updates=e7.EPOCHS * e7.UPDATES_PER_EPOCH,
                    max_learning_rate=float(cfg.training.max_learning_rate),
                    min_learning_rate=float(cfg.training.min_learning_rate),
                    warmup_fraction=float(cfg.training.warmup_fraction),
                ),
                "val_local_rr_mae": 1.0,
                "early_stopping_improved": int(epoch == 1),
                "early_stopping_wait": epoch - 1,
                "early_stopping_triggered": int(epoch == 30),
                "early_stopping_min_epoch": 30,
            }
        )
    return pd.DataFrame(rows)


def test_early_stop_history_replay_and_planned_lr(tmp_path, frequencies):
    _, cfg = derived(tmp_path, frequencies)
    history = _history(cfg)
    assert e7.validate_history(history, cfg) == 1
    for bad in (
        history.iloc[:-1],
        history.assign(optimizer_update=0),
        history.assign(train_loss_total=2.0),
        history.assign(last_learning_rate=1.0),
        history.assign(early_stopping_wait=0),
    ):
        with pytest.raises((ValueError, FloatingPointError)):
            e7.validate_history(bad, cfg)


def metric_matrix() -> pd.DataFrame:
    records = []
    arm_offsets = {arm: index * 0.1 for index, arm in enumerate(ARMS)}
    for arm in ARMS:
        for seed_index, seed in enumerate(e7.SEEDS):
            base = 1.0 + seed_index
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "split": "val",
                    **{
                        metric + "_mean": base + arm_offsets[arm]
                        for metric in e7.ERRORS
                    },
                    e7.PCC + "_mean": 0.9 - arm_offsets[arm],
                }
            )
    return pd.DataFrame(records)


def test_complete_factorial_contrasts_and_endpoint_identity():
    simple_seed, simple_across, factorial_seed, factorial_across = e7.contrast_tables(metric_matrix())
    assert len(simple_seed) == 7 * 3 * 5
    assert len(simple_across) == 7 * 5
    assert len(factorial_seed) == 5 * 3 * 5
    assert len(factorial_across) == 5 * 5
    indexed = factorial_seed.set_index(["contrast", "seed", "metric"])
    for seed in e7.SEEDS:
        for metric in e7.PRIMARY:
            endpoint = indexed.loc[("i_endpoint", seed, metric), "utility_delta"]
            component = (
                indexed.loc[("i_local", seed, metric), "utility_delta"]
                + indexed.loc[("i_span", seed, metric), "utility_delta"]
            )
            assert endpoint == pytest.approx(component)
    for invalid in (
        metric_matrix().iloc[:-1],
        pd.concat([metric_matrix(), metric_matrix().iloc[:1]], ignore_index=True),
        metric_matrix().assign(split="test"),
        metric_matrix().assign(**{e7.PCC + "_mean": np.nan}),
    ):
        with pytest.raises((ValueError, FloatingPointError)):
            e7.contrast_tables(invalid)


def test_validation_summary_writes_complete_tables_once(tmp_path):
    frame = metric_matrix()
    output = e7.write_validation_summary(frame, tmp_path / "summary")
    assert len(pd.read_csv(output / "per_seed.csv")) == 18
    assert len(pd.read_csv(output / "across_seed.csv")) == 6 * 5
    assert len(pd.read_csv(output / "simple_effects_per_seed.csv")) == 7 * 3 * 5
    assert len(pd.read_csv(output / "simple_effects_across_seed.csv")) == 7 * 5
    assert len(pd.read_csv(output / "factorial_contrasts_per_seed.csv")) == 5 * 3 * 5
    assert len(pd.read_csv(output / "factorial_contrasts_across_seed.csv")) == 5 * 5
    assert (output / "parameter_compute_report.json").is_file()
    assert (output / "summary_receipt.json").is_file()
    with pytest.raises(FileExistsError):
        e7.write_validation_summary(frame, output)


def test_subject_macro_uses_equal_subject_weight():
    records = []
    for samp_id, values in ((10, [1.0, 3.0]), (20, [10.0])):
        for value in values:
            records.append(
                {
                    "arm": ARMS[0],
                    "seed": e7.SEEDS[0],
                    "split": "val",
                    "samp_id": samp_id,
                    **{metric: value for metric in e7.PRIMARY},
                }
            )
    table = e7.subject_macro_table(pd.DataFrame(records))
    macro = table.loc[
        table.metric.eq(e7.PRIMARY[0]) & table.row_type.eq("macro"),
        "subject_macro_mean",
    ].item()
    assert macro == pytest.approx(6.0)
    assert set(table.loc[table.row_type.eq("subject"), "samp_id"]) == {10, 20}


def test_representation_attention_rms_and_compute_diagnostics():
    generator = torch.Generator().manual_seed(5)
    value = torch.randn(2, 3, 97, 4, generator=generator)
    summary, correlation = e7.scale_representation_statistics(value)
    assert correlation.shape == (97, 97)
    np.testing.assert_allclose(np.diag(correlation), 1.0, atol=1e-12, rtol=0)
    assert 1 <= summary["entropy_effective_rank"] <= 97
    assert e7.rms_ratio(value * 0.1, value) == pytest.approx(0.1, rel=1e-6)
    weights = torch.full((2, 1, 97, 4), 1 / 97)
    attention = e7.attention_statistics(weights)
    assert attention["attention_entropy"] == pytest.approx(np.log(97), rel=1e-6)
    assert attention["attention_total_variation"] == pytest.approx(0.0)
    assert sum(attention[f"region_{index}_mass"] for index in range(4)) == pytest.approx(1.0)
    report = e7.parameter_compute_report()
    assert set(report["arms"]) == set(ARMS)
    assert report["whole_model_flops"] is None
    assert report["arms"]["s1_deep_local__mean"]["covered_residual_encoder_macs"] > 0
    assert report["arms"]["s0_shallow__mean"]["covered_residual_encoder_macs"] == 0
    assert report["arms"]["s0_shallow__frequency_attention"]["covered_attention_scorer_macs"] > 0
