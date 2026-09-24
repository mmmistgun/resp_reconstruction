from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from resp_train.crd.training import build_crd_optimizer, crd_learning_rate
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.w0_structural_factorial_v1_model import (
    ARMS,
    ARM_SPECS,
    CONV20_CONTRACT,
    Conv20Frontend,
    W0StructuralFactorialModel,
    build_w0_structural_factorial_model,
    model_contract,
    trainable_parameter_count,
)
from resp_train.temporal.blocks import ChannelLayerNorm1D, TemporalStem


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(old)


def baseline(seed: int = sf.SEEDS[0]):
    return sf.load_w0_baseline(seed)


def test_spec_and_derived_configs_are_strict(tmp_path):
    spec = sf.load_experiment_spec()
    assert tuple(spec.arms) == ARMS
    assert OmegaConf.to_container(spec.arms, resolve=True) == sf._arm_spec_payload()
    for seed in sf.SEEDS:
        reference = baseline(seed)
        frozen = OmegaConf.to_container(reference, resolve=True)
        for arm in ARMS:
            output = tmp_path / arm / f"seed_{seed}"
            cfg = sf.derived_config(reference, arm=arm, output_root=output, device="cpu")
            sf.validate_config(cfg, reference, arm=arm, output_root=output, device="cpu")
            assert OmegaConf.to_container(cfg.model.w0_structural_factorial_v1, resolve=True) == model_contract(arm)
        assert OmegaConf.to_container(reference, resolve=True) == frozen

    cfg = sf.derived_config(baseline(), arm=ARMS[0], output_root=tmp_path, device="cpu")
    cfg.training.early_stopping_patience = 14
    with pytest.raises(ValueError, match="不一致"):
        sf.validate_config(cfg, baseline(), arm=ARMS[0], output_root=tmp_path, device="cpu")


def test_conv20_is_exact_temporal_stem_contract():
    torch.manual_seed(91)
    expected = TemporalStem()
    torch.manual_seed(91)
    actual = Conv20Frontend()
    assert expected.state_dict().keys() == actual.state_dict().keys()
    assert all(torch.equal(value, actual.state_dict()[name]) for name, value in expected.state_dict().items())
    assert trainable_parameter_count(actual) == 24_672
    assert sum(isinstance(module, ChannelLayerNorm1D) for module in actual.modules()) == 3
    assert CONV20_CONTRACT["fixed_fir_values"] == 382
    x = torch.randn(1, 1, 18_000, generator=torch.Generator().manual_seed(3))
    output = actual(x)
    assert output.shape == (1, 96, 1_800)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("arm", ARMS)
def test_eight_models_have_exact_structure_parameters_and_active_optimizer(arm):
    seed = sf.SEEDS[0]
    model = W0StructuralFactorialModel(arm, seed)
    spec = ARM_SPECS[arm]
    assert trainable_parameter_count(model) == spec.trainable_parameters
    assert model_contract(arm)["factors"] == list(spec.factors)
    assert isinstance(model.base.frontend, Conv20Frontend) == (spec.frontend == "conv20")
    assert isinstance(model.branches["w"].temporal, nn.Identity) == (not spec.cwt_temporal)
    assert isinstance(model.base.refinement, nn.Identity) == (not spec.refinement)
    cfg = sf.derived_config(baseline(seed), arm=arm, output_root=Path("/tmp/sfv1"), device="cpu")
    built = build_w0_structural_factorial_model(cfg)
    optimizer, partition = build_crd_optimizer(built, cfg)
    named = {name for name, parameter in built.named_parameters() if parameter.requires_grad}
    assert set(partition.decay_names) | set(partition.no_decay_names) == named
    assert set(partition.decay_names).isdisjoint(partition.no_decay_names)
    optimizer_names = {
        name
        for group in optimizer.param_groups
        for parameter in group["params"]
        for name, candidate in built.named_parameters()
        if candidate is parameter
    }
    assert optimizer_names == named
    if not spec.cwt_temporal:
        assert not any(name.startswith("branches.w.temporal.") for name in named)
    if not spec.refinement:
        assert not any(name.startswith("base.refinement.") for name in named)


def test_shared_initialization_and_rng_are_paired_across_matrix():
    seed = sf.SEEDS[1]
    torch.manual_seed(77)
    start = torch.get_rng_state()
    models: dict[str, W0StructuralFactorialModel] = {}
    ending_states = []
    for arm in ARMS:
        torch.set_rng_state(start)
        models[arm] = W0StructuralFactorialModel(arm, seed)
        ending_states.append(torch.get_rng_state())
    assert all(torch.equal(ending_states[0], state) for state in ending_states[1:])

    shared_prefixes = ("base.local_blocks.", "branches.w.conv_", "branches.w.norm.", "branches.w.depthwise.",
                       "branches.w.parameter_fill.", "branches.w.final_projection.", "base.head.", "base.decoder_residual.")
    reference = models[ARMS[0]].state_dict()
    for arm, model in models.items():
        state = model.state_dict()
        for name, value in reference.items():
            if name.startswith(shared_prefixes):
                assert name in state and torch.equal(value, state[name]), (arm, name)

    for factor, prefix in (("cwt_temporal", "branches.w.temporal."), ("refinement", "base.refinement.")):
        enabled = [arm for arm in ARMS if getattr(ARM_SPECS[arm], factor)]
        source = models[enabled[0]].state_dict()
        for arm in enabled[1:]:
            state = models[arm].state_dict()
            assert all(torch.equal(value, state[name]) for name, value in source.items() if name.startswith(prefix))


@pytest.mark.parametrize("frontend", ["patch", "conv20"])
def test_same_frontend_four_bc_cells_have_identical_initial_waveform(frontend):
    seed = sf.SEEDS[0]
    x = torch.randn(1, 1, 18_000, generator=torch.Generator().manual_seed(5))
    w = torch.randn(1, 97, 360, generator=torch.Generator().manual_seed(6))
    outputs = []
    for arm in [name for name, spec in ARM_SPECS.items() if spec.frontend == frontend]:
        model = W0StructuralFactorialModel(arm, seed).eval()
        model.base.local_blocks = nn.ModuleList([nn.Identity() for _ in range(6)])
        with torch.no_grad():
            outputs.append(model(x, tf={"w": w})["waveform"])
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


def _nonzero_grad(module: nn.Module, suffix: str) -> bool:
    values = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if name.endswith(suffix) and parameter.grad is not None
    ]
    return bool(values) and any(bool(torch.isfinite(value).all() and value.ne(0).any()) for value in values)


def test_zero_init_gradient_startup_is_observed_over_three_updates():
    model = W0StructuralFactorialModel("sfv1_patch_tm3_ref2", sf.SEEDS[0]).train()
    branch = model.branches["w"]
    refinement = model.base.refinement
    optimizer = torch.optim.AdamW([*branch.parameters(), *refinement.parameters()], lr=1e-3, weight_decay=1e-4)
    latent = torch.randn(1, 96, 1_800, generator=torch.Generator().manual_seed(10))
    w = torch.randn(1, 97, 360, generator=torch.Generator().manual_seed(11))
    observations = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        gamma, beta = branch({"w": w})
        conditioned = latent * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * torch.tanh(beta)
        loss = refinement(conditioned).square().mean()
        loss.backward()
        observations.append(
            {
                "final": _nonzero_grad(branch.final_projection, "weight"),
                "tm_project": _nonzero_grad(branch.temporal, "project.weight"),
                "tm_depthwise": _nonzero_grad(branch.temporal, "depthwise.weight"),
                "ref_project": _nonzero_grad(refinement, "project.weight"),
                "ref_depthwise": _nonzero_grad(refinement, "depthwise.weight"),
            }
        )
        optimizer.step()
    assert observations[0] == {
        "final": True,
        "tm_project": False,
        "tm_depthwise": False,
        "ref_project": True,
        "ref_depthwise": False,
    }
    assert observations[1]["tm_project"] and observations[1]["ref_depthwise"]
    assert observations[2]["tm_depthwise"]


def _factor_seed_frame() -> pd.DataFrame:
    rows = []
    for seed in sf.SEEDS:
        for arm, spec in ARM_SPECS.items():
            a, b, c = spec.factors
            value = 10 + a + 2 * b + 3 * c + 4 * a * b + 5 * a * c + 6 * b * c + 7 * a * b * c
            rows.append({"arm": arm, "seed": seed, **{metric: float(value) for metric in sf.PRIMARY}})
    return pd.DataFrame(rows)


def test_factorial_contrasts_and_seed_aggregation_are_exact():
    frame = _factor_seed_frame()
    conditional = sf.conditional_effects_by_seed(frame)
    effects = sf.factorial_effects_by_seed(frame)
    across = sf.factorial_effects_across_seed(effects)
    summary = sf.arm_primary_summary(frame)
    assert len(conditional) == 3 * 5 * 3 * 4
    assert len(effects) == 3 * 5 * 7
    assert len(across) == 5 * 7
    assert len(summary) == 8 * 5
    expected = {"A": 7.25, "B": 8.75, "C": 10.25, "AB": 7.5, "AC": 8.5, "BC": 9.5, "ABC": 7.0}
    selected = effects[(effects.seed == sf.SEEDS[0]) & (effects.metric == sf.ERRORS[0])].set_index("effect")
    for effect, value in expected.items():
        assert selected.loc[effect, "raw_effect"] == pytest.approx(value)


def _window_metrics(rows_per_pair: int = 4) -> pd.DataFrame:
    rows = []
    for seed in sf.SEEDS:
        for arm_index, arm in enumerate(ARMS):
            for index in range(rows_per_pair):
                local_eligible = index != rows_per_pair - 1
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "dataset_row_id": index,
                        "samp_id": index % 2,
                        "split": "val",
                        "whole_rr_target_eligible": True,
                        "local_rr_target_eligible": local_eligible,
                        "joint_target_eligible": True,
                        "ibi_target_eligible": index % 2 == 0,
                        "ibi_interpretable": index == 0,
                        "envelope_spearman_target_eligible": True,
                        "joint_prediction_degenerate": False,
                        "envelope_spearman_prediction_degenerate": False,
                        sf.ERRORS[0]: 0.1 + arm_index * 0.001 + index,
                        sf.ERRORS[1]: (0.2 + arm_index * 0.001 + index) if local_eligible else np.nan,
                        sf.ERRORS[2]: 0.3 + arm_index * 0.001 + index,
                        sf.ERRORS[3]: 0.4 + arm_index * 0.001 + index,
                        sf.PCC: 0.8 - arm_index * 0.001 - index * 0.01,
                    }
                )
    return pd.DataFrame(rows)


def test_tail_subject_denominator_and_simplification_tables_are_complete():
    windows = _window_metrics()
    seed = sf.seed_primary_metrics(windows, expected_rows=4)
    tail = sf.local_rr_tail_summary(windows)
    denominators = sf.metric_denominators(windows)
    subjects = sf.subject_stratified_metrics(windows)
    subject_effects = sf.subject_factorial_effects(subjects)
    assert len(seed) == 24
    assert len(tail) == 24 and set(tail.eligible_n) == {3}
    assert len(denominators) == 24
    assert len(subjects) == 24 * 2 * 5
    assert len(subject_effects) == 3 * 2 * 5 * 7

    equal = _factor_seed_frame()
    for metric in sf.PRIMARY:
        equal[metric] = 1.0
    decisions = sf.quality_preserving_simplifications(equal)
    assert len(decisions) == 6
    assert decisions.quality_preserving.all()


def test_history_contract_accepts_epoch30_patience_stop(monkeypatch):
    monkeypatch.setattr(sf, "UPDATES_PER_EPOCH", 2)
    monkeypatch.setattr(sf, "PLANNED_UPDATES", 160)
    cfg = sf.derived_config(baseline(), arm=ARMS[0], output_root=Path("/tmp/sfv1"), device="cpu")
    rows = []
    best = float("inf")
    wait = 0
    for epoch in range(1, 31):
        value = 1.0 if epoch == 1 else 2.0
        improved, wait = sf._early_stopping_step(
            value=value,
            best=best,
            epochs_without_improvement=wait,
            min_delta=sf.EARLY_STOP_MIN_DELTA,
        )
        if improved:
            best = value
        triggered = sf._early_stopping_should_stop(
            epoch=epoch,
            min_epoch=sf.EARLY_STOP_MIN_EPOCH,
            epochs_without_improvement=wait,
            patience=sf.EARLY_STOP_PATIENCE,
        )
        rows.append(
            {
                "epoch": epoch,
                "optimizer_update": 2 * epoch,
                "train_loss_sync": 1.0,
                "train_loss_effort": 2.0,
                "train_loss_total": 1.5,
                "val_local_rr_mae": value,
                "first_learning_rate": crd_learning_rate(
                    2 * (epoch - 1),
                    total_updates=160,
                    max_learning_rate=cfg.training.max_learning_rate,
                    min_learning_rate=cfg.training.min_learning_rate,
                    warmup_fraction=cfg.training.warmup_fraction,
                ),
                "last_learning_rate": crd_learning_rate(
                    2 * epoch - 1,
                    total_updates=160,
                    max_learning_rate=cfg.training.max_learning_rate,
                    min_learning_rate=cfg.training.min_learning_rate,
                    warmup_fraction=cfg.training.warmup_fraction,
                ),
                "early_stopping_improved": int(improved),
                "early_stopping_wait": wait,
                "early_stopping_triggered": int(triggered),
                "early_stopping_min_epoch": 30,
            }
        )
    assert sf.validate_history(pd.DataFrame(rows), cfg) == 1


def test_exclusive_lifecycle_freezes_success_and_preserves_failure(tmp_path):
    lock_hash = "a" * 64
    with sf.exclusive_attempt(tmp_path / "success", phase="synthetic", lock_hash=lock_hash, arm=ARMS[0]) as output:
        (output / "result.txt").write_text("ok")
    completed = next((tmp_path / "success").glob("*/freeze_receipt.json")).parent
    manifest = json.loads((completed / "manifest.json").read_text())
    assert manifest["status"] == "completed" and "result.txt" in manifest["files"]

    with pytest.raises(RuntimeError, match="fixture"):
        with sf.exclusive_attempt(tmp_path / "failure", phase="synthetic", lock_hash=lock_hash) as output:
            (output / "partial.txt").write_text("keep")
            raise RuntimeError("fixture failure")
    failed = next((tmp_path / "failure").glob("*/lifecycle_failed.json")).parent
    assert (failed / "partial.txt").read_text() == "keep"
    assert not (failed / "freeze_receipt.json").exists()


def test_parameter_compute_report_and_critical_paths():
    report = sf.parameter_compute_report()
    assert set(report["arms"]) == set(ARMS)
    assert report["covered_macs"]["tm3"] == 201_657_600
    assert report["whole_model_flops"] is None
    assert all((sf.ROOT / path).is_file() for path in sf.critical_paths())


def test_prepare_lock_requires_clean_worktree(tmp_path, monkeypatch):
    monkeypatch.setattr(sf, "load_experiment_spec", lambda _path: object())
    monkeypatch.setattr(
        sf,
        "git_state",
        lambda _root: {"commit": "f" * 40, "branch": "fixture", "status_porcelain": " M file"},
    )
    with pytest.raises(RuntimeError, match="工作树干净"):
        sf.prepare_lock(tmp_path)
