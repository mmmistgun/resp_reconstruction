"""APOR表示矩阵的synthetic定向测试；CPU前向显式使用Identity主干fixture。"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from resp_train.paper_evidence.patch_apor_v1_model import PatchAporModel
from resp_train.paper_evidence.cwt_apor_v2.spec import ARMS, SEEDS, config, load_spec, plan
from resp_train.paper_evidence.cwt_apor_v2.model import build_model, GN_NAMES, PARAMETERS
from resp_train.paper_evidence.cwt_apor_v2.interventions import captured_forward, paired_batch, CONDITIONS
from resp_train.paper_evidence.cwt_apor_v2.runtime import selected_epoch
from resp_train.paper_evidence.cwt_apor_v2.reuse import canonical_representation
from resp_train.crd.training import crd_learning_rate


def rep(pool=50, scales=97):
    return {"arm": {"name": "A0", "pool_samples": pool}, "shape": [scales, 18000//pool]}


def cpu_model(spec):
    model = build_model(SEEDS[0], spec)
    # 本fixture不验证原生Mamba kernel；GPU验收是独立后续阶段。
    model.base.local_blocks = nn.ModuleList([nn.Identity() for _ in range(6)])
    return model.eval()


def test_matrix_and_early_stopping_config(tmp_path):
    assert len(ARMS) == 20 and len(plan()) == 60 and "A0" in ARMS and "B" not in ARMS
    assert load_spec()["model_family"] == "APOR_A0"
    for arm in ARMS:
        cfg = config(arm, SEEDS[0], tmp_path / arm)
        assert cfg.training.epochs == 80
        assert cfg.training.early_stopping_enabled
        assert (cfg.training.early_stopping_min_epoch, cfg.training.early_stopping_patience, cfg.training.early_stopping_min_delta) == (30,15,0)
        assert cfg.model.variant == "apor_a0_cwt_native"


@pytest.mark.parametrize("seed", SEEDS)
def test_baseline_parameters_buffers_and_nonzero_film_match(seed):
    native = PatchAporModel("A0", seed)
    candidate = build_model(seed, rep())
    assert sum(p.numel() for p in candidate.parameters()) == PARAMETERS
    assert candidate.state_dict().keys() == native.state_dict().keys()
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
    with torch.no_grad():
        native.branches["w"].final_projection.weight.normal_(0, .01)
        candidate.load_state_dict(native.state_dict())
    for model in (native, candidate):
        model.base.local_blocks = nn.ModuleList([nn.Identity() for _ in range(6)])
        model.eval()
    x, w = torch.randn(1,1,18000), torch.rand(1,97,360)
    with torch.no_grad():
        torch.testing.assert_close(native(x, tf={"w":w})["waveform"], candidate(x, tf={"w":w})["waveform"], rtol=0, atol=0)


@pytest.mark.parametrize("pool,scales", [(25,97),(50,33),(50,193),(100,97)])
def test_condition_sampling_physical_coordinates_and_shape(pool, scales):
    model = cpu_model(rep(pool, scales))
    branch = model.branches["w"]
    frames = 18000//pool
    times = (pool-1)/2 + torch.arange(frames)*pool
    sampled = branch.sample(times[None,None].float())
    positions = torch.arange(140)[:,None]*128+torch.linspace(0,255,5)[None,:]
    expected = positions.clamp(float(times[0]), float(times[-1]))
    torch.testing.assert_close(sampled[0,0], expected, rtol=1e-6, atol=.002)
    w = torch.rand(1,scales,frames, requires_grad=True)
    with torch.no_grad():
        branch.final_projection.weight.fill_(.001)
    gamma, beta = branch({"w":w})
    assert gamma.shape == beta.shape == (1,96,140)
    (gamma.square().mean()+beta.square().mean()).backward()
    assert torch.isfinite(w.grad).all() and w.grad.ne(0).any()


def test_apor_capture_and_single_gn_paired_replay():
    model = cpu_model(rep())
    assert tuple(name for name, m in model.branches["w"].named_modules() if isinstance(m, nn.GroupNorm)) == GN_NAMES == ("norm",)
    x, w = torch.randn(1,1,18000), torch.rand(1,97,360)
    with torch.no_grad():
        model.branches["w"].final_projection.weight.normal_(0,.005)
    frequencies = np.geomspace(.0366,7.995,97)
    shifts = torch.tensor([[60,100,200]])
    observed = []
    for condition, prediction, capture, *_ in paired_batch(model,x,w,frequencies,shifts):
        assert prediction.shape == (1,1,18000)
        assert capture.z.shape == capture.z_prime.shape == (1,96,140)
        observed.append(condition)
    assert observed == list(CONDITIONS)
    assert not model.patch_head._forward_pre_hooks
    assert not model.branches["w"].norm._forward_hooks


def history_fixture(run, values):
    cfg = config("A0", SEEDS[0], run)
    OmegaConf.save(cfg, run / "config.yaml")
    records, best, wait = [], float("inf"), 0
    for epoch, value in enumerate(values,1):
        improved = value < best
        if improved:
            best, wait = value, 0
        else:
            wait += 1
        records.append({"epoch":epoch, "optimizer_update":epoch*80, "train_loss_total":1.25,
                        "train_loss_sync":1., "train_loss_effort":1., "val_local_rr_mae":value,
                        "first_learning_rate": crd_learning_rate((epoch-1)*80,total_updates=6400,max_learning_rate=3e-4,min_learning_rate=3e-5),
                        "last_learning_rate": crd_learning_rate(epoch*80-1,total_updates=6400,max_learning_rate=3e-4,min_learning_rate=3e-5),
                        "early_stopping_improved":int(improved), "early_stopping_wait":wait,
                        "early_stopping_triggered":int(epoch>=30 and wait>=15), "early_stopping_min_epoch":30})
    frame = pd.DataFrame(records)
    frame.to_csv(run / "train_history.csv", index=False)
    return frame


def test_early_stop_earliest_tie_and_fixed_lr_budget(tmp_path):
    values = [*np.linspace(2,1,15), *([1.]*15)]
    frame = history_fixture(tmp_path,values)
    assert selected_epoch(tmp_path) == 15
    assert frame.iloc[-1].last_learning_rate > 3e-5
    frame.iloc[:-1].to_csv(tmp_path / "train_history.csv", index=False)
    with pytest.raises(ValueError):
        selected_epoch(tmp_path)


def test_maximum_80_epochs_is_still_valid(tmp_path):
    frame = history_fixture(tmp_path,np.linspace(2,1,80))
    assert selected_epoch(tmp_path) == 80
    assert frame.iloc[-1].last_learning_rate == pytest.approx(3e-5)


def test_cache_identity_only_renames_baseline():
    original = {"arm":{"name":"B","pool_samples":50}, "shape":[97,360], "frequencies_hz":[.04,8.]}
    frozen = deepcopy(original)
    new = canonical_representation(original)
    assert original == frozen and new["arm"]["name"] == "A0"
    assert new["shape"] == original["shape"] and new["frequencies_hz"] == original["frequencies_hz"]


def test_w0_engineering_cannot_authorize_apor_training():
    from resp_train.paper_evidence.cwt_apor_v2.formal_launch import audit_reuse
    source = {"spec":{"protocol":"cwt-time-frequency-v1-20260930"}, "provenance":{"files":{}, "dependencies":{}}}
    with pytest.raises(ValueError, match="矩阵或依赖"):
        audit_reuse(source,{})


def test_checkpoint_validation_accepts_early_stop_at_30(tmp_path):
    from resp_train.crd.training import build_crd_optimizer
    from resp_train.paper_evidence.cwt_apor_v2.runtime import verify_checkpoints
    from resp_train.paper_evidence.cwt_apor_v2.spec import PROTOCOL
    history_fixture(tmp_path,[*np.linspace(2,1,15), *([1.]*15)])
    cfg = OmegaConf.load(tmp_path / "config.yaml")
    cfg.model.cwt_representation = rep()
    OmegaConf.save(cfg,tmp_path / "config.yaml")
    model = build_model(SEEDS[0],rep())
    optimizer, _ = build_crd_optimizer(model,cfg)
    for name, epoch in (("checkpoint_best_local_rr.pt",15),("checkpoint_final.pt",30)):
        for p in model.parameters():
            optimizer.state[p] = {"step":torch.tensor(float(epoch*80)), "exp_avg":torch.zeros_like(p), "exp_avg_sq":torch.zeros_like(p)}
        torch.save({"epoch":epoch, "config":OmegaConf.to_container(cfg,resolve=True),
                    "model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict(),
                    "extra_state":{"protocol":PROTOCOL,"update_index":epoch*80,"total_updates":6400}},tmp_path/name)
    assert verify_checkpoints(tmp_path,cfg) == 15
