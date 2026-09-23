from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from resp_fusion.model import ARMS, Fusion, FusionModel, ModelConfig
from resp_fusion.config import load_experiment_config, model_config, SEEDS, PROTOCOL
from resp_fusion.artifacts import artifact_directory, read_json, write_json, sha256_file, verified_manifest
from resp_fusion import data as data_module
from resp_fusion.data import build_cache, build_loaders, CacheReader, select_rows, verified_cache
from resp_fusion.experiment import train, evaluate_validation, PRIMARY, comparison_identity
from resp_fusion.summary import factorial_contrasts, summarize_runs
from resp_fusion.stopping import StoppingState
from resp_train.aligned_dual_view.features import CWTBatch, spec_digest, prepare_cwt_batch
from resp_train.metrics.task import summarize_task_metrics


class CPUBlock(nn.Module):
    """仅用于 CPU 接线测试；原生 Mamba 由独立 GPU 命令验收。"""
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(kwargs["d_model"]))

    def forward(self, x):
        return x * self.scale


def factory(cfg):
    return FusionModel(cfg, mamba_factory=CPUBlock)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("method", ["concat", "film", "attention"])
def test_identity_then_condition_and_projection_gradients(method):
    model = Fusion(method)
    h = torch.randn(2, 7, 64, requires_grad=True)
    v = torch.randn(2, 7, 64, requires_grad=True)
    target = torch.randn_like(h)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    output = model(h, v)
    torch.testing.assert_close(output, h, rtol=0, atol=0)
    (output-target).square().mean().backward()
    assert h.grad.abs().sum() > 0 and v.grad.abs().sum() == 0
    if method == "attention":
        assert model.output.weight.grad.abs().sum() > 0
        for name in ("query", "key", "value"):
            assert getattr(model, name).weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    h.grad = None
    v.grad = None
    (model(h, v)-target).square().mean().backward()
    assert h.grad.abs().sum() > 0 and v.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_attention_two_tokens_query_dependence_and_temporal_locality():
    module = Fusion("attention")
    h, v = torch.randn(2, 9, 64), torch.randn(2, 9, 64)
    _, weights = module.attention(h, v)
    assert weights.shape == (2, 9, 4, 2)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 9, 4))
    assert (weights[..., 0]-weights[..., 1]).abs().max() > .01
    with torch.no_grad():
        module.query.weight.mul_(2)
        module.output.weight.copy_(torch.eye(64))
    changed_q = module.attention(h, v)[1]
    assert not torch.allclose(weights, changed_q)
    before = module(h, v)
    changed = v.clone()
    changed[:, 4] += torch.randn(2, 64)
    after = module(h, changed)
    keep = [i for i in range(9) if i != 4]
    torch.testing.assert_close(before[:, keep], after[:, keep], rtol=0, atol=0)
    assert not torch.allclose(before[:, 4], after[:, 4])


def test_six_arms_shared_initialization_function_and_other_seed():
    models = {arm: factory(ModelConfig(arm=arm)) for arm in ARMS}
    first = models["A"].state_dict()
    x = torch.randn(1, 1, 18000)
    v = torch.randn(1, 97, 1800)
    outputs = []
    for arm, model in models.items():
        for name, value in model.state_dict().items():
            if not name.startswith("fusion."):
                assert torch.equal(value, first[name]), (arm, name)
        with torch.no_grad():
            output = model(x, tf={"adv_cwt": v})
        assert output["waveform"].shape == (1, 1, 18000)
        assert output["waveform_10hz"].shape == (1, 1, 1800)
        assert torch.isfinite(output["waveform"]).all()
        outputs.append(output["waveform"])
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=1e-6, atol=1e-6)
    for pre, post in (("A", "B"), ("C", "D"), ("E", "F")):
        for name, value in models[pre].fusion.state_dict().items():
            assert torch.equal(value, models[post].fusion.state_dict()[name])
    other = factory(ModelConfig(initialization_seed=20260812))
    assert not torch.equal(other.readout.weight, models["A"].readout.weight)


@pytest.mark.parametrize("arm", list(ARMS))
def test_full_model_position_and_delayed_cwt_gradients(arm):
    model = factory(ModelConfig(arm=arm))
    order = []
    hooks = [model.fusion.register_forward_pre_hook(lambda *args: order.append("fusion"))]
    hooks += [block.register_forward_pre_hook(lambda *args: order.append("block")) for block in model.blocks]
    hooks += [model.readout.register_forward_pre_hook(lambda *args: order.append("readout"))]
    x, v = torch.randn(1, 1, 18000), torch.randn(1, 97, 1800)
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(x, tf={"adv_cwt": v})
        output["waveform"].square().mean().backward()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        condition = model.scale_projection[0].weight.grad.abs().sum()
        assert condition == 0 if step == 0 else condition > 0
        assert model.waveform_encoder[0].weight.grad.abs().sum() > 0
        if step == 1 and model.method == "attention":
            for name in ("query", "key", "value"):
                assert getattr(model.fusion, name).weight.grad.abs().sum() > 0
        optimizer.step()
    expected = (["fusion"]+["block"]*6 if model.position == "pre" else ["block"]*6+["fusion"])+["readout"]
    assert order == expected*2
    for hook in hooks:
        hook.remove()


def test_shape_representation_nonfinite_and_formal_boundaries():
    model = factory(ModelConfig())
    x, v = torch.randn(1, 1, 18000), torch.randn(1, 97, 1800)
    for bad in (v[..., :360], v.double()):
        with pytest.raises(ValueError):
            model(x, tf={"adv_cwt": bad})
    with pytest.raises(ValueError, match="身份"):
        model(x, cwt=CWTBatch(v, "wrong"))
    with pytest.raises(FloatingPointError):
        model(x*float("nan"), tf={"adv_cwt": v})
    with pytest.raises(FloatingPointError):
        model(x, tf={"adv_cwt": v*float("inf")})
    for override in ("training.early_stopping_enabled=false", "training.early_stopping_patience=20",
                     "training.early_stopping_min_epochs=20", "training.early_stopping_min_delta=0.001",
                     "data.val_split=test", "loss.effort_weight=0.5",
                     "training.batch_size=64", "training.epochs=40", "model.arm=G", "model.depth=4"):
        with pytest.raises(ValueError):
            load_experiment_config(overrides=[override])
    with pytest.raises(Exception):
        load_experiment_config(overrides=["data.test_split=test"])
    for arm in ARMS:
        assert load_experiment_config(Path(__file__).parents[1]/f"configs/adv_fusion_factorial_v1/{arm}.yaml").model.arm == arm


@pytest.fixture
def cfg(tmp_path):
    root = tmp_path / "dataset"
    (root / "training").mkdir(parents=True)
    time = np.arange(19000)/100
    rows = []
    for split, subject in (("train", 11), ("val", 22)):
        x = ((1+.3*np.sin(2*np.pi*.2*time))*np.sin(2*np.pi*4*time)).astype(np.float32)
        target = ((1+.3*np.sin(2*np.pi*.015*time))*np.sin(2*np.pi*.2*time+.1)).astype(np.float32)
        np.savez(root/f"{split}_input.npz", bcg=x)
        np.savez(root/f"{split}_target.npz", target=target, tho_bad_sec=np.zeros(190, dtype=np.uint8))
        for start in (0,10):
            rows.append({"dataset_row_id":len(rows)+1,"split":split,"samp_id":subject,"coupling_state_id":1,
                "window_start_s":start,"window_end_s":start+180,"source_npz":f"../{split}_input.npz",
                "target_source_npz":f"../{split}_target.npz","bcg_rawish_segment_soft_z_key":"bcg",
                "target_waveform_segment_soft_z_key":"target","hard_valid_ratio":1.,"state_alignment_valid_ratio":1.,
                "allowed_losses":"waveform","state_alignment_method":"constant_shift","reason":""})
    rows.append({**rows[0],"dataset_row_id":99,"split":"test","samp_id":33,
                 "source_npz":"../must_not_open.npz","target_source_npz":"../must_not_open_target.npz"})
    pd.DataFrame(rows).to_csv(root/"training/dataset_index.csv",index=False)
    return load_experiment_config(overrides=["protocol.run_role=smoke",f"data.dataset_root={root}",
        "data.max_train_windows=2","data.max_val_windows=2","training.epochs=2","training.batch_size=1",
        "training.gradient_accumulation_steps=2","training.device=cpu","training.use_amp=false","training.show_progress=false"])


def test_synthetic_cache_training_validation_and_reuse_checks(cfg, tmp_path, monkeypatch):
    actual = np.load
    accessed = []
    def guarded(path, *args, **kwargs):
        accessed.append(str(path))
        assert "must_not_open" not in str(path)
        return actual(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded)
    # 使用实际 ADV CWT，验证新的缓存协议与既有数学表示连接。
    cache = build_cache(cfg, tmp_path/"cache")
    assert not any("target.npz" in name for name in accessed)
    verified_cache(cache)
    cfg.model.arm = "F"
    run = train(cfg, cache_root=cache, output=tmp_path/"train", model_factory=factory)
    m = verified_manifest(run, "train")
    assert m["epochs_completed"] == 2 and m["arm"] == "F"
    assert pd.read_csv(run/"history.csv").optimizer_update.tolist() == [1,2]
    out = evaluate_validation(run_root=run, cache_root=cache, output=tmp_path/"validation", model_factory=factory)
    pd.testing.assert_frame_equal(pd.read_csv(run/"metrics.csv"), pd.read_csv(out/"metrics.csv"))
    with pytest.raises(FileExistsError):
        train(cfg, cache_root=cache, output=run, model_factory=factory)
    selection = select_rows(cfg)
    reader = CacheReader(cache, selection, split="val")
    item = build_loaders(cfg, cache)[1]["val"].dataset[0]
    with pytest.raises(ValueError, match="输入内容"):
        reader.get(0, identity=reader.records[0]["identity"], waveform=item["x"]+1, include_features=True)
    monkeypatch.setattr(data_module, "transform_identity", lambda: {"drift":True})
    with pytest.raises(ValueError, match="前处理"):
        CacheReader(cache, selection, split="val")


def test_factorial_contrast_sign_and_complete_matrix():
    rows=[]
    for arm,(method,position) in ARMS.items():
        for seed in SEEDS:
            # 方式效应2、3；位置效应5；交互7、11。
            number={"concat":0,"film":2,"attention":3}[method]
            if position=="post":
                number += 5+{"concat":0,"film":7,"attention":11}[method]
            rows.append({"arm":arm,"seed":seed,**{metric:number for metric in PRIMARY}})
    scores=pd.DataFrame(rows)
    result=factorial_contrasts(scores)
    assert len(result)==240
    for pair,want in (("film-concat",7),("attention-concat",11),("attention-film",4)):
        picked=result[(result.kind=="interaction") & result.contrast.str.startswith(f"({pair})")]
        assert len(picked)==15 and (picked.delta==want).all()
    with pytest.raises(ValueError): factorial_contrasts(scores.iloc[:-1])


def formal_fixture(root, arm, seed):
    cfg=load_experiment_config(overrides=[f"model.arm={arm}",f"training.seed={seed}"])
    with artifact_directory(root, kind="train", cfg=cfg) as (root, started):
        samples={split:[{"identity":{"dataset_row_id":i,"split":split},"input_sha256":"fixture-input",
                         "target_sha256":"fixture-target","rr_peak_mask_sha256":"fixture-mask"} for i in range(n)]
                 for split,n in (("train",10141),("val",2675))}
        write_json(root/"samples.json",samples)
        shared={"readout.weight":f"shared-seed-{seed}"}
        state={**shared,"fusion.weight":f"{ARMS[arm][0]}-{seed}"}
        write_json(root/"initialization.json",{"state":state,"shared_sha256":spec_digest(shared)})
        write_json(root/"model_report.json",{"synthetic_fixture":True})
        (root/"checkpoints").mkdir()
        # 混合合法的早停与满预算来源，验证同一停止政策不要求相同停止epoch。
        values = [1-epoch*.001 for epoch in range(80)] if arm == "F" else [1.]*30
        stopping=StoppingState()
        for value in values: stopping.step(value)
        last, best = stopping.epoch, stopping.best_epoch
        for ep in set((best,last)):
            (root/f"checkpoints/epoch_{ep:03d}.pt").write_bytes(b"disposable-checksum-fixture")
        pd.DataFrame({"epoch":range(1,last+1),"optimizer_update":range(80,80*last+1,80),
                      "val_local_rr_mae":values}).to_csv(root/"history.csv",index=False)
        metrics=pd.DataFrame({metric:np.full(2675,.5) for metric in PRIMARY})
        for name in ("whole_rr_target_eligible","local_rr_target_eligible","joint_target_eligible"): metrics[name]=True
        metrics["dataset_row_id"]=range(2675)
        metrics["split"]="val"
        metrics["method"]=f"adv_fusion_v1_{arm}"
        metrics.to_csv(root/"metrics.csv",index=False)
        summarize_task_metrics(metrics).to_csv(root/"metrics_summary.csv",index=False)
        identity={"protocol":PROTOCOL,"config_sha256":spec_digest(OmegaConf.to_container(cfg,resolve=True)),
                  "code_sha256":started["code"]["sha256"],"samples":{k:spec_digest(v) for k,v in samples.items()},
                  "cache_id":"fixture-cache","cache_manifest_sha256":"fixture-cache-sha"}
        manifest={"protocol":PROTOCOL,"kind":"train","identity":identity,"arm":arm,"seed":seed,
                  "method":ARMS[arm][0],"position":ARMS[arm][1],"run_role":"formal","epochs_completed":last,
                  "optimizer_updates_completed":80*last,"schedule_total_updates":6400,"updates_per_epoch":80,
                  "stopping":stopping.receipt(),
                  "selected_epoch":best,"selected_local_rr_mae":stopping.best_value,"test_waveform_read":False,
                  "selected_checkpoint":f"checkpoints/epoch_{best:03d}.pt","final_checkpoint":f"checkpoints/epoch_{last:03d}.pt",
                  "shared_initialization_sha256":spec_digest(shared),
                  "comparison_id":comparison_identity(cfg,identity["samples"],identity["cache_id"],identity["code_sha256"],started["dependencies"])}
        for filename,field in (("metrics.csv","metrics_sha256"),("metrics_summary.csv","summary_sha256"),
            ("history.csv","history_sha256"),("config.yaml","config_sha256"),("samples.json","samples_sha256"),
            ("initialization.json","initialization_sha256"),("model_report.json","model_report_sha256"),
            (manifest["selected_checkpoint"],"selected_sha256"),(manifest["final_checkpoint"],"final_sha256")):
            manifest[field]=sha256_file(root/filename)
        write_json(root/"manifest.json",manifest)
    return root


def test_full_18_summary_provenance_and_corruption(tmp_path):
    roots=[formal_fixture(tmp_path/f"{arm}_{seed}",arm,seed) for arm in ARMS for seed in SEEDS]
    with pytest.raises(ValueError): summarize_runs(roots[:-1],output=tmp_path/"partial")
    with pytest.raises(ValueError): summarize_runs([roots[0]]*18,output=tmp_path/"duplicate")
    output=summarize_runs(roots,output=tmp_path/"summary")
    verified_manifest(output,"summary")
    for name,count in (("per_seed",18),("across_seed",30),("contrasts_per_seed",240),("contrasts_across_seed",80)):
        assert len(pd.read_csv(output/f"{name}.csv"))==count
    scores=pd.read_csv(output/"per_seed.csv")
    assert set(scores.epochs_completed)=={30,80}
    assert set(scores.stop_reason)=={"max_epochs","patience_exhausted"}
    # 仅破坏 disposable fixture，生产来源只读。
    with (roots[-1]/"metrics.csv").open("a") as handle: handle.write("corrupt\n")
    with pytest.raises(ValueError,match="哈希"):
        summarize_runs(roots,output=tmp_path/"corrupt_summary")
    assert (tmp_path/"corrupt_summary/failed.json").exists()


def test_rehashed_metadata_does_not_hide_initialization_or_row_mismatch(tmp_path):
    roots=[formal_fixture(tmp_path/f"{arm}_{seed}",arm,seed) for arm in ARMS for seed in SEEDS]
    source=roots[3]  # 同 seed 的后融合拼接，与 A 必须配对。
    original_init=(source/"initialization.json").read_bytes()
    original_manifest=(source/"manifest.json").read_bytes()
    original_receipt=(source/"completed.json").read_bytes()
    def replace_fixture(filename,payload):
        # 本测试只改写临时构造的 JSON，用于验证内容语义而非只检查文件哈希。
        (source/filename).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    init=read_json(source/"initialization.json")
    init["state"]["readout.weight"]="different-shared-seed"
    init["shared_sha256"]=spec_digest({k:v for k,v in init["state"].items() if not k.startswith("fusion.")})
    replace_fixture("initialization.json",init)
    m=read_json(source/"manifest.json")
    m["initialization_sha256"]=sha256_file(source/"initialization.json")
    m["shared_initialization_sha256"]=init["shared_sha256"]
    replace_fixture("manifest.json",m)
    receipt=read_json(source/"completed.json")
    receipt["manifest_sha256"]=sha256_file(source/"manifest.json")
    replace_fixture("completed.json",receipt)
    with pytest.raises(ValueError,match="配对初始化"):
        summarize_runs(roots,output=tmp_path/"wrong_initialization")
    (source/"initialization.json").write_bytes(original_init)
    (source/"manifest.json").write_bytes(original_manifest)
    (source/"completed.json").write_bytes(original_receipt)
    metrics=pd.read_csv(source/"metrics.csv")
    metrics.loc[0,"dataset_row_id"]=999999
    metrics.to_csv(source/"metrics.csv",index=False)
    m=read_json(source/"manifest.json")
    m["metrics_sha256"]=sha256_file(source/"metrics.csv")
    replace_fixture("manifest.json",m)
    receipt=read_json(source/"completed.json")
    receipt["manifest_sha256"]=sha256_file(source/"manifest.json")
    replace_fixture("completed.json",receipt)
    with pytest.raises(ValueError,match="metrics row"):
        summarize_runs(roots,output=tmp_path/"wrong_rows")


def test_failed_lifecycle_and_historical_cache_allowlist(cfg,tmp_path,monkeypatch):
    import resp_fusion.experiment as exp
    from resp_fusion.engineering import gpu_synthetic
    with pytest.raises(ValueError,match="confirm-gpu"):
        gpu_synthetic(cfg,tmp_path/"gpu")
    assert not (tmp_path/"gpu").exists()
    with pytest.raises(ValueError,match="数据源"):
        build_cache(cfg,Path(cfg.data.dataset_root)/"output")
    monkeypatch.setattr(data_module,"extract_cwt_10hz",lambda x:np.repeat(x[None,::10],97,axis=0).copy())
    cache=build_cache(cfg,tmp_path/"cache")
    def fail(*args,**kwargs): raise FloatingPointError("synthetic gradient failure")
    monkeypatch.setattr(exp,"train_crd_one_epoch",fail)
    with pytest.raises(FloatingPointError):
        train(cfg,cache_root=cache,output=tmp_path/"failed",model_factory=factory)
    assert (tmp_path/"failed/failed.json").exists() and not (tmp_path/"failed/completed.json").exists()
    m=read_json(cache/"manifest.json")
    m["protocol"]="aligned-dual-view-v1-train-val-20260920"
    (cache/"manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError,match="旧缓存"):
        verified_cache(cache)


@pytest.mark.parametrize("improvement_epoch,expected_stop",[(None,30),(20,35),(30,45)])
def test_early_stopping_minimum_patience_and_ties(improvement_epoch,expected_stop):
    state=StoppingState()
    for epoch in range(1,expected_stop+1):
        value=.5 if improvement_epoch is not None and epoch>=improvement_epoch else 1.
        improved=state.step(value)
        assert improved == (epoch==1 or epoch==improvement_epoch)
        assert (state.reason is not None) == (epoch==expected_stop)
    receipt=state.receipt()
    assert receipt["reason"]=="patience_exhausted"
    assert receipt["best_epoch"]==(improvement_epoch or 1)
    assert receipt["stopped_epoch"]==expected_stop
    with pytest.raises(ValueError,match="不能追加"): state.step(0.)


def test_early_stopping_full_budget_nonfinite_and_unfinished():
    state=StoppingState()
    with pytest.raises(ValueError,match="未满足"): state.receipt()
    with pytest.raises(FloatingPointError): state.step(float("nan"))
    assert state.epoch==0
    for epoch in range(1,81): state.step(1/epoch)
    receipt=state.receipt()
    assert receipt["best_epoch"]==80 and receipt["reason"]=="max_epochs" and not receipt["stopped_early"]


def test_summary_rejects_false_stopping_receipt(tmp_path):
    roots=[formal_fixture(tmp_path/f"{arm}_{seed}",arm,seed) for arm in ARMS for seed in SEEDS]
    source=roots[0]
    manifest=read_json(source/"manifest.json")
    manifest["stopping"]["patience"]=20
    (source/"manifest.json").write_text(json.dumps(manifest))
    receipt=read_json(source/"completed.json")
    receipt["manifest_sha256"]=sha256_file(source/"manifest.json")
    (source/"completed.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="停止原因"):
        summarize_runs(roots,output=tmp_path/"false_stop")


def test_trainer_stops_and_saves_terminal_checkpoint_with_fixed_lr_horizon(cfg,tmp_path,monkeypatch):
    import resp_fusion.experiment as exp
    monkeypatch.setattr(data_module,"extract_cwt_10hz",lambda x:np.repeat(x[None,::10],97,axis=0).copy())
    cache=build_cache(cfg,tmp_path/"cache")
    selection,loaders=build_loaders(cfg,cache)
    formal=load_experiment_config()
    # 只替换数据与计算引擎，使用真实的formal停止合同、训练循环和checkpoint生命周期。
    monkeypatch.setattr(exp,"build_loaders",lambda *args,**kwargs:(selection,loaders))
    monkeypatch.setattr(exp,"_device",lambda _:torch.device("cpu"))
    calls=[]
    def epoch_step(*args,**kwargs):
        calls.append((kwargs["epoch"],kwargs["total_updates"],kwargs["total_epochs"]))
        return {"loss":.5},kwargs["update_index"]+1
    monkeypatch.setattr(exp,"train_crd_one_epoch",epoch_step)
    monkeypatch.setattr(exp,"validate",lambda *args,**kwargs:({"loss":.5},{}))
    monkeypatch.setattr(exp,"_check_predictions",lambda *args:None)
    monkeypatch.setattr(exp,"validation_local_rr_mean",lambda *args:1.)
    monkeypatch.setattr(exp,"collect_predictions",lambda *args,**kwargs:{})
    monkeypatch.setattr(exp,"_write_metrics",lambda *args:{"synthetic_engine_fixture":True})
    output=train(formal,cache_root=cache,output=tmp_path/"early_run",model_factory=factory)
    manifest=verified_manifest(output,"train")
    assert calls==[(epoch,80,80) for epoch in range(1,31)]
    assert manifest["epochs_completed"]==30 and manifest["optimizer_updates_completed"]==30
    assert manifest["schedule_total_updates"]==80  # fixture每epoch只有1次更新，原计划仍保留80epoch。
    assert manifest["stopping"]["reason"]=="patience_exhausted"
    assert manifest["selected_checkpoint"]=="checkpoints/epoch_001.pt"
    assert manifest["final_checkpoint"]=="checkpoints/epoch_030.pt"
    assert (output/manifest["final_checkpoint"]).exists()
    assert len(pd.read_csv(output/"history.csv"))==30
