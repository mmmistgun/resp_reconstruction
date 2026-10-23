"""合成/disposable生命周期验收；禁止读取真实样本或调用CUDA。"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from scripts import m4_s1_cwt_parameters_artifacts as io
from scripts import m4_s1_cwt_parameters_data as data_ops
from scripts import m4_s1_cwt_parameters_runtime as run
from scripts import m4_s1_cwt_parameters_spec as spec
from scripts import m4_s1_cwt_parameters_summary as summary
from scripts.run_m4_s1_cwt_parameters import main
from resp_train.crd.config import load_crd_config
from resp_train.data.factory import build_tho_data
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from m4_s1_fixtures import FixtureMamba, cpu_only


@pytest.fixture(scope="module")
def reps():
    return spec.representations()


@pytest.fixture
def sessions(monkeypatch, tmp_path, reps):
    # session/manifest测试不重复官方84次构造；初始化本身由模型测试完整覆盖。
    def audit(_):
        return {str(seed): {a: {"parameters": 1, "state_sha256": "fixture", "mamba": "fixture"}
                           for a in spec.ARMS} for seed in spec.SEEDS}
    monkeypatch.setattr(run, "audit_initialization", audit)
    monkeypatch.setattr(spec, "representations", lambda: deepcopy(reps))
    files = ("scripts/m4_s1_cwt_parameters_spec.py", "scripts/m4_s1_cwt_parameters_model.py")
    monkeypatch.setattr(io, "code_identity", lambda: {p: io.sha(spec.ROOT / p) for p in files})
    monkeypatch.setattr(spec, "OUTPUT_ROOT", tmp_path)
    return lambda name="session", scope="synthetic_only": run.prepare(tmp_path / name, scope=scope)


def test_plan_configs_and_three_shards(tmp_path):
    cells = spec.plan()
    assert len(cells) == len({c["cell"] for c in cells}) == 84
    shards = [spec.plan(i, 3) for i in range(3)]
    assert all(len(s) == 28 for s in shards)
    assert {c["cell"] for s in shards for c in s} == {c["cell"] for c in cells}
    old = load_crd_config(spec.ROOT / "configs/crd_tf_v1/crd_tf102_w_formal.yaml")
    for c in cells:
        cfg = spec.config(c["arm"], c["seed"], tmp_path)
        assert cfg.training.seed == cfg.model.initialization_seed == c["seed"]
        assert cfg.data.train_sample_seed == old.data.train_sample_seed
        assert cfg.data.val_sample_seed == old.data.val_sample_seed
        for section in ("loss", "evaluation", "window"):
            assert cfg[section] == old[section]
        assert cfg.training.batch_size == 32 and cfg.training.gradient_accumulation_steps == 4
        assert cfg.training.epochs == 80 and cfg.training.early_stopping_min_epoch == 30
        assert list(cfg.model.tf_representations) == [] and cfg.data.tf_cache_path is None
    with pytest.raises(ValueError):
        spec.plan(3, 3)
    with pytest.raises(ValueError):
        spec.config("B", 20260811, tmp_path)
    # 新协议14号seed开放不改变历史CRD入口的seed边界。
    with pytest.raises(ValueError, match="seed"):
        load_crd_config(spec.ROOT / "configs/crd_tf_v1/crd_tf102_w_formal.yaml",
                        overrides=["training.seed=20260814","model.initialization_seed=20260814"])


@pytest.mark.parametrize("section,key,value", [
    ("model","time_bias",True), ("representation","library","different"),
    ("uniform_tf_readout","retained",["encoder","output"]), ("training","epochs",81),
])
def test_scientific_spec_changes_require_explicit_implementation_contract(monkeypatch, tmp_path, section, key, value):
    cfg = yaml.safe_load(spec.SPEC.read_text())
    cfg[section][key] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    monkeypatch.setattr(spec, "SPEC", path)
    with pytest.raises(ValueError, match="合同"):
        spec.spec()


@pytest.mark.parametrize("args", [
    ["run", "--session", "/not-accessed"],
    ["prepare-data", "--session", "/not-accessed"],
    ["cache", "--session", "/not-accessed", "--all"],
    ["smoke", "--session", "/not-accessed"],
    ["prepare", "--session", "/not-accessed", "--scope", "train_validation"],
    ["run", "--session", "/not-accessed", "--confirm-training", "--arm", "B"],
    ["run", "--session", "/not-accessed", "--confirm-training", "--arm", "B", "--seed", "20260814", "--shard-count", "2"],
])
def test_cli_rejects_unconfirmed_or_ambiguous_execution(args, monkeypatch):
    monkeypatch.setattr(io, "load_session", lambda _: pytest.fail("不能触碰session"))
    monkeypatch.setattr(run, "require_cuda", lambda _: pytest.fail("不能调用CUDA"))
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_prepare_source_config_identity_and_scope_guards(sessions, monkeypatch):
    session = sessions()
    assert sessions() == session
    assert len(list((session / "resolved_configs").glob("*.yaml"))) == 84
    assert len(run.status(session)) == 84
    monkeypatch.setattr(data_ops, "audited_data", lambda _: pytest.fail("不能读取真实数据"))
    monkeypatch.setattr(run, "require_cuda", lambda _: pytest.fail("不能调用CUDA"))
    with pytest.raises(PermissionError):
        data_ops.prepare_data(session, confirmed=True)
    with pytest.raises(PermissionError):
        run.run_cell(session, "B", spec.SEEDS[0], "cuda:0", confirmed=True)
    with pytest.raises(RuntimeError, match="84"):
        run.summarize(session)
    with pytest.raises(ValueError, match="范围"):
        sessions(scope="train_validation")
    path = session / "resolved_configs" / f"B_seed{spec.SEEDS[0]}.yaml"
    path.write_text(path.read_text() + "\n# fixture tamper\n")
    with pytest.raises(ValueError, match="身份"):
        io.load_session(session)


def test_snapshot_and_session_tamper_rejected(sessions):
    session = sessions()
    (session / "source_snapshot.tar.gz").write_bytes(b"synthetic tamper")
    with pytest.raises(ValueError, match="身份"):
        io.load_session(session)
    other = sessions("other")
    payload = io.read_json(other / "session.json")
    payload["scope"] = "train_validation"
    (other / "session.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="身份"):
        io.require_data_scope(other)


def test_stage_failure_retry_success_immutable(tmp_path):
    parent = tmp_path / "stage"
    key = {"fixture": "stage"}
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with io.attempt(parent, key) as attempt:
            (attempt / "partial.txt").write_text("preserved")
            raise RuntimeError("synthetic failure")
    failed = list(parent.glob("attempt_*/failed.json"))[0].parent
    before = {p.name: io.sha(p) for p in failed.iterdir()}
    with pytest.raises(FileExistsError, match="retry"):
        with io.attempt(parent, key):
            pytest.fail("未授权覆盖失败产物")
    with io.attempt(parent, key, retry_failed=True) as success:
        (success / "result.txt").write_text("synthetic success")
    assert io.completed(parent, key) == success
    assert before == {p.name: io.sha(p) for p in failed.iterdir()}
    with pytest.raises(FileExistsError, match="成功"):
        with io.attempt(parent, key, retry_failed=True):
            pytest.fail("已完成阶段不能重跑")
    with pytest.raises(ValueError, match="越界"):
        io.artifact_path(success, "../outside")
    (success / "result.txt").write_text("changed")
    with pytest.raises(ValueError, match="身份"):
        io.completed(parent, key)


def test_actual_synthetic_cwt_calibration(sessions):
    session = sessions()
    output = run.calibrate(session)
    assert run.calibrate(session) == output
    record = io.read_json(output / "calibration.json")
    assert record["passed"] and record["subsets_and_pooling_consistent"]
    assert len(record["representations"]) == 20


@pytest.mark.parametrize("arm", ["A_UniformTFReadout", "A_NoPoolPosition"])
def test_two_update_task_loss_cpu_smoke(sessions, monkeypatch, arm):
    session = sessions()
    calibration = session / "fixture_calibration"
    calibration.mkdir()
    io.write_json(calibration / "manifest.json", {"synthetic_fixture": True})
    monkeypatch.setattr(run, "require_calibration", lambda _: calibration)
    cfg = replace(run.architecture(), dimension=32, waveform_channels=16, condition_channels=8, layers=1)
    output = run.smoke(session, arm, "cpu", confirmed=True, mamba_factory=FixtureMamba, batch_size=1, cfg_override=cfg)
    receipt = io.read_json(output / "engineering.json")
    assert receipt["fixture"] and receipt["passed"] and receipt["updates"] == 2 and receipt["accumulation"] == 4
    assert receipt["batch_size"] == 1 and receipt["mamba"] == "fixture"
    monkeypatch.setattr(run, "require_cuda", lambda _: (_ for _ in ()).throw(ValueError("CUDA guard")))
    with pytest.raises(ValueError, match="CUDA guard"):
        run.ensure_smoke(session, arm, "cuda:0")


@pytest.fixture
def disposable_data(sessions, monkeypatch, tmp_path):
    rows = []
    for split, ids, subject in (("train", [10, 11, 12], 101), ("val", [20, 21], 201)):
        source = tmp_path / f"{split}.npz"
        t = np.arange(18000 * len(ids), dtype=np.float32) / 100
        np.savez(source, x=np.sin(2*np.pi*2*t), y=np.sin(2*np.pi*.25*t))
        for i, rid in enumerate(ids):
            rows.append({"dataset_row_id": rid, "split": split, "samp_id": subject, "coupling_state_id": 1,
                "source_npz": str(source), "target_source_npz": str(source), "window_start_s": i*180.,
                "window_end_s": (i+1)*180., "hard_valid_ratio": 1., "state_alignment_valid_ratio": 1.,
                "allowed_losses": "waveform", "state_alignment_method": "fixture", "reason": "",
                "supervision_confidence_level": "high"})
    # 索引可包含test元数据，但任何test波形访问都会因文件不存在而失败。
    rows.append({**rows[-1], "dataset_row_id":99, "split":"test", "samp_id":999,
                 "source_npz":str(tmp_path/"forbidden_test.npz"), "target_source_npz":str(tmp_path/"forbidden_test.npz")})
    index = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(index, index=False)
    original = spec.config
    def fixture_config(arm, seed, output, device="cpu", **kwargs):
        cfg = original(arm, seed, output, device, **kwargs)
        cfg.data.dataset_root, cfg.data.index_csv = str(tmp_path), "index.csv"
        cfg.data.bcg_input_key, cfg.data.target_key = "x", "y"
        return cfg
    monkeypatch.setattr(spec, "config", fixture_config)
    monkeypatch.setattr(spec, "COUNTS", {"train": 3, "val": 2})
    monkeypatch.setattr(spec, "SUBJECTS", {"train": 1, "val": 1})
    lock = tmp_path / "fixture_lock.json"
    io.write_json(lock, {"dataset_index": {"path": str(index), "sha256": io.sha(index)},
        "cache_lock": {"row_identity": {f"{s}_row_content_sha256": data_ops.array_hash(np.array(ids, dtype=np.int64))
                                       for s, ids in (("train", [10, 11, 12]), ("val", [20, 21]))}}})
    monkeypatch.setattr(run.legacy, "P2_LOCK_PATH", lock)
    monkeypatch.setattr(run.legacy, "P2_LOCK_SHA256", io.sha(lock))
    monkeypatch.setattr(data_ops, "optimizer_updates_per_epoch", lambda *_: 80)
    session = sessions(scope="train_validation")
    return session


def test_disposable_data_cache_and_native_batch_alignment(disposable_data, tmp_path):
    session = disposable_data
    data_path = data_ops.prepare_data(session, confirmed=True)
    assert data_ops.prepare_data(session, confirmed=True) == data_path
    cache = data_ops.build_cache(session, "B", confirmed=True)
    assert data_ops.build_cache(session, "B", confirmed=True) == cache
    cfg = spec.config("B", spec.SEEDS[0], tmp_path)
    output = tmp_path / "access"
    output.mkdir()
    data = data_ops.training_data(session, "B", cfg, output)
    sample = data.val.dataset[0]
    assert sample["tf"]["w"].shape == (97, 360) and sample["meta"]["dataset_row_id"] == 20
    expected = run.transform(sample["x"].numpy().reshape(-1), io.load_session(session)["representations"]["B"])
    np.testing.assert_array_equal(sample["tf"]["w"].numpy(), expected)
    no_tf_output = tmp_path / "no_tf_access"
    no_tf_output.mkdir()
    no_tf = data_ops.training_data(session, "A_NoTF", spec.config("A_NoTF", spec.SEEDS[0], tmp_path), no_tf_output)
    assert "tf" not in no_tf.val.dataset[0]
    assert set(no_tf.train.rows.samp_id).isdisjoint(no_tf.val.rows.samp_id)
    # 改写仅限一次性fixture；生产入口应检测来源字节变化。
    payload = io.read_json(data_path / "data.json")
    assert not any("forbidden_test" in p for p in payload["source_files"])
    target = next(iter(payload["source_files"]))
    with Path(target).open("ab") as stream:
        stream.write(b"fixture tamper")
    with pytest.raises(ValueError, match="身份"):
        data_ops.data_receipt(session)


def test_cache_reader_rejects_wrong_rows_test_split_and_nonfinite(disposable_data):
    session = disposable_data
    data_ops.prepare_data(session, confirmed=True)
    cache = data_ops.build_cache(session, "C_0p8", confirmed=True)
    rep = io.load_session(session)["representations"]["C_0p8"]
    with pytest.raises(ValueError, match="train/validation"):
        data_ops.CacheReader(cache, "test", rep, np.array([20,21]))
    with pytest.raises(ValueError, match="row"):
        data_ops.CacheReader(cache, "val", rep, np.array([21,20], dtype=np.int64))
    reader = data_ops.CacheReader(cache, "val", rep, np.array([20,21], dtype=np.int64))
    with pytest.raises(ValueError, match="row"):
        reader.get(0, 21)
    # 已打开reader逐样本也检有限性，防止manifest核验后发生缓存损坏。
    writable = np.load(cache / "val_w.npy", mmap_mode="r+")
    writable[0, 0, 0] = np.nan
    writable.flush()
    with pytest.raises(FloatingPointError):
        reader.get(0, 20)
    with pytest.raises(ValueError, match="身份"):
        data_ops.cache_receipt(session, "C_0p8")


def test_native_validation_export_checks_public_target(disposable_data, tmp_path, monkeypatch):
    session = disposable_data
    data_ops.prepare_data(session, confirmed=True)
    cfg = spec.config("A_NoTF", spec.SEEDS[0], tmp_path, "cpu")
    access = tmp_path/"export_access"
    access.mkdir()
    data = data_ops.training_data(session,"A_NoTF",cfg,access)
    class WaveFixture(torch.nn.Module):
        def forward(self,x,**kwargs):
            t = torch.arange(18000,device=x.device,dtype=torch.float32)/100
            return {"waveform":torch.sin(2*torch.pi*.25*t)[None,None].expand(len(x),1,-1)}
    output = tmp_path/"export"
    output.mkdir()
    result = run.export_validation(session,"A_NoTF",spec.SEEDS[0],cfg,WaveFixture(),data,output)
    assert list(result.dataset_row_id) == [20,21] and set(result.split) == {"val"}
    original = run.collect_predictions
    def wrong_target(*args,**kwargs):
        values = original(*args,**kwargs)
        values["tho_ref"] = values["tho_ref"] + .1
        return values
    monkeypatch.setattr(run,"collect_predictions",wrong_target)
    with pytest.raises(ValueError,match="公共参考"):
        run.export_validation(session,"A_NoTF",spec.SEEDS[0],cfg,WaveFixture(),data,tmp_path/"unused")


def test_source_snapshot_covers_runtime_dependencies():
    files = io.code_identity()
    required = {
        "scripts/run_m4_s1_cwt_parameters.py", "scripts/m4_s1_cwt_parameters_model.py",
        "scripts/m4_linear_fusion_model.py", "scripts/p1_components_model.py",
        "scripts/m4_pool_position_model.py", "resp_train/patch_aligned_tf_training.py",
        "resp_train/paper_evidence/cwt_time_frequency_v1/features.py",
        "resp_train/crd/tf_v1_features.py", "resp_train/data/research_v2.py",
        "resp_train/metrics/task.py", "tests/m4_s1_fixtures.py",
        "configs/m4_s1_cwt_parameters_v1/design.yaml",
    }
    assert required <= files.keys()


def test_read_only_legacy_cache_import_and_provenance_rejection(disposable_data, tmp_path):
    from resp_train.paper_evidence.cwt_time_frequency_v1 import artifacts as old
    import shutil
    session = disposable_data
    data_path = data_ops.prepare_data(session, confirmed=True)
    metadata = io.read_json(data_path/"data.json")
    frozen = io.load_session(session)
    source = tmp_path/"legacy_source"
    source.mkdir()
    feature_paths = ("resp_train/paper_evidence/cwt_time_frequency_v1/features.py",
                     "resp_train/crd/tf_v1_features.py", "resp_train/data/research_v2.py")
    io.write_json(source/"session.json", {"protocol":old.PROTOCOL, "research_test_open":False,
        "representations":frozen["representations"], "provenance":{"dependencies":io.dependencies(),
            "files":{p:io.identity(spec.ROOT/p) for p in feature_paths}}})
    io.write_json(source/"session_receipt.json",io.identity(source/"session.json"))
    old_data = source/"data"/"attempt_fixture"
    old_data.mkdir(parents=True)
    for name in ("data.json","train_rows.csv","val_rows.csv"):
        shutil.copyfile(data_path/name,old_data/name)
    def finish(path,binding):
        io.write_json(path/"manifest.json", {"protocol":old.PROTOCOL,"status":"completed",
            "binding":binding,"files":{p.name:io.identity(p) for p in path.iterdir()}})
        io.write_json(path/"receipt.json",io.identity(path/"manifest.json"))
    finish(old_data,old.binding(source,"data"))
    cache = source/"cache"/"C_0p8"/"attempt_fixture"
    cache.mkdir(parents=True)
    rep = frozen["representations"]["C_0p8"]
    for split,n in spec.COUNTS.items():
        np.save(cache/f"{split}_w.npy",np.ones((n,*rep["shape"]),dtype=np.float32))
        ids = pd.read_csv(data_path/f"{split}_rows.csv").dataset_row_id.to_numpy(np.int64)
        np.save(cache/f"{split}_row_ids.npy",ids)
    io.write_json(cache/"cache.json",{"representation":rep,"row_order":metadata["row_order"],
                                    "splits":["train","val"],"finite":True})
    finish(cache,old.binding(source,"cache",arm="C_0p8",data=io.sha(old_data/"manifest.json")))
    before = {str(p.relative_to(source)):io.sha(p) for p in source.rglob("*") if p.is_file()}
    target = data_ops.import_cache(session,"C_0p8",source,confirmed=True)
    assert data_ops.cache_receipt(session,"C_0p8") == (target,cache)
    assert not list(target.glob("*.npy"))
    assert before == {str(p.relative_to(source)):io.sha(p) for p in source.rglob("*") if p.is_file()}
    bad = deepcopy(metadata)
    bad["target_key"] = "wrong_target"
    with pytest.raises(ValueError,match="target_key"):
        data_ops.verify_legacy_cache(source,"C_0p8",rep,bad,data_path)
    altered = np.load(cache/"val_w.npy",mmap_mode="r+")
    altered[0,0,0] = 3
    altered.flush()
    with pytest.raises(ValueError,match="漂移"):
        data_ops.cache_receipt(session,"C_0p8")


def test_reviewed_dataset_source_pair_raw_path_equivalence(disposable_data, tmp_path, monkeypatch):
    import hashlib
    source = tmp_path/"reviewed_source"
    relative = data_ops.REVIEWED_DATA_SOURCE["path"]
    current = (spec.ROOT/relative).read_text()
    # 还原已审阅的历史字节对；BCG-CWT合入后的公共加载器不能自动取得旧缓存准入。
    bcg_guard = '''        if (str(cfg.get("model", {}).get("name", "")) == "bcg_cwt_patch_mamba_v1"
                and bool(cfg.data.get("drop_nonfinite_windows", False))):
            raise ValueError("BCG-CWT 要求 drop_nonfinite_windows=false，非有限样本必须显式失败")
'''
    bcg_config = '''        bcg_cwt = str(model_cfg.get("name", "")) == "bcg_cwt_patch_mamba_v1"
        if bcg_cwt and (not tf_cache_path or list(representations) != ["w"]):
            raise ValueError("BCG-CWT 数据要求 tf_cache_path 和 tf_representations=[w]")
'''
    bcg_branch = '''            if bcg_cwt:
                from resp_train.data.bcg_cwt import FrozenWCWTReader

                self._tf_v1_cache = FrozenWCWTReader(str(tf_cache_path), split=split)
            elif patch_tf:
'''
    for block, replacement in ((bcg_guard, ""), (bcg_config, ""),
                               (bcg_branch, "            if patch_tf:\n")):
        assert current.count(block) == 1
        current = current.replace(block, replacement)
    assert hashlib.sha256(current.encode()).hexdigest() == data_ops.REVIEWED_DATA_SOURCE["current_sha256"]
    added = '''        patch_tf = str(model_cfg.get("name", "")) == "patch_aligned_tf_mamba"
        if patch_tf and (not tf_cache_path or list(representations) != ["w"]):
            raise ValueError("Patch-aligned TF 数据要求冻结 tf_cache_path 和 tf_representations=[w]")
'''
    branch = '''            if patch_tf:
                from resp_train.data.patch_aligned_tf import FrozenHCWTReader

                self._tf_v1_cache = FrozenHCWTReader(str(tf_cache_path), split=split)
            elif split == "test" and research_test_cache_path:
'''
    assert current.count(added) == current.count(branch) == 1
    previous = current.replace(added,"").replace(branch,'            if split == "test" and research_test_cache_path:\n')
    assert hashlib.sha256(previous.encode()).hexdigest() == data_ops.REVIEWED_DATA_SOURCE["old_sha256"]
    path = source/"source_snapshot"/relative
    path.parent.mkdir(parents=True)
    path.write_text(previous)
    files = {p:io.identity(spec.ROOT/p) for p in (
        "resp_train/paper_evidence/cwt_time_frequency_v1/features.py","resp_train/crd/tf_v1_features.py")}
    files[relative] = io.identity(path)
    cfg = data_ops.raw_config(tmp_path/"unused")
    with pytest.raises(ValueError,match="身份"):
        data_ops.verify_preprocessing_sources(source,files,cfg)
    # 生产校验在一次性历史源码根上运行，哈希常量和核验实现均保持原样。
    reviewed_root = tmp_path/"reviewed_current"
    for name in files:
        destination = reviewed_root/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(current.encode() if name == relative else (spec.ROOT/name).read_bytes())
    with monkeypatch.context() as patch:
        patch.setattr(spec,"ROOT",reviewed_root)
        changes = data_ops.verify_preprocessing_sources(source,files,cfg)
        assert changes == [data_ops.REVIEWED_DATA_SOURCE]
        bad_cfg = deepcopy(cfg)
        bad_cfg.model.name = "patch_aligned_tf_mamba"
        with pytest.raises(ValueError,match="原始波形"):
            data_ops.verify_preprocessing_sources(source,files,bad_cfg)
        bad = deepcopy(files)
        bad[relative]["sha256"] = "unreviewed"
        with pytest.raises(ValueError,match="身份"):
            data_ops.verify_preprocessing_sources(source,bad,cfg)
    namespace = {"__name__":"reviewed_dataset_fixture"}
    exec(compile(previous,str(path),"exec"),namespace)
    bundles = build_tho_data(cfg)
    for bundle in (bundles.train,bundles.val):
        old_dataset = namespace["ResearchV2WindowDataset"](bundle.index_path,bundle.rows,cfg,preload_windows=False)
        for i in range(len(old_dataset)):
            old_item,current_item = old_dataset[i],bundle.dataset[i]
            torch.testing.assert_close(old_item["x"],current_item["x"],atol=0,rtol=0)
            torch.testing.assert_close(old_item["target"],current_item["target"],atol=0,rtol=0)
            for key,value in old_item["meta"].items():
                new = current_item["meta"][key]
                if isinstance(value,torch.Tensor):
                    torch.testing.assert_close(value,new,atol=0,rtol=0)
                elif isinstance(value,float) and np.isnan(value):
                    assert np.isnan(new)
                else:
                    assert value == new


def test_training_failure_is_preserved_and_resume_explicit(sessions, monkeypatch):
    session = sessions(scope="train_validation")
    monkeypatch.setattr(run, "require_cuda", lambda _: None)
    monkeypatch.setattr(run, "require_calibration", lambda _: None)
    monkeypatch.setattr(run.legacy, "runtime", lambda _: {"torch":"fixture","cuda":None,"packages":{},
        "gpu":"cpu_fixture","LD_LIBRARY_PATH_unset":True,"LD_PRELOAD_unset":True})
    monkeypatch.setattr(run, "ensure_smoke", lambda *_: (_ for _ in ()).throw(RuntimeError("fixture engineering failed")))
    with pytest.raises(RuntimeError, match="engineering failed"):
        run.run_cell(session, "B", spec.SEEDS[0], "cuda:0", confirmed=True)
    cell = session / "cells" / f"B_seed{spec.SEEDS[0]}"
    failed = list(cell.glob("attempts/*/failed.json"))
    assert len(failed) == 1
    digest = io.sha(failed[0])
    with pytest.raises(FileExistsError, match="resume"):
        run.run_cell(session, "B", spec.SEEDS[0], "cuda:0", confirmed=True)
    with pytest.raises(RuntimeError, match="engineering failed"):
        run.run_cell(session, "B", spec.SEEDS[0], "cuda:0", confirmed=True, resume=True)
    assert len(list(cell.glob("attempts/*/failed.json"))) == 2 and io.sha(failed[0]) == digest


def test_epoch_resume_matches_uninterrupted_and_earliest_tie(monkeypatch, tmp_path):
    cfg = spec.config("B", spec.SEEDS[0], tmp_path, "cpu")
    cfg.training.epochs = 5
    cfg.training.early_stopping_min_epoch = 4
    cfg.training.early_stopping_patience = 2
    current = {"epoch": 0, "interrupt": False}
    def train(model, loader, loss, optimizer, cfg, update, total, epoch):
        current["epoch"] = epoch
        optimizer.zero_grad(set_to_none=True)
        objective = (model.weight * torch.rand_like(model.weight)).sum()
        objective.backward()
        optimizer.step()
        return {"loss_total": float(objective.detach())}, update + 80
    def validation(*args, **kwargs):
        if current["interrupt"] and current["epoch"] == 2:
            raise RuntimeError("fixture interruption")
        return {"loss_total": 1.}, {"epoch": current["epoch"]}
    monkeypatch.setattr(run.legacy, "_train_epoch", train)
    monkeypatch.setattr(run.legacy, "validate", validation)
    monkeypatch.setattr(run.legacy, "validation_local_rr_mean", lambda p, _: {1:1.,2:.5,3:.5,4:.7}[p["epoch"]])
    data = SimpleNamespace(train=SimpleNamespace(loader=[None]*318), val=SimpleNamespace(loader=[]))
    def make():
        run.legacy.set_seed(7)
        model = torch.nn.Linear(2, 1, bias=False)
        return model, torch.optim.AdamW(model.parameters(), lr=.01)
    complete, interrupted = tmp_path / "complete", tmp_path / "interrupted"
    complete.mkdir(); interrupted.mkdir()
    left, opt = make()
    full = run.legacy.run_epochs(left, data, None, opt, cfg, complete)
    current["interrupt"] = True
    right, opt = make()
    with pytest.raises(RuntimeError, match="interruption"):
        run.legacy.run_epochs(right, data, None, opt, cfg, interrupted)
    first = run.legacy.completed_epochs(interrupted)[0][1]
    original = {p.name: io.sha(p) for p in first.iterdir()}
    current["interrupt"] = False
    resumed = run.legacy.run_epochs(right, data, None, opt, cfg, interrupted, resume=True)
    torch.testing.assert_close(left.weight, right.weight, atol=0, rtol=0)
    assert full["best_epoch"] == resumed["best_epoch"] == 2
    assert full["completed_epochs"] == resumed["completed_epochs"] == 4
    assert resumed["optimizer_updates"] == 320
    assert original == {p.name: io.sha(p) for p in first.iterdir()}
    assert len(run.verify_training_result(interrupted, resumed, cfg)) == 4
    with pytest.raises(ValueError, match="history"):
        run.verify_training_result(interrupted, {**resumed, "best_epoch":3}, cfg)


def metrics_fixture():
    rows = []
    for cell in spec.plan():
        arm, seed = cell["arm"], cell["seed"]
        # 三个不等权窗口使窗口均值和受试者宏平均不同。
        for rid, sid, offset in ((10,101,0.), (11,101,0.), (20,201,2.)):
            extra = .001 * (seed - spec.SEEDS[0])
            if arm in spec.POSITION:
                s, v = spec.POSITION[arm]
                extra += .01*s + .02*v + .03*s*v
            row = {"arm":arm, "seed":seed, "dataset_row_id":rid, "samp_id":sid, "split":"val",
                "whole_rr_target_eligible":True, "local_rr_target_eligible":True,
                "local_rr_target_eligible_windows":9, "joint_target_eligible":True,
                "envelope_spearman_target_eligible":True, "joint_prediction_degenerate":False,
                "envelope_spearman_prediction_degenerate":False}
            row.update({m: .8-extra-offset*.01 if m == sf.PCC else .2+extra+offset for m in sf.PRIMARY})
            rows.append(row)
    return pd.DataFrame(rows)


def test_full_matrix_paired_effects_macro_and_failure_cases():
    frame = metrics_fixture()
    tables = summary.summary_tables(frame)
    assert len(tables["per_seed"]) == 84*5
    main = tables["paired_delta"].query("group == 'baseline'")
    assert len(main) == 27*3*5
    effects = tables["effects"]
    value = effects.query("effect == 'pool_position_interaction' and metric == 'local_rr_mae_bpm'").value
    np.testing.assert_allclose(value, .03, atol=1e-15)
    main_value = effects.query("effect == 'pool_position_score_main' and metric == 'local_rr_mae_bpm'").value
    np.testing.assert_allclose(main_value, .025, atol=1e-15)
    assert len(tables["effect_summary"]) == 9*5
    index = ["arm","seed","metric"]
    delta = tables["subject_macro"].set_index(index)["mean"] - tables["per_seed"].set_index(index)["mean"]
    np.testing.assert_allclose(delta.loc[("B",spec.SEEDS[0],"local_rr_mae_bpm")], 1/3)
    with pytest.raises(ValueError, match="84"):
        summary.summary_tables(frame.iloc[:-3])
    broken = frame.copy()
    broken.loc[0,"dataset_row_id"] = 999
    with pytest.raises(ValueError, match="身份"):
        summary.summary_tables(broken)
    broken = frame.copy()
    broken.loc[0,"local_rr_mae_bpm"] = np.nan
    with pytest.raises(FloatingPointError):
        summary.summary_tables(broken)


def test_summarize_complete_fixture_and_read_only_reuse(sessions, monkeypatch):
    session = sessions()
    frame = metrics_fixture()
    monkeypatch.setattr(spec, "COUNTS", {"train":3,"val":3})
    monkeypatch.setattr(spec, "SUBJECTS", {"train":2,"val":2})
    monkeypatch.setattr(run, "summarize_task_metrics", lambda f: pd.DataFrame({"n_samples":[len(f)]}))
    for cell in spec.plan():
        directory = session / "cells" / cell["cell"]
        directory.mkdir(parents=True)
        subset = frame[(frame.arm == cell["arm"]) & (frame.seed == cell["seed"])]
        subset.to_csv(directory/"metrics.csv", index=False)
        for name in ("config.yaml","environment.json","checkpoint.pt","history.csv"):
            (directory/name).write_text("disposable fixture")
        io.write_json(directory/"resource.json", {"current_attempt_training_seconds":1.,
            "all_epoch_train_validation_selection_seconds":1., "peak_allocated_bytes":100,
            "peak_reserved_bytes":200, "cwt_train_validation_array_bytes":None})
        result = {"arm":cell["arm"],"seed":cell["seed"],"session_sha256":io.sha(session/"session.json"),
            "best_epoch":1,"completed_epochs":30,"optimizer_updates":2400,
            "best_local_rr":float(subset.local_rr_mae_bpm.mean()),"early_stopped":True,
            "parameters":1,"metrics":"metrics.csv","best_checkpoint":"checkpoint.pt","final_checkpoint":"checkpoint.pt",
            "history":"history.csv","resource":"resource.json","artifacts":{p.name:io.identity(p) for p in directory.iterdir()}}
        io.write_json(directory/"completed.json",result)
    output = run.summarize(session)
    before = {p.name:io.sha(p) for p in output.iterdir() if p.is_file()}
    assert run.summarize(session) == output
    assert before == {p.name:io.sha(p) for p in output.iterdir() if p.is_file()}
    assert len(pd.read_csv(output/"per_seed.csv")) == 420
    assert len(list(output.glob("*.png"))) == 2 and not list(output.glob("*.pdf"))
    assert (output/"report.md").is_file()
