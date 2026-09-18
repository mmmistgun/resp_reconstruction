"""E4 validation 频带诊断的独立来源锁、生命周期及原生指标入口。"""
from __future__ import annotations

import json
import shutil
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.crd.model import build_crd_model
from resp_train.data.factory import build_window_data
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from resp_train.paper_evidence import e4_aggregation_v2 as source
from resp_train.paper_evidence.e4_aggregation_v2_model import ARMS as CANDIDATES, build_model
from resp_train.paper_evidence.e4_band_audit import CONDITIONS, PRIMARY, PCC, AuditModel, paired_changes, training_reference
from resp_train.paper_evidence.e4_band_audit_store import FeatureStore, feature_reader
from resp_train.paper_evidence.e1_scale_topology_runtime import identity, sha256_file, write_json, git_state, checked_batches
from resp_train.paper_evidence.e1_scale_topology import array_hash, SEEDS
from resp_train.paper_evidence.e4_scale_aggregation_engineering import synthetic_batch

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "e4-band-encoding-aggregation-v1-20260918"
OUTPUT = Path("runs/e4_band_encoding_aggregation_v1")
LOCK = Path("docs/experiments/e4_band_encoding_aggregation_lock_20260918.json")
DOC = Path("docs/experiments/e4_band_encoding_aggregation_protocol_20260918.md")
SCRIPT = Path("scripts/eval_e4_band_encoding_aggregation.py")
TEST = Path("tests/test_e4_band_audit.py")
ARMS = ("W0_FULL", *CANDIDATES)
COUNT = 2675


def prepare_lock(summary):
    if (ROOT/LOCK).exists():
        raise FileExistsError("频带诊断锁已存在")
    training, digest = source.load_lock()
    summary = summary.resolve()
    source.verify_attempt(summary, phase="summary", lock_hash=digest)
    receipt = json.loads((summary/"summary_receipt.json").read_text())
    if set(receipt["source_runs"]) != {f"{a}/{s}" for a in CANDIDATES for s in SEEDS}:
        raise ValueError("需要完整 E4-v2 validation 矩阵")
    entries = []
    rows = None
    for arm in ARMS:
        for seed in SEEDS:
            if arm == "W0_FULL":
                old = next(e for e in training["w0_entries"] if e["seed"] == seed)
                run, epoch = ROOT/old["run_dir"], old["selected_epoch"]
                source.verify(run/"checkpoint_best_local_rr.pt", old["checkpoint"])
                source.verify(run/"config.yaml", old["config"])
                source.verify(run/"metrics_summary.csv", old["validation_summary"])
                source.verify(run/"metrics.csv", training["source_files"][str((run/"metrics.csv").relative_to(ROOT))])
            else:
                item = receipt["source_runs"][f"{arm}/{seed}"]
                attempt = Path(item["path"])
                source.verify_attempt(attempt, phase="formal", lock_hash=digest)
                source.verify(attempt/"manifest.json", item["manifest"])
                formal = json.loads((attempt/"formal_receipt.json").read_text())
                run, epoch = attempt/formal["run_dir"], item["selected_epoch"]
                candidate_rows = pd.read_csv(attempt/"val_rows.csv")
                if rows is None:
                    rows = candidate_rows
                else:
                    pd.testing.assert_frame_equal(rows, candidate_rows)
            cfg = OmegaConf.to_container(OmegaConf.load(run/"config.yaml"), resolve=True)
            files = {name: {"path": str(run/name), **identity(run/name)} for name in
                     ("checkpoint_best_local_rr.pt", "config.yaml", "metrics.csv", "metrics_summary.csv")}
            entries.append(dict(arm=arm, seed=seed, selected_epoch=epoch, files=files, config=cfg))
    if len(rows) != COUNT or rows.samp_id.nunique() != 7 or rows.dataset_row_id.duplicated().any():
        raise ValueError("validation 行合同漂移")
    probes = []
    for _, frame in rows.groupby("samp_id", sort=True):
        frame = frame.sort_values(["segment_id", "window_start_s", "dataset_row_id"])
        probes.extend(int(frame.iloc[i].dataset_row_id) for i in (0, (len(frame)-1)//2, len(frame)-1))
    cache = training["cache_lock"]
    cache_files = {}
    for name in ("manifest", "train_w", "val_w", "frequency_file"):
        item = cache[name]
        cache_files[name] = {"path": str(ROOT/item["path"]), "size_bytes": item["size_bytes"],
                             "sha256": item.get("sha256", item.get("file_sha256"))}
    for split in ("train", "val"):
        path = ROOT/cache["root"]/f"{split}_row_ids.npy"
        info = identity(path)
        if info["sha256"] != cache["row_identity"][f"{split}_row_file_sha256"]:
            raise ValueError("cache row identity 漂移")
        cache_files[f"{split}_rows"] = {"path": str(path), **info}
    code = sorted((ROOT/"resp_train").rglob("*.py")) + [ROOT/p for p in (DOC, SCRIPT, TEST)]
    lock = dict(protocol=PROTOCOL, conditions=list(CONDITIONS), arms=list(ARMS), seeds=list(SEEDS), count=COUNT,
        entries=entries, validation_rows=rows[["dataset_row_id", "samp_id", "split"]].to_dict("records"),
        validation_row_hash=array_hash(rows.dataset_row_id.to_numpy()), plot_row_ids=probes,
        frequencies=training["frequency"]["values_hz"], cache_files=cache_files,
        dataset_index=training["dataset_index"], training_lock={"path": str(ROOT/source.LOCK_PATH), **identity(ROOT/source.LOCK_PATH)},
        validation_summary={"path": str(summary/"manifest.json"), **identity(summary/"manifest.json")},
        code_files={str(p.relative_to(ROOT)): identity(p) for p in code}, preparation_git=git_state(ROOT),
        prepared_at=datetime.now(timezone.utc).isoformat())
    write_json(ROOT/LOCK, lock)
    return ROOT/LOCK


def load_lock():
    lock = json.loads((ROOT/LOCK).read_text())
    if (lock["protocol"] != PROTOCOL or lock["conditions"] != list(CONDITIONS) or lock["arms"] != list(ARMS)
            or lock["seeds"] != list(SEEDS) or lock["count"] != COUNT
            or [(e["arm"],e["seed"]) for e in lock["entries"]] != [(a,s) for a in ARMS for s in SEEDS]):
        raise ValueError("频带诊断矩阵漂移")
    for path, info in lock["code_files"].items():
        source.verify(ROOT/path, info)
    for key in ("training_lock", "validation_summary"):
        source.verify(Path(lock[key]["path"]), lock[key])
    return lock, sha256_file(ROOT/LOCK)


@contextmanager
def attempt(phase, digest, arm=None, seed=None):
    parent = ROOT/OUTPUT/phase
    if arm is not None:
        parent = parent/arm/f"seed_{seed}"
    with source.phase_guard(parent, digest):
        for frozen in parent.glob("*/freeze_receipt.json"):
            old = json.loads((frozen.parent/"manifest.json").read_text())
            if old["implementation_lock_sha256"] == digest:
                raise FileExistsError(f"同身份已完成: {frozen.parent}")
        out = parent/f"{phase}_{digest[:12]}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:12]}"
        out.mkdir(parents=True, exist_ok=False)
        context = dict(protocol=PROTOCOL, phase=phase, arm=arm, seed=seed, implementation_lock_sha256=digest,
                       command=sys.argv, started_at=datetime.now(timezone.utc).isoformat())
        write_json(out/"lifecycle_started.json", context)
        print(f"E4 band audit attempt: {out}", flush=True)
        try:
            yield out
            write_json(out/"lifecycle_completed.json", dict(context, status="completed"))
            write_json(out/"manifest.json", dict(context, status="completed", files={str(p.relative_to(out)): identity(p)
                for p in sorted(out.rglob("*")) if p.is_file()}))
            write_json(out/"freeze_receipt.json", {"manifest": identity(out/"manifest.json")})
        except BaseException as exc:
            write_json(out/"lifecycle_failed.json", dict(context, error=str(exc), traceback=traceback.format_exc()))
            raise


def verify_attempt(path, digest, phase):
    path = Path(path).resolve()
    if (path/"lifecycle_failed.json").exists():
        raise ValueError("attempt 已失败")
    source.verify(path/"manifest.json", json.loads((path/"freeze_receipt.json").read_text())["manifest"])
    manifest = json.loads((path/"manifest.json").read_text())
    if (manifest["protocol"] != PROTOCOL or manifest["phase"] != phase or manifest["implementation_lock_sha256"] != digest
            or manifest["status"] != "completed"):
        raise ValueError("attempt identity 漂移")
    for rel, info in manifest["files"].items():
        target = (path/rel).resolve()
        if not target.is_relative_to(path):
            raise ValueError("manifest 路径越界")
        source.verify(target, info)
    return manifest


def prepare_reference():
    lock, digest = load_lock()
    with attempt("reference", digest) as out:
        write_json(out/"environment.json", git_state(ROOT, require_clean=True))
        write_json(out/"access_started.json", {"scope": "frozen train W array only", "reference": "per-scale mean over all train windows and time"})
        for key in ("train_w", "train_rows", "manifest"):
            item = lock["cache_files"][key]; source.verify(Path(item["path"]), item)
        values = np.load(lock["cache_files"]["train_w"]["path"], mmap_mode="r", allow_pickle=False)
        if values.shape != (10141,97,360):
            raise ValueError("train W shape 漂移")
        reference = training_reference(values)
        np.save(out/"reference.npy", reference, allow_pickle=False)
        write_json(out/"reference.json", {"train_count": 10141, "time_count": 360, "reference_hz_order": lock["frequencies"],
            "method": "float64 sum; float32 fixed per-scale mean", "alternative": "zero", "source": lock["cache_files"]["train_w"],
            "max_abs_mean_minus_zero": float(np.abs(reference).max()), "references_identical": bool(np.count_nonzero(reference)==0)})
    return out


def load_model(entry, device):
    for item in entry["files"].values():
        source.verify(Path(item["path"]), item)
    cfg = OmegaConf.create(entry["config"])
    checkpoint = torch.load(entry["files"]["checkpoint_best_local_rr.pt"]["path"], map_location="cpu", weights_only=False)
    if checkpoint["config"] != entry["config"] or checkpoint["epoch"] != entry["selected_epoch"]:
        raise ValueError("checkpoint config/selected epoch 不符")
    source.finite_tree(checkpoint["model_state_dict"])
    model = build_crd_model(cfg) if entry["arm"] == "W0_FULL" else build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    cfg.training.device, cfg.training.show_progress = device, False
    return model.to(device).eval(), cfg


def gpu_smoke(device):
    lock, digest = load_lock()
    with attempt("gpu_smoke", digest) as out:
        environment = source.runtime_preflight(device)
        write_json(out/"environment.json", environment)
        write_json(out/"access_started.json", {"inputs": "synthetic", "checkpoints": "15 selected", "real_inputs": False})
        records = []
        for entry in lock["entries"]:
            model, cfg = load_model(entry, device)
            for batch_size in ([1,128] if entry["seed"] == SEEDS[0] else [1]):
                batch = synthetic_batch(batch_size, 20260918, device)
                captured = {}
                def observe(offset, values):
                    captured["x"] = values["x"].cpu()
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = model(batch["x"], tf=batch["tf"])
                    wrapper = AuditModel(model,"FULL",observe=observe).eval()
                    actual = wrapper(batch["x"],tf=batch["tf"])
                    for key in actual:
                        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
                    cached = AuditModel(model,"FULL",fixed_features=lambda offset,count:captured["x"][offset:offset+count]).eval()
                    replay = cached(batch["x"],tf=batch["tf"])
                    for key in replay:
                        torch.testing.assert_close(replay[key],expected[key],rtol=1e-5,atol=1e-6)
                    for condition in CONDITIONS[1:]:
                        fixed = (lambda offset,count: captured["x"][offset:offset+count]) if condition in CONDITIONS[1:6] else None
                        probe = AuditModel(model,condition,reference=torch.linspace(-.1,.1,97),fixed_features=fixed).eval()
                        result = probe(batch["x"],tf=batch["tf"])
                        if entry["arm"] == "W0_FULL" and condition in CONDITIONS[1:6]:
                            for key in result:
                                torch.testing.assert_close(result[key],expected[key],rtol=0,atol=0)
                reserved = torch.cuda.max_memory_reserved(device)
                fraction = reserved/torch.cuda.get_device_properties(device).total_memory
                if fraction>.8:
                    raise RuntimeError(f"peak reserved {fraction} 超过 80%")
                record=dict(arm=entry["arm"],seed=entry["seed"],batch_size=batch_size,peak_reserved_bytes=reserved,
                            peak_reserved_fraction=fraction,full_replay_passed=True,cached_full_replay_passed=True,
                            conditions=list(CONDITIONS))
                write_json(out/f"{entry['arm']}_{entry['seed']}_{batch_size}.json",record); records.append(record)
            del model
        write_json(out/"gpu_smoke.json",dict(passed=True,records=records))
    return out


def evaluate(arm, seed, device, reference_path, gpu_receipt):
    lock,digest = load_lock()
    entry = next(e for e in lock["entries"] if (e["arm"],e["seed"])==(arm,seed))
    verify_attempt(reference_path,digest,"reference"); verify_attempt(gpu_receipt,digest,"gpu_smoke")
    smoke=json.loads((gpu_receipt/"gpu_smoke.json").read_text())
    expected={(a,s,1) for a in ARMS for s in SEEDS}|{(a,SEEDS[0],128) for a in ARMS}
    if (not smoke["passed"] or len(smoke["records"])!=20 or
            {(r["arm"],r["seed"],r["batch_size"]) for r in smoke["records"]}!=expected or
            not all(r["full_replay_passed"] and r["cached_full_replay_passed"] and r["conditions"]==list(CONDITIONS)
                    and r["peak_reserved_fraction"]<=.8 for r in smoke["records"])):
        raise ValueError("GPU smoke 回执不完整")
    environment = source.runtime_preflight(device)
    if environment != json.loads((gpu_receipt/"environment.json").read_text()):
        raise ValueError("GPU 验收环境/commit 不匹配")
    reference = torch.from_numpy(np.load(reference_path/"reference.npy",allow_pickle=False))
    with attempt("evaluation",digest,arm,seed) as out:
        if shutil.disk_usage(out).free < COUNT*96*97*360*2+24*2**30:
            raise RuntimeError("磁盘不足：每模型完整 X 约 16.70 GiB，另需 24 GiB 余量")
        write_json(out/"environment.json",environment)
        write_json(out/"implementation_lock.json",lock)
        write_json(out/"access_started.json",dict(split="val",count=COUNT,checkpoint=entry["files"]["checkpoint_best_local_rr.pt"],
            reference={"path":str(reference_path),"manifest":identity(reference_path/"manifest.json")},
            gpu_receipt={"path":str(gpu_receipt),"manifest":identity(gpu_receipt/"manifest.json")}))
        for key in ("val_w","val_rows","manifest","frequency_file"):
            info=lock["cache_files"][key]; source.verify(Path(info["path"]),info)
        source.verify(Path(lock["dataset_index"]["path"]), {"sha256":lock["dataset_index"]["sha256"],
            "size_bytes":Path(lock["dataset_index"]["path"]).stat().st_size})
        model,cfg = load_model(entry,device)
        OmegaConf.save(cfg,out/"resolved_config.yaml")
        data=build_window_data(cfg,split="val",max_windows=None,sample_strategy=str(cfg.data.val_sample_strategy),
                               sample_seed=int(cfg.data.val_sample_seed),shuffle=False)
        rows=pd.DataFrame(lock["validation_rows"])
        pd.testing.assert_frame_equal(data.rows[rows.columns].reset_index(drop=True),rows,check_dtype=False)
        if len(data.dataset)!=COUNT:
            raise ValueError("validation dataset 数量漂移")
        rows.to_csv(out/"rows.csv",index=False)
        frames=[]; quality={}
        for condition in CONDITIONS:
            directory=out/condition; directory.mkdir()
            store=FeatureStore(directory,rows,lock["plot_row_ids"],lock["frequencies"]) if condition=="FULL" else None
            fixed=feature_reader(out/"FULL") if condition in CONDITIONS[1:6] else None
            wrapper=AuditModel(model,condition,reference=reference,fixed_features=fixed,observe=store)
            predictions=collect_predictions(wrapper,checked_batches(data.loader,rows),device=device,max_windows=COUNT,use_amp=True)
            if wrapper.offset!=COUNT:
                raise ValueError("预测/特征未覆盖全部窗口")
            if store:
                store.finish()
            metrics=evaluate_task_predictions(predictions,cfg,include_test_only=False,method=arm)
            quality[condition]=source.validate_metrics(metrics,rows)
            summary=summarize_task_metrics(metrics)
            if not np.isfinite(summary[[m+"_mean" for m in PRIMARY]].to_numpy()).all():
                raise FloatingPointError("主指标汇总非有限")
            if condition=="FULL":
                frozen=pd.read_csv(entry["files"]["metrics.csv"]["path"])
                paired_changes(frozen,metrics,arm,seed,condition)
                for metric in PRIMARY:
                    baseline=frozen[metric].mean(); actual=summary.iloc[0][metric+"_mean"]
                    if not np.isclose(actual,baseline,rtol=1e-3,atol=0):
                        raise RuntimeError(f"FULL 复现失败: {metric}: {actual} vs {baseline}")
                full=metrics
                full_waveform=predictions["r_tho_hat"]
            elif arm=="W0_FULL" and condition in CONDITIONS[1:6]:
                if not np.array_equal(predictions["r_tho_hat"],full_waveform):
                    raise RuntimeError("W0 聚合 no-op 未逐点复现")
            metrics.to_csv(directory/"metrics.csv",index=False)
            summary.to_csv(directory/"summary.csv",index=False)
            frames.append(paired_changes(full,metrics,arm,seed,condition))
            print(f"完成 {arm}/{seed}/{condition}: {COUNT} validation windows",flush=True)
        pd.concat(frames,ignore_index=True).to_csv(out/"paired_changes.csv",index=False)
        write_json(out/"evaluation_receipt.json",dict(arm=arm,seed=seed,count=COUNT,conditions=list(CONDITIONS),
            selected_epoch=entry["selected_epoch"],checkpoint=entry["files"]["checkpoint_best_local_rr.pt"],
            reference_manifest=identity(reference_path/"manifest.json"),quality=quality,full_reproduced=True))
    return out


def summarize():
    lock,digest=load_lock()
    frames=[]; features=[]; entries={}; reference=None
    for arm in ARMS:
        for seed in SEEDS:
            paths=[]
            for freeze in (ROOT/OUTPUT/"evaluation"/arm/f"seed_{seed}").glob("*/freeze_receipt.json"):
                meta=json.loads((freeze.parent/"manifest.json").read_text())
                if meta["implementation_lock_sha256"]==digest:
                    paths.append(freeze.parent)
            if len(paths)!=1:
                raise ValueError(f"{arm}/{seed} 需要唯一成功评价，实际 {len(paths)}")
            path=paths[0]; manifest=verify_attempt(path,digest,"evaluation")
            receipt=json.loads((path/"evaluation_receipt.json").read_text())
            if receipt["conditions"]!=list(CONDITIONS) or receipt["count"]!=COUNT or not receipt["full_reproduced"]:
                raise ValueError("评价矩阵不完整")
            if (manifest["arm"],manifest["seed"],receipt["arm"],receipt["seed"])!=(arm,seed,arm,seed):
                raise ValueError("评价模型 identity 不符")
            if reference is None:
                reference=receipt["reference_manifest"]
            elif reference!=receipt["reference_manifest"]:
                raise ValueError("模型间参考值来源不同")
            frame=pd.read_csv(path/"paired_changes.csv")
            if len(frame)!=14*8*5 or set(frame.condition)!=set(CONDITIONS) or not frame.arm.eq(arm).all() or not frame.seed.eq(seed).all():
                raise ValueError("配对明细不完整")
            frames.append(frame); entries[f"{arm}/{seed}"]={"path":str(path),"manifest":identity(path/"manifest.json")}
            feature=pd.read_csv(path/"FULL"/"feature_statistics.csv")
            rows=pd.read_csv(path/"rows.csv")
            if len(feature)!=COUNT or not np.array_equal(feature.dataset_row_id,rows.dataset_row_id):
                raise ValueError("特征统计 row identity 不完整")
            feature.insert(0,"arm",arm); feature.insert(0,"seed",seed); features.append(feature)
    combined=pd.concat(frames,ignore_index=True)
    summary=[]
    for key, group in combined.groupby(["arm","condition","scope","group","metric"],dropna=False,sort=False):
        if len(group)!=3 or set(group.seed)!=set(SEEDS):
            raise ValueError("三 seed 配对不完整")
        value=group.degradation
        summary.append(dict(zip(("arm","condition","scope","group","metric"),key),
            defined_seeds=int(value.notna().sum()),degradation_mean=value.mean(),degradation_sample_sd=value.std(ddof=1),
            intervention_worse_seeds=int((value>0).sum()),intervention_better_seeds=int((value<0).sum()),equal_seeds=int((value==0).sum())))
    # 每个 seed 的 samp_id 等权宏均值单独保留，避免长记录决定方向。
    subject=combined[combined.scope.eq("samp_id")]
    macro=subject.groupby(["arm","seed","condition","metric"],sort=False).agg(
        subject_macro_degradation=("degradation","mean"),defined_subjects=("degradation","count"),
        worse_subjects=("degradation",lambda x:int((x>0).sum())),better_subjects=("degradation",lambda x:int((x<0).sum())))
    feature_windows=pd.concat(features,ignore_index=True)
    feature_summary=[]
    excluded={"arm","seed","dataset_row_id","samp_id","split"}
    for (arm,seed), frame in feature_windows.groupby(["arm","seed"],sort=False):
        groups=[("pooled","ALL",frame),*(("samp_id",str(s),f) for s,f in frame.groupby("samp_id",sort=True))]
        for scope,group,part in groups:
            for metric in [k for k in frame.columns if k not in excluded]:
                values=part[metric].astype(float)
                feature_summary.append(dict(arm=arm,seed=seed,scope=scope,group=group,metric=metric,count=len(values),
                    defined_count=int(values.notna().sum()),mean=values.mean(),median=values.median(),
                    p10=values.quantile(.1),p90=values.quantile(.9)))
    with attempt("summary",digest) as out:
        combined.to_csv(out/"paired_seed_subject_changes.csv",index=False)
        pd.DataFrame(summary).to_csv(out/"three_seed_direction.csv",index=False)
        macro.to_csv(out/"subject_macro_by_seed.csv")
        feature_windows.to_csv(out/"feature_statistics_by_window.csv",index=False)
        pd.DataFrame(feature_summary).to_csv(out/"feature_statistics_by_seed_subject.csv",index=False)
        write_json(out/"summary_receipt.json",dict(models=15,conditions=14,evaluated_windows=15*14*COUNT,sources=entries,
            reference_manifest=reference,delta="error=intervened-FULL; PCC=FULL-intervened; positive=worse"))
    return out
