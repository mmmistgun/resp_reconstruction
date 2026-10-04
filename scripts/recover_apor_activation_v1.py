"""恢复共享validation参考写入冲突；串行完成原矩阵，保留全部失败产物。"""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scripts import apor_activation_v1_runtime as exp
from scripts import eval_apor_activation_v1_research_test as test
from resp_train.crd.experiment import set_seed

SESSION = ROOT / "runs/apor_activation_v1/session_20260930T120541Z_324e56049c8a"
FAILED = SESSION / "formal/U1/seed_20260812/attempt_20260930T120835Z_c7836048f0ce"
DOC = ROOT / "docs/experiments/apor_activation_v1_recovery_20260930.md"


def assert_same_tree(left, right):
    """逐tensor精确比较，确保迁移没有改变权重、optimizer或更新状态。"""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError("checkpoint tensor改变")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError("checkpoint字段改变")
        for key in left:
            assert_same_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError("checkpoint序列改变")
        for a, b in zip(left, right):
            assert_same_tree(a, b)
    elif left != right:
        raise ValueError("checkpoint标量改变")


def migrate_checkpoint(source, destination, old_cfg, new_cfg):
    original = torch.load(source, map_location="cpu", weights_only=False)
    old = OmegaConf.to_container(old_cfg, resolve=True)
    new = OmegaConf.to_container(new_cfg, resolve=True)
    expected = copy.deepcopy(old)
    expected["outputs"]["run_root"] = new["outputs"]["run_root"]
    if original["config"] != old or new != expected:
        raise ValueError("恢复只允许改变outputs.run_root")
    exp.finite_tree(original, label="original_checkpoint")
    migrated = {**original, "config": new}
    with destination.open("xb") as stream:
        torch.save(migrated, stream)
    saved = torch.load(destination, map_location="cpu", weights_only=False)
    saved["config"] = old
    assert_same_tree(original, saved)
    return {"original": {"path": str(source), **exp.sf.identity(source)},
            "derived": {"path": str(destination), **exp.sf.identity(destination)},
            "model_state": exp.state_dict_identity(original["model_state_dict"]),
            "only_changed_field": "config.outputs.run_root", "other_payload_exact": True}


def recover_validation():
    seed = 20260812
    prior = exp.completed(FAILED.parent, SESSION, "formal", True, arm="U1", seed=seed)
    if prior:
        return prior
    failure = exp.read_json(FAILED / "failed.json")["traceback"]
    if "BlockingIOError" not in failure or "validation_reference.lock" not in failure:
        raise ValueError("不是本修复覆盖的validation参考文件锁故障")
    runs = list((FAILED / "training").glob("*/checkpoint_final.pt"))
    if len(runs) != 1:
        raise ValueError("训练终态checkpoint不唯一")
    old_run = runs[0].parent
    old_cfg = OmegaConf.load(old_run / "config.yaml")
    exp.sf.validate_history(pd.read_csv(old_run / "train_history.csv"), old_cfg)
    old_files = {str(p.relative_to(FAILED)): exp.sf.identity(p) for p in FAILED.rglob("*") if p.is_file()}
    acceptance = Path(exp.read_json(FAILED / "engineering_source.json")["path"])
    exp.verify_gpu(acceptance, SESSION)
    with exp.attempt(FAILED.parent, SESSION, "formal", arm="U1", seed=seed, recovery=True) as output:
        shutil.copy2(__file__, output / "recovery_source.py")
        shutil.copy2(DOC, output / "recovery_protocol.md")
        for name in ("environment.json", "engineering_source.json", "initialization.json"):
            shutil.copy2(FAILED / name, output / name)
        cfg = exp.config("U1", seed, output / "training", str(old_cfg.training.device))
        rows = exp.audit_sources(cfg, output)
        for split in ("train", "val"):
            if not rows[split].equals(pd.read_csv(FAILED / f"{split}_rows.csv")):
                raise ValueError("恢复样本与原训练不一致")
        run = output / "training" / "recovered_validation"
        run.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(cfg, run / "config.yaml")
        for name in ("train_history.csv", "optimizer_parameter_groups.json", "audit.csv", "run_manifest.json"):
            shutil.copy2(old_run / name, run / name)
        checkpoints = [migrate_checkpoint(old_run / name, run / name, old_cfg, cfg)
                       for name in ("checkpoint_best_local_rr.pt", "checkpoint_final.pt")]
        device = str(cfg.training.device)
        environment = exp.cuda_environment(device)
        accepted = exp.read_json(acceptance / "environment.json")
        if environment["dependencies"] != accepted["dependencies"] or environment["device_name"] != accepted["device_name"]:
            raise ValueError("恢复运行环境不符合原GPU验收")
        set_seed(seed)
        experiment = exp.ModuleExperiment(cfg, rows, session=SESSION)
        experiment.run_dir, experiment.device = run, torch.device(device)
        data = experiment._build_data()
        model = exp.build_model(cfg).to(device)
        checkpoint = torch.load(run / "checkpoint_best_local_rr.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"补齐已完成训练的validation导出: {output}", flush=True)
        metrics = experiment._evaluate_model(model, data.val.loader)
        previous = np.load(old_run / "validation_prediction.npy", mmap_mode="r", allow_pickle=False)
        recovered = np.load(run / "validation_prediction.npy", mmap_mode="r", allow_pickle=False)
        if not np.array_equal(previous, recovered):
            raise ValueError("恢复预测与故障前保存的预测不完全一致")
        if not pd.read_csv(old_run / "prediction_rows.csv").equals(pd.read_csv(run / "prediction_rows.csv")):
            raise ValueError("恢复预测行顺序改变")
        metrics.to_csv(run / "metrics.csv", index=False)
        exp.summarize_task_metrics(metrics).to_csv(run / "metrics_summary.csv", index=False)
        exp.write_json(run / "runtime_summary.json", {"device": device, "peak_allocated_mib": None,
                       "peak_reserved_mib": None, "reason": "原训练结束后写入失败，训练峰值未保存；恢复不重训"})
        receipt = exp.validate_run(run, cfg, rows["val"], output / "initialization.json")
        for relative, identity in old_files.items():
            exp.sf.verify_identity(FAILED / relative, identity)
        exp.write_json(output / "training_recovery.json", {"original_attempt": str(FAILED), "original_files": old_files,
                       "checkpoints": checkpoints, "optimizer_updates_during_recovery": 0,
                       "validation_prediction_exact": True, "environment": environment,
                       "run_manifest_role": "原训练manifest原样复制，恢复来源由本receipt记录"})
        exp.write_json(output / "result.json", {**receipt, "run_dir": str(run.relative_to(output)), "arm": "U1", "seed": seed})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-training", required=True, action="store_true")
    parser.add_argument("--confirm-research-test", required=True, action="store_true")
    parser.parse_args()
    torch.set_num_threads(4)
    exp.load_session(SESSION)
    # 持有与原调度器相同的互斥锁；仅一个validation写入者可运行。
    with exp.file_mutex(SESSION / ".parallel_controller.lock"), exp.session_mutex(SESSION):
        with exp.attempt(SESSION / "recovery", SESSION, "recovery") as output:
            shutil.copy2(__file__, output / "recovery_source.py")
            shutil.copy2(DOC, output / "recovery_protocol.md")
            recovered = recover_validation()
            torch.cuda.empty_cache()
            acceptances = sorted((SESSION / "engineering/cuda_0").glob("attempt_*/freeze_receipt.json"))
            if len(acceptances) != 1:
                raise ValueError("cuda:0验收来源不唯一")
            exp.run_formal(SESSION, "U1", 20260813, "cuda:0", acceptances[0].parent, retry_failed=True)
            validation = exp.summarize(SESSION)
            exp.write_json(output / "result.json", {"recovered_formal": str(recovered), "validation_summary": str(validation)})
    allowlist = test.prepare_allowlist(SESSION)
    summary = test.parallel(allowlist, ["cuda:0", "cuda:1"], retry_failed=True)
    result = {"validation_summary": str(validation), "test_allowlist": str(allowlist), "test_summary": str(summary)}
    path = SESSION / "pipeline_result.json"
    if path.exists():
        if exp.read_json(path) != result:
            raise ValueError("pipeline结果漂移")
    else:
        exp.write_json(path, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
