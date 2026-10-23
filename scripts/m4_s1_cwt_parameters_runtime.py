"""M4-S1：CPU准备、受控缓存/GPU验收、84-cell训练恢复和完整validation汇总。"""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
from pathlib import Path
import sys
import time
import traceback
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train import patch_aligned_tf_training as legacy
from resp_train.crd.tf_v1_features import cwt_magnitude_features
from resp_train.crd.training import build_crd_optimizer, optimizer_updates_per_epoch
from resp_train.engine import collect_predictions, validate
from resp_train.losses.task import RespirationTaskLoss
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.cwt_time_frequency_v1.features import transform
from scripts import m4_s1_cwt_parameters_artifacts as io
from scripts import m4_s1_cwt_parameters_data as data_ops
from scripts import m4_s1_cwt_parameters_spec as spec
from scripts.m4_linear_fusion_model import LinearFusionModel
from scripts.m4_s1_cwt_parameters_model import M4S1CWTModel, architecture, build_model
from scripts.m4_s1_cwt_parameters_summary import summary_tables, write_report


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name == "condition.frequencies_hz":
            continue
        tensor = value.detach().cpu().contiguous()
        legacy.finite_tree(tensor)
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_initialization(representations=None, *, cfg=None, mamba_factory=None):
    reps = representations if representations is not None else spec.representations()
    canonical = reps["B"]["frequencies_hz"]
    records = {}
    for seed in spec.SEEDS:
        baseline = LinearFusionModel("L1", seed, canonical, cfg=cfg, mamba_factory=mamba_factory)
        expected = dict(baseline.named_parameters())
        records[str(seed)] = {}
        for arm, option in spec.ARMS.items():
            rep = reps[option.representation] if option.representation else None
            previous_rng = torch.get_rng_state().clone()
            model = M4S1CWTModel(arm, seed, rep, cfg=cfg, mamba_factory=mamba_factory, canonical_frequencies=canonical)
            if not torch.equal(torch.get_rng_state(), previous_rng):
                raise ValueError("模型构造改变外部CPU随机状态")
            params = dict(model.named_parameters())
            if not params.keys() <= expected.keys() or any(not torch.equal(value, expected[name]) for name, value in params.items()):
                raise ValueError(f"共有参数初始化不匹配: {arm}/{seed}")
            if arm == "B":
                left, right = baseline.state_dict(), model.state_dict()
                if (list(left) != list(right) or list(expected) != list(params)
                        or any(not torch.equal(left[k], right[k]) for k in left)):
                    raise ValueError("B与原生L1的完整初始化不一致")
            count = sum(p.numel() for p in model.parameters())
            if mamba_factory is None and cfg is None and arm in spec.CWT_SETTINGS and count != 1046945:
                raise ValueError("CWT参数臂参数量不符")
            zero = model.condition is None or all(torch.count_nonzero(p) == 0 for p in model.condition.output.parameters())
            if not zero:
                raise ValueError("初始条件残差投影必须为零")
            records[str(seed)][arm] = {"state_sha256": state_hash(model.state_dict()), "parameters": count,
                "common_parameters_exact": True, "external_rng_unchanged": True,
                "initial_condition_residual_zero": True,
                "removed_parameter_names": sorted(expected.keys() - params.keys()),
                "mamba": "official_mamba2" if option.mamba and mamba_factory is None else "fixture" if option.mamba else "identity"}
            del model
        del baseline
    return records


def prepare(session, *, scope="synthetic_only"):
    if scope not in ("synthetic_only", "train_validation"):
        raise ValueError("未知session范围")
    session = Path(session).resolve()
    if not session.is_relative_to(spec.OUTPUT_ROOT.resolve()) and scope != "synthetic_only":
        raise ValueError(f"正式session须位于 {spec.OUTPUT_ROOT}")
    session.mkdir(parents=True, exist_ok=True)
    with io.mutex(session / ".prepare.lock"):
        if (session / "session_receipt.json").exists():
            previous = io.load_session(session)
            if previous["scope"] != scope:
                raise ValueError("已有session范围不能改变")
            return session
        if set(p.name for p in session.iterdir()) - {".prepare.lock"}:
            raise FileExistsError("session已存在准备产物，请使用新identity")
        try:
            reps = spec.representations()
            initialization = audit_initialization(reps)
            resolved = session / "resolved_configs"
            resolved.mkdir()
            files = {}
            for cell in spec.plan():
                arm = cell["arm"]
                key = spec.ARMS[arm].representation
                cfg = spec.config(arm, cell["seed"], session / "cells" / cell["cell"], "cuda:0",
                                  representation=reps[key] if key else None)
                path = resolved / f"{cell['cell']}.yaml"
                with path.open("x", encoding="utf-8") as stream:
                    OmegaConf.save(cfg, stream)
                files[str(path.relative_to(session))] = io.identity(path)
            io.create_session(session, reps, initialization, files, scope=scope)
        except BaseException as exc:
            io.write_json(session / "prepare_failed.json", {"error": repr(exc), "traceback": traceback.format_exc()})
            raise
    return session


def synthetic_signal():
    t = np.arange(18000, dtype=np.float32) / 100
    x = ((1 + .2 * np.sin(2 * np.pi * .25 * t)) * np.sin(2 * np.pi * 2 * t)
         + .1 * np.sin(2 * np.pi * .18 * t) + .05 * np.sin(2 * np.pi * 13 * t)).astype(np.float32)
    y = np.sin(2 * np.pi * .25 * t).astype(np.float32)
    return x, y


def calibrate(session, *, retry_failed=False):
    frozen = io.load_session(session)
    key = io.binding(session, "calibration")
    prior = io.completed(Path(session) / "calibration", key)
    if prior:
        return prior
    with io.attempt(Path(session) / "calibration", key, retry_failed=retry_failed) as output:
        signal, _ = synthetic_signal()
        values = {a: transform(signal, r) for a, r in frozen["representations"].items()}
        old, frequencies = cwt_magnitude_features(signal)
        np.testing.assert_array_equal(frequencies, frozen["representations"]["B"]["frequencies_hz"])
        np.testing.assert_allclose(values["B"], old, rtol=1e-6, atol=1e-7)
        for arm, mask in (("H", (frequencies > .8) & (frequencies <= 8)),
                          ("C_0p8", frequencies <= .8), ("C_2", frequencies <= 2), ("C_4", frequencies <= 4)):
            np.testing.assert_allclose(values[arm], values["B"][mask], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(values["P_025"].reshape(97, 360, 2).mean(-1), values["B"], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(values["B"].reshape(97, 180, 2).mean(-1), values["P_100"], rtol=1e-6, atol=1e-7)
        for arm, array in values.items():
            if array.dtype != np.float32 or list(array.shape) != frozen["representations"][arm]["shape"] or not np.isfinite(array).all():
                raise ValueError("合成CWT shape/dtype/finite校准失败")
        io.write_json(output / "calibration.json", {"passed": True, "synthetic_only": True,
            "representations": frozen["representations"], "baseline_max_abs": float(np.abs(old - values["B"]).max()),
            "subsets_and_pooling_consistent": True, "rtol": 1e-6, "atol": 1e-7})
        io.load_session(session)
    return output


def require_calibration(session):
    path = io.completed(Path(session) / "calibration", io.binding(session, "calibration"))
    if path is None or io.read_json(path / "calibration.json")["passed"] is not True:
        raise RuntimeError("须先完成合成CWT校准")
    return path


def require_cuda(device):
    parsed = torch.device(device)
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("正式GPU入口要求显式cuda:<index>")
    torch.cuda.set_device(parsed)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU不支持BF16")
    return parsed


def baseline_equivalence(rep, device, *, cfg=None, mamba_factory=None):
    """打开条件输出后检查原生L1及新B；GPU验收不以零残差相等替代路径验证。"""
    x, _ = synthetic_signal()
    w = transform(x, rep)
    x = torch.from_numpy(x)[None, None].to(device)
    tf = {"w": torch.from_numpy(w)[None].to(device)}
    checks = []
    for seed in spec.SEEDS:
        original = LinearFusionModel("L1", seed, rep["frequencies_hz"], cfg=cfg, mamba_factory=mamba_factory).to(device).eval()
        adapted = M4S1CWTModel("B", seed, rep, cfg=cfg, mamba_factory=mamba_factory,
                              canonical_frequencies=rep["frequencies_hz"]).to(device).eval()
        with torch.no_grad():
            values = torch.arange(original.condition.output.weight.numel(), device=device).reshape_as(original.condition.output.weight)
            original.condition.output.weight.copy_(.02 * values.float().sin())
            adapted.load_state_dict(original.state_dict(), strict=True)
        for bf16 in (False, True):
            original.zero_grad(set_to_none=True)
            adapted.zero_grad(set_to_none=True)
            tol = {"rtol": 2e-2, "atol": 2e-3} if bf16 else {"rtol": 2e-5, "atol": 2e-6}
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=bf16):
                left = original(x, tf=tf)["waveform"]
                right = adapted(x, tf=tf)["waveform"]
            torch.testing.assert_close(left, right, **tol)
            left.float().square().mean().backward()
            right.float().square().mean().backward()
            adapted_parameters = dict(adapted.named_parameters())
            for name, parameter in original.named_parameters():
                grad = adapted_parameters[name].grad
                if parameter.grad is None or grad is None:
                    raise ValueError(f"基准等价检查缺少梯度: {name}")
                torch.testing.assert_close(parameter.grad, grad, **tol)
            checks.append({"seed": seed, "bf16": bf16, "max_abs": float((left - right).abs().max().detach()),
                           "forward_and_gradients_match": True, **tol})
        del original, adapted
    return checks


def smoke(session, arm, device, *, confirmed=False, retry_failed=False, mamba_factory=None, batch_size=32, cfg_override=None):
    if confirmed is not True:
        raise PermissionError("GPU验收要求显式--confirm-synthetic")
    frozen = io.load_session(session)
    if arm not in spec.ARMS:
        raise ValueError("未知验收臂")
    if mamba_factory is None:
        require_cuda(device)
    elif torch.device(device).type != "cpu":
        raise ValueError("合成替身只允许CPU")
    calibration = require_calibration(session)
    environment = legacy.runtime(device)
    key = io.binding(session, "engineering", arm=arm, device=str(device), environment=environment,
                     fixture=mamba_factory is not None, batch_size=batch_size,
                     architecture=asdict(cfg_override or architecture()))
    parent = Path(session) / "engineering" / arm / io.json_hash(key)[:16]
    prior = io.completed(parent, key)
    if prior:
        return prior
    with io.attempt(parent, key, retry_failed=retry_failed) as output:
        representation = spec.ARMS[arm].representation
        rep = frozen["representations"][representation] if representation else None
        cfg = spec.config(arm, spec.SEEDS[0], output, device, representation=rep)
        legacy.set_seed(spec.SEEDS[0])
        x, y = synthetic_signal()
        batch = {"x": torch.from_numpy(x)[None, None].repeat(batch_size, 1, 1),
                 "target": torch.from_numpy(y)[None, None].repeat(batch_size, 1, 1)}
        if rep:
            batch["tf"] = {"w": torch.from_numpy(transform(x, rep))[None].repeat(batch_size, 1, 1)}
        equivalence = baseline_equivalence(rep, device, cfg=cfg_override, mamba_factory=mamba_factory) if arm == "B" else None
        model = M4S1CWTModel(arm, spec.SEEDS[0], rep, cfg=cfg_override, mamba_factory=mamba_factory).to(device)
        loss = RespirationTaskLoss(cfg).to(device)
        optimizer, _ = build_crd_optimizer(model, cfg)
        watch = model.waveform_encoder.project.weight if model.condition is None else model.condition.encoder[1].weight
        before = watch.detach().clone()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        training, updates = legacy._train_epoch(model, [batch] * 8, loss, optimizer, cfg, 0, 6400, 1)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        legacy.finite_tree((training, model.state_dict(), optimizer.state_dict()))
        if (updates != 2 or watch.grad is None or not torch.isfinite(watch.grad).all()
                or watch.grad.abs().sum() == 0 or torch.equal(watch, before)):
            raise RuntimeError("两次更新后条件/时域分支没有有效任务梯度")
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"活动参数梯度缺失或非有限: {name}")
        validation = validate(model, [batch], loss, device=device, use_amp=True, show_progress=False)
        legacy.finite_tree(validation)
        reserved = torch.cuda.max_memory_reserved(device) if torch.device(device).type == "cuda" else None
        total_memory = torch.cuda.get_device_properties(device).total_memory if torch.device(device).type == "cuda" else None
        if total_memory and reserved / total_memory > .95:
            raise RuntimeError(f"验收显存超过95%: {reserved}/{total_memory}")
        io.write_json(output / "engineering.json", {"passed": True, "arm": arm, "contract": spec.contract(arm),
            "session_sha256": io.sha(Path(session) / "session.json"), "calibration": io.identity(calibration / "manifest.json"),
            "environment": environment, "batch_size": batch_size, "accumulation": 4, "updates": updates,
            "mamba": "official_mamba2" if spec.ARMS[arm].mamba and mamba_factory is None else "fixture" if spec.ARMS[arm].mamba else "identity",
            "fixture": mamba_factory is not None, "architecture": asdict(cfg_override or architecture()),
            "parameters": sum(p.numel() for p in model.parameters()), "train_elapsed_seconds": elapsed,
            "timing_scope": "two_update_engineering_including_initial_kernel_cost",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if total_memory else None,
            "peak_reserved_bytes": reserved, "total_memory_bytes": total_memory,
            "baseline_equivalence": equivalence, "training": training, "validation": validation})
        io.load_session(session)
    return output


def ensure_smoke(session, arm, device):
    # 同一正式arm/设备/环境自动复用成功验收；失败必须单独检查后显式重试。
    return smoke(session, arm, device, confirmed=True)


def register_environment(session, environment):
    path = Path(session) / "training_environment.json"
    with io.mutex(Path(session) / ".environment.lock", blocking=True):
        if path.exists():
            previous = io.read_json(path)
            keys = ("torch", "cuda", "packages", "gpu", "LD_LIBRARY_PATH_unset", "LD_PRELOAD_unset")
            if any(previous[k] != environment[k] for k in keys):
                raise ValueError("配对训练的软件/GPU型号/动态库环境变化")
        else:
            io.write_json(path, environment)


def cell_config(session, arm, seed, device):
    frozen = io.load_session(session)
    path = Path(session) / "resolved_configs" / f"{arm}_seed{seed}.yaml"
    cfg = OmegaConf.load(path)
    cfg.training.device = str(device)
    rep_key = spec.ARMS[arm].representation
    expected = spec.config(arm, seed, Path(session) / "cells" / f"{arm}_seed{seed}", device,
                          representation=frozen["representations"][rep_key] if rep_key else None)
    if legacy.config_identity(cfg) != legacy.config_identity(expected):
        raise ValueError("resolved config与当前cell合同不符")
    return cfg


def verify_training_result(cell, result, cfg):
    """核验最早最小Local RR、早停轨迹和已保存的完整epoch身份。"""
    epochs = legacy.completed_epochs(cell)
    history = [io.read_json(path / "record.json") for _, path in epochs]
    if not history:
        raise ValueError("缺少完整训练epoch")
    per_epoch = 80
    best, best_epoch, wait, stopped = float("inf"), None, 0, False
    for epoch, row in enumerate(history, 1):
        legacy.finite_tree(row)
        if stopped or row["epoch"] != epoch or row["optimizer_update"] != epoch * per_epoch:
            raise ValueError("history epoch/update/停止轨迹不符")
        score = float(row["val_local_rr_mae"])
        if score < best - float(cfg.training.early_stopping_min_delta):
            best, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
        stopped = epoch >= int(cfg.training.early_stopping_min_epoch) and wait >= int(cfg.training.early_stopping_patience)
        if (row["best_epoch"], row["wait"], row["stopped"]) != (best_epoch, wait, stopped):
            raise ValueError("history选点/早停状态不一致")
    n = len(history)
    if n > int(cfg.training.epochs) or (n < int(cfg.training.epochs) and not stopped):
        raise ValueError("训练尚未到合法停止点")
    expected = {"best_epoch": best_epoch, "best_local_rr": best, "completed_epochs": n,
                "optimizer_updates": n * per_epoch, "early_stopped": stopped}
    if any(result[k] != v for k, v in expected.items()):
        raise ValueError("完成记录与history不符")
    by_epoch = dict(epochs)
    for field, epoch in (("best_checkpoint", best_epoch), ("final_checkpoint", n)):
        path = io.artifact_path(cell, result[field])
        if path != (by_epoch[epoch] / "checkpoint.pt").resolve():
            raise ValueError("checkpoint路径与所选epoch不符")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        legacy.finite_tree(saved)
        if (saved["epoch"] != epoch or saved["config"] != legacy.config_identity(cfg)
                or saved["update_index"] != epoch * per_epoch
                or saved["total_updates"] != int(cfg.training.epochs) * per_epoch
                or saved["history"] != history[:epoch]):
            raise ValueError("checkpoint配置/更新数/history不符")
    return history


def verified_cell(session, arm, seed):
    cell = Path(session) / "cells" / f"{arm}_seed{seed}"
    receipt = io.read_json(cell / "completed.json")
    frozen = io.read_json(Path(session) / "session.json")
    if (receipt["arm"] != arm or receipt["seed"] != seed
            or receipt["session_sha256"] != io.sha(Path(session) / "session.json")):
        raise ValueError("完成cell标签或session绑定不符")
    if (receipt["parameters"] != frozen["initialization"][str(seed)][arm]["parameters"]
            or not 30 <= receipt["completed_epochs"] <= 80
            or not 1 <= receipt["best_epoch"] <= receipt["completed_epochs"]
            or receipt["optimizer_updates"] != 80 * receipt["completed_epochs"]
            or not np.isfinite(receipt["best_local_rr"])):
        raise ValueError("完成cell参数量、选点或更新预算非法")
    required = {"config.yaml", "environment.json", receipt["metrics"], receipt["best_checkpoint"],
                receipt["final_checkpoint"], receipt["history"], receipt["resource"]}
    if not required <= receipt["artifacts"].keys():
        raise ValueError("完成记录缺少必要产物")
    for name, expected in receipt["artifacts"].items():
        io.verify(io.artifact_path(cell, name), expected)
    return receipt


def export_validation(session, arm, seed, cfg, model, data, output):
    prediction = collect_predictions(model, data.val.loader, device=torch.device(cfg.training.device),
                                     max_windows=len(data.val.dataset), use_amp=bool(cfg.training.use_amp))
    for name in ("r_tho_hat", "tho_ref"):
        if not np.isfinite(prediction[name]).all():
            raise FloatingPointError("validation波形非有限")
    if not np.array_equal(prediction["dataset_row_id"], data.val.rows.dataset_row_id.to_numpy(np.int64)):
        raise ValueError("validation预测row顺序变化")
    data_path, _ = data_ops.data_receipt(session, verify_inputs=False)
    reference = np.load(data_path / "validation_reference.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(reference, prediction["tho_ref"]):
        raise ValueError("validation target与公共参考不一致")
    metrics = evaluate_task_predictions(prediction, cfg, include_test_only=False, method=spec.ARMS[arm].name)
    sf.validate_metrics(metrics, data.val.rows)
    metrics.insert(0, "arm", arm)
    metrics.insert(1, "seed", seed)
    metrics.insert(2, "evaluation_split", "validation")
    metrics.to_csv(Path(output) / "validation_metrics.csv", index=False)
    summarize_task_metrics(metrics).to_csv(Path(output) / "validation_summary.csv", index=False)
    return metrics


def run_cell(session, arm, seed, device, *, confirmed=False, resume=False):
    if confirmed is not True:
        raise PermissionError("正式训练要求显式--confirm-training")
    if arm not in spec.ARMS or seed not in spec.SEEDS:
        raise ValueError("未知训练cell")
    frozen = io.require_data_scope(session)
    require_cuda(device)
    require_calibration(session)
    session = Path(session).resolve()
    cell = session / "cells" / f"{arm}_seed{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    with io.mutex(cell / ".run.lock"):
        if (cell / "completed.json").exists():
            return verified_cell(session, arm, seed)
        if (cell / "config.yaml").exists() and not resume:
            raise FileExistsError("cell已有记录；恢复须显式--resume")
        output = cell / "attempts" / uuid4().hex
        output.mkdir(parents=True, exist_ok=False)
        try:
            io.write_json(output / "started.json", {"arm": arm, "seed": seed, "command": sys.argv,
                "session_sha256": io.sha(session / "session.json"), "resume": resume})
            cfg = cell_config(session, arm, seed, device)
            if (cell / "config.yaml").exists():
                if legacy.config_identity(OmegaConf.load(cell / "config.yaml")) != legacy.config_identity(cfg):
                    raise ValueError("恢复配置变化")
            else:
                with (cell / "config.yaml").open("x", encoding="utf-8") as stream:
                    OmegaConf.save(cfg, stream)
            environment = legacy.runtime(device)
            register_environment(session, environment)
            if not (cell / "environment.json").exists():
                io.write_json(cell / "environment.json", environment)
            io.write_json(output / "environment.json", environment)
            # 完整B的GPU数值验收先于任意候选训练。
            baseline_acceptance = ensure_smoke(session, "B", device)
            acceptance = baseline_acceptance if arm == "B" else ensure_smoke(session, arm, device)
            io.write_json(output / "engineering.json", {"baseline": str(baseline_acceptance),
                "baseline_manifest": io.identity(baseline_acceptance / "manifest.json"),
                "arm": str(acceptance), "arm_manifest": io.identity(acceptance / "manifest.json")})
            legacy.set_seed(seed)
            data = data_ops.training_data(session, arm, cfg, output)
            if optimizer_updates_per_epoch(len(data.train.loader), int(cfg.training.gradient_accumulation_steps)) != 80:
                raise ValueError("每epoch更新数不符")
            model = build_model(cfg)
            initial = frozen["initialization"][str(seed)][arm]
            if (state_hash(model.state_dict()) != initial["state_sha256"]
                    or sum(p.numel() for p in model.parameters()) != initial["parameters"]):
                raise ValueError("训练模型与prepare初始化不符")
            io.write_json(output / "initialization.json", initial)
            model = model.to(device)
            loss = RespirationTaskLoss(cfg).to(device)
            optimizer, partition = build_crd_optimizer(model, cfg)
            io.write_json(output / "optimizer_groups.json", {"decay": partition.decay_names, "no_decay": partition.no_decay_names})
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            training_started = time.perf_counter()
            result = legacy.run_epochs(model, data, loss, optimizer, cfg, cell, resume=resume, log_path=output / "train.log")
            torch.cuda.synchronize(device)
            training_elapsed = time.perf_counter() - training_started
            history = verify_training_result(cell, result, cfg)
            cache_rep = spec.ARMS[arm].representation
            cache_bytes = None
            if cache_rep:
                _, cache_dir = data_ops.cache_receipt(session, cache_rep)
                cache_bytes = sum((cache_dir / f"{s}_w.npy").stat().st_size for s in spec.COUNTS)
            io.write_json(output / "resource.json", {"current_attempt_training_seconds": training_elapsed,
                "current_attempt_includes_checkpoint_serialization": True,
                "all_epoch_train_validation_selection_seconds": sum(r["elapsed_seconds"] for r in history),
                "epoch_timing_excludes_checkpoint_serialization": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cwt_train_validation_array_bytes": cache_bytes, "resumed": resume})
            # 导出失败时，--resume复用合法训练终点；旧失败attempt和epochs不改写。
            io.write_json(output / "training_complete.json", result)
            selected = torch.load(io.artifact_path(cell, result["best_checkpoint"]), map_location="cpu", weights_only=False)
            legacy.finite_tree(selected)
            model.load_state_dict(selected["model_state_dict"], strict=True)
            metrics = export_validation(session, arm, seed, cfg, model, data, output)
            score = float(metrics.loc[metrics.local_rr_target_eligible, "local_rr_mae_bpm"].mean())
            if not np.isclose(score, result["best_local_rr"], rtol=1e-6, atol=1e-7):
                raise ValueError("导出Local RR与history选点值不同")
            pd.json_normalize(history).to_csv(output / "train_history.csv", index=False)
            io.load_session(session)
            files = [p for p in output.rglob("*") if p.is_file()]
            # 所有完整epoch都进入完成manifest，恢复轨迹和记录文件同样受保护。
            files += [p for _, folder in legacy.completed_epochs(cell) for p in folder.iterdir() if p.is_file()]
            files += [cell / "config.yaml", cell / "environment.json"]
            result.update(arm=arm, seed=seed, parameters=initial["parameters"],
                session_sha256=io.sha(session / "session.json"),
                metrics=str((output / "validation_metrics.csv").relative_to(cell)),
                history=str((output / "train_history.csv").relative_to(cell)),
                resource=str((output / "resource.json").relative_to(cell)),
                artifacts={str(p.relative_to(cell)): io.identity(p) for p in files})
            io.write_json(cell / "completed.json", result)
            return result
        except BaseException as exc:
            io.write_json(output / "failed.json", {"error": repr(exc), "traceback": traceback.format_exc()})
            raise
        finally:
            gc.collect()


def status(session):
    io.load_session(session)
    rows = []
    for cell in spec.plan():
        path = Path(session) / "cells" / cell["cell"]
        if (path / "completed.json").exists():
            verified_cell(session, cell["arm"], cell["seed"])
            state = "completed_verified"
        elif list((path / "attempts").glob("*/failed.json")):
            state = "failed_or_resumable"
        elif (path / "config.yaml").exists():
            state = "partial"
        else:
            state = "pending"
        rows.append({**cell, "status": state})
    return rows


def summarize(session, *, retry_failed=False):
    io.load_session(session)
    session = Path(session)
    if any(not (session / "cells" / c["cell"] / "completed.json").is_file() for c in spec.plan()):
        raise RuntimeError("84个cell尚未全部完成，拒绝partial矩阵汇总")
    frames, sources, parameters, selections, native, efficiency = [], {}, [], [], [], []
    for cell in spec.plan():
        arm, seed = cell["arm"], cell["seed"]
        directory = session / "cells" / cell["cell"]
        result = verified_cell(session, arm, seed)
        frame = pd.read_csv(io.artifact_path(directory, result["metrics"]))
        if (len(frame) != spec.COUNTS["val"] or frame.samp_id.nunique() != spec.SUBJECTS["val"]
                or set(frame.split) != {"val"} or not frame.arm.eq(arm).all() or not frame.seed.eq(seed).all()):
            raise ValueError("validation分母或标签不符")
        sf.validate_metrics(frame, frame[["dataset_row_id", "samp_id", "split"]])
        exported_score = float(frame.loc[frame.local_rr_target_eligible, "local_rr_mae_bpm"].mean())
        if not np.isclose(exported_score, result["best_local_rr"], rtol=1e-6, atol=1e-7):
            raise ValueError("完成记录的选点值与逐窗口指标不符")
        frames.append(frame)
        sources[cell["cell"]] = io.identity(directory / "completed.json")
        parameters.append({"arm": arm, "seed": seed, "parameters": result["parameters"]})
        selections.append({"arm": arm, "seed": seed, **{k: result[k] for k in
            ("best_epoch", "completed_epochs", "best_local_rr", "optimizer_updates", "early_stopped")}})
        resources = io.read_json(io.artifact_path(directory, result["resource"]))
        legacy.finite_tree(resources)
        efficiency.append({"arm": arm, "seed": seed, **resources})
        detail = summarize_task_metrics(frame)
        detail.insert(0, "arm", arm)
        detail.insert(1, "seed", seed)
        native.append(detail)
    key = io.binding(session, "summary", sources=sources)
    prior = io.completed(session / "summary", key)
    if prior:
        return prior
    with io.attempt(session / "summary", key, retry_failed=retry_failed) as output:
        tables = summary_tables(pd.concat(frames, ignore_index=True))
        tables.update(parameters=pd.DataFrame(parameters), selection=pd.DataFrame(selections), efficiency=pd.DataFrame(efficiency),
                      native_summary=pd.concat(native, ignore_index=True))
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False)
        io.write_json(output / "sources.json", sources)
        write_report(output, tables)
        io.load_session(session)
    return output
