"""TF-S6/S7：APOR真实W-GN拓扑、同批FULL及140-patch FiLM案例。"""
from __future__ import annotations
import shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from omegaconf import OmegaConf
from resp_train.engine.train import _extract_meta, _prediction_dict_from_arrays
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.e4_r3_norm import native_stats, frozen_norm, replay_norm, check_formula
from . import artifacts as io
from .data import data_receipt, training_data
from .engineering import require_gpu
from .model import build_model, GN_NAMES, forward_with_capture
from .runtime import cell_result
from .signals import reference_curves
from .spec import SEEDS, OUTPUT_ROOT

CONDITIONS = ("FULL__NAT", "FULL__FIXED") + tuple(
    f"{band}_{kind}__{mode}" for band in ("H", "H2") for kind in ("MEAN", "SHIFT1", "SHIFT2", "SHIFT3") for mode in ("NAT", "FIXED"))


def verify_full_replay(frame, source):
    current = frame[frame.condition.eq("FULL__NAT")].reset_index(drop=True)
    sf.validate_metrics(current, source[["dataset_row_id", "samp_id", "split"]])
    for metric in sf.PRIMARY:
        mask = sf._metric_mask(source, metric)
        if not np.array_equal(mask, sf._metric_mask(current, metric)):
            raise ValueError("FULL 重放资格发生变化")
        np.testing.assert_allclose(current.loc[mask, metric].mean(), source.loc[mask, metric].mean(), rtol=1e-3, atol=0)


def transform_condition(w, frequencies, condition, shifts):
    if condition not in CONDITIONS or w.ndim != 3 or tuple(w.shape[1:]) != (len(frequencies), 360):
        raise ValueError("干预条件/输入 shape 非法")
    if not torch.isfinite(w).all():
        raise FloatingPointError("干预输入非有限")
    if condition.startswith("FULL"):
        return w
    band, kind = condition.split("__")[0].split("_")
    f = torch.as_tensor(frequencies, device=w.device)
    mask = (f > (.8 if band == "H" else 2.)) & (f <= 8.)
    if not mask.any():
        raise ValueError("干预频带为空")
    result = w.clone()
    region = w[:, mask]
    if kind == "MEAN":
        result[:, mask] = region.mean(-1, keepdim=True)
    else:
        if shifts.shape != (len(w), 3) or shifts.dtype != torch.int64 or bool(((shifts < 60) | (shifts > 300)).any()):
            raise ValueError("shift 数组不符合冻结合同")
        offset = shifts[:, int(kind[-1])-1].to(w.device)
        indices = (torch.arange(360, device=w.device)[None, :] - offset[:, None]) % 360
        result[:, mask] = region.gather(2, indices[:, None].expand(-1, region.shape[1], -1))
    if not torch.equal(result[:, ~mask], w[:, ~mask]):
        raise RuntimeError("受保护频带改变")
    return result


def captured_forward(model, x, w, fixed=None):
    if model.training or torch.is_grad_enabled():
        raise ValueError("干预仅支持 eval/no_grad")
    modules = {name: m for name, m in model.branches["w"].named_modules() if isinstance(m, nn.GroupNorm)}
    if tuple(modules) != GN_NAMES:
        raise ValueError("W-GN 拓扑漂移")
    stats, used, errors, handles = {}, {}, {}, []

    def hook(name, module):
        def apply(_, inputs, output):
            natural = native_stats(module, inputs[0], output)
            reconstructed = frozen_norm(module, inputs[0], natural, output.dtype)
            torch.testing.assert_close(reconstructed, output, rtol=1e-5, atol=1e-6)
            stats[name] = natural
            selected = natural if fixed is None else fixed[name]
            used[name] = selected
            result = output if fixed is None else replay_norm(module, inputs[0], selected, natural, output)
            if fixed is not None:
                check_formula(result, frozen_norm(module, inputs[0], selected, output.dtype))
            errors[name] = float((reconstructed.float()-output.float()).abs().max())
            return result
        return apply

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(hook(name, module)))
        prediction, capture = forward_with_capture(model, x, tf={"w": w})
    finally:
        for handle in handles:
            handle.remove()
    if tuple(stats) != GN_NAMES:
        raise ValueError("GN 捕获不完整")
    return prediction["waveform"], capture, stats, used, errors


def paired_batch(model, x, w, frequencies, shifts):
    baseline_stats = baseline_z = baseline_prediction = None
    protected_x, protected_w = x.clone(), w.clone()
    with torch.inference_mode():
        for condition in CONDITIONS:
            changed = transform_condition(w, frequencies, condition, shifts)
            fixed = baseline_stats if condition.endswith("FIXED") else None
            with torch.autocast(x.device.type, enabled=x.is_cuda, dtype=torch.bfloat16):
                prediction, capture, stats, used, errors = captured_forward(model, x, changed, fixed)
            if condition == "FULL__NAT":
                baseline_stats, baseline_z, baseline_prediction = stats, capture.z.detach().clone(), prediction.detach().clone()
            else:
                torch.testing.assert_close(capture.z, baseline_z, rtol=0, atol=0)
            if condition == "FULL__FIXED":
                torch.testing.assert_close(prediction, baseline_prediction, rtol=1e-5, atol=1e-6)
            if not torch.equal(x, protected_x) or not torch.equal(w, protected_w):
                raise RuntimeError("干预修改了公共输入")
            values = [prediction, capture.z, capture.z_prime, capture.gamma_raw, capture.beta_raw]
            if any(not torch.isfinite(v).all() for v in values):
                raise FloatingPointError("干预输出/FiLM 张量非有限")
            yield condition, prediction, capture, changed, stats, used, baseline_stats, errors


def evaluate_loader(model, loader, cfg, rep, shifts, case_ids, output, seed):
    offset, metrics = 0, []
    first_stats = True
    with (output / "gn_stats.csv").open("x") as stat_file:
        for batch in loader:
            size = len(batch["x"])
            device = next(model.parameters()).device
            x, w = batch["x"].to(device), batch["tf"]["w"].to(device)
            batch_shifts = torch.as_tensor(shifts[offset:offset+size], dtype=torch.int64, device=device)
            metadata = [_extract_meta(batch["meta"], i) for i in range(size)]
            targets = batch["target"].numpy()
            full_prediction = full_z_prime = None
            for condition, prediction, capture, changed, natural, used, baseline, errors in paired_batch(model, x, w, rep["frequencies_hz"], batch_shifts):
                predicted = prediction.float().cpu().numpy()
                blob = _prediction_dict_from_arrays(predicted, targets, metadata)
                frame = evaluate_task_predictions(blob, cfg, include_test_only=False, method=condition)
                frame.insert(0, "arm", condition)
                frame.insert(0, "seed", seed)
                frame["condition"] = condition
                metrics.append(frame)
                if condition == "FULL__NAT":
                    full_prediction, full_z_prime = predicted.copy(), capture.z_prime.detach().clone()
                stats_rows = []
                for layer in GN_NAMES:
                    # 每层一次批量传到 CPU，避免按窗口/组的数百万次 CUDA 标量同步。
                    stat_arrays = {prefix+"_"+field: getattr(mapping[layer], field).float().cpu().numpy()
                                   for prefix, mapping in (("natural", natural), ("used", used), ("full", baseline))
                                   for field in ("mean", "rstd")}
                    for i, meta in enumerate(metadata):
                        for group in range(natural[layer].mean.shape[1]):
                            record = {"seed": seed, "condition": condition, "dataset_row_id": meta["dataset_row_id"],
                                      "samp_id": meta["samp_id"], "layer": layer, "group": group, "self_replay_max_abs": errors[layer]}
                            record.update({name: float(values[i, group]) for name, values in stat_arrays.items()})
                            stats_rows.append(record)
                pd.DataFrame(stats_rows).to_csv(stat_file, index=False, header=first_stats)
                first_stats = False
                for i, meta in enumerate(metadata):
                    row_id = int(meta["dataset_row_id"])
                    if row_id not in case_ids:
                        continue
                    g, b = .5*torch.tanh(capture.gamma_raw[i]), .5*torch.tanh(capture.beta_raw[i])
                    tensors = {"gamma_raw": capture.gamma_raw[i], "beta_raw": capture.beta_raw[i], "g": g, "b": b,
                               "Z": capture.z[i], "Z_prime": capture.z_prime[i],
                               "modulation_delta": capture.z_prime[i].float()-capture.z[i].float(),
                               "paired_Z_prime_delta": capture.z_prime[i].float()-full_z_prime[i].float(),
                               "actual_cwt": changed[i]}
                    arrays = {k: v.detach().float().cpu().numpy() for k, v in tensors.items()}
                    arrays.update(bcg=batch["x"][i].numpy(), reference=targets[i], prediction=predicted[i],
                                  prediction_delta=predicted[i]-full_prediction[i], frequency_hz=np.array(rep["frequencies_hz"]),
                                  cwt_time_seconds=np.array(rep["time_seconds"]), waveform_time_seconds=np.arange(18000)/100,
                                  latent_time_seconds=(np.arange(140)*128+127.5)/100,
                                  condition_sample_time_seconds=(np.arange(140)[:, None]*128+np.linspace(0,255,5)[None,:])/100)
                    for prefix, raw in (("reference", targets[i]), ("prediction", predicted[i])):
                        arrays.update({prefix+"_"+k: v for k, v in reference_curves(raw, cfg).items()})
                    with (output / f"case_{row_id}_{condition}.npz").open("xb") as stream:
                        np.savez_compressed(stream, **arrays)
                    metric_record = {k: None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v
                                     for k, v in frame.iloc[i].to_dict().items()}
                    io.write_json(output / f"case_{row_id}_{condition}_metrics.json", metric_record)
            offset += size
    if offset != len(loader.dataset):
        raise ValueError("干预分母不完整")
    result = pd.concat(metrics, ignore_index=True)
    result.to_csv(output / "metrics.csv", index=False)
    return result


def run_interventions(session, seed, device, retry=False):
    frozen = io.load_session(session)
    if seed not in SEEDS:
        raise ValueError("未知 seed")
    require_gpu(session, device)
    source = cell_result(session, "A0", seed)
    if source is None:
        raise RuntimeError("B 对应 cell 未完成")
    key = io.binding(session, "mechanisms", seed=seed)
    parent = session / "mechanisms" / f"seed_{seed}"
    prior = io.completed(parent, key)
    if prior:
        return prior
    data_path, _ = data_receipt(session)
    cases = io.read_json(data_path / "cases.json")["dataset_row_ids"]
    estimate = len(cases)*len(CONDITIONS)*(9*96*140+97*360+8*18000)*4 + 2*1024**3
    if shutil.disk_usage(session).free < estimate:
        raise RuntimeError(f"案例和 GN 导出需至少 {estimate} bytes 空闲空间")
    with io.mutex(OUTPUT_ROOT / f".device_{torch.device(device).index}.mutex"):
        with io.attempt(parent, key, retry) as output:
            checkpoint_path = source[1]["sources"]["checkpoint"]
            io.verify(checkpoint_path["path"], checkpoint_path)
            cfg = OmegaConf.load(source[1]["sources"]["config"]["path"])
            cfg.training.device = device
            cfg.training.batch_size = 8  # 固定推理 batch；同批 FULL 负责配对基准。
            model = build_model(seed, frozen["representations"]["A0"]).to(device).eval()
            checkpoint = torch.load(checkpoint_path["path"], map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            # 数据训练合同固定128；先构建后只替换 validation loader。
            cfg.training.batch_size = 128
            data = training_data(session, "A0", cfg)
            from torch.utils.data import DataLoader
            loader = DataLoader(data.val.dataset, batch_size=8, shuffle=False, num_workers=0)
            frame = evaluate_loader(model, loader, cfg, frozen["representations"]["A0"],
                                    np.load(data_path / "val_shift_frames.npy", allow_pickle=False), set(cases), output, seed)
            verify_full_replay(frame, pd.read_csv(source[1]["sources"]["metrics"]["path"]))
            io.write_json(output / "source.json", {"checkpoint": checkpoint_path, "conditions": list(CONDITIONS),
                          "batch_size": 8, "case_ids": cases, "disk_estimate_bytes": estimate, "test": False})
            io.verify_provenance(frozen["provenance"], session)
    return output


def summarize_interventions(session, retry=False):
    from .summary import tables
    io.load_session(session)
    key = io.binding(session, "mechanisms_summary")
    prior = io.completed(session / "mechanisms_summary", key)
    if prior:
        return prior
    frames = []
    for seed in SEEDS:
        path = io.completed(session / "mechanisms" / f"seed_{seed}", io.binding(session, "mechanisms", seed=seed))
        if path is None:
            raise RuntimeError("三个 seed 的机制评价未完成")
        frames.append(pd.read_csv(path / "metrics.csv"))
    full = pd.concat(frames, ignore_index=True)
    with io.attempt(session / "mechanisms_summary", key, retry) as output:
        result = tables(full, {(c, s) for c in CONDITIONS for s in SEEDS}, [(c, "FULL__NAT") for c in CONDITIONS if c != "FULL__NAT"])
        for name, table in result.items():
            table.to_csv(output / f"{name}.csv", index=False)
    return output
