#!/usr/bin/env python3
"""有界 validation 诊断：原批次六/五十步 DDIM 轨迹与逐 t 损失梯度。"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.data.research_v2 import ResearchV2WindowDataset
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.respdiff.model import finite
from resp_train.respdiff_bcg.config import load_config
from resp_train.respdiff_bcg.data import ChunkDataset
from resp_train.respdiff_bcg.model import TIMESTEPS
from resp_train.respdiff_bcg.runtime import (
    LOGGER, artifact_manifest, environment, make_model, run_logging, seed_all,
    sha256, source_file_inventory, write_json,
)
from resp_train.respdiff_bcg.signal import parent_noise, restore_parent

BATCHES = (0, 177, 178)
PROBE_STEPS = (0, 9, 19, 29, 39, 49)
PROBE_SEED = 20261004


def summary_tensor(value):
    value = value.detach().float()
    finite("diagnostic tensor", value)
    absolute = value.abs().flatten()
    return {"max_abs": float(absolute.max()), "p99_abs": float(torch.quantile(absolute, .99)),
            "p999_abs": float(torch.quantile(absolute, .999)),
            "rms": float(value.square().mean().sqrt())}


@torch.inference_mode()
def trace_ddim(model, condition, initial, steps):
    """保持生产采样公式、FP32 与完整 batch，只增加轨迹记录。"""
    steps = tuple(steps)
    if model.training or steps[0] != 49 or steps[-1] != 0 or any(a <= b for a, b in zip(steps, steps[1:])):
        raise ValueError("要求 eval 模式及从 49 严格递减到 0 的 steps")
    if condition.shape != initial.shape:
        raise ValueError("condition/noise shape 不匹配")
    current = initial.clone()
    states, x0s, noises, records = [current.cpu().numpy()], [], [], []
    for index, step in enumerate(steps):
        t = torch.full((len(condition),), step, dtype=torch.long, device=condition.device)
        predicted = model.diffusion_model(condition, current, t)
        alpha = model.alpha_torch[step]
        following = model.alpha_torch[steps[index + 1]] if index + 1 < len(steps) else torch.ones_like(alpha)
        x0 = (current - (1 - alpha).sqrt() * predicted) / alpha.sqrt()
        next_state = following.sqrt() * x0 + (1 - following).sqrt() * predicted
        record = {"t": step, "next_t": steps[index + 1] if index + 1 < len(steps) else -1}
        for name, value in (("xt", current), ("epsilon_hat", predicted), ("x0_hat", x0), ("next", next_state)):
            record.update({f"{name}_{key}": val for key, val in summary_tensor(value).items()})
        spectrum = torch.fft.rfft(x0 - x0.mean(-1, keepdim=True), norm="ortho")
        power = spectrum.abs().square()
        power[..., 1:-1] *= 2
        frequencies = torch.fft.rfftfreq(x0.shape[-1], 1 / 20, device=x0.device)
        record["x0_above_2hz_energy_fraction"] = float(power[..., frequencies > 2].sum() / power.sum().clamp_min(1e-30))
        records.append(record)
        x0s.append(x0.cpu().numpy())
        noises.append(predicted.cpu().numpy())
        current = next_state
        states.append(current.cpu().numpy())
        if len(steps) == 6 or index in (0, len(steps) - 1) or (index + 1) % 10 == 0:
            LOGGER.info("DDIM nfe=%d t=%d x0_max=%.6g x0_p99=%.6g next_max=%.6g",
                        len(steps), step, record["x0_hat_max_abs"], record["x0_hat_p99_abs"], record["next_max_abs"])
    return current.cpu().numpy(), {"states": np.stack(states), "x0_hat": np.stack(x0s),
                                  "epsilon_hat": np.stack(noises), "timesteps": np.array(steps)}, records


def gradient_geometry(left, right):
    a = sum(float(g.detach().double().square().sum()) for g in left)
    b = sum(float(g.detach().double().square().sum()) for g in right)
    dot = sum(float((g.detach().double() * h.detach().double()).sum()) for g, h in zip(left, right))
    if not np.isfinite([a, b, dot]).all():
        raise FloatingPointError("分项梯度非有限")
    return {"noise_grad_norm": a ** .5, "weighted_fft_grad_norm": b ** .5,
            "gradient_norm_ratio": (b / a) ** .5 if a > 0 else None,
            "gradient_cosine": dot / (a * b) ** .5 if a > 0 and b > 0 else None}


def probe_timestep(model, condition, target, noise, step):
    # cuDNN RNN backward 要求 train 模式；本网络无 dropout、BN 不保存运行统计。
    model.train()
    t = torch.full((len(condition),), step, dtype=torch.long, device=condition.device)
    alpha = model.alpha_torch[t]
    xt = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
    predicted = model.diffusion_model(condition, xt, t)
    error = noise - predicted
    reconstructed = (xt - (1 - alpha).sqrt() * predicted) / alpha.sqrt()
    noise_rows = error.square().mean((1, 2))
    fft_rows = (torch.fft.fft(reconstructed, norm="ortho").abs()
                - torch.fft.fft(target, norm="ortho").abs()).square().mean((1, 2))
    loss_noise, loss_fft = noise_rows.mean(), .01 * fft_rows.mean()
    finite("probe loss_noise", loss_noise)
    finite("probe weighted_fft", loss_fft)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    g_noise = torch.autograd.grad(loss_noise, parameters, retain_graph=True)
    g_fft = torch.autograd.grad(loss_fft, parameters)
    geometry = gradient_geometry(g_noise, g_fft)
    row = {"t": step, "loss_noise": float(loss_noise.detach()),
           "weighted_fft_loss": float(loss_fft.detach()), **geometry,
           **{f"x0_{key}": value for key, value in summary_tensor(reconstructed).items()},
           "epsilon_hat_max_abs": float(predicted.detach().abs().max()),
           "epsilon_max_abs": float(noise.abs().max())}
    sample = pd.DataFrame({"t": step, "loss_noise": noise_rows.detach().cpu().numpy(),
                           "weighted_fft_loss": .01 * fft_rows.detach().cpu().numpy()})
    # 保存输出层上的误差尾部分布，帮助区分总梯度与少数极端坐标。
    details = {"true_noise": noise.detach().cpu().numpy(), "predicted_noise": predicted.detach().cpu().numpy(),
               "x0_hat": reconstructed.detach().cpu().numpy()}
    model.eval()
    return row, sample, details


def completed_parents(predictions):
    parents = sorted({index // 13 for index in predictions})
    return [parent for parent in parents if all(parent * 13 + chunk in predictions for chunk in range(13))]


def state_digest(model):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(source, output, device, *, confirm=False):
    if not confirm:
        raise ValueError("需要 --confirm-validation-diagnostic 才能读取真实 validation 输入并调用 GPU")
    if not str(device).startswith("cuda:"):
        raise ValueError("此有限 GPU 诊断要求明确 cuda:N")
    output.mkdir(parents=True, exist_ok=False)
    with run_logging(output):
        stage = "provenance"
        try:
            receipt = json.loads((source / "receipt.json").read_text())
            if receipt["status"] != "complete" or receipt["mode"] != "train":
                raise ValueError("来源必须是完成的 train/validation run")
            names = ("final.pt", "val_parents.csv", "source_identity.json", "data_identity.json",
                     "resolved_config.yaml", "validation_waveforms.npz", "validation_batches.json")
            for name in names:
                if sha256(source / name) != receipt["artifacts"][name]:
                    raise ValueError(f"来源产物 hash 变化：{name}")
            identity = json.loads((source / "source_identity.json").read_text())
            for relative, expected in identity["files"].items():
                if sha256(ROOT / relative) != expected:
                    raise ValueError(f"来源代码漂移：{relative}")
            cfg = load_config(source / "resolved_config.yaml")
            if int(cfg.training.seed) != 20260811 or int(cfg.inference.batch_size) != 64:
                raise ValueError("诊断合同固定 seed=20260811、原验证 batch=64")
            rows = pd.read_csv(source / "val_parents.csv")
            if len(rows) != 2675 or set(rows.split) != {"val"} or int(rows.iloc[876].dataset_row_id) != 10382:
                raise ValueError("来源父窗口身份与预先选择不匹配")
            batches = json.loads((source / "validation_batches.json").read_text())
            if batches["ordered_chunk_indices"] != list(range(len(rows) * 13)):
                raise ValueError("来源推理批次不满足连续顺序")
            indices = [index for batch in BATCHES for index in range(batch * 64, (batch + 1) * 64)]
            parent_indices = sorted({index // 13 for index in indices})
            selected_rows = rows.iloc[parent_indices]
            data_identity = json.loads((source / "data_identity.json").read_text())
            index_path = Path(data_identity["index_path"])
            if sha256(index_path) != data_identity["index_sha256"]:
                raise ValueError("原始数据索引变化")
            inventory = source_file_inventory(selected_rows)
            if any(info != data_identity["source_file_stats"][path] for path, info in inventory.items()):
                raise ValueError("选中 validation 源文件 stat 变化")
            manifest = {"source_run": str(source.resolve()), "checkpoint_sha256": receipt["artifacts"]["final.pt"],
                        "original_batch_indices": BATCHES, "global_chunk_indices": indices,
                        "selected_parent_indices": parent_indices, "selected_parent_row_ids": selected_rows.dataset_row_id.tolist(),
                        "selection": "first original batch and both batches containing known outlier row 10382",
                        "sampling_steps": [list(TIMESTEPS), list(range(49, -1, -1))],
                        "probe_batch_index": 0, "probe_steps": PROBE_STEPS, "probe_seed": PROBE_SEED,
                        "scope": "diagnostic selection; not an unbiased performance estimate",
                        "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "source_file_stats": inventory, "script_sha256": sha256(Path(__file__))}
            write_json(output / "diagnostic_plan.json", manifest)
            write_json(output / "environment.json", environment())
            LOGGER.info("有限诊断：seed=20260811 batches=%s，192 chunks，%d 个输入父窗口", BATCHES, len(parent_indices))
            stage = "load_model_and_validation"
            seed_all(int(cfg.training.seed))
            torch.set_num_threads(1)
            torch.cuda.set_device(device)
            checkpoint = torch.load(source / "final.pt", map_location="cpu", mmap=True, weights_only=False)
            if checkpoint["schema"] != "respdiff-bcg-v1" or checkpoint["update"] != 6400:
                raise ValueError("checkpoint schema/update 不匹配")
            model = make_model(cfg).to(device)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            original_state = state_digest(model)
            del checkpoint
            dataset = ChunkDataset(ResearchV2WindowDataset(index_path, rows, cfg, preload_windows=False), rows)
            saved = np.load(source / "validation_waveforms.npz")
            saved_low, saved_ref = saved["prediction_20hz"], saved["tho_ref"]
            saved.close()
            outputs = {6: {}, 50: {}}
            trace_records, batch_data = [], {}
            for batch_index in BATCHES:
                stage = f"batch_{batch_index}_load"
                current_indices = list(range(batch_index * 64, (batch_index + 1) * 64))
                items = [dataset[index] for index in current_indices]
                condition = torch.stack([item["x"] for item in items])
                target = torch.stack([item["target"] for item in items])
                initial = torch.stack([parent_noise(split="val", row_id=int(rows.iloc[index // 13].dataset_row_id),
                              seed=int(cfg.inference.noise_seed))[index % 13] for index in current_indices])
                batch_data[batch_index] = (condition, target, initial)
                np.savez(output / f"batch_{batch_index}_inputs.npz", global_chunk_indices=current_indices,
                         condition=condition.numpy(), target=target.numpy(), initial_noise=initial.numpy())
            for steps in (TIMESTEPS, tuple(range(49, -1, -1))):
                for batch_index in BATCHES:
                    current_indices = list(range(batch_index * 64, (batch_index + 1) * 64))
                    condition, target, initial = (tensor.to(device) for tensor in batch_data[batch_index])
                    stage = f"batch_{batch_index}_ddim_{len(steps)}"
                    LOGGER.info("开始 %s", stage)
                    final, trace, records = trace_ddim(model, condition, initial, steps)
                    outputs[len(steps)].update(zip(current_indices, final))
                    trace_records.extend({"original_batch": batch_index, "nfe": len(steps), **r} for r in records)
                    np.savez(output / f"batch_{batch_index}_ddim{len(steps)}_trace.npz",
                             global_chunk_indices=current_indices, **trace)
                    del trace
                    del condition, target, initial
                if len(steps) == 6:
                    stage = "six_step_replay_check"
                    complete = completed_parents(outputs[6])
                    replay = np.stack([restore_parent(np.stack([outputs[6][parent * 13 + c] for c in range(13)]),
                                            chunk_indices=range(13))[0] for parent in complete])
                    error = float(np.max(np.abs(replay - saved_low[complete])))
                    np.testing.assert_allclose(replay, saved_low[complete], rtol=1e-4, atol=1e-4)
                    write_json(output / "six_step_replay_check.json", {"passed": True, "max_abs_error": error,
                              "rtol": 1e-4, "atol": 1e-4, "complete_parent_indices": complete})
                    LOGGER.info("六步保存结果复现通过：max_abs_error=%.6g，完整父窗口=%d", error, len(complete))
            pd.DataFrame(trace_records).to_csv(output / "sampling_trace_summary.csv", index=False)
            stage = "reconstruction_and_replay_check"
            complete = completed_parents(outputs[6])
            if complete != completed_parents(outputs[50]) or 876 not in complete:
                raise ValueError("采样分支的完整父窗口集合不一致")
            metric_frames, parent_stats = [], []
            for nfe in (6, 50):
                low, high = zip(*(restore_parent(np.stack([outputs[nfe][parent * 13 + c] for c in range(13)]),
                                                   chunk_indices=range(13)) for parent in complete))
                low, high = np.stack(low), np.stack(high)
                payload = {"r_tho_hat": high, "tho_ref": saved_ref[complete],
                           "dataset_row_id": rows.iloc[complete].dataset_row_id.to_numpy(dtype=np.int64),
                           "samp_id": rows.iloc[complete].samp_id.to_numpy(dtype=np.int64),
                           "split": np.full(len(complete), "val")}
                metrics = evaluate_task_predictions(payload, cfg, method=f"diagnostic_ddim{nfe}")
                metrics["nfe"] = nfe
                metric_frames.append(metrics)
                np.savez(output / f"selected_parents_ddim{nfe}.npz", prediction_20hz=low, **payload)
                for i, parent in enumerate(complete):
                    centered = low[i] - low[i].mean()
                    power = abs(np.fft.rfft(centered, norm="ortho")) ** 2
                    power[1:-1] *= 2
                    frequencies = np.fft.rfftfreq(3600, 1 / 20)
                    parent_stats.append({"nfe": nfe, "parent_index": parent,
                        "dataset_row_id": int(rows.iloc[parent].dataset_row_id), "max_abs": float(abs(low[i]).max()),
                        "std": float(low[i].std()), "above_2hz_energy_fraction": float(power[frequencies > 2].sum() / power.sum())})
            pd.concat(metric_frames, ignore_index=True).to_csv(output / "selected_parent_metrics.csv", index=False)
            pd.DataFrame(parent_stats).to_csv(output / "selected_parent_signal_stats.csv", index=False)
            stage = "timestep_loss_gradient_probe"
            condition, target = (tensor.to(device) for tensor in batch_data[0][:2])
            generator = torch.Generator().manual_seed(PROBE_SEED)
            noise = torch.randn(target.shape, generator=generator, dtype=torch.float32).to(device)
            summaries, samples = [], []
            for step in PROBE_STEPS:
                summary, sample, details = probe_timestep(model, condition, target, noise, step)
                sample["global_chunk_index"] = np.arange(64)
                summaries.append(summary)
                samples.append(sample)
                np.savez(output / f"probe_t{step}.npz", **details)
                LOGGER.info("probe t=%d noise=%.6g weighted_fft=%.6g grad_ratio=%s x0_max=%.6g", step,
                            summary["loss_noise"], summary["weighted_fft_loss"],
                            summary["gradient_norm_ratio"], summary["x0_max_abs"])
            pd.DataFrame(summaries).to_csv(output / "timestep_probe_summary.csv", index=False)
            pd.concat(samples, ignore_index=True).to_csv(output / "timestep_probe_per_chunk.csv", index=False)
            stage = "final_checks"
            if state_digest(model) != original_state:
                raise ValueError("诊断意外改变模型参数或 buffer")
            if inventory != source_file_inventory(selected_rows):
                raise ValueError("诊断期间 validation 源文件变化")
            for relative, expected in identity["files"].items():
                if sha256(ROOT / relative) != expected:
                    raise ValueError(f"诊断期间模型源码变化：{relative}")
            if sha256(Path(__file__)) != manifest["script_sha256"]:
                raise ValueError("诊断脚本运行期间变化")
            LOGGER.info("诊断完成：参数保持不变；写入独立诊断回执")
            write_json(output / "receipt.json", {"status": "complete", "mode": "validation_diagnostic",
                       "formal_evidence": False, "optimizer_updates": 0, "artifacts": artifact_manifest(output)})
            print(f"完成：{output / 'receipt.json'}", flush=True)
        except BaseException as exc:
            LOGGER.error("诊断失败：stage=%s %s: %s", stage, type(exc).__name__, exc)
            LOGGER.debug("异常堆栈", exc_info=True)
            write_json(output / "failure.json", {"status": "failed", "stage": stage,
                       "error_type": type(exc).__name__, "error": str(exc)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/respdiff_bcg_v1/train_b64_seed20260811_r01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirm-validation-diagnostic", action="store_true")
    args = parser.parse_args()
    run(args.source, args.output, torch.device(args.device), confirm=args.confirm_validation_diagnostic)
