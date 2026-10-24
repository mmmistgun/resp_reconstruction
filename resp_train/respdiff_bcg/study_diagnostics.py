"""分 t 误差、保 batch 的条件错配与反推轨迹诊断。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader

from resp_train.respdiff.model import finite
from .baseband import absolute_stats
from .inference_v2 import evaluate_setting
from .runtime import LOGGER, Progress, write_json
from .sampling_accelerated import AccelerationSpec, accelerated_prefixes
from .study_contract import NAME, PROBE_STEPS, PROBE_SEEDS, PAIRING_SEEDS, StudySamplerSpec
from .study_state import preserved_evaluation


def chunk_keys(dataset, indices):
    keys = []
    for index in indices:
        parent, chunk = divmod(int(index), 13)
        row = dataset.rows.iloc[parent]
        keys.append((str(row.split), int(row.dataset_row_id), chunk))
    return keys


def forward_noise(keys, seed, device):
    values = []
    for key in keys:
        identity = json.dumps([NAME, "forward", seed, list(key)])
        local = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % (2**63)
        values.append(torch.randn(1, 600, generator=torch.Generator(device="cpu").manual_seed(local)))
    return torch.stack(values).to(device)


def condition_permutation(keys, seed):
    count = len(keys)
    if not count or len(set(keys)) != count:
        raise ValueError("条件错配要求唯一的非空 chunk 身份")
    if count == 1:
        return [0], {"status": "single_chunk_identity", "same_parent": 1, "fixed_points": 1}
    digest = hashlib.sha256(json.dumps([seed, keys]).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    parents = np.asarray([key[1] for key in keys])
    # 先最小化同 parent 配对，再用独立随机代价确定置换；自身配对严格最贵。
    cost = rng.random((count, count)) + (parents[:, None] == parents[None, :]) * (count + 1)
    np.fill_diagonal(cost, (count + 1)**3)
    left, right = linear_sum_assignment(cost)
    if left.tolist() != list(range(count)) or np.any(left == right):
        raise ValueError("没有得到无自身配对的双射")
    return right.tolist(), {"status": "derangement", "same_parent": int(np.sum(parents == parents[right])),
                             "fixed_points": 0}


def error_statistics(target, noise, predicted, alpha):
    if target.shape != noise.shape or noise.shape != predicted.shape:
        raise ValueError("诊断 target/noise/prediction shape 不匹配")
    for name, value in (("target", target), ("noise", noise), ("prediction", predicted), ("alpha", alpha)):
        finite(name, value)
    if alpha.numel() != 1 or not 0 < float(alpha) < 1:
        raise ValueError("分 t 诊断只接受单个有效 alpha_bar")
    error = (noise.double() - predicted.double()).flatten(1)
    rms = error.square().mean(1).sqrt()
    target_rms = target.double().flatten(1).square().mean(1).sqrt()
    gain = float(((1 - alpha.double()) / alpha.double()).sqrt())
    quantiles = torch.quantile(error.abs(), error.new_tensor([.99, .999]), dim=1)
    records = []
    for i in range(len(target)):
        tr, er = float(target_rms[i]), float(rms[i])
        records.append({"epsilon_mse": er**2, "epsilon_rms": er,
            "epsilon_p99": float(quantiles[0, i]), "epsilon_p999": float(quantiles[1, i]),
            "epsilon_max": float(error[i].abs().max()), "target_rms": tr,
            "x0_rmse": er * gain, "x0_normalized_rmse": er * gain / tr if tr else None,
            "zero_x0_epsilon_mse": tr**2 / gain**2,
            "epsilon_mse_over_zero_x0": (er * gain / tr)**2 if tr else None,
            "zero_target_rms": tr == 0, "rms_amplification": gain})
    return records, error


def run_forward_probes(model, dataset, output, *, batch_size=64, steps=PROBE_STEPS,
                       noise_seeds=PROBE_SEEDS, pairing_seeds=PAIRING_SEEDS):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator(device="cpu").manual_seed(0))
    device = next(model.parameters()).device
    progress = Progress("分 t / condition probe", len(loader), every=1)
    records, permutations, temporal = [], [], {}
    with preserved_evaluation(model):
        for batch_index, batch in enumerate(loader):
            indices = batch["index"].tolist()
            keys = chunk_keys(dataset, indices)
            target, condition = batch["target"].to(device), batch["x"].to(device)
            views = [("correct", list(range(len(keys))))]
            for seed in pairing_seeds:
                perm, status = condition_permutation(keys, seed)
                view = f"permuted_{seed}"
                views.append((view, perm))
                permutations.append({"batch": batch_index, "view": view, "keys": keys,
                                     "donor_indices": perm, **status})
            for seed in noise_seeds:
                noise = forward_noise(keys, seed, device)
                for t in steps:
                    alpha = model.alpha_torch[t]
                    xt = alpha.sqrt() * target + (1-alpha).sqrt() * noise
                    timestep = torch.full((len(target),), t, dtype=torch.long, device=device)
                    for view, perm in views:
                        prediction = model.diffusion_model(condition[perm], xt, timestep)
                        stats, error = error_statistics(target, noise, prediction, alpha)
                        for key, chunk, stat in zip(keys, indices, stats):
                            records.append({"split": key[0], "dataset_row_id": key[1], "chunk_index": key[2],
                                "global_chunk_index": chunk, "batch": batch_index, "timestep": t,
                                "noise_seed": seed, "view": view, **stat})
                        group = (t, seed, view)
                        square = error.square().sum(0).cpu().numpy()
                        maximum = error.abs().amax(0).cpu().numpy()
                        if group not in temporal:
                            temporal[group] = [square, maximum, len(error)]
                        else:
                            temporal[group][0] += square
                            temporal[group][1] = np.maximum(temporal[group][1], maximum)
                            temporal[group][2] += len(error)
            progress.update(batch_index + 1)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "timestep_metrics.csv", index=False)
    base = frame[frame.view == "correct"]
    pairs = frame[frame.view != "correct"].merge(base[["global_chunk_index", "timestep", "noise_seed", "epsilon_mse", "x0_rmse"]],
        on=["global_chunk_index", "timestep", "noise_seed"], suffixes=("", "_correct"), validate="many_to_one")
    pairs["epsilon_mse_delta"] = pairs.epsilon_mse - pairs.epsilon_mse_correct
    pairs["x0_rmse_delta"] = pairs.x0_rmse - pairs.x0_rmse_correct
    pairs.to_csv(output / "condition_pairing_metrics.csv", index=False)
    position = []
    for (t, seed, view), (square, maximum, count) in temporal.items():
        position.extend({"timestep": t, "noise_seed": seed, "view": view, "position_20hz": i,
                         "epsilon_rms": float(np.sqrt(square[i]/count)), "epsilon_max": float(maximum[i]),
                         "n_chunks": count} for i in range(len(square)))
    pd.DataFrame(position).to_csv(output / "position_error_metrics.csv", index=False)
    write_json(output / "condition_permutations.json", permutations)
    return frame


def _trace_record(event, keys, batch_index, spec):
    current = event["current"].reshape(-1, 1, 600)
    epsilon = event["epsilon"].reshape_as(current)
    alpha = event["alpha"]
    x0 = (current - (1-alpha).sqrt() * epsilon) / alpha.sqrt()
    finite("trace x0", x0)
    flat = int(x0.abs().argmax())
    sample, position = divmod(flat, 600)
    key = keys[sample % len(keys)]
    return {"sampler": spec.sampler, "S": spec.nfe, "N": spec.n_trajectories,
            "batch": batch_index, "trajectory_id": event["trajectory_ids"][sample // len(keys)],
            "timestep": event["timestep"], "worst_dataset_row_id": key[1],
            "worst_chunk_index": key[2], "worst_position_20hz": position,
            **{f"xt_{k}": v for k,v in absolute_stats(current).items()},
            **{f"x0_{k}": v for k,v in absolute_stats(x0).items()},
            **{f"next_{k}": v for k,v in absolute_stats(event["reverse"]).items()}}


def sample_paired(model, dataset, spec, output, *, batch_size=64, pairing_seeds=PAIRING_SEEDS,
                  trace=True):
    """同一 batch 内错配条件，保持 target 和随机轨迹身份不变。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = next(model.parameters()).device
    views = ["correct", *[f"permuted_{seed}" for seed in pairing_seeds]]
    collected = {view: [] for view in views}
    trace_rows, mappings = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator(device="cpu").manual_seed(0))
    with preserved_evaluation(model):
        for batch_index, batch in enumerate(loader):
            keys = chunk_keys(dataset, batch["index"].tolist())
            condition = batch["x"].to(device)
            for view in views:
                if view == "correct":
                    perm, status = list(range(len(keys))), {"status": "paired"}
                else:
                    perm, status = condition_permutation(keys, int(view.split("_")[-1]))
                mappings.append({"batch": batch_index, "view": view, "keys": keys,
                                 "donor_indices": perm, **status})
                def observer(event):
                    # 第一条轨迹用于逐步定位；ensemble 质量仍使用全部 N 条。
                    if trace and view == "correct" and 0 in event["trajectory_ids"]:
                        trace_rows.append(_trace_record({**event, "alpha": model.alpha_torch[event["timestep"]]},
                                                        keys, batch_index, spec))
                iterator = accelerated_prefixes(model, condition[perm], keys, spec, AccelerationSpec(1),
                    checkpoints=(spec.n_trajectories,), observer=observer if trace else None)
                for _, mean in iterator:
                    collected[view].append(mean.cpu().numpy())
            LOGGER.info("%s S%d N%d paired sampling batch=%d/%d", spec.sampler, spec.nfe,
                        spec.n_trajectories, batch_index+1, len(loader))
    write_json(output / "sampling_identity.json", {"spec": spec.identity(), "batch_size": batch_size,
               "pairings": mappings, "trace_trajectory_ids": [0] if trace else [], "trajectory_group_size": 1})
    pd.DataFrame(trace_rows).to_csv(output / "reverse_trace_summary.csv", index=False)
    return {view: np.concatenate(values).reshape(len(dataset.rows), 13, 1, 600)
            for view, values in collected.items()}


def evaluate_anchors(model, dataset, cfg, output, *, batch_size=64, pairing_seeds=(), tiny=False,
                     trace=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    curves = []
    for spec in (StudySamplerSpec("ddim", 6, 1 if tiny else 16), StudySamplerSpec("ddpm", 50, 1 if tiny else 4)):
        chunks = sample_paired(model, dataset, spec, output / f"{spec.sampler}_sampling",
                               batch_size=batch_size, pairing_seeds=pairing_seeds, trace=trace)
        for view, values in chunks.items():
            path = output / f"{spec.sampler}_S{spec.nfe}_N{spec.n_trajectories}_{view}"
            result = evaluate_setting(values, dataset, cfg, path, spec, spec.n_trajectories,
                                      method_prefix=f"RespDiff-study-v1/anchor/S{spec.nfe}/{view}")
            curves.append({"view": view, **result})
    pd.DataFrame(curves).to_csv(output / "anchor_metrics.csv", index=False)
    return curves
