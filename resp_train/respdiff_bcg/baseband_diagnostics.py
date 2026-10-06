"""固定训练 probe 与父窗口验证诊断；所有非有限数值显式失败。"""

import json
import math

import numpy as np
import pandas as pd
import torch

from resp_train.respdiff.model import finite
from .baseband import absolute_stats
from .runtime import LOGGER, write_json
from .signal import parent_noise


def gradient_norm(loss, parameters, *, retain_graph=False):
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    squared = loss.new_zeros((), dtype=torch.float64)
    for value in gradients:
        if value is not None:
            finite("probe parameter gradient", value)
            squared += value.detach().double().square().sum()
    result = float(squared.sqrt())
    if not math.isfinite(result):
        raise FloatingPointError("probe gradient norm 非有限")
    return result


class TrainingProbe:
    """同一 train chunk、噪声与 t 网格；不更新参数、.grad、buffer 或全局 RNG。"""
    def __init__(self, model, dataset, cfg, output, *, tiny=False):
        self.model, self.cfg, self.output = model, cfg, output
        self.updates = [0, 2] if tiny else list(cfg.diagnostics.probe_updates)
        count = min(2 if tiny else int(cfg.diagnostics.probe_chunks), len(dataset))
        self.indices = list(range(count))
        items = [dataset[index] for index in self.indices]
        self.condition = torch.stack([item["x"] for item in items])
        self.target = torch.stack([item["target"] for item in items])
        generator = torch.Generator().manual_seed(int(cfg.diagnostics.noise_seed))
        self.noise = torch.randn(self.target.shape, generator=generator)
        initial = []
        for index in self.indices:
            parent, chunk = divmod(index, 13)
            row = dataset.rows.iloc[parent]
            initial.append(parent_noise(split=str(row.split), row_id=int(row.dataset_row_id),
                                        seed=int(cfg.inference.noise_seed))[chunk])
        self.initial = torch.stack(initial)
        np.savez(output / "probe_inputs.npz", condition=self.condition.numpy(), target=self.target.numpy(),
                 forward_noise=self.noise.numpy(), reverse_initial_noise=self.initial.numpy())
        write_json(output / "probe_identity.json", {"chunk_indices": self.indices,
            "updates": self.updates, "timesteps": list(cfg.diagnostics.timesteps),
            "forward_noise_seed": int(cfg.diagnostics.noise_seed),
            "reverse_noise_seed": int(cfg.inference.noise_seed), "split": "train",
            "first_step_definition": "DDIM reverse t=49 from parent-keyed initial noise",
            "quantiles": "absolute magnitude, linear interpolation"})
        with (output / "training_probes.jsonl").open("x"):
            pass

    def __call__(self, update):
        if update not in self.updates:
            return
        model = self.model
        device = next(model.parameters()).device
        modes = [(module, module.training) for module in model.modules()]
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        diagnostics = model.last_diagnostics
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        devices = [device.index] if device.type == "cuda" else []
        LOGGER.info("固定训练 probe：update=%d，6 个 timestep，chunks=%d", update, len(self.indices))
        try:
            with torch.random.fork_rng(devices=devices):
                condition, target, noise = (value.to(device) for value in
                                             (self.condition, self.target, self.noise))
                model.eval()
                with torch.inference_mode():
                    steps = torch.full((len(target),), 49, dtype=torch.long, device=device)
                    initial = self.initial.to(device)
                    prediction = model.diffusion_model(condition, initial, steps)
                    alpha = model.alpha_torch[49]
                    x0 = (initial - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
                    first = {f"first_step_x0_{key}": value for key, value in absolute_stats(x0).items()}
                # cuDNN RNN backward 要求 train mode；网络没有 dropout，BN 不保存 running stats。
                model.train()
                for step in self.cfg.diagnostics.timesteps:
                    steps = torch.full((len(target),), int(step), dtype=torch.long, device=device)
                    losses = model.training_loss(condition, target, noise=noise, step=steps)
                    noise_norm = gradient_norm(losses["loss_noise"], parameters, retain_graph=True)
                    spec_norm = gradient_norm(losses["loss_spec_weighted"], parameters)
                    if noise_norm == 0:
                        raise FloatingPointError("probe 主损失参数梯度为零，r_t 无定义")
                    ratio = spec_norm / noise_norm
                    if not math.isfinite(ratio):
                        raise FloatingPointError("probe r_t 非有限")
                    record = {"update": update, "timestep": int(step), "objective": model.objective,
                        **{key: float(value.detach()) for key, value in losses.items()},
                        **{key: value for key, value in model.last_diagnostics.items() if key != "timestep"},
                        **first, "grad_noise_norm": noise_norm, "grad_weighted_spec_norm": spec_norm,
                        "r_t": ratio, "spectral_gradient_exceeds_noise": ratio > 1}
                    with (self.output / "training_probes.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                    LOGGER.info("probe update=%d t=%d r_t=%.6g first-step |x0| max=%.6g",
                                update, step, ratio, first["first_step_x0_max"])
        finally:
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
            for module, training in modes:
                module.training = training
            model.last_diagnostics = diagnostics


def waveform_diagnostics(values, *, fs=100):
    """去均值后单边能量比；幅值与 top 1% 能量使用原始波形。"""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (18000,) or not np.isfinite(values).all():
        raise ValueError("输出诊断要求有限的 18000 点父窗口")
    centered = values - values.mean()
    spectrum = np.abs(np.fft.rfft(centered, norm="ortho")) ** 2
    spectrum[1:-1] *= 2  # Parseval 单边谱的非 DC/Nyquist 权重。
    frequencies = np.fft.rfftfreq(len(values), d=1 / fs)
    total = float(spectrum.sum())
    energy = values ** 2
    raw_total = float(energy.sum())
    top_count = math.ceil(.01 * len(values))
    result = {"resp_energy_ratio": float(spectrum[(frequencies >= .05) & (frequencies <= .70)].sum() / total)
              if total else None,
              "above_1hz_energy_ratio": float(spectrum[frequencies > 1].sum() / total) if total else None,
              "std": float(values.std()), "max_abs": float(np.abs(values).max()),
              "top_1pct_sample_energy_ratio": float(np.partition(energy, -top_count)[-top_count:].sum() / raw_total)
              if raw_total else None,
              "zero_centered_energy": total == 0, "zero_raw_energy": raw_total == 0}
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("validation 输出诊断非有限")
    return result


def validation_diagnostics(output):
    with np.load(output / "validation_waveforms.npz", allow_pickle=False) as payload:
        predictions = payload["r_tho_hat"]
        targets = payload["tho_ref"]
        row_ids = payload["dataset_row_id"]
        sample_ids = payload["samp_id"]
    metrics = pd.read_csv(output / "validation_per_parent.csv")
    if len(metrics) != len(row_ids) or not np.array_equal(metrics.dataset_row_id, row_ids):
        raise ValueError("validation metrics 与输出身份不一致")
    records = []
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        pred = waveform_diagnostics(prediction)
        ref = waveform_diagnostics(target)
        records.append({"dataset_row_id": int(row_ids[index]), "samp_id": int(sample_ids[index]),
            **{f"prediction_{key}": value for key, value in pred.items()},
            **{f"target_{key}": value for key, value in ref.items()},
            "prediction_target_std_ratio": pred["std"] / ref["std"] if ref["std"] else None,
            "zero_target_std": ref["std"] == 0})
    frame = pd.DataFrame(records)
    numeric = frame.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise FloatingPointError("validation 比值溢出")
    frame.to_csv(output / "validation_output_diagnostics.csv", index=False)
    summary = {"n_parents": len(frame), "reference": "1Hz LPF THO at 100Hz",
        "spectral_definition": "demeaned full-parent rectangular rFFT, one-sided Parseval energy",
        "top_1pct_definition": "largest 180 squared time samples / full-parent raw squared energy",
        "undefined_policy": "null only for explicit zero denominator; counts reported"}
    for channel in ("prediction", "target"):
        for quantile in (.95, .99):
            summary[f"{channel}_parent_max_p{int(100 * quantile)}"] = float(
                frame[f"{channel}_max_abs"].quantile(quantile))
    for column in frame.columns:
        if "ratio" in column:
            defined = frame[column].notna()
            summary[column] = {"mean": float(frame.loc[defined, column].mean()) if defined.any() else None,
                               "n_defined": int(defined.sum()), "n_undefined": int((~defined).sum())}
    selected = frame.dataset_row_id == 10382
    # 异常历史 row 必须显式报告存在性；不额外读取 train/test 或更换样本。
    summary["row_10382"] = {"present": bool(selected.any()),
        "output": json.loads(frame.loc[selected].to_json(orient="records")),
        "metrics": json.loads(metrics.loc[metrics.dataset_row_id == 10382].to_json(orient="records"))}
    write_json(output / "validation_output_summary.json", summary)
