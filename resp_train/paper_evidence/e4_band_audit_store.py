"""完整特征的无损磁盘记录、流式描述统计及预先固定窗口的图。"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from resp_train.paper_evidence.e4_band_audit import finite, REGIONS
from resp_train.paper_evidence.e1_scale_topology_runtime import write_json


def feature_reader(directory):
    import json
    metadata = json.loads((directory/"features.json").read_text())
    array = np.load(directory/"x.npy", mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != tuple(metadata["shape"]) or str(array.dtype) != metadata["numpy_dtype"]:
        raise ValueError("保存的 X shape/dtype 漂移")

    def read(offset, count):
        values = torch.from_numpy(np.array(array[offset:offset+count], copy=True))
        if len(values) != count:
            raise ValueError("固定 X 不足当前 batch")
        return values.view(torch.bfloat16) if metadata["torch_dtype"] == "torch.bfloat16" else values
    return read


class FeatureStore:
    def __init__(self, directory, rows, plot_row_ids, frequencies):
        self.directory, self.rows = directory, rows
        self.plot_ids, self.frequencies = set(plot_row_ids), frequencies
        self.array = self.alpha = None
        self.records = []
        self.mass_sum = self.mass_square_sum = None
        self.count = 0

    def __call__(self, offset, values):
        if offset != self.count:
            raise ValueError("X 写入顺序漂移")
        x, alpha, delta, z = (values[k].detach().cpu() for k in ("x", "alpha", "delta", "z"))
        for value in (x, alpha, delta, z):
            finite(value)
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("未支持的 X 精度")
        bits = x.view(torch.uint16).numpy() if x.dtype == torch.bfloat16 else x.numpy()
        if self.array is None:
            self.array = np.lib.format.open_memmap(self.directory/"x.npy", mode="w+", dtype=bits.dtype,
                                                  shape=(len(self.rows), *x.shape[1:]))
            write_json(self.directory/"features.json", {"torch_dtype": str(x.dtype), "numpy_dtype": str(bits.dtype),
                "shape": list(self.array.shape), "encoding": "BF16 raw bits as uint16; FP32 unchanged"})
            dynamic = alpha.shape[-1] == 360
            self.alpha = np.lib.format.open_memmap(self.directory/"alpha.npy", mode="w+", dtype=np.float32,
                shape=((len(self.rows) if dynamic else 1), *alpha.shape[1:]))
            write_json(self.directory/"alpha.json", {"dynamic": dynamic, "shape": list(self.alpha.shape),
                "axes": ["window_or_shared", "channel_or_shared", "scale", "time_or_shared"]})
        self.array[offset:offset+len(x)] = bits
        if self.alpha.shape[0] == 1:
            if offset and not np.array_equal(self.alpha, alpha.numpy()):
                raise ValueError("静态权重随 batch 改变")
            self.alpha[:] = alpha.numpy()
        else:
            self.alpha[offset:offset+len(x)] = alpha.numpy()
        # 逐窗口处理 FP32 统计，避免同时物化整个 batch 的多个 B×96×97×360 副本。
        for i in range(len(x)):
            a = alpha[i if len(alpha)>1 else 0].float()
            d = delta[i if len(delta)>1 else 0].float()
            value = x[i].float()
            mean = x[i].mean(1).float()
            terms = value*d
            ideal = terms.sum(1)
            actual = z[i].float()-mean
            denominator = float(torch.linalg.vector_norm(terms, dim=(0, 2)).sum())
            rms_mean = float(mean.square().mean().sqrt())
            rms_ideal = float(ideal.square().mean().sqrt())
            rms_actual = float(actual.square().mean().sqrt())
            record = self.rows.iloc[offset+i][["dataset_row_id", "samp_id", "split"]].to_dict()
            record.update(mean_rms=rms_mean, delta_ideal_rms=rms_ideal, delta_executed_rms=rms_actual,
                delta_to_mean=rms_ideal/rms_mean if rms_mean else None,
                executed_delta_to_mean=rms_actual/rms_mean if rms_mean else None,
                mean_is_zero=rms_mean == 0,
                cancellation_ratio=float(ideal.norm())/denominator if denominator else None,
                cancellation_defined=denominator>0,
                scale_std_rms=float(value.std(1, correction=0).square().mean().sqrt()),
                alpha_tv_mean=float((a-1/97).abs().sum(1).mean()/2),
                alpha_tv_max=float((a-1/97).abs().sum(1).max()/2))
            mass = torch.stack([a[:, start:stop].sum(1) for start, stop in REGIONS], 1)
            for k, (start, stop) in enumerate(REGIONS):
                record[f"R{k}_mass_mean"] = float(mass[:, k].mean())
                record[f"R{k}_feature_rms"] = float(value[:, start:stop].square().mean().sqrt())
                record[f"R{k}_weighted_contribution_rms"] = float((value[:, start:stop]*a[:, start:stop]).sum(1).square().mean().sqrt())
                record[f"R{k}_delta_rms"] = float(terms[:, start:stop].sum(1).square().mean().sqrt())
            self.records.append(record)
            time_mass = mass.mean(0).expand(4,360).double().numpy()
            if self.mass_sum is None:
                self.mass_sum, self.mass_square_sum = np.zeros_like(time_mass), np.zeros_like(time_mass)
            self.mass_sum += time_mass
            self.mass_square_sum += time_mass**2
            if record["dataset_row_id"] in self.plot_ids:
                self.plot(record["dataset_row_id"], a, mass, mean, ideal, actual)
        self.count += len(x)
        self.array.flush(); self.alpha.flush()

    def plot(self, row_id, alpha, mass, mean, ideal, actual):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
        # 槽位并非等距 Hz；明确以索引绘制并标注真实频率，重复频率仍有独立槽位。
        weights = alpha.mean(0).expand(97, 360).numpy()
        limit = max(float(np.abs(weights*97-1).max()), 1e-7)
        picture = axes[0].imshow(weights*97-1, origin="lower", aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit,
                                  extent=(0,180,-.5,96.5))
        ticks=[0,24,48,72,96]
        axes[0].set_yticks(ticks, [f"{i}: {self.frequencies[i]:.3f} Hz" for i in ticks])
        axes[0].set_title(f"row {row_id}: relative weight to 1/97 (channel mean for regional arm)")
        fig.colorbar(picture, ax=axes[0])
        for k in range(4):
            axes[1].plot(np.arange(360)*.5, mass.mean(0)[k].expand(360), label=f"R{k}")
            axes[1].axhline((REGIONS[k][1]-REGIONS[k][0])/97, color=f"C{k}", linestyle="--", alpha=.4)
        axes[1].set_ylabel("Region mass"); axes[1].legend(ncol=4)
        for label, value in (("mean", mean), ("ideal correction", ideal), ("executed correction", actual)):
            axes[2].plot(np.arange(360)*.5, value.square().mean(0).sqrt(), label=label)
        axes[2].set_ylabel("Feature RMS over channels"); axes[2].set_xlabel("Window time (s)"); axes[2].legend()
        fig.savefig(self.directory/f"row_{int(row_id)}.png", dpi=120); plt.close(fig)
        if alpha.shape[0] == 96 and not (self.directory/"region_channels.png").exists():
            fig, ax = plt.subplots(figsize=(6, 9))
            p = ax.imshow(mass.mean(-1).numpy(), aspect="auto")
            ax.set_xticks(range(4), [f"R{k}" for k in range(4)])
            ax.set_ylabel("Channel"); ax.set_title("Per-channel region mass")
            fig.colorbar(p, ax=ax); fig.savefig(self.directory/"region_channels.png", dpi=120); plt.close(fig)

    def finish(self):
        if self.count != len(self.rows):
            raise ValueError("X/描述统计未覆盖完整样本")
        self.array.flush(); self.alpha.flush()
        pd.DataFrame(self.records).to_csv(self.directory/"feature_statistics.csv", index=False)
        np.savez(self.directory/"region_mass_time_summary.npz", mean=self.mass_sum/self.count,
                 sd=np.sqrt(np.maximum(0,(self.mass_square_sum-self.mass_sum**2/self.count)/max(1,self.count-1))),
                 count=self.count)
