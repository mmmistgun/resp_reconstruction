"""固定 W0 案例的 R3 四条件配对导出与离线机制图。"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from resp_train.paper_evidence.w0_qualitative_catalog import ANALYSIS_PROTOCOL, Bundle, selection_rows, table_page
from resp_train.paper_evidence.w0_test_qualitative import EPOCH, PRIMARY, SEED, save_arrays, sha256, write_json
from resp_train.paper_evidence.w0_test_qualitative_runtime import ROOT, finish, record, sources, verify

CONDITIONS = ("FULL__NAT", "SHIFT_1__NAT", "FULL__ALL_W_GN_FIXED", "SHIFT_1__ALL_W_GN_FIXED")
BATCH_SIZE = 8


def metric_metadata(rows, batch_ids):
    """将 pandas 元数据转为评价入口要求的 NumPy 标量数组。"""
    # pandas 字符串列可能导出 object 数组，逐项取出后是没有 item() 的 Python str。
    return {key: rows.loc[batch_ids, key].to_numpy(dtype=str if key == "split" else np.int64)
            for key in ("dataset_row_id", "samp_id", "split", "coupling_state_id")}


def condition_outputs(model, x, w, shifts):
    """复用已验收的 GN 控制与原生 FiLM 捕获，所有条件共享同批 FULL 统计。"""
    import torch
    from torch import nn
    from resp_train.paper_evidence.e4_r3_norm import GN_NAMES, NormIntervention, Trace, transform_w
    from resp_train.paper_evidence.w0_cwt_film_behavior import forward_with_capture
    from resp_train.engine.train import _waveform_output

    class CaptureProxy(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        @property
        def branches(self):
            return self.inner.branches

        def forward(self, x, *, tf):
            output, self.captured = forward_with_capture(self.inner, x, tf=tf)
            return output

    proxy = CaptureProxy(model).eval()
    baseline = None
    full_prediction = None
    full_z = None
    reference = torch.zeros(97, device=x.device)  # 四条件均不使用 TRAIN_MEAN 参考。
    with torch.inference_mode():
        for condition in CONDITIONS:
            wrapper = NormIntervention(proxy, condition, reference, shifts, baseline).eval()
            with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
                result = wrapper(x, tf={"w": w})
            capture = proxy.captured
            g, b = .5 * torch.tanh(capture.gamma_raw), .5 * torch.tanh(capture.beta_raw)
            tensors = {"prediction": _waveform_output(result), "gamma_raw": capture.gamma_raw,
                       "beta_raw": capture.beta_raw, "g": g, "b": b, "z": capture.z,
                       "z_prime": capture.z_prime, "scale_delta": g.float()*capture.z.float(),
                       "total_delta": capture.z_prime.float()-capture.z.float(),
                       "cwt_w": transform_w(w, condition.split('__')[0], reference, shifts)}
            if any(not torch.isfinite(v).all() for v in tensors.values()):
                raise FloatingPointError("干预输出非有限")
            arrays = {key: value.float().cpu().numpy() for key, value in tensors.items()}
            if condition == CONDITIONS[0]:
                baseline = Trace(natural=wrapper.trace.natural)
                full_prediction, full_z = arrays["prediction"], arrays["z"]
            else:
                np.testing.assert_allclose(arrays["z"], full_z, rtol=1e-5, atol=1e-6)
            if condition.startswith("FULL__"):
                np.testing.assert_allclose(arrays["prediction"], full_prediction, rtol=1e-5, atol=1e-6)
            stats = {}
            for name in GN_NAMES:
                natural, used, original = wrapper.trace.natural[name], wrapper.trace.used[name], baseline.natural[name]
                if condition.endswith("ALL_W_GN_FIXED"):
                    torch.testing.assert_close(used.mean, original.mean, rtol=0, atol=0)
                    torch.testing.assert_close(used.rstd, original.rstd, rtol=0, atol=0)
                stats[name] = {prefix + '_' + field: getattr(values, field).float().cpu().numpy()
                               for prefix, values in (("natural", natural), ("used", used), ("full", original))
                               for field in ("mean", "rstd")}
            wrapper.trace.points.clear()
            yield condition, arrays, stats


def intervene(source: Path, cases: Path, output: Path, *, device: str, command: str) -> Path:
    import torch
    from omegaconf import OmegaConf
    from resp_train.crd.config import check_crd_dependencies
    from resp_train.crd.experiment import _validate_checkpoint_config
    from resp_train.crd.model import build_crd_model
    from resp_train.metrics.task import evaluate_task_predictions, compute_log_rms_envelopes
    from resp_train.paper_evidence.e4_r3_norm import make_shifts
    from resp_train.paper_evidence.w0_cwt_film_behavior_runtime import environment

    dev = torch.device(device)
    if dev.type != "cuda" or dev.index is None or not torch.cuda.is_available():
        raise ValueError("真实干预要求显式可用 cuda:<index>")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError('; '.join(problems))
    bundle = Bundle(source, "export")
    ids = selection_rows(cases, source)
    rows = bundle.csv("test_rows.csv")
    rows = rows.set_index("dataset_row_id", drop=False)
    if len(rows) != 2310 or rows.index.duplicated().any() or not set(ids) <= set(rows.index):
        raise ValueError("干预来源窗口集合不完整")
    all_shifts = make_shifts(len(rows))
    selected_shifts = all_shifts[rows.index.get_indexer(ids)]
    baseline, _, _ = sources()
    run = Path(baseline["run_dir"])
    verify(run / "checkpoint_best_local_rr.pt", baseline["checkpoint"])
    verify(run / "config.yaml", baseline["config"])
    cfg = OmegaConf.load(run / "config.yaml")
    checkpoint = torch.load(run / "checkpoint_best_local_rr.pt", map_location="cpu", weights_only=False)
    _validate_checkpoint_config(checkpoint.get("config"), cfg)
    if checkpoint["epoch"] != EPOCH or int(cfg.training.seed) != SEED:
        raise ValueError("checkpoint 身份错误")
    if any(not torch.isfinite(v).all() for v in checkpoint["model_state_dict"].values() if torch.is_tensor(v)):
        raise FloatingPointError("checkpoint 非有限")
    model = build_crd_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(dev).eval()
    del checkpoint
    output = output.resolve()
    if output == bundle.root or bundle.root in output.parents:
        raise ValueError("干预结果使用独立输出目录")
    output.mkdir(parents=True, exist_ok=False)
    config = {"analysis_protocol": ANALYSIS_PROTOCOL, "seed": SEED, "epoch": EPOCH, "conditions": CONDITIONS, "batch_size": BATCH_SIZE,
              "amp_dtype": "bfloat16", "row_ids": ids, "r3_slice": [73, 97],
              "shift_rule": "E4 make_shifts(2310), PCG64 seed 20260920, SHIFT_1, frozen export row order",
              "shift_frames": selected_shifts[:, 0].tolist(), "cwt_frame_sec": .5,
              "full_replay_rtol": 1e-5, "full_replay_atol": 1e-6,
              "source": bundle.identity(), "cases": Bundle(cases, "case_selection").identity(),
              "checkpoint": baseline["checkpoint"], "command": command, "environment": environment(device)}
    write_json(output / "config.json", config)
    write_json(output / "access_started.json", {"scope": "selected saved research-test windows", "rows": ids, "model_inference": True})
    OmegaConf.save(cfg, output / "training_config.yaml")
    shutil.copy2(bundle.path("coordinates.npz"), output / "coordinates.npz")
    (output / "source_code").mkdir()
    code_sources = [Path(__file__), ROOT / "resp_train/paper_evidence/e4_r3_norm.py",
                    ROOT / "resp_train/paper_evidence/w0_cwt_film_behavior.py"]
    for path in code_sources:
        shutil.copy2(path, output / "source_code" / path.name)
    metric_frames, gn_records, window_records, replay_records = [], [], [], []
    saved_metrics = bundle.csv("metrics.csv")
    for start in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[start:start+BATCH_SIZE]
        loaded = []
        for row_id in batch_ids:
            with np.load(bundle.path(f"windows/row_{row_id}.npz"), allow_pickle=False) as blob:
                if int(blob["dataset_row_id"]) != row_id:
                    raise ValueError("保存窗口身份漂移")
                loaded.append({"bcg": blob["bcg"], "target": blob["waveforms"][0], "w": blob["cwt_w"]})
        bcg = np.stack([r["bcg"] for r in loaded]).astype(np.float32)
        target = np.stack([r["target"] for r in loaded]).astype(np.float32)
        w = np.stack([r["w"] for r in loaded]).astype(np.float32)
        if not all(np.isfinite(a).all() for a in (bcg, target, w)):
            raise FloatingPointError("保存输入非有限")
        metadata = metric_metadata(rows, batch_ids)
        local_full = {}
        for condition, arrays, stats in condition_outputs(model, torch.from_numpy(bcg[:, None]).to(dev),
                                                          torch.from_numpy(w).to(dev), torch.from_numpy(selected_shifts[start:start+len(batch_ids)])):
            prediction = arrays["prediction"].reshape(len(batch_ids), -1)
            metrics = evaluate_task_predictions({"r_tho_hat": prediction, "tho_ref": target, **metadata}, cfg, method=condition)
            if not np.isfinite(metrics[list(PRIMARY)].to_numpy(float)).all():
                raise FloatingPointError("干预五主指标非有限")
            metrics["condition"] = condition
            metric_frames.append(metrics)
            if condition == CONDITIONS[0]:
                for _, row in metrics.iterrows():
                    old = saved_metrics.loc[saved_metrics.method.eq('W0') & saved_metrics.dataset_row_id.eq(row.dataset_row_id)].iloc[0]
                    replay_records.extend({"dataset_row_id": int(row.dataset_row_id), "metric": key,
                                           "current_full": float(row[key]), "saved_full": float(old[key]),
                                           "delta": float(row[key]-old[key])} for key in PRIMARY)
            envelopes = compute_log_rms_envelopes(prediction, cfg)
            target_env = compute_log_rms_envelopes(target, cfg)
            for i, row_id in enumerate(batch_ids):
                values = {key: value[i] for key, value in arrays.items()}
                values.update(dataset_row_id=np.asarray(row_id), target=target[i], bcg=bcg[i],
                              log_rms_envelope=envelopes[i], target_log_rms_envelope=target_env[i])
                if condition == CONDITIONS[0]:
                    local_full[row_id] = {key: values[key].copy() for key in ("prediction", "z_prime")}
                values["prediction_minus_full"] = values["prediction"] - local_full[row_id]["prediction"]
                values["z_prime_minus_full"] = values["z_prime"] - local_full[row_id]["z_prime"]
                values["relative_feature_change"] = np.linalg.norm(values["z_prime_minus_full"], axis=0) / (np.linalg.norm(values["z"], axis=0)+1e-12)
                folder = output / "windows" / f"row_{row_id}"
                folder.mkdir(parents=True, exist_ok=True)
                save_arrays(folder / f"{condition}.npz", values)
                for layer, tensors in stats.items():
                    for group in range(tensors["used_mean"].shape[1]):
                        gn_records.append({"dataset_row_id": row_id, "condition": condition, "layer": layer, "group": group,
                                           **{key: float(value[i, group]) for key, value in tensors.items()}})
                window_records.append({"dataset_row_id": row_id, "condition": condition,
                                       "file": str((folder / f"{condition}.npz").relative_to(output))})
        print(f"intervened {min(start+BATCH_SIZE,len(ids))}/{len(ids)} selected windows", flush=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    full = metrics.loc[metrics.condition.eq(CONDITIONS[0])].set_index('dataset_row_id')
    paired = metrics[['dataset_row_id', 'condition', *PRIMARY]].copy()
    for key in PRIMARY:
        paired[key] = paired[key] - paired.dataset_row_id.map(full[key])
    paired.to_csv(output / "paired_deltas.csv", index=False)
    pd.DataFrame(replay_records).to_csv(output / "saved_full_comparison.csv", index=False)
    pd.DataFrame(gn_records).to_csv(output / "gn_statistics.csv", index=False)
    pd.DataFrame(window_records).to_csv(output / "window_index.csv", index=False)
    finish(output, {"phase": "r3_intervention", "seed": SEED, "epoch": EPOCH, "rows": ids,
                    "analysis_protocol": ANALYSIS_PROTOCOL,
                    "conditions": CONDITIONS, "export_source": bundle.identity(), "config_sha256": sha256(output / 'config.json'),
                    "source_code": [record(p) for p in code_sources], "model_inference": True, "training": False,
                    "original_dataset_read": False, "paired_reference": "same-batch FULL__NAT"})
    return output


def render_intervention(source: Path, output: Path, *, row_ids: list[int] | None, zoom: tuple[float, float] | None, command: str) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bundle = Bundle(source, "r3_intervention")
    metrics = bundle.csv('metrics.csv')
    ids = row_ids if row_ids is not None else bundle.receipt['rows']
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(bundle.receipt['rows']):
        raise ValueError("绘图窗口无效")
    if zoom and not 0 <= zoom[0] < zoom[1] <= 180:
        raise ValueError("zoom 必须位于 0–180 秒")
    output = output.resolve()
    if output == bundle.root or bundle.root in output.parents:
        raise ValueError("图形输出使用独立目录")
    output.mkdir(parents=True, exist_ok=False)
    (output / 'figures').mkdir()
    links = {}
    for position, row_id in enumerate(ids):
        arrays = {}
        for condition in CONDITIONS:
            with np.load(bundle.path(f'windows/row_{row_id}/{condition}.npz'), allow_pickle=False) as blob:
                arrays[condition] = dict(blob)
        fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True, layout='constrained')
        base = arrays[CONDITIONS[0]]
        t = np.arange(18000)/100
        for condition in CONDITIONS[:2]:
            axes[0].plot((np.arange(360)*50+24.5)/100, arrays[condition]['cwt_w'][73:97].mean(0), label=condition)
        axes[1].plot(t, base['target'], color='black', lw=1, label='THO reference')
        for condition in CONDITIONS:
            a = arrays[condition]
            axes[1].plot(t, a['prediction'].reshape(-1), lw=.7, label=condition)
            axes[2].plot((np.arange(35)*5+5), a['log_rms_envelope']-np.median(a['log_rms_envelope']), label=condition)
            axes[3].plot((np.arange(1800)+.5)/10, a['relative_feature_change'], label=condition)
            axes[4].plot(t, a['prediction_minus_full'].reshape(-1), label=condition, lw=.7)
        axes[2].plot(np.arange(35)*5+5, base['target_log_rms_envelope']-np.median(base['target_log_rms_envelope']), color='black', label='THO reference')
        for ax, label in zip(axes, ('R3 mean log-magnitude', 'Stored waveform amplitude', 'Centered log-RMS', 'Relative feature change', 'Prediction minus FULL'), strict=True):
            ax.set_ylabel(label)
            ax.set_xlim(*(zoom or (0,180)))
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=.15)
        selected = metrics.loc[metrics.dataset_row_id.eq(row_id)]
        captions = [f"{r.condition}: PCC={r.lag_aware_signed_pcc:.4f}, trajectory MAE={r.envelope_trajectory_mae:.4f}" for _,r in selected.iterrows()]
        fig.suptitle(f'R3 temporal intervention | W0 seed {SEED} | row {row_id}\n'+'\n'.join(captions)+'\nMetrics: full 180 s; feature/prediction changes relative to same-batch FULL')
        axes[-1].set_xlabel('Time within window (s)')
        name = f'figures/row_{row_id}.png'
        try:
            with (output/name).open('xb') as handle:
                fig.savefig(handle, format='png', dpi=160, bbox_inches='tight')
        finally:
            plt.close(fig)
        links[position] = [('机制组合图',name)]
    (output/'index.html').write_text(table_page(pd.DataFrame({'dataset_row_id':ids}), 'R3 干预例图',links), encoding='utf-8')
    metrics.loc[metrics.dataset_row_id.isin(ids)].to_csv(output/'metrics.csv',index=False)
    finish(output, {'phase':'intervention_render','analysis_protocol':ANALYSIS_PROTOCOL,'source':bundle.identity(),'rows':ids,'zoom':zoom,'command':command,'model_inference':False})
    return output
