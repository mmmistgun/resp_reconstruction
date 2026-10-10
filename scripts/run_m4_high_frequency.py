"""原始 M4 高频机制：独立来源清单、受控推理和完整三 seed 配对汇总。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tarfile
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from scripts import eval_p1_components as source
from scripts import p1_components_runtime as training
from scripts.p1_components_model import P1ComponentsModel, frequency_metadata
from scripts import m4_high_frequency_core as core
from resp_train.engine.train import _extract_meta, _prediction_dict_from_arrays
from resp_train.metrics import task as task_metrics

shared = source.shared
sf = shared.sf
sha, read_json, write_json = training.sha, training.read_json, training.write_json
PROTOCOL = "m4-high-frequency-v1-20261009"
SPEC = "docs/experiments/m4_high_frequency_v1_protocol_20261009.md"
REFERENCE = ROOT / "runs/p1_components_v1/research_test_v1_20261007"
REFERENCE_SHA = "8ac8b2305f66cd4f54f6a317f7c1dad9336bd88d393206fa84b5f300ad68fa47"
BATCH_SIZE = 8
COUNTS = {"val": (2675, 7), "test": (2310, 8)}


def code_identity():
    result = source.code_identity()
    for name in (SPEC, "scripts/m4_high_frequency_core.py", "scripts/run_m4_high_frequency.py",
                 "scripts/plot_m4_high_frequency.py", "tests/test_m4_high_frequency.py",
                 "resp_train/paper_evidence/e4_r3_norm.py"):
        result[name] = sha(ROOT / name)
    return result


def contract(split):
    count, subjects = COUNTS[split]
    return {"split": split, "count": count, "subjects": subjects, "batch_size": BATCH_SIZE,
            "conditions": list(core.CONDITIONS), "seeds": list(core.SEEDS),
            "shift_seed": core.SHIFT_SEED, "shift_bins": [60, 300], "high_hz": [0.8, 8.0],
            "amp": "bfloat16", "baseline": "same_batch_FULL__NAT",
            "case_rule": "subject_first_middle_last_in_frozen_row_order",
            "evidence_role": "development validation" if split == "val" else shared.EVIDENCE_ROLE}


def prepare(output, split):
    """只读复用既有选点来源，不打开数据集 index、test cache 或 test 指标。"""
    if sha(REFERENCE / "allowlist.json") != REFERENCE_SHA:
        raise ValueError("P1 来源 allowlist 身份不符")
    original = source.load_allowlist(REFERENCE)
    entries = [e for e in original["entries"] if e["arm"] == "M4"]
    if [e["seed"] for e in entries] != list(core.SEEDS):
        raise ValueError("缺少完整三个原始 M4 checkpoint")
    for entry in entries:
        for key in ("checkpoint", "config", "completion"):
            if sha(entry[key]["path"]) != entry[key]["sha256"]:
                raise ValueError(f"原始 M4 来源变化: {key}")
    output = Path(output).resolve()
    # 独立目录是一次性 identity，准备失败也保留供诊断，不在原目录补写。
    output.mkdir(parents=True, exist_ok=False)
    try:
        identities = code_identity()
        with tarfile.open(output / "source_snapshot.tar.gz", "x:gz") as archive:
            for name in identities:
                archive.add(ROOT / name, arcname=name)
        if identities != code_identity():
            raise RuntimeError("准备期间源码变化")
        manifest = {"protocol": PROTOCOL, "contract": contract(split), "entries": entries,
            "reference_allowlist": {"path": str(REFERENCE / "allowlist.json"), "sha256": REFERENCE_SHA},
            "dataset_index": original["dataset_index"], "frequencies_hz": original["frequencies_hz"],
            "development_subjects": original["development_subjects"],
            "development_row_files": original["development_row_files"],
            "code_sha256": identities, "snapshot_sha256": sha(output / "source_snapshot.tar.gz"),
            "command": sys.argv, "real_data_read": False, "model_inference_used": False}
        write_json(output / "manifest.json", manifest)
        write_json(output / "prepared.json", {"manifest_sha256": sha(output / "manifest.json")})
    except BaseException:
        write_json(output / "prepare_failed.json", {"traceback": traceback.format_exc()})
        raise
    return output


def load_manifest(root):
    root = Path(root)
    if sha(root / "manifest.json") != read_json(root / "prepared.json")["manifest_sha256"]:
        raise ValueError("机制 manifest 身份变化")
    manifest = read_json(root / "manifest.json")
    split = manifest["contract"]["split"]
    if (manifest["protocol"] != PROTOCOL or manifest["contract"] != contract(split)
            or manifest["code_sha256"] != code_identity()
            or manifest["snapshot_sha256"] != sha(root / "source_snapshot.tar.gz")
            or manifest["reference_allowlist"]["sha256"] != REFERENCE_SHA
            or sha(manifest["reference_allowlist"]["path"]) != REFERENCE_SHA):
        raise ValueError("机制合同或代码来源变化")
    original = read_json(manifest["reference_allowlist"]["path"])
    if manifest["entries"] != [e for e in original["entries"] if e["arm"] == "M4"]:
        raise ValueError("M4 checkpoint 清单变化")
    for entry in manifest["entries"]:
        for key in ("checkpoint", "config", "completion"):
            if sha(entry[key]["path"]) != entry[key]["sha256"]:
                raise ValueError(f"M4 来源变化: {key}")
    return manifest


def completed(root, parent, *, retry=False):
    successes, failures, pending = [], [], []
    for path in sorted(Path(parent).glob("attempt_*")):
        if (path / "receipt.json").exists():
            receipt = read_json(path / "receipt.json")
            if receipt["manifest_sha256"] != sha(Path(root) / "manifest.json"):
                raise ValueError("完成记录属于另一 manifest")
            for name, digest in receipt["artifacts"].items():
                if sha(shared.contained(path, name)) != digest:
                    raise ValueError(f"完成产物变化: {name}")
            successes.append(path)
        elif (path / "failed.json").exists():
            failures.append(path)
        else:
            pending.append(path)
    if len(successes) > 1 or (successes and pending):
        raise ValueError("存在重复成功或未完成 attempt")
    if successes:
        return successes[0]
    if (pending or failures) and not retry:
        raise RuntimeError("存在失败/中断产物；检查原因后使用 --retry 新建 attempt")
    return None


@contextmanager
def attempt(root, parent):
    path = Path(parent) / f"attempt_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "started.json", {"protocol": PROTOCOL, "command": sys.argv})
    try:
        yield path
        files = {str(p.relative_to(path)): sha(p) for p in path.rglob("*") if p.is_file()}
        write_json(path / "receipt.json", {"manifest_sha256": sha(Path(root) / "manifest.json"), "artifacts": files})
    except BaseException:
        write_json(path / "failed.json", {"traceback": traceback.format_exc()})
        raise


def check_rows(rows, split):
    count, subjects = COUNTS[split]
    if (len(rows) != count or rows.samp_id.nunique() != subjects
            or rows.dataset_row_id.duplicated().any() or set(rows.split) != {split}):
        raise ValueError("机制数据 split/窗口/受试者数量不符")
    if split == "test":
        shared.check_test_rows(rows)


def load_data(cfg, manifest, output, confirmed):
    split = manifest["contract"]["split"]
    if split == "test":
        shared.require_confirmation(confirmed)
    write_json(output / "access_started.json", {"split": split, "contract": manifest["contract"],
        "dataset_index": manifest["dataset_index"]})
    if sha(manifest["dataset_index"]["path"]) != manifest["dataset_index"]["sha256"]:
        raise ValueError("dataset index 身份变化")
    working = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    working.data.preload_windows = False
    if split == "test":
        working.model.name, working.model.tf_representations = "crd_v1", []
        working.data.tf_cache_path = None
    bundle = shared.build_window_data(working, split=split, max_windows=None, shuffle=False,
        sample_strategy=cfg.data[f"{split}_sample_strategy"], sample_seed=int(cfg.data[f"{split}_sample_seed"]))
    check_rows(bundle.rows, split)
    bundle.rows.to_csv(output / "rows.csv", index=False)
    if split == "val":
        if sha(output / "rows.csv") != manifest["development_row_files"]["val"]:
            raise ValueError("validation row identity/order 与原始训练不同")
        if not np.array_equal(frequency_metadata(cfg.data.tf_cache_path), manifest["frequencies_hz"]):
            raise ValueError("validation CWT 坐标与 checkpoint 不同")
        dataset = bundle.dataset
    else:
        if set(bundle.rows.samp_id.astype(int)) & set(manifest["development_subjects"]):
            raise ValueError("test 与 development 受试者重叠")
        reader = source.BandTestReader(manifest["frequencies_hz"], "M4", confirmed=confirmed)
        reader.source.verify_rows(bundle.rows.dataset_row_id)
        dataset = shared.HTestDataset(bundle.dataset, reader)
    return bundle.rows, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                   pin_memory=True, drop_last=False)


def guarded_batches(loader, rows):
    offset = 0
    for batch in loader:
        size = len(batch["x"])
        expected = rows.iloc[offset:offset + size]
        if size < 1 or len(expected) != size or set(batch.get("tf", {})) != {"w"}:
            raise ValueError("batch 未按完整有序 W 数据集构造")
        for key in shared.IDENTITY_COLUMNS:
            actual = batch["meta"][key]
            actual = actual.cpu().numpy() if torch.is_tensor(actual) else np.asarray(actual)
            if not np.array_equal(actual, expected[key].to_numpy()):
                raise ValueError(f"batch 样本错序: {key}")
        for value, shape in ((batch["x"], (size, 1, 18000)), (batch["target"], (size, 1, 18000)),
                             (batch["tf"]["w"], (size, 97, 360))):
            if value.shape != shape or not value.is_floating_point():
                raise ValueError("batch shape/dtype 不符")
            core.finite(value)
        yield batch, expected
        offset += size
    if offset != len(rows):
        raise ValueError("loader 未完整覆盖数据")


def canonical_envelopes(waveform, cfg):
    protocol = task_metrics.TaskMetricConfig.from_config(cfg)
    _, canonical = task_metrics.canonicalize_numpy(np.asarray(waveform).reshape(1, -1),
        fs=protocol.fs, low_hz=protocol.band_low_hz, high_hz=protocol.band_high_hz, scale_eps=protocol.scale_eps)
    envelope = task_metrics._log_rms_envelope(canonical[0], protocol)
    times = (np.arange(len(envelope)) * protocol.envelope_step + (protocol.envelope_window - 1) / 2) / protocol.fs
    return canonical[0], times, envelope - np.median(envelope)


def signal_diagnostic(w, target, cfg, frequencies, shifts):
    """直接高频平均 log 幅度的 10 秒均值与正式 THO 包络相关；定义独立于旧 APOR 代理。"""
    from scipy.stats import pearsonr
    _, times, envelope = canonical_envelopes(target, cfg)
    high = w[np.asarray(frequencies) > .8].mean(0)
    centers = (np.arange(341) * 50 + 499.5) / 100
    records = []
    for label, curve in [("FULL", high)] + [(f"SHIFT_{i+1}", np.roll(high, shift)) for i, shift in enumerate(shifts)]:
        smooth = np.convolve(curve, np.ones(20) / 20, mode="valid")
        if times.min() < centers.min() - 1e-9 or times.max() > centers.max() + 1e-9:
            raise ValueError("包络物理支持超出 CWT 平滑支持")
        aligned = np.interp(times, centers, smooth)
        valid = np.ptp(aligned) > 1e-12 and np.ptp(envelope) > 1e-12
        correlation = float(pearsonr(aligned, envelope).statistic) if valid else np.nan
        if valid and not np.isfinite(correlation):
            raise FloatingPointError("非退化信号相关非有限")
        records.append({"transform": label, "signed_r": correlation, "valid": valid,
                        "reason": "" if valid else "constant_high_curve_or_target_envelope"})
    return records


def infer(model, loader, rows, cfg, device, output, frequencies, seed):
    cases = set(core.select_cases(rows))
    rows[rows.dataset_row_id.isin(cases)].to_csv(output / "case_selection.csv", index=False)
    shifts = core.row_shifts(rows.dataset_row_id)
    pd.DataFrame({"dataset_row_id": rows.dataset_row_id, **{f"shift_{i+1}": shifts[:, i] for i in range(3)}}).to_csv(
        output / "shifts.csv", index=False)
    (output / "cases").mkdir()
    tables = {c: [] for c in core.CONDITIONS}
    diagnostics, signals, spectra = [], [], {}
    offset = 0
    model.to(device).eval()
    with torch.no_grad():
        for batch_number, (batch, batch_rows) in enumerate(guarded_batches(loader, rows)):
            size = len(batch_rows)
            x, w = batch["x"].to(device), batch["tf"]["w"].to(device)
            target = batch["target"].cpu().numpy()
            meta = [_extract_meta(batch["meta"], i) for i in range(size)]
            batch_shifts = shifts[offset:offset + size]
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                for condition, prediction, trace, baseline, changed in core.paired_batch(model, x, w, frequencies, batch_shifts):
                    pred = prediction.float().cpu().numpy()
                    # 正式指标在 autocast 外计算，完全复用既有公式及资格逻辑。
                    with torch.autocast(torch.device(device).type, enabled=False):
                        predictions = _prediction_dict_from_arrays(pred, target, meta, pred_key="r_tho_hat", target_key="tho_ref")
                        metrics = shared.evaluate_task_predictions(predictions, cfg,
                            include_test_only=bool(rows.iloc[0].split == "test"), method=PROTOCOL)
                    sf.validate_metrics(metrics, batch_rows)
                    tables[condition].append(metrics)
                    delta = trace.points["residual"] - baseline.points["residual"]
                    context_delta = trace.points["context"].float() - baseline.points["context"].float()
                    rms = lambda t: t.float().square().mean(tuple(range(1, t.ndim))).sqrt().cpu().numpy()
                    for i, row in enumerate(batch_rows.itertuples()):
                        diagnostics.append({"condition": condition, "dataset_row_id": row.dataset_row_id,
                            "samp_id": row.samp_id, "residual_delta_rms": float(rms(delta)[i]),
                            "context_delta_rms": float(rms(context_delta)[i]),
                            "gn_formula_max_abs": max(trace.formula_error.values())})
                        if int(row.dataset_row_id) in cases:
                            arrays = {name: value[i].float().cpu().numpy() for name, value in trace.points.items()}
                            arrays.update(w=changed[i].float().cpu().numpy(), prediction=pred[i, 0],
                                target=target[i, 0], residual_delta=delta[i].cpu().numpy())
                            np.savez_compressed(output / "cases" / f"row_{row.dataset_row_id}__{condition}.npz", **arrays)
            if seed == core.SEEDS[0]:
                for i, row in enumerate(batch_rows.itertuples()):
                    raw_w = batch["tf"]["w"][i].numpy()
                    freq, normalized, valid = core.modulation_psd(raw_w)
                    if int(row.samp_id) not in spectra:
                        spectra[int(row.samp_id)] = [np.zeros_like(normalized), np.zeros(97, dtype=np.int64), 0]
                    total, counts, _ = spectra[int(row.samp_id)]
                    total[valid] += normalized[valid]
                    counts += valid
                    spectra[int(row.samp_id)][2] += 1
                    for record in signal_diagnostic(raw_w, target[i, 0], cfg, frequencies, batch_shifts[i]):
                        signals.append({"dataset_row_id": row.dataset_row_id, "samp_id": row.samp_id, **record})
            offset += size
            print(f"seed={seed} batch={batch_number+1} windows={offset}/{len(rows)}", flush=True)
    all_metrics, reference = [], None
    for condition, parts in tables.items():
        frame = pd.concat(parts, ignore_index=True)
        sf.validate_metrics(frame, rows)
        identity = frame[[*shared.IDENTITY_COLUMNS, *[k for k in shared.TARGET_COLUMNS if k in frame]]]
        if reference is None:
            reference = identity
        else:
            pd.testing.assert_frame_equal(identity, reference)
        frame.insert(0, "condition", condition)
        frame.insert(0, "seed", seed)
        all_metrics.append(frame)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "path_diagnostics.csv", index=False)
    if signals:
        pd.DataFrame(signals).to_csv(output / "signal_association.csv", index=False)
        subjects = sorted(spectra)
        np.savez_compressed(output / "modulation_psd.npz", modulation_hz=freq, carrier_hz=frequencies,
            subjects=subjects, psd_sum=np.stack([spectra[s][0] for s in subjects]),
            valid_count=np.stack([spectra[s][1] for s in subjects]), window_count=[spectra[s][2] for s in subjects])
    write_json(output / "checks.json", {"rows": len(rows), "conditions": len(core.CONDITIONS),
        "batches": batch_number+1, "full_native_replay_max_abs": 0., "full_fixed_replay_max_abs": 0.,
        "fixed_gn_stats_match": True, "z_unchanged": True, "case_rows": sorted(cases)})


def evaluate(root, seed, device, *, confirmed=False, retry=False):
    root = Path(root).resolve()
    # 先检查 split 门控，再加载任何 checkpoint 或数据。
    split = read_json(root / "manifest.json")["contract"]["split"]
    if split == "test":
        shared.require_confirmation(confirmed)
    manifest = load_manifest(root)
    entries = [e for e in manifest["entries"] if e["seed"] == seed]
    if len(entries) != 1:
        raise ValueError("seed 不属于冻结矩阵")
    entry = entries[0]
    parent = root / "evaluation" / f"seed_{seed}"
    parent.mkdir(parents=True, exist_ok=True)
    with training.mutex(parent / ".run.lock"):
        previous = completed(root, parent, retry=retry)
        if previous:
            return previous
        with attempt(root, parent) as output:
            if torch.device(device).type != "cuda":
                raise ValueError("真实数据推理要求 CUDA；CPU 验证仅使用合成测试")
            torch.cuda.set_device(device)
            environment = training.legacy.runtime(device)
            if any(environment[k] != entry["training_environment"][k] for k in ("torch", "cuda", "packages", "gpu")):
                raise ValueError("推理环境与原始训练环境不一致")
            write_json(output / "environment.json", environment)
            cfg = OmegaConf.load(entry["config"]["path"])
            if OmegaConf.to_container(cfg, resolve=True) != entry["training_config"]:
                raise ValueError("原始训练配置变化")
            checkpoint = torch.load(entry["checkpoint"]["path"], map_location="cpu", weights_only=False)
            if (checkpoint["epoch"] != entry["selected_epoch"]
                    or checkpoint["config"] != training.legacy.config_identity(cfg)):
                raise ValueError("checkpoint 配置或选点不符")
            training.legacy.finite_tree(checkpoint["model_state_dict"])
            source.checkpoint_frequencies(checkpoint["model_state_dict"], "M4", manifest["frequencies_hz"])
            model = P1ComponentsModel("M4", seed, manifest["frequencies_hz"],
                cfg=shared.PatchTFConfig(**dict(cfg.model.architecture)))
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if sum(p.numel() for p in model.parameters()) != entry["parameters"]:
                raise ValueError("模型参数量不符")
            del checkpoint
            cfg.training.device, cfg.training.batch_size = device, BATCH_SIZE
            cfg.training.show_progress = False
            OmegaConf.save(cfg, output / "evaluation_config.yaml")
            rows, loader = load_data(cfg, manifest, output, confirmed)
            infer(model, loader, rows, cfg, device, output, np.asarray(manifest["frequencies_hz"]), seed)
            if code_identity() != manifest["code_sha256"]:
                raise RuntimeError("推理期间源码变化")
    return output


def paired_tables(metrics, conditions=core.CONDITIONS, seeds=core.SEEDS):
    """严格一对一配对；资格由既有 target 定义，拒绝错误非有限值与缺 seed。"""
    if set(metrics.seed) != set(seeds) or set(metrics.condition) != set(conditions):
        raise ValueError("拒绝 partial seed/condition 汇总")
    frames, common_baseline = [], None
    identity = [*shared.IDENTITY_COLUMNS]
    for seed in seeds:
        baseline = metrics[(metrics.seed == seed) & (metrics.condition == "FULL__NAT")].reset_index(drop=True)
        sf.validate_metrics(baseline, baseline)
        keys_to_match = [*identity, *[k for k in shared.TARGET_COLUMNS if k in baseline]]
        if common_baseline is None:
            common_baseline = baseline[keys_to_match].copy()
        elif not baseline[keys_to_match].equals(common_baseline):
            raise ValueError("跨 seed 样本或 target 资格不同")
        for condition in conditions:
            current = metrics[(metrics.seed == seed) & (metrics.condition == condition)].reset_index(drop=True)
            sf.validate_metrics(current, baseline)
            for key in shared.TARGET_COLUMNS:
                if key in baseline:
                    if key not in current or not np.array_equal(current[key], baseline[key]):
                        raise ValueError("配对 target 资格发生变化")
            for metric in sf.PRIMARY:
                valid = np.isfinite(baseline[metric].to_numpy(float))
                delta = current[metric].to_numpy(float) - baseline[metric].to_numpy(float)
                if metric == sf.PCC:
                    delta = -delta
                frame = baseline[identity].copy()
                frame["seed"], frame["condition"], frame["metric"] = seed, condition, metric
                frame["baseline"], frame["value"] = baseline[metric].to_numpy(), current[metric].to_numpy()
                frame["eligible"], frame["deterioration"] = valid, delta
                frames.append(frame)
    paired = pd.concat(frames, ignore_index=True)
    keys = ["seed", "condition", "metric"]
    def aggregate(frame, keys):
        return frame.groupby(keys, sort=False).agg(windows=("dataset_row_id", "size"),
            eligible=("eligible", "sum"), baseline=("baseline", "mean"), value=("value", "mean"),
            deterioration=("deterioration", "mean")).reset_index()
    pooled = aggregate(paired, keys)
    subject = aggregate(paired, [*keys, "samp_id"])
    if (pooled.eligible <= 0).any() or (subject.eligible <= 0).any():
        raise ValueError("主指标存在无 eligible 样本的受试者/条件")
    macro = subject.groupby(keys, sort=False)[["baseline", "value", "deterioration"]].mean().reset_index()
    pooled["aggregation"], macro["aggregation"] = "window", "subject_macro"
    by_seed = pd.concat([pooled, macro], ignore_index=True)
    summary = by_seed.groupby(["condition", "metric", "aggregation"], sort=False).agg(
        baseline=("baseline", "mean"), value=("value", "mean"), deterioration=("deterioration", "mean"),
        seed_sd=("deterioration", "std"), worse_seeds=("deterioration", lambda v: int((v > 0).sum())),
        seeds=("seed", "nunique")).reset_index()
    return paired, subject, by_seed, summary


def summarize(root, *, retry=False):
    root = Path(root).resolve()
    manifest = load_manifest(root)
    paths, frames, reference = {}, [], None
    for seed in core.SEEDS:
        path = completed(root, root / "evaluation" / f"seed_{seed}")
        if path is None:
            raise ValueError("完整三 seed 推理尚未完成")
        frame = pd.read_csv(path / "metrics.csv")
        rows = pd.read_csv(path / "rows.csv")
        check_rows(rows, manifest["contract"]["split"])
        ids = frame[frame.condition == "FULL__NAT"][[*shared.IDENTITY_COLUMNS]].reset_index(drop=True)
        if reference is None:
            reference = ids
        else:
            pd.testing.assert_frame_equal(ids, reference)
        for condition in core.CONDITIONS:
            sf.validate_metrics(frame[frame.condition == condition].reset_index(drop=True), rows)
        if set(frame.seed) != {seed}:
            raise ValueError("产物 seed 不符")
        paths[str(seed)] = {"path": str(path), "receipt_sha256": sha(path / "receipt.json")}
        frames.append(frame)
    parent = root / "summary"
    parent.mkdir(exist_ok=True)
    with training.mutex(parent / ".run.lock"):
        previous = completed(root, parent, retry=retry)
        if previous:
            return previous
        with attempt(root, parent) as output:
            tables = paired_tables(pd.concat(frames, ignore_index=True))
            for name, table in zip(("paired_windows", "paired_subjects", "paired_seeds", "three_seed_summary"), tables):
                table.to_csv(output / f"{name}.csv", index=False)
            write_json(output / "sources.json", paths)
            write_json(output / "interpretation.json", {"deterioration_positive": True,
                "pcc_delta": "FULL minus intervention", "error_delta": "intervention minus FULL",
                "primary_mechanistic_outcomes": [sf.PRIMARY[2], sf.PCC],
                "all_primary_metrics_reported": list(sf.PRIMARY), "seed_sd_is_not_population_ci": True,
                "evidence_role": manifest["contract"]["evidence_role"]})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=COUNTS, default="val")
    p = sub.add_parser("run")
    p.add_argument("--session", required=True)
    p.add_argument("--seed", type=int, choices=core.SEEDS, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--confirm-research-test", action="store_true")
    p.add_argument("--retry", action="store_true")
    p = sub.add_parser("summarize")
    p.add_argument("--session", required=True)
    p.add_argument("--retry", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps({"protocol": PROTOCOL, "validation": contract("val"), "research_test": contract("test")}, ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        print(prepare(args.output, args.split))
    elif args.command == "run":
        print(evaluate(args.session, args.seed, args.device, confirmed=args.confirm_research_test, retry=args.retry))
    else:
        print(summarize(args.session, retry=args.retry))


if __name__ == "__main__":
    main()
