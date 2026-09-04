"""只读取冻结指标的跨输出时长节律汇总；不导入模型、数据集或 evaluator。"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROTOCOL_ID = "paper-harmonized-rhythm-test-v1-20260905"
OUTPUT_DIR = Path("runs/paper_evidence_v1/harmonized_rhythm_test_summary")
V2_PROTOCOL_ID = "paper-harmonized-rhythm-test-v2-20260905"
V2_OUTPUT_DIR = Path("runs/paper_evidence_v1/harmonized_rhythm_test_summary_v2")
BASE = Path("runs/paper_evidence_v1")
SOURCES = Path("configs/paper_evidence_v1/p0_comparison_sources.json")
SEEDS = (20260811, 20260812, 20260813)
ROW_HASH = "184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8"
PINS = {
    BASE / "context_length_research_test_summary/artifact_manifest.json": "d69d47a80e514a9556000296938dff642d2642656c61ecff82d57c1a1baeeb66",
    BASE / "center90_context_research_test_summary/artifact_manifest.json": "2da65f83dc93d6a65b9c9e2c5024c67aa00a0520f2cf36338e35c77589d9f50b",
    BASE / "p0_comparison_audit/manifest.json": "1e05c3de9a75c08ee016af379e075adc1b71b4d1162f7bbf81d716b4942d2542",
}
IBI_CONFIG = {"ibi_peak_distance_samples": 142, "ibi_match_tolerance_sec": 0.5, "ibi_coverage_threshold": 0.8}
VALUES = ("native_whole_rr_mae_bpm", "ibi_medae_sec", "ibi_coverage", "ibi_interpretable_fraction")
COUNTS = ("n_samples", "native_whole_rr_n", "ibi_valid_n", "ibi_target_eligible_n", "ibi_interpretable_n")
HISTORICAL_LABELS = {
    "crd_c201_decoder_10hz_cap": "C201",
    "crd_tf102_w": "W0",
    "crd_tfw_v2_w3_full_6v_film_d6": "W3",
}
CENTER_SELECTORS = {
    "center30": "full_validation_center30_rr_mae_bpm_strict_lower_tie_earlier",
    "center60": "full_validation_center_rr_mae_bpm_strict_lower_tie_earlier",
    "center90": "full_validation_center90_rr_mae_bpm_strict_lower_tie_earlier",
}


class FrozenReader:
    """核验冻结 hash 后解析同一份 bytes，记录全部实际读取文件。"""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.records: dict[str, dict[str, Any]] = {}
        self.cache: dict[str, bytes] = {}

    def read(self, path: Path | str, expected: str | None = None, size: int | None = None) -> Any:
        path = Path(path)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        relative = path.relative_to(self.root)
        if path.suffix not in {".csv", ".json", ".yaml", ".py", ".md"}:
            raise ValueError(f"禁止读取非指标/provenance 文件: {relative}")
        key = str(relative)
        data = self.cache.get(key)
        if data is None:
            data = path.read_bytes()
            self.cache[key] = data
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"冻结 SHA-256 不匹配: {relative}")
        if size is not None and len(data) != int(size):
            raise ValueError(f"冻结 size 不匹配: {relative}")
        verified = expected is not None or self.records.get(key, {}).get("frozen_hash_verified", False)
        self.records[key] = {"path": key, "sha256": digest, "size_bytes": len(data), "frozen_hash_verified": verified}
        if path.suffix == ".csv":
            return pd.read_csv(io.BytesIO(data))
        if path.suffix == ".json":
            return json.loads(data)
        if path.suffix == ".yaml":
            return yaml.safe_load(data)
        return data.decode("utf-8")

    def bundle(self, manifest_path: Path) -> dict[str, Any]:
        manifest = self.read(manifest_path, PINS[manifest_path])
        records = manifest["files"]
        if isinstance(records, dict):
            records = [{"filename": k, **v} for k, v in records.items()]
        if len({r["filename"] for r in records}) != len(records):
            raise ValueError("manifest 文件身份重复")
        for record in records:
            self.read(manifest_path.parent / record["filename"], record["sha256"], record["size_bytes"])
        return manifest

    def verify_unchanged(self) -> None:
        for key, record in self.records.items():
            if hashlib.sha256((self.root / key).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"汇总期间输入发生变化: {key}")


def _bool_column(frame: pd.DataFrame, key: str) -> np.ndarray:
    values = frame[key]
    if values.isna().any() or not values.isin([True, False, 0, 1, "True", "False"]).all():
        raise ValueError(f"{key} 不是完整布尔列")
    return values.map({True: True, False: False, "True": True, "False": False}).to_numpy(dtype=bool)


def _close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=1e-10, atol=1e-12, equal_nan=True):
        raise ValueError(f"summary 数值不一致: {label}: {actual} != {expected}")


def summarize_samples(frame: pd.DataFrame, prefix: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """保持每个样本先计算 MedAE、checkpoint 再取均值的原合同。"""
    ibi = f"{prefix}ibi_"
    rr = f"{prefix}rr_mae_bpm" if prefix else "whole_rr_abs_error_bpm"
    rr_eligible = f"{prefix}rr_target_eligible" if prefix else "whole_rr_target_eligible"
    needed = {rr, rr_eligible, *(ibi + k for k in ("medae_sec", "coverage", "interpretable", "target_eligible"))}
    if not needed.issubset(frame):
        raise ValueError(f"metrics 缺少字段: {sorted(needed - set(frame))}")
    eligible = _bool_column(frame, ibi + "target_eligible")
    interpretable = _bool_column(frame, ibi + "interpretable")
    rr_mask = _bool_column(frame, rr_eligible)
    medae = pd.to_numeric(frame[ibi + "medae_sec"], errors="raise").to_numpy(float)
    coverage = pd.to_numeric(frame[ibi + "coverage"], errors="raise").to_numpy(float)
    rr_values = pd.to_numeric(frame[rr], errors="raise").to_numpy(float)
    for key, values, mask in ((rr, rr_values, rr_mask), (ibi + "medae_sec", medae, interpretable), (ibi + "coverage", coverage, eligible)):
        if np.isinf(values).any() or not np.array_equal(np.isfinite(values), mask):
            raise ValueError(f"{key} finite/eligible 合同不符")
        if (values[mask] < 0).any():
            raise ValueError(f"{key} 负值")
    if (coverage[eligible] > 1).any() or not np.array_equal(interpretable, eligible & (coverage >= 0.8)):
        raise ValueError("IBI coverage/interpretable 合同不符")
    mean = lambda x: float(np.mean(x)) if len(x) else np.nan
    result = {
        "n_samples": len(frame), "native_rr_source_column": rr,
        "native_whole_rr_mae_bpm": mean(rr_values[rr_mask]), "native_whole_rr_n": int(rr_mask.sum()),
        "ibi_medae_sec": mean(medae[interpretable]), "ibi_valid_n": int(interpretable.sum()),
        "ibi_coverage": mean(coverage[eligible]), "ibi_target_eligible_n": int(eligible.sum()),
        "ibi_interpretable_fraction": mean(interpretable[eligible]), "ibi_interpretable_n": int(interpretable.sum()),
    }
    support = pd.DataFrame({"dataset_row_id": frame["dataset_row_id"].to_numpy(), "eligible": eligible, "interpretable": interpretable})
    return result, support.sort_values("dataset_row_id").reset_index(drop=True)


def validate_identity(frame: pd.DataFrame, reference: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["dataset_row_id", "split", "input_set", "samp_id", "coupling_state_id"]
    if len(frame) != 2310 or frame[columns].isna().any().any() or not frame["split"].eq("test").all():
        raise ValueError("独立测试集 count/split/metadata 错误")
    ids = pd.to_numeric(frame["dataset_row_id"], errors="raise").to_numpy()
    if not np.isfinite(ids).all() or not np.equal(ids, ids.astype(np.int64)).all():
        raise ValueError("dataset_row_id 必须是整数")
    if len(np.unique(ids)) != 2310 or hashlib.sha256(np.sort(ids.astype("<i8")).tobytes()).hexdigest() != ROW_HASH:
        raise ValueError("dataset_row_id 重复或冻结身份不符")
    identity = frame[columns].sort_values("dataset_row_id").reset_index(drop=True)
    if identity["samp_id"].nunique() != 8 or (reference is not None and not identity.equals(reference)):
        raise ValueError("跨来源 row/metadata 错位")
    return identity


def validate_stored(row: dict[str, Any], stored: pd.Series, prefix: str, *, rtm: bool = False) -> None:
    rr = row["native_rr_source_column"]
    if rtm:
        mapping = {"n_samples": "sample_rows", "native_whole_rr_mae_bpm": rr, "native_whole_rr_n": rr + "_eligible_count"}
    else:
        ibi = prefix + "ibi_"
        mapping = {
            "n_samples": "n_samples", "native_whole_rr_mae_bpm": rr + "_mean", "native_whole_rr_n": rr + "_n",
            "ibi_medae_sec": ibi + "medae_sec_mean", "ibi_valid_n": ibi + "medae_sec_n",
            "ibi_coverage": ibi + "coverage_mean", "ibi_target_eligible_n": ibi + "coverage_n",
            "ibi_interpretable_fraction": ibi + "interpretable_fraction",
        }
        mapping["ibi_target_eligible_n"] = ibi + ("target_eligible_n" if prefix else "interpretable_n")
        _close(row["ibi_target_eligible_n"], stored[ibi + "coverage_n"], ibi + "coverage_n")
    for target, source in mapping.items():
        _close(row[target], stored[source], source)


def validate_metric_config(cfg: dict[str, Any], *, historical: bool) -> None:
    expected = dict(IBI_CONFIG)
    if historical:
        expected.update(local_rr_window_sec=60, local_rr_step_sec=15)
    if any(cfg["evaluation"].get(k) != v for k, v in expected.items()):
        raise ValueError("冻结 IBI/Local RR 参数漂移")
    if cfg["window"]["target_fs"] != 100 or cfg["loss"]["max_lag_sec"] != 0.3:
        raise ValueError("IBI fs/lag 参数漂移")


def aggregate_arms(seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in seeds.groupby(["task", "model", "input_sec", "output_sec"], sort=False):
        first = group.iloc[0]
        deterministic = bool(first["deterministic"])
        if deterministic:
            if len(group) != 1 or group["seed"].notna().any():
                raise ValueError("deterministic-no-seed 身份错误")
        elif len(group) != 3 or sorted(group["seed"].astype(int)) != list(SEEDS):
            raise ValueError("学习模型必须完整三个独立 seeds")
        fields = ["task", "model", "method_id", "input_sec", "output_sec", "deterministic", "seed_semantics", "conclusion_lock_status", "checkpoint_selector", "native_rr_source_column", "fft_rr_spacing_bpm"]
        if any(group[field].nunique(dropna=False) != 1 for field in fields):
            raise ValueError("同 arm provenance 不一致")
        row = {k: first[k] for k in fields}
        row.update(seed_count=0 if deterministic else 3, record_count=len(group), seeds="" if deterministic else ";".join(map(str, SEEDS)))
        for metric in VALUES:
            values = group[metric].to_numpy(float)
            # 不对缺失的 seed 再做 nanmean；任何缺失会显式传到 arm 结果。
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_sample_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[metric + "_finite_records"] = int(np.isfinite(values).sum())
        for count in COUNTS:
            row[count + "_by_seed"] = json.dumps(group[count].astype(int).tolist())
            row[count + "_min"] = int(group[count].min())
            row[count + "_max"] = int(group[count].max())
        rows.append(row)
    return pd.DataFrame(rows)


def audit_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, FrozenReader]:
    reader = FrozenReader(root)
    for path in ("resp_train/metrics/task.py", "resp_train/paper_evidence/center30_metrics.py",
                 "resp_train/paper_evidence/center_context_metrics.py", "resp_train/paper_evidence/center90_metrics.py"):
        reader.read(path)
    for manifest in PINS:
        reader.bundle(manifest)
    p0 = BASE / "p0_comparison_audit"
    p0_manifest = reader.read(p0 / "manifest.json")
    sources = reader.read(SOURCES, p0_manifest["source_manifest_sha256"])
    audit = reader.read(p0 / "source_artifact_audit.csv")
    compatibility = reader.read(p0 / "protocol_compatibility.csv").set_index("method_id")
    primary = reader.read(p0 / "paper_primary_metrics_table.csv")
    reference = None
    rows: list[dict[str, Any]] = []
    supports: dict[tuple[str, str, int, int], pd.DataFrame] = {}

    def add(frame: pd.DataFrame, stored: pd.DataFrame, metadata: dict[str, Any], prefix: str, rtm: bool = False) -> dict[str, Any]:
        nonlocal reference
        reference = validate_identity(frame, reference)
        if len(stored) != 1:
            raise ValueError("每项必须唯一 stored summary")
        result, support = summarize_samples(frame, prefix)
        validate_stored(result, stored.iloc[0], prefix, rtm=rtm)
        row = {**metadata, **result, "fft_rr_spacing_bpm": 60.0 / metadata["output_sec"]}
        rows.append(row)
        if not metadata["deterministic"] and metadata["input_sec"] == metadata["output_sec"]:
            supports[(metadata["task"], metadata["model"], metadata["input_sec"], metadata["seed"])] = support
        return row

    for directory, expected_count in (("context_length_research_test_summary", 42), ("center90_context_research_test_summary", 18)):
        receipt = reader.read(BASE / directory / "summary_receipt.json")
        if receipt["status"] != "complete" or len(receipt["inputs"]) != expected_count:
            raise ValueError("中心任务冻结 receipt 不完整")
        for item in receipt["inputs"]:
            task = item.get("task", "center90")
            length = int(task.removeprefix("center"))
            prefix = "center_" if length == 60 else task + "_"
            evaluation = Path(item["evaluation_dir"])
            frame = reader.read(evaluation / "research_test_metrics.csv", item.get("research_test_metrics_sha256", item.get("metrics_sha256")))
            stored = reader.read(evaluation / "research_test_metrics_summary.csv", item.get("research_test_summary_sha256", item.get("summary_sha256")))
            manifest = reader.read(evaluation / "artifact_manifest.json", item["artifact_manifest_sha256"])
            evaluation_receipt = reader.read(
                evaluation / "evaluation_receipt.json", item["evaluation_receipt_sha256"]
            )
            evaluation_identity = evaluation_receipt.get("identity", {})
            formal_manifest = reader.read(
                Path(evaluation_identity["run_dir"]) / "artifact_manifest.json",
                evaluation_identity["artifact_manifest_sha256"],
            )
            selector = formal_manifest.get("selector")
            if selector != CENTER_SELECTORS[task]:
                raise ValueError(f"{task} formal selector identity 漂移")
            config_record = next(r for r in manifest["files"] if r["filename"] == "resolved_evaluation_config.yaml")
            cfg = reader.read(evaluation / "resolved_evaluation_config.yaml", config_record["sha256"], config_record["size_bytes"])
            validate_metric_config(cfg, historical=False)
            seed_table_name = "test_seed_arm_summary.csv" if length == 90 else f"{task}_seed_arm_test_summary.csv"
            seed_table = reader.read(BASE / directory / seed_table_name)
            selected = seed_table.loc[seed_table.experiment_id.eq(item["experiment_id"]) & seed_table.seed.eq(item["seed"])]
            if len(selected) != 1:
                raise ValueError("中心任务 seed table identity 错误")
            identity = selected.iloc[0]
            model = ("C201" if "C201" in item["experiment_id"] else "W-reduced") + f"-center{length}"
            row = add(frame, stored, {
                "task": task, "model": model, "method_id": identity["variant"], "input_sec": int(identity["input_sec"]),
                "output_sec": length, "seed": int(item["seed"]), "deterministic": False, "seed_semantics": "three_training_seeds",
                "conclusion_lock_status": "confirmed", "checkpoint_selector": selector,
                "source_metrics": str(evaluation.relative_to(root) / "research_test_metrics.csv"), "ibi_summary_source": "verified_existing_summary",
            }, prefix)
            validate_stored(row, identity, prefix)

    def historical_read(path: str, method_id: str, kind: str) -> pd.DataFrame:
        records = audit.loc[audit.method_id.eq(method_id) & audit.artifact_kind.eq(kind) & audit.path.eq(str((root / path).resolve()))]
        unique = records[["sha256", "size_bytes"]].drop_duplicates()
        if len(unique) != 1:
            raise ValueError(f"P0 artifact 身份缺失或冲突: {path}")
        return reader.read(path, str(unique.iloc[0].sha256), int(unique.iloc[0].size_bytes))

    for method in sources["methods"]:
        method_id = method["method_id"]
        if method["quality_scope"] != "independent_test":
            continue
        if compatibility.loc[method_id, "compatibility"] != "compatible":
            raise ValueError(f"P0 方法未通过兼容审计: {method_id}")
        matrix = method.get("record_matrix")
        if matrix:
            metrics = historical_read(matrix["sample_metrics"], method_id, "sample_metrics")
            summaries = historical_read(matrix["summary"], method_id, "summary")
            expected = {(c, s) for c in matrix["candidate_ids"] for s in matrix["seeds"]}
            if set(zip(metrics.candidate_id, metrics.seed)) != expected or set(zip(summaries.candidate_id, summaries.seed)) != expected:
                raise ValueError("RTM 候选/seed 矩阵错误")
            records = [{"seed": seed, "candidate_id": candidate, **matrix} for candidate, seed in sorted(expected)]
            rtm_dir = Path(matrix["sample_metrics"]).parent
            rtm_manifest = reader.read(rtm_dir / "artifact_manifest.json", "a18d1459a3e4e9f11728c2bcebffd64f1f016f65129345c359794eee2a39ce4b")
            config_record = next(r for r in rtm_manifest["artifacts"] if r["filename"] == "checkpoint_inputs.json")
            checkpoint_inputs = reader.read(rtm_dir / "checkpoint_inputs.json", config_record["sha256"], config_record["size_bytes"])
        else:
            records = method["records"]
        for record in records:
            if record["split"] != "test":
                raise ValueError("历史来源必须是 test")
            if matrix:
                candidate = record["candidate_id"]
                frame = metrics.loc[metrics.candidate_id.eq(candidate) & metrics.seed.eq(record["seed"])].copy()
                stored = summaries.loc[summaries.candidate_id.eq(candidate) & summaries.seed.eq(record["seed"])].copy()
                config_input = [r for r in checkpoint_inputs if r["candidate_id"] == candidate and r["seed"] == record["seed"]]
                if len(config_input) != 1:
                    raise ValueError("RTM config identity 错误")
                cfg = reader.read(config_input[0]["config_path"], config_input[0]["config_sha256"])
            else:
                candidate = method_id
                frame = historical_read(record["sample_metrics"], method_id, "sample_metrics")
                stored = historical_read(record["summary"], method_id, "summary")
                directory = root / Path(record["sample_metrics"]).parent
                if method["deterministic"]:
                    cfg = reader.read(directory / "resolved_config.yaml")
                elif candidate == "crd_tfw_v2_w3_full_6v_film_d6":
                    provenance = reader.read(directory / "research_test_metrics_manifest.json")
                    # 仅使用 checkpoint 路径定位旁边的配置，不打开 checkpoint 文件。
                    cfg = reader.read(Path(provenance["checkpoint"]).parent / "config.yaml")
                else:
                    cfg = reader.read(directory / "config.yaml")
            validate_metric_config(cfg, historical=True)
            add(frame, stored, {
                "task": "historical180", "model": HISTORICAL_LABELS.get(candidate, candidate), "method_id": candidate,
                "input_sec": 180, "output_sec": 180, "seed": record.get("seed"), "deterministic": method["deterministic"],
                "seed_semantics": "deterministic-no-seed" if method["deterministic"] else "three_training_seeds",
                "conclusion_lock_status": "confirmed" if method["conclusion_lock_confirmed"] else "pending_user_confirmation_audit_only",
                "checkpoint_selector": method["contracts"]["checkpoint_selector"], "source_metrics": record["sample_metrics"],
                "ibi_summary_source": "existing_sample_metrics_only_summary_has_no_ibi" if matrix else "verified_existing_summary",
            }, "", rtm=bool(matrix))

    seeds = pd.DataFrame(rows).sort_values(["output_sec", "model", "input_sec", "seed"]).reset_index(drop=True)
    expected_context = {(f"center{out}", f"{model}-center{out}", inp, seed) for out, inputs in ((30, (30, 45, 60, 90)), (60, (60, 90, 180)), (90, (90, 135, 180))) for model in ("C201", "W-reduced") for inp in inputs for seed in SEEDS}
    actual_context = seeds.loc[seeds.task.ne("historical180"), ["task", "model", "input_sec", "seed"]]
    if len(seeds) != 101 or len(actual_context) != 60 or set(actual_context.itertuples(index=False, name=None)) != expected_context:
        raise ValueError("全任务必须完整覆盖 60 中心 checkpoints + 41 历史记录")
    arms = aggregate_arms(seeds)
    for _, old in primary.iterrows():
        new = arms.loc[arms.method_id.eq(old.method_id)].iloc[0]
        _close(new.native_whole_rr_mae_bpm_mean, old.whole_rr_abs_error_bpm_mean, old.method_id)
        _close(new.native_whole_rr_mae_bpm_sample_sd, old.whole_rr_abs_error_bpm_sample_sd, old.method_id)
    support_rows = []
    for family in ("C201", "W-reduced"):
        endpoints = ("C201",) if family == "C201" else ("W3", "W0")
        edges = [(30, 60, f"{family}-center30", f"{family}-center60"), (60, 90, f"{family}-center60", f"{family}-center90")]
        edges += [(90, 180, f"{family}-center90", end) for end in endpoints]
        for left, right, lm, rm in edges:
            for seed in SEEDS:
                a = supports[(f"center{left}", lm, left, seed)]
                b = supports[("historical180" if right == 180 else f"center{right}", rm, right, seed)]
                row = {"from_model": lm, "to_model": rm, "from_sec": left, "to_sec": right, "seed": seed}
                left_metric = seeds.loc[seeds.model.eq(lm) & seeds.input_sec.eq(left) & seeds.seed.eq(seed), "ibi_medae_sec"].item()
                right_metric = seeds.loc[seeds.model.eq(rm) & seeds.input_sec.eq(right) & seeds.seed.eq(seed), "ibi_medae_sec"].item()
                row.update(from_ibi_medae_sec=left_metric, to_ibi_medae_sec=right_metric,
                           ibi_decreased=bool(right_metric < left_metric))
                for mask in ("eligible", "interpretable"):
                    av, bv = a[mask].to_numpy(bool), b[mask].to_numpy(bool)
                    union = int((av | bv).sum())
                    row.update({f"{mask}_intersection_n": int((av & bv).sum()), f"{mask}_union_n": union,
                                f"{mask}_changed_n": int((av ^ bv).sum()), f"{mask}_jaccard": float((av & bv).sum() / union) if union else np.nan})
                support_rows.append(row)
    return seeds, pd.DataFrame(support_rows), reader


def build_comparability(
    arms: pd.DataFrame,
    supports: pd.DataFrame,
    *,
    protocol_id: str = PROTOCOL_ID,
) -> dict[str, Any]:
    ladder = rhythm_ladder(arms)
    directions = [
        {"from_model": left, "to_model": right, "seeds_decreased": int(group.ibi_decreased.sum()), "seed_count": len(group)}
        for (left, right), group in supports.groupby(["from_model", "to_model"], sort=False)
    ]
    return {
        "protocol_id": protocol_id, "evidence_scope": "descriptive_cross_task_only",
        "native_whole_rr": {"definition": "各自完整输出上去均值、symmetric Hann、N 点 rFFT 的频带 dominant-frequency 绝对误差",
            "output_sec_to_fft_rr_spacing_bpm": {str(n): 60 / n for n in (30, 60, 90, 180)},
            "interpretation": "60/T 是理论 bin 间距，不是误差下界或完整频谱分辨能力；较粗网格可将不同节律量化到同一 bin。",
            "nonmonotonic_explanations": ["输出时长与频谱网格变化", "target 的时间支撑与 Pi_T 投影变化", "独立训练任务与监督时长变化", "中心任务由 native RR 选 checkpoint，历史由 60s Local RR 选 checkpoint", "W-reduced、W3、W0 表征身份不同"],
            "causal_attribution": "这些是解释限制，现有汇总不能分离各因素贡献"},
        "ibi": {**IBI_CONFIG, "sample_rate_hz": 100, "max_lag_sec": 0.3,
            "implementation": "resp_train.metrics.task._ibi_metrics/_detect_peaks/_ordered_event_match",
            "peak_prominence": "max(0.2*std(ddof=0), 0.08*(P95-P5))",
            "matching": "有序匹配优先最大匹配数，再最小总时间差；仅相邻 target/prediction 峰对形成 IBI",
            "coverage": "有效相邻匹配周期数 / (target峰数-1)，在 target-eligible 样本上直接均值",
            "interpretable_fraction_denominator": "target-eligible 样本数",
            "nan_contract": "MedAE 仅 interpretable 时有限；coverage 仅 target-eligible 时有限；禁止 Inf 与额外过滤",
            "aggregation": "每个样本的周期误差先取 median；checkpoint 对有效样本直接 mean；三 seed arithmetic mean 与 ddof=1 SD",
            "bridge_scope": "共享事件算法使 IBI 成为当前更直接的跨输出节律桥梁；不同支撑上的事件、Pi_T、检测边界和可解释子集仍不同",
            "equal_io_target_eligible_n_range": [int(ladder.ibi_target_eligible_n_min.min()), int(ladder.ibi_target_eligible_n_max.max())],
            "equal_io_coverage_mean_range": [float(ladder.ibi_coverage_mean.min()), float(ladder.ibi_coverage_mean.max())],
            "equal_io_interpretable_fraction_mean_range": [float(ladder.ibi_interpretable_fraction_mean.min()), float(ladder.ibi_interpretable_fraction_mean.max())],
            "adjacent_interpretable_jaccard_range": [float(supports.interpretable_jaccard.min()), float(supports.interpretable_jaccard.max())],
            "eligible_changed_rows_max": int(supports.eligible_changed_n.max()),
            "adjacent_seed_directions": directions,
            "selection_bias_limit": "target eligibility 集合一致可以排除该层面的集合变化；可解释样本比例接近不能排除其成员变化或周期组成变化"},
        "local_rr": {"historical180": "固定60s窗口、15s step，共9窗口，先样本内平均；历史方法内部可比",
            "center60": "native RR 为单个完整中心60s窗口，与历史Local RR具有相同观测长度，但时间位置、Pi作用域、窗口聚合不同，不能直接等同历史9窗均值",
            "center30": "输出不足60s，不能定义60s Local RR", "center90": "冻结metrics未提供统一60s Local RR",
            "cross_task_comparable_column_generated": False},
        "identity": {"w_models": ["W-reduced-center30", "W-reduced-center60", "W-reduced-center90", "W3", "W0"],
            "rtm": "五候选完整报告为 audit-only，保留结论锁待用户确认状态，不进入论文已冻结主表",
            "d4": "validation-only，test汇总不纳入"},
        "followup": {"required_for_this_summary": False, "status": "not_implemented_or_authorized",
            "possible_metric": "common_center30_rr_mae_bpm", "design": "所有输出取父窗口相同中心30s，统一Pi_30与FFT，一样本一次",
            "approval_prerequisites": ["validation-selected checkpoint完整矩阵", "test重新访问范围", "两GPU任务数与成本", "全部context arms或仅等长轨迹", "历史180模型名单"]},
        "weighted_score_used": False, "cross_task_ranking_used": False, "checkpoint_model_or_window_reselection_used": False,
        "context_vs_output": "固定输出增加外侧输入上下文，与同时增加输入输出持续时间，是两个不同问题",
    }


def rhythm_ladder(arms: pd.DataFrame) -> pd.DataFrame:
    mask = arms.input_sec.eq(arms.output_sec) & (arms.model.str.startswith(("C201", "W-reduced")) | arms.model.isin(["W0", "W3"]))
    return arms.loc[mask].sort_values(["output_sec", "model"]).reset_index(drop=True)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _artifacts(output: Path) -> list[dict[str, Any]]:
    return [{"filename": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()]


def run_summary(
    *,
    repo_root: Path,
    command: str,
    output_dir: Path = OUTPUT_DIR,
    protocol_id: str = PROTOCOL_ID,
    schema_version: str = "harmonized-rhythm-test-summary-v1",
    supersedes: dict[str, Any] | None = None,
) -> Path:
    root = repo_root.resolve()
    output = root / output_dir
    if output.exists():
        raise FileExistsError(f"汇总输出目录禁止覆盖: {output}")
    git = lambda *args: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("正式汇总要求干净实现 commit")
    seeds, supports, reader = audit_inputs(root)
    arms = aggregate_arms(seeds)
    ladder = rhythm_ladder(arms)
    comparability = build_comparability(arms, supports, protocol_id=protocol_id)
    reader.verify_unchanged()
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("汇总读取期间 Git identity 变化")
    # 独占创建固定目录；任何写出失败留下目录，后续调用一律拒绝复用。
    output.mkdir(parents=True, exist_ok=False)
    for filename, frame in (("native_whole_rr_ibi_all_arms.csv", arms), ("equal_input_output_rhythm_ladder.csv", ladder),
                            ("per_checkpoint_rhythm_summary.csv", seeds), ("equal_io_support_overlap.csv", supports)):
        with (output / filename).open("x", encoding="utf-8") as handle:
            frame.to_csv(handle, index=False)
    _write_json(output / "rhythm_comparability.json", comparability)
    reader.verify_unchanged()
    _write_json(output / "summary_receipt.json", {
        "protocol_id": protocol_id, "schema_version": schema_version, "status": "complete",
        "supersedes": supersedes,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {"command": command, "cwd": str(root), "git_commit": commit, "git_dirty": False,
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyyaml": yaml.__version__},
        "inputs": list(reader.records.values()), "source_hashes_unchanged": True,
        "counts": {"arms": len(arms), "equal_io_ladder_arms": len(ladder), "evaluation_records": len(seeds),
            "learning_checkpoints": int((~seeds.deterministic).sum()), "deterministic_records": int(seeds.deterministic.sum()),
            "sample_metric_rows": int(seeds.n_samples.sum()), "unique_test_rows": 2310, "row_id_sha256": ROW_HASH},
        "aggregation": "sample-direct mean; arithmetic seed mean; sample SD ddof=1; deterministic-no-seed SD undefined",
        "access": {"frozen_test_metrics_read": True, "frozen_summary_read": True, "provenance_read": True,
            "checkpoint_content_read": False, "dataset_or_index_read": False, "signal_or_target_array_read": False,
            "validation_metrics_read": False, "model_inference_used": False, "training_used": False, "gpu_used": False,
            "cache_used": False, "benchmark_used": False, "checkpoint_reselection_used": False},
        "artifacts": _artifacts(output),
    })
    _write_json(
        output / "artifact_manifest.json",
        {
            "protocol_id": protocol_id,
            "schema_version": schema_version,
            "status": "complete",
            "supersedes": supersedes,
            "files": _artifacts(output),
        },
    )
    return output


def run_summary_v2(*, repo_root: Path, command: str) -> Path:
    return run_summary(
        repo_root=repo_root,
        command=command,
        output_dir=V2_OUTPUT_DIR,
        protocol_id=V2_PROTOCOL_ID,
        schema_version="harmonized-rhythm-test-summary-v2",
        supersedes={
            "protocol_id": PROTOCOL_ID,
            "output_dir": str(OUTPUT_DIR),
            "scope": "center30_center60_center90_checkpoint_selector_labels",
            "numeric_metrics_changed": False,
        },
    )
