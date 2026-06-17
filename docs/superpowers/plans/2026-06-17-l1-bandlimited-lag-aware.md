# L1 低频波形损失与 Lag-Aware 评价实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不改模型结构的前提下，先建立 split 独立性审计，再新增 lag-aware 评价指标和 band-limited waveform loss，用于验证 PSG 胸带呼吸信号的低频波形可恢复性。

**架构：** 保留 `WeakSyncLoss` 现有 envelope、spectrum、smooth、高频和相对包络分量，新增默认关闭的低频波形分量。评价侧新增可容忍小范围时间偏移的 best-lag correlation，训练侧第一轮只加入 band-limited waveform loss，不把 lag search 放进训练目标。数据侧只做独立性审计，不在本计划中重写数据切分。

**技术栈：** Python、PyTorch FFT、NumPy、pandas、OmegaConf、pytest、现有 `resp_train` 训练与评价框架。

---

## 执行状态

- 已完成：split 独立性审计，提交 `8943425 feat: 增加数据划分独立性审计`。
- 已完成：lag-aware 评价指标，提交 `1ea3ee2 feat: 增加 lag-aware 呼吸评价指标`。
- 已完成：默认关闭的 band-limited waveform loss，提交 `b239473 feat: 增加带限波形损失`。
- 已完成：L1 实验命令、阶段说明和 run 汇总指标接入。
- 已完成：完整测试、pilot 同口径 split 审计、L0/L1 pilot run。

### Pilot 结果摘要

pilot 口径：`4096/1024` 窗口、`3` epoch、固定 `train_sample_seed=20260610` 和 `val_sample_seed=20260611`。匹配 pilot 的 split 审计显示 `overlap_samp_id_count=0`、`overlap_segment_count=0`。

| run | band_waveform_weight | val_loss | band_limited_corr_mean | best_lag_corr_mean | best_lag_sec_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20260617_120905_055953` | 0.0 | 0.611827 | 0.783169 | 0.843794 | 0.109170 |
| `20260617_121328_250481` | 0.2 | 0.714597 | 0.778134 | 0.841708 | 0.121895 |

初步结论：`band_waveform_weight=0.2` 没有改善低频相关或 lag-aware 相关，且加权 `val_loss` 上升。下一轮不建议直接加大该权重；更合理的是尝试更小权重、降低 spectrum 权重，或把 lag tolerance 引入训练目标前先看诊断图确认失败形态。

收尾诊断已生成到 `runs/tho_research_v2/l1_closeout_20260617/`。逐窗图像复核显示，L1 仍主要贴近主频和频谱峰，但没有稳定修复相位连续性或局部波形形态；后续应推进 L2 的 phase / lag-aware training loss，而不是继续围绕当前 L1 权重做大规模实验。

---

## 范围与非目标

本计划包含：

- 审计当前 train/val 在 `samp_id` 和 `segment_id` 层面的独立性风险。
- 在评价指标中加入 `band_limited_corr`、`best_lag_corr` 和 `best_lag_sec`。
- 在 `WeakSyncLoss` 中加入默认关闭的 `band_waveform` 分量。
- 给出 L0 与 L1 pilot 实验命令。

本计划不包含：

- 不实现 subject-level split 生成器。
- 不做 M1-M5 模型结构实验。
- 不加入 STFT complex loss。
- 不继续扩大 `relative_envelope_weight`。
- 不把 ±lag search 直接加入训练 loss。

## 文件结构

- 创建：`resp_train/data/independence.py`
  - 职责：根据已过滤和抽样后的 train/val rows 生成 split 独立性审计表。
- 创建：`scripts/audit_split_independence.py`
  - 职责：命令行运行当前配置的数据独立性审计，输出 CSV 和关键风险摘要。
- 创建：`tests/test_split_independence.py`
  - 职责：覆盖 `samp_id` overlap、`segment_id` overlap、窗口数量偏斜和分布表。
- 修改：`resp_train/metrics/signal.py`
  - 职责：新增低频滤波相关指标和 best-lag correlation。
- 修改：`resp_train/metrics/evaluate.py`
  - 职责：把 lag-aware 指标写入 `metrics.csv`。
- 修改：`tests/test_signal_metrics.py`
  - 职责：覆盖 band-limited correlation 与 best-lag behavior。
- 修改：`tests/test_eval_metrics.py`
  - 职责：覆盖新增评价列。
- 修改：`resp_train/losses/weak.py`
  - 职责：新增 differentiable band-limited waveform loss。
- 修改：`tests/test_losses.py`
  - 职责：覆盖默认关闭、分量返回、相位错位惩罚。
- 修改：`configs/tho_research_v2.yaml`
  - 职责：显式加入默认关闭的 loss/evaluation 配置。
- 修改：`docs/tho_small_training.md`
  - 职责：记录新一轮实验口径，不写入未执行的结果。
- 修改：`scripts/README.md`
  - 职责：补充独立性审计和 L1 pilot 命令。
- 修改：`scripts/summarize_tho_runs.py`
  - 职责：把新增 lag-aware 指标纳入 run 汇总。
- 修改：`tests/test_diagnostics_scripts.py`
  - 职责：覆盖新增汇总列。

---

### 任务 1：新增 Split 独立性审计

**文件：**
- 创建：`resp_train/data/independence.py`
- 创建：`scripts/audit_split_independence.py`
- 创建：`tests/test_split_independence.py`
- 修改：`scripts/README.md`

- [ ] **步骤 1：编写失败的独立性审计测试**

在 `tests/test_split_independence.py` 中创建：

```python
import pandas as pd

from resp_train.data.independence import audit_split_independence


def test_audit_split_independence_detects_samp_and_segment_overlap():
    train = pd.DataFrame(
        [
            {"dataset_row_id": 1, "samp_id": 88, "segment_id": "a", "allowed_losses": "waveform", "valid_ratio": 0.9},
            {"dataset_row_id": 2, "samp_id": 89, "segment_id": "b", "allowed_losses": "rate", "valid_ratio": 0.8},
        ]
    )
    val = pd.DataFrame(
        [
            {"dataset_row_id": 3, "samp_id": 88, "segment_id": "a", "allowed_losses": "waveform", "valid_ratio": 0.95},
            {"dataset_row_id": 4, "samp_id": 90, "segment_id": "c", "allowed_losses": "rate", "valid_ratio": 0.7},
        ]
    )

    report = audit_split_independence(train, val)

    summary = report["summary"]
    assert int(summary.loc[0, "train_windows"]) == 2
    assert int(summary.loc[0, "val_windows"]) == 2
    assert int(summary.loc[0, "overlap_samp_id_count"]) == 1
    assert int(summary.loc[0, "overlap_segment_count"]) == 1
    assert bool(summary.loc[0, "has_samp_id_leakage"]) is True
    assert bool(summary.loc[0, "has_segment_leakage"]) is True


def test_audit_split_independence_reports_distribution_shift():
    train = pd.DataFrame(
        [
            {"dataset_row_id": 1, "samp_id": 1, "segment_id": "a", "allowed_losses": "waveform", "valid_ratio": 0.9},
            {"dataset_row_id": 2, "samp_id": 2, "segment_id": "b", "allowed_losses": "waveform", "valid_ratio": 0.8},
        ]
    )
    val = pd.DataFrame(
        [
            {"dataset_row_id": 3, "samp_id": 3, "segment_id": "c", "allowed_losses": "rate", "valid_ratio": 0.4},
        ]
    )

    report = audit_split_independence(train, val, categorical_columns=("allowed_losses",), numeric_columns=("valid_ratio",))

    distribution = report["categorical_distribution"]
    numeric = report["numeric_distribution"]
    assert set(distribution["column"]) == {"allowed_losses"}
    assert set(distribution["split"]) == {"train", "val"}
    assert numeric.loc[numeric["split"].eq("train"), "valid_ratio_mean"].iloc[0] == 0.85
    assert numeric.loc[numeric["split"].eq("val"), "valid_ratio_mean"].iloc[0] == 0.4
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_split_independence.py -v
```

预期：FAIL，报错包含 `No module named 'resp_train.data.independence'`。

- [ ] **步骤 3：实现独立性审计模块**

创建 `resp_train/data/independence.py`：

```python
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


DEFAULT_CATEGORICAL_COLUMNS = ("allowed_losses", "residual_quality_class", "input_set")
DEFAULT_NUMERIC_COLUMNS = ("valid_ratio", "input_finite_ratio", "target_finite_ratio")


def audit_split_independence(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    *,
    categorical_columns: Iterable[str] = DEFAULT_CATEGORICAL_COLUMNS,
    numeric_columns: Iterable[str] = DEFAULT_NUMERIC_COLUMNS,
) -> dict[str, pd.DataFrame]:
    """审计 train/val 是否在个体和片段层面独立，并给出基础分布对照。"""
    train = train_rows.copy()
    val = val_rows.copy()
    _require_columns(train, ("samp_id", "segment_id"))
    _require_columns(val, ("samp_id", "segment_id"))

    train_samp = set(train["samp_id"].dropna().astype(str))
    val_samp = set(val["samp_id"].dropna().astype(str))
    train_segments = set(_segment_keys(train))
    val_segments = set(_segment_keys(val))
    overlap_samp = sorted(train_samp & val_samp)
    overlap_segments = sorted(train_segments & val_segments)

    summary = pd.DataFrame(
        [
            {
                "train_windows": int(len(train)),
                "val_windows": int(len(val)),
                "train_samp_id_count": int(len(train_samp)),
                "val_samp_id_count": int(len(val_samp)),
                "overlap_samp_id_count": int(len(overlap_samp)),
                "overlap_segment_count": int(len(overlap_segments)),
                "has_samp_id_leakage": bool(overlap_samp),
                "has_segment_leakage": bool(overlap_segments),
                "max_train_windows_per_samp_id": _max_group_size(train, "samp_id"),
                "max_val_windows_per_samp_id": _max_group_size(val, "samp_id"),
            }
        ]
    )

    overlap_samp_frame = pd.DataFrame({"samp_id": overlap_samp})
    overlap_segment_frame = pd.DataFrame({"segment_key": overlap_segments})
    categorical = _categorical_distribution(train, val, categorical_columns)
    numeric = _numeric_distribution(train, val, numeric_columns)
    per_samp = _per_samp_summary(train, val)
    return {
        "summary": summary,
        "overlap_samp_id": overlap_samp_frame,
        "overlap_segment": overlap_segment_frame,
        "categorical_distribution": categorical,
        "numeric_distribution": numeric,
        "per_samp_id": per_samp,
    }


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"独立性审计缺少必需列: {missing}")


def _segment_keys(frame: pd.DataFrame) -> list[str]:
    return (
        frame[["samp_id", "segment_id"]]
        .dropna()
        .astype(str)
        .agg("::".join, axis=1)
        .tolist()
    )


def _max_group_size(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame.groupby(column, dropna=False).size().max())


def _categorical_distribution(
    train: pd.DataFrame,
    val: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for column in columns:
        if column not in train.columns or column not in val.columns:
            continue
        for split_name, frame in (("train", train), ("val", val)):
            counts = frame[column].fillna("__MISSING__").astype(str).value_counts(dropna=False)
            total = max(int(counts.sum()), 1)
            for value, count in counts.items():
                records.append(
                    {
                        "column": str(column),
                        "split": split_name,
                        "value": str(value),
                        "count": int(count),
                        "ratio": float(count) / float(total),
                    }
                )
    return pd.DataFrame.from_records(records)


def _numeric_distribution(
    train: pd.DataFrame,
    val: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for split_name, frame in (("train", train), ("val", val)):
        record: dict[str, object] = {"split": split_name}
        for column in columns:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            record[f"{column}_mean"] = float(values.mean())
            record[f"{column}_median"] = float(values.median())
            record[f"{column}_p10"] = float(values.quantile(0.10))
            record[f"{column}_p90"] = float(values.quantile(0.90))
        records.append(record)
    return pd.DataFrame.from_records(records)


def _per_samp_summary(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    records = []
    for split_name, frame in (("train", train), ("val", val)):
        grouped = frame.groupby("samp_id", dropna=False).size().reset_index(name="window_count")
        grouped["split"] = split_name
        records.append(grouped[["split", "samp_id", "window_count"]])
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["split", "samp_id", "window_count"])
```

- [ ] **步骤 4：实现命令行审计脚本**

创建 `scripts/audit_split_independence.py`：

```python
from __future__ import annotations

import argparse
from pathlib import Path

from resp_train.config import load_config
from resp_train.data.factory import build_tho_data
from resp_train.data.independence import audit_split_independence


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 THO train/val split 的个体和片段独立性")
    parser.add_argument("--config", required=True, help="训练配置路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖")
    parser.add_argument("--output-dir", required=True, help="审计 CSV 输出目录")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    data = build_tho_data(cfg)
    report = audit_split_independence(data.train.rows, data.val.rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in report.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    summary = report["summary"].iloc[0].to_dict()
    print(
        "split_independence "
        f"train_windows={summary['train_windows']} "
        f"val_windows={summary['val_windows']} "
        f"overlap_samp_id_count={summary['overlap_samp_id_count']} "
        f"overlap_segment_count={summary['overlap_segment_count']}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **步骤 5：运行审计测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_split_independence.py -v
```

预期：PASS，2 个测试通过。

- [ ] **步骤 6：补充脚本文档**

在 `scripts/README.md` 增加独立性审计命令：

```markdown
## Split 独立性审计

在解释 L1 或模型实验前，先检查 train/val 是否共享 `samp_id` 或 `(samp_id, segment_id)`：

```bash
./.venv/bin/python scripts/audit_split_independence.py \
  --config configs/tho_research_v2.yaml \
  --output-dir runs/audits/split_independence_research_v2
```

若 `summary.csv` 中 `overlap_samp_id_count` 或 `overlap_segment_count` 大于 0，当前结果只能作为 within-subject / within-dataset 开发指标，不应作为跨个体泛化结论。
```

- [ ] **步骤 7：Commit**

运行：

```bash
git add resp_train/data/independence.py scripts/audit_split_independence.py tests/test_split_independence.py scripts/README.md
git commit -m "feat: 增加数据划分独立性审计"
```

---

### 任务 2：新增 Lag-Aware 评价指标

**文件：**
- 修改：`resp_train/metrics/signal.py`
- 修改：`resp_train/metrics/evaluate.py`
- 修改：`tests/test_signal_metrics.py`
- 修改：`tests/test_eval_metrics.py`
- 修改：`configs/tho_research_v2.yaml`

- [ ] **步骤 1：编写 signal metrics 失败测试**

在 `tests/test_signal_metrics.py` 中新增：

```python
def test_band_limited_corr_ignores_high_frequency_noise():
    fs = 100
    t = np.arange(0, 60, 1 / fs)
    target = np.sin(2 * np.pi * 0.25 * t)
    pred = target + 0.5 * np.sin(2 * np.pi * 8.0 * t)

    corr = band_limited_corr(pred, target, fs=fs, low_hz=0.05, high_hz=0.7, order=4)

    assert corr > 0.98


def test_best_lag_correlation_recovers_positive_delay_seconds():
    fs = 100
    t = np.arange(0, 60, 1 / fs)
    target = np.sin(2 * np.pi * 0.25 * t)
    delay_sec = 0.5
    shift = int(round(delay_sec * fs))
    pred = np.roll(target, shift)

    metrics = best_lag_correlation(pred, target, fs=fs, max_lag_sec=1.0, low_hz=0.05, high_hz=0.7)

    assert metrics["best_lag_corr"] > 0.99
    assert abs(metrics["best_lag_sec"] - delay_sec) <= 1 / fs
```

同时在导入区加入：

```python
from resp_train.metrics.signal import band_limited_corr, best_lag_correlation
```

- [ ] **步骤 2：运行 signal metrics 测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_signal_metrics.py::test_band_limited_corr_ignores_high_frequency_noise tests/test_signal_metrics.py::test_best_lag_correlation_recovers_positive_delay_seconds -v
```

预期：FAIL，报错包含 `cannot import name 'band_limited_corr'` 或 `cannot import name 'best_lag_correlation'`。

- [ ] **步骤 3：实现 lag-aware 指标**

在 `resp_train/metrics/signal.py` 的 `bandpass_filter` 后加入：

```python
def band_limited_corr(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    fs: float,
    low_hz: float = 0.05,
    high_hz: float = 0.7,
    order: int = 4,
) -> float:
    """先限制到呼吸频带，再计算预测与胸带参考的相关系数。"""
    pred_band = bandpass_filter(pred, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    target_band = bandpass_filter(target, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    return _corrcoef_or_nan(pred_band, target_band)


def best_lag_correlation(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    fs: float,
    max_lag_sec: float = 1.0,
    low_hz: float = 0.05,
    high_hz: float = 0.7,
    order: int = 4,
) -> dict[str, float]:
    """在 ±max_lag_sec 内寻找最高低频相关，正 lag 表示 pred 相对 target 滞后。"""
    pred_band = bandpass_filter(pred, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    target_band = bandpass_filter(target, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    if pred_band.shape != target_band.shape:
        raise ValueError(f"pred 和 target 长度必须一致，当前 {pred_band.shape} != {target_band.shape}")
    max_lag = int(round(float(max_lag_sec) * float(fs)))
    if max_lag < 0:
        raise ValueError(f"max_lag_sec 必须非负，当前={max_lag_sec}")

    best_corr = float("-inf")
    best_lag = 0
    n = pred_band.size
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            pred_slice = pred_band[: n + lag]
            target_slice = target_band[-lag:]
        elif lag > 0:
            pred_slice = pred_band[lag:]
            target_slice = target_band[: n - lag]
        else:
            pred_slice = pred_band
            target_slice = target_band
        if pred_slice.size < 2:
            continue
        corr = _corrcoef_or_nan(pred_slice, target_slice)
        if np.isfinite(corr) and corr > best_corr:
            best_corr = float(corr)
            best_lag = lag
    if not np.isfinite(best_corr):
        return {"best_lag_corr": float("nan"), "best_lag_sec": float("nan")}
    return {"best_lag_corr": best_corr, "best_lag_sec": float(best_lag) / float(fs)}
```

- [ ] **步骤 4：运行 signal metrics 测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_signal_metrics.py::test_band_limited_corr_ignores_high_frequency_noise tests/test_signal_metrics.py::test_best_lag_correlation_recovers_positive_delay_seconds -v
```

预期：PASS。

- [ ] **步骤 5：编写 evaluate 失败测试**

在 `tests/test_eval_metrics.py::test_evaluate_prediction_dict_returns_window_metrics` 的断言后加入：

```python
    assert frame.loc[0, "band_limited_corr"] > 0.99
    assert frame.loc[0, "best_lag_corr"] > 0.99
    assert abs(frame.loc[0, "best_lag_sec"]) < 1e-6
```

再新增：

```python
def test_evaluate_prediction_dict_reports_best_lag_for_shifted_prediction():
    fs = 100
    t = np.arange(0, 60, 1 / fs)
    target = np.sin(2 * np.pi * 0.25 * t).astype(np.float32)
    pred = np.roll(target, int(0.5 * fs)).astype(np.float32)
    cfg = _cfg()
    cfg.evaluation = {"max_lag_sec": 1.0, "lag_bandpass_order": 4}
    preds = {
        "r_tho_hat": pred.reshape(1, 1, -1),
        "tho_ref": target.reshape(1, 1, -1),
        "dataset_row_id": np.asarray([1]),
        "split": np.asarray(["val"]),
        "input_set": np.asarray(["research_v2_waveform"]),
        "residual_quality_class": np.asarray(["waveform"]),
    }

    frame = evaluate_prediction_dict(preds, cfg, method="model")

    assert frame.loc[0, "best_lag_corr"] > 0.99
    assert abs(frame.loc[0, "best_lag_sec"] - 0.5) <= 1 / fs
```

- [ ] **步骤 6：运行 evaluate 测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_eval_metrics.py::test_evaluate_prediction_dict_returns_window_metrics tests/test_eval_metrics.py::test_evaluate_prediction_dict_reports_best_lag_for_shifted_prediction -v
```

预期：FAIL，报错包含 `band_limited_corr` 列不存在或 `cannot import name`。

- [ ] **步骤 7：把指标接入 evaluate**

修改 `resp_train/metrics/evaluate.py` 的 import：

```python
from resp_train.metrics.signal import (
    band_limited_corr,
    best_lag_correlation,
    estimate_peak_rate_bpm,
    estimate_spectral_rate_bpm,
    relative_envelope_metrics,
    rms_envelope,
    spectrum_similarity,
)
```

在 `evaluate_prediction_dict()` 中 `env_window` 后加入：

```python
    max_lag_sec = float(cfg.get("evaluation", {}).get("max_lag_sec", 1.0))
    lag_bandpass_order = int(cfg.get("evaluation", {}).get("lag_bandpass_order", 4))
```

在每个窗口循环中 `rel_env = ...` 后加入：

```python
        lag_metrics = best_lag_correlation(
            pred,
            target,
            fs=fs,
            max_lag_sec=max_lag_sec,
            low_hz=low_hz,
            high_hz=high_hz,
            order=lag_bandpass_order,
        )
```

在 record 中加入：

```python
                "band_limited_corr": band_limited_corr(
                    pred,
                    target,
                    fs=fs,
                    low_hz=low_hz,
                    high_hz=high_hz,
                    order=lag_bandpass_order,
                ),
                "best_lag_corr": lag_metrics["best_lag_corr"],
                "best_lag_sec": lag_metrics["best_lag_sec"],
```

- [ ] **步骤 8：配置默认 evaluation 参数**

在 `configs/tho_research_v2.yaml` 的 `loss` 块后、`training` 块前加入：

```yaml
evaluation:
  max_lag_sec: 1.0
  lag_bandpass_order: 4
```

- [ ] **步骤 9：运行 evaluate 相关测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_signal_metrics.py tests/test_eval_metrics.py -v
```

预期：PASS。

- [ ] **步骤 10：Commit**

运行：

```bash
git add resp_train/metrics/signal.py resp_train/metrics/evaluate.py tests/test_signal_metrics.py tests/test_eval_metrics.py configs/tho_research_v2.yaml
git commit -m "feat: 增加低频 lag-aware 评价指标"
```

---

### 任务 3：新增默认关闭的 Band-Limited Waveform Loss

**文件：**
- 修改：`resp_train/losses/weak.py`
- 修改：`tests/test_losses.py`
- 修改：`configs/tho_research_v2.yaml`

- [ ] **步骤 1：编写失败的 loss 测试**

在 `tests/test_losses.py` 中新增：

```python
def test_band_waveform_weight_zero_keeps_total_loss_unchanged():
    cfg = _cfg()
    cfg.loss.band_waveform_weight = 0.0
    loss_fn = WeakSyncLoss(cfg)
    pred = torch.randn(2, 1, 1800)
    target = torch.randn(2, 1, 1800)

    total, parts = loss_fn(pred, target)
    expected = (
        cfg.loss.envelope_weight * parts["envelope"]
        + cfg.loss.spectrum_weight * parts["spectrum"]
        + cfg.loss.smooth_weight * parts["smooth"]
        + cfg.loss.high_freq_weight * parts["high_freq"]
        + cfg.loss.get("relative_envelope_weight", 0.0) * parts["relative_envelope"]
    )

    assert "band_waveform" in parts
    assert torch.allclose(total, expected)


def test_band_waveform_loss_penalizes_low_frequency_phase_mismatch():
    cfg = _cfg()
    cfg.loss.envelope_weight = 0.0
    cfg.loss.spectrum_weight = 0.0
    cfg.loss.smooth_weight = 0.0
    cfg.loss.high_freq_weight = 0.0
    cfg.loss.band_waveform_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 60, 1 / fs)
    target = torch.sin(2 * torch.pi * 0.25 * time).reshape(1, 1, -1)
    good = target.clone()
    bad = torch.sin(2 * torch.pi * 0.25 * time + torch.pi / 2).reshape(1, 1, -1)

    good_total, good_parts = loss_fn(good, target)
    bad_total, bad_parts = loss_fn(bad, target)

    assert good_parts["band_waveform"] < 0.01
    assert bad_parts["band_waveform"] > good_parts["band_waveform"] + 0.5
    assert bad_total > good_total + 0.5
```

同时把 `test_weak_sync_loss_returns_components_and_scalar` 的分量断言改为：

```python
    assert set(parts) == {"envelope", "spectrum", "smooth", "high_freq", "relative_envelope", "band_waveform"}
```

- [ ] **步骤 2：运行 loss 测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_losses.py::test_weak_sync_loss_returns_components_and_scalar tests/test_losses.py::test_band_waveform_weight_zero_keeps_total_loss_unchanged tests/test_losses.py::test_band_waveform_loss_penalizes_low_frequency_phase_mismatch -v
```

预期：FAIL，报错包含 `"band_waveform" in parts` 失败。

- [ ] **步骤 3：实现 band-limited waveform loss**

在 `resp_train/losses/weak.py::__init__` 中加入：

```python
        self.band_waveform_weight = float(cfg.loss.get("band_waveform_weight", 0.0))
```

在 `forward()` 中 `relative_env = ...` 后加入：

```python
        band_waveform = self._band_waveform_loss(pred, target)
```

在 total 里加入：

```python
            + self.band_waveform_weight * band_waveform
```

在返回 parts 中加入：

```python
            "band_waveform": band_waveform.detach(),
```

在 `_spectrum_loss()` 后加入：

```python
    def _band_waveform_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_band = self._fft_bandpass(pred)
        target_band = self._fft_bandpass(target)
        pred_norm = self._zscore(pred_band)
        target_norm = self._zscore(target_band)
        corr = torch.mean(pred_norm * target_norm, dim=-1)
        return torch.mean(1.0 - corr)

    def _fft_bandpass(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)
        spec = torch.fft.rfft(centered, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.fs).to(x.device)
        mask = ((freqs >= self.low_hz) & (freqs <= self.high_hz)).to(spec.dtype)
        if not bool(torch.any(mask.bool())):
            raise ValueError(
                f"低频波形损失频带为空: low_hz={self.low_hz} high_hz={self.high_hz} "
                f"fs={self.fs} n={x.shape[-1]}"
            )
        filtered = torch.fft.irfft(spec * mask.view(1, 1, -1), n=x.shape[-1], dim=-1)
        return filtered
```

- [ ] **步骤 4：更新默认配置**

在 `configs/tho_research_v2.yaml` 的 `loss` 下加入：

```yaml
  band_waveform_weight: 0.0
```

默认关闭，确保现有 L0 训练口径不漂移。

- [ ] **步骤 5：运行 loss 测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_losses.py -v
```

预期：PASS。

- [ ] **步骤 6：运行训练 smoke 测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_engine_smoke.py tests/test_tho_experiment.py -v
```

预期：PASS。

- [ ] **步骤 7：Commit**

运行：

```bash
git add resp_train/losses/weak.py tests/test_losses.py configs/tho_research_v2.yaml
git commit -m "feat: 增加低频波形损失"
```

---

### 任务 4：补充实验命令与结果记录框架

**文件：**
- 修改：`docs/tho_small_training.md`
- 修改：`scripts/README.md`
- 修改：`scripts/summarize_tho_runs.py`
- 修改：`tests/test_diagnostics_scripts.py`
- 可创建：`docs/experiments/l1_bandlimited_lag_aware.md`

- [ ] **步骤 1：补充训练文档中的下一轮目标**

在 `docs/tho_small_training.md` 的“下一步优先方向”附近加入：

```markdown
### L1 低频波形损失与 Lag-Aware 评价

下一轮先不改模型结构，只比较：

- L0：当前 `WeakSyncLoss`。
- L1：在 L0 基础上加入 `loss.band_waveform_weight`。

评价必须同时查看原有 RR / spectrum / envelope 指标，以及新增的：

- `band_limited_corr`
- `best_lag_corr`
- `best_lag_sec`

`best_lag_sec` 只作为弱同步诊断，不直接说明模型更好；若 lag 集中偏向同一方向，应优先回查输入与胸带参考的对齐。
```

- [ ] **步骤 2：补充 README 中的 pilot 命令**

在 `scripts/README.md` 增加：

```markdown
## L1 低频波形损失 Pilot

L0 baseline：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=4096 \
  --set data.max_val_windows=1024 \
  --set training.epochs=20 \
  --set training.patience=5 \
  --set outputs.run_root=runs/tho_research_v2_l0_lagaware
```

L1 band-limited waveform loss：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=4096 \
  --set data.max_val_windows=1024 \
  --set training.epochs=20 \
  --set training.patience=5 \
  --set loss.band_waveform_weight=0.1 \
  --set outputs.run_root=runs/tho_research_v2_l1_bandwave_010
```

需要 GPU 的正式训练应在沙盒外运行，不要为了绕过 GPU 权限改用 CPU。
```

- [ ] **步骤 3：编写新增汇总列的失败测试**

在 `tests/test_diagnostics_scripts.py::_write_minimal_run` 的 `metrics.csv` 记录中加入：

```python
                "band_limited_corr": 0.67,
                "best_lag_corr": 0.72,
                "best_lag_sec": 0.4,
```

新增测试：

```python
def test_summarize_runs_includes_lag_aware_metrics(tmp_path):
    root = tmp_path / "runs"
    _write_minimal_run(root / "run_a")
    output = tmp_path / "summary.csv"

    frame = summarize_runs(root, output)

    assert frame["model_band_limited_corr_mean"].tolist() == [0.67]
    assert frame["model_best_lag_corr_median"].tolist() == [0.72]
    assert frame["model_best_lag_sec_mean"].tolist() == [0.4]
```

- [ ] **步骤 4：运行汇总测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_diagnostics_scripts.py::test_summarize_runs_includes_lag_aware_metrics -v
```

预期：FAIL，报错包含 `model_band_limited_corr_mean`。

- [ ] **步骤 5：把新增指标加入汇总脚本**

修改 `scripts/summarize_tho_runs.py` 的 `METRIC_COLUMNS`：

```python
METRIC_COLUMNS = [
    "rr_spec_abs_error",
    "rr_peak_abs_error",
    "envelope_corr",
    "relative_envelope_corr",
    "relative_envelope_mae",
    "band_limited_corr",
    "best_lag_corr",
    "best_lag_sec",
    "spectrum_similarity",
]
```

- [ ] **步骤 6：运行汇总测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_diagnostics_scripts.py::test_summarize_runs_includes_lag_aware_metrics tests/test_diagnostics_scripts.py::test_summarize_runs_writes_one_row_per_run tests/test_diagnostics_scripts.py::test_summarize_runs_includes_relative_envelope_metrics -v
```

预期：PASS。

- [ ] **步骤 7：创建实验记录模板**

创建 `docs/experiments/l1_bandlimited_lag_aware.md`：

```markdown
# L1 低频波形损失与 Lag-Aware 评价实验

## 实验问题

目标是验证 PSG 胸带呼吸信号的低频波形形态是否能被更直接地约束，而不是只改善 envelope、spectrum 或 RR 指标。

## 数据独立性审计

| 审计项 | 数值 | 解释 |
|---|---:|---|
| train windows | NaN | 未运行 |
| val windows | NaN | 未运行 |
| overlap samp_id count | NaN | 大于 0 时只解释为 within-subject 开发结果 |
| overlap segment count | NaN | 大于 0 时只解释为 window-level 开发结果 |

## 实验对照

| 实验 | run | band_waveform_weight | best epoch | best val loss | band_limited_corr mean | best_lag_corr mean | best_lag_sec median | rr_peak_abs_error mean | spectrum_similarity mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L0 | 未运行 | 0.0 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | 未运行 |
| L1 | 未运行 | 0.1 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | 未运行 |

## 判定规则

- L1 必须提升 `band_limited_corr` 或 `best_lag_corr`，同时不能明显恶化 RR 和 spectrum 指标。
- 如果 `best_lag_sec` 绝对值系统性偏大，先回查对齐，不把它解释为模型收益。
- 如果 L1 只改善 loss 但诊断图仍像 BCG 尖峰，本轮不进入模型结构实验。
```

- [ ] **步骤 8：运行文档相关轻量检查**

运行：

```bash
rg -n "band_waveform_weight|best_lag_corr|split_independence|L1 低频波形" docs scripts configs
```

预期：输出包含 `docs/tho_small_training.md`、`scripts/README.md`、`docs/experiments/l1_bandlimited_lag_aware.md` 和 `configs/tho_research_v2.yaml`。

- [ ] **步骤 9：Commit**

运行：

```bash
git add docs/tho_small_training.md scripts/README.md scripts/summarize_tho_runs.py tests/test_diagnostics_scripts.py docs/experiments/l1_bandlimited_lag_aware.md
git commit -m "docs: 记录低频波形损失实验方案"
```

---

### 任务 5：整体验证与 Pilot 执行

**文件：**
- 不新增源码文件
- 生成：`runs/audits/split_independence_research_v2/*.csv`
- 生成：`runs/tho_research_v2_l0_lagaware/` 下的时间戳 run 子目录
- 生成：`runs/tho_research_v2_l1_bandwave_010/` 下的时间戳 run 子目录
- 修改：`docs/experiments/l1_bandlimited_lag_aware.md`

- [ ] **步骤 1：运行完整相关测试**

运行：

```bash
./.venv/bin/python -m pytest \
  tests/test_split_independence.py \
  tests/test_signal_metrics.py \
  tests/test_eval_metrics.py \
  tests/test_losses.py \
  tests/test_engine_smoke.py \
  tests/test_tho_experiment.py \
  -v
```

预期：PASS。

- [ ] **步骤 2：运行独立性审计**

运行：

```bash
./.venv/bin/python scripts/audit_split_independence.py \
  --config configs/tho_research_v2.yaml \
  --output-dir runs/audits/split_independence_research_v2
```

预期：退出码 0，输出形如：

```text
split_independence train_windows=1024 val_windows=256 overlap_samp_id_count=数字 overlap_segment_count=数字
```

验收：

- 如果 `overlap_samp_id_count` 或 `overlap_segment_count` 大于 0，本轮实验结论标记为 within-subject / window-level 开发指标。
- 如果 `overlap_samp_id_count` 和 `overlap_segment_count` 都等于 0，仍需查看 `per_samp_id.csv`，确认验证集没有被少数 `samp_id` 支配。

- [ ] **步骤 3：GPU 执行 L0 pilot**

运行：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=4096 \
  --set data.max_val_windows=1024 \
  --set training.epochs=20 \
  --set training.patience=5 \
  --set training.device=cuda:0 \
  --set outputs.run_root=runs/tho_research_v2_l0_lagaware
```

预期：退出码 0，run 目录中存在 `config.yaml`、`train_history.csv`、`metrics.csv`、`predictions.npz`。

- [ ] **步骤 4：GPU 执行 L1 pilot**

运行：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=4096 \
  --set data.max_val_windows=1024 \
  --set training.epochs=20 \
  --set training.patience=5 \
  --set training.device=cuda:0 \
  --set loss.band_waveform_weight=0.1 \
  --set outputs.run_root=runs/tho_research_v2_l1_bandwave_010
```

预期：退出码 0，run 目录中存在 `config.yaml`、`train_history.csv`、`metrics.csv`、`predictions.npz`。

- [ ] **步骤 5：汇总 L0/L1 结果**

运行：

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_l0_lagaware \
  --output runs/l0_lagaware_summary.csv

./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_l1_bandwave_010 \
  --output runs/l1_bandwave_010_summary.csv
```

预期：退出码 0，`runs/l0_lagaware_summary.csv` 和 `runs/l1_bandwave_010_summary.csv` 存在。

- [ ] **步骤 6：绘制诊断图**

```bash
./.venv/bin/python scripts/plot_tho_predictions.py \
  --run-dir "$(find runs/tho_research_v2_l1_bandwave_010 -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)" \
  --sort-by best_lag_corr \
  --limit 32
```

预期：退出码 0，run 目录下生成诊断图。

- [ ] **步骤 7：更新实验记录**

根据 `runs/audits/split_independence_research_v2/summary.csv`、`runs/l0_lagaware_summary.csv`、`runs/l1_bandwave_010_summary.csv` 和 L0/L1 的 `metrics.csv` 更新 `docs/experiments/l1_bandlimited_lag_aware.md` 表格。

记录必须包含：

- split 独立性审计结果。
- L0/L1 的 run 路径。
- `band_limited_corr`、`best_lag_corr`、`best_lag_sec`、`rr_peak_abs_error`、`spectrum_similarity`。
- 是否允许进入 L2 或 STFT loss 的判断。

- [ ] **步骤 8：Commit**

运行：

```bash
git add docs/experiments/l1_bandlimited_lag_aware.md runs/l0_lagaware_summary.csv runs/l1_bandwave_010_summary.csv
git commit -m "docs: 记录低频波形损失实验结果"
```

---

## 自检清单

- [ ] 计划覆盖了数据独立性审计、lag-aware evaluation、band-limited waveform loss 和 L0/L1 pilot。
- [ ] 计划没有包含 M1-M5 模型结构实现。
- [ ] 所有新增功能都有失败测试、实现、通过测试和 commit 步骤。
- [ ] 默认配置保持 L0 行为不变：`band_waveform_weight: 0.0`。
- [ ] 训练命令显式使用 GPU；若 GPU 不可见，应报告环境阻塞，不降级为 CPU 正式训练。
