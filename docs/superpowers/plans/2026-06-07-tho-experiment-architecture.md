# THO 实验工程骨架重建实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将首版 THO 小规模训练脚本重建为「数据工厂 + 公共实验基类 + THO 任务实验类」的实验工程骨架，并默认使用分层随机抽样。

**架构：** 数据层集中处理索引审计、抽样、Dataset 和 DataLoader；公共 `BaseExperiment` 承担 run 生命周期、训练循环、best checkpoint 和 early stopping；`ThoExperiment` 只保留 THO 任务语义，包括 baseline、评价指标和诊断预测。脚本保持原路径，但瘦身为 CLI 入口。

**技术栈：** Python、PyTorch、NumPy、Pandas、SciPy、OmegaConf、tqdm、pytest。

---

## 文件结构

创建文件：

- `resp_train/data/factory.py`：统一构建窗口数据、抽样后的索引行、Dataset、DataLoader 和审计摘要。
- `resp_train/experiments/__init__.py`：导出实验层公开类。
- `resp_train/experiments/base.py`：公共实验基类、早停状态和实验数据容器。
- `resp_train/experiments/tho.py`：THO 任务实验类和 checkpoint 评价入口。
- `tests/test_data_factory.py`：数据工厂行为测试。
- `tests/test_base_experiment.py`：公共实验基类行为测试。
- `tests/test_tho_experiment.py`：THO 实验 smoke 测试。

修改文件：

- `configs/tho_small.yaml`：改为默认分层随机抽样，新增训练策略配置。
- `resp_train/config.py`：补充新配置字段校验。
- `resp_train/data/index.py`：加入 `random`、`stratified_random` 和 debug `head` 抽样策略。
- `resp_train/data/__init__.py`：导出数据工厂类型和函数。
- `resp_train/engine/train.py`：加入 AMP、梯度裁剪和更通用的 prediction collection 键名支持。
- `resp_train/engine/__init__.py`：导出新增训练工具。
- `resp_train/metrics/baseline.py`：抽出可复用的 baseline dataset 评价函数。
- `scripts/train_tho_small.py`：瘦身为调用 `ThoExperiment.train()` 的 CLI。
- `scripts/eval_tho_small.py`：瘦身为调用 THO checkpoint 评价入口的 CLI。
- `scripts/audit_tho_dataset.py`：复用数据工厂口径。
- `scripts/baseline_tho_hilbert.py`：复用数据工厂口径。
- `docs/tho_small_training.md`：更新新骨架、抽样策略、训练策略和 smoke 命令。
- `scripts/README.md`：更新脚本职责。
- `docs/experiments/tho_small_mixed_zscore_20260607.md`：标注既有 run 为前缀采样下的首版 smoke/现象记录。

不移动脚本目录，不引入新第三方依赖。

---

### 任务 1：配置改为新实验默认值

**文件：**
- 修改：`configs/tho_small.yaml`
- 修改：`resp_train/config.py`
- 修改：`tests/test_config.py`

- [x] **步骤 1：更新配置测试**

在 `tests/test_config.py` 中调整默认配置断言，确保默认抽样不再是 `head`，并覆盖新增训练策略字段。

```python
def test_load_default_config_has_expected_values():
    cfg = load_config("configs/tho_small.yaml")

    assert Path(cfg.data.dataset_root).name == "20260530_tho_ramp5_stage2_1"
    assert cfg.data.input_set == "mixed_zscore"
    assert cfg.data.max_train_windows == 1024
    assert cfg.data.max_val_windows == 256
    assert cfg.data.train_sample_strategy == "stratified_random"
    assert cfg.data.val_sample_strategy == "stratified_random"
    assert cfg.data.train_sample_seed == 20260601
    assert cfg.data.val_sample_seed == 20260602
    assert cfg.data.stratify_column == "residual_quality_class"
    assert bool(cfg.data.filter_unusable) is True
    assert cfg.training.epochs == 3
    assert cfg.training.patience == 5
    assert cfg.training.min_delta == 0.0
    assert cfg.training.lr_scheduler == "none"
    assert cfg.training.grad_clip_norm is None
    assert bool(cfg.training.use_amp) is False
    assert cfg.model.name == "unet1d_tiny"
    assert cfg.outputs.max_prediction_windows == 32
```

新增覆盖 dotlist 的断言：

```python
def test_load_config_applies_sampling_overrides():
    cfg = load_config(
        "configs/tho_small.yaml",
        overrides=[
            "data.train_sample_strategy=random",
            "data.val_sample_strategy=head",
            "data.train_sample_seed=7",
            "data.val_sample_seed=8",
            "training.patience=2",
        ],
    )

    assert cfg.data.train_sample_strategy == "random"
    assert cfg.data.val_sample_strategy == "head"
    assert cfg.data.train_sample_seed == 7
    assert cfg.data.val_sample_seed == 8
    assert cfg.training.patience == 2
```

- [x] **步骤 2：运行测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_config.py -v
```

预期：FAIL，失败点包含 `train_sample_strategy` 或 `patience` 字段不存在。

- [x] **步骤 3：更新 `configs/tho_small.yaml`**

将 `data` 和 `training` 段改为：

```yaml
data:
  dataset_root: /mnt/disk_code/marques/dataset/RespPairs/20260530_tho_ramp5_stage2_1
  index_csv: training/dataset_index.csv
  input_set: mixed_zscore
  train_split: train
  val_split: val
  max_train_windows: 1024
  max_val_windows: 256
  filter_unusable: true
  valid_ratio_min: 0.99
  input_finite_ratio_min: 0.99
  target_finite_ratio_min: 0.99
  unusable_residual_classes: []
  preload_windows: true
  train_sample_strategy: stratified_random
  val_sample_strategy: stratified_random
  train_sample_seed: 20260601
  val_sample_seed: 20260602
  stratify_column: residual_quality_class

training:
  epochs: 3
  batch_size: 8
  learning_rate: 0.001
  num_workers: 0
  seed: 20260601
  device: auto
  patience: 5
  min_delta: 0.0
  lr_scheduler: none
  grad_clip_norm: null
  use_amp: false
```

- [x] **步骤 4：增强配置校验**

在 `resp_train/config.py` 的 `_validate_config` 中追加必需字段，并验证枚举值。

```python
required = [
    "data.dataset_root",
    "data.index_csv",
    "data.input_set",
    "data.train_sample_strategy",
    "data.val_sample_strategy",
    "data.train_sample_seed",
    "data.val_sample_seed",
    "data.stratify_column",
    "window.duration_samples",
    "model.name",
    "training.epochs",
    "training.patience",
    "training.min_delta",
    "training.lr_scheduler",
    "training.use_amp",
    "outputs.run_root",
]
```

在字段检查后加入：

```python
sample_strategies = {"head", "random", "stratified_random"}
for key in ("data.train_sample_strategy", "data.val_sample_strategy"):
    value = OmegaConf.select(cfg, key)
    if value not in sample_strategies:
        raise ValueError(f"{key} 必须是 {sorted(sample_strategies)} 之一，当前为: {value}")

lr_schedulers = {"none", "cosine"}
lr_scheduler = OmegaConf.select(cfg, "training.lr_scheduler")
if lr_scheduler not in lr_schedulers:
    raise ValueError(f"training.lr_scheduler 必须是 {sorted(lr_schedulers)} 之一，当前为: {lr_scheduler}")
```

- [x] **步骤 5：运行配置测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_config.py -v
```

预期：PASS。

- [x] **步骤 6：Commit**

```bash
git add configs/tho_small.yaml resp_train/config.py tests/test_config.py
git commit -m "feat(配置): 默认使用分层随机抽样"
```

---

### 任务 2：实现窗口抽样策略

**文件：**
- 修改：`resp_train/data/index.py`
- 修改：`tests/test_data_index_audit.py`

- [x] **步骤 1：补充抽样测试数据**

在 `tests/test_data_index_audit.py` 中新增一个多行样例函数，用于测试抽样策略。

```python
def _sampling_rows():
    records = []
    qualities = [
        "near_zero_residual",
        "near_zero_residual",
        "near_zero_residual",
        "near_zero_residual",
        "stable_nonzero_residual",
        "stable_nonzero_residual",
        "stable_nonzero_residual",
        "stable_nonzero_residual",
        "high_residual",
        "high_residual",
    ]
    for idx, quality in enumerate(qualities):
        records.append(
            {
                "dataset_row_id": idx + 1,
                "input_set": "mixed_zscore",
                "split": "train",
                "usable": True,
                "residual_quality_class": quality,
                "samp_id": 100 + idx,
                "segment_id": 1,
                "window_id_in_segment": 1,
            }
        )
    return pd.DataFrame.from_records(records)
```

- [x] **步骤 2：编写抽样行为测试**

在同一文件中新增测试：

```python
def test_filter_index_defaults_to_stratified_random():
    cfg = OmegaConf.create(
        {
            "data": {
                "input_set": "mixed_zscore",
                "filter_unusable": True,
                "train_sample_strategy": "stratified_random",
                "train_sample_seed": 42,
                "stratify_column": "residual_quality_class",
            }
        }
    )

    filtered = filter_index(_sampling_rows(), cfg, split="train", max_windows=5)

    assert len(filtered) == 5
    assert set(filtered["residual_quality_class"]) == {
        "near_zero_residual",
        "stable_nonzero_residual",
        "high_residual",
    }
    assert filtered["dataset_row_id"].tolist() == sorted(filtered["dataset_row_id"].tolist())


def test_filter_index_random_is_reproducible():
    cfg = OmegaConf.create(
        {
            "data": {
                "input_set": "mixed_zscore",
                "filter_unusable": True,
                "train_sample_strategy": "random",
                "train_sample_seed": 123,
                "stratify_column": "residual_quality_class",
            }
        }
    )

    first = filter_index(_sampling_rows(), cfg, split="train", max_windows=4)
    second = filter_index(_sampling_rows(), cfg, split="train", max_windows=4)

    assert first["dataset_row_id"].tolist() == second["dataset_row_id"].tolist()
    assert first["dataset_row_id"].tolist() != [1, 2, 3, 4]


def test_filter_index_head_is_debug_prefix_only_when_explicit():
    cfg = OmegaConf.create(
        {
            "data": {
                "input_set": "mixed_zscore",
                "filter_unusable": True,
                "train_sample_strategy": "head",
                "train_sample_seed": 123,
                "stratify_column": "residual_quality_class",
            }
        }
    )

    filtered = filter_index(_sampling_rows(), cfg, split="train", max_windows=4)

    assert filtered["dataset_row_id"].tolist() == [1, 2, 3, 4]


def test_filter_index_uses_independent_val_strategy_and_seed():
    cfg = OmegaConf.create(
        {
            "data": {
                "input_set": "mixed_zscore",
                "filter_unusable": True,
                "train_sample_strategy": "random",
                "val_sample_strategy": "random",
                "train_sample_seed": 1,
                "val_sample_seed": 2,
                "stratify_column": "residual_quality_class",
            }
        }
    )

    train_rows = filter_index(_sampling_rows(), cfg, split="train", max_windows=4)
    val_rows = filter_index(_sampling_rows().assign(split="val"), cfg, split="val", max_windows=4)

    assert train_rows["dataset_row_id"].tolist() != val_rows["dataset_row_id"].tolist()


def test_filter_index_rejects_missing_stratify_column():
    cfg = OmegaConf.create(
        {
            "data": {
                "input_set": "mixed_zscore",
                "filter_unusable": True,
                "train_sample_strategy": "stratified_random",
                "train_sample_seed": 42,
                "stratify_column": "missing_column",
            }
        }
    )

    with pytest.raises(ValueError, match="missing_column"):
        filter_index(_sampling_rows(), cfg, split="train", max_windows=4)
```

- [x] **步骤 3：运行测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_data_index_audit.py -v
```

预期：FAIL，失败点包含默认仍按 `head` 前缀返回或缺少分层列校验。

- [x] **步骤 4：实现抽样策略**

在 `resp_train/data/index.py` 中修改 `filter_index` 签名：

```python
def filter_index(
    df: pd.DataFrame,
    cfg: DictConfig,
    *,
    split: str,
    max_windows: int | None,
    sample_strategy: str | None = None,
    sample_seed: int | None = None,
) -> pd.DataFrame:
```

在文件内新增辅助函数：

```python
def _resolve_sample_strategy(cfg: DictConfig, split: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    if split == str(cfg.data.get("train_split", "train")):
        return str(cfg.data.get("train_sample_strategy", "stratified_random"))
    if split == str(cfg.data.get("val_split", "val")):
        return str(cfg.data.get("val_sample_strategy", "stratified_random"))
    return "stratified_random"


def _resolve_sample_seed(cfg: DictConfig, split: str, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    if split == str(cfg.data.get("train_split", "train")):
        return int(cfg.data.get("train_sample_seed", cfg.training.get("seed", 0)))
    if split == str(cfg.data.get("val_split", "val")):
        return int(cfg.data.get("val_sample_seed", cfg.training.get("seed", 0) + 1))
    return int(cfg.data.get("train_sample_seed", 0))
```

实现 `_sample_rows`、`_random_sample`、`_stratified_random_sample`：

```python
def _sample_rows(
    filtered: pd.DataFrame,
    cfg: DictConfig,
    *,
    split: str,
    max_windows: int | None,
    sample_strategy: str | None,
    sample_seed: int | None,
) -> pd.DataFrame:
    ordered = filtered.sort_values("dataset_row_id").reset_index(drop=True)
    if max_windows is None or len(ordered) <= int(max_windows):
        return ordered

    limit = int(max_windows)
    strategy = _resolve_sample_strategy(cfg, split, sample_strategy)
    seed = _resolve_sample_seed(cfg, split, sample_seed)
    if strategy == "head":
        sampled = ordered.head(limit)
    elif strategy == "random":
        sampled = ordered.sample(n=limit, random_state=seed, replace=False)
    elif strategy == "stratified_random":
        sampled = _stratified_random_sample(ordered, cfg, n=limit, seed=seed)
    else:
        raise ValueError(f"未知抽样策略: {strategy}")
    return sampled.sort_values("dataset_row_id").reset_index(drop=True)
```

`_stratified_random_sample` 规则：

```python
def _stratified_random_sample(df: pd.DataFrame, cfg: DictConfig, *, n: int, seed: int) -> pd.DataFrame:
    column = str(cfg.data.get("stratify_column", "residual_quality_class"))
    if column not in df.columns:
        raise ValueError(f"分层抽样列不存在: {column}")

    counts = df[column].value_counts(dropna=False).sort_index()
    raw = counts / counts.sum() * n
    quotas = raw.astype(int)
    remainder = raw - quotas
    remaining = n - int(quotas.sum())
    for key in remainder.sort_values(ascending=False).index:
        if remaining <= 0:
            break
        if quotas.loc[key] < counts.loc[key]:
            quotas.loc[key] += 1
            remaining -= 1

    while int(quotas.sum()) < n:
        available = counts[counts > quotas]
        if available.empty:
            break
        key = (available - quotas[available.index]).sort_values(ascending=False).index[0]
        quotas.loc[key] += 1

    parts = []
    for offset, (key, quota) in enumerate(quotas.items()):
        if int(quota) <= 0:
            continue
        group = df[df[column] == key]
        parts.append(group.sample(n=int(quota), random_state=seed + offset, replace=False))
    if not parts:
        return df.head(0).copy()
    return pd.concat(parts, ignore_index=True)
```

更新 `filter_index` 主体：

```python
filtered = df[(df["input_set"] == cfg.data.input_set) & (df["split"] == split)].copy()
if bool(cfg.data.get("filter_unusable", True)) and "usable" in filtered.columns:
    filtered = filtered[filtered["usable"]].copy()
return _sample_rows(
    filtered,
    cfg,
    split=split,
    max_windows=max_windows,
    sample_strategy=sample_strategy,
    sample_seed=sample_seed,
)
```

- [x] **步骤 5：运行抽样测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_data_index_audit.py -v
```

预期：PASS。

- [x] **步骤 6：Commit**

```bash
git add resp_train/data/index.py tests/test_data_index_audit.py
git commit -m "feat(数据): 添加可复现分层随机抽样"
```

---

### 任务 3：新增数据工厂

**文件：**
- 创建：`resp_train/data/factory.py`
- 修改：`resp_train/data/__init__.py`
- 创建：`tests/test_data_factory.py`

- [x] **步骤 1：编写数据工厂测试**

创建 `tests/test_data_factory.py`。测试文件中提供临时数据集构造函数：

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from resp_train.data.factory import build_tho_data, build_window_data


def _write_npz(root: Path, samp_id: int) -> str:
    npz_dir = root / "whole_night" / "mixed_bcg_to_tho" / str(samp_id)
    npz_dir.mkdir(parents=True, exist_ok=True)
    path = npz_dir / "sample.npz"
    signal = np.linspace(0, 1, 64, dtype=np.float32)
    np.savez(path, bcg=signal, tho=signal * 2)
    return f"../whole_night/mixed_bcg_to_tho/{samp_id}/sample.npz"


def _index_rows(root: Path) -> pd.DataFrame:
    records = []
    row_id = 1
    for split in ("train", "val"):
        for idx, quality in enumerate(["near_zero_residual", "stable_nonzero_residual", "near_zero_residual", "high_residual"]):
            records.append(
                {
                    "dataset_row_id": row_id,
                    "input_set": "mixed_zscore",
                    "split": split,
                    "samp_id": 100 + idx,
                    "segment_id": 1,
                    "window_id_in_segment": idx + 1,
                    "source_npz": _write_npz(root, 100 + idx),
                    "bcg_signal_key": "bcg",
                    "target_signal_key": "tho",
                    "valid_sec_key": "valid",
                    "segment_decision": "include_candidate",
                    "window_start_sample": 0,
                    "window_end_sample": 32,
                    "window_duration_samples": 32,
                    "target_fs": 100,
                    "valid_ratio": 1.0,
                    "input_finite_ratio": 1.0,
                    "target_finite_ratio": 1.0,
                    "residual_quality_class": quality,
                    "base_alignment_method": "keep_original",
                    "apply_decision": "approved",
                    "reason": "ok",
                }
            )
            row_id += 1
    return pd.DataFrame.from_records(records)


def _cfg(root: Path):
    return OmegaConf.create(
        {
            "data": {
                "dataset_root": str(root),
                "index_csv": "training/dataset_index.csv",
                "input_set": "mixed_zscore",
                "train_split": "train",
                "val_split": "val",
                "max_train_windows": 3,
                "max_val_windows": 2,
                "filter_unusable": True,
                "valid_ratio_min": 0.99,
                "input_finite_ratio_min": 0.99,
                "target_finite_ratio_min": 0.99,
                "unusable_residual_classes": [],
                "preload_windows": True,
                "train_sample_strategy": "stratified_random",
                "val_sample_strategy": "head",
                "train_sample_seed": 1,
                "val_sample_seed": 2,
                "stratify_column": "residual_quality_class",
            },
            "window": {"duration_samples": 32, "target_fs": 100},
            "training": {"batch_size": 2, "num_workers": 0},
        }
    )


def _prepare_dataset(tmp_path: Path):
    root = tmp_path / "dataset"
    training_dir = root / "training"
    training_dir.mkdir(parents=True)
    _index_rows(root).to_csv(training_dir / "dataset_index.csv", index=False)
    return root
```

新增测试：

```python
def test_build_window_data_returns_rows_dataset_loader_and_audit(tmp_path: Path):
    root = _prepare_dataset(tmp_path)
    cfg = _cfg(root)

    bundle = build_window_data(
        cfg,
        split="train",
        max_windows=cfg.data.max_train_windows,
        sample_strategy=cfg.data.train_sample_strategy,
        sample_seed=cfg.data.train_sample_seed,
        shuffle=False,
    )

    assert len(bundle.rows) == 3
    assert len(bundle.dataset) == 3
    assert bundle.audit_summary["n_windows"].sum() == 8
    batch = next(iter(bundle.loader))
    assert batch["x"].shape == (2, 1, 32)
    assert batch["target"].shape == (2, 1, 32)


def test_build_tho_data_uses_independent_train_and_val_sampling(tmp_path: Path):
    root = _prepare_dataset(tmp_path)
    cfg = _cfg(root)

    data = build_tho_data(cfg)

    assert len(data.train.rows) == 3
    assert len(data.val.rows) == 2
    assert data.train.rows["split"].unique().tolist() == ["train"]
    assert data.val.rows["split"].unique().tolist() == ["val"]
    assert data.val.rows["dataset_row_id"].tolist() == [5, 6]


def test_build_tho_data_rejects_empty_train_split(tmp_path: Path):
    root = _prepare_dataset(tmp_path)
    cfg = _cfg(root)
    cfg.data.train_split = "missing"

    with pytest.raises(RuntimeError, match="train.*为空"):
        build_tho_data(cfg)
```

- [x] **步骤 2：运行测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_data_factory.py -v
```

预期：FAIL，报错包含 `No module named 'resp_train.data.factory'`。

- [x] **步骤 3：实现 `resp_train/data/factory.py`**

创建数据容器：

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from resp_train.data.audit import add_usable_flag, summarize_audit
from resp_train.data.dataset import RespWindowDataset
from resp_train.data.index import filter_index, read_index


@dataclass(frozen=True)
class WindowDataBundle:
    index_path: Path
    rows: pd.DataFrame
    dataset: RespWindowDataset
    loader: DataLoader
    audited: pd.DataFrame
    audit_summary: pd.DataFrame


@dataclass(frozen=True)
class ThoDataBundle:
    train: WindowDataBundle
    val: WindowDataBundle
    audited: pd.DataFrame
    audit_summary: pd.DataFrame
```

实现公共构建函数：

```python
def build_window_data(
    cfg: DictConfig,
    *,
    split: str,
    max_windows: int | None,
    sample_strategy: str | None = None,
    sample_seed: int | None = None,
    shuffle: bool,
    audited: pd.DataFrame | None = None,
) -> WindowDataBundle:
    dataset_root = Path(str(cfg.data.dataset_root))
    index_csv = Path(str(cfg.data.index_csv))
    index_path = dataset_root / index_csv
    audited_frame = add_usable_flag(read_index(dataset_root, index_csv), cfg) if audited is None else audited.copy()
    audit_summary = summarize_audit(audited_frame)
    rows = filter_index(
        audited_frame,
        cfg,
        split=split,
        max_windows=max_windows,
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
    )
    dataset = RespWindowDataset(
        index_path,
        rows,
        cfg,
        preload_windows=bool(cfg.data.get("preload_windows", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=bool(shuffle),
        num_workers=int(cfg.training.num_workers),
    )
    return WindowDataBundle(
        index_path=index_path,
        rows=rows,
        dataset=dataset,
        loader=loader,
        audited=audited_frame,
        audit_summary=audit_summary,
    )
```

实现 THO train/val 工厂：

```python
def build_tho_data(cfg: DictConfig) -> ThoDataBundle:
    dataset_root = Path(str(cfg.data.dataset_root))
    index_csv = Path(str(cfg.data.index_csv))
    audited = add_usable_flag(read_index(dataset_root, index_csv), cfg)
    audit_summary = summarize_audit(audited)
    train = build_window_data(
        cfg,
        split=str(cfg.data.train_split),
        max_windows=cfg.data.get("max_train_windows"),
        sample_strategy=str(cfg.data.train_sample_strategy),
        sample_seed=int(cfg.data.train_sample_seed),
        shuffle=True,
        audited=audited,
    )
    val = build_window_data(
        cfg,
        split=str(cfg.data.val_split),
        max_windows=cfg.data.get("max_val_windows"),
        sample_strategy=str(cfg.data.val_sample_strategy),
        sample_seed=int(cfg.data.val_sample_seed),
        shuffle=False,
        audited=audited,
    )
    if len(train.dataset) == 0:
        raise RuntimeError("train 数据为空，请检查 input_set、train_split、可用性过滤和抽样配置。")
    if len(val.dataset) == 0:
        raise RuntimeError("val 数据为空，请检查 input_set、val_split、可用性过滤和抽样配置。")
    return ThoDataBundle(train=train, val=val, audited=audited, audit_summary=audit_summary)
```

- [x] **步骤 4：导出数据工厂**

更新 `resp_train/data/__init__.py`：

```python
from resp_train.data.factory import ThoDataBundle, WindowDataBundle, build_tho_data, build_window_data

__all__ = ["ThoDataBundle", "WindowDataBundle", "build_tho_data", "build_window_data"]
```

- [x] **步骤 5：运行数据工厂测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_data_factory.py -v
```

预期：PASS。

- [x] **步骤 6：Commit**

```bash
git add resp_train/data/factory.py resp_train/data/__init__.py tests/test_data_factory.py
git commit -m "feat(数据): 添加统一窗口数据工厂"
```

---

### 任务 4：增强训练引擎能力

**文件：**
- 修改：`resp_train/engine/train.py`
- 修改：`resp_train/engine/__init__.py`
- 修改：`tests/test_engine_smoke.py`

- [x] **步骤 1：编写引擎增强测试**

在 `tests/test_engine_smoke.py` 中新增梯度裁剪测试：

```python
def test_train_one_epoch_accepts_grad_clip_norm():
    cfg = _cfg()
    loader = DataLoader(DictDataset(), batch_size=2, shuffle=False)
    model = build_model(cfg)
    loss_fn = WeakSyncLoss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    summary = train_one_epoch(
        model,
        loader,
        loss_fn,
        optimizer,
        torch.device("cpu"),
        grad_clip_norm=0.1,
        use_amp=False,
    )

    assert summary["loss"] > 0
```

新增 prediction key 测试：

```python
def test_collect_predictions_accepts_custom_output_keys():
    loader = DataLoader(DictDataset(), batch_size=2, shuffle=False)
    model = torch.nn.Identity()

    preds = collect_predictions(
        model,
        loader,
        device=torch.device("cpu"),
        max_windows=3,
        pred_key="custom_pred",
        target_key="custom_target",
    )

    assert preds["custom_pred"].shape[0] == 3
    assert preds["custom_target"].shape[0] == 3
    assert preds["dataset_row_id"].tolist() == [0, 1, 2]
```

- [x] **步骤 2：运行引擎测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_engine_smoke.py -v
```

预期：FAIL，失败点包含 `grad_clip_norm` 或 `pred_key` 参数不存在。

- [x] **步骤 3：更新 `train_one_epoch` 签名和实现**

将 `train_one_epoch` 签名扩展为：

```python
def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
```

在训练循环中使用 AMP 和梯度裁剪：

```python
amp_enabled = bool(use_amp and resolved_device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
...
with torch.cuda.amp.autocast(enabled=amp_enabled):
    pred = model(sensor)
    loss, parts = loss_fn(pred, target)
if amp_enabled:
    scaler.scale(loss).backward()
    if grad_clip_norm is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
    optimizer.step()
```

保持 CPU 默认路径不变。

- [x] **步骤 4：更新 `collect_predictions` 键名参数**

将签名扩展为：

```python
def collect_predictions(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    max_windows: int,
    pred_key: str = "r_tho_hat",
    target_key: str = "tho_ref",
) -> dict[str, np.ndarray]:
```

返回字典改为：

```python
return {
    pred_key: pred_arr,
    target_key: target_arr,
    "dataset_row_id": np.asarray([...], dtype=np.int64),
    "split": np.asarray([...]),
    "input_set": np.asarray([...]),
    "residual_quality_class": np.asarray([...]),
}
```

- [x] **步骤 5：确认导出**

检查 `resp_train/engine/__init__.py`，确保导出 `collect_predictions`、`save_checkpoint`、`train_one_epoch`、`validate`。

- [x] **步骤 6：运行引擎测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_engine_smoke.py -v
```

预期：PASS。

- [x] **步骤 7：Commit**

```bash
git add resp_train/engine/train.py resp_train/engine/__init__.py tests/test_engine_smoke.py
git commit -m "feat(训练): 支持梯度裁剪和预测键名配置"
```

---

### 任务 5：新增公共实验基类

**文件：**
- 创建：`resp_train/experiments/__init__.py`
- 创建：`resp_train/experiments/base.py`
- 修改：`resp_train/engine/__init__.py`
- 创建：`tests/test_base_experiment.py`

- [x] **步骤 1：编写公共实验基类测试**

创建 `tests/test_base_experiment.py`：

```python
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from resp_train.experiments.base import BaseExperiment, ExperimentData


class TinyDataset(Dataset):
    def __init__(self, length: int = 4):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.tensor([[float(idx), float(idx + 1)]])
        target = x * 0.0
        return {"x": x.float(), "target": target.float(), "meta": {"dataset_row_id": idx}}


class ConstantLoss(torch.nn.Module):
    def forward(self, pred, target):
        loss = pred.sum() * 0.0 + 1.0
        return loss, {"constant": loss.detach()}


class ToyExperiment(BaseExperiment):
    task_name = "toy"

    def build_data(self):
        loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)
        return ExperimentData(
            train_loader=loader,
            val_loader=loader,
            audit_frame=pd.DataFrame({"split": ["train", "val"]}),
            audit_summary=pd.DataFrame({"n_windows": [4]}),
            extras={},
        )

    def build_model(self):
        return torch.nn.Conv1d(1, 1, kernel_size=1)

    def build_loss(self):
        return ConstantLoss()

    def run_baseline(self, data, run_dir):
        pd.DataFrame({"baseline": [1.0]}).to_csv(run_dir / "baseline_metrics.csv", index=False)

    def evaluate_best(self, model, data, run_dir):
        pd.DataFrame({"metric": [1.0]}).to_csv(run_dir / "metrics.csv", index=False)
        torch.save({"ok": True}, run_dir / "predictions_marker.pt")


def _cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "outputs": {"run_root": str(tmp_path / "runs")},
            "training": {
                "seed": 1,
                "device": "cpu",
                "epochs": 2,
                "learning_rate": 0.01,
                "patience": 1,
                "min_delta": 0.0,
                "lr_scheduler": "none",
                "grad_clip_norm": None,
                "use_amp": False,
            },
        }
    )


def test_base_experiment_runs_lifecycle_and_writes_outputs(tmp_path: Path):
    run_dir = ToyExperiment(_cfg(tmp_path)).train()

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "train_history.csv").exists()
    assert (run_dir / "baseline_metrics.csv").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "predictions_marker.pt").exists()


def test_base_experiment_early_stopping_records_reason(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.training.epochs = 5
    cfg.training.patience = 1

    run_dir = ToyExperiment(cfg).train()
    history = pd.read_csv(run_dir / "train_history.csv")

    assert history["epoch"].max() < 5
    assert (run_dir / "train.log").read_text(encoding="utf-8").find("early_stop") >= 0
```

- [x] **步骤 2：运行测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_base_experiment.py -v
```

预期：FAIL，报错包含 `No module named 'resp_train.experiments'`。

- [x] **步骤 3：实现 `resp_train/experiments/base.py`**

创建数据容器和基类：

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig

from resp_train.engine import save_checkpoint, train_one_epoch, validate
from resp_train.utils.run import create_run_dir, resolve_device, save_config, set_seed, setup_logger


@dataclass(frozen=True)
class ExperimentData:
    train_loader: Any
    val_loader: Any
    audit_frame: pd.DataFrame
    audit_summary: pd.DataFrame
    extras: dict[str, Any]
```

实现 `BaseExperiment` 生命周期：

```python
class BaseExperiment:
    task_name = "base"

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.run_dir: Path | None = None
        self.device: torch.device | None = None

    def train(self) -> Path:
        run_dir = create_run_dir(self.cfg.outputs.run_root)
        self.run_dir = run_dir
        save_config(self.cfg, run_dir)
        logger = setup_logger(run_dir)
        set_seed(int(self.cfg.training.seed))
        device = resolve_device(str(self.cfg.training.device))
        self.device = device
        logger.info("task=%s device=%s", self.task_name, device)

        data = self.build_data()
        data.audit_summary.to_csv(run_dir / "audit.csv", index=False)
        self.run_baseline(data, run_dir)

        model = self.build_model().to(device)
        loss_fn = self.build_loss()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)

        history_records = []
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        patience = int(self.cfg.training.get("patience", 0))
        min_delta = float(self.cfg.training.get("min_delta", 0.0))

        for epoch in range(1, int(self.cfg.training.epochs) + 1):
            train_metrics = train_one_epoch(
                model,
                data.train_loader,
                loss_fn,
                optimizer,
                device=device,
                grad_clip_norm=self.cfg.training.get("grad_clip_norm"),
                use_amp=bool(self.cfg.training.get("use_amp", False)),
            )
            val_metrics = validate(model, data.val_loader, loss_fn, device=device)
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                **{f"train_{k}": v for k, v in train_metrics.items() if k != "loss"},
                **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"},
            }
            history_records.append(record)
            logger.info("epoch=%s train_loss=%.6f val_loss=%.6f", epoch, record["train_loss"], record["val_loss"])

            improved = record["val_loss"] < (best_loss - min_delta)
            if improved:
                best_loss = record["val_loss"]
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint(run_dir / "checkpoint.pt", model=model, optimizer=optimizer, epoch=epoch, metrics=record, cfg=self.cfg)
            else:
                stale_epochs += 1
                if patience > 0 and stale_epochs >= patience:
                    logger.info("early_stop epoch=%s best_epoch=%s best_val_loss=%.6f", epoch, best_epoch, best_loss)
                    break
            if scheduler is not None:
                scheduler.step()

        pd.DataFrame(history_records).to_csv(run_dir / "train_history.csv", index=False)
        checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.evaluate_best(model, data, run_dir)
        return run_dir
```

实现默认钩子：

```python
    def build_data(self) -> ExperimentData:
        raise NotImplementedError

    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def build_loss(self) -> torch.nn.Module:
        raise NotImplementedError

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=float(self.cfg.training.learning_rate))

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        name = str(self.cfg.training.get("lr_scheduler", "none"))
        if name == "none":
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.cfg.training.epochs))
        raise ValueError(f"未知 lr_scheduler: {name}")

    def run_baseline(self, data: ExperimentData, run_dir: Path) -> None:
        return None

    def evaluate_best(self, model: torch.nn.Module, data: ExperimentData, run_dir: Path) -> None:
        raise NotImplementedError
```

- [x] **步骤 4：导出实验层**

创建 `resp_train/experiments/__init__.py`：

```python
from resp_train.experiments.base import BaseExperiment, ExperimentData

__all__ = ["BaseExperiment", "ExperimentData"]
```

同时确认 `resp_train/engine/__init__.py` 公开导出 `save_checkpoint`，供公共实验基类复用训练引擎入口。

- [x] **步骤 5：运行公共实验基类测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_base_experiment.py -v
```

预期：PASS。

- [x] **步骤 6：Commit**

```bash
git add resp_train/experiments/__init__.py resp_train/experiments/base.py tests/test_base_experiment.py
git commit -m "feat(实验): 添加公共实验基类"
```

---

### 任务 6：让 baseline 可复用数据工厂

**文件：**
- 修改：`resp_train/metrics/baseline.py`
- 修改：`tests/test_eval_metrics.py` 或新增 `tests/test_baseline_metrics.py`

- [x] **步骤 1：新增 baseline dataset 评价测试**

新增 `tests/test_baseline_metrics.py`：

```python
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from resp_train.metrics.baseline import evaluate_baseline_dataset


class TinyRespDataset:
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        t = np.linspace(0, 2 * np.pi, 512, dtype=np.float32)
        x = np.sin(t + idx).astype(np.float32)
        y = np.sin(t + idx * 0.1).astype(np.float32)
        return {
            "x": __import__("torch").from_numpy(x).view(1, -1),
            "target": __import__("torch").from_numpy(y).view(1, -1),
            "meta": {
                "dataset_row_id": idx,
                "split": "val",
                "input_set": "mixed_zscore",
                "samp_id": 1,
                "segment_id": 1,
                "window_id_in_segment": idx + 1,
                "residual_quality_class": "near_zero_residual",
            },
        }


def test_evaluate_baseline_dataset_returns_metrics_frame():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100},
            "loss": {"spectrum_low_hz": 0.05, "spectrum_high_hz": 5.0, "envelope_window_sec": 0.2},
            "baseline": {"bandpass_low_hz": 0.05, "bandpass_high_hz": 5.0, "filter_order": 2},
        }
    )

    frame = evaluate_baseline_dataset(TinyRespDataset(), cfg)

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert set(frame.columns) >= {"method", "dataset_row_id", "rr_spec_abs_error", "spectrum_similarity"}
```

- [x] **步骤 2：运行测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_baseline_metrics.py -v
```

预期：FAIL，报错包含 `cannot import name 'evaluate_baseline_dataset'`。

- [x] **步骤 3：抽出 `evaluate_baseline_dataset`**

在 `resp_train/metrics/baseline.py` 中新增：

```python
def evaluate_baseline_dataset(dataset: Any, cfg: DictConfig) -> pd.DataFrame:
    fs = float(cfg.window.get("target_fs", 100.0))
    low_hz = float(cfg.baseline.get("bandpass_low_hz", cfg.loss.get("spectrum_low_hz", 0.05)))
    high_hz = float(cfg.baseline.get("bandpass_high_hz", cfg.loss.get("spectrum_high_hz", 0.7)))
    order = int(cfg.baseline.get("filter_order", 4))
    envelope_window = max(1, int(round(float(cfg.loss.get("envelope_window_sec", 2.0)) * fs)))
    records: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        records.append(_evaluate_baseline_sample(sample, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order, envelope_window=envelope_window))
    return pd.DataFrame.from_records(records)
```

把原循环中的单样本逻辑移动到 `_evaluate_baseline_sample`。保留 `run_baseline(cfg, output)`，但内部改为：

```python
from resp_train.data.factory import build_window_data

bundle = build_window_data(
    cfg,
    split=str(cfg.data.get("val_split", "val")),
    max_windows=cfg.data.get("max_val_windows"),
    sample_strategy=str(cfg.data.get("val_sample_strategy", "stratified_random")),
    sample_seed=int(cfg.data.get("val_sample_seed", cfg.data.get("train_sample_seed", 0) + 1)),
    shuffle=False,
)
df = evaluate_baseline_dataset(bundle.dataset, cfg)
```

- [x] **步骤 4：运行 baseline 测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_baseline_metrics.py tests/test_eval_metrics.py -v
```

预期：PASS。

- [x] **步骤 5：Commit**

```bash
git add resp_train/metrics/baseline.py tests/test_baseline_metrics.py
git commit -m "refactor(指标): 复用数据集计算平凡基线"
```

---

### 任务 7：实现 THO 实验类

**文件：**
- 创建：`resp_train/experiments/tho.py`
- 修改：`resp_train/experiments/__init__.py`
- 创建：`tests/test_tho_experiment.py`

- [x] **步骤 1：编写 THO 实验 smoke 测试**

创建 `tests/test_tho_experiment.py`，复用任务 3 的临时数据思路，但配置补齐模型、loss 和输出。

```python
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from resp_train.experiments.tho import ThoExperiment


def _prepare_dataset(root: Path):
    training_dir = root / "training"
    npz_dir = root / "whole_night" / "mixed_bcg_to_tho" / "1"
    training_dir.mkdir(parents=True)
    npz_dir.mkdir(parents=True)
    t = np.linspace(0, 4 * np.pi, 128, dtype=np.float32)
    np.savez(npz_dir / "sample.npz", bcg=np.sin(t).astype(np.float32), tho=np.cos(t).astype(np.float32))
    records = []
    row_id = 1
    for split in ("train", "val"):
        for start in (0, 16, 32, 48):
            records.append(
                {
                    "dataset_row_id": row_id,
                    "input_set": "mixed_zscore",
                    "split": split,
                    "samp_id": 1,
                    "segment_id": 1,
                    "window_id_in_segment": row_id,
                    "source_npz": "../whole_night/mixed_bcg_to_tho/1/sample.npz",
                    "bcg_signal_key": "bcg",
                    "target_signal_key": "tho",
                    "valid_sec_key": "valid",
                    "segment_decision": "include_candidate",
                    "window_start_sample": start,
                    "window_end_sample": start + 32,
                    "window_duration_samples": 32,
                    "target_fs": 100,
                    "valid_ratio": 1.0,
                    "input_finite_ratio": 1.0,
                    "target_finite_ratio": 1.0,
                    "residual_quality_class": "near_zero_residual" if row_id % 2 else "stable_nonzero_residual",
                    "base_alignment_method": "keep_original",
                    "apply_decision": "approved",
                    "reason": "ok",
                }
            )
            row_id += 1
    pd.DataFrame.from_records(records).to_csv(training_dir / "dataset_index.csv", index=False)
```

配置函数：

```python
def _cfg(tmp_path: Path):
    root = tmp_path / "dataset"
    _prepare_dataset(root)
    return OmegaConf.create(
        {
            "data": {
                "dataset_root": str(root),
                "index_csv": "training/dataset_index.csv",
                "input_set": "mixed_zscore",
                "train_split": "train",
                "val_split": "val",
                "max_train_windows": 4,
                "max_val_windows": 4,
                "filter_unusable": True,
                "valid_ratio_min": 0.99,
                "input_finite_ratio_min": 0.99,
                "target_finite_ratio_min": 0.99,
                "unusable_residual_classes": [],
                "preload_windows": True,
                "train_sample_strategy": "stratified_random",
                "val_sample_strategy": "stratified_random",
                "train_sample_seed": 1,
                "val_sample_seed": 2,
                "stratify_column": "residual_quality_class",
            },
            "window": {"target_fs": 100, "duration_samples": 32, "duration_sec": 0.32},
            "model": {"name": "unet1d_tiny", "in_channels": 1, "out_channels": 1, "base_channels": 4},
            "loss": {
                "envelope_weight": 1.0,
                "spectrum_weight": 0.2,
                "smooth_weight": 0.001,
                "envelope_window_sec": 0.08,
                "spectrum_low_hz": 0.05,
                "spectrum_high_hz": 5.0,
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "num_workers": 0,
                "seed": 1,
                "device": "cpu",
                "patience": 2,
                "min_delta": 0.0,
                "lr_scheduler": "none",
                "grad_clip_norm": None,
                "use_amp": False,
            },
            "baseline": {"bandpass_low_hz": 0.05, "bandpass_high_hz": 5.0, "filter_order": 2},
            "outputs": {"run_root": str(tmp_path / "runs"), "max_prediction_windows": 2},
        }
    )
```

测试：

```python
def test_tho_experiment_smoke_writes_run_outputs(tmp_path: Path):
    run_dir = ThoExperiment(_cfg(tmp_path)).train()

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "audit.csv").exists()
    assert (run_dir / "baseline_metrics.csv").exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "train_history.csv").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "predictions.npz").exists()
```

- [x] **步骤 2：运行测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_tho_experiment.py -v
```

预期：FAIL，报错包含 `No module named 'resp_train.experiments.tho'`。

- [x] **步骤 3：实现 `ThoExperiment`**

创建 `resp_train/experiments/tho.py`：

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from resp_train.data.factory import build_tho_data
from resp_train.engine import collect_predictions
from resp_train.experiments.base import BaseExperiment, ExperimentData
from resp_train.losses.weak import WeakSyncLoss
from resp_train.metrics.baseline import evaluate_baseline_dataset
from resp_train.metrics.evaluate import evaluate_prediction_dict
from resp_train.models.registry import build_model


class ThoExperiment(BaseExperiment):
    task_name = "tho"

    def build_data(self) -> ExperimentData:
        tho_data = build_tho_data(self.cfg)
        return ExperimentData(
            train_loader=tho_data.train.loader,
            val_loader=tho_data.val.loader,
            audit_frame=tho_data.audited,
            audit_summary=tho_data.audit_summary,
            extras={"tho_data": tho_data},
        )

    def build_model(self):
        return build_model(self.cfg)

    def build_loss(self):
        return WeakSyncLoss(self.cfg)

    def run_baseline(self, data: ExperimentData, run_dir: Path) -> None:
        tho_data = data.extras["tho_data"]
        evaluate_baseline_dataset(tho_data.val.dataset, self.cfg).to_csv(run_dir / "baseline_metrics.csv", index=False)

    def evaluate_best(self, model: torch.nn.Module, data: ExperimentData, run_dir: Path) -> None:
        tho_data = data.extras["tho_data"]
        eval_preds = collect_predictions(model, tho_data.val.loader, device=self.device, max_windows=len(tho_data.val.dataset))
        evaluate_prediction_dict(eval_preds, self.cfg, method=str(self.cfg.model.name)).to_csv(run_dir / "metrics.csv", index=False)
        diag_preds = collect_predictions(
            model,
            tho_data.val.loader,
            device=self.device,
            max_windows=int(self.cfg.outputs.max_prediction_windows),
        )
        np.savez(run_dir / "predictions.npz", **diag_preds)
```

- [x] **步骤 4：导出 `ThoExperiment`**

更新 `resp_train/experiments/__init__.py`：

```python
from resp_train.experiments.base import BaseExperiment, ExperimentData
from resp_train.experiments.tho import ThoExperiment

__all__ = ["BaseExperiment", "ExperimentData", "ThoExperiment"]
```

- [x] **步骤 5：运行 THO 实验测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_tho_experiment.py -v
```

预期：PASS。

- [x] **步骤 6：Commit**

```bash
git add resp_train/experiments/__init__.py resp_train/experiments/tho.py tests/test_tho_experiment.py
git commit -m "feat(实验): 添加 THO 实验编排"
```

---

### 任务 8：迁移训练和评价脚本到实验层

**文件：**
- 修改：`scripts/train_tho_small.py`
- 修改：`scripts/eval_tho_small.py`
- 修改：`tests/test_diagnostics_scripts.py`

- [x] **步骤 1：补充脚本结构测试**

在 `tests/test_diagnostics_scripts.py` 中增加静态测试：

```python
def test_train_script_delegates_to_tho_experiment():
    source = Path("scripts/train_tho_small.py").read_text(encoding="utf-8")

    assert "ThoExperiment" in source
    assert "train_one_epoch" not in source
    assert "run_baseline" not in source


def test_eval_script_delegates_to_tho_checkpoint_evaluator():
    source = Path("scripts/eval_tho_small.py").read_text(encoding="utf-8")

    assert "evaluate_tho_checkpoint" in source
    assert "RespWindowDataset" not in source
    assert "filter_index" not in source
```

- [x] **步骤 2：运行脚本结构测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_diagnostics_scripts.py -v
```

预期：FAIL，当前脚本仍直接编排训练和评价。

- [x] **步骤 3：在 `resp_train/experiments/tho.py` 增加 checkpoint 评价入口**

实现：

```python
def evaluate_tho_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path | None,
    output_path: str | Path,
    metrics_output_path: str | Path | None,
    overrides: list[str] | None = None,
) -> Path:
    resolved_checkpoint = Path(checkpoint_path)
    resolved_config = _resolve_config_path(config_path, resolved_checkpoint)
    cfg = load_config(resolved_config, overrides=overrides)
    experiment = ThoExperiment(cfg)
    experiment.evaluate_checkpoint(
        resolved_checkpoint,
        output=Path(output_path),
        metrics_output=Path(metrics_output_path) if metrics_output_path else None,
    )
    return Path(output_path)
```

把 `_resolve_config_path` 和 `_validate_checkpoint_config` 从 `scripts/eval_tho_small.py` 移到 `resp_train/experiments/tho.py`，保持原校验字段，并把新增抽样字段加入校验：

```python
"data.val_sample_strategy",
"data.val_sample_seed",
"data.stratify_column",
```

在 `ThoExperiment` 中实现实例方法：

```python
def evaluate_checkpoint(self, checkpoint_path: Path, *, output: Path, metrics_output: Path | None) -> None:
    device = self.device or resolve_device(str(self.cfg.training.device))
    model = self.build_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _validate_checkpoint_config(checkpoint.get("config"), self.cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    data = self.build_data()
    tho_data = data.extras["tho_data"]
    output.parent.mkdir(parents=True, exist_ok=True)
    diag_preds = collect_predictions(model, tho_data.val.loader, device=device, max_windows=int(self.cfg.outputs.max_prediction_windows))
    np.savez(output, **diag_preds)
    if metrics_output is not None:
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        eval_preds = collect_predictions(model, tho_data.val.loader, device=device, max_windows=len(tho_data.val.dataset))
        evaluate_prediction_dict(eval_preds, self.cfg, method=str(self.cfg.model.name)).to_csv(metrics_output, index=False)
```

- [x] **步骤 4：瘦身训练脚本**

将 `scripts/train_tho_small.py` 主体改为：

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages, load_config
from resp_train.experiments.tho import ThoExperiment


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 THO 呼吸努力估计实验")
    parser.add_argument("--config", default="configs/tho_small.yaml", help="配置文件路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()

    missing = check_required_packages()
    if missing:
        raise SystemExit(f"缺少依赖: {missing}; 请先确认是否安装。")

    cfg = load_config(args.config, overrides=args.overrides)
    run_dir = ThoExperiment(cfg).train()
    print(run_dir)


if __name__ == "__main__":
    main()
```

- [x] **步骤 5：瘦身评价脚本**

将 `scripts/eval_tho_small.py` 主体改为：

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.experiments.tho import evaluate_tho_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="用 checkpoint 生成 THO 验证预测和指标")
    parser.add_argument("--config", default="", help="配置文件路径；为空时优先使用 checkpoint 同目录 config.yaml")
    parser.add_argument("--checkpoint", required=True, help="训练产生的 checkpoint.pt")
    parser.add_argument("--output", required=True, help="预测 NPZ 输出路径")
    parser.add_argument("--metrics-output", default="", help="可选指标 CSV 输出路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()

    output = evaluate_tho_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config or None,
        output_path=args.output,
        metrics_output_path=args.metrics_output or None,
        overrides=args.overrides,
    )
    print(f"写出预测: {output}")


if __name__ == "__main__":
    main()
```

- [x] **步骤 6：运行脚本结构测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_diagnostics_scripts.py -v
```

预期：PASS。

- [x] **步骤 7：Commit**

```bash
git add scripts/train_tho_small.py scripts/eval_tho_small.py resp_train/experiments/tho.py tests/test_diagnostics_scripts.py
git commit -m "refactor(脚本): 训练评价入口委托实验层"
```

---

### 任务 9：迁移审计和 baseline 脚本口径

**文件：**
- 修改：`scripts/audit_tho_dataset.py`
- 修改：`scripts/baseline_tho_hilbert.py`
- 修改：`scripts/README.md`
- 修改：`tests/test_diagnostics_scripts.py`

- [x] **步骤 1：补充脚本复用测试**

在 `tests/test_diagnostics_scripts.py` 中新增：

```python
def test_audit_and_baseline_scripts_use_data_factory():
    audit_source = Path("scripts/audit_tho_dataset.py").read_text(encoding="utf-8")
    baseline_source = Path("scripts/baseline_tho_hilbert.py").read_text(encoding="utf-8")

    assert "build_tho_data" in audit_source
    assert "build_window_data" in baseline_source or "build_tho_data" in baseline_source
```

- [x] **步骤 2：运行测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_diagnostics_scripts.py -v
```

预期：FAIL，当前脚本未复用数据工厂。

- [x] **步骤 3：更新审计脚本**

将 `scripts/audit_tho_dataset.py` 改为加载配置后调用 `build_tho_data(cfg)`，并输出 `data.audit_summary`：

```python
from resp_train.data.factory import build_tho_data

...
cfg = load_config(args.config, overrides=args.overrides)
data = build_tho_data(cfg)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
data.audit_summary.to_csv(output, index=False)
print(f"写出审计: {output}")
```

保留原有 `--config`、`--output` 和 `--set` 参数。

- [x] **步骤 4：更新 baseline 脚本**

将 `scripts/baseline_tho_hilbert.py` 改为使用 `build_tho_data(cfg)` 和 `evaluate_baseline_dataset(data.val.dataset, cfg)`：

```python
from resp_train.data.factory import build_tho_data
from resp_train.metrics.baseline import evaluate_baseline_dataset

...
cfg = load_config(args.config, overrides=args.overrides)
data = build_tho_data(cfg)
frame = evaluate_baseline_dataset(data.val.dataset, cfg)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
frame.to_csv(output, index=False)
print(f"写出 baseline: {output}")
```

- [x] **步骤 5：运行脚本复用测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_diagnostics_scripts.py -v
```

预期：PASS。

- [x] **步骤 6：Commit**

```bash
git add scripts/audit_tho_dataset.py scripts/baseline_tho_hilbert.py tests/test_diagnostics_scripts.py
git commit -m "refactor(脚本): 审计和基线复用数据工厂"
```

---

### 任务 10：更新文档和实验记录

**文件：**
- 修改：`docs/tho_small_training.md`
- 修改：`scripts/README.md`
- 修改：`docs/experiments/tho_small_mixed_zscore_20260607.md`

- [x] **步骤 1：更新训练说明**

在 `docs/tho_small_training.md` 中更新以下内容：

- 默认抽样改为 `data.train_sample_strategy=stratified_random` 和 `data.val_sample_strategy=stratified_random`。
- 说明 `head` 只用于 debug。
- 说明 train 和 val 抽样 seed 独立。
- 说明 `BaseExperiment` 和 `ThoExperiment` 的职责。
- smoke 命令改为：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_small.yaml \
  --set data.train_sample_seed=1 \
  --set data.val_sample_seed=2 \
  --set data.max_train_windows=16 \
  --set data.max_val_windows=8 \
  --set training.epochs=1 \
  --set model.base_channels=4 \
  --set outputs.max_prediction_windows=4
```

- [x] **步骤 2：更新脚本索引**

在 `scripts/README.md` 中说明：

- `train_tho_small.py` 是薄入口，内部委托 `ThoExperiment`。
- `eval_tho_small.py` 委托 checkpoint 评价函数。
- `audit_tho_dataset.py` 和 `baseline_tho_hilbert.py` 复用数据工厂口径。
- 脚本仍暂时平铺，不移动目录。

- [x] **步骤 3：更新实验记录**

在 `docs/experiments/tho_small_mixed_zscore_20260607.md` 的「当前判断」或「下一步」前加入：

```markdown
## 采样口径修正

这 4 个 run 使用首版前缀采样口径。真实索引检查显示，前缀窗口会集中到少数 `samp_id` 和片段，因此这些结果只作为 smoke test 和现象记录，不作为正式消融结论。

后续消融应基于新实验骨架的 `stratified_random` 默认策略，并固定 `val_sample_seed`，再做多 seed 比较。
```

将「下一步」改为先重跑默认 loss 与 `smooth_weight=0.001` 的分层随机对照，不继续扩大旧前缀采样结论。

- [x] **步骤 4：运行文档检查**

运行：

```bash
rg -n "data.sample_strategy|默认.*head|前缀采样.*正式|继续做窄范围 smooth" docs scripts
git diff --check
```

预期：

- `rg` 不应再命中把 `data.sample_strategy` 或默认 `head` 当作当前配置的内容。
- `git diff --check` exit 0。

- [x] **步骤 5：Commit**

```bash
git add docs/tho_small_training.md scripts/README.md docs/experiments/tho_small_mixed_zscore_20260607.md
git commit -m "docs(实验): 更新分层随机实验口径"
```

---

### 任务 11：全量验证和 smoke run

**文件：**
- 只产生被 `.gitignore` 忽略的 `runs/` 输出。

- [x] **步骤 1：运行重点测试**

运行：

```bash
./.venv/bin/python -m pytest tests/test_data_index_audit.py tests/test_data_factory.py tests/test_base_experiment.py tests/test_tho_experiment.py -v
```

预期：PASS。

- [x] **步骤 2：运行全量测试**

运行：

```bash
./.venv/bin/python -m pytest tests -q
```

预期：PASS。

- [x] **步骤 3：运行真实数据小 smoke**

运行：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_small.yaml \
  --set data.train_sample_seed=1 \
  --set data.val_sample_seed=2 \
  --set data.max_train_windows=16 \
  --set data.max_val_windows=8 \
  --set training.epochs=1 \
  --set model.base_channels=4 \
  --set outputs.max_prediction_windows=4
```

预期：

- 命令打印 `runs/tho_small/<timestamp>`。
- run 目录包含 `config.yaml`、`audit.csv`、`baseline_metrics.csv`、`checkpoint.pt`、`train_history.csv`、`metrics.csv`、`predictions.npz`、`train.log`。

- [x] **步骤 4：检查 run 配置记录**

运行：

```bash
run_dir=$(ls -td runs/tho_small/* | head -1)
./.venv/bin/python -c "from omegaconf import OmegaConf; import sys; cfg=OmegaConf.load(sys.argv[1] + '/config.yaml'); print(cfg.data.train_sample_strategy, cfg.data.val_sample_strategy, cfg.data.train_sample_seed, cfg.data.val_sample_seed)" "$run_dir"
```

预期输出包含：

```text
stratified_random stratified_random 1 2
```

- [x] **步骤 5：提交最终验证相关代码**

如果前面任务已经逐步提交，且 smoke run 只产生忽略文件，本步骤不需要新 commit。若验证过程中修复了代码或文档，按修改文件提交：

```bash
git add <修复过的文件>
git commit -m "fix(实验): 修正实验骨架验证问题"
```

---

## 最终验收清单

- [x] 默认训练和验证抽样策略均为 `stratified_random`。
- [x] `head` 只能通过显式配置启用。
- [x] train 和 val 抽样 seed 独立记录。
- [x] 训练支持 best checkpoint 和 early stopping。
- [x] 训练和评价复用同一数据工厂。
- [x] `BaseExperiment` 只承担公共生命周期，不包含 THO 专属指标。
- [x] `ThoExperiment` 承担 THO 任务数据、baseline、loss 和评价。
- [x] 训练、评价、审计、baseline 脚本均为薄入口或复用数据工厂。
- [x] 旧实验记录标注为前缀采样下的 smoke/现象记录。
- [x] 重点测试、全量测试和真实数据小 smoke 均通过。
