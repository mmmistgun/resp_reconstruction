# 胸带参考小规模训练骨架实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建胸带参考 `r_tho_hat(t)` 的小规模训练、诊断和评价闭环。

**架构：** 使用纯 PyTorch 训练工程，按 `configs/`、`resp_train/`、`scripts/` 分层组织。数据层读取 Stage 2.1 `dataset_index.csv` 并缓存整晚 NPZ；模型层通过注册表提供默认 `unet1d_tiny`；训练层输出轻量 run 目录，包含审计、平凡基线、指标、checkpoint 和少量诊断预测。

**技术栈：** Python、PyTorch、NumPy、Pandas、SciPy、tqdm、OmegaConf、pytest。

---

## 文件结构

创建文件：

- `configs/tho_small.yaml`：首版训练默认配置。
- `resp_train/__init__.py`：包标记。
- `resp_train/config.py`：配置加载、合并和依赖检查。
- `resp_train/data/__init__.py`：数据子包导出。
- `resp_train/data/index.py`：读取、校验和过滤 `dataset_index.csv`。
- `resp_train/data/audit.py`：生成 `usable` 标记和 `audit.csv` 摘要。
- `resp_train/data/cache.py`：整晚 NPZ 缓存。
- `resp_train/data/dataset.py`：PyTorch Dataset，按窗口切片。
- `resp_train/metrics/__init__.py`：指标子包导出。
- `resp_train/metrics/signal.py`：包络、频谱、主频、峰谷和基础指标。
- `resp_train/metrics/baseline.py`：val 子集平凡基线的可复用函数。
- `resp_train/metrics/evaluate.py`：模型输出的窗口级评价指标。
- `resp_train/losses/__init__.py`：损失子包导出。
- `resp_train/losses/weak.py`：包络、频谱和平滑损失。
- `resp_train/models/__init__.py`：模型子包导出。
- `resp_train/models/registry.py`：模型注册表。
- `resp_train/models/unet1d.py`：默认 `unet1d_tiny`。
- `resp_train/engine/__init__.py`：训练子包导出。
- `resp_train/engine/train.py`：训练、验证、checkpoint 和诊断预测。
- `resp_train/utils/__init__.py`：工具子包导出。
- `resp_train/utils/run.py`：run 目录、日志、配置快照、随机种子。
- `scripts/audit_tho_dataset.py`：生成审计摘要的命令入口。
- `scripts/baseline_tho_hilbert.py`：val 子集平凡基线入口。
- `scripts/train_tho_small.py`：小规模训练入口。
- `scripts/eval_tho_small.py`：checkpoint 评价入口。
- `tests/test_config.py`：配置与依赖检查测试。
- `tests/test_data_index_audit.py`：索引校验与审计测试。
- `tests/test_dataset_cache.py`：缓存和窗口切片测试。
- `tests/test_signal_metrics.py`：信号指标测试。
- `tests/test_losses.py`：损失函数测试。
- `tests/test_model_registry.py`：模型注册与输出形状测试。
- `tests/test_engine_smoke.py`：一批次训练 smoke test。

不修改当前未跟踪的 `.gitignore`。run 目录忽略策略留作独立维护项，避免把用户工作区中的无关文件混入实现提交。

---

### 任务 1：配置、包结构与依赖检查

**文件：**
- 创建：`configs/tho_small.yaml`
- 创建：`resp_train/__init__.py`
- 创建：`resp_train/config.py`
- 创建：`tests/test_config.py`

- [ ] **步骤 1：编写配置测试**

```python
# tests/test_config.py
from pathlib import Path

from resp_train.config import REQUIRED_PACKAGES, load_config, check_required_packages


def test_load_default_config_has_expected_values():
    cfg = load_config("configs/tho_small.yaml")

    assert Path(cfg.data.dataset_root).name == "20260530_tho_ramp5_stage2_1"
    assert cfg.data.input_set == "mixed_zscore"
    assert cfg.data.max_train_windows == 1024
    assert cfg.data.max_val_windows == 256
    assert bool(cfg.data.filter_unusable) is True
    assert cfg.training.epochs == 3
    assert cfg.model.name == "unet1d_tiny"
    assert cfg.outputs.max_prediction_windows == 32


def test_required_packages_list_is_explicit():
    assert REQUIRED_PACKAGES == [
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
        "omegaconf",
    ]


def test_check_required_packages_returns_missing_names(monkeypatch):
    import importlib

    real_import = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "omegaconf":
            raise ImportError("missing omegaconf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    assert check_required_packages() == ["omegaconf"]


def test_load_config_applies_dotlist_overrides():
    cfg = load_config(
        "configs/tho_small.yaml",
        overrides=[
            "data.max_train_windows=16",
            "data.max_val_windows=8",
            "training.epochs=1",
        ],
    )

    assert cfg.data.max_train_windows == 16
    assert cfg.data.max_val_windows == 8
    assert cfg.training.epochs == 1
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_config.py -v`

预期：FAIL，报错包含 `No module named 'resp_train'` 或 `cannot import name 'load_config'`。

- [ ] **步骤 3：创建配置文件**

```yaml
# configs/tho_small.yaml
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

window:
  target_fs: 100
  duration_samples: 18000
  duration_sec: 180

model:
  name: unet1d_tiny
  in_channels: 1
  out_channels: 1
  base_channels: 16

loss:
  envelope_weight: 1.0
  spectrum_weight: 0.2
  smooth_weight: 0.01
  envelope_window_sec: 2.0
  spectrum_low_hz: 0.05
  spectrum_high_hz: 0.7

training:
  epochs: 3
  batch_size: 8
  learning_rate: 0.001
  num_workers: 0
  seed: 20260601
  device: auto

baseline:
  bandpass_low_hz: 0.05
  bandpass_high_hz: 0.7
  filter_order: 4

outputs:
  run_root: runs/tho_small
  max_prediction_windows: 32
```

- [ ] **步骤 4：创建配置加载代码**

```python
# resp_train/__init__.py
"""胸带参考呼吸努力估计训练包。"""
```

```python
# resp_train/config.py
from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from omegaconf import DictConfig


REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "tqdm",
    "omegaconf",
]


def check_required_packages(packages: Iterable[str] = REQUIRED_PACKAGES) -> list[str]:
    missing: list[str] = []
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    return missing


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> "DictConfig":
    from omegaconf import OmegaConf

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Any) -> None:
    from omegaconf import OmegaConf

    required = [
        "data.dataset_root",
        "data.index_csv",
        "data.input_set",
        "window.duration_samples",
        "model.name",
        "training.epochs",
        "outputs.run_root",
    ]
    for key in required:
        if OmegaConf.select(cfg, key) is None:
            raise ValueError(f"配置缺少必需字段: {key}")
```

- [ ] **步骤 5：运行测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_config.py -v`

预期：PASS。若失败原因是缺少 `omegaconf` 或其他依赖，停止并报告缺失包名、用途和建议命令：`nv pip install omegaconf`，等待用户确认。

- [ ] **步骤 6：Commit**

```bash
git add configs/tho_small.yaml resp_train/__init__.py resp_train/config.py tests/test_config.py
git commit -m "feat: add tho small training config"
```

### 任务 2：索引读取、校验与诊断审计

**文件：**
- 创建：`resp_train/data/__init__.py`
- 创建：`resp_train/data/index.py`
- 创建：`resp_train/data/audit.py`
- 创建：`scripts/audit_tho_dataset.py`
- 创建：`tests/test_data_index_audit.py`

- [ ] **步骤 1：编写索引与审计测试**

```python
# tests/test_data_index_audit.py
import pandas as pd
from omegaconf import OmegaConf

from resp_train.data.audit import add_usable_flag, summarize_audit
from resp_train.data.index import filter_index, validate_index_columns


def _rows():
    return pd.DataFrame(
        [
            {
                "dataset_row_id": 1,
                "input_set": "mixed_zscore",
                "split": "train",
                "samp_id": 88,
                "segment_id": 1,
                "window_id_in_segment": 1,
                "source_npz": "../whole_night/mixed_bcg_to_tho/88/mixed_bcg_to_tho.npz",
                "bcg_signal_key": "bcg_mixed_refined_to_tho_zscore",
                "target_signal_key": "tho_ref",
                "valid_sec_key": "mixed_train_valid_sec",
                "segment_decision": "include_candidate",
                "window_start_sample": 0,
                "window_end_sample": 18000,
                "window_duration_samples": 18000,
                "target_fs": 100,
                "valid_ratio": 1.0,
                "input_finite_ratio": 1.0,
                "target_finite_ratio": 1.0,
                "residual_quality_class": "near_zero_residual",
                "base_alignment_method": "keep_original",
                "apply_decision": "approved",
                "reason": "ok",
            },
            {
                "dataset_row_id": 2,
                "input_set": "legacy_v1",
                "split": "train",
                "samp_id": 88,
                "segment_id": 1,
                "window_id_in_segment": 2,
                "source_npz": "../whole_night/legacy_v1/88/stage_1_4_applied.npz",
                "bcg_signal_key": "bcg_refined_to_tho",
                "target_signal_key": "tho_ref",
                "valid_sec_key": "valid_after_refinement_sec",
                "segment_decision": "include_candidate",
                "window_start_sample": 0,
                "window_end_sample": 18000,
                "window_duration_samples": 18000,
                "target_fs": 100,
                "valid_ratio": 0.5,
                "input_finite_ratio": 1.0,
                "target_finite_ratio": 1.0,
                "residual_quality_class": "bad",
                "base_alignment_method": "keep_original",
                "apply_decision": "approved",
                "reason": "low valid",
            },
        ]
    )


def test_validate_index_columns_accepts_required_columns():
    validate_index_columns(_rows())


def test_filter_index_selects_input_set_and_split_and_limit():
    cfg = OmegaConf.create({"data": {"input_set": "mixed_zscore"}})
    filtered = filter_index(_rows(), cfg, split="train", max_windows=1)
    assert filtered["dataset_row_id"].tolist() == [1]


def test_add_usable_flag_uses_thresholds_and_classes():
    cfg = OmegaConf.create(
        {
            "data": {
                "valid_ratio_min": 0.99,
                "input_finite_ratio_min": 0.99,
                "target_finite_ratio_min": 0.99,
                "unusable_residual_classes": ["bad"],
            }
        }
    )
    audited = add_usable_flag(_rows(), cfg)
    assert bool(audited.loc[0, "usable"]) is True
    assert bool(audited.loc[1, "usable"]) is False


def test_summarize_audit_groups_by_split_input_set_quality():
    cfg = OmegaConf.create(
        {
            "data": {
                "valid_ratio_min": 0.99,
                "input_finite_ratio_min": 0.99,
                "target_finite_ratio_min": 0.99,
                "unusable_residual_classes": [],
            }
        }
    )
    summary = summarize_audit(add_usable_flag(_rows(), cfg))
    assert set(summary.columns) >= {
        "split",
        "input_set",
        "residual_quality_class",
        "n_windows",
        "n_usable",
        "usable_ratio",
    }
    assert summary["n_windows"].sum() == 2
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_data_index_audit.py -v`

预期：FAIL，报错包含 `No module named 'resp_train.data'`。

- [ ] **步骤 3：实现索引读取与校验**

```python
# resp_train/data/__init__.py
"""数据索引、审计、缓存和 Dataset。"""
```

```python
# resp_train/data/index.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


REQUIRED_INDEX_COLUMNS = [
    "dataset_row_id",
    "input_set",
    "split",
    "samp_id",
    "segment_id",
    "window_id_in_segment",
    "source_npz",
    "bcg_signal_key",
    "target_signal_key",
    "valid_sec_key",
    "segment_decision",
    "window_start_sample",
    "window_end_sample",
    "window_duration_samples",
    "target_fs",
    "valid_ratio",
    "input_finite_ratio",
    "target_finite_ratio",
    "residual_quality_class",
    "base_alignment_method",
    "apply_decision",
    "reason",
]


def read_index(dataset_root: str | Path, index_csv: str | Path) -> pd.DataFrame:
    index_path = Path(dataset_root) / index_csv
    if not index_path.exists():
        raise FileNotFoundError(f"索引文件不存在: {index_path}")
    df = pd.read_csv(index_path)
    validate_index_columns(df)
    return df


def validate_index_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_INDEX_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"索引缺少必需列: {missing}")


def filter_index(
    df: pd.DataFrame,
    cfg: DictConfig,
    *,
    split: str,
    max_windows: int | None,
) -> pd.DataFrame:
    filtered = df[(df["input_set"] == cfg.data.input_set) & (df["split"] == split)].copy()
    filtered = filtered.sort_values("dataset_row_id").reset_index(drop=True)
    if max_windows is not None:
        filtered = filtered.head(int(max_windows)).reset_index(drop=True)
    return filtered
```

- [ ] **步骤 4：实现审计逻辑和脚本**

```python
# resp_train/data/audit.py
from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig


def add_usable_flag(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    audited = df.copy()
    unusable_classes = set(cfg.data.get("unusable_residual_classes", []))
    audited["usable"] = (
        (audited["valid_ratio"] >= float(cfg.data.valid_ratio_min))
        & (audited["input_finite_ratio"] >= float(cfg.data.input_finite_ratio_min))
        & (audited["target_finite_ratio"] >= float(cfg.data.target_finite_ratio_min))
        & (audited["segment_decision"] == "include_candidate")
        & (~audited["residual_quality_class"].isin(unusable_classes))
    )
    audited["usable"] = audited["usable"].astype(bool)
    return audited


def summarize_audit(audited: pd.DataFrame) -> pd.DataFrame:
    grouped = audited.groupby(["split", "input_set", "residual_quality_class"], dropna=False)
    summary = grouped.agg(
        n_windows=("dataset_row_id", "count"),
        n_usable=("usable", "sum"),
        valid_ratio_mean=("valid_ratio", "mean"),
        valid_ratio_min=("valid_ratio", "min"),
        input_finite_ratio_mean=("input_finite_ratio", "mean"),
        target_finite_ratio_mean=("target_finite_ratio", "mean"),
        base_alignment_method_main=("base_alignment_method", _mode_or_empty),
        apply_decision_main=("apply_decision", _mode_or_empty),
    ).reset_index()
    summary["usable_ratio"] = summary["n_usable"] / summary["n_windows"].clip(lower=1)
    return summary


def _mode_or_empty(series: pd.Series) -> str:
    mode = series.mode(dropna=True)
    if mode.empty:
        return ""
    return str(mode.iloc[0])
```

```python
# scripts/audit_tho_dataset.py
from __future__ import annotations

import argparse
from pathlib import Path

from resp_train.config import load_config
from resp_train.data.audit import add_usable_flag, summarize_audit
from resp_train.data.index import read_index


def main() -> None:
    parser = argparse.ArgumentParser(description="生成胸带小规模训练数据审计表")
    parser.add_argument("--config", default="configs/tho_small.yaml")
    parser.add_argument("--output", default="audit.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = read_index(cfg.data.dataset_root, cfg.data.index_csv)
    audited = add_usable_flag(df, cfg)
    summary = summarize_audit(audited)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"写出审计摘要: {output} rows={len(summary)}")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 5：运行测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_data_index_audit.py -v`

预期：PASS。

- [ ] **步骤 6：用真实索引运行审计 smoke test**

运行：`./.venv/bin/python scripts/audit_tho_dataset.py --config configs/tho_small.yaml --output /tmp/tho_audit.csv`

预期：退出码 0，输出包含 `写出审计摘要`，且 `/tmp/tho_audit.csv` 存在。

- [ ] **步骤 7：Commit**

```bash
git add resp_train/data/__init__.py resp_train/data/index.py resp_train/data/audit.py scripts/audit_tho_dataset.py tests/test_data_index_audit.py
git commit -m "feat: add dataset audit"
```

### 任务 3：整晚缓存与窗口 Dataset

**文件：**
- 创建：`resp_train/data/cache.py`
- 创建：`resp_train/data/dataset.py`
- 创建：`tests/test_dataset_cache.py`

- [ ] **步骤 1：编写缓存与 Dataset 测试**

```python
# tests/test_dataset_cache.py
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.data.dataset import RespWindowDataset


def test_dataset_slices_npz_window_and_returns_metadata(tmp_path):
    root = tmp_path / "dataset"
    npz_dir = root / "whole_night" / "mixed_bcg_to_tho" / "88"
    npz_dir.mkdir(parents=True)
    np.savez(
        npz_dir / "sample.npz",
        bcg=np.arange(200, dtype=np.float32),
        tho=np.arange(200, dtype=np.float32) * 2,
        valid=np.ones(2, dtype=np.uint8),
    )
    df = pd.DataFrame(
        [
            {
                "dataset_row_id": 10,
                "split": "train",
                "input_set": "mixed_zscore",
                "source_npz": "../whole_night/mixed_bcg_to_tho/88/sample.npz",
                "bcg_signal_key": "bcg",
                "target_signal_key": "tho",
                "valid_sec_key": "valid",
                "window_start_sample": 10,
                "window_end_sample": 30,
                "window_duration_samples": 20,
                "target_fs": 100,
                "samp_id": 88,
                "segment_id": 1,
                "window_id_in_segment": 2,
                "residual_quality_class": "near_zero_residual",
                "usable": True,
            }
        ]
    )
    cfg = OmegaConf.create({"window": {"duration_samples": 20}, "data": {"filter_unusable": True}})
    dataset = RespWindowDataset(root / "training" / "dataset_index.csv", df, cfg, preload_windows=True)

    sample = dataset[0]
    assert torch.equal(sample["x"], torch.arange(10, 30, dtype=torch.float32).view(1, -1))
    assert torch.equal(sample["target"], (torch.arange(10, 30, dtype=torch.float32) * 2).view(1, -1))
    assert sample["meta"]["dataset_row_id"] == 10


def test_dataset_filters_unusable_when_configured(tmp_path):
    df = pd.DataFrame(
        [
            {
                "dataset_row_id": 1,
                "split": "train",
                "input_set": "mixed_zscore",
                "source_npz": "missing.npz",
                "bcg_signal_key": "bcg",
                "target_signal_key": "tho",
                "valid_sec_key": "valid",
                "window_start_sample": 0,
                "window_end_sample": 20,
                "window_duration_samples": 20,
                "target_fs": 100,
                "samp_id": 88,
                "segment_id": 1,
                "window_id_in_segment": 1,
                "residual_quality_class": "bad",
                "usable": False,
            }
        ]
    )
    cfg = OmegaConf.create({"window": {"duration_samples": 20}, "data": {"filter_unusable": True}})
    dataset = RespWindowDataset(tmp_path / "training" / "dataset_index.csv", df, cfg)
    assert len(dataset) == 0
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_dataset_cache.py -v`

预期：FAIL，报错包含 `No module named 'resp_train.data.dataset'`。

- [ ] **步骤 3：实现整晚缓存**

```python
# resp_train/data/cache.py
from __future__ import annotations

from pathlib import Path

import numpy as np


class WholeNightCache:
    def __init__(self, index_csv_path: str | Path):
        self.index_csv_path = Path(index_csv_path)
        self.base_dir = self.index_csv_path.parent
        self._cache: dict[Path, dict[str, np.ndarray]] = {}

    def resolve(self, source_npz: str) -> Path:
        path = (self.base_dir / source_npz).resolve()
        if not path.exists():
            raise FileNotFoundError(f"源 NPZ 不存在: {path}")
        return path

    def get_arrays(self, source_npz: str, keys: list[str]) -> dict[str, np.ndarray]:
        path = self.resolve(source_npz)
        if path not in self._cache:
            with np.load(path) as data:
                missing = [key for key in keys if key not in data.files]
                if missing:
                    raise KeyError(f"{path} 缺少数组: {missing}")
                self._cache[path] = {key: np.asarray(data[key]) for key in keys}
        cached = self._cache[path]
        missing_cached = [key for key in keys if key not in cached]
        if missing_cached:
            with np.load(path) as data:
                for key in missing_cached:
                    if key not in data.files:
                        raise KeyError(f"{path} 缺少数组: {key}")
                    cached[key] = np.asarray(data[key])
        return {key: cached[key] for key in keys}
```

- [ ] **步骤 4：实现 Dataset**

```python
# resp_train/data/dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from resp_train.data.cache import WholeNightCache


class RespWindowDataset(Dataset):
    def __init__(
        self,
        index_csv_path: str | Path,
        rows: pd.DataFrame,
        cfg: DictConfig,
        *,
        preload_windows: bool = False,
    ) -> None:
        self.index_csv_path = Path(index_csv_path)
        self.cfg = cfg
        self.rows = rows.copy().reset_index(drop=True)
        if bool(cfg.data.get("filter_unusable", True)) and "usable" in self.rows.columns:
            self.rows = self.rows[self.rows["usable"]].reset_index(drop=True)
        self.cache = WholeNightCache(self.index_csv_path)
        self._preloaded: list[dict[str, Any]] | None = None
        if preload_windows:
            self._preloaded = [self._load_item(i) for i in range(len(self.rows))]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._preloaded is not None:
            return self._preloaded[idx]
        return self._load_item(idx)

    def _load_item(self, idx: int) -> dict[str, Any]:
        row = self.rows.iloc[idx]
        start = int(row["window_start_sample"])
        end = int(row["window_end_sample"])
        expected = int(self.cfg.window.duration_samples)
        if end <= start or (end - start) != expected:
            raise ValueError(f"窗口长度异常 row={row['dataset_row_id']} start={start} end={end}")
        arrays = self.cache.get_arrays(
            str(row["source_npz"]),
            [str(row["bcg_signal_key"]), str(row["target_signal_key"])],
        )
        x = arrays[str(row["bcg_signal_key"])][start:end].astype(np.float32, copy=False)
        y = arrays[str(row["target_signal_key"])][start:end].astype(np.float32, copy=False)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"窗口包含非有限值 row={row['dataset_row_id']}")
        return {
            "x": torch.from_numpy(x.copy()).view(1, -1),
            "target": torch.from_numpy(y.copy()).view(1, -1),
            "meta": {
                "dataset_row_id": int(row["dataset_row_id"]),
                "split": str(row.get("split", "")),
                "input_set": str(row.get("input_set", "")),
                "samp_id": int(row["samp_id"]),
                "segment_id": int(row["segment_id"]),
                "window_id_in_segment": int(row["window_id_in_segment"]),
                "residual_quality_class": str(row.get("residual_quality_class", "")),
            },
        }
```

- [ ] **步骤 5：运行测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_dataset_cache.py -v`

预期：PASS。

- [ ] **步骤 6：真实数据 Dataset smoke test**

运行：

```bash
./.venv/bin/python - <<'PY'
from resp_train.config import load_config
from resp_train.data.audit import add_usable_flag
from resp_train.data.dataset import RespWindowDataset
from resp_train.data.index import read_index, filter_index

cfg = load_config("configs/tho_small.yaml")
df = read_index(cfg.data.dataset_root, cfg.data.index_csv)
audited = add_usable_flag(df, cfg)
rows = filter_index(audited, cfg, split=cfg.data.train_split, max_windows=2)
ds = RespWindowDataset(
    f"{cfg.data.dataset_root}/training/dataset_index.csv",
    rows,
    cfg,
    preload_windows=False,
)
sample = ds[0]
print(sample["x"].shape, sample["target"].shape, sample["meta"])
PY
```

预期：输出 `torch.Size([1, 18000]) torch.Size([1, 18000])`。

- [ ] **步骤 7：Commit**

```bash
git add resp_train/data/cache.py resp_train/data/dataset.py tests/test_dataset_cache.py
git commit -m "feat: add respiration window dataset"
```

### 任务 4：信号指标、RR/主频读数与平凡基线

**文件：**
- 创建：`resp_train/metrics/__init__.py`
- 创建：`resp_train/metrics/signal.py`
- 创建：`resp_train/metrics/baseline.py`
- 创建：`scripts/baseline_tho_hilbert.py`
- 创建：`tests/test_signal_metrics.py`

- [ ] **步骤 1：编写指标测试**

```python
# tests/test_signal_metrics.py
import numpy as np

from resp_train.metrics.signal import (
    bandpass_filter,
    estimate_peak_rate_bpm,
    estimate_spectral_rate_bpm,
    rms_envelope,
    spectrum_similarity,
)


def test_rms_envelope_preserves_length():
    x = np.ones(100, dtype=np.float32)
    env = rms_envelope(x, window_samples=11)
    assert env.shape == x.shape
    assert np.allclose(env[10:-10], 1.0)


def test_spectral_rate_detects_sine_frequency():
    fs = 100
    t = np.arange(0, 60, 1 / fs)
    x = np.sin(2 * np.pi * 0.25 * t)
    bpm = estimate_spectral_rate_bpm(x, fs=fs, low_hz=0.05, high_hz=0.7)
    assert abs(bpm - 15.0) < 1.0


def test_peak_rate_detects_sine_cycles():
    fs = 100
    t = np.arange(0, 60, 1 / fs)
    x = np.sin(2 * np.pi * 0.2 * t)
    bpm = estimate_peak_rate_bpm(x, fs=fs, distance_sec=2.0)
    assert abs(bpm - 12.0) < 2.0


def test_spectrum_similarity_identical_is_one():
    fs = 100
    t = np.arange(0, 60, 1 / fs)
    x = np.sin(2 * np.pi * 0.25 * t)
    assert spectrum_similarity(x, x, fs=fs, low_hz=0.05, high_hz=0.7) > 0.99


def test_bandpass_filter_returns_same_shape():
    fs = 100
    x = np.random.default_rng(0).normal(size=1000).astype(np.float32)
    y = bandpass_filter(x, fs=fs, low_hz=0.05, high_hz=0.7, order=2)
    assert y.shape == x.shape
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_signal_metrics.py -v`

预期：FAIL，报错包含 `No module named 'resp_train.metrics'`。

- [ ] **步骤 3：实现信号指标**

```python
# resp_train/metrics/__init__.py
"""信号读数与评价指标。"""
```

```python
# resp_train/metrics/signal.py
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


def rms_envelope(x: np.ndarray, window_samples: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    window = max(1, int(window_samples))
    kernel = np.ones(window, dtype=np.float32) / window
    power = np.convolve(np.square(x), kernel, mode="same")
    return np.sqrt(np.maximum(power, 0.0)).astype(np.float32)


def bandpass_filter(
    x: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float32)).astype(np.float32)


def estimate_spectral_rate_bpm(
    x: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
) -> float:
    x = np.asarray(x, dtype=np.float32)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x - np.mean(x)))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return float("nan")
    band_freqs = freqs[mask]
    band_power = spectrum[mask]
    if np.all(band_power <= 0):
        return float("nan")
    return float(band_freqs[int(np.argmax(band_power))] * 60.0)


def estimate_peak_rate_bpm(x: np.ndarray, *, fs: float, distance_sec: float) -> float:
    x = np.asarray(x, dtype=np.float32)
    distance = max(1, int(distance_sec * fs))
    peaks, _ = find_peaks(x, distance=distance)
    duration_min = x.size / fs / 60.0
    if duration_min <= 0:
        return float("nan")
    return float(len(peaks) / duration_min)


def spectrum_similarity(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
) -> float:
    pred_dist = _band_distribution(pred, fs=fs, low_hz=low_hz, high_hz=high_hz)
    target_dist = _band_distribution(target, fs=fs, low_hz=low_hz, high_hz=high_hz)
    return float(np.sum(np.sqrt(pred_dist * target_dist)))


def _band_distribution(x: np.ndarray, *, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    power = np.square(np.abs(np.fft.rfft(x - np.mean(x))))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    band = power[mask].astype(np.float64)
    total = np.sum(band)
    if total <= 0:
        return np.ones_like(band, dtype=np.float64) / max(1, band.size)
    return band / total
```

- [ ] **步骤 4：实现可复用平凡基线**

```python
# resp_train/metrics/baseline.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from resp_train.data.audit import add_usable_flag
from resp_train.data.dataset import RespWindowDataset
from resp_train.data.index import filter_index, read_index
from resp_train.metrics.signal import (
    bandpass_filter,
    estimate_peak_rate_bpm,
    estimate_spectral_rate_bpm,
    rms_envelope,
    spectrum_similarity,
)


def run_baseline(cfg: DictConfig, output: str | Path) -> pd.DataFrame:
    df = read_index(cfg.data.dataset_root, cfg.data.index_csv)
    audited = add_usable_flag(df, cfg)
    rows = filter_index(audited, cfg, split=cfg.data.val_split, max_windows=cfg.data.max_val_windows)
    dataset = RespWindowDataset(
        Path(cfg.data.dataset_root) / cfg.data.index_csv,
        rows,
        cfg,
        preload_windows=False,
    )
    records: list[dict[str, Any]] = []
    fs = float(cfg.window.target_fs)
    for sample in dataset:
        x = sample["x"].squeeze(0).numpy()
        y = sample["target"].squeeze(0).numpy()
        pred = bandpass_filter(
            x,
            fs=fs,
            low_hz=float(cfg.baseline.bandpass_low_hz),
            high_hz=float(cfg.baseline.bandpass_high_hz),
            order=int(cfg.baseline.filter_order),
        )
        pred_env = rms_envelope(pred, int(fs * float(cfg.loss.envelope_window_sec)))
        target_env = rms_envelope(y, int(fs * float(cfg.loss.envelope_window_sec)))
        pred_rr_spec = estimate_spectral_rate_bpm(pred, fs=fs, low_hz=cfg.loss.spectrum_low_hz, high_hz=cfg.loss.spectrum_high_hz)
        target_rr_spec = estimate_spectral_rate_bpm(y, fs=fs, low_hz=cfg.loss.spectrum_low_hz, high_hz=cfg.loss.spectrum_high_hz)
        pred_rr_peak = estimate_peak_rate_bpm(pred, fs=fs, distance_sec=2.0)
        target_rr_peak = estimate_peak_rate_bpm(y, fs=fs, distance_sec=2.0)
        meta = sample["meta"]
        records.append(
            {
                **meta,
                "method": "bandpass_rms",
                "pred_rr_spec_bpm": pred_rr_spec,
                "target_rr_spec_bpm": target_rr_spec,
                "rr_spec_abs_error": abs(pred_rr_spec - target_rr_spec),
                "pred_rr_peak_bpm": pred_rr_peak,
                "target_rr_peak_bpm": target_rr_peak,
                "rr_peak_abs_error": abs(pred_rr_peak - target_rr_peak),
                "envelope_corr": pd.Series(pred_env).corr(pd.Series(target_env)),
                "spectrum_similarity": spectrum_similarity(pred, y, fs=fs, low_hz=cfg.loss.spectrum_low_hz, high_hz=cfg.loss.spectrum_high_hz),
            }
        )
    frame = pd.DataFrame(records)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
```

- [ ] **步骤 5：实现平凡基线脚本**

```python
# scripts/baseline_tho_hilbert.py
from __future__ import annotations

import argparse

from resp_train.config import load_config
from resp_train.metrics.baseline import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 val 子集低通/Hilbert 平凡基线")
    parser.add_argument("--config", default="configs/tho_small.yaml")
    parser.add_argument("--output", default="baseline_metrics.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    frame = run_baseline(cfg, args.output)
    print(f"写出平凡基线指标: {args.output} rows={len(frame)}")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 6：运行测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_signal_metrics.py -v`

预期：PASS。

- [ ] **步骤 7：真实数据基线 smoke test**

运行：`./.venv/bin/python scripts/baseline_tho_hilbert.py --config configs/tho_small.yaml --output /tmp/baseline_metrics.csv`

预期：退出码 0，输出包含 `写出平凡基线指标`，且 `/tmp/baseline_metrics.csv` 存在。基线指标可差，不作为失败。

- [ ] **步骤 8：Commit**

```bash
git add resp_train/metrics/__init__.py resp_train/metrics/signal.py resp_train/metrics/baseline.py scripts/baseline_tho_hilbert.py tests/test_signal_metrics.py
git commit -m "feat: add signal metrics and baseline"
```

### 任务 5：弱同步损失

**文件：**
- 创建：`resp_train/losses/__init__.py`
- 创建：`resp_train/losses/weak.py`
- 创建：`tests/test_losses.py`

- [ ] **步骤 1：编写损失测试**

```python
# tests/test_losses.py
import torch
from omegaconf import OmegaConf

from resp_train.losses.weak import WeakSyncLoss


def test_weak_sync_loss_returns_components_and_scalar():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100},
            "loss": {
                "envelope_weight": 1.0,
                "spectrum_weight": 0.2,
                "smooth_weight": 0.01,
                "envelope_window_sec": 2.0,
                "spectrum_low_hz": 0.05,
                "spectrum_high_hz": 0.7,
            },
        }
    )
    loss_fn = WeakSyncLoss(cfg)
    pred = torch.randn(2, 1, 1800, requires_grad=True)
    target = torch.randn(2, 1, 1800)
    total, parts = loss_fn(pred, target)
    assert total.ndim == 0
    assert set(parts) == {"envelope", "spectrum", "smooth"}
    total.backward()
    assert pred.grad is not None
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_losses.py -v`

预期：FAIL，报错包含 `No module named 'resp_train.losses'`。

- [ ] **步骤 3：实现弱同步损失**

```python
# resp_train/losses/__init__.py
"""弱同步训练损失。"""
```

```python
# resp_train/losses/weak.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


class WeakSyncLoss(torch.nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        fs = float(cfg.window.target_fs)
        window = max(1, int(float(cfg.loss.envelope_window_sec) * fs))
        self.envelope_window = window
        self.env_weight = float(cfg.loss.envelope_weight)
        self.spec_weight = float(cfg.loss.spectrum_weight)
        self.smooth_weight = float(cfg.loss.smooth_weight)
        self.low_hz = float(cfg.loss.spectrum_low_hz)
        self.high_hz = float(cfg.loss.spectrum_high_hz)
        self.fs = fs

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        env = self._envelope_loss(pred, target)
        spec = self._spectrum_loss(pred, target)
        smooth = torch.mean(torch.abs(pred[..., 1:] - pred[..., :-1]))
        total = self.env_weight * env + self.spec_weight * spec + self.smooth_weight * smooth
        return total, {"envelope": env.detach(), "spectrum": spec.detach(), "smooth": smooth.detach()}

    def _envelope_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_env = self._rms_envelope(pred)
        target_env = self._rms_envelope(target)
        pred_norm = self._zscore(pred_env)
        target_norm = self._zscore(target_env)
        corr = torch.mean(pred_norm * target_norm, dim=-1)
        return torch.mean(1.0 - corr)

    def _rms_envelope(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.envelope_window // 2
        power = x.square()
        smoothed = F.avg_pool1d(power, kernel_size=self.envelope_window, stride=1, padding=pad)
        if smoothed.shape[-1] > x.shape[-1]:
            smoothed = smoothed[..., : x.shape[-1]]
        return torch.sqrt(torch.clamp(smoothed, min=1e-8))

    def _spectrum_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dist = self._band_distribution(pred)
        target_dist = self._band_distribution(target)
        return torch.mean(torch.sum(torch.abs(pred_dist - target_dist), dim=-1))

    def _band_distribution(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)
        power = torch.fft.rfft(centered, dim=-1).abs().square().squeeze(1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.fs).to(x.device)
        mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        band = power[:, mask]
        return band / torch.clamp(band.sum(dim=-1, keepdim=True), min=1e-8)

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(dim=-1, keepdim=True)) / torch.clamp(x.std(dim=-1, keepdim=True), min=1e-6)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_losses.py -v`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add resp_train/losses/__init__.py resp_train/losses/weak.py tests/test_losses.py
git commit -m "feat: add weak sync losses"
```

### 任务 6：模型注册表与 `unet1d_tiny`

**文件：**
- 创建：`resp_train/models/__init__.py`
- 创建：`resp_train/models/registry.py`
- 创建：`resp_train/models/unet1d.py`
- 创建：`tests/test_model_registry.py`

- [ ] **步骤 1：编写模型测试**

```python
# tests/test_model_registry.py
import torch
from omegaconf import OmegaConf

from resp_train.models.registry import build_model, list_models


def test_unet1d_tiny_registered_and_preserves_shape():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "unet1d_tiny",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
            }
        }
    )
    assert "unet1d_tiny" in list_models()
    model = build_model(cfg)
    x = torch.randn(2, 1, 18000)
    y = model(x)
    assert y.shape == x.shape
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_model_registry.py -v`

预期：FAIL，报错包含 `No module named 'resp_train.models'`。

- [ ] **步骤 3：实现注册表和模型**

```python
# resp_train/models/__init__.py
"""模型注册与默认时域模型。"""
```

```python
# resp_train/models/registry.py
from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from omegaconf import DictConfig

from resp_train.models.unet1d import UNet1DTiny


_REGISTRY: dict[str, Callable[[DictConfig], nn.Module]] = {
    "unet1d_tiny": lambda cfg: UNet1DTiny(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
    )
}


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(cfg: DictConfig) -> nn.Module:
    name = str(cfg.model.name)
    if name not in _REGISTRY:
        raise KeyError(f"未知模型: {name}; 可用模型: {list_models()}")
    return _REGISTRY[name](cfg)
```

```python
# resp_train/models/unet1d.py
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet1DTiny(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.down1 = nn.Conv1d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = ConvBlock(c * 2, c * 2)
        self.down2 = nn.Conv1d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.bottleneck = ConvBlock(c * 4, c * 4)
        self.up2 = nn.ConvTranspose1d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose1d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(c * 2, c)
        self.out = nn.Conv1d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        d2 = self.up2(b)
        d2 = self._match_length(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self._match_length(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)

    @staticmethod
    def _match_length(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == ref.shape[-1]:
            return x
        if x.shape[-1] > ref.shape[-1]:
            return x[..., : ref.shape[-1]]
        pad = ref.shape[-1] - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad))
```

- [ ] **步骤 4：运行测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_model_registry.py -v`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add resp_train/models/__init__.py resp_train/models/registry.py resp_train/models/unet1d.py tests/test_model_registry.py
git commit -m "feat: add tiny 1d unet"
```

### 任务 7：run 工具与训练引擎

**文件：**
- 创建：`resp_train/utils/__init__.py`
- 创建：`resp_train/utils/run.py`
- 创建：`resp_train/engine/__init__.py`
- 创建：`resp_train/engine/train.py`
- 创建：`tests/test_engine_smoke.py`

- [ ] **步骤 1：编写训练 smoke 测试**

```python
# tests/test_engine_smoke.py
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from resp_train.engine.train import train_one_epoch
from resp_train.losses.weak import WeakSyncLoss
from resp_train.models.registry import build_model


class DictDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x = torch.randn(4, 1, 512)
        self.y = torch.randn(4, 1, 512)

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {"x": self.x[idx], "target": self.y[idx], "meta": {"dataset_row_id": idx}}


def test_train_one_epoch_returns_loss_summary():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100},
            "model": {"name": "unet1d_tiny", "in_channels": 1, "out_channels": 1, "base_channels": 4},
            "loss": {
                "envelope_weight": 1.0,
                "spectrum_weight": 0.2,
                "smooth_weight": 0.01,
                "envelope_window_sec": 0.2,
                "spectrum_low_hz": 0.05,
                "spectrum_high_hz": 0.7,
            },
        }
    )
    model = build_model(cfg)
    loss_fn = WeakSyncLoss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(DictDataset(), batch_size=2)
    summary = train_one_epoch(model, loader, loss_fn, optimizer, device=torch.device("cpu"))
    assert summary["loss"] > 0
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_engine_smoke.py -v`

预期：FAIL，报错包含 `No module named 'resp_train.engine'`。

- [ ] **步骤 3：实现 run 工具**

```python
# resp_train/utils/__init__.py
"""运行目录、日志和随机种子工具。"""
```

```python
# resp_train/utils/run.py
from __future__ import annotations

import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def create_run_dir(run_root: str | Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_root) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config(cfg: DictConfig, run_dir: Path) -> None:
    OmegaConf.save(cfg, run_dir / "config.yaml")


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("resp_train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
```

- [ ] **步骤 4：实现训练/验证函数**

```python
# resp_train/engine/__init__.py
"""训练与验证循环。"""
```

```python
# resp_train/engine/train.py
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: list[float] = []
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(device)
        target = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss, _ = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        totals.append(float(loss.detach().cpu()))
    return {"loss": float(np.mean(totals)) if totals else float("nan")}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: list[float] = []
    for batch in tqdm(loader, desc="val", leave=False):
        x = batch["x"].to(device)
        target = batch["target"].to(device)
        pred = model(x)
        loss, _ = loss_fn(pred, target)
        totals.append(float(loss.detach().cpu()))
    return {"loss": float(np.mean(totals)) if totals else float("nan")}
```

- [ ] **步骤 5：运行测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_engine_smoke.py -v`

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add resp_train/utils/__init__.py resp_train/utils/run.py resp_train/engine/__init__.py resp_train/engine/train.py tests/test_engine_smoke.py
git commit -m "feat: add training engine"
```

### 任务 8：训练、评价脚本与 run 产物

**文件：**
- 修改：`resp_train/engine/train.py`
- 创建：`resp_train/metrics/evaluate.py`
- 创建：`tests/test_eval_metrics.py`
- 创建：`scripts/train_tho_small.py`
- 创建：`scripts/eval_tho_small.py`

- [ ] **步骤 1：编写评价指标测试**

```python
# tests/test_eval_metrics.py
import numpy as np
from omegaconf import OmegaConf

from resp_train.metrics.evaluate import evaluate_prediction_dict


def test_evaluate_prediction_dict_returns_window_metrics():
    fs = 100
    t = np.arange(0, 60, 1 / fs)
    target = np.sin(2 * np.pi * 0.25 * t).astype(np.float32)
    preds = {
        "r_tho_hat": target.reshape(1, 1, -1),
        "tho_ref": target.reshape(1, 1, -1),
        "dataset_row_id": np.asarray([1]),
        "split": np.asarray(["val"]),
        "input_set": np.asarray(["mixed_zscore"]),
        "residual_quality_class": np.asarray(["near_zero_residual"]),
    }
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": fs},
            "loss": {"envelope_window_sec": 2.0, "spectrum_low_hz": 0.05, "spectrum_high_hz": 0.7},
        }
    )

    frame = evaluate_prediction_dict(preds, cfg, method="model")

    assert frame.loc[0, "method"] == "model"
    assert frame.loc[0, "dataset_row_id"] == 1
    assert frame.loc[0, "rr_spec_abs_error"] < 1.0
    assert frame.loc[0, "spectrum_similarity"] > 0.99
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_eval_metrics.py -v`

预期：FAIL，报错包含 `No module named 'resp_train.metrics.evaluate'`。

- [ ] **步骤 3：实现评价指标函数**

```python
# resp_train/metrics/evaluate.py
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from resp_train.metrics.signal import (
    estimate_peak_rate_bpm,
    estimate_spectral_rate_bpm,
    rms_envelope,
    spectrum_similarity,
)


def evaluate_prediction_dict(predictions: dict[str, np.ndarray], cfg: DictConfig, *, method: str) -> pd.DataFrame:
    fs = float(cfg.window.target_fs)
    low_hz = float(cfg.loss.spectrum_low_hz)
    high_hz = float(cfg.loss.spectrum_high_hz)
    env_window = int(fs * float(cfg.loss.envelope_window_sec))
    records: list[dict[str, Any]] = []
    preds = predictions["r_tho_hat"]
    targets = predictions["tho_ref"]
    for idx in range(preds.shape[0]):
        pred = np.asarray(preds[idx]).reshape(-1)
        target = np.asarray(targets[idx]).reshape(-1)
        pred_env = rms_envelope(pred, env_window)
        target_env = rms_envelope(target, env_window)
        pred_rr_spec = estimate_spectral_rate_bpm(pred, fs=fs, low_hz=low_hz, high_hz=high_hz)
        target_rr_spec = estimate_spectral_rate_bpm(target, fs=fs, low_hz=low_hz, high_hz=high_hz)
        pred_rr_peak = estimate_peak_rate_bpm(pred, fs=fs, distance_sec=2.0)
        target_rr_peak = estimate_peak_rate_bpm(target, fs=fs, distance_sec=2.0)
        records.append(
            {
                "method": method,
                "dataset_row_id": int(predictions["dataset_row_id"][idx]),
                "split": str(predictions.get("split", [""])[idx]),
                "input_set": str(predictions.get("input_set", [""])[idx]),
                "residual_quality_class": str(predictions.get("residual_quality_class", [""])[idx]),
                "pred_rr_spec_bpm": pred_rr_spec,
                "target_rr_spec_bpm": target_rr_spec,
                "rr_spec_abs_error": abs(pred_rr_spec - target_rr_spec),
                "pred_rr_peak_bpm": pred_rr_peak,
                "target_rr_peak_bpm": target_rr_peak,
                "rr_peak_abs_error": abs(pred_rr_peak - target_rr_peak),
                "envelope_corr": pd.Series(pred_env).corr(pd.Series(target_env)),
                "spectrum_similarity": spectrum_similarity(pred, target, fs=fs, low_hz=low_hz, high_hz=high_hz),
            }
        )
    return pd.DataFrame(records)
```

- [ ] **步骤 4：扩展训练引擎，支持 checkpoint 和带元数据的 predictions**

在 `resp_train/engine/train.py` 中追加：

```python
from pathlib import Path
from typing import Any


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_windows: int,
) -> dict[str, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    meta_records: list[dict[str, Any]] = []
    if max_windows <= 0:
        raise ValueError("max_windows 必须大于 0")
    for batch in loader:
        x = batch["x"].to(device)
        pred = model(x).cpu().numpy()
        target = batch["target"].cpu().numpy()
        preds.append(pred)
        targets.append(target)
        batch_size = pred.shape[0]
        for idx in range(batch_size):
            meta_records.append(_extract_meta(batch["meta"], idx))
        if len(meta_records) >= max_windows:
            break
    if not preds:
        raise RuntimeError("没有可收集的预测窗口")
    pred_arr = np.concatenate(preds, axis=0)[:max_windows]
    target_arr = np.concatenate(targets, axis=0)[:max_windows]
    meta_records = meta_records[:max_windows]
    return {
        "r_tho_hat": pred_arr,
        "tho_ref": target_arr,
        "dataset_row_id": np.asarray([int(m["dataset_row_id"]) for m in meta_records], dtype=np.int64),
        "split": np.asarray([str(m.get("split", "")) for m in meta_records]),
        "input_set": np.asarray([str(m.get("input_set", "")) for m in meta_records]),
        "residual_quality_class": np.asarray([str(m.get("residual_quality_class", "")) for m in meta_records]),
    }


def _extract_meta(meta: dict[str, Any], idx: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in meta.items():
        if torch.is_tensor(value):
            item = value[idx].item()
        elif isinstance(value, (list, tuple)):
            item = value[idx]
        else:
            item = value
        result[key] = item
    return result
```

- [ ] **步骤 5：实现训练入口**

```python
# scripts/train_tho_small.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from resp_train.config import check_required_packages, load_config
from resp_train.data.audit import add_usable_flag, summarize_audit
from resp_train.data.dataset import RespWindowDataset
from resp_train.data.index import filter_index, read_index
from resp_train.engine.train import collect_predictions, save_checkpoint, train_one_epoch, validate
from resp_train.losses.weak import WeakSyncLoss
from resp_train.metrics.baseline import run_baseline
from resp_train.metrics.evaluate import evaluate_prediction_dict
from resp_train.models.registry import build_model
from resp_train.utils.run import create_run_dir, resolve_device, save_config, set_seed, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="训练胸带参考小规模模型")
    parser.add_argument("--config", default="configs/tho_small.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，例如 data.max_train_windows=16")
    args = parser.parse_args()

    missing = check_required_packages()
    if missing:
        raise SystemExit(f"缺少依赖: {missing}; 请先确认是否安装。")

    cfg = load_config(args.config, overrides=args.overrides)
    run_dir = create_run_dir(cfg.outputs.run_root)
    save_config(cfg, run_dir)
    logger = setup_logger(run_dir)
    set_seed(int(cfg.training.seed))
    device = resolve_device(str(cfg.training.device))
    logger.info("device=%s", device)

    df = read_index(cfg.data.dataset_root, cfg.data.index_csv)
    audited = add_usable_flag(df, cfg)
    summarize_audit(audited).to_csv(run_dir / "audit.csv", index=False)
    baseline_frame = run_baseline(cfg, run_dir / "baseline_metrics.csv")
    logger.info("baseline_windows=%s", len(baseline_frame))

    train_rows = filter_index(audited, cfg, split=cfg.data.train_split, max_windows=cfg.data.max_train_windows)
    val_rows = filter_index(audited, cfg, split=cfg.data.val_split, max_windows=cfg.data.max_val_windows)
    index_path = Path(cfg.data.dataset_root) / cfg.data.index_csv
    train_ds = RespWindowDataset(index_path, train_rows, cfg, preload_windows=bool(cfg.data.preload_windows))
    val_ds = RespWindowDataset(index_path, val_rows, cfg, preload_windows=bool(cfg.data.preload_windows))
    logger.info("train_windows=%s val_windows=%s", len(train_ds), len(val_ds))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("可训练或可验证窗口为空，请检查 filter_unusable、input_set 和 split 配置。")

    train_loader = DataLoader(train_ds, batch_size=int(cfg.training.batch_size), shuffle=True, num_workers=int(cfg.training.num_workers))
    val_loader = DataLoader(val_ds, batch_size=int(cfg.training.batch_size), shuffle=False, num_workers=int(cfg.training.num_workers))

    model = build_model(cfg).to(device)
    loss_fn = WeakSyncLoss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.learning_rate))
    history_records = []
    best_loss = float("inf")
    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device=device)
        val_metrics = validate(model, val_loader, loss_fn, device=device)
        record = {"epoch": epoch, "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"]}
        history_records.append(record)
        logger.info("epoch=%s train_loss=%.6f val_loss=%.6f", epoch, record["train_loss"], record["val_loss"])
        if record["val_loss"] < best_loss:
            best_loss = record["val_loss"]
            save_checkpoint(run_dir / "checkpoint.pt", model=model, optimizer=optimizer, epoch=epoch, metrics=record)

    pd.DataFrame(history_records).to_csv(run_dir / "train_history.csv", index=False)
    eval_preds = collect_predictions(model, val_loader, device=device, max_windows=len(val_ds))
    evaluate_prediction_dict(eval_preds, cfg, method=str(cfg.model.name)).to_csv(run_dir / "metrics.csv", index=False)
    diag_preds = collect_predictions(model, val_loader, device=device, max_windows=int(cfg.outputs.max_prediction_windows))
    np.savez(run_dir / "predictions.npz", **diag_preds)
    print(run_dir)


if __name__ == "__main__":
    main()
```

- [ ] **步骤 6：实现评价入口**

```python
# scripts/eval_tho_small.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from resp_train.config import load_config
from resp_train.data.audit import add_usable_flag
from resp_train.data.dataset import RespWindowDataset
from resp_train.data.index import filter_index, read_index
from resp_train.engine.train import collect_predictions
from resp_train.metrics.evaluate import evaluate_prediction_dict
from resp_train.models.registry import build_model
from resp_train.utils.run import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="用 checkpoint 生成验证预测和指标")
    parser.add_argument("--config", default="configs/tho_small.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    device = resolve_device(str(cfg.training.device))
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    df = read_index(cfg.data.dataset_root, cfg.data.index_csv)
    audited = add_usable_flag(df, cfg)
    val_rows = filter_index(audited, cfg, split=cfg.data.val_split, max_windows=cfg.data.max_val_windows)
    dataset = RespWindowDataset(Path(cfg.data.dataset_root) / cfg.data.index_csv, val_rows, cfg, preload_windows=False)
    loader = DataLoader(dataset, batch_size=int(cfg.training.batch_size), shuffle=False, num_workers=0)
    preds = collect_predictions(model, loader, device=device, max_windows=int(cfg.outputs.max_prediction_windows))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **preds)
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        evaluate_prediction_dict(preds, cfg, method=str(cfg.model.name)).to_csv(metrics_path, index=False)
    print(f"写出预测: {output}")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 7：运行全测试**

运行：`./.venv/bin/python -m pytest tests -v`

预期：PASS。

- [ ] **步骤 8：运行小规模训练 smoke test**

运行：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_small.yaml \
  --set data.max_train_windows=16 \
  --set data.max_val_windows=8 \
  --set training.epochs=1 \
  --set model.base_channels=4 \
  --set outputs.max_prediction_windows=8
```

预期：退出码 0，输出 run 目录路径；目录中存在 `config.yaml`、`checkpoint.pt`、`train.log`、`audit.csv`、`baseline_metrics.csv`、`train_history.csv`、`metrics.csv`、`predictions.npz`。

- [ ] **步骤 9：Commit**

```bash
git add resp_train/engine/train.py resp_train/metrics/evaluate.py tests/test_eval_metrics.py scripts/train_tho_small.py scripts/eval_tho_small.py
git commit -m "feat: add tho small training scripts"
```

### 任务 9：最终验证与使用文档

**文件：**
- 创建：`docs/tho_small_training.md`

- [ ] **步骤 1：创建使用文档**

```markdown
# 胸带小规模训练使用说明

## 范围

首版只训练 `r_tho_hat(t)`，不训练 RR 头，也不把 RR 作为损失项。RR/主频、峰谷读数、频谱相似度和 envelope corr 仅作为评价指标。

## 默认数据

- 数据根目录：`/mnt/disk_code/marques/dataset/RespPairs/20260530_tho_ramp5_stage2_1`
- 索引：`training/dataset_index.csv`
- 默认输入集：`mixed_zscore`
- 默认窗口：180 秒，100 Hz，18000 点

## 快速 smoke test

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_small.yaml \
  --set data.max_train_windows=16 \
  --set data.max_val_windows=8 \
  --set training.epochs=1 \
  --set model.base_channels=4 \
  --set outputs.max_prediction_windows=8
```

## 主要产物

每次运行输出到 `runs/tho_small/<timestamp>/`：

- `config.yaml`：本次运行的完整配置快照。
- `audit.csv`：按 split、input_set、residual_quality_class 分组的数据可用性摘要。
- `baseline_metrics.csv`：val 子集平凡基线指标。
- `train_history.csv`：每轮训练和验证损失。
- `metrics.csv`：模型窗口级评价指标。
- `checkpoint.pt`：验证损失最优 checkpoint。
- `predictions.npz`：少量诊断窗口预测，不作为完整归档。

## 说明

BCG 原始信号的频谱主峰法和峰谷检测法只用于诊断；质量较好的片段可能能达到这些规则要求，质量较差的片段不应被规则法失败直接判为研究失败。
```

- [ ] **步骤 2：运行依赖检查**

运行：

```bash
./.venv/bin/python - <<'PY'
from resp_train.config import check_required_packages
missing = check_required_packages()
print("missing=", missing)
PY
```

预期：`missing= []`。若有缺失，停止并报告包名、用途和建议安装命令，等待用户确认。

- [ ] **步骤 3：运行全量测试**

运行：`./.venv/bin/python -m pytest tests -v`

预期：PASS。

- [ ] **步骤 4：运行审计和基线 smoke test**

运行：

```bash
./.venv/bin/python scripts/audit_tho_dataset.py --config configs/tho_small.yaml --output /tmp/tho_audit.csv
./.venv/bin/python scripts/baseline_tho_hilbert.py --config configs/tho_small.yaml --output /tmp/baseline_metrics.csv
```

预期：两个命令退出码 0，并写出 CSV。基线指标差不作为失败。

- [ ] **步骤 5：运行小规模训练**

运行：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_small.yaml \
  --set data.max_train_windows=16 \
  --set data.max_val_windows=8 \
  --set training.epochs=1 \
  --set model.base_channels=4 \
  --set outputs.max_prediction_windows=8
```

预期：退出码 0，生成 `runs/tho_small/<timestamp>/`，包含 `config.yaml`、`checkpoint.pt`、`train.log`、`audit.csv`、`baseline_metrics.csv`、`train_history.csv`、`metrics.csv`、`predictions.npz`。

- [ ] **步骤 6：Commit**

```bash
git add docs/tho_small_training.md
git commit -m "docs: document tho small training workflow"
```
