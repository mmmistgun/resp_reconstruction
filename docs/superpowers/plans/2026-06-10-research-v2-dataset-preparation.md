# Research v2 数据准备改造实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 `/mnt/disk_code/marques/resp_prepare` 中新增并行 Research v2 数据准备分支，以 coupling state、reference-assisted alignment 和监督可信度为核心，导出可与当前 Stage 2.1 baseline 对照的研究型 dataset index。

**架构：** 保留 Stage A-E 和现有 Stage 1.5/2.1 产物不变；新增 `resp/research_v2/` 模块和 `scripts/build_research_v2_*.py` 阶段脚本。R1 构建多表征 signal bank，R2 用稳定 lag 先验和缓慢变化约束发现 coupling state，R3 输出 raw/state-aligned 两套输入，R4 生成窗口级 confidence，R5 导出 180s research dataset index。

**技术栈：** Python、NumPy、SciPy、Pandas、PyYAML、loguru、pytest、现有 `resp_prepare` 工具模块。

---

## 执行位置

本计划的实现位置是：

```bash
cd /mnt/disk_code/marques/resp_prepare
PY=/home/marques/.conda/envs/lighting/bin/python
```

当前训练仓库只保存设计和计划文档。实现时不要覆盖现有 Stage A-E、Stage 1.5、Stage 2.1 或导出数据集；Research v2 全部写入独立目录。

## 文件结构

创建文件：

- `/mnt/disk_code/marques/resp_prepare/resp/research_v2/__init__.py`：导出 Research v2 公共函数。
- `/mnt/disk_code/marques/resp_prepare/resp/research_v2/paths.py`：统一解析 Research v2 输出目录和阶段文件名。
- `/mnt/disk_code/marques/resp_prepare/resp/research_v2/masks.py`：从 Stage B/C/D/E 产物派生 hard/soft mask。
- `/mnt/disk_code/marques/resp_prepare/resp/research_v2/signals.py`：构建 BCG/THO 多表征 signal bank。
- `/mnt/disk_code/marques/resp_prepare/resp/research_v2/lag.py`：稳健 lag 估计、平滑轨迹和歧义标记。
- `/mnt/disk_code/marques/resp_prepare/resp/research_v2/states.py`：coupling state 发现和状态摘要。
- `/mnt/disk_code/marques/resp_prepare/resp/research_v2/alignment.py`：state-level shift/drift 对齐和有效边界 mask。
- `/mnt/disk_code/marques/resp_prepare/resp/research_v2/confidence.py`：rate/phase/event/waveform/normalization/alignment confidence。
- `/mnt/disk_code/marques/resp_prepare/resp/research_v2/dataset.py`：Research v2 dataset index 生成逻辑。
- `/mnt/disk_code/marques/resp_prepare/scripts/build_research_v2_signal_bank.py`：R1 阶段入口。
- `/mnt/disk_code/marques/resp_prepare/scripts/build_research_v2_coupling_states.py`：R2 阶段入口。
- `/mnt/disk_code/marques/resp_prepare/scripts/build_research_v2_alignment.py`：R3 阶段入口。
- `/mnt/disk_code/marques/resp_prepare/scripts/build_research_v2_confidence.py`：R4 阶段入口。
- `/mnt/disk_code/marques/resp_prepare/scripts/build_research_v2_dataset.py`：R5 阶段入口。
- `/mnt/disk_code/marques/resp_prepare/tests/test_research_v2_masks.py`：mask 语义测试。
- `/mnt/disk_code/marques/resp_prepare/tests/test_research_v2_lag.py`：稳健 lag 和跳变约束测试。
- `/mnt/disk_code/marques/resp_prepare/tests/test_research_v2_states.py`：coupling state 发现测试。
- `/mnt/disk_code/marques/resp_prepare/tests/test_research_v2_alignment.py`：state alignment 测试。
- `/mnt/disk_code/marques/resp_prepare/tests/test_research_v2_confidence_dataset.py`：confidence 和 dataset index 测试。

修改文件：

- `/mnt/disk_code/marques/resp_prepare/configs/default.yaml`：新增 `research_v2` 配置节。
- `/mnt/disk_code/marques/resp_prepare/COMMANDS.md`：新增 Research v2 阶段命令和最小验证命令。
- `/mnt/disk_code/marques/resp_prepare/AGENTS.md`：新增 Research v2 分支职责、边界和禁止覆盖 baseline 的规则。
- `/mnt/disk_code/marques/resp_prepare/findings.md`：实现完成后记录 Research v2 首轮样本级诊断结论。

---

### 任务 1：新增 Research v2 配置和路径工具

**文件：**
- 修改：`/mnt/disk_code/marques/resp_prepare/configs/default.yaml`
- 创建：`/mnt/disk_code/marques/resp_prepare/resp/research_v2/__init__.py`
- 创建：`/mnt/disk_code/marques/resp_prepare/resp/research_v2/paths.py`
- 测试：`/mnt/disk_code/marques/resp_prepare/tests/test_research_v2_masks.py`

- [ ] **步骤 1：编写失败的配置测试**

创建 `tests/test_research_v2_masks.py`，先写配置和路径断言。

```python
from pathlib import Path

from resp.config import load_config
from resp.research_v2.paths import research_v2_root, stage_path


def test_research_v2_config_and_paths():
    cfg = load_config("configs/default.yaml", Path.cwd())

    assert cfg["research_v2"]["out_root"] == "artifacts/research_v2"
    assert cfg["research_v2"]["signal_bank"]["out_npz"] == "research_v2_signal_bank.npz"
    assert cfg["research_v2"]["states"]["min_state_s"] == 180
    assert cfg["research_v2"]["states"]["preferred_state_s"] == 300
    assert cfg["research_v2"]["dataset"]["window_s"] == 180

    root = research_v2_root(cfg)
    assert root == Path.cwd() / "artifacts/research_v2"
    assert stage_path(cfg, "signal_bank", 88, "out_npz") == root / "signal_bank" / "88" / "research_v2_signal_bank.npz"
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
$PY -m pytest tests/test_research_v2_masks.py::test_research_v2_config_and_paths -v
```

预期：FAIL，报错包含 `No module named 'resp.research_v2'` 或 `KeyError: 'research_v2'`。

- [ ] **步骤 3：新增 Research v2 配置**

在 `configs/default.yaml` 的 `stage_2_1_dataset` 后、`logging` 前新增：

```yaml
research_v2:
  out_root: artifacts/research_v2
  excluded_samp_ids:
    - 54
    - 582
  legacy_excluded_from_training_samp_ids:
    - 1301
  signal_bank:
    out_npz: research_v2_signal_bank.npz
    out_meta: research_v2_signal_bank.json
    summary_csv: summary_research_v2_signal_bank.csv
    bcg_rawish:
      detrend_window_s: 300.0
      detrend_decimate: 10
      bandpass_low_hz: 0.03
      bandpass_high_hz: 20.0
      filter_order: 4
    resp_band:
      low_hz: 0.05
      high_hz: 0.7
      filter_order: 4
    legacy_band:
      low_hz: 0.05
      high_hz: 0.5
      filter_order: 4
  masks:
    transient_motion_pad_s: 15
    posture_candidate_min_s: 20
    transition_guard_s: 45
    hard_invalid_long_unusable_s: 300
  lag:
    window_s: 180
    step_s: 30
    max_lag_s: 10
    min_valid_ratio: 0.75
    min_corr_peak: 0.20
    peak_margin_min: 0.03
    smooth_window_count: 5
    max_state_lag_iqr_s: 1.0
    lag_jump_ambiguous_s: 2.0
  states:
    out_npz: research_v2_coupling_states.npz
    out_meta: research_v2_coupling_states.json
    summary_csv: summary_research_v2_coupling_states.csv
    regions_csv: research_v2_coupling_state_regions.csv
    min_state_s: 180
    preferred_state_s: 300
    boundary_context_s: 150
    merge_gap_s: 60
  alignment:
    out_npz: research_v2_alignment.npz
    out_meta: research_v2_alignment.json
    summary_csv: summary_research_v2_alignment.csv
    constant_shift_iqr_max_s: 1.0
    constant_shift_abs_max_s: 8.0
    linear_drift_min_abs_s_per_hour: 0.5
    linear_drift_residual_p90_max_s: 1.0
  confidence:
    out_npz: research_v2_confidence.npz
    out_meta: research_v2_confidence.json
    summary_csv: summary_research_v2_confidence.csv
    high_threshold: 0.75
    medium_threshold: 0.45
  dataset:
    index_csv: dataset_index_research_v2.csv
    out_meta: dataset_index_research_v2.json
    summary_csv: summary_research_v2_dataset.csv
    window_s: 180
    step_s: 30
    short_window_candidates_s:
      - 60
      - 120
    min_hard_valid_ratio: 0.80
    min_alignment_valid_ratio: 0.80
    min_supervision_level: medium
```

说明：`excluded_samp_ids` 放入现有 `stage_1_3_alignment_manifest.csv` 中 `alignment_method=exclude_from_auto_alignment` 的两份样本 `54` 和 `582`。`1301` 是历史项目约束下未纳入训练候选的样本，单独保留在 `legacy_excluded_from_training_samp_ids`，不要和“人工明确排除的两份自动对齐样本”混淆。

- [ ] **步骤 4：创建路径工具**

创建 `resp/research_v2/__init__.py`：

```python
"""Research v2 数据准备工具。"""
```

创建 `resp/research_v2/paths.py`：

```python
from __future__ import annotations

from pathlib import Path
from typing import Any


def _resolve(project_root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else project_root / path


def research_v2_root(cfg: dict[str, Any]) -> Path:
    return _resolve(Path(cfg["_project_root"]), cfg["research_v2"]["out_root"])


def stage_dir(cfg: dict[str, Any], stage: str, samp_id: int | str) -> Path:
    return research_v2_root(cfg) / stage / str(samp_id)


def stage_path(cfg: dict[str, Any], stage: str, samp_id: int | str, key: str) -> Path:
    filename = cfg["research_v2"][stage][key]
    return stage_dir(cfg, stage, samp_id) / str(filename)


def summary_path(cfg: dict[str, Any], stage: str) -> Path:
    filename = cfg["research_v2"][stage]["summary_csv"]
    return research_v2_root(cfg) / stage / str(filename)
```

- [ ] **步骤 5：运行测试验证通过**

运行：

```bash
$PY -m pytest tests/test_research_v2_masks.py::test_research_v2_config_and_paths -v
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add configs/default.yaml resp/research_v2/__init__.py resp/research_v2/paths.py tests/test_research_v2_masks.py
git commit -m "feat: add research v2 config and paths"
```

---

### 任务 2：实现 R1 signal bank 和质量 mask 重解释

**文件：**
- 创建：`/mnt/disk_code/marques/resp_prepare/resp/research_v2/masks.py`
- 创建：`/mnt/disk_code/marques/resp_prepare/resp/research_v2/signals.py`
- 创建：`/mnt/disk_code/marques/resp_prepare/scripts/build_research_v2_signal_bank.py`
- 修改：`/mnt/disk_code/marques/resp_prepare/tests/test_research_v2_masks.py`

- [ ] **步骤 1：编写 mask 语义测试**

追加到 `tests/test_research_v2_masks.py`：

```python
import numpy as np

from resp.research_v2.masks import derive_research_masks


def test_derive_research_masks_keeps_transient_motion_soft():
    stage_b = {
        "tho_extreme_motion_sec": np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8),
        "bcg_extreme_motion_sec": np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8),
        "tho_unusable_sec": np.zeros(6, dtype=np.uint8),
        "bcg_unusable_sec": np.zeros(6, dtype=np.uint8),
    }
    stage_c = {
        "tho_bad_sec": np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8),
        "bcg_bad_sec": np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8),
    }
    stage_d = {
        "tho_motion_sec": np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8),
        "bcg_motion_sec": np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8),
        "tho_low_quality_sec": np.zeros(6, dtype=np.uint8),
        "bcg_low_quality_sec": np.zeros(6, dtype=np.uint8),
    }

    masks = derive_research_masks(stage_b, stage_c, stage_d, n_sec=6, transient_pad_s=0)

    assert masks["transient_motion_sec"].tolist() == [0, 1, 1, 0, 0, 0]
    assert masks["hard_invalid_sec"].sum() == 0
    assert masks["normalization_unreliable_sec"].tolist() == [0, 1, 1, 0, 0, 0]
    assert masks["amplitude_unreliable_sec"].tolist() == [0, 1, 1, 0, 0, 0]
```

- [ ] **步骤 2：编写 signal bank 测试**

创建最小合成测试到同一文件：

```python
from resp.research_v2.signals import build_signal_bank_arrays


def test_build_signal_bank_arrays_contains_expected_keys():
    fs = 100
    t = np.arange(fs * 20, dtype=np.float32) / fs
    stage_a = {
        "bcg_100": np.sin(2 * np.pi * 0.25 * t).astype(np.float32),
        "tho_100": np.cos(2 * np.pi * 0.25 * t).astype(np.float32),
    }
    stage_c = {
        "bcg_on_axis": stage_a["bcg_100"],
        "tho_on_axis": stage_a["tho_100"],
    }
    stage_e = {
        "tho_normalized_zscore": stage_a["tho_100"],
    }

    arrays = build_signal_bank_arrays(stage_a, stage_c, stage_e, fs)

    assert set(arrays) >= {
        "bcg_rawish_wideband",
        "bcg_resp_band_0p05_0p7",
        "bcg_legacy_on_axis_0p05_0p5",
        "tho_event_phase_ref",
        "tho_waveform_ref",
        "tho_rate_ref",
    }
    assert arrays["bcg_resp_band_0p05_0p7"].shape == stage_a["bcg_100"].shape
```

- [ ] **步骤 3：运行测试验证失败**

```bash
$PY -m pytest tests/test_research_v2_masks.py -v
```

预期：FAIL，报错包含 `No module named` 或函数未定义。

- [ ] **步骤 4：实现 mask 派生**

创建 `resp/research_v2/masks.py`：

```python
from __future__ import annotations

import numpy as np


def _bool_sec(arrays: dict[str, np.ndarray], key: str, n_sec: int) -> np.ndarray:
    raw = np.asarray(arrays.get(key, np.zeros(n_sec, dtype=np.uint8)), dtype=bool)
    out = np.zeros(n_sec, dtype=bool)
    keep = min(n_sec, raw.size)
    out[:keep] = raw[:keep]
    return out


def _pad(mask: np.ndarray, pad_s: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if pad_s <= 0 or not arr.any():
        return arr.copy()
    out = arr.copy()
    idx = np.flatnonzero(arr)
    for i in idx:
        lo = max(0, int(i) - int(pad_s))
        hi = min(arr.size, int(i) + int(pad_s) + 1)
        out[lo:hi] = True
    return out


def derive_research_masks(
    stage_b: dict[str, np.ndarray],
    stage_c: dict[str, np.ndarray],
    stage_d: dict[str, np.ndarray],
    *,
    n_sec: int,
    transient_pad_s: int,
    hard_invalid_long_unusable_s: int = 300,
) -> dict[str, np.ndarray]:
    tho_unusable = _bool_sec(stage_b, "tho_unusable_sec", n_sec)
    bcg_unusable = _bool_sec(stage_b, "bcg_unusable_sec", n_sec)
    tho_em = _bool_sec(stage_b, "tho_extreme_motion_sec", n_sec)
    bcg_em = _bool_sec(stage_b, "bcg_extreme_motion_sec", n_sec)
    tho_motion = _bool_sec(stage_d, "tho_motion_sec", n_sec)
    bcg_motion = _bool_sec(stage_d, "bcg_motion_sec", n_sec)
    tho_lowq = _bool_sec(stage_d, "tho_low_quality_sec", n_sec)
    bcg_lowq = _bool_sec(stage_d, "bcg_low_quality_sec", n_sec)

    hard_invalid = tho_unusable | bcg_unusable
    transient_motion = _pad(tho_em | bcg_em | tho_motion | bcg_motion, transient_pad_s)
    amplitude_unreliable = transient_motion | tho_lowq | bcg_lowq
    normalization_unreliable = transient_motion | hard_invalid
    posture_candidate = transient_motion.copy()

    return {
        "hard_invalid_sec": hard_invalid.astype(np.uint8),
        "transient_motion_sec": transient_motion.astype(np.uint8),
        "posture_transition_candidate_sec": posture_candidate.astype(np.uint8),
        "amplitude_unreliable_sec": amplitude_unreliable.astype(np.uint8),
        "normalization_unreliable_sec": normalization_unreliable.astype(np.uint8),
    }
```

- [ ] **步骤 5：实现 signal bank 构建**

创建 `resp/research_v2/signals.py`：

```python
from __future__ import annotations

from typing import Any

import numpy as np

from resp import base_layer


def build_signal_bank_arrays(
    stage_a: dict[str, np.ndarray],
    stage_c: dict[str, np.ndarray],
    stage_e: dict[str, np.ndarray],
    fs: int,
    *,
    rawish_low_hz: float = 0.03,
    rawish_high_hz: float = 20.0,
    resp_low_hz: float = 0.05,
    resp_high_hz: float = 0.7,
    legacy_low_hz: float = 0.05,
    legacy_high_hz: float = 0.5,
    order: int = 4,
) -> dict[str, np.ndarray]:
    bcg_100 = np.asarray(stage_a["bcg_100"], dtype=np.float32)
    tho_on_axis = np.asarray(stage_c["tho_on_axis"], dtype=np.float32)
    bcg_on_axis = np.asarray(stage_c["bcg_on_axis"], dtype=np.float32)
    n = min(bcg_100.size, tho_on_axis.size, bcg_on_axis.size)
    bcg_100 = bcg_100[:n]
    tho_on_axis = tho_on_axis[:n]
    bcg_on_axis = bcg_on_axis[:n]

    bcg_rawish = base_layer.zero_phase_bandpass(bcg_100, fs, rawish_low_hz, rawish_high_hz, order)
    bcg_resp = base_layer.zero_phase_bandpass(bcg_100, fs, resp_low_hz, resp_high_hz, order)
    bcg_legacy = base_layer.zero_phase_bandpass(bcg_on_axis, fs, legacy_low_hz, legacy_high_hz, order)
    tho_event = base_layer.zero_phase_bandpass(tho_on_axis, fs, resp_low_hz, resp_high_hz, order)
    if "tho_normalized_zscore" in stage_e:
        tho_waveform = np.asarray(stage_e["tho_normalized_zscore"], dtype=np.float32)[:n]
    else:
        tho_waveform = tho_event.copy()

    return {
        "bcg_rawish_wideband": bcg_rawish.astype(np.float32, copy=False),
        "bcg_resp_band_0p05_0p7": bcg_resp.astype(np.float32, copy=False),
        "bcg_legacy_on_axis_0p05_0p5": bcg_legacy.astype(np.float32, copy=False),
        "tho_event_phase_ref": tho_event.astype(np.float32, copy=False),
        "tho_waveform_ref": tho_waveform.astype(np.float32, copy=False),
        "tho_rate_ref": tho_event.astype(np.float32, copy=False),
    }
```

- [ ] **步骤 6：实现 R1 脚本**

创建 `scripts/build_research_v2_signal_bank.py`，复用现有 CSV/NPZ 写法。脚本必须支持 `--sample-ids` 和 `--skip-existing`。

```python
"""Research v2 R1：构建多表征 signal bank 和多级质量 mask。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from resp import io_utils  # noqa: E402
from resp.config import load_config  # noqa: E402
from resp.research_v2.masks import derive_research_masks  # noqa: E402
from resp.research_v2.paths import stage_path, summary_path  # noqa: E402
from resp.research_v2.signals import build_signal_bank_arrays  # noqa: E402


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def main() -> int:
    parser = argparse.ArgumentParser(description="Research v2 R1 signal bank")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample-ids", nargs="*", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, _PROJECT_ROOT)
    fs = int(cfg["timebase"]["target_fs"])
    out_root = Path(cfg["paths"]["out_root"])
    norm_root = Path(cfg["paths"]["normalized_root"])
    sample_ids = args.sample_ids or io_utils.scan_samp_ids(Path(cfg["paths"]["raw_root"]))
    rows = []
    for sid in sorted(set(sample_ids)):
        out_npz = stage_path(cfg, "signal_bank", sid, "out_npz")
        out_meta = stage_path(cfg, "signal_bank", sid, "out_meta")
        if args.skip_existing and out_npz.exists():
            rows.append({"samp_id": sid, "status": "skipped_existing", "out_npz": str(out_npz)})
            continue
        base_dir = out_root / str(sid)
        try:
            stage_a = _load_npz(base_dir / "stage_a.npz")
            stage_b = _load_npz(base_dir / "stage_b.npz")
            stage_c = _load_npz(base_dir / "stage_c.npz")
            stage_d = _load_npz(base_dir / "stage_d.npz")
            stage_e = _load_npz(norm_root / str(sid) / "stage_e.npz")
            arrays = build_signal_bank_arrays(stage_a, stage_c, stage_e, fs)
            n_sec = min(v.size for v in arrays.values()) // fs
            masks = derive_research_masks(
                stage_b,
                stage_c,
                stage_d,
                n_sec=n_sec,
                transient_pad_s=int(cfg["research_v2"]["masks"]["transient_motion_pad_s"]),
                hard_invalid_long_unusable_s=int(cfg["research_v2"]["masks"]["hard_invalid_long_unusable_s"]),
            )
            io_utils.save_npz(out_npz, **arrays, **masks)
            io_utils.save_json(out_meta, {"samp_id": sid, "stage": "research_v2_signal_bank", "target_fs": fs, "outputs": {"npz": str(out_npz)}})
            rows.append({"samp_id": sid, "status": "ok", "duration_s": n_sec, "out_npz": str(out_npz), "out_meta": str(out_meta)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"samp_id": sid, "status": "error", "reason": str(exc)})

    summary = summary_path(cfg, "signal_bank")
    summary.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with summary.open("w", newline="", encoding="gbk") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(row["status"] in {"ok", "skipped_existing"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **步骤 7：运行测试和 smoke**

```bash
$PY -m pytest tests/test_research_v2_masks.py -v
$PY scripts/build_research_v2_signal_bank.py --sample-ids 88 220 1478
```

预期：测试 PASS；脚本为 `88/220/1478` 写出类似 `artifacts/research_v2/signal_bank/88/research_v2_signal_bank.npz` 的样本级文件和 summary CSV，`88/220/1478` 均为 `status=ok`。

- [ ] **步骤 8：Commit**

```bash
git add resp/research_v2/masks.py resp/research_v2/signals.py scripts/build_research_v2_signal_bank.py tests/test_research_v2_masks.py
git commit -m "feat: build research v2 signal bank"
```

---

### 任务 3：实现稳健 lag 估计和跳变约束

**文件：**
- 创建：`/mnt/disk_code/marques/resp_prepare/resp/research_v2/lag.py`
- 测试：`/mnt/disk_code/marques/resp_prepare/tests/test_research_v2_lag.py`

- [ ] **步骤 1：编写失败的 lag 测试**

创建 `tests/test_research_v2_lag.py`：

```python
import numpy as np

from resp.research_v2.lag import robust_state_lag_summary, smooth_lag_track


def test_smooth_lag_track_suppresses_single_window_jump():
    lag = np.array([0.2, 0.25, 0.2, 4.8, 0.3, 0.25, 0.2], dtype=np.float32)
    corr = np.ones_like(lag, dtype=np.float32) * 0.8
    status = np.zeros_like(lag, dtype=np.int8)

    out = smooth_lag_track(lag, corr, status, smooth_window_count=3, lag_jump_ambiguous_s=2.0)

    assert np.nanmax(np.abs(out["smoothed_lag_s"][:3] - 0.2)) < 0.1
    assert out["lag_ambiguous"][3] == 1
    assert abs(float(np.nanmedian(out["smoothed_lag_s"])) - 0.25) < 0.1


def test_robust_state_lag_summary_uses_stable_windows():
    lag = np.array([0.1, 0.2, 0.15, 3.0, 0.2, 0.1], dtype=np.float32)
    ambiguous = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)

    summary = robust_state_lag_summary(lag, ambiguous)

    assert abs(summary["lag_median_s"] - 0.15) < 0.05
    assert summary["lag_iqr_s"] < 0.2
    assert summary["lag_ambiguous_ratio"] == 1 / 6
```

- [ ] **步骤 2：运行测试验证失败**

```bash
$PY -m pytest tests/test_research_v2_lag.py -v
```

预期：FAIL，模块不存在。

- [ ] **步骤 3：实现稳健 lag 工具**

创建 `resp/research_v2/lag.py`：

```python
from __future__ import annotations

import numpy as np


STATUS_OK = 0


def _rolling_nanmedian(x: np.ndarray, win: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    half = max(0, int(win) // 2)
    for i in range(arr.size):
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        vals = arr[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[i] = np.float32(np.median(vals.astype(np.float64, copy=False)))
    return out


def smooth_lag_track(
    lag_s: np.ndarray,
    corr_peak: np.ndarray,
    lag_status: np.ndarray,
    *,
    smooth_window_count: int,
    lag_jump_ambiguous_s: float,
) -> dict[str, np.ndarray]:
    lag = np.asarray(lag_s, dtype=np.float32)
    status = np.asarray(lag_status, dtype=np.int16)
    usable = np.isfinite(lag)
    base = np.where(usable, lag, np.nan).astype(np.float32)
    smoothed = _rolling_nanmedian(base, max(1, int(smooth_window_count)))
    jump = np.isfinite(lag) & np.isfinite(smoothed) & (np.abs(lag - smoothed) >= float(lag_jump_ambiguous_s))
    ambiguous = jump | (status != STATUS_OK)
    stable_lag = smoothed.copy()
    stable_lag[ambiguous & np.isfinite(smoothed)] = smoothed[ambiguous & np.isfinite(smoothed)]
    return {
        "smoothed_lag_s": stable_lag.astype(np.float32, copy=False),
        "lag_ambiguous": ambiguous.astype(np.uint8),
    }


def robust_state_lag_summary(lag_s: np.ndarray, lag_ambiguous: np.ndarray) -> dict[str, float]:
    lag = np.asarray(lag_s, dtype=np.float64)
    amb = np.asarray(lag_ambiguous, dtype=bool)
    keep = np.isfinite(lag) & (~amb)
    vals = lag[keep]
    if vals.size == 0:
        vals = lag[np.isfinite(lag)]
    if vals.size == 0:
        return {"lag_median_s": float("nan"), "lag_iqr_s": float("nan"), "lag_ambiguous_ratio": float(amb.mean()) if amb.size else 0.0}
    q25, q75 = np.nanpercentile(vals, [25, 75])
    return {
        "lag_median_s": float(np.nanmedian(vals)),
        "lag_iqr_s": float(q75 - q25),
        "lag_ambiguous_ratio": float(amb.mean()) if amb.size else 0.0,
    }
```

- [ ] **步骤 4：运行测试验证通过**

```bash
$PY -m pytest tests/test_research_v2_lag.py -v
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add resp/research_v2/lag.py tests/test_research_v2_lag.py
git commit -m "feat: add robust research v2 lag tools"
```

---

### 任务 4：实现 R2 coupling state 发现

**文件：**
- 创建：`/mnt/disk_code/marques/resp_prepare/resp/research_v2/states.py`
- 创建：`/mnt/disk_code/marques/resp_prepare/scripts/build_research_v2_coupling_states.py`
- 测试：`/mnt/disk_code/marques/resp_prepare/tests/test_research_v2_states.py`

- [ ] **步骤 1：编写失败的状态发现测试**

创建 `tests/test_research_v2_states.py`：

```python
import numpy as np

from resp.research_v2.states import discover_coupling_states


def test_discover_coupling_states_does_not_split_single_lag_jump():
    window_start_s = np.arange(0, 900, 30, dtype=np.float32)
    window_end_s = window_start_s + 180
    window_center_s = window_start_s + 90
    smoothed_lag_s = np.ones_like(window_start_s, dtype=np.float32) * 0.2
    lag_ambiguous = np.zeros_like(window_start_s, dtype=np.uint8)
    lag_ambiguous[10] = 1
    masks = {
        "hard_invalid_sec": np.zeros(1200, dtype=np.uint8),
        "posture_transition_candidate_sec": np.zeros(1200, dtype=np.uint8),
    }

    states = discover_coupling_states(
        window_start_s,
        window_end_s,
        window_center_s,
        smoothed_lag_s,
        lag_ambiguous,
        masks,
        min_state_s=180,
        preferred_state_s=300,
        merge_gap_s=60,
    )

    assert len(states) == 1
    assert states[0]["state_start_s"] == 0
    assert states[0]["state_end_s"] >= 900
    assert states[0]["state_lag_ambiguous_ratio"] > 0


def test_discover_coupling_states_splits_confirmed_posture_change():
    window_start_s = np.arange(0, 1200, 30, dtype=np.float32)
    window_end_s = window_start_s + 180
    window_center_s = window_start_s + 90
    smoothed_lag_s = np.where(window_center_s < 600, 0.1, 1.2).astype(np.float32)
    lag_ambiguous = np.zeros_like(window_start_s, dtype=np.uint8)
    posture = np.zeros(1500, dtype=np.uint8)
    posture[590:650] = 1
    masks = {
        "hard_invalid_sec": np.zeros(1500, dtype=np.uint8),
        "posture_transition_candidate_sec": posture,
    }

    states = discover_coupling_states(
        window_start_s,
        window_end_s,
        window_center_s,
        smoothed_lag_s,
        lag_ambiguous,
        masks,
        min_state_s=180,
        preferred_state_s=300,
        merge_gap_s=60,
    )

    assert len(states) == 2
    assert states[0]["state_lag_median_s"] < 0.5
    assert states[1]["state_lag_median_s"] > 0.8
```

- [ ] **步骤 2：运行测试验证失败**

```bash
$PY -m pytest tests/test_research_v2_states.py -v
```

预期：FAIL，函数未定义。

- [ ] **步骤 3：实现状态发现核心函数**

创建 `resp/research_v2/states.py`：

```python
from __future__ import annotations

import numpy as np

from resp.research_v2.lag import robust_state_lag_summary


def _state_row(state_id: int, start_s: int, end_s: int, lag: np.ndarray, ambiguous: np.ndarray, reason: str) -> dict[str, float | int | str]:
    summary = robust_state_lag_summary(lag, ambiguous)
    return {
        "coupling_state_id": int(state_id),
        "state_start_s": int(start_s),
        "state_end_s": int(end_s),
        "state_duration_s": int(max(0, end_s - start_s)),
        "state_lag_median_s": summary["lag_median_s"],
        "state_lag_iqr_s": summary["lag_iqr_s"],
        "state_lag_ambiguous_ratio": summary["lag_ambiguous_ratio"],
        "state_reason": reason,
    }


def discover_coupling_states(
    window_start_s: np.ndarray,
    window_end_s: np.ndarray,
    window_center_s: np.ndarray,
    smoothed_lag_s: np.ndarray,
    lag_ambiguous: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    min_state_s: int,
    preferred_state_s: int,
    merge_gap_s: int,
    lag_change_threshold_s: float = 0.8,
) -> list[dict[str, float | int | str]]:
    starts = np.asarray(window_start_s, dtype=np.float64)
    ends = np.asarray(window_end_s, dtype=np.float64)
    centers = np.asarray(window_center_s, dtype=np.float64)
    lag = np.asarray(smoothed_lag_s, dtype=np.float32)
    amb = np.asarray(lag_ambiguous, dtype=np.uint8)
    if starts.size == 0:
        return []

    posture = np.asarray(masks.get("posture_transition_candidate_sec", np.zeros(int(np.nanmax(ends)) + 1, dtype=np.uint8)), dtype=bool)
    hard = np.asarray(masks.get("hard_invalid_sec", np.zeros_like(posture, dtype=np.uint8)), dtype=bool)
    candidate_boundaries: list[int] = []
    for sec in np.flatnonzero(posture):
        if not candidate_boundaries or int(sec) - candidate_boundaries[-1] > int(merge_gap_s):
            candidate_boundaries.append(int(sec))

    boundaries = [int(np.nanmin(starts))]
    for boundary_s in candidate_boundaries:
        pre = (centers >= boundary_s - preferred_state_s) & (centers < boundary_s)
        post = (centers >= boundary_s) & (centers < boundary_s + preferred_state_s)
        pre_vals = lag[pre & np.isfinite(lag)]
        post_vals = lag[post & np.isfinite(lag)]
        if pre_vals.size == 0 or post_vals.size == 0:
            continue
        if abs(float(np.nanmedian(post_vals)) - float(np.nanmedian(pre_vals))) >= lag_change_threshold_s:
            boundaries.append(boundary_s)
    boundaries.append(int(np.nanmax(ends)))
    boundaries = sorted(set(boundaries))

    states: list[dict[str, float | int | str]] = []
    state_id = 0
    for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=False):
        if hi - lo < min_state_s:
            continue
        inside = (centers >= lo) & (centers < hi)
        if not inside.any():
            continue
        invalid_ratio = float(hard[lo:hi].mean()) if hi <= hard.size else 0.0
        if invalid_ratio >= 0.8:
            continue
        state_id += 1
        reason = "posture_confirmed_lag_change" if lo != boundaries[0] else "initial_state"
        states.append(_state_row(state_id, lo, hi, lag[inside], amb[inside], reason))
    if not states:
        states.append(_state_row(1, int(np.nanmin(starts)), int(np.nanmax(ends)), lag, amb, "fallback_single_state"))
    return states
```

- [ ] **步骤 4：实现 R2 脚本**

创建 `scripts/build_research_v2_coupling_states.py`。脚本读取 R1 signal bank，调用现有 `resp.align_lag.run_lag_diagnostics` 计算逐窗 lag，再用 `smooth_lag_track` 和 `discover_coupling_states` 输出 R2 产物。

核心保存字段必须包含：

```python
io_utils.save_npz(
    out_npz,
    window_start_s=result.window_start_s,
    window_end_s=result.window_end_s,
    window_center_s=result.window_center_s,
    raw_lag_s=result.lag_s,
    smoothed_lag_s=lag_track["smoothed_lag_s"],
    lag_ambiguous=lag_track["lag_ambiguous"],
    corr_peak=result.corr_peak,
    corr_peak_margin=result.corr_peak_margin,
    coupling_state_id_sec=state_id_sec.astype(np.int32),
)
```

CSV 区域字段必须包含：

```text
samp_id,coupling_state_id,state_start_s,state_end_s,state_duration_s,
state_lag_median_s,state_lag_iqr_s,state_lag_ambiguous_ratio,state_reason
```

- [ ] **步骤 5：运行测试和 sample smoke**

```bash
$PY -m pytest tests/test_research_v2_lag.py tests/test_research_v2_states.py -v
$PY scripts/build_research_v2_signal_bank.py --sample-ids 88 220 956 1478 --skip-existing
$PY scripts/build_research_v2_coupling_states.py --sample-ids 88 220 956 1478
```

预期：测试 PASS；脚本为 `88/220/956/1478` 写出类似 `artifacts/research_v2/coupling_states/88/research_v2_coupling_states.npz` 的样本级文件，且 summary 中 `88/220/956/1478` 有至少 1 个 state。检查 `956` 不因单窗 lag 跳变生成大量短状态。

- [ ] **步骤 6：Commit**

```bash
git add resp/research_v2/states.py scripts/build_research_v2_coupling_states.py tests/test_research_v2_states.py
git commit -m "feat: discover research v2 coupling states"
```

---

### 任务 5：实现 R3 state-level reference-assisted alignment

**文件：**
- 创建：`/mnt/disk_code/marques/resp_prepare/resp/research_v2/alignment.py`
- 创建：`/mnt/disk_code/marques/resp_prepare/scripts/build_research_v2_alignment.py`
- 测试：`/mnt/disk_code/marques/resp_prepare/tests/test_research_v2_alignment.py`

- [ ] **步骤 1：编写失败的 alignment 测试**

创建 `tests/test_research_v2_alignment.py`：

```python
import numpy as np

from resp.research_v2.alignment import apply_constant_shift, choose_alignment_method


def test_choose_alignment_method_constant_shift():
    state = {
        "state_lag_median_s": 0.5,
        "state_lag_iqr_s": 0.2,
        "state_duration_s": 300,
    }

    method = choose_alignment_method(state, constant_shift_iqr_max_s=1.0, constant_shift_abs_max_s=8.0)

    assert method["state_alignment_method"] == "constant_shift"
    assert method["state_alignment_lag_s"] == 0.5


def test_apply_constant_shift_marks_edge_invalid():
    x = np.arange(10, dtype=np.float32)
    shifted, valid = apply_constant_shift(x, sample_shift=2)

    assert shifted[0] == 2
    assert valid[:8].all()
    assert not valid[8:].any()
```

- [ ] **步骤 2：运行测试验证失败**

```bash
$PY -m pytest tests/test_research_v2_alignment.py -v
```

预期：FAIL，模块不存在。

- [ ] **步骤 3：实现 alignment 工具**

创建 `resp/research_v2/alignment.py`：

```python
from __future__ import annotations

import numpy as np


def choose_alignment_method(
    state: dict[str, float | int | str],
    *,
    constant_shift_iqr_max_s: float,
    constant_shift_abs_max_s: float,
) -> dict[str, float | str | int]:
    lag = float(state.get("state_lag_median_s", float("nan")))
    iqr = float(state.get("state_lag_iqr_s", float("nan")))
    if np.isfinite(lag) and np.isfinite(iqr) and iqr <= constant_shift_iqr_max_s and abs(lag) <= constant_shift_abs_max_s:
        return {
            "state_alignment_method": "constant_shift",
            "state_alignment_lag_s": lag,
            "state_alignment_drift_s_per_hour": 0.0,
            "state_alignment_is_reference_assisted": 1,
        }
    return {
        "state_alignment_method": "none",
        "state_alignment_lag_s": 0.0,
        "state_alignment_drift_s_per_hour": 0.0,
        "state_alignment_is_reference_assisted": 1,
    }


def apply_constant_shift(x: np.ndarray, sample_shift: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float32)
    src = np.arange(arr.size, dtype=np.int64) + int(sample_shift)
    valid = (src >= 0) & (src < arr.size)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    out[valid] = arr[src[valid]]
    return out, valid
```

- [ ] **步骤 4：实现 R3 脚本**

创建 `scripts/build_research_v2_alignment.py`。脚本读取 R1 signal bank 和 R2 states，对 `bcg_rawish_wideband`、`bcg_resp_band_0p05_0p7`、`bcg_legacy_on_axis_0p05_0p5` 逐 state 应用同一个 shift，输出 raw/timebase-aligned 和 state-aligned 版本。

输出 NPZ 必须包含：

```text
bcg_rawish_wideband_to_tho_timebase
bcg_resp_band_to_tho_timebase
bcg_legacy_on_axis_to_tho_timebase
bcg_rawish_wideband_state_aligned
bcg_resp_band_state_aligned
bcg_legacy_on_axis_state_aligned
state_alignment_valid_sec
state_alignment_edge_invalid_sec
coupling_state_id_sec
```

输出 CSV 必须包含：

```text
samp_id,coupling_state_id,state_alignment_method,state_alignment_lag_s,
state_alignment_drift_s_per_hour,state_alignment_is_reference_assisted,
state_alignment_valid_ratio
```

- [ ] **步骤 5：运行测试和 smoke**

```bash
$PY -m pytest tests/test_research_v2_alignment.py -v
$PY scripts/build_research_v2_alignment.py --sample-ids 88 220 956 1478
```

预期：测试 PASS；R3 summary 中至少部分 state 为 `constant_shift`，`state_alignment_valid_ratio` 大于 `0.8`。

- [ ] **步骤 6：Commit**

```bash
git add resp/research_v2/alignment.py scripts/build_research_v2_alignment.py tests/test_research_v2_alignment.py
git commit -m "feat: add research v2 state alignment"
```

---

### 任务 6：实现 R4 supervision confidence

**文件：**
- 创建：`/mnt/disk_code/marques/resp_prepare/resp/research_v2/confidence.py`
- 创建：`/mnt/disk_code/marques/resp_prepare/scripts/build_research_v2_confidence.py`
- 测试：`/mnt/disk_code/marques/resp_prepare/tests/test_research_v2_confidence_dataset.py`

- [ ] **步骤 1：编写失败的 confidence 测试**

创建 `tests/test_research_v2_confidence_dataset.py`：

```python
from resp.research_v2.confidence import confidence_level, combine_task_confidence


def test_confidence_level_thresholds():
    assert confidence_level(0.8, medium_threshold=0.45, high_threshold=0.75) == "high"
    assert confidence_level(0.5, medium_threshold=0.45, high_threshold=0.75) == "medium"
    assert confidence_level(0.2, medium_threshold=0.45, high_threshold=0.75) == "low"


def test_combine_task_confidence_uses_task_specific_minimums():
    scores = {
        "rate_confidence_score": 0.9,
        "phase_confidence_score": 0.7,
        "event_confidence_score": 0.6,
        "waveform_confidence_score": 0.8,
        "normalization_confidence_score": 0.3,
        "alignment_confidence_score": 0.7,
    }

    combined = combine_task_confidence(scores)

    assert combined["rate_task_confidence"] == 0.9
    assert combined["phase_task_confidence"] == 0.7
    assert combined["waveform_task_confidence"] == 0.3
```

- [ ] **步骤 2：运行测试验证失败**

```bash
$PY -m pytest tests/test_research_v2_confidence_dataset.py -v
```

预期：FAIL，模块不存在。

- [ ] **步骤 3：实现 confidence 工具**

创建 `resp/research_v2/confidence.py`：

```python
from __future__ import annotations

import numpy as np


def confidence_level(score: float, *, medium_threshold: float, high_threshold: float) -> str:
    value = float(score)
    if value >= float(high_threshold):
        return "high"
    if value >= float(medium_threshold):
        return "medium"
    return "low"


def score_from_error(error: float, good: float, bad: float) -> float:
    if not np.isfinite(error):
        return 0.0
    if error <= good:
        return 1.0
    if error >= bad:
        return 0.0
    return float(1.0 - (error - good) / (bad - good))


def combine_task_confidence(scores: dict[str, float]) -> dict[str, float]:
    rate = float(scores.get("rate_confidence_score", 0.0))
    phase = min(float(scores.get("alignment_confidence_score", 0.0)), float(scores.get("phase_confidence_score", 0.0)))
    event = min(float(scores.get("alignment_confidence_score", 0.0)), float(scores.get("event_confidence_score", 0.0)))
    waveform = min(
        float(scores.get("alignment_confidence_score", 0.0)),
        float(scores.get("waveform_confidence_score", 0.0)),
        float(scores.get("normalization_confidence_score", 0.0)),
    )
    return {
        "rate_task_confidence": rate,
        "phase_task_confidence": phase,
        "event_task_confidence": event,
        "waveform_task_confidence": waveform,
        "supervision_confidence_score": max(rate, phase, event, waveform),
    }
```

- [ ] **步骤 4：实现 R4 脚本**

创建 `scripts/build_research_v2_confidence.py`。第一版使用状态和窗口统计生成保守 confidence：

- `alignment_confidence_score`：由 state lag IQR、ambiguous ratio、alignment valid ratio 组成。
- `rate_confidence_score`：由 BCG/THO 呼吸频段谱峰接近程度和峰清晰度组成。
- `phase_confidence_score`：由 smoothed lag 稳定性和 corr margin 组成。
- `event_confidence_score`：第一版使用 phase confidence 的保守副本，并在 meta 中记录 `event_detection_mode=phase_proxy_v1`。
- `waveform_confidence_score`：由 band-limited correlation 组成。
- `normalization_confidence_score`：由 transient motion、posture transition、normalization unreliable ratio 组成。

输出 NPZ 至少包含每个 180s/30s 窗口对应数组：

```text
window_start_s
window_end_s
coupling_state_id
rate_confidence_score
phase_confidence_score
event_confidence_score
waveform_confidence_score
normalization_confidence_score
alignment_confidence_score
supervision_confidence_score
```

输出 CSV 使用同名字段，并增加所有 `*_level`。

- [ ] **步骤 5：运行测试和 smoke**

```bash
$PY -m pytest tests/test_research_v2_confidence_dataset.py -v
$PY scripts/build_research_v2_confidence.py --sample-ids 88 220 956 1478
```

预期：测试 PASS；R4 summary 显示每个样本都有窗口级 confidence，且 confidence score 在 `[0, 1]`。

- [ ] **步骤 6：Commit**

```bash
git add resp/research_v2/confidence.py scripts/build_research_v2_confidence.py tests/test_research_v2_confidence_dataset.py
git commit -m "feat: estimate research v2 supervision confidence"
```

---

### 任务 7：实现 R5 research dataset index

**文件：**
- 创建：`/mnt/disk_code/marques/resp_prepare/resp/research_v2/dataset.py`
- 创建：`/mnt/disk_code/marques/resp_prepare/scripts/build_research_v2_dataset.py`
- 修改：`/mnt/disk_code/marques/resp_prepare/tests/test_research_v2_confidence_dataset.py`

- [ ] **步骤 1：编写失败的 dataset 测试**

追加到 `tests/test_research_v2_confidence_dataset.py`：

```python
from resp.research_v2.dataset import allowed_losses_from_scores


def test_allowed_losses_from_scores_keeps_rate_only_window():
    row = {
        "rate_confidence_level": "medium",
        "phase_confidence_level": "low",
        "event_confidence_level": "low",
        "waveform_confidence_level": "low",
    }

    assert allowed_losses_from_scores(row) == "rate"


def test_allowed_losses_from_scores_includes_waveform_when_supported():
    row = {
        "rate_confidence_level": "high",
        "phase_confidence_level": "high",
        "event_confidence_level": "medium",
        "waveform_confidence_level": "medium",
    }

    assert allowed_losses_from_scores(row) == "rate;phase;event;waveform"
```

- [ ] **步骤 2：运行测试验证失败**

```bash
$PY -m pytest tests/test_research_v2_confidence_dataset.py -v
```

预期：FAIL，函数未定义。

- [ ] **步骤 3：实现 dataset 工具**

创建 `resp/research_v2/dataset.py`：

```python
from __future__ import annotations


def _is_medium_or_high(level: str) -> bool:
    return str(level) in {"medium", "high"}


def allowed_losses_from_scores(row: dict[str, str]) -> str:
    losses: list[str] = []
    if _is_medium_or_high(row.get("rate_confidence_level", "")):
        losses.append("rate")
    if _is_medium_or_high(row.get("phase_confidence_level", "")):
        losses.append("phase")
    if _is_medium_or_high(row.get("event_confidence_level", "")):
        losses.append("event")
    if _is_medium_or_high(row.get("waveform_confidence_level", "")):
        losses.append("waveform")
    return ";".join(losses)
```

- [ ] **步骤 4：实现 R5 脚本**

创建 `scripts/build_research_v2_dataset.py`。脚本读取 R1/R3/R4 输出，按 `window_s=180`、`step_s=30` 生成不能跨 coupling state 的窗口。

输出字段必须包含：

```text
dataset_row_id
split
samp_id
coupling_state_id
window_start_s
window_end_s
window_duration_s
source_npz
bcg_input_key
bcg_input_aligned_key
target_event_phase_key
target_waveform_key
target_rate_key
hard_valid_ratio
state_alignment_valid_ratio
transient_motion_ratio
posture_transition_ratio
amplitude_reliable_ratio
normalization_reliable_ratio
rate_confidence_score
rate_confidence_level
phase_confidence_score
phase_confidence_level
event_confidence_score
event_confidence_level
waveform_confidence_score
waveform_confidence_level
alignment_confidence_score
alignment_confidence_level
supervision_confidence_score
supervision_confidence_level
state_alignment_method
state_alignment_lag_s
state_alignment_drift_s_per_hour
state_alignment_is_reference_assisted
allowed_losses
reason
```

split 规则第一版复用现有 Stage 2.1 `dataset_index.csv` 的 `samp_id -> split` 映射，保证 baseline 和 Research v2 可比。

- [ ] **步骤 5：运行测试和 smoke**

```bash
$PY -m pytest tests/test_research_v2_confidence_dataset.py -v
$PY scripts/build_research_v2_dataset.py --sample-ids 88 220 956 1478
```

预期：测试 PASS；`artifacts/research_v2/dataset/dataset_index_research_v2.csv` 存在；所有行的 `allowed_losses` 非空；没有窗口跨 `coupling_state_id`。

- [ ] **步骤 6：Commit**

```bash
git add resp/research_v2/dataset.py scripts/build_research_v2_dataset.py tests/test_research_v2_confidence_dataset.py
git commit -m "feat: export research v2 dataset index"
```

---

### 任务 8：全链路最小样本验证和文档更新

**文件：**
- 修改：`/mnt/disk_code/marques/resp_prepare/COMMANDS.md`
- 修改：`/mnt/disk_code/marques/resp_prepare/AGENTS.md`
- 修改：`/mnt/disk_code/marques/resp_prepare/findings.md`

- [ ] **步骤 1：运行全链路最小样本**

```bash
$PY scripts/build_research_v2_signal_bank.py --sample-ids 88 220 956 1478
$PY scripts/build_research_v2_coupling_states.py --sample-ids 88 220 956 1478
$PY scripts/build_research_v2_alignment.py --sample-ids 88 220 956 1478
$PY scripts/build_research_v2_confidence.py --sample-ids 88 220 956 1478
$PY scripts/build_research_v2_dataset.py --sample-ids 88 220 956 1478
```

预期：所有命令退出码为 0。

- [ ] **步骤 2：检查产物数量**

运行：

```bash
$PY - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("artifacts/research_v2")
for rel in [
    "signal_bank/summary_research_v2_signal_bank.csv",
    "coupling_states/summary_research_v2_coupling_states.csv",
    "alignment/summary_research_v2_alignment.csv",
    "confidence/summary_research_v2_confidence.csv",
    "dataset/dataset_index_research_v2.csv",
]:
    path = root / rel
    df = pd.read_csv(path, encoding="gbk")
    print(rel, df.shape)
PY
```

预期：五个 CSV 均存在；dataset index 行数大于 0。

- [ ] **步骤 3：检查 lag 跳变不会产生大量短状态**

运行：

```bash
$PY - <<'PY'
from pathlib import Path
import pandas as pd
regions = pd.read_csv("artifacts/research_v2/coupling_states/research_v2_coupling_state_regions.csv", encoding="gbk")
print(regions.groupby("samp_id").size())
short = regions[regions["state_duration_s"] < 180]
print("short_lt_180", len(short))
assert len(short) == 0
PY
```

预期：没有 `<180s` 的训练 state；`956` 这类 lag 不稳定样本不会被切成大量短状态。

- [ ] **步骤 4：更新 `COMMANDS.md`**

新增 Research v2 命令段：

```md
## Research v2 数据准备

最小样本 smoke：

```bash
$PY scripts/build_research_v2_signal_bank.py --sample-ids 88 220 956 1478
$PY scripts/build_research_v2_coupling_states.py --sample-ids 88 220 956 1478
$PY scripts/build_research_v2_alignment.py --sample-ids 88 220 956 1478
$PY scripts/build_research_v2_confidence.py --sample-ids 88 220 956 1478
$PY scripts/build_research_v2_dataset.py --sample-ids 88 220 956 1478
```

Research v2 产物写入 `artifacts/research_v2/`，不会覆盖 Stage A-E、Stage 1.5 或 Stage 2.1 baseline。
```

- [ ] **步骤 5：更新 `AGENTS.md`**

在项目规则中增加：

```md
- Research v2 是并行研究分支，产物必须写入 `artifacts/research_v2/`，禁止覆盖现有 Stage A-E、Stage 1.5、Stage 2.1 和导出数据集。
- Research v2 中现有 Stage 1.2-1.4 人工结论只作为先验；除明确排除样本外，其余样本允许重新估计 coupling state 和监督可信度。
- 逐窗 lag 不能直接作为切段真值；必须通过稳健化、缓慢变化约束和体动/转身上下文确认状态边界。
```

- [ ] **步骤 6：更新 `findings.md`**

记录最小样本 smoke 的结果：

运行以下命令生成可粘贴记录：

```bash
$PY - <<'PY'
from pathlib import Path
import pandas as pd
dataset = pd.read_csv("artifacts/research_v2/dataset/dataset_index_research_v2.csv", encoding="gbk")
regions = pd.read_csv("artifacts/research_v2/coupling_states/research_v2_coupling_state_regions.csv", encoding="gbk")
state_count_956 = int((regions["samp_id"] == 956).sum())
print("### Research v2 最小样本 smoke")
print("")
print("- 样本：`88, 220, 956, 1478`")
print("- 产物根目录：`artifacts/research_v2/`")
print(f"- R1-R5 均完成，dataset index 行数为 `{len(dataset)}`")
print(f"- coupling state 不允许由单个 lag 跳变直接切段；`956` 的状态数为 `{state_count_956}`")
print("- 该分支不覆盖 Stage 2.1 baseline。")
PY
```

将命令输出追加到 `findings.md` 的“近期诊断与决策”下。

- [ ] **步骤 7：运行最终验证**

```bash
$PY -m pytest tests/test_research_v2_masks.py tests/test_research_v2_lag.py tests/test_research_v2_states.py tests/test_research_v2_alignment.py tests/test_research_v2_confidence_dataset.py -v
$PY -m py_compile \
  scripts/build_research_v2_signal_bank.py \
  scripts/build_research_v2_coupling_states.py \
  scripts/build_research_v2_alignment.py \
  scripts/build_research_v2_confidence.py \
  scripts/build_research_v2_dataset.py \
  resp/research_v2/*.py
```

预期：pytest 全部 PASS；py_compile 无输出且退出码 0。

- [ ] **步骤 8：Commit**

```bash
git add COMMANDS.md AGENTS.md findings.md
git commit -m "docs: document research v2 dataset workflow"
```

---

## 全量执行前检查

在执行全量样本前运行：

```bash
git status --short
$PY scripts/build_research_v2_signal_bank.py --sample-ids 88 220 956 1478 --skip-existing
$PY scripts/build_research_v2_coupling_states.py --sample-ids 88 220 956 1478
$PY scripts/build_research_v2_alignment.py --sample-ids 88 220 956 1478
$PY scripts/build_research_v2_confidence.py --sample-ids 88 220 956 1478
$PY scripts/build_research_v2_dataset.py --sample-ids 88 220 956 1478
```

检查内容：

- Research v2 产物只出现在 `artifacts/research_v2/`。
- 现有 `artifacts/stage_2_1_dataset/dataset_index.csv` 未被修改。
- `dataset_index_research_v2.csv` 至少包含 raw 和 state-aligned 输入键。
- 每行 `allowed_losses` 非空。
- `rate` 窗口数不少于 `waveform` 窗口数。

## 计划覆盖度自检

- Stage A-E 保留和重解释：任务 1、2、8 覆盖。
- R1 signal bank：任务 2 覆盖。
- 质量 mask 多语义：任务 2 覆盖。
- R2 稳定 lag 先验、缓慢变化约束、允许推翻旧人工结论：任务 3、4、8 覆盖。
- R3 raw/state-aligned 输入和 reference-assisted metadata：任务 5 覆盖。
- R4 连续分数与离散等级：任务 6 覆盖。
- R5 180s research dataset index 和 allowed losses：任务 7 覆盖。
- 最小样本验证和文档沉淀：任务 8 覆盖。
