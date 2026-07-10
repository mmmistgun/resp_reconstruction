# BCG 呼吸带二次谐波显著窗口分层分析实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 建立一条可复现的离线分析链：在验证集冻结 BCG 二次谐波窗口定义，在测试集固定分层，比较四个 G 系列模型的任务表现与谐波纠正率，并生成阶段汇报材料。

**架构：** 将信号计算和标签规则放入独立的 `resp_train.analysis.second_harmonic` 模块；命令行脚本只负责加载 research v2 窗口、冻结/应用阈值、连接现有 metrics 和导出选定窗口预测。窗口标签完全由输入 BCG 与目标 THO 产生，模型输出只参与后续纠正率统计，确保所有模型使用同一分层。

**技术栈：** Python 3.11、NumPy、SciPy、pandas、PyTorch、OmegaConf、Matplotlib、pytest。

---

## 文件结构

- 创建 `resp_train/analysis/__init__.py`：公开二次谐波分析的稳定数据类型和入口。
- 创建 `resp_train/analysis/second_harmonic.py`：滤波、Welch 特征、覆盖状态、阈值验证、标签与模型纠正状态；不做文件 I/O。
- 创建 `scripts/analyze_bcg_second_harmonic.py`：从 research v2 数据加载输入/目标，执行 `discover`、`freeze`、`apply` 和 `summarize-metrics`。
- 创建 `scripts/export_harmonic_predictions.py`：读取冻结后的阳性 row id 和 checkpoint spec，仅对选定测试窗口推理并保存预测。
- 创建 `scripts/plot_bcg_second_harmonic.py`：绘制验证阈值复核图及测试代表案例波形/频谱图。
- 修改 `scripts/stratified_eval_analysis.py`：增加按外部 `dataset_row_id -> stratum` 标签汇总的公共入口，复用现有 paired metric 逻辑。
- 修改 `scripts/README.md`：记录完整工作流、CPU 命令、用户执行的 GPU 命令和产物语义。
- 创建 `tests/test_second_harmonic_analysis.py`：核心信号、边界和标签规则测试。
- 创建 `tests/test_analyze_bcg_second_harmonic.py`：数据回连、阈值冻结保护、覆盖率与 metrics 汇总测试。
- 创建 `tests/test_export_harmonic_predictions.py`：dry-run、选定 row、checkpoint spec 与 NPZ schema 测试。
- 修改 `tests/test_stratified_eval_analysis.py`：外部分层 paired 汇总回归测试。
- 最终修改 `docs/stage_reports/2026-07-08-tho-research-v2-group-meeting-stage-report.md`：写入实际覆盖率、四模型结果、证据边界和图表索引。

所有运行产物写入新目录 `runs/bcg_second_harmonic_20260710/`，不覆盖现有 metrics、checkpoint、日志或图表。

### 任务 1：建立纯函数信号特征与标签核心

**文件：**
- 创建：`resp_train/analysis/__init__.py`
- 创建：`resp_train/analysis/second_harmonic.py`
- 创建：`tests/test_second_harmonic_analysis.py`

- [ ] **步骤 1：编写合成信号特征的失败测试**

```python
def test_extract_harmonic_features_detects_dominant_second_harmonic():
    fs = 100.0
    t = np.arange(18000) / fs
    f0 = 0.20
    tho = np.sin(2 * np.pi * f0 * t)
    bcg = 0.35 * np.sin(2 * np.pi * f0 * t) + np.sin(2 * np.pi * 2 * f0 * t)

    cfg = HarmonicFeatureConfig(
        fs=fs,
        low_hz=0.05,
        high_hz=0.7,
        filter_order=4,
        welch_nperseg=4096,
        neighborhood_hz=0.02,
    )
    result = extract_harmonic_features(bcg, tho, cfg=cfg)

    assert result.status == "eligible"
    assert abs(result.bcg_peak_hz / result.tho_reference_hz - 2.0) < 0.08
    assert result.harmonic_to_fundamental_ratio > 1.0
    assert result.harmonic_band_fraction > 0.4
```

同时加入：纯基频 BCG 的比值较低、`2*f_tho > 0.7` 返回
`second_harmonic_out_of_band`、稳健 RR 与频谱 RR 冲突返回
`tho_reference_unstable`、全零 BCG 不产生无穷值、非有限输入抛出 `ValueError`。

- [ ] **步骤 2：运行核心测试确认失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_second_harmonic_analysis.py -q
```

预期：FAIL，提示 `resp_train.analysis.second_harmonic` 不存在。

- [ ] **步骤 3：实现不可变配置和结果类型**

```python
@dataclass(frozen=True)
class HarmonicFeatureConfig:
    fs: float = 100.0
    low_hz: float = 0.05
    high_hz: float = 0.7
    filter_order: int = 4
    welch_nperseg: int = 4096
    neighborhood_hz: float = 0.02
    energy_floor: float = 1e-12
    tho_rr_agreement_bpm: float = 1.0


@dataclass(frozen=True)
class HarmonicThresholds:
    version: str
    tho_rr_agreement_bpm: float
    peak_relative_tolerance: float
    harmonic_to_fundamental_min: float
    harmonic_band_fraction_min: float
    correction_ratio_drop_min: float


@dataclass(frozen=True)
class HarmonicFeatures:
    status: str
    tho_reference_hz: float
    tho_robust_rr_bpm: float
    tho_spectral_rr_bpm: float
    bcg_peak_hz: float
    peak_to_tho_ratio: float
    peak_second_harmonic_relative_error: float
    fundamental_energy: float
    second_harmonic_energy: float
    band_energy: float
    harmonic_to_fundamental_ratio: float
    harmonic_band_fraction: float
```

配置构造时检查采样率、频带、Welch 长度、邻域、能量下限和 THO 一致性容差均有效。

- [ ] **步骤 4：实现滤波、THO 一致性与 Welch 邻域能量**

```python
def bandpass_resp(signal: np.ndarray, cfg: HarmonicFeatureConfig) -> np.ndarray:
    x = _as_finite_1d(signal)
    sos = scipy_signal.butter(
        cfg.filter_order,
        [cfg.low_hz, cfg.high_hz],
        btype="bandpass",
        fs=cfg.fs,
        output="sos",
    )
    return scipy_signal.sosfiltfilt(sos, x)


def neighborhood_energy(freqs, power, center_hz, half_width_hz):
    mask = np.abs(freqs - center_hz) <= half_width_hz
    if not mask.any():
        return 0.0
    return float(np.trapezoid(power[mask], freqs[mask])) if mask.sum() > 1 else float(power[mask].sum())
```

`extract_harmonic_features` 复用当前 `estimate_robust_peak_rate_bpm` 与频谱 RR 语义，先检查 THO
参考一致性和二倍频带宽，再计算 BCG 主峰、`E1`、`E2`、`E_band`。比值分母使用
`max(E1, energy_floor)`，并保留原始能量。

- [ ] **步骤 5：实现冻结标签和模型纠正状态**

```python
def classify_harmonic_window(features, thresholds):
    if features.status != "eligible":
        return features.status
    peak = features.peak_second_harmonic_relative_error <= thresholds.peak_relative_tolerance
    prominent = (
        features.harmonic_to_fundamental_ratio >= thresholds.harmonic_to_fundamental_min
        and features.harmonic_band_fraction >= thresholds.harmonic_band_fraction_min
    )
    if peak and prominent:
        return "strong_harmonic"
    if peak:
        return "peak_doubling"
    if prominent:
        return "harmonic_prominent"
    return "harmonic_negative"
```

`classify_model_correction` 仅接收已经判为阳性的输入特征和模型输出特征：主峰回到 THO 基频
容差内且谐波比下降超过 `correction_ratio_drop_min` 为 `corrected`；只满足下降为
`partially_corrected`；其余为 `not_corrected`。

- [ ] **步骤 6：运行核心测试确认通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_second_harmonic_analysis.py -q
```

预期：全部 PASS，且无 divide-by-zero warning。

- [ ] **步骤 7：提交核心模块**

```bash
git add resp_train/analysis/__init__.py resp_train/analysis/second_harmonic.py tests/test_second_harmonic_analysis.py
git commit -m "分析: 添加BCG二次谐波特征与标签"
```

### 任务 2：实现 research v2 窗口特征发现和阈值冻结

**文件：**
- 创建：`scripts/analyze_bcg_second_harmonic.py`
- 创建：`tests/test_analyze_bcg_second_harmonic.py`

- [ ] **步骤 1：编写窗口回连和 discover 输出失败测试**

测试构造一个最小 research v2 数据包，其中验证集包含纯基频、强二次谐波、超出带宽和 THO
不稳定窗口；调用 `discover_features` 后断言：

```python
assert set(features["dataset_row_id"]) == {1, 2, 3, 4}
assert features.loc[features.dataset_row_id == 2, "status"].item() == "eligible"
assert features.loc[features.dataset_row_id == 3, "status"].item() == "second_harmonic_out_of_band"
assert {"samp_id", "split", "window_start_s", "source_npz"} <= set(features.columns)
```

再断言 duplicate `dataset_row_id`、错 split、切片长度不符或缺少信号键会显式失败。

- [ ] **步骤 2：运行脚本测试确认失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_analyze_bcg_second_harmonic.py -q
```

预期：FAIL，提示目标脚本或 `discover_features` 不存在。

- [ ] **步骤 3：实现 `discover` 子命令和逐窗口 CSV**

CLI：

```bash
./.venv/bin/python scripts/analyze_bcg_second_harmonic.py discover \
  --config configs/tho_research_v2.yaml \
  --split val \
  --output-dir runs/bcg_second_harmonic_20260710/validation
```

实现必须通过 `read_research_v2_index` 和 `ResearchV2WindowDataset` 读取模型实际 `x` 与
`target`，不得直接读取备用呼吸带字段。输出：

- `validation_harmonic_features.csv`
- `validation_distribution_summary.csv`
- `proposed_harmonic_thresholds.json`
- `analysis_manifest.json`

proposal JSON 使用固定、可复现的候选生成规则：THO 两类 RR 一致性容差候选为
`[0.5, 1.0, 2.0] bpm`，倍频相对容差候选为 `[0.05, 0.10, 0.15]`，能量阈值候选来自在对应
THO 容差下可判定验证窗口的 `p75/p85/p90`，纠正最小下降候选为 `[0.10, 0.20, 0.30]`。
文件状态必须是 `"proposal"`，不能被 `apply` 接受。

- [ ] **步骤 4：实现 `freeze` 子命令的防误用约束**

冻结命令接收 proposal JSON 中一个完整候选组合的 `candidate_id`，写出新的不可变文件：

```bash
./.venv/bin/python scripts/analyze_bcg_second_harmonic.py freeze \
  --proposal runs/bcg_second_harmonic_20260710/validation/proposed_harmonic_thresholds.json \
  --candidate-id candidate_000 \
  --output runs/bcg_second_harmonic_20260710/harmonic_thresholds.json
```

输出记录 `status="frozen"`、源 proposal SHA-256、候选全部数值、验证集路径、创建时间和人工复核
备注。若输出已存在则失败，除非用户显式提供 `--allow-identical-existing` 且内容哈希一致；绝不提供
覆盖不同阈值的开关。

- [ ] **步骤 5：增加阈值生命周期测试**

```python
def test_apply_rejects_proposal_threshold_file(tmp_path):
    proposal = tmp_path / "proposal.json"
    proposal.write_text('{"status":"proposal"}', encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        load_frozen_thresholds(proposal)
```

同时测试冻结文件已存在且不同、未知 candidate id、proposal 哈希缺失、test split 不能运行
proposal 生成。

- [ ] **步骤 6：运行脚本测试确认通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_analyze_bcg_second_harmonic.py -q
```

预期：全部 PASS。

- [ ] **步骤 7：提交特征发现与冻结流程**

```bash
git add scripts/analyze_bcg_second_harmonic.py tests/test_analyze_bcg_second_harmonic.py
git commit -m "分析: 添加二次谐波阈值冻结流程"
```

### 任务 3：生成验证集人工复核图并冻结真实阈值

**文件：**
- 创建：`scripts/plot_bcg_second_harmonic.py`
- 修改：`tests/test_analyze_bcg_second_harmonic.py`

- [ ] **步骤 1：编写分层抽样和绘图失败测试**

测试输入 30 行 feature CSV，断言每个候选指标分别选择高值、阈值附近和低值对照，row id 不重复；
使用 Agg 后端生成 PNG，并断言图包含 BCG/THO 呼吸带时域与频谱两个面板。

- [ ] **步骤 2：运行绘图测试确认失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_analyze_bcg_second_harmonic.py -q
```

预期：FAIL，提示 `plot_review_cases` 不存在。

- [ ] **步骤 3：实现验证复核图命令**

```bash
./.venv/bin/python scripts/plot_bcg_second_harmonic.py validation-review \
  --config configs/tho_research_v2.yaml \
  --features runs/bcg_second_harmonic_20260710/validation/validation_harmonic_features.csv \
  --proposal runs/bcg_second_harmonic_20260710/validation/proposed_harmonic_thresholds.json \
  --output-dir runs/bcg_second_harmonic_20260710/validation/figures \
  --cases-per-group 6
```

每张图标题显示 row id、subject、THO 两类 RR、BCG/THO 主频、倍频偏差、`E2/E1` 和
`E2/E_band`。同时写 `review_case_manifest.csv`，记录抽样组和图路径。

- [ ] **步骤 4：运行测试和真实验证集 discover**

运行：

```bash
./.venv/bin/python -m pytest tests/test_second_harmonic_analysis.py tests/test_analyze_bcg_second_harmonic.py -q
./.venv/bin/python scripts/analyze_bcg_second_harmonic.py discover --config configs/tho_research_v2.yaml --split val --output-dir runs/bcg_second_harmonic_20260710/validation
```

预期：测试 PASS；真实运行写出完整 validation 特征、分布、proposal 和 manifest，不需要 GPU。

- [ ] **步骤 5：生成人工复核图并暂停选择候选**

运行上一步 `validation-review` 命令。检查高值组确实呈现双峰/二次谐波，边界组用于选择候选，
低值组不应系统性出现同类形态。把选择理由写入独立文本
`runs/bcg_second_harmonic_20260710/validation/threshold_review.md`，然后使用任务 2 的 `freeze`
命令冻结被批准的 `candidate_id`。此处是必须的人工检查点；没有冻结 JSON 不进入测试集。

- [ ] **步骤 6：提交绘图实现，不提交派生运行产物**

```bash
git add scripts/plot_bcg_second_harmonic.py tests/test_analyze_bcg_second_harmonic.py
git commit -m "分析: 添加二次谐波阈值复核图"
```

### 任务 4：固定应用测试标签并汇总覆盖率

**文件：**
- 修改：`scripts/analyze_bcg_second_harmonic.py`
- 修改：`tests/test_analyze_bcg_second_harmonic.py`

- [ ] **步骤 1：编写 apply 和覆盖率失败测试**

```python
assert set(labels["stratum"]) == {
    "strong_harmonic",
    "peak_doubling",
    "harmonic_prominent",
    "harmonic_negative",
    "tho_reference_unstable",
    "second_harmonic_out_of_band",
}
assert coverage.loc[coverage.status == "eligible_total", "n_windows"].item() == 4
assert coverage["n_subjects"].notna().all()
```

测试还必须证明测试特征不会写回或改变 `harmonic_thresholds.json` 的字节内容。

- [ ] **步骤 2：运行目标测试确认失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_analyze_bcg_second_harmonic.py -q
```

预期：FAIL，提示 `apply_frozen_thresholds` 或 coverage 输出未实现。

- [ ] **步骤 3：实现 `apply` 子命令**

```bash
./.venv/bin/python scripts/analyze_bcg_second_harmonic.py apply \
  --config configs/tho_research_v2.yaml \
  --split test \
  --thresholds runs/bcg_second_harmonic_20260710/harmonic_thresholds.json \
  --output-dir runs/bcg_second_harmonic_20260710/test
```

输出 `test_harmonic_labels.csv`、`coverage_summary.csv` 和 `analysis_manifest.json`。coverage 至少包含
总窗口、THO 不稳定、二倍频出带、eligible、每个子层、合并阳性层的窗口数/比例/受试者数。
合并阳性定义固定为 `strong_harmonic | peak_doubling | harmonic_prominent`。

- [ ] **步骤 4：运行测试和测试集 apply**

运行：

```bash
./.venv/bin/python -m pytest tests/test_analyze_bcg_second_harmonic.py -q
./.venv/bin/python scripts/analyze_bcg_second_harmonic.py apply --config configs/tho_research_v2.yaml --split test --thresholds runs/bcg_second_harmonic_20260710/harmonic_thresholds.json --output-dir runs/bcg_second_harmonic_20260710/test
```

预期：测试 PASS；测试集产物使用冻结文件哈希，且 coverage 行数、窗口总数、受试者总数一致。

- [ ] **步骤 5：提交测试标签实现**

```bash
git add scripts/analyze_bcg_second_harmonic.py tests/test_analyze_bcg_second_harmonic.py
git commit -m "分析: 固定应用测试集二次谐波标签"
```

### 任务 5：复用现有逐窗口 metrics 比较四个模型

**文件：**
- 修改：`scripts/stratified_eval_analysis.py`
- 修改：`tests/test_stratified_eval_analysis.py`
- 修改：`scripts/analyze_bcg_second_harmonic.py`

- [ ] **步骤 1：编写外部分层 paired 汇总失败测试**

新增 `run_external_strata_analysis` 测试：标签 CSV 只定义一次 stratum；四模型、两次训练均按
相同 row id 连接；断言输出包含每模型绝对指标、相对 `g0_time_only` 的配对 delta、每层窗口数和
跨训练标准差。缺失/重复 row id 必须失败，不能静默 inner join 丢窗口。

- [ ] **步骤 2：运行分层测试确认失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_stratified_eval_analysis.py -q
```

预期：FAIL，提示 `run_external_strata_analysis` 不存在。

- [ ] **步骤 3：实现公共外部分层接口**

```python
def run_external_strata_analysis(
    spec: StratifiedAnalysisSpec,
    strata: pd.DataFrame,
    *,
    stratum_column: str = "stratum",
) -> dict[str, pd.DataFrame]:
    required = {"dataset_row_id", stratum_column}
    missing = required - set(strata.columns)
    if missing:
        raise ValueError(f"外部分层标签缺少列: {sorted(missing)}")
    if strata["dataset_row_id"].duplicated().any():
        raise ValueError("外部分层标签存在重复 dataset_row_id")
    return _external_strata_frames(spec, strata, stratum_column=stratum_column)
```

复用 `_joined_metrics`、`_paired_record` 和 `_aggregate_seed_rows`。派生
`breath_count_zero_cross_bpm_error` 后汇总当前五项核心指标；同时保留 robust RR 的
median/p95/frac>1、count bpm 的 median/p95，以及每层 `n_subjects`。

- [ ] **步骤 4：实现 `summarize-metrics` 子命令**

```bash
./.venv/bin/python scripts/analyze_bcg_second_harmonic.py summarize-metrics \
  --labels runs/bcg_second_harmonic_20260710/test/test_harmonic_labels.csv \
  --eval-root runs/test_eval_g_series_20260709_local_rr_canonical \
  --dataset-index /mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/training/dataset_index.csv \
  --output-dir runs/bcg_second_harmonic_20260710/model_metrics
```

固定四个 label 和三个训练编号，或从现有 G 系列 manifest 验证后读取；不得自动吸收其他历史 run。
输出 `model_stratified_metrics_seed.csv`、`model_stratified_metrics_summary.csv`、
`paired_delta_vs_time_seed.csv`、`paired_delta_vs_time_summary.csv` 和 manifest。

- [ ] **步骤 5：运行测试与现有 metrics 汇总**

运行：

```bash
./.venv/bin/python -m pytest tests/test_stratified_eval_analysis.py tests/test_analyze_bcg_second_harmonic.py -q
./.venv/bin/python scripts/analyze_bcg_second_harmonic.py summarize-metrics --labels runs/bcg_second_harmonic_20260710/test/test_harmonic_labels.csv --eval-root runs/test_eval_g_series_20260709_local_rr_canonical --dataset-index /mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/training/dataset_index.csv --output-dir runs/bcg_second_harmonic_20260710/model_metrics
```

预期：12 个 metrics 文件全部匹配；每个模型/训练编号在同一 stratum 的 `n_windows` 一致。

- [ ] **步骤 6：提交任务指标分层**

```bash
git add scripts/stratified_eval_analysis.py tests/test_stratified_eval_analysis.py scripts/analyze_bcg_second_harmonic.py
git commit -m "分析: 按二次谐波窗口汇总四模型指标"
```

### 任务 6：准备选定窗口预测导出和用户 GPU 命令

**文件：**
- 创建：`scripts/export_harmonic_predictions.py`
- 创建：`tests/test_export_harmonic_predictions.py`

- [ ] **步骤 1：编写 dry-run 和选定 row 导出失败测试**

测试最小 checkpoint spec：dry-run 只打印 12 个任务，不加载 torch checkpoint；实际单任务用假模型后
断言 NPZ 只含标签 CSV 中合并阳性 row，且 schema 为：

```python
assert set(blob.files) == {"dataset_row_id", "r_tho_hat", "tho_ref"}
assert blob["r_tho_hat"].shape == (n_selected, 18000)
assert np.array_equal(blob["dataset_row_id"], expected_row_ids)
```

空阳性集合、checkpoint 不存在、config/checkpoint 不匹配和推理返回 row 缺失均显式失败。

- [ ] **步骤 2：运行导出测试确认失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_export_harmonic_predictions.py -q
```

预期：FAIL，提示导出脚本不存在。

- [ ] **步骤 3：实现 checkpoint spec、dry-run 和单任务推理**

脚本读取 `configs/eval_specs/g_series_test_eval_20260705.csv`，从 checkpoint 同目录解析 config，使用
research v2 index 精确选择合并阳性 row，按标签文件顺序构造 dataset。输出路径为：

```text
runs/bcg_second_harmonic_20260710/predictions/<label>_<seed>_harmonic_predictions.npz
```

每个文件旁写 JSON manifest，记录 checkpoint/config、checkpoint SHA-256、labels SHA-256、split、设备、
row 数和数组 schema。使用临时文件加 `os.replace` 原子写入；若目标存在且 manifest 哈希不一致则失败。

- [ ] **步骤 4：验证 dry-run**

运行：

```bash
./.venv/bin/python scripts/export_harmonic_predictions.py \
  --spec configs/eval_specs/g_series_test_eval_20260705.csv \
  --labels runs/bcg_second_harmonic_20260710/test/test_harmonic_labels.csv \
  --output-dir runs/bcg_second_harmonic_20260710/predictions \
  --device cuda:0 --device cuda:1 --max-parallel 4 --dry-run
```

预期：列出 12 个 checkpoint、设备分配、相同阳性 row 数和不冲突输出路径，不访问 GPU。

- [ ] **步骤 5：把正式命令交给用户执行**

用户确认 dry-run 后，由用户执行去掉 `--dry-run` 的同一命令。agent 不擅自执行。完成后检查 12 个
NPZ 和 manifest 均存在、row id 完全一致、无非有限预测、文件总大小合理。

- [ ] **步骤 6：提交预测导出实现**

```bash
git add scripts/export_harmonic_predictions.py tests/test_export_harmonic_predictions.py
git commit -m "评价: 添加谐波窗口预测导出"
```

### 任务 7：统计模型谐波纠正率并生成代表图

**文件：**
- 修改：`scripts/analyze_bcg_second_harmonic.py`
- 修改：`scripts/plot_bcg_second_harmonic.py`
- 修改：`tests/test_analyze_bcg_second_harmonic.py`

- [ ] **步骤 1：编写模型纠正率汇总失败测试**

构造 corrected、partially corrected、not corrected 三类预测，断言窗口级状态正确；分模型/训练编号汇总
包含三类比例，跨训练汇总包含 mean/std；缺少任何阳性 row 或标签哈希不一致时失败。

- [ ] **步骤 2：运行目标测试确认失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_analyze_bcg_second_harmonic.py -q
```

预期：FAIL，提示 `summarize_prediction_corrections` 不存在。

- [ ] **步骤 3：实现 `summarize-corrections` 子命令**

```bash
./.venv/bin/python scripts/analyze_bcg_second_harmonic.py summarize-corrections \
  --labels runs/bcg_second_harmonic_20260710/test/test_harmonic_labels.csv \
  --thresholds runs/bcg_second_harmonic_20260710/harmonic_thresholds.json \
  --predictions-dir runs/bcg_second_harmonic_20260710/predictions \
  --output-dir runs/bcg_second_harmonic_20260710/corrections
```

输出 `model_harmonic_correction.csv`、`model_harmonic_correction_seed_summary.csv`、
`model_harmonic_correction_summary.csv` 和 manifest。每行同时保留输入与输出频谱特征、标签、模型、
训练编号和纠正状态，便于案例追溯。

- [ ] **步骤 4：实现测试代表图命令**

```bash
./.venv/bin/python scripts/plot_bcg_second_harmonic.py model-cases \
  --config configs/tho_research_v2.yaml \
  --labels runs/bcg_second_harmonic_20260710/test/test_harmonic_labels.csv \
  --corrections runs/bcg_second_harmonic_20260710/corrections/model_harmonic_correction.csv \
  --predictions-dir runs/bcg_second_harmonic_20260710/predictions \
  --output-dir runs/bcg_second_harmonic_20260710/figures \
  --max-cases 8
```

案例选择规则固定覆盖：四模型一致 corrected、一致失败、模型间分歧、阈值边界。每张图显示输入 BCG、
THO、四模型输出的呼吸带时域和频谱；不得仅挑选有利于当前 anchor 的案例。

- [ ] **步骤 5：运行测试、正式汇总和绘图**

运行：

```bash
./.venv/bin/python -m pytest tests/test_second_harmonic_analysis.py tests/test_analyze_bcg_second_harmonic.py tests/test_export_harmonic_predictions.py tests/test_stratified_eval_analysis.py -q
```

预期：全部 PASS。随后运行正式 corrections 和 model-cases 命令，检查纠正率分母与合并阳性窗口数一致。

- [ ] **步骤 6：提交纠正率与绘图实现**

```bash
git add scripts/analyze_bcg_second_harmonic.py scripts/plot_bcg_second_harmonic.py tests/test_analyze_bcg_second_harmonic.py
git commit -m "分析: 汇总模型二次谐波纠正率"
```

### 任务 8：记录工作流、验证并写回阶段汇报

**文件：**
- 修改：`scripts/README.md`
- 修改：`docs/stage_reports/2026-07-08-tho-research-v2-group-meeting-stage-report.md`
- 按证据需要修改：`docs/experiments/current_evidence_ledger.md`

- [ ] **步骤 1：在脚本 README 记录可复制工作流**

新增小节按顺序列出 `discover -> validation-review -> freeze -> apply -> summarize-metrics -> prediction dry-run -> 用户 GPU 推理 -> summarize-corrections -> model-cases`。明确：

- validation 定阈值，test 只应用；
- `2*f_tho > 0.7 Hz` 和 THO 不稳定窗口单独报告覆盖率；
- 正式 GPU 命令由用户执行；
- 所有输出均为旁路派生结果，不覆盖历史产物。

- [ ] **步骤 2：运行静态和目标测试**

运行：

```bash
./.venv/bin/python -m py_compile \
  resp_train/analysis/second_harmonic.py \
  scripts/analyze_bcg_second_harmonic.py \
  scripts/export_harmonic_predictions.py \
  scripts/plot_bcg_second_harmonic.py
./.venv/bin/python -m pytest \
  tests/test_second_harmonic_analysis.py \
  tests/test_analyze_bcg_second_harmonic.py \
  tests/test_export_harmonic_predictions.py \
  tests/test_stratified_eval_analysis.py -q
```

预期：编译无输出；全部测试 PASS。

- [ ] **步骤 3：运行共享路径回归测试**

运行：

```bash
./.venv/bin/python -m pytest tests/test_signal_metrics.py tests/test_eval_metrics.py tests/test_research_v2_data.py tests/test_diagnostics_scripts.py -q
```

预期：全部 PASS。若只新增 analysis 模块而未改 `resp_train/metrics/`，这些测试仍用于证明没有改变
现有指标、mask 或数据读取语义。

- [ ] **步骤 4：核验正式产物完整性**

检查 manifest/CSV/JSON：阈值状态为 frozen；validation/test split 无混用；12 个模型预测 manifest 的
labels 哈希一致；所有模型同层窗口数一致；coverage 分母闭合；不存在 NaN/Inf 被静默计入纠正率；
checkpoint 与 `configs/eval_specs/g_series_test_eval_20260705.csv` 一致。

- [ ] **步骤 5：把真实结果写回阶段底稿**

在阶段报告新增“低质量 BCG：二次谐波显著窗口”，按以下固定结构写入实际数值：

1. 离线 THO 参考的操作性定义与非推理用途边界。
2. 总测试窗口、THO 不稳定排除、二倍频出带排除、可判定覆盖率、阳性窗口和受试者数。
3. 四模型在阳性层与 harmonic-negative 层的五项任务指标。
4. 四模型 corrected/partially/not corrected 比例。
5. 一张不偏向单模型选择的代表图。
6. 样本重叠、样本量与探索性结论限制。

只有“输出谐波下降 + 呼吸率或周期数量恢复”同时成立时才使用“纠正/解决”表述；否则写成“部分
抑制”或“尚无充分证据”。若结果改变当前 active shortlist 或下一步实验判断，再同步更新
`current_evidence_ledger.md`；否则不改长期证据账本。

- [ ] **步骤 6：文档轻量检查**

运行：

```bash
rg -n "TO""DO|TB""D|待""定|待""补|低维频带摘要|一对一波形翻译" \
  scripts/README.md \
  docs/stage_reports/2026-07-08-tho-research-v2-group-meeting-stage-report.md
git diff --check
```

预期：占位符和问题词无输出；`git diff --check` 无输出。

- [ ] **步骤 7：提交文档和最终证据记录**

```bash
git add scripts/README.md docs/stage_reports/2026-07-08-tho-research-v2-group-meeting-stage-report.md
git add docs/experiments/current_evidence_ledger.md  # 仅当阶段判断确实改变
git commit -m "文档: 补充BCG二次谐波分层结果"
```

- [ ] **步骤 8：最终工作区和提交范围检查**

运行：

```bash
git status --short
git log --oneline -10
```

预期：本任务修改均有清晰提交；用户原有未提交文件仍保持原状态，没有被意外纳入本任务提交。
