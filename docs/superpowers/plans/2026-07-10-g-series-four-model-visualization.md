# G 系列四模型测试集可视化工具实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（- [ ]）语法来跟踪进度。

**目标：** 导出由验证集冻结的 4 个 G 系列 checkpoint 的完整测试集波形缓存，并以最多 48 个 CPU 进程生成带输入、THO、频谱和逐窗口 canonical 指标的对比图。

**架构：** 阶段 A 只做 GPU 推理和 .npy 缓存；BCG/THO 只保存一份，4 个模型各保存预测，全部检查完成才写 manifest.json。阶段 B 只读取缓存与 2026-07-09 canonical metrics，严格按 dataset_row_id 对齐、仅用 BCG 过滤稳定窗口，再用 48 个 memmap worker 并行绘图，主进程统一写索引。

**技术栈：** Python 3.11、NumPy memmap、pandas、SciPy、PyTorch、OmegaConf、Matplotlib Agg、concurrent.futures、pytest。

---

## 文件结构

- 创建 configs/eval_specs/g_series_four_model_visualization.csv：显式冻结 4 个 label/seed/checkpoint。
- 创建 scripts/export_g_series_comparison_cache.py：阶段 A，校验 spec/config，导出共享信号与 4 份预测 .npy，写缓存 manifest。
- 创建 scripts/plot_g_series_comparison.py：阶段 B，加载/对齐缓存与 metrics、过滤、并行渲染和收集索引。
- 创建 tests/test_export_g_series_comparison_cache.py：spec、缓存 schema、row 对齐与 dry-run 测试。
- 创建 tests/test_plot_g_series_comparison.py：指标对齐、BCG-only 过滤、四面板图、48 worker 和失败收集测试。
- 修改 scripts/README.md：记录用户 GPU 导出和 CPU 绘图命令、产物和证据边界。

### 任务 1：冻结比较 checkpoint 并建立导出契约

**文件：**

- 创建：configs/eval_specs/g_series_four_model_visualization.csv
- 创建：scripts/export_g_series_comparison_cache.py
- 创建：tests/test_export_g_series_comparison_cache.py

- [ ] **步骤 1：编写 checkpoint spec 的失败测试**

~~~python
from scripts.export_g_series_comparison_cache import REQUIRED_MODELS, build_export_plan, load_comparison_specs


def test_load_comparison_specs_requires_exact_four_models(tmp_path):
    path = tmp_path / "spec.csv"
    path.write_text(
        "label,seed,checkpoint,selection_source\n"
        "g0_time_only,20260837,/tmp/time.pt,validation_topk_legacy_task_selection\n"
        "g0_f0_native_stft_pre_mixer,20260700,/tmp/f0.pt,validation_topk_legacy_task_selection\n"
        "g3_c_wide_8p0,20260700,/tmp/wide.pt,validation_topk_legacy_task_selection\n"
        "g3_c_bandenergy,20260700,/tmp/band.pt,validation_topk_legacy_task_selection\n",
        encoding="utf-8",
    )
    specs = load_comparison_specs(path, require_paths=False)
    assert [item.label for item in specs] == list(REQUIRED_MODELS)
    assert [item.seed for item in specs] == [20260837, 20260700, 20260700, 20260700]


def test_build_export_plan_rejects_existing_output_dir(tmp_path):
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    with pytest.raises(FileExistsError, match="输出目录已存在"):
        build_export_plan(_four_specs(tmp_path), output_dir=output_dir, devices=["cuda:0"])
~~~

同时覆盖缺列、重复/未知 label、重复 checkpoint、错误 selection_source、空 devices 和 --resume 时 manifest 哈希不匹配。

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
./.venv/bin/python -m pytest tests/test_export_g_series_comparison_cache.py -q
~~~

预期：FAIL，提示 scripts.export_g_series_comparison_cache 不存在。

- [ ] **步骤 3：写入固定 spec 并实现计划/哈希类型**

创建：

~~~csv
label,seed,checkpoint,selection_source
g0_time_only,20260837,runs/g_series_stft_input/g0_time_only/time_only/20260629_155634_106514/checkpoint_top1.pt,validation_topk_legacy_task_selection
g0_f0_native_stft_pre_mixer,20260700,runs/g_series_stft_input/g0_f0_native_stft_pre_mixer/dual/20260629_155803_880047/checkpoint_top1.pt,validation_topk_legacy_task_selection
g3_c_wide_8p0,20260700,runs/g_series_stft_input/g3_c_wide_8p0/dual/20260630_211450_861579/checkpoint_top3.pt,validation_topk_legacy_task_selection
g3_c_bandenergy,20260700,runs/g_series_stft_input/g3_c_bandenergy/dual/20260630_211919_419308/checkpoint_top3.pt,validation_topk_legacy_task_selection
~~~

在脚本中定义：

~~~python
REQUIRED_MODELS = (
    "g0_time_only", "g0_f0_native_stft_pre_mixer", "g3_c_wide_8p0", "g3_c_bandenergy"
)


@dataclass(frozen=True)
class ComparisonSpec:
    label: str
    seed: int
    checkpoint: Path
    selection_source: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
~~~

load_comparison_specs 要求 4 个 label 顺序和集合均等于 REQUIRED_MODELS，所有 checkpoint 存在，所有 source 都为 validation_topk_legacy_task_selection。build_export_plan 创建唯一的 dataset_row_id.npy、bcg_input.npy、tho_ref.npy、predictions/<label>/r_tho_hat.npy 路径；除相同 manifest 的 --resume 外，已有输出目录一律失败。

- [ ] **步骤 4：实现原子 .npy 与 manifest 基础校验**

~~~python
def validate_signal_matrix(values: np.ndarray, *, name: str, expected_rows: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} 必须为 [N, T]，当前 shape={array.shape}")
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise ValueError(f"{name} 行数不一致: {array.shape[0]} != {expected_rows}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含非有限值")
    return array.astype(np.float32, copy=False)
~~~

保存时写 .<name>.<pid>.<uuid>.tmp.npy，np.save 完成后以 os.replace 发布。manifest 必须包含 schema_version、status、spec_sha256、4 个 checkpoint/config SHA-256、dataset_index_sha256、dataset_row_id_sha256、数组 schema、code_version 和 created_at_utc；只有任务 2 全部通过后才写 status="complete"。

- [ ] **步骤 5：运行测试确认通过**

运行：

~~~bash
./.venv/bin/python -m pytest tests/test_export_g_series_comparison_cache.py -q
~~~

预期：PASS；测试不加载 PyTorch、不访问 GPU，也不创建 runs 目录。

- [ ] **步骤 6：提交 checkpoint 与契约**

~~~bash
git add configs/eval_specs/g_series_four_model_visualization.csv scripts/export_g_series_comparison_cache.py tests/test_export_g_series_comparison_cache.py
git commit -m "评价: 固定四模型可视化checkpoint清单"
~~~

### 任务 2：实现阶段 A 共享信号和四模型预测导出

**文件：**

- 修改：scripts/export_g_series_comparison_cache.py
- 修改：tests/test_export_g_series_comparison_cache.py

- [ ] **步骤 1：编写共享缓存和 row/target 对齐失败测试**

~~~python
def test_write_cache_arrays_are_memmap_readable(tmp_path):
    ids = np.asarray([11, 12, 13], dtype=np.int64)
    bcg = np.arange(24, dtype=np.float32).reshape(3, 8)
    tho = -bcg
    paths = write_cache_arrays(
        tmp_path, dataset_row_id=ids, bcg_input=bcg, tho_ref=tho,
        predictions={"g0_time_only": tho + 0.25},
    )
    assert np.load(paths["bcg_input"], mmap_mode="r").shape == (3, 8)
    np.testing.assert_array_equal(np.load(paths["dataset_row_id"]), ids)


def test_validate_prediction_batch_rejects_target_or_id_drift():
    with pytest.raises(ValueError, match="dataset_row_id 顺序"):
        validate_prediction_batch(np.asarray([2, 1]), np.asarray([1, 2]), np.ones((2, 4)), np.ones((2, 4)))
~~~

同时测试非有限值、重复 id、prediction/THO shape 不同、临时文件不能视为 complete、4 份 config 的 test 数据字段不一致。

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
./.venv/bin/python -m pytest tests/test_export_g_series_comparison_cache.py -q
~~~

预期：FAIL，提示 write_cache_arrays 或 validate_prediction_batch 未定义。

- [ ] **步骤 3：导出共享 BCG/THO 并比较 4 份 config**

从第一个 checkpoint 的 config.yaml 用 load_config、read_research_v2_index、ResearchV2WindowDataset 读取完整 test split。其余 config 必须在以下字段一致，否则中止：

~~~python
COMPARISON_CONFIG_KEYS = (
    "data.dataset_root", "data.index_csv", "data.format", "data.input_set",
    "data.test_split", "data.max_test_windows", "data.test_sample_strategy", "data.test_sample_seed",
    "window.target_fs", "loss.spectrum_low_hz", "loss.spectrum_high_hz",
)
~~~

用 numpy.lib.format.open_memmap 逐样本写共享数组，不能先把 2,310 个输入和目标收集在普通内存：

~~~python
row_ids = open_memmap(tmp_ids, mode="w+", dtype=np.int64, shape=(len(dataset),))
bcg_out = open_memmap(tmp_bcg, mode="w+", dtype=np.float32, shape=(len(dataset), n_samples))
tho_out = open_memmap(tmp_tho, mode="w+", dtype=np.float32, shape=(len(dataset), n_samples))
for position, sample in enumerate(dataset):
    row_ids[position] = int(sample["meta"]["dataset_row_id"])
    bcg_out[position] = sample["x"].detach().cpu().numpy().reshape(-1)
    tho_out[position] = _sample_target(sample)
~~~

_sample_target 只接受当前 dataset schema 的 target 或 y，其他情况抛 KeyError。flush、唯一性/有限性/shape 检查后原子发布 3 个文件。

- [ ] **步骤 4：实现每模型 GPU 预测任务和完成 manifest**

每任务加载 checkpoint/config，调用现有 build_model、collect_predictions，再验证其 target 与共享 target 一致：

~~~python
predictions = collect_predictions(model, loader, device=device, max_windows=len(dataset))
ids = np.asarray(predictions["dataset_row_id"], dtype=np.int64).reshape(-1)
pred = validate_signal_matrix(predictions["r_tho_hat"], name=f"{task.label}.r_tho_hat", expected_rows=len(expected_ids))
target = validate_signal_matrix(predictions["tho_ref"], name=f"{task.label}.tho_ref", expected_rows=len(expected_ids))
validate_prediction_batch(ids, expected_ids, target, shared_target)
atomic_save_npy(task.output_path, pred)
~~~

通过 ProcessPoolExecutor(mp_context=mp.get_context("spawn")) 调度 4 个 GPU task；--max-parallel 限制进程，重复 --device 轮转分配。任何任务失败都不写 complete manifest。CLI 提供 --spec、--output-dir、可重复 --device、--max-parallel、--resume、--dry-run；dry-run 只显示 4 个 checkpoint、输出文件、GPU 分配和约 1 GB 空间预算。

- [ ] **步骤 5：运行测试和用户 GPU 前 dry-run**

运行：

~~~bash
./.venv/bin/python -m pytest tests/test_export_g_series_comparison_cache.py -q
./.venv/bin/python scripts/export_g_series_comparison_cache.py --spec configs/eval_specs/g_series_four_model_visualization.csv --output-dir /tmp/g_series_cache_dry_run --device cuda:0 --device cuda:1 --max-parallel 2 --dry-run
~~~

预期：测试 PASS；dry-run 不创建缓存、不加载模型、不访问 GPU。

- [ ] **步骤 6：提交阶段 A**

~~~bash
git add scripts/export_g_series_comparison_cache.py tests/test_export_g_series_comparison_cache.py
git commit -m "评价: 导出四模型测试波形缓存"
~~~

### 任务 3：实现 canonical metrics 对齐和 BCG-only 稳定过滤

**文件：**

- 创建：scripts/plot_g_series_comparison.py
- 创建：tests/test_plot_g_series_comparison.py

- [ ] **步骤 1：编写 metrics 对齐与稳定过滤失败测试**

~~~python
from scripts.plot_g_series_comparison import align_canonical_metrics, select_plot_rows


def test_align_canonical_metrics_requires_one_row_per_cache_id():
    ids = np.asarray([10, 20, 30])
    metrics = _metric_frame([10, 20, 30])
    aligned = align_canonical_metrics(ids, {label: metrics.copy() for label in REQUIRED_MODELS})
    assert aligned.index.tolist() == [10, 20, 30]

    missing = _metric_frame([10, 20])
    frames = {label: metrics.copy() for label in REQUIRED_MODELS}
    frames["g3_c_bandenergy"] = missing
    with pytest.raises(ValueError, match="缺少 cache row"):
        align_canonical_metrics(ids, frames)


def test_select_plot_rows_excludes_exact_top_twenty_percent_input_stable():
    features = pd.DataFrame(
        {
            "dataset_row_id": [1, 2, 3, 4, 5],
            "spectral_peak_fraction": [0.9, 0.7, 0.5, 0.3, 0.1],
            "local_rr_valid_frac": [1.0, 1.0, 1.0, 0.8, 0.2],
            "local_rr_iqr_bpm": [0.1, 0.2, 0.5, 1.0, 3.0],
        }
    )
    selected = select_plot_rows(features, filter_mode="exclude-input-stable", stable_fraction=0.2)
    assert selected.loc[selected.dataset_row_id == 1, "plot_status"].item() == "input_stable_excluded"
    assert selected.plot_status.eq("input_stable_excluded").sum() == 1
~~~

另测试缺少 canonical 列、metrics split 不全为 test、重复 row id、缓存/预测长度不一致和 stable_fraction 不在 (0, 1)。

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
./.venv/bin/python -m pytest tests/test_plot_g_series_comparison.py -q
~~~

预期：FAIL，提示 scripts.plot_g_series_comparison 不存在。

- [ ] **步骤 3：实现缓存/metrics 加载和严格 row 对齐**

定义当前绘图必须存在的 metrics 列：

~~~python
CANONICAL_METRIC_COLUMNS = (
    "dataset_row_id", "split",
    "pred_rr_peak_band_robust_bpm", "target_rr_peak_band_robust_bpm",
    "rr_peak_band_robust_abs_error", "breath_count_zero_cross_abs_error",
    "best_lag_corr_4s", "best_lag_sec_4s", "relative_envelope_corr_lag4s",
    "local_rr_mae", "local_rr_corr", "local_rr_valid_frac",
)
~~~

load_cache 读取 manifest.json 并拒绝非 complete 状态，使用 np.load(..., mmap_mode="r") 打开 3 个共享数组和 4 个预测数组；检查 manifest label/seed/checkpoint 与固定 spec 相同。align_canonical_metrics 对每个 label 读取 <metrics-dir>/<label>_<seed>_test_metrics.csv，要求一行一个 dataset_row_id、split 全为 test、集合与 cache id 完全一致，随后按 cache id 顺序 reindex。表格只派生：

~~~python
frame["breath_count_zero_cross_bpm_error"] = frame["breath_count_zero_cross_abs_error"] / 3.0
~~~

不得重新计算或替换任何 canonical metric。

- [ ] **步骤 4：实现 BCG-only 特征和确定性过滤**

对 bcg_input.npy 的每个窗口，用现有 bandpass_filter、scipy_signal.welch 和 local_rr_rate_trace 计算：

~~~python
def input_stability_features(bcg, *, fs, low_hz, high_hz, order):
    band = bandpass_filter(bcg, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    freqs, power = scipy_signal.welch(band, fs=fs, nperseg=min(4096, band.size))
    in_band = (freqs >= low_hz) & (freqs <= high_hz)
    peak_fraction = float(power[in_band].max() / max(power[in_band].sum(), np.finfo(float).eps))
    rates = local_rr_rate_trace(band, fs=fs, window_sec=40.0, step_sec=10.0, low_hz=low_hz, high_hz=high_hz)
    valid = rates[np.isfinite(rates)]
    iqr = float(np.subtract(*np.percentile(valid, [75, 25]))) if valid.size >= 2 else float("inf")
    return peak_fraction, float(valid.size / rates.size), iqr
~~~

使用 rank 百分位和 row id tie-breaker：

~~~python
score = (
    0.50 * rank_pct(spectral_peak_fraction)
    + 0.30 * rank_pct(local_rr_valid_frac)
    + 0.20 * (1.0 - rank_pct(local_rr_iqr_bpm))
)
excluded = frame.sort_values(
    ["input_stability_score", "dataset_row_id"], ascending=[False, True]
).head(math.ceil(0.20 * len(frame)))
~~~

--filter all 将全部标为 retained；--filter exclude-input-stable 排除上述窗口。该函数只接收 BCG 与评价配置，不能接收 THO、prediction 或 metrics。

- [ ] **步骤 5：运行对齐/过滤测试确认通过**

运行：

~~~bash
./.venv/bin/python -m pytest tests/test_plot_g_series_comparison.py -q
~~~

预期：PASS；稳定过滤可重复，缺失/重复 metrics 均显式失败。

- [ ] **步骤 6：提交对齐与过滤实现**

~~~bash
git add scripts/plot_g_series_comparison.py tests/test_plot_g_series_comparison.py
git commit -m "可视化: 对齐四模型指标并过滤稳定输入"
~~~

### 任务 4：实现四面板图、48 进程渲染和索引收集

**文件：**

- 修改：scripts/plot_g_series_comparison.py
- 修改：tests/test_plot_g_series_comparison.py

- [ ] **步骤 1：编写图形、worker 解析和失败收集测试**

~~~python
def test_render_one_window_writes_four_panel_png(tmp_path):
    cache = _write_synthetic_cache(tmp_path, n_windows=1, n_samples=1000)
    task = _synthetic_render_task(row_id=101, output_dir=tmp_path / "figures")
    initialize_render_worker(cache, task.metrics_by_label, fs=100.0, low_hz=0.05, high_hz=0.7, order=4)
    result = render_one_window(task)
    assert result.status == "written"
    assert result.figure_path.exists()
    assert result.figure_path.suffix == ".png"


def test_resolve_workers_uses_48_cap(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 102)
    assert resolve_workers("auto", n_tasks=2310) == 48
    assert resolve_workers("auto", n_tasks=7) == 7
    assert resolve_workers(12, n_tasks=100) == 12
~~~

另测试每个 OMP/MKL/OpenBLAS/NUMEXPR 环境变量为 1、worker 异常只能生成 failure record、结果由主进程按 row id 排序、已有 PNG 拒绝覆盖。

- [ ] **步骤 2：运行测试确认失败**

运行：

~~~bash
./.venv/bin/python -m pytest tests/test_plot_g_series_comparison.py -q
~~~

预期：FAIL，提示 render_one_window、resolve_workers 或 initialize_render_worker 未定义。

- [ ] **步骤 3：实现共享尺度四面板图**

render_one_window 从 worker 全局只读 memmap 取一个 row，对 BCG、THO 和 4 个预测使用同一 0.05–0.7 Hz、4 阶零相位带通。使用 GridSpec(4, 1, height_ratios=(1.0, 1.2, 1.0, 1.35)) 生成：

~~~python
input_ax.plot(time, bcg, color="#7f7f7f", linewidth=0.55, label="BCG soft-z input")
input_ax.plot(time, bcg_band, color="#1f77b4", linewidth=0.9, label="BCG resp-band (0.05–0.7 Hz)")

output_ax.plot(time, tho, color="#111111", linewidth=1.35, label="THO target")
for label, color in MODEL_COLORS.items():
    output_ax.plot(time, prediction[label], color=color, linewidth=0.75, alpha=0.85, label=label)
~~~

输入面板以 bcg 与 bcg_band 的联合 0.5/99.5 分位数设置同一个 y 轴；输出面板以 THO 和 4 个预测的联合分位数设置同一个 y 轴。不得调用 _scale_for_display、robust_zscore 或逐曲线归一化。频谱面板对 BCG 呼吸带、THO 和 4 个 raw output 使用 Welch，截取 0.05–0.7 Hz 后各自除以频带内功率和，y 轴标为 normalized band power。表格按模型列显示 robust RR pred/target/error、count bpm error、lag4 corr/sec、relative envelope corr、local RR MAE/corr/valid。

- [ ] **步骤 4：实现 48 进程调度和原子产物写入**

在任何 NumPy/SciPy/Matplotlib 导入前设置父进程默认线程环境；worker initializer 再次设置并打开 memmap：

~~~python
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def resolve_workers(value: str | int, *, n_tasks: int) -> int:
    if str(value).lower() == "auto":
        return min(48, max(1, (os.cpu_count() or 1) - 2), n_tasks)
    workers = int(value)
    if workers < 1:
        raise ValueError("--workers 必须 >= 1")
    return min(workers, n_tasks)
~~~

使用 ProcessPoolExecutor 的 spawn context、initializer 和 initargs 传递缓存路径、轻量 metrics、fs、频带和阶数。worker 将 PNG 保存到 .<filename>.<pid>.<uuid>.tmp.png 后 os.replace，只返回 RenderResult。主进程将成功和失败结果写入按 dataset_row_id 排序的 window_index.csv 和 filter_summary.csv。若任一 task 失败，写 plot_failure_manifest.json 后非零退出，绝不写 plot_manifest.json；全部成功才写含缓存摘要、metrics SHA-256、过滤参数、worker 数和图数量的 plot_manifest.json。

- [ ] **步骤 5：实现 CLI、运行测试和 CPU smoke**

CLI：

~~~bash
./.venv/bin/python scripts/plot_g_series_comparison.py --cache-dir runs/g_series_four_model_cache --metrics-dir runs/test_eval_g_series_20260709_local_rr_canonical --output-dir /tmp/g_series_plots_smoke --filter exclude-input-stable --stable-fraction 0.20 --workers 2 --max-plots 2
~~~

--max-plots 仅限 smoke/选图，按保留 row id 升序截取；默认 null 绘制全部保留窗口。正式全量命令使用 --workers auto，在该机器上解析为 48。运行：

~~~bash
./.venv/bin/python -m pytest tests/test_plot_g_series_comparison.py -q
~~~

预期：PASS；smoke 在已有完整缓存后生成 2 张 PNG、window_index.csv、filter_summary.csv 和 plot_manifest.json。

- [ ] **步骤 6：提交并行绘图实现**

~~~bash
git add scripts/plot_g_series_comparison.py tests/test_plot_g_series_comparison.py
git commit -m "可视化: 并行绘制四模型测试窗口"
~~~

### 任务 5：记录工作流、验证实现并设置用户 GPU 检查点

**文件：**

- 修改：scripts/README.md
- 修改：tests/test_export_g_series_comparison_cache.py
- 修改：tests/test_plot_g_series_comparison.py

- [ ] **步骤 1：编写 CLI 默认值的失败测试**

在两份脚本测试中 monkeypatch sys.argv，断言导出默认拒绝覆盖、绘图默认 filter="exclude-input-stable"、stable_fraction=0.20、workers="auto"：

~~~python
def test_plot_cli_defaults_to_input_stable_filter(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_g_series_comparison.py", "--cache-dir", "cache", "--metrics-dir", "metrics", "--output-dir", "plots"],
    )
    args = _parse_args()
    assert args.filter == "exclude-input-stable"
    assert args.stable_fraction == 0.20
    assert args.workers == "auto"
~~~

同时断言 resolve_workers("auto", n_tasks=2310) 在 102 核 mock 下为 48。

- [ ] **步骤 2：运行参数测试确认失败**

运行：

~~~bash
./.venv/bin/python -m pytest tests/test_export_g_series_comparison_cache.py tests/test_plot_g_series_comparison.py -q
~~~

预期：FAIL，直到阶段 A/B 的 CLI 默认值、拒绝覆盖和 worker 解析全部实现。

- [ ] **步骤 3：更新 scripts/README.md 的诊断分析章节**

新增 G 系列四模型测试集可视化小节，写入以下命令与语义：

~~~bash
# 先由用户执行 GPU 缓存导出；执行前先附加 --dry-run 复核。
./.venv/bin/python scripts/export_g_series_comparison_cache.py --spec configs/eval_specs/g_series_four_model_visualization.csv --output-dir runs/g_series_four_model_cache --device cuda:0 --device cuda:1 --max-parallel 2

# 缓存完整后进行 CPU 绘图；auto 在 102 核机器上使用 48 个进程。
./.venv/bin/python scripts/plot_g_series_comparison.py --cache-dir runs/g_series_four_model_cache --metrics-dir runs/test_eval_g_series_20260709_local_rr_canonical --output-dir runs/g_series_four_model_plots --filter exclude-input-stable --stable-fraction 0.20 --workers auto
~~~

明确：GPU 导出由用户手动执行；绘图不重推理；exclude-input-stable 只使用 BCG、排除稳定度最高 20%，不是目标侧或模型侧过滤；checkpoint 来自 legacy validation top-k，不是当前 robust-RR 重选。

- [ ] **步骤 4：运行全量相关测试和静态检查**

运行：

~~~bash
./.venv/bin/python -m pytest tests/test_export_g_series_comparison_cache.py tests/test_plot_g_series_comparison.py tests/test_run_g_series_test_eval.py tests/test_signal_metrics.py -q
./.venv/bin/python -m py_compile scripts/export_g_series_comparison_cache.py scripts/plot_g_series_comparison.py
git diff --check
~~~

预期：pytest 全部 PASS，py_compile 与 git diff --check 均以 0 退出码结束。

- [ ] **步骤 5：提供用户 GPU 命令并设置产物收集检查点**

不要启动正式 GPU 导出。向用户交付步骤 3 的导出命令及一个附带 --dry-run 的命令，等待用户报告完成。完成后按以下顺序收集：

1. 读取 manifest.json，确认 status=complete、固定 4 个 label/seed/checkpoint/hash 和测试窗口数。
2. 以 mmap 打开 3 个共享数组与 4 个预测数组，确认 row id 唯一、shape 一致、值全有限。
3. 加载 4 份 canonical metrics，确认每个 dataset_row_id 恰好一行且可与缓存完全对齐。
4. 先运行 --max-plots 2 --workers 2 的 CPU smoke；成功后再以 --workers auto 渲染全量保留窗口。
5. 检查 window_index.csv、filter_summary.csv、plot_manifest.json、PNG 数量和失败记录；若有失败，报告 row id 与异常，不把部分结果表述为完整图集。

- [ ] **步骤 6：提交文档与测试收尾**

~~~bash
git add scripts/README.md tests/test_export_g_series_comparison_cache.py tests/test_plot_g_series_comparison.py
git commit -m "文档: 补充四模型可视化工作流"
~~~

## 计划自检

- 规格的冻结 checkpoint、.npy 内存映射、GPU/CPU 两阶段、canonical metrics、四面板图、输入侧 20% 过滤、48 进程、原子写入和用户执行 GPU 边界，分别由任务 1–5 覆盖。
- 所有新函数、数组 schema、CLI 参数和错误状态均在首次使用前定义；后续任务只引用这些名称。
- 每个实现任务均包含失败测试、失败命令、最小实现、通过命令和独立提交。
- 本计划不启动正式 GPU 导出，也不修改任何现有 checkpoint、metrics、图或历史 run。
