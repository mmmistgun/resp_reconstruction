# G 系列四模型测试集可视化工具设计

日期：2026-07-10

## 目标与范围

为 THO research v2 的 4 个当前比较模型提供可复用的测试集逐窗口可视化工具。每张图比较同一个 `dataset_row_id` 的输入、THO 参考和 4 个模型输出；工具服务诊断与汇报选图，不改变训练、split、checkpoint 或现有测试指标。

当前范围只覆盖以下模型，每个模型只使用 1 个由验证集确定的 checkpoint：

| 模型标签 | seed | checkpoint | 验证选择来源 |
|---|---:|---|---|
| `g0_time_only` | `20260837` | `checkpoint_top1.pt` | `runs/g_series_stft_input_topk_best_by_rr.csv` |
| `g0_f0_native_stft_pre_mixer` | `20260700` | `checkpoint_top1.pt` | 同上 |
| `g3_c_wide_8p0` | `20260700` | `checkpoint_top3.pt` | 同上 |
| `g3_c_bandenergy` | `20260700` | `checkpoint_top3.pt` | 同上 |

选择链沿用当时已经冻结的 validation top-k task selection：
`rr_peak_band_abs_error_mean -> frac_gt_1 -> frac_gt_2 -> rr_spec_abs_error_mean -> breath_count_zero_cross_abs_error_mean`。
工具必须在产物 manifest 中将其标为 `validation_topk_legacy_task_selection`，不得称为按当前 robust-RR 口径重新排序。

不复用、拼接或校验任何历史预测缓存。已有二次谐波阳性窗口预测只覆盖部分测试集，且不属于本工具输入。

## 两阶段架构

### 阶段 A：全量预测缓存导出

新增 GPU 导出入口，固定导出完整 held-out test 的 2,310 个可用窗口。此命令由用户执行；助手只提供实现、dry-run 和可复制命令，不主动启动。

导出目录使用内存映射友好的 `.npy` 布局：

```text
<output-dir>/
  manifest.json
  dataset_row_id.npy                 # (2310,)
  bcg_input.npy                      # (2310, 18000), float32
  tho_ref.npy                        # (2310, 18000), float32
  predictions/
    g0_time_only/r_tho_hat.npy       # (2310, 18000), float32
    g0_f0_native_stft_pre_mixer/r_tho_hat.npy
    g3_c_wide_8p0/r_tho_hat.npy
    g3_c_bandenergy/r_tho_hat.npy
```

`manifest.json` 必须记录：4 个 checkpoint 的绝对路径与 SHA-256、resolved config 路径与哈希、测试 split 与行数、`dataset_row_id` 哈希、数据索引哈希、数组 schema、导出脚本与代码版本、选择来源和过滤无关的输入范围。

导出前检查 checkpoint/config 一致性；导出后检查 4 个预测数组、BCG 和 THO 的第一维完全一致、row id 唯一且全部有限。默认拒绝覆盖已有缓存；恢复运行只能复用 manifest 与哈希完全一致的完成文件。

采用 `.npy` 而不是 `.npz`，以便阶段 B 用 `np.load(..., mmap_mode="r")` 每次读取一个窗口。预计缓存约 1 GB；该预算只包含 2 份共享波形和 4 份预测波形，不将 THO/BCG 重复写入每个模型文件。

### 阶段 B：CPU 绘图与产物收集

新增绘图入口，只读取阶段 A 的缓存和已存在的 canonical 测试逐窗口指标：
`runs/test_eval_g_series_20260709_local_rr_canonical/*_test_metrics.csv`。它不重新推理、不重算指标、不写入历史 run 目录。

绘图前，工具必须按 `dataset_row_id` 对齐 4 份 metrics、共享缓存和 4 份预测；要求测试 split 一致、每个 row id 恰好出现一次、target 指标列齐全。不满足时停止并报出具体文件、标签和 row id。

每个保留窗口生成 1 张 PNG，包含：

1. BCG soft-z 输入与 `0.05–0.7 Hz`、4 阶零相位 Butterworth 呼吸带分量的叠加图；文案统一称为“呼吸带分量”，不把带通误称为低通。
2. THO 参考波形与 4 个模型输出的叠加图。
3. BCG 呼吸带分量、THO 和 4 个输出在 `0.05–0.7 Hz` 的 Welch 归一化频谱图。
4. 逐窗口指标表：预测/目标 robust RR 与绝对误差、周期计数 bpm 误差、`best_lag_corr_4s`、`best_lag_sec_4s`、`relative_envelope_corr_lag4s`、`local_rr_mae`、`local_rr_corr` 和 `local_rr_valid_frac`。

同一波形面板内所有曲线共用 y 轴范围，保留 soft-z 幅值关系；不得对不同曲线分别 z-score 后再宣称幅度可比较。频谱使用每条信号在绘图区间内归一化的 Welch 功率，因此只用于比较主频与频率结构，不比较绝对能量。

绘图输出还包括：

- `window_index.csv`：每个 row id 的保留状态、过滤特征、过滤原因、图片路径和四模型指标索引。
- `plot_manifest.json`：缓存 manifest 摘要、metrics 文件与哈希、绘图参数、过滤规则、总/保留/排除窗口数、代码版本。
- `filter_summary.csv`：输入侧稳定度阈值与各状态计数。

## 输入侧稳定窗口过滤

工具支持：

- `--filter all`：保留 2,310 个窗口。
- `--filter exclude-input-stable`：默认模式，排除输入侧最稳定的 20% 窗口。

稳定度只由 `bcg_input.npy` 计算，不能读取 THO、任一模型的预测或当前测试 metrics。对每个窗口计算：

- 呼吸带 Welch 频谱主峰集中度；越高表示主频越明确。
- 40 秒窗口、10 秒步长的 BCG 局部 RR 有效性与稳定度；越稳定表示局部节律变化越小。

将两项输入特征转换为测试集内分位数并组成稳定度分数；分数最高的 20% 标为 `input_stable_excluded`，其余为 `retained`。所有原始特征、分位数、规则版本和结果都写入 `window_index.csv`，使该过滤能复现且不被误解为模型/目标侧的后验挑样。

## 命令职责与资源边界

- GPU 缓存导出是长时间运行命令，由用户在实现完成后手动执行。
- 导出完成后，助手检查 manifest、shape、完整性和指标对齐，再执行或组织 CPU 侧绘图与产物汇总。
- 所有正式产物写入新的用户指定输出根目录；不覆盖 `runs/test_eval_g_series_*`、checkpoint、历史图或日志。

## 错误处理与测试

必须在实现中覆盖：

- 选择表中标签/seed/checkpoint 缺失或重复；checkpoint 与 config 不匹配。
- 缓存的不完整写入、非有限数、shape 不一致、重复/缺失 row id。
- 4 份 canonical metrics 不能一一对齐，或缺少当前 canonical 指标列。
- 过滤后没有窗口可画，或输出路径已存在。

单元测试至少覆盖：验证选择的词典序、缓存 manifest 与数组完整性、row 对齐、输入侧过滤的确定性、Welch 频率范围和指标表列。另提供不依赖 GPU 的合成数据 smoke，验证单图和索引产物可生成。
