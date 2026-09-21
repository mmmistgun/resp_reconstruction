# E4 性能、代价与机制证据数据

日期：2026-09-22。结论与实验范围见 [E4 结项报告](../e4_closeout_20260922.md)。本目录从已冻结的汇总 CSV 和 benchmark JSON 整理，可直接用于统计核对和论文数据引用。

## 数据文件

| 文件 | 行数 | 内容 |
|---|---:|---|
| [performance_cost.csv](performance_cost.csv) | 14 | 六模型的 validation/test 五指标均值与三 seed SD、参数量、耗时、显存；W0 随两轮效率测量分别列出 |
| [benchmark_measurement_groups.csv](benchmark_measurement_groups.csv) | 42 | 各轮/模型/场景的三个测量组，保留 median、IQR、allocated/reserved 显存、硬件及重复次数 |
| [mechanism_paired_points.csv](mechanism_paired_points.csv) | 144 | 固定全部四层 W 分支 GN 后，三种错位的包络/PCC 配对变化，包含三 seed、pooled 和七个 samp_id |
| [mechanism_subject_directions.csv](mechanism_subject_directions.csv) | 42 | 每错位×指标×受试者中变差、改善和相同的 seed 数，分母均为 3 |
| [mechanism_signal_associations.csv](mechanism_signal_associations.csv) | 64 | 原样及三种错位的 R3–THO 周期/相对包络相关，包含 pooled 与受试者均值、signed/absolute 统计及有效数 |
| [mechanism_signal_paired_changes.csv](mechanism_signal_paired_changes.csv) | 48 | 三种错位相对原样的信号关联配对变化、定义有效数与方向计数 |

结构候选相对 W0 的全部五指标配对差及 seed 方向另见 [完整比较表](../e4_closeout_comparison_20260922.csv)。机制检查的其余条件、其余指标继续保存在原 summary，并由 [产物索引](../e4_closeout_artifact_index_20260922.json)定位。

## 性能与代价字段

- `arm`：`W0_FULL` 为均匀尺度平均；`w0_mr4_residual` 为四区域残差融合；其余依次为静态尺度加权、尺度注意力、频率感知注意力、逐通道区域加权。
- `benchmark_group`：A 是首轮残差融合比较，B 是四候选比较。W0 的性能来自同一冻结对照，效率来自各自测量轮次；不能跨 A/B 混合速度排序。
- `split`：`val` / `test`；每 seed 分别为 2675/2310 窗口。`test` 保留既有 research-test-informed development 证据属性。
- `*_mean` / `*_seed_sd`：五主指标的三训练 seed 均值与 sample SD（ddof=1）。Whole/Local RR 单位 bpm，前四项越小越好，lag-aware signed PCC 越大越好。
- `parameters` / `added_parameters`：全模型参数和相对 W0 的新增参数。
- `eval_ms_mean` / `train_ms_mean`：三个独立测量组的组内 median 之均值；`*_ms_measurement_sd` 是这些 median 的 sample SD，不是训练 seed SD。
- `*_peak_allocated_gib_max` / `*_peak_reserved_gib_max`：三个测量组对应峰值的最大值，单位 GiB。allocated 与 reserved 分别保留。

效率条件为 RTX 4070 Ti SUPER，推理 batch=1、训练 batch=128，synthetic cached-input model-only。输入预处理不计入。组明细 `median_ms`、`iqr_ms` 单位为毫秒，显存原始字段单位为 bytes。

## 机制数据字段和解释

`condition` 固定为 `SHIFT_1/2/3__ALL_W_GN_FIXED`。每窗口三个 R3 循环平移偏移事先固定在 30–150 s 范围，跨三个模型 seed 共用；不是所有窗口共用三个统一偏移长度。四层 W-GN 使用该窗口原输入统计。

`scope=pooled` 是完整窗口汇总；`scope=samp_id` 是对应受试者分组，`group` 给出其 ID。`full` / `intervened` 为原样和干预的指标值。`degradation` 对误差指标取 `intervened-full`，对 PCC 取 `full-intervened`，正值统一表示变差。`relative_degradation_pct` 对误差指标取相对变化百分比，PCC 不定义该百分比。

方向表的 `worse_seeds` / `better_seeds` / `equal_seeds` 分别按配对差严格大于、小于、等于零计数。每格分母是同一受试者上的三个模型 seed，不能解释为三个独立受试者，也不代表超过某种临床或统计阈值。

信号表固定使用 `representation=R3_MEAN`、零滞后、完整支持区：R3 输入是 2 Hz 的 CWT 对数幅度图；周期曲线采用原任务带投影，相对包络采用 10 s 窗、5 s 步长的 log-RMS。`signed_*` 和 `absolute_*` 分别保留有符号相关与预定义绝对相关统计。配对变化表保留 `paired_defined_n`；这些信号分析由三个模型 seed 共享，仅计算一次。

机制数据支持当前 W0 对幅度时间对应的任务依赖，固定 GN 仅覆盖 W 分支。Local RR 的受试者等权方向有差异；完整五指标、跨路径交互及生理归因边界见结项报告。

## 来源与复核

2026-09-22 已完成 [15 份完整 X 的存储清理](../e4_feature_x_cleanup_20260922.json)。本目录依赖冻结 CSV、benchmark JSON 和 summary 来源，不读取 X；原 manifest 保留，完整 X 的缺失由清理记录说明。清理后重新整理的六份 CSV 与本目录版本逐字节一致。

[source_manifest.json](source_manifest.json) 登记结项索引、每个输入文件的 SHA-256、整理脚本身份、统计定义及运行环境。[manifest.json](manifest.json) 绑定六份 CSV 和来源清单的字节身份；本 README 为阅读说明。

从四份原 `seed_metrics.csv` 复算的 140 项性能均值/SD 与总表一致；耗时统计与各测量组明细一致；42 个方向记录与配对数据一致。所有 CSV 保留完整导出精度。

整理入口：[collect_e4_closeout_data.py](../../../scripts/collect_e4_closeout_data.py)。指定新的输出目录即可复核，已有目录拒绝覆盖：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./.venv/bin/python scripts/collect_e4_closeout_data.py \
  --output /tmp/e4_closeout_data_check
```
