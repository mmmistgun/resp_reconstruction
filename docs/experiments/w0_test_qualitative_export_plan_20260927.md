# W0 测试集定性分析导出协议

协议 ID：`w0-test-qualitative-export-v1-20260927`。日期：2026-09-27。

状态：完整 2310 窗口导出及 `qualitative-export-v2` 离线收尾已完成，PNG 绘图待用户执行。产物目录为 `runs/w0_test_qualitative_v1/seed_20260812_export_01`，完成状态见其中 `receipt.json` 与 `artifact_manifest.json`。本专项仅导出固定模型的定性分析资料，不改变历史冻结结果。

## 1. 固定范围

保存测试集逐窗口预测、FiLM 数值和四联图，使后续论文分析可以直接读取导出文件，避免每次绘图重新加载模型。

来源固定为原始 `crd_tf102_w`、默认 `c_gamma=c_beta=0.5`、seed 20260812 的原有 validation-selected epoch 15 checkpoint。完整 test 为 2310 windows、8 个 `samp_id`；运行时核验对应 manifest、checkpoint 与 row identity。

单 seed 选择依据：已有冻结 `runs/w0_film_gamma_test_v1/summary/summary_0d2bbbcc5ee6_20260919T082329Z_78d16bc7986e/seed_metrics.csv` 中 GAMMA_050 的 seed 20260812 在 Whole RR、Local RR、global modulation、PCC 四项表现最好，trajectory 与最优 seed 接近（0.138999 对 0.138897）。其 test 指标依次为 0.587029 bpm、0.589975 bpm、0.167431、0.877993。这是按已有 test 表现选择的定性展示实例，论文需如实说明；总体性能继续引用三 seed 汇总。

固定执行边界：

1. 单 seed 采用上述来源，保留选择依据与既有 checkpoint epoch。
2. 第二幅图采用现有 F0 固定呼吸频带，第三幅图采用已有协议化 Python IEWT。
3. Codex 执行范围为实现、文档、synthetic CPU 验证及用户已授权的已有产物离线收尾。真实数据推理和全量绘图由用户运行本协议命令。
4. `export` 入口必须显式传入 `--confirm-research-test-export`；只开放本协议的固定 test 访问。绘图入口仅读取完成的导出产物。

## 2. 已有实现入口

| 功能 | 入口 | 复用范围 |
|---|---|---|
| W0 波形导出与绘图 | `resp_train/paper_evidence/p6_multi_attribute.py` 的 `export_p6_validation_waveforms` / `render_waveform_panel` | 参考数据关联、预测保存与指标回放方式；现有入口只允许预定 validation 样例 |
| FiLM 捕获 | `resp_train/paper_evidence/w0_cwt_film_behavior.py` 的 `forward_with_capture` | 捕获实际 Z、gamma_raw、beta_raw、Z_prime |
| FiLM 统计 | 同模块 `compute_batch_statistics` | 有效调制、相对强度、时间变化、边缘比例和闭合误差 |
| 固定呼吸频带 | `resp_train/baselines/fixed_band.py` | 使用数据集已有 F0 信号及其实际信号键 |
| IEWT | `resp_train/baselines/iewt.py` 的 `extract_respiration_iewt` | 沿用冻结预处理、分块和输出波形 |
| 任务指标 | `resp_train/metrics/task.py` | 沿用既有五主指标、资格规则、时段与对齐口径 |

## 3. 详细结果的数据合同

采用新的独立输出 identity，拒绝覆盖已有目录；单 seed 按窗口保存压缩 NPZ，公共坐标单独保存。所有文件通过 `dataset_row_id` 关联，并保留 `samp_id`、split、窗口时间、采样率、信号键与对齐元数据。

- 波形：BCG、THO reference、F0、IEWT 和 W0；保存完整 180 秒波形，评价区间固定为完整 `[0,180)` 秒。BCG 明确标注为已有状态对齐、segment soft-z 的宽带模型输入。
- W0 预测：完整模型原生输出及对应窗口顺序、checkpoint 身份和 epoch。
- FiLM 张量：`gamma_raw`、`beta_raw`、实际有效 `g=c_gamma*tanh(gamma_raw)`、`b=c_beta*tanh(beta_raw)`、调制前 `Z` 与调制后 `Z_prime`；保留实际推理精度信息和通道/时间维度定义。
- FiLM 表：逐窗口和时间块的 g/b 幅值、时间变化、边缘比例、`r_scale`、`r_shift`、`r_total`、低能量标记及调制公式闭合误差。复用已有定义，不另造同名统计量。
- 指标表：逐窗口、逐方法、逐 seed 的五主指标与资格/退化标记；绘图读取此表。基线与 W0 使用相同窗口和评价时段。
- 来源：resolved config、命令、代码身份、checkpoint/cache/index/source metrics 的身份、运行环境、访问回执、完成/失败状态及 artifact manifest。

同一次 W0 forward 捕获预测与 FiLM。固定 batch=128、尾 batch=6、BF16 AMP、eval + inference_mode、shuffle=false；不补齐或更换 batch。五主指标使用 `evaluate_task_predictions(..., include_test_only=False)`。导出后按 row 对齐既有冻结 test metrics，并核对 target eligibility。`1e-6` 作为描述性参考精度保留在差异表中；`qualitative-export-v2` 的完成条件为来源身份、完整性与有限性检查通过。差异统计不阻断定性导出，绘图使用本次保存波形对应的指标；历史论文汇总保持冻结。

本修订由用户于 2026-09-27 明确确认。该决定改变导出验收规则，不改变数据、模型、核心指标或历史结论，也不将数值差异视为已确定的 BF16 原因。

非有限 input/target/prediction/FiLM/五主指标显式失败；退化局部 RR 用 NaN 和显式 valid 标记表示，其原生指标仍按既有 39 bpm 惩罚计算，不删除样本。保存的局部 RR 可用 target eligibility 和 prediction validity 重建原生 Local RR MAE。

## 4. 四联图

所选 seed 的每个窗口生成独立 PNG 四联图；全部窗口可按 row 索引浏览，图形生成独立于推理导出。

1. BCG 波形，注明真实信号层级、时间轴和幅值单位/归一化。
2. THO reference + F0 固定呼吸频带结果。
3. THO reference + IEWT 结果。
4. THO reference + W0 结果，标明 seed。

四幅图共享时间轴；明确标出评价时段，reference 的颜色在后三幅保持一致。各重建子图标注 PCC、Whole RR error、Local RR MAE；trajectory/global modulation error 放入图旁或底部指标栏，并注明统计时段。若 PCC 使用 lag-aware 定义，明确标注，不为视觉拟合额外移动曲线、翻转符号或拟合幅度。

后三幅叠加曲线采用指标原生 `canonicalize_numpy`（0.05–0.7 Hz 及既有归一化），图中明确标注 canonical amplitude；原始输出仍完整保存在 NPZ 的 `waveforms` 中。`best_lag_sec` 作为指标附注，展示波形保持原有时间坐标。局部放大不重新计算全窗指标，图题始终注明完整 180 秒评价。

FiLM 另存 g/b 通道×时间热图与调制强度时间曲线，保留主四联图的可读性。latent 通道不是频带，不能把通道热图解释为高频归因图。

补充产物：CWT 频率轴、时间轴、尺度映射及实际网络输入表示；可重绘的局部 RR（60 s / 15 s）与 log-RMS 包络（10 s / 5 s）。绘图索引包含主体、窗口起止时间、五主指标与 `W0 - baseline` 原值差，便于查看不同主体、典型窗口及困难窗口。四项 error 差值为负表示 W0 更好，PCC 差值为正表示 W0 更好。

同时保存原始 log-RMS 与逐窗口 median-centered log-RMS；轨迹附图显示后者，与 envelope trajectory MAE 的定义一致。

CWT 实际频率允许重复，热图以尺度索引排列并用实际 Hz 标注；latent 轴采用 0.1 秒网格名义中心，不解释为因果时延。FiLM `g/b` 在原生 dtype 计算后提升至 FP32 存储；`g*Z` 是提升至 FP32 后的描述性分解，`Z_prime-Z` 使用实际捕获张量计算。两者的浮点舍入语义记录于 `tensor_semantics.json`。

## 5. 高频 BCG 的解释范围

FiLM 数值描述条件调制实际发生的强度和变化；仅凭 gamma/beta 非零，不能得出高频段改善重建的结论。论文功能解释可结合已冻结的 validation 频带干预，入口为 `paper_p5_w0_functional_evidence_protocol_20260905.md`；其中 RESP/CARRIER/CARRIER_L/CARRIER_H 的操作是“仅保留该频带”。

本次导出不自动增加 test 频带干预。若需要 test 上直接比较高频贡献，需要另外明确干预矩阵和证据边界，不能从全频 FiLM 张量直接推断。展示案例保留其来源和选择规则，不根据 test 案例重选 checkpoint 或改变模型。

## 6. 实施与验收

实现入口：

- `scripts/export_w0_test_qualitative.py`
- `resp_train/paper_evidence/w0_test_qualitative.py`
- `resp_train/paper_evidence/w0_test_qualitative_runtime.py`
- `tests/test_w0_test_qualitative.py`

复用现有 training source lock（SHA-256 `2bf4b72a15111e1caa76b6bed19abbde3242edc451795176c307d160cc49368c`）和 gamma test source lock（`0d2bbbcc5ee6fee8e442475600c450de55c8e03ba31f18f5043e0557ac1ae712`）中的 W0 来源，具体文件名见代码 `SOURCE_LOCKS`；不复跑这些历史阶段。当前源码快照、来源首尾哈希、环境和产物 manifest 共同记录本次导出身份。

允许的本地快速验证使用 synthetic fixture，覆盖 row 关联、非有限值拒绝、压缩文件读写、原生局部 RR 指标重建、退化有效性标记、四联图指标匹配、离线重绘与防覆盖。真实数据验证按确认的执行范围进行：全部 rows 与所选单 checkpoint 覆盖完整、预测与冻结指标在预设容差内一致、FiLM 数值完整、四联图可由已保存产物重绘。历史结果不补写，不重训，不改变指标、split 或 checkpoint selector。

### 用户执行命令

在仓库根目录运行；下面目录必须尚不存在。完整 FiLM 八个 `[96,1800]` FP32 张量的未压缩体积约 12.8 GB，另含波形/CWT/轨迹/源码。建议预留至少 20 GB 磁盘空间以及图形空间；压缩率和运行时间依实际数据与硬件而定。源 NPZ 与 cache 哈希核验会产生只读 I/O。

```bash
./.venv/bin/python scripts/export_w0_test_qualitative.py export \
  --device cuda:0 \
  --output runs/w0_test_qualitative_v1/seed_20260812_export_01 \
  --confirm-research-test-export

./.venv/bin/python scripts/export_w0_test_qualitative.py render \
  --source runs/w0_test_qualitative_v1/seed_20260812_export_01 \
  --output runs/w0_test_qualitative_v1/seed_20260812_figures_01
```

完成后打开绘图目录的 `index.html`，按 row/主体/时间或显示的指标搜索。`window_index.csv` 提供全部五主指标与相对 F0/IEWT 差值，便于排序筛选。论文选定窗口后，可使用 `render --rows <row_id> ... --zoom 30 60` 输出局部放大到新的独立目录；`--rows` 省略时绘制全部窗口。

### 产物与验收

- `windows/row_<id>.npz`：`bcg`，`waveforms`（reference/F0/IEWT/W0），canonical 曲线，CWT，八个 FiLM 张量，逐通道/5 s 时间块统计，逐 latent 时刻强度与低能量标记，RR/包络轨迹，RR peak validity mask。
- `coordinates.npz`：CWT 实际频率、scales、池化中心时间，latent 名义中心、FiLM 5 s 时间块中心。
- `metrics.csv`：2310×3=6930 行，三种方法完整逐窗口指标，IEWT 分块边界与选中模式。
- `film_statistics.csv`、`window_index.csv`：各 2310 行；`test_rows.csv` 保存来源、信号键、对齐与质量元数据。
- `anchor_deltas.csv`：2310×5=11550 行，保存逐窗口原值与回放差异；`anchor_summary.csv` 保存逐指标最大/平均绝对差、均值有符号差与参考精度内的数量。
- `source_manifest.json`、`source_code/`、配置、环境、`access_started.json`、`receipt.json`、`artifact_manifest.json`：复现与完整性信息。
- 全量绘图：6930 张 PNG（三类图×2310窗口）；选定窗口可另存局部放大图到独立绘图目录。

只有 `artifact_manifest.json` 完成且 `receipt.json` 标记 complete 才算成功。异常直接报错退出，已生成数据保持原样。绘图会校验所读文件哈希并拒绝不完整导出。

### 已有产物离线收尾

```bash
./.venv/bin/python scripts/export_w0_test_qualitative.py finalize \
  --source runs/w0_test_qualitative_v1/seed_20260812_export_01
```

`finalize` 检查保存的训练配置、执行配置、源码快照与执行 commit、访问来源记录和冻结指标身份；检查完整窗口、FiLM 统计、数组 shape/finite、RR 有效性标记及指标 row/subject 对应关系。核验后追加窗口索引、差异汇总、来源清单、收尾代码快照和完成清单。已有窗口与指标不改写。

该步骤只读取已保存导出和冻结指标表，不读取原始信号、cache 数组或 checkpoint，不重新推理。来源核验依据是已保存的配置/源码/访问元数据及冻结参考指标；本次离线收尾不能补证原导出未完成的原始 source bytes 首尾核验，此事实记录在 `source_manifest.json`。对已完成目录拒绝再次收尾。

本次定向测试命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv/bin/python -m pytest tests/test_w0_test_qualitative.py -q
```

synthetic 测试不构成真实 test 完成记录。导出与离线收尾的实际完成状态以各目录的 receipt/manifest 为准；全量绘图由用户执行。
