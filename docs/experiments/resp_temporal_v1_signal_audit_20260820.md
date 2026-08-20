# RTM-v1 train-only 信号审计规范

日期：2026-08-20

协议 ID：

```text
resp-temporal-v1-train-signal-audit-20260820
```

状态：**完整 train-only 审计已从干净 commit `2ce0050` 执行、验收并关闭；receipt、manifest 与全部 artifact hash 已复核，signal-substrate/family/candidate lock 已获用户确认。不得重跑审计或根据结果改题。**

## 1. 目的与权威性

本文是 `docs/experiments/resp_temporal_v1_protocol_20260820.md` 的前置审计附件。它只回答：

1. train BCG 中哪些固定频率机制对正式呼吸目标可观测；
2. 载波调制应在何种采样率下完成非线性解调；
3. 呼吸相位、速率变化、努力趋势和整窗状态分别要求哪些时间尺度；
4. 公共 stem、latent grid 与多尺度降采样需要满足哪些抗混叠和上下文条件。

它不比较神经网络 validation 质量，不选择参数量相近的模型，也不形成任何家族优劣结论。审计结果不得自动改题、调带、增加 proxy 或升格 implementation probe；receipt 完成后必须暂停，由用户人工锁定 substrate、family 与 candidate。

## 2. 数据与访问边界

- 只读取 research-v2 的完整 eligible `train` split，固定 `10141` windows、`32` 个 `samp_id`；
- 有序 `dataset_row_id` 的 SHA-256 固定为 `f290e569140a2ff7745cf1a5cfa6a4da943644d76498c9b85517d3ae0702c45e`；
- 输入固定为 `bcg_rawish_segment_soft_z_key`，shape `[1,18000]`、100 Hz、180秒；
- target 固定为 `target_waveform_segment_soft_z_key`；
- admission、subject/session 隔离、finite 与 eligibility 语义沿用主协议；
- 不读取 validation signal/target/prediction、任何 checkpoint 或 research-test；
- 不运行模型训练、模型推理、GPU smoke、benchmark 或超参数搜索；
- 不覆盖已有审计、run、checkpoint、日志或 summary。

当前 dataset index 是多 split 共享的元数据容器。脚本可读取该 CSV 以核对其 SHA-256，但必须立即仅保留 `split=train` 行；只有这些行可以进入适配器和 NPZ reader。Receipt 必须同时登记 `shared_index_metadata_read=true`、`signal_splits_accessed=["train"]` 与全部 validation/research-test/model/checkpoint access flag 为 false。这里的 `validation_accessed=false` 指没有读取 validation signal、target、prediction 或派生量；共享 index 的存在不得被误写成 validation signal access。

v1 不开放子集、抽样 seed、`max_windows` 或 dry-run 科学结果。完整 CPU 计算属于长任务，只由用户手动执行；若完整审计不可运行，先暂停并修订协议，不得根据部分结果换样本。

## 3. 固定信号机制、频带与端点

为避免连续调带，只使用以下边界：

| 名称 | BCG 频带 | 端点 | 物理/信号角色 |
|---|---:|---|---|
| displacement | `0.05–0.80 Hz` | 两端包含 | 直接呼吸位移与慢变化 |
| carrier-low | `(0.80–3.00] Hz` | 0.80排除、3.00包含 | 低频心动/机械载波及其呼吸调制 |
| carrier-high | `(3.00–8.00] Hz` | 3.00排除、8.00包含 | 更高频机械载波及其呼吸调制 |
| target | `0.05–0.70 Hz` | 两端包含 | 冻结正式呼吸输出频带 |

0.80 Hz 固定归 displacement，3.00 Hz 固定归 carrier-low；频带互不重叠。不尝试新 cutoff、可学习滤波器或依据 target 选择窄带。

全部波形运算使用 float64。固定 band extraction 是整段180秒去均值后的 `rfft/irfft` 矩形投影，FFT norm=`backward`，无 padding；端点按上表显式应用。Carrier 解调固定为：

```text
100-Hz BCG
→ 固定 carrier band extraction
→ scipy.signal.hilbert analytic magnitude
→ log(magnitude + 1e-8)
→ remove whole-window mean
→ 固定 0.05–0.70-Hz projection
```

Displacement 直接使用固定低频投影。固定互补组合只允许：三个 proxy 分别去均值并按 whole-window RMS 标准化，等权相加并除以 `sqrt(3)`，再投影到 target band。低动态 component 固定贡献全零，同时显式保留 dynamic/coverage 计数；不根据 target 拟合符号、频带或权重。上述结果均是 observation proxy，不是预测模型，不能用于宣称模型能力上界。

## 4. 审计 A：target 时间尺度

对每个 eligible train target，在冻结正式 `Pi` 后记录：

- Whole RR：冻结的5个60秒、50% overlap、逐频率 bin 中位数功率语义；
- Local RR：60秒窗、15秒步长，共9个位置；记录均值、范围、IQR、相邻绝对变化中位数与最大值；
- effort：10秒 RMS、5秒步长的 log-effort，共35点；记录范围、IQR、相邻绝对变化中位数与最大值；
- effort 自相关衰减：对去均值序列 $z$ 固定使用 $\sum_{t=0}^{N-k-1}z_tz_{t+k}/\sum_{t=0}^{N-1}z_t^2$，记录首次不高于 `exp(-1)` 的离散 lag，步长5秒；窗口内未 crossing 时记 missing 并登记 `censored=true`，不得填180秒；
- whole-window symmetric Hann periodogram 在 `0.05–0.70 Hz` 内的功率质心、主峰与 RMS 带宽；
- `0.05–0.10 Hz` 能量占 target-band 总能量的比例；
- 冻结 target envelope modulation 与 Low/Medium/High strata。

该审计只把信号映射为 respiratory-cycle/local morphology、tens-of-seconds rate transition、effort trend 与180秒 global context。不得由 train target 反推模型宽度、参数上限或唯一网络深度。

Target band 低动态时仍保留固定9个 Local-RR 与35个 effort sequence 行：RR/频谱字段记 missing、eligibility=false，effort 有限值与 autocorrelation eligibility 分开登记；不得删除整个 window 或用任意频率填充。

## 5. 审计 B：BCG proxy 可观测性

对 displacement、carrier-low envelope、carrier-high envelope 和固定等 RMS 组合分别报告：

- 与 canonical target 的 respiratory-band magnitude-squared coherence；固定5个60秒、50% overlap segment、symmetric Hann、逐 band-bin 直接均值；
- Whole RR 与 Local RR 描述性误差；target eligible 但 proxy 低动态时沿用冻结的 `39 bpm` penalty，不得删行；
- 与 target 10秒/5秒 log-effort 的 Spearman；target eligible 而 proxy degenerate 时固定为 `-1`；
- `±0.3 s` 整数 sample lag 内最大绝对相关的位置，同时保留该位置的 signed correlation；tie-break 固定为最小绝对 lag，再取较小 lag；仅描述对齐，不作为选择目标；
- target-frozen envelope strata 的分层结果；
- window → `(samp_id, stratum, proxy)` mean，再对 `samp_id` direct mean；低样本组同时报告 count，不作 subject effect 推断。

Carrier proxy 优于或补充 displacement 只能解释为“公共 stem 需要保留该观测机制”，不能解释为某个神经网络家族更优。固定组合也只是未拟合的互补性探针，不是 supervised baseline。

## 6. 审计 C：100→20→10 Hz 解调顺序与抗混叠

必须比较：

1. `100 Hz carrier → envelope → target-band projection → 10 Hz`；
2. `100 Hz carrier → anti-aliased 20 Hz → envelope → target-band projection → anti-aliased 10 Hz`；
3. `100 Hz carrier → anti-aliased 10 Hz → envelope → target-band projection`。

Band extraction 始终先在100 Hz完成。所有抗混叠重采样使用 `scipy.signal.resample_poly`、`padtype=line` 与显式奇数长度 Kaiser FIR taps，beta=`8.6`：

| Map | up/down | taps | cutoff |
|---|---:|---:|---:|
| 100→20 | 1/5 | 255 | 9.0 Hz |
| 100→10 | 1/10 | 511 | 4.5 Hz |
| 20→10 | 1/2 | 127 | 4.5 Hz |

逐 path 报告相对 path1 的 NRMSE、zero-lag correlation、target-band energy ratio，以及各自 coherence、RR、effort 与 lag 描述。Receipt 不自动输出“通过/失败”或根据结果调整阈值。其人工解释边界固定为：

- path2与path1的一致性支持先到20 Hz再解调的可行性；
- path3相对path1/2丢失互补信息，支持必须在20 Hz或更高采样率完成 carrier-sensitive filtering + nonlinearity；
- carrier-low/high 均未提供稳定互补证据，只说明当前 train proxy 未支持显式 carrier 路径，不得宣称高频 BCG 无用；
- 任一路径差异都不能重开旧时频协议或读取 validation。

## 7. 审计 D：10/2/(1或0.5) Hz 多尺度 grid

只检查 `10 / 2 / 1 / 0.5 Hz` 离散集合，不搜索 grid 或 cutoff。Fine 10 Hz 是共同 latent；2 Hz Nyquist=1 Hz，可承担完整 `0.05–0.70 Hz` 波形；1 Hz和0.5 Hz Nyquist分别为0.5/0.25 Hz，不得承担完整波形，只能解释为 context/effort grid。

从10 Hz到2/1/0.5 Hz的显式低通分别固定为：

| Map | taps | cutoff |
|---|---:|---:|
| 10→2 | 255 | 0.85 Hz |
| 10→1 | 255 | 0.45 Hz |
| 10→0.5 | 511 | 0.225 Hz |

比较对象为 target、displacement、carrier-low path1 与 carrier-high path1。Boxcar 固定为 `kernel=stride=decimation factor`、无 padding 的 block average。为避免把半个 pooling block 的时间偏移误写成混叠，显式 FIR 使用 reflect padding、`scipy.signal.fftconvolve` 零相位滤波，并在相同 block center 线性取样。二者均先用 `scipy.signal.resample` Fourier reconstruction 回10 Hz，再以频域相位因子补偿 `(factor-1)/(2×10)` 秒的共同 block-center offset，仅作确定性差异描述。

每项记录 source 在目标 Nyquist 以上的功率比例、boxcar-vs-lowpass NRMSE、round-trip NRMSE/correlation、effort Spearman，以及每个 grid 对10秒 effort、60秒 Local RR、180秒 global window的点数。只有 target 的2-Hz grid计算 waveform RR/coherence round-trip；1/0.5-Hz waveform 字段必须 missing，并显式登记 `full_waveform_eligible=false`。正式多尺度只能 feature-level 融合，不建立分支 waveform head，不按 validation 调权。

## 8. 配置、产物、receipt 与不可覆盖语义

唯一配置为：

```text
configs/resp_temporal_v1/signal_audit_train_v1.yaml
```

schema=`rtm-v1-signal-audit-config-v1`。Audit config SHA-256 固定为 `20591d70947d44d5b2f0e49ffec2d75eb29af1b20d9fa964d59442b3ced5ffb5`，`configs/tho_research_v2.yaml` SHA-256 固定为 `104f6c03e662b530fffb5bd376a5d754fdebdfc5b6edaf92a6c958d47951b8d7`；不接受 `--set`、split、band、subset、output 或 operator 覆盖。固定输出目录：

```text
runs/resp_temporal_v1/train_signal_audit/rtm_v1_train_only_v1/
```

目标目录已存在时在读取数据前 fail closed。完整结果先写同父目录唯一临时目录，全部文件和 receipt 校验通过后再原子发布；不得覆盖、删除或重用既有正式目录。临时目录仅用于本次未发布写入失败的清理。

固定产物：

```text
resolved_config.json
target_timescale_window_metrics.csv
target_timescale_sequence_metrics.csv
target_timescale_sample_metrics.csv
proxy_observability_window_metrics.csv
proxy_observability_sample_metrics.csv
sampling_order_window_metrics.csv
sampling_order_sample_metrics.csv
multiscale_grid_window_metrics.csv
multiscale_grid_sample_metrics.csv
sample_direct_summary.csv
artifact_manifest.json
access_receipt.json
access_receipt.sha256
```

Receipt schema=`rtm-v1-signal-audit-receipt-v1`，写前严格拒绝缺字段与未知字段。至少登记：

- 协议、audit/source config hash、代码 commit、`git_dirty=false`、命令、Python/platform/依赖版本；
- shared index SHA-256、有序 row-id SHA-256、逐已选 window 的 input/target content SHA-256；
- split、window/samp_id、exclusion flag/count；
- 每张表的行数，以及每个 numeric column 的 total/finite/null/Inf count；Inf 必须为0，null 只能由显式 eligibility/censoring语义产生；
- 所有非 receipt 产物的 SHA-256/bytes/rows、manifest SHA-256，以及 receipt 自身 `.sha256`；
- `validation_accessed=false`、`validation_target_accessed=false`、`validation_prediction_accessed=false`、`research_test_accessed=false`、`checkpoint_accessed=false`、`model_training_used=false`、`model_inference_used=false`、`gpu_used=false`。

## 9. 用户手动运行与验收

Codex 不执行完整数据审计。用户在实现提交后、工作树干净且固定输出目录不存在时只运行：

```bash
./.venv/bin/python scripts/audit_resp_temporal_v1_signal.py --config configs/resp_temporal_v1/signal_audit_train_v1.yaml
```

验收必须同时满足：

1. `access_receipt.sha256` 与 `access_receipt.json` 一致，receipt/manifest/artifact hashes 全部复核通过；
2. actual windows=`10141`、actual `samp_id`=`32`、row-id SHA-256 等于冻结值；
3. target-window/target-sequence/proxy/sampling/grid rows 分别为 `10141 / 446204 / 40564 / 60846 / 243384`，每个 target window 固定含9个 Local-RR 与35个 log-effort sequence 点；
4. 全部 numeric column `n_infinite=0`，`n_finite+n_null=n_total`，null count 与 eligibility/censoring 一致；
5. access flags 全部满足 train-only/no-model/no-checkpoint/no-GPU 边界；
6. 固定14项产物齐全，目标目录此前不存在且未覆盖任何旧产物。

完整审计随后由用户从干净 commit `2ce00509184954ec764e9cf80041f6c5f6aadfda` 执行。Receipt SHA-256=`1dbdc18836c13d8c2cd888c186477c40b044e190afed3e8817c082bd9b235428`，manifest SHA-256=`19ab1ece34e40b0b970f1f620f20015742c7ae1de57df3f20a2f904403763eb2`，sample-direct summary SHA-256=`61b5c5d17b11b71731bf399ac2d12c684c724ac1aebe4979b577797a96b49483`。固定行数、`10141/32` train identity、row/content hashes、finite/null closure 与全部 access flags 均通过，审计输出未读取 validation、research-test 或 checkpoint，未训练/推理模型且未使用 GPU。

用户随后明确回复“确认上述 RTM-v1 lock 草案”。Signal substrate 固定为显式 anti-aliased `100→20 Hz`、20-Hz learned carrier-sensitive filtering + nonlinearity、再显式 anti-aliased `20→10 Hz`；不增加手工 analytic-envelope 输入。多尺度固定 `10/2/1 Hz`、只允许显式 low-pass + decimation，1 Hz只承担 context/effort，关闭0.5 Hz和普通 average pooling。Family 固定 T0 + TCN + BiMamba2 + BiLSTM + feature-level multiscale，关闭 global token mixer。规范性 lock 为：

- `docs/experiments/resp_temporal_v1_signal_substrate_lock_20260820.json`，SHA-256=`11bfcad00f4532d4bdfe1413a375b5f06f46eb8ac67dfcd475701872322fee69`；
- `docs/experiments/resp_temporal_v1_candidate_lock_20260820.json`，SHA-256=`b4a2c83310fa2ce9519e3ca25814aea0b179458ab52d6380a932545c99c25f9b`。

## 10. 当前停止线

- 完整审计已关闭，不得重复执行、删改或覆盖冻结产物；
- 科学 substrate/family/candidate 已锁定，但 exact implementation 尚未完成；不得把旧 implementation probe 原样升格为 formal candidate；
- 不授权 GPU engineering、formal training、validation summary 或 research-test；
- 不得借用 `train_tho.py`、`train_crd.py`、旧 evaluator 或已有 checkpoint 旁路推进；
- 下一阶段只允许实现 locked common substrate、10/2/1-Hz multiscale 与五项 strict configs，并运行轻量确定性 CPU 测试；GPU engineering 仍需新的用户授权。
