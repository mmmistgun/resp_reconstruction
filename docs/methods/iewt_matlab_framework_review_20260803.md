# IEWT MATLAB 代码与 Python 基线接入框架梳理

日期：2026-08-03

状态：2026-08-04 已将两处 1 Hz 低通改为零相位并通过定向测试与单 sample CPU smoke；等待重新运行完整 validation

## 1. 本文范围

本文记录两部分内容：

1. `hy_ewt/` 中现有 MATLAB 文件实际包含什么、各文件之间如何调用；
2. 若后续将其转换为 Python 传统方法基线，应如何接入当前 THO 呼吸重建仓库。

本文不修改数据、split、target、metrics、checkpoint 或 designated test 状态，也不把旧 MATLAB 结果迁移为当前实验结论。

用户描述的目录名是 `ly_ewt`，仓库中实际存在的目录名是 `hy_ewt`。下文均以实际目录为准。

## 2. 先给结论

新增目录 `hy_ewt/改进EWT6.0-草稿/` 已补上此前最关键的缺口。现在可以追踪出一条不依赖 target 的完整非可视化提取链：

```text
raw BCG
  -> pre（3 阶多项式 detrend + 1 Hz 三阶 Butterworth 因果低通）
  -> Test_EEWT1D（0–1 Hz 频谱上包络、平顶/平底与候选频带判定）
  -> Test_EEWT
  -> EEWT1D
  -> BoundariesDetect
  -> EEWT Meyer filter bank
  -> 单频带或多频带求和输出
```

`Test_EEWT1D.m`、`Test_EEWT.m`、`EEWT1D.m`、`BoundariesDetect.m`、全部 Meyer 函数、`pre.m`、`low.m` 和旧批处理辅助函数现已存在。因此，从“源码是否足以开始转换”看，答案是“具备实现条件”。

以下差异已识别并在 Python 协议中显式收口，不再构成实现阻塞：

- 原代码固定按 1000 Hz 设计预处理滤波器，当前输入是 100 Hz；
- 主批处理实际用 35 秒上下文生成每 30 秒输出，但同一草稿中存在两种不同裁剪/拼接写法；
- 多个阈值按频谱 bin 编号而非 Hz 编写，直接改成 180 秒整窗会改变科学含义；
- 当前环境没有 MATLAB/Octave，也没有配套 MATLAB 输入—输出 oracle，因此第一版不声明 Python 与 MATLAB 逐点数值等价；
- 草稿包含若干边界条件和旧调用签名问题，需要先决定是“忠实保留”还是“修复后定义新版基线”。

Python 实现、确定性测试和单 sample CPU smoke 现已通过。首轮因果版本完整 validation 暴露出 `94.99%` 的样本最佳 lag 命中 `±0.30 s` 边界，且理论群延迟与两次因果低通一致；用户因此在 2026-08-04 明确要求将两处低通改为零相位。旧因果结果只作诊断，不作为当前零相位 IEWT 结果。

### 2.1 已确认的 Python 基线适配协议（2026-08-03）

用户已确认以下方向：

- 实验和研究叙述沿用名称 `IEWT`；代码注释中保留原 MATLAB 函数使用 `EEWT` 命名的来源说明；
- 输入使用与当前深度学习相同的 `bcg_rawish_wideband_state_aligned_segment_soft_z`；
- 当前实现直接工作在 100 Hz，不把输入升采样回 1000 Hz；
- 预处理保持三阶多项式 detrend，以及按当前采样率重新设计的三阶、1 Hz Butterworth；2026-08-04 起使用 `sosfiltfilt` 零相位实现；
- 每个 180 秒 sample 拆为 6 个 30 秒输出块；
- 每块使用 35 秒上下文：第一块读取 `0–35 s` 并保留前 30 秒，后续块读取“左侧 5 秒 + 当前 30 秒”并丢弃左侧 5 秒；
- 6 块拼接后输出严格为 18000 点；
- IEWT 只读取 BCG，不读取 THO target，不按 target 选频带、修极性或调参数；
- 评价继续复用当前冻结的 `Pi`、五项 primary metrics、IBI/coverage、eligibility 与逐 sample direct mean；
- 暂不要求 MATLAB 数值对照。因此第一版只能声称是依据现有 MATLAB 源码定义的“协议化 Python IEWT 基线”，不能声称与某个 MATLAB 环境逐点等价。

历史 `DeriveRes.m` 在拼接后又使用同一个 1 Hz 低通处理完整输出。Python 基线保留该后处理位置，但与预处理一致改用 `sosfiltfilt` 零相位实现，以消除方法内部确定性群延迟；阶数和截止频率不变。

## 3. 当前 Python 项目框架

### 3.1 当前任务口径

当前唯一实验协议是 `docs/experiments/loss_metrics_restart_plan_20260729.md`。与传统基线接入直接相关的冻结口径为：

- 数据集：research v2；
- 输入配置列：`bcg_rawish_segment_soft_z_key`；
- 该列在当前索引中解析为整晚 NPZ 内的 `bcg_rawish_wideband_state_aligned_segment_soft_z`；
- target 配置列：`target_waveform_segment_soft_z_key`；
- 采样率：100 Hz；
- 每个样本：180 秒，即 18000 点；
- dataset admission、train/val/test subject/session 隔离关系保持不变；
- 正式评价前，prediction 与 target 都经过公共输出算子 `Pi = S o B`；
- `B` 是整窗 `0.05–0.70 Hz` 零相位硬 FFT 投影，`S` 是整窗尺度规范化；
- validation 使用五项 primary metrics 与 IBI/coverage 的冻结定义，逐 sample direct mean；
- designated test 继续封存，不能用于开发 IEWT 参数、频带或选带规则。

### 3.2 数据流

当前确定性传统方法适合沿用已有 `F0_fixed_band_bcg` 的生命周期：

```text
configs/tho_research_v2.yaml
        |
        v
build_window_data(..., split=val, shuffle=False)
        |
        v
ResearchV2WindowDataset
  x      : [B, 1, 18000]，100 Hz BCG 输入
  target : [B, 1, 18000]，THO waveform target
  meta   : dataset_row_id / split / samp_id / coupling_state_id / ...
        |
        v
确定性 IEWT 变换（无训练、无 checkpoint）
        |
        v
evaluate_task_predictions(...)
        |
        +--> sample_metrics.csv
        +--> summary.csv
        +--> resolved_config.yaml
        +--> run_manifest.json
```

数据加载相关实现位于：

- `resp_train/data/factory.py`：构建 index、dataset 和 DataLoader；
- `resp_train/data/research_v2.py`：按索引从整晚 NPZ 切出 18000 点输入和 target，并对 shape/finite 做强校验；
- `resp_train/protocols/respiration.py`：公共 `B`、`S`、`Pi`；
- `resp_train/metrics/task.py`：冻结 metrics、eligibility、失败值和 direct-mean summary；
- `resp_train/baselines/fixed_band.py` 与 `scripts/eval_tho_fixed_band_baseline.py`：现有确定性传统基线模板。

### 3.3 确定性基线的科研角色

IEWT 不训练，也没有随机初始化或 checkpoint selector。因此，在算法、输入、分窗和参数冻结后：

- validation 只产生一组确定性结果；
- 不构造多 seed mean ± SD；
- smoke 只证明实现可运行，不形成科研结论；
- 正式 validation 结果必须保留逐 sample 指标和完整运行清单；
- test 只能在最终方法集合全部冻结后显式执行。

## 4. `hy_ewt/` 文件组织

### 4.1 标准 EWT 核心

`hy_ewt/EWT/` 主要来自 Jerome Gilles 的 EWT 1D MATLAB 实现。当前默认路径真正会调用的文件是：

| 文件 | 作用 |
|---|---|
| `Test_EWT1D.m` | 包装参数、调用 EWT、按低频主峰选择一个子带作为呼吸输出 |
| `EWT1D.m` | 半谱边界检测、镜像扩展、Meyer 滤波器组分解 |
| `EWT_Boundaries_Detect.m` | 频谱预处理、正则化、边界检测方法分派 |
| `LocalMaxMin2.m` | 默认 `locmaxmin` 边界检测：选大峰，在相邻峰之间找最低谷 |
| `EWT_Boundaries_Completion.m` | 检测边界不足时补全边界 |
| `EWT_Meyer_FilterBank.m` | 根据边界生成 Meyer scaling/wavelet filter bank |
| `EWT_Meyer_Scaling.m` | 生成最低频 scaling filter |
| `EWT_Meyer_Wavelet.m` | 生成各频带 Meyer wavelet filter |
| `EWT_beta.m` | Meyer 过渡区使用的七次多项式 beta 函数 |
| `iEWT1D.m` | 全子带逆变换；当前呼吸提取入口默认不调用 |

默认配置下，`RemoveTrend.m` 和 `SpectrumRegularize.m` 都走 `none` 分支。尺度空间、Otsu、显示、HHT/EMD 等文件属于其他可选检测或可视化路径，不是 `Test_EWT1D.m` 当前默认呼吸提取路径的必要组成。

### 4.2 上层实验脚本

| 文件 | 观察到的用途 | 与 IEWT 的关系 |
|---|---|---|
| `normal6.m` | 正常呼吸数据的批处理、不同传统方法比较、呼吸间期误差与绘图 | IEWT 代码大多已注释；更像混合实验草稿 |
| `SA1.m` | 根据睡眠呼吸暂停标签构造事件片段 | 不直接实现 IEWT |
| `SA2.m` | 拼接暂停与正常片段，比较 MOR/IEWT/DWT/LPF 的呼吸间期 | 调用现已补充的 `Test_EEWT1D` |
| `SA3.m` | 60 秒、50% overlap，运动/无效片段筛除，比较暂停与正常段 RMS | 调用现已补充的 `Test_EEWT1D` |
| `untitled.m` | 多段探索代码：峰谷幅差 Spearman、方法比较、Bland–Altman 绘图 | 调用现已补充的 `Test_EEWT1D` |
| `movement_detectionCheyne.m` | 基于短窗频域能量和阈值的体动检测 | 当前 research v2 已有独立质量/admission 体系，不宜直接混入 |
| `segwithmovCheyne.m` | 避开体动位置切 30/60 秒片段 | 当前 180 秒 dataset 窗口已冻结，是否复用必须单独决定 |
| `resp_peaks2.m` | 正峰/负峰检测并清理连续同类型极值 | 属于旧实验评价逻辑，不应替换当前冻结 IBI detector |
| `FTpsd.m` | `periodogram` 功率谱封装 | 主要服务旧体动检测 |

`MOR.mat`、`RMS.mat`、`cor.mat` 和 Excel 文件是旧实验结果或中间产物；它们不是可执行算法定义，也不能直接当作当前 validation 证据。`频谱划分.fig/.jpg` 是算法思路图；新增源码现已能解释图中的主要步骤，但旧图本身仍不是数值 oracle。

### 4.3 新增 IEWT/EEWT 核心

新增目录已经把图中“上包络和平顶/平底分区”落实为代码：

| 文件 | 作用 | 当前判断 |
|---|---|---|
| `Test_EEWT1D.m` | 改进方法总入口；构造 `0–1 Hz` 频谱上包络、检测平顶/平底、判断单/多候选频带 | 核心已提供 |
| `Test_EEWT.m` | 调用 EEWT 分解并处理可选绘图/逆变换 | 核心已提供 |
| `EEWT1D.m` | 使用自定义边界而非标准 EWT 的 `locmaxmin` 边界，执行镜像扩展和滤波器组分解 | 核心已提供 |
| `BoundariesDetect.m` | 在相邻上包络平台之间寻找频谱谷值作为分区边界 | 核心已提供，但边界 tie 情况需验证 |
| `EEWT_Meyer_*`、`EEWT_beta.m` | 改名后的标准 Meyer filter bank 实现 | 与标准 EWT 数学实现一致 |
| `pre.m` | 三阶多项式 detrend 后调用 `low.m` | 核心预处理已提供 |
| `low.m` | 1000 Hz 下三阶、1 Hz 截止 Butterworth 低通 | 参数已明确，需适配当前 100 Hz |
| `windows_30.m` | 以 30 秒为输出步长，为每块提供额外 5 秒上下文 | 历史批处理语义已可观察 |
| `CrossFading.m` | 另一套 overlap/cross-fade 拼接探索 | 是否属于最终方法不明确 |

非可视化主路径的直接依赖已经闭合。`Comp=1` 的展示分支在 `Test_EEWT.m` 中调用了 `Show_EWT` 而不是同目录的 `Show_EEWT`，但正式基线可以固定 `Bound=0, Comp=0`，不让绘图分支进入算法依赖。

## 5. 当前可见的标准 EWT 算法

以下描述严格对应标准入口 `EWT/Test_EWT1D.m`，用于与第 6 节新增的改进 EEWT 入口对照。

### 5.1 包装参数

`Test_EWT1D(f, fs, Bound, Comp)` 固定：

```text
globtrend       = none
reg             = none
detect          = locmaxmin
N               = 6
completion      = 1
InitBounds      = [2, 25]   # locmaxmin 默认路径实际上不使用
log spectrum    = false
```

文件说明建议输入先做去趋势并低通到约 1 Hz。这个预处理不在 `Test_EWT1D` 内部执行，而是依赖调用者。

### 5.2 频谱边界检测

设输入长度为 `L`：

1. `EWT1D` 对输入做 MATLAB `fft`；
2. 取 `abs(ff(1:round(L/2)))` 作为待分区半谱；
3. `LocalMaxMin2(..., N=6)` 选择最大的局部峰，并在相邻峰之间寻找最低局部谷；
4. `EWT1D` 执行 `boundaries = boundaries(2:end)`；
5. 将剩余边界从半谱整数 index 归一化到 `[0, pi]`。

当前代码在典型路径中得到 4 个内部边界，对应 5 个 Meyer 子带。实际子带数仍应在 MATLAB 对照测试中确认，不能只依赖注释中的“maximum number of bands”。

### 5.3 Meyer 滤波器组

`EWT1D` 先镜像扩展输入，再在扩展信号 FFT 上应用实值 Meyer filter bank：

- 第一个 filter 是 `[0, w1]` 的 scaling function；
- 后续 filter 是相邻边界之间的 Meyer wavelet；
- 最后一个 wavelet 延伸到 `pi`；
- 过渡区宽度由所有相邻边界的最小相对间隔确定；
- 各分量由 `real(ifft(conj(filter) * fft(extended_signal)))` 得到；
- 最后裁掉镜像扩展，恢复原长度。

`EWT_beta(x)` 为：

```text
0                                      x < 0
x^4 * (35 - 84x + 70x^2 - 20x^3)      0 <= x <= 1
1                                      x > 1
```

### 5.4 呼吸子带选择

`Test_EWT1D` 不返回全部子带之和，而是只选择一个子带：

1. 再次计算原始输入的 FFT 幅值；
2. 只查看 `0–1 Hz`，具体 MATLAB slice 为 `1 : L/fs + 1`；
3. 找到该区间全局最大幅值 bin；
4. 根据该 bin 落在哪个 EWT 分区，返回对应的单一 EWT component。

这意味着算法的最终输出高度依赖调用前的去均值、去趋势和低通。如果 DC 未清除，0 Hz 可能成为最大峰并导致选择最低频分量。

## 6. 新增代码揭示的 IEWT 算法

### 6.1 上包络与平台检测

`Test_EEWT1D.m` 先计算输入的 `0–1 Hz` 单边幅值谱：

```matlab
y1 = abs(fft(presig)) .* 2 / L;
sig_fft = y1(1:L/fs+1);
```

随后：

1. 在 `sig_fft` 上找局部峰；
2. 取相邻峰最小距离 `mirror`，并强制 `mirror >= 3 bins`；
3. 令 `w = ceil((mirror-1)/2)`；
4. 对每个 bin 取半径 `w` 邻域最大值，构成滑动最大上包络 `upper`；
5. 从 `upper` 中找平台起点与严格局部低点；
6. 在这些候选点组成的稀疏序列上再次判断平台极大/极小，并向右延伸得到每个平台的 `[start, end]`。

这部分与 `频谱划分.jpg` 展示的“频谱上包络和平顶/平底”一致，说明新增目录确实包含此前缺失的改进点。

### 6.2 自适应边界与 EEWT 分解

`BoundariesDetect.m` 在每个平台末端和下一个平台起点之间寻找原始 `sig_fft` 的最低点；最后一个平台后若仍有频谱，则继续在尾段找最低点。边界传入 `EEWT1D.m` 后：

- 按原始输入长度归一化到 `[0, pi]`；
- 对输入做与标准 EWT 相同的镜像扩展；
- 构造 Meyer scaling/wavelet filter bank；
- 对每个分区分别 IFFT 并裁回原长度。

因此 IEWT 相对标准 EWT 的主要变化在**边界生成与呼吸频带选择**，Meyer 滤波器组本身没有改变。

### 6.3 单频/多频带选择

先以全局最高平台幅值的 50% 为阈值，保留候选平台。若候选超过一个，算法把输入分成四个无重叠时间块做 spectrogram，并结合两类判据：

- 各时间块在候选平台中心频率上的强峰是否唯一；
- 各候选 EEWT 区间内的局部谱峰是否在多数时间块中支持同一或多个区间。

判为多频结构时，输出相应多个 EEWT component 的和；否则进入“倍频”分支，按候选平台中心 bin 选择一个 component。`beipin/duopin` 等额外输出只记录旧实验片段编号，不参与输出波形计算。

### 6.4 预处理已明确

`pre.m` 的有效代码是：

1. `detrend(sig, 3)`：移除三阶多项式趋势；
2. 使用 `low.m` 设计的三阶 Butterworth 低通；
3. 用 MATLAB `filter` 单向因果滤波，不是 `filtfilt`。

`low.m` 固定 `Fs=1000 Hz`、`Fc=1 Hz`。因此原 IEWT 的相位和启动瞬态包含因果 IIR 语义。当前项目若改成 100 Hz 重新设计同阶同截止滤波器，属于保持物理参数的适配，不是逐点复用原系数。

### 6.5 历史分窗与拼接

`windows_30.m` 名称容易误导：当 `num=30000` 时，它实际输出 35000 点，即 35 秒上下文窗：

- 第一块读取 `0–35 s`；
- 后续每块读取“前 5 秒 + 当前 30 秒”；
- 输出步长仍是 30 秒。

`DeriveRes.m` 中至少出现两种拼接策略：

1. 第一块保留前 27.5 秒，后续块去掉头尾各 2.5 秒；
2. 第一块保留前 30 秒，后续块丢弃开头 5 秒后保留 30 秒。

第二种策略能自然恢复 `rows * 30 秒` 的总长度，且更接近“5 秒上下文、30 秒输出”的定义。Python 基线已明确采用第二种策略；这是协议化适配选择，不声称它是所有历史 MATLAB 草稿的唯一行为。

## 7. 不能静默继承的旧实验逻辑

旧 MATLAB 脚本和当前协议的统计对象不同，不能整段翻译后直接称为当前基线：

| 旧 MATLAB 逻辑 | 当前仓库口径 | 处理原则 |
|---|---|---|
| 1000 Hz 原始 BCG | 100 Hz、18000 点 research v2 soft-z 输入 | 必须明确重采样/输入层级，不默认等价 |
| IEWT 主批处理常用 35 秒上下文窗、30 秒输出步长；另有 60 秒探索 | 冻结的 180 秒 sample | 必须冻结 full-window 或分块/拼接策略 |
| 自定义体动检测后重新切片 | dataset admission 已冻结 | 不在 IEWT 内新增第二套样本排除规则 |
| `resp_peaks2` 与 2–7 秒 RI 过滤 | 当前冻结 RR/IBI detector 与 eligibility | 只把 IEWT 当 waveform prediction，评价复用现行 metrics |
| `abs(corr(..., Spearman))` | signed PCC 与冻结 effort Spearman | 不复用旧绝对相关评价 |
| 旧 Bland–Altman、Eabs/Erel/Pr | 五项 primary + IBI/coverage | 旧统计只作历史背景 |
| 按 prediction 行为删点或插值事件序列 | eligibility 只能由 target 决定 | prediction 失败必须按冻结最差值计入 |

尤其不能把旧体动检测、旧 RI 过滤或旧峰谷逻辑塞进当前 DataLoader，从而改变 validation 样本集合或让 prediction 逃避评价。

## 8. Python 接入建议

### 8.1 建议的代码边界

进入 Python 转换时，建议新增：

```text
resp_train/baselines/iewt.py
scripts/eval_tho_iewt_baseline.py
tests/test_iewt.py
tests/test_iewt_baseline.py
```

职责建议为：

- `iewt.py`：纯 NumPy/SciPy 的确定性算法和 loader 评价适配；
- `eval_tho_iewt_baseline.py`：只负责参数解析、构建 validation loader、保存产物；
- `test_iewt.py`：边界、filter bank、子带选择、确定性和异常输入测试；后续若恢复 MATLAB 对照，可在此补充 parity fixture；
- `test_iewt_baseline.py`：验证数据键、metadata、metrics 复用、输出产物与 test 封存。

算法函数应保持纯函数，不直接读取仓库数据，不在内部访问 target，也不根据 validation/test 调参。建议的最小接口是：

```python
def extract_respiration_iewt(
    signal: np.ndarray,
    *,
    fs: float,
    config: IEWTConfig,
) -> IEWTResult:
    ...
```

`IEWTResult` 至少保留：

- `waveform`：与输入等长的输出；
- `boundaries_bins` / `boundaries_hz`；
- `selected_band_index`；
- 可选的 `components`，仅用于单测或 smoke 诊断，正式逐 sample CSV 不保存大数组。

### 8.2 输入与输出约束

Python 实现应显式保证：

- 接受 `[N]` 和 batch 外层逐 sample 调用，时间轴含义固定；
- 输入、边界、components 和输出全部 finite，否则立即失败；
- 输出长度严格等于输入长度；
- 不读取 target；
- 不根据 prediction 失败删除 sample；
- 保留 `dataset_row_id` 等追溯字段；
- 正式评价继续调用 `evaluate_task_predictions`，让公共 `Pi` 和所有 metrics 保持一致。

### 8.3 性能边界

标准 EWT 对每个 sample 至少需要多次 FFT、边界扫描和多个子带 IFFT。完整 validation 有 2675 个 180 秒窗口，适合：

- batch loader 流式读取，避免完整波形重复常驻内存；
- 先用少量合成信号和固定真实样本做正确性验收；
- 正确性冻结后再考虑向量化或缓存；
- 优化前后必须保持边界和输出数值一致。

本方法不需要 GPU。正式全量 validation 是否耗时较长，应在 smoke 后给出实测，不预先启动后台或长时间任务。

## 9. 实现前必须冻结的问题

名称、输入、采样率适配、35 秒上下文、30 秒输出步长、拼接方式、target 隔离和“不要求 MATLAB 逐点对照”均已确认。由于 100 Hz 实现仍保持 35 秒上下文，`0–1 Hz` 一共仍是 36 个频谱 bin，原代码的 `freq > 6` 保持相同物理频率含义，无需改写成新的经验阈值。

以下实现口径已一并确认：

1. 保留 `DeriveRes.m` 中拼接后的第二次整段 1 Hz 低通；2026-08-04 起两处低通均冻结为零相位 `sosfiltfilt`。
2. 边界条件采用“按注释意图做最小确定性修复”：局部区间并列谷值取最左者、无平台时选第一 EEWT mode 并令 `I=1`、boundary 非空/严格递增检查失败时显式报错、任何 sample 都不静默删除。

至此不存在阻止开始实现的协议 blocker。正式 validation 仍必须等实现、定向测试和小规模 smoke 通过后再运行。

## 10. 建议的分阶段转换顺序

### 阶段 A：冻结实现契约与确定性样例

- 以新增 `Test_EEWT1D.m`、`Test_EEWT.m`、`EEWT1D.m`、`BoundariesDetect.m` 和 `pre.m` 为算法来源；
- 将本文确认的 100 Hz、35 秒上下文、30 秒输出、零相位滤波和异常语义写成不可变配置默认值；
- 选取不涉及 test 的合成信号和少量 train 窗口，形成 Python 确定性回归样例；
- 核实原始 EWT 代码的许可与引用要求。

### 阶段 B：移植纯算法

- 先实现 MATLAB 索引语义、局部极值/平顶规则和并列规则；
- 再实现镜像扩展、Meyer beta、scaling/wavelet filter bank；
- 最后实现 IEWT 特有边界检测与选带；
- 每一步使用可计算的不变量和固定回归样例验收；暂不把 MATLAB parity 作为实现门槛。

### 阶段 C：接入当前 baseline 生命周期

- 复用 research v2 admission 与 validation split；
- 只从 BCG 输入生成 prediction；
- 复用公共 `Pi`、metrics、summary 和 metadata；
- 保存 resolved config、manifest、逐 sample 指标和诊断字段；
- 继续拒绝未确认的 designated test。

### 阶段 D：验证

最低验收集合：

- 常量、单频、双频、接近边界频率、奇偶长度和极短输入；
- boundaries 严格递增、selected band 合法、components 可重建，以及固定输入输出回归；
- 输入/输出 shape 与 finite guard；
- 同一输入重复运行完全一致；
- loader smoke 不改变 sample 数和 metadata；
- 当前协议定向测试仍通过。

### 阶段 E：validation 基线

只有算法定义、输入层级、分窗策略、参数、定向测试和 smoke 全部冻结后，才运行完整 validation。结果作为新的确定性传统基线独立报告，不与旧 MATLAB 表格或历史 run 混合，也不声明 MATLAB 逐点等价。

## 11. 已识别的实现风险

### 11.1 MATLAB 索引与并列规则

`LocalMaxMin2.m` 使用 1-based index、包含端点的 slice、`sort(..., 'descend')` 和对平谷中心的 `round`。Python 的 0-based index、切片右开边界、排序稳定性和 rounding 规则不同，必须逐项对照。

### 11.2 输入方向

`EWT1D.m` 的镜像拼接按列向量书写。旧脚本有时显式转置、有时不转置。Python 端应统一压成一维时间序列，拒绝含糊的二维方向。

### 11.3 边界补全代码的单位疑点

`EWT_Boundaries_Completion.m` 用 `pi` 补全边界，但调用位置上的 boundary 仍表现为半谱 index。默认路径可能因为总能返回足够数量而没有触发该分支。Python 版应对“局部峰不足”构造定向测试，采用显式失败或已冻结的确定性 fallback，并避免声称逐点等价。

### 11.4 最大峰非唯一

`Test_EWT1D.m` 的 `find(sig_fft == max(sig_fft))` 可能返回多个 index，后续 `if` 假设标量。Python 版本统一采用最左 index，并把它作为协议化确定性规则；不将这一选择描述为未经验证的 MATLAB 逐点复现。

### 11.5 预处理与 DC

选带搜索包含 0 Hz，而 `Test_EWT1D` 自身不去均值。旧说明要求调用前去趋势并低通 1 Hz，证明预处理属于算法契约的一部分。当前 soft-z 输入不等于已证明 DC/趋势语义完全一致。

### 11.6 当前数据的对齐背景

当前输入 key 名称包含 `state_aligned`，索引还记录 `state_alignment_method`、`state_alignment_lag_s` 和 `state_alignment_is_reference_assisted`。IEWT baseline 应像现有 fixed-band baseline 一样把这些字段附到逐 sample 结果，避免把上游对齐收益错误归因于 IEWT 本身。

### 11.7 旧代码并非可直接运行的软件包

`normal6.m` 和 `untitled.m` 混合了不同日期的实验块、硬编码 Windows 绝对路径、已注释分支和绘图代码；部分变量依赖工作区已有状态。它们适合作为研究过程记录，不适合逐行翻译成 Python 入口。Python 基线应从明确的算法函数和当前数据生命周期重新组织。

### 11.8 新增草稿存在签名漂移

当前 `Test_EEWT1D` 需要 5 个参数，但 `DeriveRes.m` 和新增 `Untitled.m` 的部分旧代码只传 4 个参数；`pre.m` 只接受 1 个参数，而 `Untitled.m` 有传入 `fs` 的旧调用。这说明目录保留了不同时间的脚本片段，不能用“所有 `.m` 文件均可直接运行”作为完整性标准。核心候选接口应冻结为：

```matlab
presig = pre(signal);
[eewt, upper, sig_fft, boundaries, ..., I] = ...
    Test_EEWT1D(presig, fs, 0, 0, data_piece);
```

其中 `data_piece` 只服务旧诊断标签，在 Python 正式接口中可替换为可选 metadata，不应影响波形。

### 11.9 新增核心的边界条件风险

静态阅读发现以下需要定向测试覆盖的路径：

- `BoundariesDetect` 用“全谱等于区间最小值”寻找边界；若相同幅值在别处重复，可能返回多个位置或错误位置；
- `Test_EEWT1D` 在 `max_start` 为空时没有给输出 `I` 赋值；
- 候选区间完全找不到局部峰时，后续数组索引可能进入非法的 0 index 路径；
- EEWT filter bank 要求 boundary 非空、严格递增且位于 `(0, pi)`，当前入口没有集中验证；
- `W=L/4`、`L/fs+1` 和 spectrogram 的 4 个时间块隐含 `L` 可被 4 整除且 `L/fs` 为整数；
- “倍频”分支使用 `freq > 6` 的 bin index，而不是采样率无关的 Hz；
- MATLAB 的 1 Hz 低通是因果 `filter`；Python 在首轮 validation 诊断后经用户明确确认改为零相位 `sosfiltfilt`，因此不再声称保留 MATLAB 相位语义。

这些风险不代表算法不可转换，但说明 Python 版必须先定义输入契约与失败语义，不能通过静默丢弃异常 sample 来让完整 validation 跑通。

## 12. Python 实现与验收记录

新增实现：

- `resp_train/baselines/iewt.py`：三阶多项式去趋势、当前采样率下的三阶 1 Hz 零相位 Butterworth、频谱上包络、平台检测、自适应边界、Meyer filter bank、单/多频带选择、35/30 秒拼接和整段零相位后低通；
- `scripts/eval_tho_iewt_baseline.py`：research v2 validation/test 生命周期、产物保存和 designated-test 显式确认；
- `tests/test_iewt.py`：边界并列规则、Meyer filter bank、确定性、component 选带重建、35/30 秒拼接及非法输入；
- `tests/test_iewt_baseline.py`：输入层级、BCG-only 评价、冻结 metrics 复用与可追溯 summary。

实现采用本文冻结的确定性修复，不声称逐点复现某个 MATLAB 版本：局部并列谷值取最左者，无有效 boundary 显式失败，无平台时选首个 mode，所有 shape/finite 违约立即报告。

验收结果：

- `tests/test_iewt.py tests/test_iewt_baseline.py`：`12 passed`；
- 当前全量测试：`267 passed`；
- 单个真实 validation sample 的 CPU 端到端 smoke 通过，读取的 source key 为 `bcg_rawish_wideband_state_aligned_segment_soft_z`，生成逐 sample metrics、summary、resolved config 和 manifest；该结果仅用于实现验收。

本轮没有：

- 启动训练、GPU、完整 validation 或 designated test；
- 改变数据、split、target、admission、metrics 或输出格式；
- 把旧 `.mat/.xls/.xlsx` 结果写入当前协议结论；
- 覆盖 `hy_ewt/` 中任何用户文件。
