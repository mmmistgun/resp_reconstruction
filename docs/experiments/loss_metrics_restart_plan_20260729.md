# Loss 与 Metrics 重构实验记录

日期：2026-07-29

最后更新：2026-08-02

状态：实现、GPU batch 128 验收与 `B0_pilot` 已通过；正式三 seed 预算冻结为 50 epochs，尚未启动

## 1. 定位

本文用于从头定义下一阶段实验的 loss、metrics 及二者之间的对应关系。

本轮设计不继承此前实验路线、默认 loss、主指标、排序规则、checkpoint 选择逻辑或既有结论。代码与文档不另建 `archive/`：旧版本依靠 Git 历史追溯，历史 run 原地保留但不进入新实验比较。

Loss、metrics、聚合方式和 checkpoint 选择规则已经完成一致性检查并按冻结定义实现。本次只运行了定向单测、轻量生命周期验收与一次性 CPU `B0_smoke`；不启动 pilot 或正式训练，也不重评历史 checkpoint。

## 2. 当前目标与边界

当前要完成：

1. 明确模型训练真正需要优化的目标。
2. 定义 loss 的组成、数学形式、适用样本、归一化方式和权重策略。
3. 明确实验成功需要由哪些 metrics 证明。
4. 定义每个 metric 的计算对象、信号预处理、统计单位、聚合方式和解释边界。
5. 对齐训练目标、验证评价、checkpoint 选择和最终结论，避免四者口径割裂。
6. 在设计冻结后，形成代码修改清单、测试清单和首批最小实验矩阵。

实现阶段仍不做：

- 不启动正式训练、designated test、数据重算或历史结果迁移。
- 不为旧 loss、metrics、checkpoint、runner 或配置增加兼容层。
- 不默认复用旧 metrics、旧 checkpoint 排序或旧 baseline。
- 不在 loss 和 metrics 尚未冻结时讨论具体模型结构调参。

### 2.1 验证生命周期

本文中的合成信号、边界条件和确定性样例验证属于**一次性实现验收**，只在以下情况执行：

- 新 loss/metric 首次实现；
- 数学定义、预处理、mask、聚合或边界语义发生变化；
- 相关代码修改后运行对应回归测试。

它们不是每次训练开始前的检查项。一次性验收通过并固化为测试后，普通实验启动不重复运行整套合成验证；单次 run 只执行训练流程本身必需的配置解析、数据/shape/finite 断言和产物记录。

### 2.2 已冻结的数据与任务口径

本轮继续使用 `configs/tho_research_v2.yaml` 对应的数据口径：

- 数据集：2026-06-20 research v2 soft-z。
- 输入：`bcg_rawish_segment_soft_z_key`。
- target 载体：`target_waveform_segment_soft_z_key`。
- 采样率：100 Hz。
- 单样本长度：180 秒，即 18000 点。
- split：沿用当前 train/val/test subject/session 隔离关系，不重新划分。
- dataset admission：沿用 `filter_unusable=true` 的既有固定规则，即 `allowed_losses` 包含 `waveform`、`reason` 为空、`hard_valid_ratio≥0.80`、`state_alignment_valid_ratio≥0.80`；这是数据口径本身，不是每次训练前重新拟合的阈值。

上述 admission 由 dataset 构造/加载阶段统一完成；metrics 不重复应用或发明第二套质量 mask，只在已经 admission 的样本上追加各 metric 明确定义的 target-only dynamic 或事件资格。

2026-08-02 在当前配置上完成一次性索引与 split 独立性审计，未读取波形内容：admission 后 train/val/test 分别为 `10141 / 2675 / 2310` 个窗口；train–val、train–test、val–test 的 `samp_id` 与 segment overlap 均为 0。该审计只在本次实现验收以及未来数据/split 口径变化时重跑，不并入每次训练启动流程。

现有 `test` split 已在历史实验中产生过评价结果，因此本轮不能把它重新称为“从未触碰的 held-out test”。在不改变既有 split 的前提下，本文统一称其为 **designated test split**：新协议仍禁止读取其 target 来设计频带、阈值或 detector，并只在新方案全部冻结后评价一次，但结论必须承认它带有历史观察背景。若未来需要严格的 untouched final test，必须另立数据划分任务并获得新的未触碰 subject/session；本轮不静默改变 split。

本轮任务目标不是逐点复制原始 THO waveform，也不恢复绝对物理幅值或带外细节，而是恢复：

1. 全局与局部呼吸节律；
2. 呼吸频带内的低频形态和正确极性；
3. 相对呼吸强弱变化。

## 3. 重启声明与历史结果处理

本轮实验采用新的实验口径。以下内容默认全部重新定义：

- 训练 loss。
- validation/test metrics。
- 主指标、护栏指标与诊断指标的角色。
- metric 的预处理、mask、有效性条件和聚合方式。
- checkpoint 选择、early stopping 和最终模型选择规则。
- baseline 和候选实验之间的比较协议。

历史 run、checkpoint、summary 和旧指标整体退出新实验比较，不提供兼容迁移、批量重算或旁路重评支持。历史实验产物继续原地保留；旧代码、旧配置和旧说明由 Git 历史溯源，不在当前工作树重复归档。

新实验 baseline 必须在新 loss、metrics 和 checkpoint 规则下重新训练。若未来确实需要恢复某项历史比较，应另立独立任务说明目的和成本，不在当前设计中预留兼容分支。

## 4. 设计问题的第一性拆分

### 4.1 可观察量

待明确哪些量可以从输入、target、预测和元数据中可靠观察：

- 输入中真实可辨认的呼吸信息是什么？
- target 中哪些属性稳定可信，哪些存在噪声、时延、极性或多解？
- 哪些波形属性可以直接测量，哪些只能通过代理量评价？
- 样本质量、可学习性和评价有效性如何观察？
- metric 失败时，能否区分模型错误、target 不可靠和评价器错误？

### 4.2 可控制量

待明确实验中允许主动控制的变量：

- loss 项及其权重。
- loss 生效的样本、时间段、频带或质量条件。
- target 和 prediction 的预处理。
- 训练阶段、curriculum、采样和权重调度。
- metric 的有效性门槛、分层和聚合方式。
- checkpoint 选择和停止条件。

### 4.3 必须保证的性质

新方案至少应保证：

- loss 优化方向与最终任务目标一致。
- 主 metric 能直接回答实验是否成功，而不是只衡量易优化的代理量。
- metric 不因极少数无效窗口、错误峰值或聚合方式而产生误导。
- 均值改善不能掩盖关键子集或长尾的系统性退化。
- checkpoint 选择规则与正式结论使用相同的核心口径。
- 每项核心 metric 都有可验证实现、边界条件和人工波形复核路径。
- 新实验能够追溯配置、数据划分、seed、代码版本、checkpoint 和逐样本结果。

## 5. Loss 设计区

状态：第一版数学定义、参数和边界语义已冻结并实现；确定性回归验收已通过

设计来源：`docs/temp/loss 20260729.md`。本节将该草稿整理为正式实验设计；局部节律尺度由草稿中的 40 秒调整为 30 秒，其他尚未明确的实现细节不从旧代码中默认继承。

### 5.1 训练目标定义

模型可以在网络内部产生原始 head 输出 $\hat y$，但本任务的规范化重建结果定义为 $\hat x=\Pi(\hat y)$。Loss/metric 的**数值计算对象**是 canonical $\hat x/x$；规范化前的 target 频带信号 $b=B(y)$ 只用于 target eligibility，raw/band 中间量只用于 finite guard。只有 $\hat x$ 正式导出；$\hat y$ 不解释为原始 waveform 重建结果。因此带外、DC、整窗绝对尺度和原始传感器 waveform 都不属于本任务的可辨识目标。

Loss 不再要求一个通用波形误差同时承担所有任务，而是精简为四个互不重复的作用域：

1. `L_sync`：负责呼吸频带波形的有符号同步，容忍很小的传感器延迟。
2. `L_rhythm`：负责全局和局部呼吸节律，不约束绝对能量。
3. `L_effort`：负责相对呼吸努力随时间的变深、变浅趋势，不要求固定线性幅值映射。
4. `L_pol`：只在训练早期锚定极性，随后退火到零，避免长期与 signed PCC 重复。

核心目标可以暂时概括为：在允许不超过 0.3 秒小延迟的前提下，恢复方向正确的呼吸频带波形、全局与 30 秒局部节律，以及相对努力趋势。

本版暂不把点对点 raw-waveform 重建、绝对幅值、逐事件拓扑或局部 RR 回归加入核心 loss。它们后续只能在证明具有独立作用后再进入，不能重新堆成冗余 loss 集合。

### 5.2 公共定义与统一作用域

设模型原始 head 输出为 $\hat y$，soft-z target 载体为 $y$。$B(\cdot)$ 是固定、无参数、可微的呼吸频带投影；$S(\cdot)$ 用一个整窗尺度规范化表示；二者组成任务输出算子 $\Pi=S\circ B$：

$$
\hat b=B(\hat y),
\qquad
b=B(y),
\qquad
\hat x=\Pi(\hat y)=S(\hat b),
\qquad
x=\Pi(y)=S(b)
$$

`B` 不来自旧训练代码，也不作用于模型输入；它是本轮为了明确输出空间而新增的确定性 terminal projection，位于模型 raw head 之后、所有 loss/metrics 之前。其作用是删除本任务不评价的 DC 与带外分量。本轮冻结为整窗、可微、零相位 FFT 频带投影：

$$
B(u)
=
\operatorname{irfft}
\left[
M(f)\operatorname{rfft}(u-\bar u)
\right]
$$

其中：

$$
M(f)=
\begin{cases}
1,&0.05\leq f\leq0.70\ \mathrm{Hz}\\
0,&\text{otherwise}
\end{cases}
$$

由于任务只要求窗内相对努力，不恢复跨窗口绝对幅值，再定义：

$$
S(b)
=
\frac{b-\bar b}
{\sqrt{\frac{1}{N}\sum_t(b_t-\bar b)^2+\epsilon_{\mathrm{scale}}}},
\qquad
\epsilon_{\mathrm{scale}}=10^{-8}
$$

这一步只固定不可辨识的整窗增益；同一个 180 秒窗口内的相对强弱变化仍完整保留。对近零输出，分母中的 $\epsilon_{\mathrm{scale}}$ 只负责避免除零，且该输出仍会被同步、节律、努力和无效输出规则惩罚；它不等于已经证明近零处梯度温和。

投影和尺度规范化都在完整 180 秒、18000 点上执行。`B` 的频带端点包含在内，`n_fft=18000`，不 padding、不使用可训练参数，并关闭 AMP。`rfft/irfft` 固定使用 `norm="backward"` 的默认配对和 one-sided 实谱。先对整窗执行一次 $\Pi$，局部 loss/metrics 再从 $\hat x$ 和 $x$ 中切窗；不对每个局部窗重复滤波或重新做整窗尺度规范化。Loss、validation 和 test 必须使用同一实现语义。

这里明确接受硬矩形 DFT mask 的**循环边界全局投影**语义：窗口末端可以影响开头，并可能产生长程 ringing，$\pm0.3$ 秒中央裁剪也不声称能消除它。选择它是因为本轮定义的是固定 180 秒离线频带表示，而不是普通有限支撑滤波器；因此结果不得外推为 streaming/causal 重建能力。若未来做流式任务，必须另立输出算子和边界协议。

#### 5.2.1 Train-only 频带审计与选择

频带选择只使用当前配置按 `train_sample_seed=20260610` 固定抽样的 1024 个 train 窗口，不读取 val/test target。审计结果：

- `0.05–0.10 Hz` 能量占 `0.05–0.70 Hz` 的 median 为 1.75%，p95 为 15.51%，p99 为 30.29%。
- `0.05–0.70 Hz` 下整窗主峰低于 6 bpm 的窗口为 `20/1024`，占 1.95%；切换到 `0.10–0.70 Hz` 后，主峰变化集中在这 20 个窗口。
- 人工复核两个受影响最大的 train 窗口，低于 0.10 Hz 的主峰主要来自孤立大瞬态和缓慢回基线，不像可信的持续慢呼吸。

上述审计曾支持把下限提高到 `0.10 Hz`，但当前导出索引没有 apnea 类型、事件起止或 apnea burden 字段，无法证明所有 `0.05–0.10 Hz` 成分都与连续 OSA 无关。为避免在任务输出中不可逆删除潜在慢变化，也避免为 waveform、loss、RR、IBI 和 effort 维护多套频带，本轮最终统一选择 `0.05–0.70 Hz`。这不等于把所有低频峰都认定为可靠呼吸；低频漂移、短窗周期数不足和 OSA/慢呼吸混淆作为统一频带的已知解释边界保留，但不再通过拆分频带处理。

四项 loss 的作用域固定如下：

| loss 项 | 信号域 | 时间作用域 | 负责内容 | 明确不负责 | 生效阶段 |
|---|---|---|---|---|---|
| $\mathcal L_{\mathrm{sync}}$ | 呼吸频带波形 | 整个 180 秒窗口，小延迟搜索范围 $\pm0.3$ 秒 | 有符号波形同步、整体形态方向、小延迟容忍 | 局部节律分布、绝对幅值 | 全训练阶段 |
| $\mathcal L_{\mathrm{rhythm}}$ | 呼吸频带归一化功率谱 | 全局 180 秒 + 局部 30 秒窗口 | 整体和局部节律、主频/谐波分布 | 绝对能量、相位、极性 | 全训练阶段 |
| $\mathcal L_{\mathrm{effort}}$ | 对齐后的呼吸频带 log-RMS 包络 | 整个 180 秒趋势，10 秒包络、5 秒采样 | 相对努力变深/变浅趋势 | 绝对幅值标定、快速波形细节 | 全训练阶段 |
| $\mathcal L_{\mathrm{pol}}$ | 对齐并按窗标准化后的呼吸频带波形 | 中央共同有效区间 | 训练早期极性锚定 | 长期形态优化、节律 | 前 15% optimizer steps，随后为零 |

这里的 `B` 是本轮新任务定义中的频带投影，$\Pi=S\circ B$ 才是正式输出算子；二者都不静默沿用旧配置中的 Butterworth 阶数、`0.05 Hz` 下限或其他历史滤波参数。

### 5.3 小范围延迟同步损失

以下所有时间样本索引均采用 Python/NumPy 的零基约定 $t\in\{0,1,\ldots,N-1\}$。采样率为 100 Hz，最大 lag 为 0.3 秒，因此使用整数采样点网格：

$$
\mathcal K=\{-30,-29,\ldots,29,30\},
\qquad
\tau_k=\frac{k}{100}\text{ s}
$$

正 $k$ 表示预测相对 target 滞后：将较晚的预测样本 $\hat x_{t+k}$ 与 target 的 $x_t$ 比较。所有候选 lag 统一使用中央共同支撑区间：

$$
\mathcal I=\{30,31,\ldots,N-31\}
$$

不为不同 lag 使用不同长度的 overlap，也不 padding。对每个样本、每个候选 lag，令：

$$
u_{k,t}=\hat x_{t+k},
\qquad
v_t=x_t,
\qquad t\in\mathcal I
$$

计算有符号 PCC：

$$
c_k=
\frac{
\sum_t
\left(u_{k,t}-\bar u_k\right)
\left(v_t-\bar v\right)
}{
\sqrt{\sum_t\left(u_{k,t}-\bar u_k\right)^2+\epsilon_{\mathrm{corr}}}
\sqrt{\sum_t\left(v_t-\bar v\right)^2+\epsilon_{\mathrm{corr}}}
}
$$

实现时把 $c_k$ 截断到 $[-1,1]$，关键相关计算关闭 AMP，并取 $\epsilon_{\mathrm{corr}}=10^{-8}$。

Target 或 prediction 出现 NaN/Inf 都视为数据/计算错误并使训练立即失败，不能用 eligibility 静默吞掉。对有限 target，只有规范化前频带信号 $b=B(y)$ 在中央共同区间的中心化总体方差大于 $10^{-8}$ 时，该样本才进入 `L_sync`；这一低动态资格不依赖 prediction，也不会被 $S$ 放大后绕过。Loss 端不设置 prediction-variance 硬分支：所有有限 prediction 都按稳定公式计算；严格常量时 $c_k=0$，近常量但仍含形态时可以得到非零相关。

训练期按样本直接选择无惩罚的最佳 signed PCC：

$$
 k^*_{\mathrm{train}}
 =
 \arg\max_{k\in\mathcal K}c_k,
\qquad
\tau^*_{\mathrm{train}}=k^*_{\mathrm{train}}/100
$$

若并列，依次选择 $|k|$ 更小者、再选择数值更小的 $k$。索引选择使用 stop-gradient，梯度只通过被选中的相关值传播。同步损失定义为：

$$
\boxed{
\mathcal L_{\mathrm{sync},i}
=
1-c_{i,k^*_{\mathrm{train}}}
}
$$

完全一致且某个允许 lag 下 $c=1$ 时，该 sample 的 `L_sync=0`。先逐 sample 计算，再只对 `L_sync` target-eligible sample 取 batch 算术均值；若 batch 中没有 eligible sample，则返回与计算图相连的 0 并记录计数。

后续训练 loss 统一使用同一个 hard best-lag 对齐信号：

$$
\hat x^{\,a}_t=\hat x_{t+k^*_{\mathrm{train}}},
\qquad
x^{a}_t=x_t,
\qquad t\in\mathcal I
$$

`L_effort` 和 `L_pol` 统一使用 $(\hat x^a,x^a)$，不允许各自重新选择延迟；它们的 target eligibility 必须先包含 `L_sync` 的中央共同区间 dynamic 条件，因此不会对 sync-ineligible sample 使用任意 $k$。这里必须使用 signed PCC，不能使用 $|c_k|$ 或 $c_k^2$，否则极性翻转会被错误视为等价解。

评价期另行定义 $\tau^*_{\mathrm{eval}}$。训练与评价使用同一个无惩罚 lag 含义，但仍保留不同名称以区分生命周期；lag 只用于容忍残余对齐误差，不作为需要模型最小化的科学目标。

### 5.4 全局与 30 秒局部节律频谱损失

对时间尺度 $s$，在呼吸频带 $\mathcal B$ 内定义归一化功率谱：

$$
P^{(s)}_{t,f}(x)
=
\left|
\operatorname{STFT}_s(x)_{t,f}
\right|^2
$$

$$
p^{(s)}_{t,f}(x)
=
\frac{
P^{(s)}_{t,f}(x)+\epsilon_{\mathrm{power}}
}{
\sum_{g\in\mathcal B}\left(P^{(s)}_{t,g}(x)+\epsilon_{\mathrm{power}}\right)
}
$$

频谱分布误差采用频率维总变差距离。对 batch 中第 $i$ 个样本和尺度 $s$，令 $\mathcal G_i^{(s)}$ 为只由 target 决定的 eligible frame 集合：

$$
\mathcal L_{\mathrm{spec},i}^{(s)}
=
\frac{1}{|\mathcal G_i^{(s)}|}
\sum_{t\in\mathcal G_i^{(s)}}
\frac{1}{2}
\sum_{f\in\mathcal B}
\left|
p^{(s)}_{t,f}(\hat x)
-
p^{(s)}_{t,f}(x)
\right|
$$

只有 $|\mathcal G_i^{(s)}|>0$ 的样本产生该尺度的 sample loss。再令 $\mathcal H^{(s)}$ 为这些 sample 的集合：

$$
\mathcal L_{\mathrm{spec}}^{(s)}
=
\frac{1}{|\mathcal H^{(s)}|}
\sum_{i\in\mathcal H^{(s)}}
\mathcal L_{\mathrm{spec},i}^{(s)}
$$

即 local frame 先在 sample 内等权，再让 eligible sample 在 batch 内等权；global 和 local 各自没有 eligible sample 时，各自返回 graph-connected 0 并记录有效数。

总节律损失固定为全局与局部等权：

$$
\boxed{
\mathcal L_{\mathrm{rhythm}}
=
\frac{1}{2}\mathcal L_{\mathrm{spec}}^{(180s)}
+
\frac{1}{2}\mathcal L_{\mathrm{spec}}^{(30s)}
}
$$

本轮冻结的谱参数为：

| 尺度 | window | hop | `n_fft` | 帧数 | 其他 |
|---|---:|---:|---:|---:|---|
| 全局 | 180 秒 / 18000 点 | 不适用 | 18000 | 1 | 起点 0，整窗单帧 |
| 局部 | 30 秒 / 3000 点 | 10 秒 / 1000 点 | 3000 | 16 | 起点为 0、10、…、150 秒，`center=False` |

每帧先减去自身均值，再乘 symmetric Hann window（`periodic=False`）；不 padding、不做 zero-padding，频带端点包含在内。STFT 固定 `normalized=False`、`onesided=True`、`return_complex=True`，谱计算关闭 AMP。取 $\epsilon_{\mathrm{power}}=10^{-8}$。

非有限 target/prediction 直接使训练失败。对有限 target，frame eligibility 使用规范化前的 $b=B(y)$，并复用与正式谱完全相同的“逐帧去均值 → symmetric Hann → STFT”算子和参数：

$$
A_{W_t}(b)>10^{-8}
\quad\text{且}\quad
\sum_{f\in\mathcal B}
\left|\operatorname{STFT}_s(b)_{t,f}\right|^2
>\epsilon_{\mathrm{power}}
$$

实际谱距离仍在 canonical $x$ 与 $\hat x$ 上计算。低动态 eligibility 只能由 target 决定；prediction 低能量或近常量不得被排除，而是通过平滑后的近均匀谱承担相应误差。

30 秒窗口的解释固定为“局部节律分布”：它足以覆盖多个常见呼吸周期，同时容易对应半分钟尺度的局部变化。统一频带下限 0.05 Hz 在 30 秒中只有 1.5 个周期，因此频带低端只视为被保留的慢变化证据，不把单个 30 秒 frame 的低端谱峰过度解释成精确 RR。由于使用归一化功率谱，该项不约束绝对能量，也不负责极性和相位；$\pm0.3$ 秒小延迟对功率谱基本无影响，因此本项不重复使用 $\tau^*_{\mathrm{train}}$ 对齐。

### 5.5 相对努力趋势损失

训练与评价统一使用 10 秒 RMS、5 秒步长、valid 模式且不 padding。对任意一维信号 $u$：

$$
e_j(u)
=
\sqrt{
\frac{1}{1000}
\sum_{r=0}^{999}u_{j+r}^2
+\epsilon_{\mathrm{env}}
},
\qquad
j=0,500,1000,\ldots
\quad\text{且}\quad j+1000\leq\operatorname{len}(u)
$$

窗口采用左闭右开区间 $[j,j+1000)$。完整 180 秒信号得到 35 个包络点；60 秒局部窗口得到 11 个包络点；lag 对齐后的 17940 点中央共同区间得到 34 个包络点。取 $\epsilon_{\mathrm{env}}=10^{-8}$。

训练 loss 使用数值等价、更加稳定的 log-RMS：

$$
q_j(u)
=
\frac{1}{2}
\log\left(
\frac{1}{1000}\sum_{r=0}^{999}u_{j+r}^2
+\epsilon_{\mathrm{env}}
\right)
$$

努力趋势损失为：

$$
\boxed{
\mathcal L_{\mathrm{effort}}
=
1-
\rho\left(q(\hat x^{a}),q(x^{a})\right)
}
$$

该项只评价相对努力趋势，不要求预测和目标之间存在固定线性幅值映射。这里的 $\rho$ 使用第 5.3 节相同的中心化 Pearson 数值公式、$\epsilon_{\mathrm{corr}}=10^{-8}$ 和 clamp 语义，但不再搜索 lag。10 秒 RMS 覆盖正常呼吸下约 2–3 个周期，5 秒步长避免让强自相关的高采样率包络重复加权。

Sample eligibility 同时要求：满足 `L_sync` 的 target dynamic 条件，且 target 的 $q(x^a)$ 有限、总体方差大于 $10^{-8}$。Loss 端不设置 prediction-variance 硬分支；所有有限 prediction 都按稳定 PCC 公式计算，严格常量时 $\rho=0$、`L_effort=1`。先逐 sample 计算，再只对该项 eligible sample 取 batch 算术均值；无 eligible sample 时返回 graph-connected 0 并记录计数。

### 5.6 训练早期极性锚定损失

数据载体本身已经是 soft-z，本项不再次执行 soft-z 压缩。令 $Z_w(\cdot)$ 表示在 lag 对齐后的单个样本共同有效区间上执行普通标准化：

$$
Z_w(u)
=
\frac{u-\bar u}
{\sqrt{\operatorname{var}(u)+10^{-8}}}
$$

其中 `var` 使用总体方差（`correction=0`）。令 $N_a=|\mathcal I|=17940$ 为对齐后点数，则：

极性锚定使用与 PyTorch `SmoothL1(beta=\delta)` 一致的函数：

$$
s_\delta(r)=
\begin{cases}
\dfrac{r^2}{2\delta}, & |r|\leq\delta\\[4pt]
|r|-\dfrac{\delta}{2}, & |r|>\delta
\end{cases}
$$

则：

$$
\boxed{
\mathcal L_{\mathrm{pol}}
=
\frac{1}{N_a}
\sum_t
s_\delta\left(Z_w(\hat x^{a})_t-Z_w(x^{a})_t\right)
}
$$

本轮冻结：

$$
\delta=0.5
$$

该项只作为训练早期稳定器。令 $S_{\mathrm{total}}$ 为考虑梯度累积后的总 optimizer update 数，$g\in\{0,1,\ldots,S_{\mathrm{total}}-1\}$ 为持久化的 `global_optimizer_step`。在第 $g$ 次 optimizer update **之前**计算：

$$
p=\frac{g}{S_{\mathrm{total}}}
$$

权重在前 15% optimizer updates 内线性退火：

$$
\lambda_{\mathrm{pol}}(p)
=
0.05
\max\left(0,1-\frac{p}{0.15}\right)
$$

`L_pol` 复用 `L_sync` 的 target-only eligibility；prediction 近常量不排除。先逐 sample 对 $N_a$ 个点取均值，再对 eligible sample 取 batch 算术均值；无 eligible sample 时返回 graph-connected 0 并记录计数。权重从 0.05 降到 0；退火结束后，极性和同步由 signed `L_sync` 负责，不让 SmoothL1 波形项长期重复施压。一次性验收必须包含精确反相 $\hat x=-x$ 样例，确认该项在反相点仍提供非零、方向正确的梯度。

### 5.7 总损失与默认权重

第一版总损失为：

$$
\boxed{
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{sync}}
+0.5\mathcal L_{\mathrm{rhythm}}
+0.25\mathcal L_{\mathrm{effort}}
+\lambda_{\mathrm{pol}}(p)\mathcal L_{\mathrm{pol}}
}
$$

其中固定权重为 `sync=1.0`、`rhythm=0.5`、`effort=0.25`，$\lambda_{\mathrm{pol}}(0)=0.05$ 并在前 15% optimizer steps 内退火到 0。本轮不做 loss 权重搜索。

精确参数及作用汇总如下：

| 参数 | 冻结值 | 控制的含义 |
|---|---:|---|
| $B$ 频带 | `0.05–0.70 Hz`，端点包含 | 正式输出允许保留的统一呼吸频带 |
| FFT convention | `rfft/irfft norm="backward"`；STFT `normalized=False, onesided=True` | 固定功率尺度、频率 bin 与近零语义 |
| $\epsilon_{\mathrm{scale}}$ | $10^{-8}$ | 整窗尺度规范化的数值稳定项 |
| target dynamic threshold | $A_W(B(y))>10^{-8}$ | 在尺度规范化前排除近零 target 区间 |
| lag 网格 | `-30…30 samples`，步长 1 sample | 允许的固定延迟为 $\pm0.30$ 秒，分辨率 0.01 秒 |
| lag 共同支撑 | `17940 samples` | 所有 lag 比较使用相同中央长度 |
| $\epsilon_{\mathrm{corr}}$ | $10^{-8}$ | sync 与 effort Pearson 分母的数值稳定项 |
| global rhythm | `180 s / n_fft=18000` | 整窗节律分布 |
| local rhythm | `30 s / hop 10 s / n_fft=3000` | 半分钟局部节律分布，共 16 帧 |
| $\epsilon_{\mathrm{power}}$ | $10^{-8}$ | 每个频带 bin 的谱分布平滑项；分子分母逐 bin 一致加入 |
| RMS envelope | `10 s / step 5 s / valid` | 相对努力观察尺度 |
| $\epsilon_{\mathrm{env}}$ | $10^{-8}$ | log-RMS 与 RMS 的数值稳定项 |
| $Z_w$ variance | `correction=0`，稳定项 $10^{-8}$ | 极性项的对齐区间标准化 |
| SmoothL1 $\delta$ | `0.5` | 极性锚定从二次区进入线性区的标准化误差阈值 |
| $\lambda_{\mathrm{pol}}$ | `0.05 → 0`，前 15% optimizer steps 线性退火 | 只在训练初期帮助确定正负方向 |
| 总 loss 权重 | `1.0 / 0.5 / 0.25` | `sync / rhythm / effort` 的固定第一版相对权重 |

这些值是本轮预注册的第一版 protocol defaults，不通过 validation/test 搜索。所有分量统一“sample 内按各自有效点/帧聚合，再对该分量 target-eligible sample 做 batch 算术均值”；具体 eligibility 见各节。一次性实现验收只检查公式、量级和梯度是否符合定义；若某个值确需改变，必须作为新的单因素实验记录，而不是在同一结果上事后微调。

默认核心结构只有：

```text
同步波形 + 节律 + 努力趋势 + 短期极性稳定
```

### 5.8 Loss 组合与最小训练记录

当前权重作为第一版冻结值。只在一次性实现验收中用固定 mini-batch 确认数值和梯度方向无异常，不在每次训练前重复校准，也不据此启动权重搜索。

训练优化侧每个 epoch 只持久化五个数值：

- `loss_total`；
- `loss_sync`；
- `loss_rhythm`；
- `loss_effort`；
- `loss_pol`。

四个分量记录加权前的 epoch mean；固定权重、极性退火和其他参数由 resolved config 保存。Validation 侧每个 epoch 只额外持久化 `val_core_loss` 与 `val_local_rr_mae`，精确定义和角色见第 7.2 节。默认不记录加权分量、有效样本比例、lag 分布、全局/局部 rhythm 子项、梯度范数、loss 相关性或分层 loss。若未来出现明确优化问题，再另立诊断任务。

### 5.9 Loss 一次性实现验收

每个 loss 首次实现或语义变更时至少验证一次：

- 完全一致输入在允许 lag 下达到 $c=1$ 时，`L_sync=0`。
- 对预期错误方向产生正确惩罚。
- 对应当容忍的变换保持不变或按定义变化。
- 极端值、常量信号、短有效段和无效 mask 下数值稳定。
- 不产生 NaN、Inf 或无梯度路径。
- batch 聚合、sample 聚合和 mask 归一化符合定义。
- 用人工构造样例验证排序关系，而不只验证代码能够运行。
- 对 $\pm0.3$ 秒内外延迟、极性翻转、倍频、半频、幅值缩放和局部努力变化分别做定向测试。
- 验证 30 秒局部谱项确实响应局部节律变化，而全局 180 秒谱项主要响应整窗节律分布。
- 验证 `L_effort` 对整体幅值缩放基本不敏感，但能区分相对努力趋势方向。
- 验证极性项退火到零后不再参与总 loss 和梯度。
- 对 raw head 呼吸频带 RMS 从 $10^{-8}$ 到 $10^{-3}$ 以及 exact-zero 的输入，检查 $\Pi$、四项 loss 和梯度均 finite，并确认近零规范化梯度没有压倒其余分量；该检查只做实现资格验收，不在每次训练前重复。

验收通过后将确定性样例固化为回归测试，不要求每个训练 run 启动前重复执行。

## 6. Metrics 设计区

状态：primary、IBI 与两个 test-only 补充指标的算法、边界、无效值和选模规则已冻结

设计来源：

- `docs/temp/metrics overall.md`
- `docs/temp/metrics  ibi.md`

两份草稿只提供 metric 候选与参数依据，不自动视为最终口径。本节已用领域先验和 train-only 信号审计冻结统一频带、RR 估计器、事件检测器、无效窗口、聚合与 checkpoint 规则；实现后仍须完成第 6.11 节的一次性确定性验收，但不再用 validation/test 调参。

### 6.1 评价问题定义

新评价体系分成三个任务轴，不再把大量相关指标放入同一排序：

1. **节律**：整窗主节律、60 秒局部平均节律，以及逐呼吸 IBI 稳定性。
2. **相对努力**：180 秒整体趋势与 60 秒局部趋势。
3. **联合波形与极性**：允许小延迟后的有符号呼吸频带相关性。

第一版保留五个 `primary` 报告指标：

1. Whole-window RR MAE。
2. Local RR MAE。
3. Global effort-trend Spearman。
4. Local effort-trend Spearman。
5. Lag-aware signed band-limited PCC。

这里的 `primary` 表示 validation 与 designated test 的正式任务轴，不表示五项可以直接平均成一个总分，也不表示五项都参与 checkpoint 排序；第 7.2 节只指定 Local RR 为唯一 selector。补充指标保留两组：IBI-MedAE + coverage 可用于 validation/test，但不参与 checkpoint 排序；coherence 与 nDTW 只在 designated test 计算，不进入训练期、validation、early stopping 或 checkpoint 选择。CCC 删除：在 canonical 输出已经去除整窗均值/尺度、再复用 signed PCC 最佳 lag 的条件下，它与 signed PCC 高度重叠，不能形成新的科学任务轴。

### 6.2 公共预处理与冻结频带

Loss 与 metrics 共用第 5.2 节定义的整窗任务输出算子 $\Pi=S\circ B$，其中 FFT 频带投影冻结为：

$$
\mathcal B=0.05\text{--}0.70\ \mathrm{Hz}
=3\text{--}42\ \mathrm{bpm}
$$

频带依据只来自领域先验和第 5.2.1 节的 train-only 审计。Val 用于模型选择但不修改 metric 定义；designated test 只能在全部口径冻结后按既定频带报告结果，不能反向修改频带。

为统一下述算法的“非退化”含义，对任意待评价区间 $W$ 定义中心化能量：

$$
A_W(u)
=
\frac{1}{|W|}
\sum_{t\in W}(u_t-\bar u_W)^2
$$

先要求 target 与 prediction 均通过 finite 检查；target 非有限是数据错误，prediction 非有限是 checkpoint 错误，二者都不能作为 eligibility 缺失。对有限区间，只有 $A_W(u)>\epsilon_{\mathrm{dyn}}=10^{-8}$ 时才称为 dynamic。Target dynamic 固定在尺度规范化前的 $b=B(y)$ 上判断，并叠加第 2.2 节既有 dataset admission/质量规则；核心 loss/metric 的实际计算对象仍是 canonical $x=S(b)$。Prediction dynamic 在正式输出 $\hat x$ 上判断，因为 raw head 的整体尺度不是任务目标。这样既不让 $S$ 把近零 target 噪声放大成“有效呼吸”，也不因 prediction raw head 的任意整体增益取消评价。Prediction 不满足 dynamic 条件时按第 6.8 节的失败值处理，不能据此排除样本。Metric 统计量使用 float64 计算，最终输出再写为普通标量。

### 6.3 节律指标

#### 6.3.1 Whole-window RR MAE

每个 180 秒样本只产生一个整体 RR：

$$
\mathrm{RR\text{-}AE}_{180,i}
=
\left|\widehat{RR}_i-RR_i\right|
$$

单位为 bpm，越小越好。它回答整段主节律是否正确，不表达逐呼吸变化。

这里的 RR 明确定义为 **dominant spectral RR**，prediction 和 target 都从规范化重建 $\hat x$ 与 $x$ 按同一算法估计，不读取 `tho_rate_ref`；该字段实际是带通信号，不是真实 RR 标签。

Whole-window RR 使用 median-Welch。Target 整窗 dynamic 时该样本才有 RR 评价资格；prediction 不 dynamic 时不运行谱峰插值，直接按 39 bpm 失败误差计入。对有资格的信号：

1. 将 180 秒信号切成 5 个 60 秒 segment，50% overlap。
2. 每段执行 constant detrend，乘 symmetric Hann，再取 `rfft(..., n=6000, norm="backward")` 的 one-sided $|X_f|^2$；不 zero-padding、不做额外 density scaling。
3. 在每个频率 bin 对 5 段功率取 median。
4. 在 `0.05–0.70 Hz` 内取最大功率 bin；最大值并列时固定取较低频率。若峰不在频带边界，对 log-power 使用三点抛物线亚 bin 插值。
5. 将峰频率乘 60 得到 bpm。若最大值位于频带边界，则直接使用边界 bin，不做外推。

设峰 bin 及相邻三个 log-power 为 $\ell_{-1},\ell_0,\ell_{+1}$，插值偏移固定为：

$$
\delta_f
=
\frac{1}{2}
\frac{\ell_{-1}-\ell_{+1}}
{\ell_{-1}-2\ell_0+\ell_{+1}}
$$

并截断到 $[-0.5,0.5]$ bin；分母为 0 或结果非有限时令 $\delta_f=0$。若原始峰 bin 索引为 $m$，最终：

$$
\hat f=(m+\delta_f)\frac{f_s}{n_{\mathrm{fft}}},
\qquad
\widehat{RR}=60\hat f
$$

log-power 使用 $\log(P+10^{-12})$。不加入自相关投票或事后谐波翻转，避免产生未冻结的多分支估计器；因此名称和解释始终限定为 `dominant spectral RR`，不把它表述成逐呼吸平均 RR。

这一选择经过相同 1024 个 train 窗口的 train-only 审计：Whole median-Welch 与 IBI-derived RR 绝对差 `≤1 bpm`、`≤2 bpm` 的比例分别为 84.77%、94.04%，双参照确认的 $2\times$ 或 $0.5\times$ 错峰均为 0；9216 个 Local 60 秒窗中，对应比例分别为 81.04%、92.14%，确认的 $2\times$ 与 $0.5\times$ 分歧分别仅 4 个、8 个。人工复核显示极少数局部分歧来自瞬态、滤波振铃或长基线摆动，而非稳定谐波主导。因此不让 IBI 或峰计数反向修正正式 spectral RR。

#### 6.3.2 Local RR MAE

评价窗口采用：

$$
\text{window}=60\text{ s},
\qquad
\text{step}=15\text{ s}
$$

每个 180 秒样本产生：

$$
K=\left\lfloor\frac{180-60}{15}\right\rfloor+1=9
$$

个高度相关的局部估计，窗口起点为 0、15、…、120 秒，均采用左闭右开边界。每个 60 秒局部窗使用 constant detrend、symmetric Hann、`n_fft=6000`、不 zero-padding的单窗 periodogram，并使用与 Whole RR 相同的并列、边界和 log-power 三点亚 bin 插值规则。

令 $\mathcal G_i$ 为仅由 target dynamic 条件决定的局部窗集合。对 $k\in\mathcal G_i$，若 prediction dynamic 且返回有限 RR，则使用实际绝对误差；否则令 $e_{i,k}=39$ bpm。样本级指标为：

$$
\mathrm{Local\ RR\ MAE}_i
=
\frac{1}{|\mathcal G_i|}
\sum_{k\in\mathcal G_i}e_{i,k}
$$

若 $|\mathcal G_i|=0$，该样本因 target 原因不具备 Local RR 评价资格。Local RR prediction-valid fraction 的分母同样固定为 $|\mathcal G_i|$，但只把 prediction dynamic 且返回有限 RR 的窗口计入分子。单位为 bpm，越小越好。必须同时保存 target eligibility 和 prediction-valid fraction；9 个重叠窗口不能当成 9 个独立统计样本。

60 秒而非 30 秒用于评价，是因为正常 12–20 bpm 下包含约 12–20 次呼吸，原始频率分辨率约 1 bpm；更细的逐呼吸变化交给 IBI，而不是继续缩短谱估计窗。在统一下限 0.05 Hz 处，60 秒只有 3 个周期，因此 3–6 bpm 区间的 Local RR 必须保留“短窗低周期数、易受慢漂移或 OSA 结构影响”的解释限制。

#### 6.3.3 IBI-MedAE 与覆盖率

IBI 用于检查“平均 RR 正确但逐呼吸周期变化错误”的情况。参考同类事件时间为 $t_i$，匹配后的预测事件为 $\hat t_j$：

$$
IBI_i=t_{i+1}-t_i,
\qquad
\widehat{IBI}_i=\hat t_{j+1}-\hat t_j
$$

IBI 事件固定定义为规范化呼吸频带波形的**正向波峰**，不能解释为真实吸气起点。`tho_event_phase_ref` 和 `tho_rate_ref` 均不作为事件 GT。先按第 6.5 节的 $k^*_{\mathrm{eval}}$ 构造共同区间信号 $\hat x^{e}_t=\hat x_{t+k^*_{\mathrm{eval}}}$、$x^e_t=x_t$（$t\in\mathcal I$），再分别检测事件。Prediction 与 target 使用同一个冻结检测器：

$$
\text{min peak distance}
=
\left\lfloor\frac{100}{0.70}\right\rfloor
=142\text{ samples}=1.42\text{ s}
$$

$$
\text{prominence}
=
\max\left(
0.2\,\operatorname{std}(u),
0.08\,[P_{95}(u)-P_5(u)]
\right)
$$

检测器固定采用 `scipy.signal.find_peaks` 的正峰语义，只设置 `distance=142` 和上式的 `prominence`，不设置 height 或 width；`std` 使用 `ddof=0`，$P_5/P_{95}$ 使用 linear percentile。使用 floor 是为了允许 0.70 Hz 离散正弦出现 142/143 samples 交替峰距；若取 143 会错误删除合法的 142-sample 相邻峰。端点不补事件，也不在每个窗口内自适应选择峰或谷。所有事件先用整数 sample index 表示，最终时间统一除以 100 转为秒。

随后做一对一、保持时间顺序的动态规划匹配，允许 $|\hat t_j-t_i|\leq0.5$ 秒（即 50 samples，端点包含）。第一目标最大化匹配事件数，第二目标最小化总 $|\Delta t|$；仍并列时选择索引对序列 $(i,j)$ 字典序更小的方案。由此不依赖库内部任意 tie-break。

只有参考和预测事件索引都连续的相邻匹配，才能形成有效 IBI 对。

$$
\mathrm{IBI\text{-}MedAE}_i
=
\operatorname{median}
\left|
(\hat t_{j+1}-\hat t_j)-(t_{i+1}-t_i)
\right|
$$

单位为秒，越小越好。它不是两个 IBI 分布中位数之差。

Target 的规范化前频带信号 $b$ 在共同区间 dynamic，且 canonical $x^e$ 至少有两个检测峰时，该样本才有 IBI 评价资格。Prediction 峰不足不会取消资格，而是按失败覆盖处理。必须配套报告：

$$
\mathrm{IBI\ coverage}_i
=
\frac{\text{有效匹配 IBI 数}}
{\text{参考 IBI 总数}}
$$

IBI-MedAE 不进入自动 checkpoint 排序。本轮明确不增加 breath-event precision/recall/F1，IBI 只与 coverage 配套报告。样本级 coverage 小于 0.80 时，该样本的 IBI-MedAE 记为缺失且不可解释，但 coverage 仍保留为 0–1 的实际值。若 target 有 IBI 而 prediction 无法形成有效事件对，则 `coverage=0`。匹配区间内的漏检或多检都会使相应 IBI 对无效并降低 coverage；匹配范围之外的额外事件不被完整评价，因此 IBI 只能作为逐呼吸节律补充结果。

### 6.4 相对努力趋势指标

直接在第 5.2 节得到的规范化呼吸频带信号 $\hat x$ 与 $x$ 上，按第 5.5 节相同的 valid 规则计算 10 秒 RMS、5 秒步长包络；不再次调用 $B$ 或 $S$：

$$
E_x[j]
=
\sqrt{
\frac{1}{1000}
\sum_{r=0}^{999}x[j+r]^2
+10^{-8}
}
$$

完整 180 秒的包络窗口起点为 0、5、…、170 秒，共 35 点；起点为 $s$ 的 60 秒局部窗只使用 $s+0,s+5,\ldots,s+50$ 秒，共 11 点。全部采用左闭右开、valid、无 padding 边界。包络每 5 秒采样一次，避免在 100 Hz 的强自相关序列上制造虚假的样本量。

该 metric 不假设预测与 THO 存在线性幅值标定，因此采用 Spearman 相关。Spearman 固定定义为“分别使用 average ranks 处理并列值，再计算两组 rank 的 Pearson 相关”；target 有资格但 prediction ranks 为常量时直接返回 `-1`，不使用库默认的 NaN。

#### 6.4.1 Global effort-trend Spearman

在完整 180 秒上计算：

$$
\rho_{\mathrm{env,global},i}
=
\rho_S\left(E_{\hat x,i}^{180s},E_{x,i}^{180s}\right)
$$

每个样本有 35 个包络点，越大越好。完整 180 秒规范化前 target $b$ 必须 dynamic，且 canonical target 包络有限、总体方差大于 $10^{-8}$；否则该样本的 Global effort 不可评价。Target 有效但 prediction 包络为常量时按 `-1` 计入。它回答整窗哪些阶段呼吸相对较深或较浅的排序是否一致，不评价绝对幅值。

#### 6.4.2 Local effort-trend Spearman

局部评价与 Local RR 使用相同的 60 秒窗口、15 秒步长。每个局部窗口严格包含 11 个 valid 包络采样点：

$$
\rho_{\mathrm{env},i,k}
=
\rho_S\left(E_{\hat x,i,k},E_{x,i,k}\right)
$$

样本内部直接取有效局部相关性的中位数：

$$
\rho_{\mathrm{env,local},i}
=
\operatorname{median}_{k\in\mathrm{valid}}
\left(\rho_{\mathrm{env},i,k}\right)
$$

这里不使用 Fisher-z，也不保留多种聚合策略。中位数直接、稳健，并避免对仅 11 个包络点得到的局部 Spearman 做额外分布假设。

参考包络为常量时，Spearman 没有定义。Local 窗口只有在规范化前 target $b$ dynamic，且 canonical target 包络有限、总体方差大于 $10^{-8}$ 时才有资格。Target 有效但 prediction 包络为常量时，Spearman 按最差值 `-1` 处理，不能排除 prediction 逃避评价。若某样本的 9 个局部窗均不具备 target 资格，则该样本的 Local effort Spearman 缺失；这是 target-driven 缺失，不用 prediction 失败值填补。

### 6.5 联合波形与极性指标

主综合指标为 180 秒 lag-aware signed band-limited PCC。评价复用第 5.3 节的整数 lag 网格 $\mathcal K$、固定中央区间 $\mathcal I$、数值稳定 PCC $c_k$ 和并列规则：

$$
k^*_{\mathrm{eval},i}
=
\arg\max_{k\in\mathcal K}c_{i,k},
\qquad
\tau^*_{\mathrm{eval},i}=k^*_{\mathrm{eval},i}/100
$$

$$
\rho_{\mathrm{joint},i}
=
c_{i,k^*_{\mathrm{eval},i}}
$$

Target 在中央共同区间 dynamic 时该样本才有资格。若 prediction 不 dynamic，则固定令 $k^*_{\mathrm{eval}}=0$、$\rho_{\mathrm{joint}}=-1$；否则按上式选择。越大越好。必须使用 signed PCC，不能取绝对值或平方。该指标同时检查呼吸频带波形形态和极性，但允许不超过 0.3 秒的整窗固定延迟。

`L_sync` 与该 metric 共用呼吸频带投影、lag 网格、边界裁剪、PCC 公式和无惩罚 hard argmax 语义。训练使用 $k^*_{\mathrm{train}}$ 计算 `1-c[k*]` 并对齐后续 loss；metric 使用 $k^*_{\mathrm{eval}}$ 报告允许范围内实际达到的最佳 signed PCC。两者都不把 $|k|$ 大小加入优化目标，也不引入额外平滑参数。

配套保存以下诊断量，不进入主评分。Lag 分布只在 target eligible 且 prediction 非退化的样本上计算；target eligibility fraction 以全部 sample 为分母，prediction-degenerate fraction 以 target-eligible sample 为分母：

- median $|\tau^*_{\mathrm{eval}}|$；
- p95 $|\tau^*_{\mathrm{eval}}|$；
- $\tau^*_{\mathrm{eval}}$ 命中 $\pm0.3$ 秒边界的比例；
- target eligibility 与 prediction-degenerate 比例。

### 6.6 Test-only 补充指标

以下两项只在最终 designated test 计算。它们不写入训练期 epoch metrics，不进入 validation summary、early stopping、checkpoint 选择或模型迭代，也不组成新的总分。二者直接使用第 5.2 节得到的 canonical $\hat x/x$，不再次调用 $B$、$S$ 或按各自区间重新标准化。

#### 6.6.1 Respiratory-band coherence

在完整 180 秒上计算标准 magnitude-squared coherence。固定使用 5 个 60 秒 segment、50% overlap，segment 起点为 0、30、60、90、120 秒。每段先减去自身均值，再乘 symmetric Hann（`periodic=False`），随后执行 `rfft(..., n=6000, norm="backward")`；不 padding、不 zero-padding，统计量使用 float64。

令第 $r$ 段 prediction/target 的复频谱分别为 $X_r(f)$ 和 $Y_r(f)$，则：

$$
S_{\hat x\hat x}(f)=\frac{1}{5}\sum_{r=1}^{5}|X_r(f)|^2,
\qquad
S_{xx}(f)=\frac{1}{5}\sum_{r=1}^{5}|Y_r(f)|^2
$$

$$
S_{\hat x x}(f)=\frac{1}{5}\sum_{r=1}^{5}X_r(f)Y_r(f)^*,
\qquad
C(f)
=
\frac{|S_{\hat x x}(f)|^2}
{S_{\hat x\hat x}(f)S_{xx}(f)}
$$

若某个 bin 的分母为 0，则该 bin 的 $C(f)=0$；其余结果因浮点误差截断到 $[0,1]$，不加入任意绝对 epsilon。`0.05–0.70 Hz` 在 60 秒 FFT 上恰好对应端点包含的 bin 3–42，共 40 个 bin。样本级指标直接对这些 bin 做不加权算术平均：

$$
C_{\mathrm{resp}}
=
\frac{1}{40}
\sum_{m=3}^{42}C(f_m)
$$

越大越好。它不预先使用 $k^*_{\mathrm{eval}}$ 对齐：固定时延主要改变互谱相位而不改变理想 magnitude-squared coherence，预对齐只会重复放宽时间容差。这里也不做频率功率加权或频率有效性阈值，避免把 coherence 变成另一个主频/谱能量指标。由于整窗只提供 5 个 Welch segment，有限样本 coherence 存在正偏；本指标只解释为跨区段的呼吸频域耦合补充结果，不设置通过阈值，也不作显著性解释。

#### 6.6.2 Constrained nDTW

由于 $\hat x/x$ 已严格限带到最高 0.70 Hz，nDTW 固定从 100 Hz 等间隔抽取到 10 Hz 后计算：

$$
u_i=\hat x_{10i},
\qquad
v_i=x_{10i},
\qquad i=0,1,\ldots,1799
$$

10 Hz 在 0.70 Hz 处仍有约 14.3 点/周期，同时避免在几乎重复的 10 ms 点上构造庞大路径。这里不做额外抗混叠滤波，因为 $B$ 已删除 0.70 Hz 以上成分；不再次执行 $Z_w$，因为 $\Pi$ 已固定不可辨识的整窗均值和尺度。

局部点代价固定为 signed canonical 波形的 L1 距离：

$$
d(i,j)=|u_i-v_j|
$$

路径从 $(0,0)$ 到 $(1799,1799)$，只允许 $(1,1)$、$(1,0)$、$(0,1)$ 三种步进，并满足 Sakoe–Chiba 约束：

$$
|i-j|\leq3
$$

即局部 warping 不超过 0.3 秒。动态规划首先最小化累计 L1 代价；累计代价完全相同时选择路径更短者，再并列时优先对角步。令得到的路径为 $P^*$，则：

$$
\mathrm{nDTW}
=
\frac{
\sum_{(i,j)\in P^*}|u_i-v_j|
}{|P^*|}
$$

越小越好。nDTW 不预先使用 $k^*_{\mathrm{eval}}$ 对齐，避免先做全局 lag、再允许局部 warping 而重复放宽时间容差；不增加 slope constraint、warping penalty 或幅值截断。它只解释为“有限局部时间变形下的波形差异”，不能解释成 RR、IBI 或真实时间同步准确性。

### 6.7 Metric 角色总表

| metric | 回答的问题 | 时间尺度 | 聚合顺序 | 角色 | 运行阶段 | 当前状态 |
|---|---|---|---|---|---|---|
| Whole-window RR MAE | 整段主节律是否正确 | 180 秒 | sample → direct mean | `primary` | validation + test | 已冻结 |
| Local RR MAE | 局部平均节律是否正确 | 60 秒/15 秒 | local → sample → direct mean | `primary + checkpoint selector` | validation + test | 已冻结 |
| IBI-MedAE | 逐呼吸周期变化是否正确 | 逐事件 | event → sample → direct mean | `supplementary` | validation + test | 已冻结 |
| IBI coverage | IBI 结果覆盖多少参考周期 | 逐事件 | event ratio → sample → direct mean | `coverage companion` | validation + test | 已冻结；不增加 event F1 |
| Global envelope Spearman | 整窗相对努力趋势是否一致 | 180 秒 | envelope point → sample → direct mean | `primary` | validation + test | 已冻结 |
| Local envelope Spearman | 局部相对努力趋势是否一致 | 60 秒/15 秒 | envelope point → local median → sample → direct mean | `primary` | validation + test | 已冻结 |
| Lag-aware signed PCC | 方向正确的呼吸频带波形是否同步 | 180 秒，$\pm0.3$ 秒 lag | sample → direct mean | `primary` | validation + test | 已冻结 |
| Respiratory-band coherence | 跨区段频域耦合是否一致 | Welch 60 秒/50% overlap，40 个频带 bin | frequency → sample → direct mean | `supplementary` | test only | 已冻结 |
| Constrained nDTW | 有限局部变形下的波形差异 | 10 Hz、180 秒，warping ≤0.3 秒 | path → sample → direct mean | `supplementary` | test only | 已冻结 |

### 6.8 Eligibility 与无效 prediction 处理

Eligibility 只能由 target、固定质量元数据和预先冻结的规则决定，prediction 不得参与样本是否被评价的判定。全局/局部 RR 与 signed PCC 使用第 6.2 节在 $b$ 上的 dynamic 条件；Global/Local effort 还要求对应 target 包络有限且非常量；IBI 使用共同区间的 $b$ dynamic 加至少两个 target 正峰。三类资格分别保存，不能用一个总 valid mask 代替。

- Target、$B(y)$ 或 $x$ 出现 NaN/Inf：视为数据错误，训练/validation/test 立即失败，不算作普通 target-ineligible。
- 有限 target 低于对应 target 动态/变化门槛：该项不具备评价资格，排除时必须保留 eligibility 计数。
- Raw head、$B(\hat y)$ 或 $\hat x$ 出现 NaN/Inf：视为模型错误，训练立即失败；validation 时整个 checkpoint 判为不合格，不按窗口静默丢弃。
- Target 有效但 prediction 为数值常量、无有效峰或其他退化输出：不得排除，按预定义最差值计入。

冻结的最差值语义：

| 情况 | 处理 |
|---|---|
| signed PCC 的 prediction 退化 | `-1` |
| Global/Local effort Spearman 的 prediction 退化 | `-1` |
| Whole/Local RR 无法从 prediction 估计 | RR 绝对误差记为 `42-3=39 bpm` |
| IBI 无有效 prediction 事件对 | `coverage=0`，`IBI-MedAE=NaN`，且不能因 MedAE 缺失提高汇总 |

Local RR prediction-valid fraction 定义为：target-eligible 局部窗中 prediction dynamic 且能返回有限 RR 的比例。Prediction 无效窗已经按 39 bpm 计错；该比例只在选中 checkpoint 的完整 validation/test 报告中帮助解释误差来源，不再作为 checkpoint eligibility 门槛。

两个 test-only 指标复用 signed PCC 的整窗 target eligibility，不增加各自的 sample mask 或 valid fraction。Coherence 的 prediction 退化时所有零分母 bin 按 0 处理，样本结果自然为 0。nDTW 对任何有限 prediction（包括常量）均按实际路径代价计算，不排除样本，也不人为指定上界或失败常数；nDTW 没有自然有限上界，且不参与选模，增加任意 cap 只会污染指标定义。任一 raw/canonical prediction 非有限仍使整个 checkpoint 评价失败。

Loss 中若某个分量在整个 batch 没有任何 target-valid 样本，该分量返回与计算图相连的 0，并记录有效数为 0；不得返回 NaN，也不得改变其他分量的固定权重。低幅但经 $\Pi$ 后仍 dynamic 的 prediction 不因原始 head 幅值小而判无效，因为本任务不恢复绝对尺度。

### 6.9 聚合与报告层级

本轮沿用既有评价入口的**逐 sample 直接平均**，暂不实现 subject-balanced 聚合、比例分子分母跨 sample pooling、paired-seed delta 或 bootstrap。局部窗和事件先收敛成 180 秒 sample 指标仍属于 metric 自身定义，不属于额外统计均衡：

1. **sample 内**：Local RR 对 target-eligible 局部误差取算术均值；Local effort 对 eligible 局部 Spearman 取中位数；IBI 对有效 IBI 误差取中位数。
2. **标量指标汇总**：对所有 target-eligible 180 秒 sample 的指标值直接取算术平均。Prediction 导致的失败已经按第 6.8 节写成有限最差值，不能在求均值前删除。
3. **比例项汇总**：先在每个 sample 内计算 IBI coverage 和 Local RR prediction-valid fraction，再对 sample ratio 直接取算术平均；不跨 sample 合并分子分母。
4. **IBI 条件结果**：IBI-MedAE 只对 sample coverage $\geq0.80$ 的样本级 MedAE 直接取均值，同时单独报告 sample-level coverage 的直接均值和可解释 sample fraction；IBI-MedAE 不得脱离二者单独引用。
5. **headline**：沿用 `mean` 作为 checkpoint 和主表数值；median、p95 等只在既有汇总入口已经提供时作为描述，不新增 subject 层统计。

多 seed 的跨 seed 配对、mean ± SD 或统计推断属于实验结果最终统计层，等 seed 数量和实验矩阵确定后再单独决定，不进入当前 metrics 实现。正式产物仍保留逐样本数值、sample-level eligibility/coverage 及数据集中的 subject 标识 `samp_id`，使未来可以独立增加 subject-balanced 分析，而无需改变当前默认汇总。

本轮默认不实现 easy/hard、质量、RR 区间或其他探索性分层；若后续产生明确科研问题，再作为独立分析任务设计。

### 6.10 与 Loss 时间尺度的关系

训练和评价的局部尺度有意不完全相同：

- `L_rhythm` 使用 30 秒窗口、10 秒 hop，目标是提供更密集、可解释的局部频谱训练信号。
- Local RR metric 使用 60 秒窗口、15 秒 step，目标是得到更稳定、约 1 bpm 分辨率的局部平均 RR。
- 更细的逐呼吸变化由 IBI-MedAE 检查，不用继续缩短 Local RR 窗口。

这不是同一 metric 的口径漂移，而是“训练监督分辨率”和“稳定评价分辨率”的职责分离。当前阶段不安排 loss–metric 桥接验证；若未来需要验证 30 秒训练代理量是否传导到 60 秒 RR 或 IBI，应另立独立任务，不纳入本轮实现范围。

节律配置必须显式拆开，禁止复用一个含糊的 `local_window_seconds`：

```text
loss.rhythm.local_window_seconds = 30
loss.rhythm.local_hop_seconds = 10
metrics.local_rr.window_seconds = 60
metrics.local_rr.step_seconds = 15
```

effort loss 与 metrics 统一使用 10 秒 RMS 包络、5 秒采样。二者只保留算法职责差异：loss 对 log-RMS 使用可微 Pearson，metrics 对同一基础包络使用 Spearman；Global metric 作用于 180 秒，Local metric 再按 60 秒/15 秒聚合。

### 6.11 Metric 一次性实现验收

每个 metric 首次实现或语义变更时至少验证一次：

- 构造完全正确、完全错误和部分错误的信号对，验证排序符合直觉。
- 分别注入幅度缩放、单调非线性幅值映射、时延、极性翻转、倍频、漏周期和额外周期。
- 验证固定延迟不影响 IBI，但局部时变延迟会增加 IBI 误差。
- 验证 IBI-MedAE 不能通过只保留容易事件或生成多余事件获得虚假优势。
- 验证全局和局部 effort 指标对整体幅值缩放及单调映射的预期不变性。
- 检查预处理和 mask 是否造成意外不变性或信息泄漏。
- 检查短有效段、常量信号、低能量信号和无有效事件时的返回语义。
- 与少量人工构造的确定性样例交叉核对。
- 验证固定时延同频信号仍具有高 coherence、无关频率或跨区段关系不一致时 coherence 降低，并确认零分母 bin 返回 0。
- 验证 nDTW 对完全一致信号为 0、对约束内局部形变不高于未变形直接路径、对超过 0.3 秒形变和极性翻转产生更大代价。

这些测试只验证各 metric 自身定义与边界行为，不承担 loss–metric 桥接验证。验收通过后固化为回归测试，不要求每个训练或 test run 启动前重复执行。

## 7. Loss、Metrics 与模型选择的一致性

### 7.1 任务映射

第一版任务映射冻结如下：

| 任务目标 | 训练 loss | primary metric | guardrail | 诊断 metric | 允许的不变性/容差 | 失败判据 |
|---|---|---|---|---|---|---|
| 有符号呼吸频带波形同步 | $\mathcal L_{\mathrm{sync}}$ | Lag-aware signed band-limited PCC | 全 prediction finite | $|\tau^*|$ median、p95、边界命中率 | 容忍 $\pm0.3$ 秒延迟；不容忍极性翻转 | 非有限使评价失败；选中 checkpoint 的 validation 平均 PCC 小于 0 表示方向/极性科学失败，不触发重选 epoch |
| 全局与局部呼吸节律 | $\mathcal L_{\mathrm{rhythm}}$：180 秒 + 30 秒 | Whole-window RR MAE；Local RR MAE（60 秒/15 秒） | Local RR 是唯一 checkpoint selector；无额外阈值 | IBI-MedAE + coverage；Local RR prediction-valid fraction | 不要求绝对能量、相位或极性一致 | 无效 RR 计 39 bpm；prediction-valid fraction 只解释结果，不作门槛 |
| 相对呼吸努力趋势 | $\mathcal L_{\mathrm{effort}}$：10 秒 log-RMS、5 秒采样 | Global/Local effort-trend Spearman | 无额外 guardrail | 无默认附加诊断 | 容忍整体幅值缩放和单调非线性标定 | target 常量时不可评价；prediction 常量记 `-1` |
| 训练早期极性稳定 | $\mathcal L_{\mathrm{pol}}$，前 15% optimizer steps 退火至零 | 不单独增加指标；由 signed PCC 覆盖 | 同第一行 | 无默认附加诊断 | 仅允许小延迟，不允许符号翻转 | 选中 checkpoint 的 validation signed PCC 小于 0 表示极性目标失败，不回头重选 epoch |

### 7.2 Validation、checkpoint 与 early stopping

每个 epoch 在**固定且完整的 validation 窗口**上只计算两个 epoch 级数值。正式实验不沿用当前配置的 `max_val_windows=256` 上限；实现时只取消窗口上限，不改变既有 validation subject/session split。

第一项是只用于观察优化过程、不参与 checkpoint 排序的稳定核心 validation loss：

$$
\mathcal L_{\mathrm{val,core}}
=
\mathcal L_{\mathrm{sync}}
+0.5\mathcal L_{\mathrm{rhythm}}
+0.25\mathcal L_{\mathrm{effort}}
$$

三个分量分别在完整 validation 上按各自 target-eligible sample 汇总后再组合，不能先求 batch total loss 再对 batch 等权平均。这里明确排除 $\mathcal L_{\mathrm{pol}}$：它只负责训练早期稳定且权重随 optimizer step 变化，若纳入会使不同 epoch 的 validation 数值不再对应同一个目标函数。训练中的 $\mathcal L_{\mathrm{pol}}$ 仍按第 5.6–5.7 节保留。

第二项是唯一用于 checkpoint 选择的完整 validation Local RR MAE，按第 6.3.2 与 6.9 节执行 local → sample → direct mean，并已把 prediction 无法估计 RR 的 target-eligible 局部窗按 39 bpm 计错。选择规则只有：

$$
e^*
=
\arg\min_e
\mathrm{Local\ RR\ MAE}_{\mathrm{val},e}
$$

不设置 `0.25 bpm` 或其他容差；数值完全相同时选择更早 epoch。Validation raw/canonical prediction 非有限仍按第 6.8 节使评价失败；若固定 validation 中没有任何 target-eligible Local RR sample，则属于数据/协议错误，不是 checkpoint 平局。除此之外不再设置 signed PCC、prediction-valid fraction 或其他 checkpoint 门槛。

第一批重启实验采用固定最大 epoch 预算并**关闭 early stopping**。原因不是 early stopping 与 checkpoint 必须使用不同目标；若未来启用，二者完全可以都监控 Local RR。当前关闭是因为新协议没有足够学习曲线来冻结 `patience/min_delta`，且 Local RR 可能非单调改善。固定候选时间范围也使正式 run 的最大训练预算一致；训练过程中持续保存 Local RR 最优 checkpoint，因此继续训练不会覆盖早期最优模型。

训练过程中长期候选只需维护当前最小 Local RR checkpoint；训练结束后保留 `checkpoint_best_local_rr` 与 `checkpoint_final`，不永久保存每个 epoch，也不保留基于容差的候选池。

训练结束后，只在 `checkpoint_best_local_rr` 上计算完整 validation 的五项 primary metrics、IBI-MedAE + coverage 及必要的 eligibility/coverage。除非出现非有限值等协议错误，不为 Whole RR 或 effort 预设没有证据支持的绝对通过阈值；这些值如实报告，并在有正式对照后按预先定义的相对标准解释。Validation direct-mean signed PCC 小于 0 单独标记为方向/极性科学失败。上述结果都不得用于查看其余 epoch 后事后改选 checkpoint；IBI、Whole RR、PCC、effort 与两个 test-only 指标不参与 epoch 排序。

这一设计承认 loss 最低 epoch 与任务 metric 最优 epoch 可能不同，因此直接让最高优先级的 Local RR 决定 checkpoint；同时不在每个 epoch 计算全部任务指标，避免重新形成未预注册的多指标搜索。`val_core_loss` 只回答优化代理量是否继续改善，完整 metrics 回答选中模型是否真正完成任务，二者角色不能互换。

### 7.3 Designated test 封存规则

Designated test 只在 loss、五项 primary、IBI、两个 test-only 指标、训练预算、seed、候选模型以及每个 run 的 validation checkpoint 全部冻结后，在本轮协议下评价一次。最终 test 报告五项 primary、IBI-MedAE + coverage，以及 coherence、nDTW；结果不得反向修改频带、阈值、事件检测器、checkpoint 或模型选择。

该 split 已有历史观察背景，所以本轮结果只能表述为 `designated test evidence`，不能包装成全新的无偏 held-out 证据。若本轮再次根据 test 调整方案，它还将进一步退化为直接的 development evidence。严格 held-out 结论只能来自未来新建且从未查看的 final split；当前不做任何 test-target 审计，也不为此预留兼容或桥接流程。

## 8. 实验比较协议

本节只定义新协议的第一批 baseline 建立流程。第一批不同时比较多个模型，也不继承历史 run 的数值结论。

### 8.1 Baseline

新 baseline 冻结为纯时域 `PatchMixer1D`，实验 ID 为 `B0_time_only_patch_mixer`：

| 配置 | 冻结值 |
|---|---|
| 模型注册名 | `patch_mixer1d` |
| 输入 | 单通道 `bcg_rawish_segment_soft_z_key` |
| 输出 | 单通道 raw head；正式结果统一经过 $\Pi=S\circ B$ |
| `base_channels` | `16` |
| `patch_len` / `patch_stride` | `256 / 128` samples |
| `mixer_layers` | `2` |
| `overlap_window` | `hann` |
| `output_smoothing_kernel` | `1`，即不增加输出平滑 |
| STFT/其他辅助输入 | 不使用，也不在 baseline 配置中保留无效 STFT 字段 |

选择纯时域 PatchMixer，而不是旧阶段的 `G3_C_wide_8p0`，是为了先把新输出空间、loss、Local RR checkpoint 和完整 metrics 验证清楚；STFT 输入不能与新协议同时作为第一批变量。选择它而不是 `unet1d_tiny`，是因为 PatchMixer 已有完整数据训练的稳定实现基础，同时仍保持单分支和清晰归因。历史 PatchMixer run 只说明结构可运行，其 checkpoint、loss 和旧 metrics 不进入本轮 baseline。

第一版训练默认值一并冻结：

| 配置 | 冻结值 | 说明 |
|---|---:|---|
| train / validation 数据 | 完整现有 split；`max_train_windows=null`、`max_val_windows=null` | 不用 1024/256 子集代替正式训练或选模 |
| dataset admission / sample seeds | 继续使用第 2.2 节及现有 `20260610 / 20260611` | 数据口径已冻结，不重新抽 split |
| 初始化 | 从头训练；不载入旧 checkpoint 或预训练权重 | 历史模型只提供结构参考 |
| optimizer | Adam，`betas=(0.9,0.999)`、`eps=1e-8`、`weight_decay=0` | 沿用 PyTorch Adam 稳定默认值，不增加正则参数 |
| learning rate | `1e-3` | 第一版不做 LR 搜索 |
| batch size | `128` | 与该骨干完整数据训练的稳定 substrate 一致 |
| gradient accumulation | 不使用 | optimizer step 与 batch 一一对应，$L_{pol}$ 进度按实际 optimizer step 计算 |
| LR scheduler | `none` | 不新增 scheduler 超参数 |
| AMP | `false` | 第一版避免 FFT/PCC 关键计算与混合精度语义交织 |
| gradient clipping | `null` | 不预设没有证据的裁剪阈值 |
| preload / workers | `preload_windows=true`、`num_workers=0` | 沿用完整数据正式 run 的稳定加载方式 |
| DataLoader | train `shuffle=true`；validation `shuffle=false`；`drop_last=false`；CUDA 自动 pin memory | 训练随机顺序由训练 seed 控制，validation 集合和顺序固定 |
| early stopping | 关闭 | 理由见第 7.2 节 |
| checkpoint | `checkpoint_best_local_rr` + `checkpoint_final` | 不保留旧 `best_task` 兼容语义 |
| 旧手工 signal baseline | 默认关闭 | 不让旧滤波 baseline 输出进入新默认 summary |

若一次性 smoke 证明 `batch_size=128` 在新 loss 实现下无法运行，这属于实现约束而不是允许自动变化的超参数：在 pilot 前统一改成 `64`、回写文档并对后续所有 run 固定使用；不得按模型或 seed 临时改变。其他上述默认值若需改变，同样必须在 pilot 前记录理由，不能根据正式 validation/test 结果事后调整。

### 8.2 首批最小实验矩阵

第一批采用“实现 smoke → 单 seed pilot → 三 seed 正式 baseline”三步，不把 STFT 或其他模型同时加入矩阵：

| 阶段 | 实验 ID | 范围 | seed | 最大 epoch | 产出与停止条件 | 状态 |
|---|---|---|---:|---:|---|---|
| 实现验收 | `B0_smoke` | `max_train_windows=32`、`max_val_windows=32`、`batch_size=8`；另做 batch 128 单 batch GPU 验收；不是科研结果 | `20260802` | `2` | 检查真实数据 forward/backward、finite、日志、Local RR checkpoint、产物契约与正式 batch 显存可行性 | 已通过 |
| 预算 pilot | `B0_pilot` | 完整 train + 完整 validation | `20260802` | `50` | 不看 test；若非有限、无法产生 Local RR checkpoint，或选中 checkpoint 的 validation signed PCC `<0`，则停止正式 baseline 并另立诊断任务 | 已通过；最佳 epoch=7 |
| 正式 baseline | `B0_formal` | 完整 train + 完整 validation，全部从头训练 | `20260811 / 20260812 / 20260813` | `50`（由 pilot 规则冻结） | 三个 seed 分别报告及作描述性 mean ± SD；不做 paired delta、bootstrap 或显著性检验 | 待运行 |

正式最大 epoch 使用预先冻结的自适应规则：若 `B0_pilot` 的最小 validation Local RR epoch 位于 `41–50`（端点包含），正式预算固定为 `80`；否则固定为 `50`。只使用最佳 epoch 的位置作此决定，不根据 Whole RR、PCC、effort、IBI 或 test 调预算。Pilot 仅用于冻结训练时间范围，不进入正式 baseline 数值汇总；正式三个 seed 不复用 pilot seed。

`B0_formal` 的职责只是建立新协议 baseline，不声称优于历史模型，也不做 loss 权重、模型结构或输入分支消融。三个正式 seed 逐 seed 保留五项 primary、IBI + coverage 和逐样本结果；跨 seed 只提供描述性算术 mean ± SD，不增加配对统计推断。

第一阶段不打开 designated test。只有未来候选模型、训练预算和 validation 选择全部冻结后，才按第 7.3 节对最终模型集合统一评价一次；因此 coherence 与 nDTW 在本阶段不会运行。

设计要求：

- 第一批只验证新协议能否建立可信 baseline，不追求模型排名或大规模搜索。
- Pilot 只能决定正式最大 epoch 的 `50/80` 分支，不能参与正式数值汇总。
- 三个正式 run 固定数据、split、模型、训练设置和 epoch 预算，只改变训练 seed。
- 任何正式 run 都必须保留 resolved config、split/sample seed、训练 seed、命令、代码版本、checkpoint 和逐样本 metrics。

## 9. 实现前冻结清单

只有以下项目全部完成后，才进入代码修改：

- [x] 核心任务目标、数据载体与 split 边界已写清。
- [x] 规范化输出空间 $\Pi=S\circ B$ 已定义。
- [x] 总 loss 公式、各项定义、权重和生效条件已冻结。
- [x] 每个 loss 的不变性、边界条件和单测样例已定义。
- [x] primary、coverage companion、supplementary metrics 已分组。
- [x] 每个核心 metric 与 IBI 的预处理、mask、统计单位和聚合方式已冻结。
- [x] 核心 metric 的合成信号测试和人工真实样本复核方案已定义。
- [x] validation、checkpoint、early stopping 和 designated test 封存口径已对齐。
- [x] coherence 与 nDTW 的精确 test-only 实现、失败语义和解释边界已冻结；CCC 因与 signed PCC 冗余删除。
- [x] 历史结果整体退出新比较，不提供兼容迁移或重评路径。
- [x] 首批 baseline、pilot/正式 seed、预算冻结规则、最小实验矩阵和停止条件已明确。
- [x] 代码影响面、新增配置、一次性验收、回归测试和文档更新清单已完成。

## 10. 已实现代码影响面

实现直接替换当前 THO 主路径，不维护新旧协议切换或兼容层；模型注册表和数据基础设施保留。旧实现不复制到仓库内 `archive/`，需要时从 Git 恢复：

- 新增 `resp_train/protocols/respiration.py`：Torch/NumPy 共用的 $B$、$S$、$\Pi$、频带和边界基础算子。
- 新增 `resp_train/losses/task.py`，只实现冻结的四项 loss、optimizer-step 极性退火和 eligible 聚合；删除旧 loss 模块与旧权重分支。
- 新增 `resp_train/metrics/task.py`，只实现五项 primary、IBI + coverage、test-only coherence/nDTW、逐 sample 结果和 direct-mean summary；删除旧 metrics 模块与旧指标列。
- 重写 `resp_train/experiments/tho.py` 为独立的当前生命周期：每 epoch 最小日志、完整 Local RR 选模、两个 checkpoint 和选中 checkpoint 完整 validation 评价；删除旧 `BaseExperiment`、gate、best-task、topK 和 early-stopping 路径。
- 直接更新 `configs/tho_research_v2.yaml` 为纯时域 PatchMixer baseline 冻结配置；旧 run 的 resolved config 继续留在各自 run 目录作溯源。
- 当前入口固定为 `scripts/train_tho.py` 和 `scripts/eval_tho.py`；删除旧 small/test 入口，不提供转发壳。Test 必须显式传入 `--confirm-designated-test` 并遵守第 7.3/8.2 节。
- 修改 `resp_train/engine/train.py`：只增加 loss 模块训练/评价状态、optimizer-step 与 eligible sum/count 聚合的通用钩子。
- 更新包导出、`AGENTS.md` 和 `scripts/README.md`，只描述当前新协议；旧 probe 代码、配置和长期说明退出当前工作树，由 Git 历史追溯。
- 新增定向测试：输出投影、四项 loss/梯度/退火、RR/IBI/effort/PCC/coherence/nDTW、无效值、Local RR checkpoint 和配置契约。
- 旧 loss/metrics/checkpoint/runner 测试退出当前测试集合，旧模型与当前数据基础设施测试保留。历史 run、checkpoint、CSV、图表和原始数据不删除或改写。

若后续设计改变 target、数据构造、窗口、split、标签或 mask，需单独增加影响面说明，不能混在普通 loss/metric 实现中静默修改。

## 11. 决策日志

| 日期 | 主题 | 决策 | 理由 | 尚存风险 | 是否冻结 |
|---|---|---|---|---|---|
| 2026-07-29 | 实验重启 | 新实验不继承旧 loss、metrics、排序和结论；先完成设计，再修改代码 | 避免历史口径继续约束新任务定义，也避免设计未定时反复改代码 | 新 baseline、历史可比性和实现影响面尚未确定 | 是 |
| 2026-07-29 | Loss 精简 | 核心 loss 只保留同步、节律、努力趋势和短期极性稳定四项；权重及精确参数见第 5.7 节 | 每项只负责一个明确目标，减少波形、频谱、极性和幅值约束之间的长期重复 | 数值、梯度和退火边界已由确定性测试验收；真实数据 smoke 仍待运行 | 已实现 |
| 2026-07-29 | 局部节律尺度 | 局部频谱窗口从草稿的 40 秒改为 30 秒，第一版 hop 保持 10 秒 | 半分钟尺度更容易解释，同时仍覆盖多个正常呼吸周期 | 快速呼吸、极慢呼吸和非稳态窗口上的频率分辨率仍需合成信号验证 | 是 |
| 2026-07-31 | 输出空间 | raw head 仅作内部量；正式输出为 $\Pi=S\circ B$ 后的呼吸频带、整窗单位尺度表示 | 任务从来不是原始 waveform 或绝对幅值重建；显式删除带外与尺度零空间 | FFT 投影的有限值、中心化、尺度和 Torch/NumPy 一致性已由定向测试验收 | 已实现 |
| 2026-07-31 | 频带候选 | 曾依据 1024 个 train 窗口审计建议 `0.10–0.70 Hz`，未查看 val/test target | 多数 `<0.10 Hz` 主导案例像瞬态与慢基线 | 审计没有 OSA 事件标签，不能排除连续 OSA 相关慢变化 | 2026-08-01 已替代 |
| 2026-08-01 | 统一频带 | waveform、四项 loss、五项 primary、IBI 与 test-only 指标统一使用 `0.05–0.70 Hz`（3–42 bpm） | 避免不可逆删除潜在连续 OSA 慢变化，也避免多套频带带来的解释分叉 | 0.05–0.10 Hz 更易受漂移影响，30/60 秒短窗在低端仅有 1.5/3 个周期 | 当前阶段冻结 |
| 2026-07-31 | Metrics 第一版结构 | 正式报告保留 Whole RR、Local RR、Global/Local effort Spearman 和 lag-aware signed PCC 五项；IBI-MedAE + coverage 保留；曾暂定 coherence、nDTW、CCC 仅 test | 分别覆盖整体节律、局部平均节律、相对努力和联合波形/极性，同时限制补充指标角色 | test-only 指标当时尚未精确定义 | 2026-08-01 已替代 |
| 2026-07-31 | 训练与评价局部尺度分工 | 训练节律 loss 使用 30 秒/10 秒，Local RR 评价使用 60 秒/15 秒，逐呼吸变化由 IBI 补充 | 训练需要较密监督，谱 RR 评价需要足够周期数和约 1 bpm 分辨率 | 当前不做跨尺度桥接验证；若未来需要，另立独立任务 | 是 |
| 2026-08-01 | Lag 规则 | 100 Hz 下搜索 `-30…30` samples，统一中央 179.4 秒支撑；训练、后续 loss 对齐和评价统一使用无惩罚 hard argmax，`L_sync=1-c[k*]` | lag 只容忍残余对齐误差，不作为值得模型优化的科学目标 | 边界、符号和反相语义已由确定性测试验收 | 已实现 |
| 2026-07-31 | STFT 与 envelope 边界 | 节律为 180 秒单帧 + 30 秒/10 秒 16 帧；effort 为 valid 10 秒 RMS/5 秒步长，不 padding | 分别覆盖整窗、半分钟节律及 2–3 个呼吸周期的努力尺度 | 窗口数、归一化和边界已由确定性测试验收 | 已实现 |
| 2026-07-31 | 频谱归一化修正 | $p_f=(P_f+\epsilon)/\sum_g(P_g+\epsilon)$，分母对每个 bin 的平滑项一并求和 | 原草稿只在分母总和后加一次 $\epsilon$，导致 $\sum_f p_f\neq1$ | 公式已按逐 bin 一致平滑实现；回归测试覆盖 identity、频率错误与 finite gradient | 已实现 |
| 2026-07-31 | Metrics 聚合候选 | 曾建议 sample → subject → subject-macro、比例 pooling 与 paired-seed 报告 | 可减少 subject 窗口数不均造成的权重差异 | 不属于当前常用默认汇总，增加实现和解释层级 | 2026-08-01 已搁置 |
| 2026-08-01 | 当前 Metrics 聚合 | sample 内先完成局部窗/事件聚合，数据集层沿用逐 sample direct mean；sample ratio 也直接平均 | 保留既有、常见且简单的实验汇总口径，把 subject-balanced 与跨 seed 推断留到最终统计阶段 | sample 数多的 subject 仍会有更高权重，作为已知限制记录 | 当前阶段冻结 |
| 2026-07-31 | IBI 配套指标 | IBI-MedAE 只配套 IBI coverage，不增加 event precision/recall/F1，也不进入自动 checkpoint 排序 | 保持评价体系精简；通过双侧连续事件约束让匹配区间内漏检或多检降低 coverage | 无法完整评价匹配范围外的额外预测事件，因此 IBI 只能作诊断 | 是 |
| 2026-07-31 | RR 与 IBI 算法 | RR 采用 dominant spectral RR、无倍/半频分支；IBI 使用正峰、固定 prominence/distance 与顺序 DP 匹配 | Train-only 审计未见 Whole 系统性谐波误选，Local 分歧极少且来自非稳态坏窗 | IBI 仍是 detector-dependent 条件指标 | 定义冻结 |
| 2026-08-01 | Effort 统一口径 | loss 与 metrics 共用 10 秒 RMS/5 秒采样；target 包络只要求有限且非常量 | 统一基础观察对象并保持核心口径精简 | 常量 target 的 effort 指标不可评价 | 当前阶段冻结 |
| 2026-07-31 | 无效输出 | target-only eligibility 在尺度规范化前的 $b=B(y)$ 上判断；prediction NaN/Inf 使 checkpoint 失败，核心指标的其他退化 prediction 按预定义最差值计入 | 防止 $S$ 放大近零 target，也防止模型通过制造无效输出逃避评价 | test-only 的特殊语义见第 6.8 节 | 当前阶段冻结 |
| 2026-07-31 | Checkpoint | 曾采用完整 validation、关闭 early stopping，以及 Local RR → Whole RR → PCC → Local/Global effort 的容差级联 | 当时试图兼顾多个任务轴且不构造加权总分 | `0.25 bpm / 0.005 / 0.01` 没有新协议下的波动或最小实质差异依据，级联存在阈值悬崖 | 2026-08-02 已替代 |
| 2026-07-31 | 精简实现范围 | 不做 loss–metric 桥接验证，不做历史兼容迁移；曾暂定 coherence、nDTW、CCC 仅在最终 test | 聚焦核心训练与正式评价路径，避免重新积累历史包袱 | test-only 指标当时尚未收口 | 2026-08-01 已替代 |
| 2026-07-31 | Test 防泄漏 | 频带、阈值和 detector 不使用 test target；所有设计与模型冻结后 designated test 在本轮只评价一次 | 防止 test 参与新 metric 设计或迭代选模；同时承认该 split 已被历史实验观察 | 只能称 designated test evidence；严格 held-out 需未来新 split | 是 |
| 2026-08-01 | Test-only 指标收口 | 保留固定 Welch coherence 与 10 Hz constrained nDTW；删除 CCC | Coherence 补充跨区段频域耦合，nDTW 补充有限局部变形下的波形差异；CCC 在 $\Pi$ 和共享 best lag 后与 signed PCC 基本重复，不能形成独立任务轴 | Coherence 只有 5 个 Welch segment，存在有限样本正偏；nDTW 允许的局部形变不能解释为节律正确 | 是 |
| 2026-08-02 | Checkpoint 精简 | 保留 $\mathcal L_{pol}$；每个 epoch 只记录稳定 `val_core_loss` 和完整 validation Local RR，严格选择 Local RR 最小 epoch；删除五级容差、PCC/valid-fraction 门槛和逐 epoch 全 metrics | 直接防止 loss 最低但最高优先级任务 metric 非最优，同时避免未预注册的多指标搜索和任意容差 | 单一 Local RR 选模不保证 PCC/effort 同时最优；这些轴在选中 checkpoint 上如实报告，不据此重选 | 是 |
| 2026-08-02 | 新 baseline | 第一批只使用纯时域 `patch_mixer1d`，不同时引入 STFT 或其他模型；旧 checkpoint 和数值不进入比较 | 先验证新输出空间、loss、metrics 和 checkpoint 协议，保持失败归因清晰 | 该 baseline 不代表最终最强模型；结构扩展留到后续独立阶段 | 是 |
| 2026-08-02 | Pilot 与正式预算 | 完整数据单 seed pilot 跑 50 epoch；若最优 Local RR epoch 在 41–50，正式三 seed 统一跑 80，否则跑 50；关闭 early stopping | 用预注册的有限分支避免凭空决定训练长度，也不给正式结果事后调预算 | 单个 pilot seed 只能决定候选时间范围，不能证明跨 seed 收敛完全一致 | 是 |

## 12. 分阶段收口

### 阶段 A：本轮已完成

- 冻结数据/任务口径、规范化输出空间、统一频带与防泄漏边界。
- 冻结四项 loss 的公式、参数、lag、STFT、envelope 与无效值语义。
- 冻结五项 primary、IBI + coverage、RR/IBI 算法、逐 sample direct-mean 汇总与 checkpoint 规则。
- 冻结 test-only coherence 与 nDTW 的精确实现和失败语义；CCC 因与 signed PCC 冗余删除。
- 冻结纯时域 PatchMixer baseline、单 seed pilot、`50/80` epoch 预算分支和三 seed 正式 baseline 矩阵。

### 阶段 B：实现前设计收口

已完成，无剩余设计 blocker。

Subject-macro、跨 sample 比例 pooling、paired-seed delta、bootstrap 与显著性检验均明确搁置，不是开始代码实现前的 blocker；若未来需要，作为独立的最终统计任务增加，不改变本轮默认指标输出。

### 阶段 C：当前实施

Loss、metrics、validation/checkpoint、配置、现行入口与定向测试已经统一替换；当前测试集合 `240 passed`。合成信号、轻量实验生命周期、真实数据 CPU `B0_smoke`、GPU batch 128 验收和完整 `B0_pilot` 均已通过；此后普通训练不重复整套资格验证。Loss–metric 桥接验证不属于本阶段，若未来确有必要另立任务。下一步只运行三 seed 正式 baseline，不打开 designated test。

## 13. 留档与兼容策略

- 不创建 `archive/`，避免旧代码继续占据当前目录和被误当成可运行入口。
- 已提交过的旧代码、旧文档与旧配置由 Git 历史负责恢复；未修改历史 `runs/`、checkpoint、CSV、图表或原始数据。
- 当前代码只承诺冻结协议、现行配置和两个现行入口；模型实现与数据基础设施继续保留，旧实验 runner、旧 loss/metrics 及其测试不再维护。

## 14. 实现验收记录（2026-08-02）

- `./.venv/bin/python -m pytest -q tests`：`240 passed`。覆盖保留的数据/模型能力，以及输出投影、四项 loss、metrics、无效输出、lag、配置冻结、Local RR checkpoint 平局和 designated-test 封存。
- CPU `B0_smoke`：32 train + 32 validation、batch 8、2 epochs、seed `20260802`，运行成功；`val_local_rr_mae` 从 `2.022188` 降至 `1.059387 bpm`。
- Smoke 生成 resolved config、数据审计、最小复现 manifest、两个 checkpoint、训练历史和选中 checkpoint 的完整 validation metrics/summary。验收目录为 `/tmp/tho_restart_b0_smoke_final/20260802_012915_672737`。
- Validation checkpoint 复评成功；test 入口在缺少显式确认时拒绝执行。
- 以上仅属实现验收，不构成科研结果；后续 pilot 结果单独记录于第 15 节。当前仍未运行 designated test 或三 seed 正式 baseline。

## 15. B0 Pilot 结果与正式预算冻结（2026-08-02）

运行目录：`runs/tho_restart_b0_pilot/20260802_171938_601573`。运行基于 Git commit `da41f8f`，启动时工作树干净；使用完整 `10141` 个 train 窗口、`2675` 个 validation 窗口、`batch_size=128`、seed `20260802`、固定 50 epochs，未执行 designated test 推理或指标计算。

Checkpoint selector 在第 7 epoch 达到最低 validation Local RR MAE `0.640705 bpm`，因此按第 8.2 节预注册规则，正式三 seed 最大 epoch 冻结为 **50**，不扩展到 80。`val_core_loss` 的最低点在第 50 epoch（`0.350582`），而 Local RR 最优较早出现；这正是将 Local RR 作为唯一 selector、而不按最低 loss 选 checkpoint 的预期情形，不据此改变选模规则。

选中 checkpoint 的完整 validation 结果：

| 指标 | 结果 |
|---|---:|
| Whole-window RR MAE | `0.549586 bpm` |
| Local RR MAE | `0.640705 bpm` |
| Local RR prediction-valid fraction | `1.000000` |
| Global effort Spearman | `0.485396` |
| Local effort Spearman | `0.494586` |
| Lag-aware signed PCC | `0.839307` |
| IBI-MedAE | `0.082923 s`（`1960` 个可解释 sample） |
| IBI coverage | `0.846381` |
| IBI interpretable fraction | `0.732710` |

停止条件均未触发：训练和评价无 NaN/Inf，Local RR checkpoint 正常产生，平均 signed PCC 为正。需要保留的唯一明显风险是 `18.58%` 的样本最佳 lag 命中 `±0.30 s` 边界，且 p95 为 `0.30 s`；正负边界命中近似对称（`254/243`）。当前不基于 pilot validation 扩大 lag，因为这会改变已冻结的时间容忍定义并使正式 baseline 口径漂移。该现象随正式结果如实报告；若未来要判断是否存在范围截断，作为独立 lag-sensitivity 任务处理，不阻塞当前三 seed baseline。
