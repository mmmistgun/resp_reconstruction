# Loss 与 Metrics 重构实验记录

日期：2026-07-29

最后更新：2026-09-01

状态：最终 loss 与 metrics 已冻结；旧时频模型第一阶段独立测试集评价已完成；CRD-v1.1 与第 45 节 C0/C1/C2 控制线均已关闭；第 46–48 节 CRD-TF v1 P0–P6a与独立测试集评价已完成并冻结；第 49 节 CRD-TF-W v2 的 P0/P−1/correction/P1/P3/P4/P5 已完成并冻结，P2 关闭；第 51 节论文实验补充任务的 P0/P1/P2 与三份完整 W cache 已完成并冻结，P3 六项配置已冻结并等待用户手动执行

## 1. 定位

本文用于从头定义下一阶段实验的 loss、metrics 及二者之间的对应关系。

本轮设计不继承此前实验路线、默认 loss、主指标、排序规则、checkpoint 选择逻辑或既有结论。代码与文档不另建 `archive/`：旧版本依靠 Git 历史追溯，历史 run 原地保留但不进入新实验比较。

Loss、metrics、聚合方式和 checkpoint 选择规则已经完成一致性检查并按冻结定义实现。定向单测、轻量生命周期验收、一次性 CPU `B0_smoke`、GPU batch 128 验收、预算 pilot、三 seed 正式 baseline、loss 消融、M1 纯时域结构探针以及 T1–T4 时频实验均已完成。纯时域模型只承担协议 baseline 与结构对照，不是本项目后续模型研究重点。2026-08-07 起，现有 `test` 明确转为可重复观察的 research-test，具体角色见第 7.3 节。

## 2. 当前目标与边界

当前要完成：

1. 明确模型训练真正需要优化的目标。
2. 定义 loss 的组成、数学形式、适用样本、归一化方式和权重策略。
3. 明确实验成功需要由哪些 metrics 证明。
4. 定义每个 metric 的计算对象、信号预处理、统计单位、聚合方式和解释边界。
5. 对齐训练目标、验证评价、checkpoint 选择和最终结论，避免四者口径割裂。
6. 在设计冻结后，形成代码修改清单、测试清单和首批最小实验矩阵。

实现阶段仍不做：

- 不引入额外测试数据集，不做数据重算或历史结果迁移。
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

现有 `test` split 统一称为 **独立测试集（independent test split）**。其独立性由固定subject/session隔离以及不参与当前阶段的训练、checkpoint选择和candidate选择保证，并不要求整个研究过程只能访问一次；允许在阶段性模型整理后重复评价，也允许评价结果形成后续独立研究问题。为保持跨阶段可比性，当前频带、阈值、detector和metric定义仍只依据train/领域先验冻结；若未来根据独立测试集结果改变其中任何一项，必须登记为test-informed新阶段，既有结果不追溯改写。

本次术语统一不改变任何数据、split、subject/session隔离、target、Pi、loss、metrics、selector、checkpoint、数值结果或冻结产物hash。

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

状态：消融后最终训练目标只保留 `L_sync + 0.25 L_effort`；第 5.4、5.6、5.7 节保留为已淘汰候选的历史定义，当前实现以第 5.10 节为准

设计来源：`docs/temp/loss 20260729.md`。本节将该草稿整理为正式实验设计；局部节律尺度由草稿中的 40 秒调整为 30 秒，其他尚未明确的实现细节不从旧代码中默认继承。

### 5.1 训练目标定义

模型可以在网络内部产生原始 head 输出 $\hat y$，但本任务的规范化重建结果定义为 $\hat x=\Pi(\hat y)$。Loss/metric 的**数值计算对象**是 canonical $\hat x/x$；规范化前的 target 频带信号 $b=B(y)$ 只用于 target eligibility，raw/band 中间量只用于 finite guard。只有 $\hat x$ 正式导出；$\hat y$ 不解释为原始 waveform 重建结果。因此带外、DC、整窗绝对尺度和原始传感器 waveform 都不属于本任务的可辨识目标。

Loss 不再要求一个通用波形误差同时承担所有任务。第一版四项候选经第 18、20 节消融后，最终只保留两个互补作用域：

1. `L_sync`：负责呼吸频带波形的有符号同步，容忍很小的传感器延迟。
2. `L_effort`：负责相对呼吸努力随时间的变深、变浅趋势，不要求固定线性幅值映射。

核心目标概括为：在允许不超过 0.3 秒小延迟的前提下，恢复方向正确的呼吸频带波形及相对努力趋势。Whole/Local RR 和 IBI 继续作为正式评价轴，但不再配置独立节律 loss；这是 validation 消融结果，而不是认为节律不重要。

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

这一步只固定不可辨识的整窗增益；同一个 180 秒窗口内的相对强弱变化仍完整保留。对近零输出，分母中的 $\epsilon_{\mathrm{scale}}$ 只负责避免除零，且该输出仍会被同步、努力和无效输出规则惩罚；它不等于已经证明近零处梯度温和。

投影和尺度规范化都在完整 180 秒、18000 点上执行。`B` 的频带端点包含在内，`n_fft=18000`，不 padding、不使用可训练参数，并关闭 AMP。`rfft/irfft` 固定使用 `norm="backward"` 的默认配对和 one-sided 实谱。先对整窗执行一次 $\Pi$，局部 loss/metrics 再从 $\hat x$ 和 $x$ 中切窗；不对每个局部窗重复滤波或重新做整窗尺度规范化。Loss、validation 和 test 必须使用同一实现语义。

这里明确接受硬矩形 DFT mask 的**循环边界全局投影**语义：窗口末端可以影响开头，并可能产生长程 ringing，$\pm0.3$ 秒中央裁剪也不声称能消除它。选择它是因为本轮定义的是固定 180 秒离线频带表示，而不是普通有限支撑滤波器；因此结果不得外推为 streaming/causal 重建能力。若未来做流式任务，必须另立输出算子和边界协议。

#### 5.2.1 Train-only 频带审计与选择

频带选择只使用当前配置按 `train_sample_seed=20260610` 固定抽样的 1024 个 train 窗口，不读取 val/test target。审计结果：

- `0.05–0.10 Hz` 能量占 `0.05–0.70 Hz` 的 median 为 1.75%，p95 为 15.51%，p99 为 30.29%。
- `0.05–0.70 Hz` 下整窗主峰低于 6 bpm 的窗口为 `20/1024`，占 1.95%；切换到 `0.10–0.70 Hz` 后，主峰变化集中在这 20 个窗口。
- 人工复核两个受影响最大的 train 窗口，低于 0.10 Hz 的主峰主要来自孤立大瞬态和缓慢回基线，不像可信的持续慢呼吸。

上述审计曾支持把下限提高到 `0.10 Hz`，但当前导出索引没有 apnea 类型、事件起止或 apnea burden 字段，无法证明所有 `0.05–0.10 Hz` 成分都与连续 OSA 无关。为避免在任务输出中不可逆删除潜在慢变化，也避免为 waveform、loss、RR、IBI 和 effort 维护多套频带，本轮最终统一选择 `0.05–0.70 Hz`。这不等于把所有低频峰都认定为可靠呼吸；低频漂移、短窗周期数不足和 OSA/慢呼吸混淆作为统一频带的已知解释边界保留，但不再通过拆分频带处理。

最终两项 loss 的作用域固定如下：

| loss 项 | 信号域 | 时间作用域 | 负责内容 | 明确不负责 | 生效阶段 |
|---|---|---|---|---|---|
| $\mathcal L_{\mathrm{sync}}$ | 呼吸频带波形 | 整个 180 秒窗口，小延迟搜索范围 $\pm0.3$ 秒 | 有符号波形同步、整体形态方向、小延迟容忍 | 局部节律分布、绝对幅值 | 全训练阶段 |
| $\mathcal L_{\mathrm{effort}}$ | 对齐后的呼吸频带 log-RMS 包络 | 整个 180 秒趋势，10 秒包络、5 秒采样 | 相对努力变深/变浅趋势 | 绝对幅值标定、快速波形细节 | 全训练阶段 |

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

`L_effort` 使用 $(\hat x^a,x^a)$，不允许重新选择延迟；其 target eligibility 必须先包含 `L_sync` 的中央共同区间 dynamic 条件，因此不会对 sync-ineligible sample 使用任意 $k$。这里必须使用 signed PCC，不能使用 $|c_k|$ 或 $c_k^2$，否则极性翻转会被错误视为等价解。

评价期另行定义 $\tau^*_{\mathrm{eval}}$。训练与评价使用同一个无惩罚 lag 含义，但仍保留不同名称以区分生命周期；lag 只用于容忍残余对齐误差，不作为需要模型最小化的科学目标。

### 5.4 全局与 30 秒局部节律频谱损失（已由消融删除）

本节记录消融前 `L_rhythm` 的精确定义，仅用于解释 `A1_no_rhythm`；它已退出当前配置、训练计算、梯度和日志。

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

### 5.6 训练早期极性锚定损失（已由消融删除）

本节记录消融前 `L_pol` 的精确定义，仅用于解释 `A3_no_pol`；它已退出当前配置、训练计算、optimizer-step 状态和日志。

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

### 5.7 消融前候选总损失与默认权重（历史）

第一版消融候选总损失为：

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

最终权重由预注册消融冻结。只在一次性实现验收中用固定 mini-batch 确认数值和梯度方向无异常，不在每次训练前重复校准，也不据此启动权重搜索。

训练优化侧每个 epoch 只持久化三个训练数值：

- `loss_total`；
- `loss_sync`；
- `loss_effort`。

两个分量记录加权前的 epoch mean；固定权重由 resolved config 保存。Validation 侧每个 epoch 只额外持久化 `val_core_loss` 与 `val_local_rr_mae`，精确定义和角色见第 7.2 节。默认不记录加权分量、有效样本比例、lag 分布、梯度范数、loss 相关性或分层 loss。若未来出现明确优化问题，再另立诊断任务。

### 5.9 Loss 一次性实现验收

每个 loss 首次实现或语义变更时至少验证一次：

- 完全一致输入在允许 lag 下达到 $c=1$ 时，`L_sync=0`。
- 对预期错误方向产生正确惩罚。
- 对应当容忍的变换保持不变或按定义变化。
- 极端值、常量信号、短有效段和无效 mask 下数值稳定。
- 不产生 NaN、Inf 或无梯度路径。
- batch 聚合、sample 聚合和 mask 归一化符合定义。
- 用人工构造样例验证排序关系，而不只验证代码能够运行。
- 对 $\pm0.3$ 秒内外延迟、极性翻转、幅值缩放和局部努力变化分别做定向测试。
- 验证 `L_effort` 对整体幅值缩放基本不敏感，但能区分相对努力趋势方向。
- 对 raw head 呼吸频带 RMS 从 $10^{-8}$ 到 $10^{-3}$ 以及 exact-zero 的输入，检查 $\Pi$、两项 loss 和梯度均 finite，并确认近零规范化梯度没有压倒其余分量；该检查只做实现资格验收，不在每次训练前重复。

验收通过后将确定性样例固化为回归测试，不要求每个训练 run 启动前重复执行。

### 5.10 消融后最终训练 Loss

第 18 节的三个 leave-one-term-out 实验支持删除 `L_rhythm`、保留 `L_effort`、删除 `L_pol`；第 20 节联合删除实验未发现不良交互。因此当前唯一训练目标冻结为：

$$
\boxed{
\mathcal L_{\mathrm{final}}
=
\mathcal L_{\mathrm{sync}}
+0.25\mathcal L_{\mathrm{effort}}
}
$$

当前配置和实现不再包含 rhythm 频谱参数、rhythm 权重、polarity 权重/退火或 SmoothL1 参数。旧 run sidecar 中这些字段只用于追溯；当前训练配置一旦重新出现旧 rhythm/polarity 权重字段即明确失败，不提供静默忽略或重新启用路径。节律仍由 Whole/Local RR 与 IBI 正式评价，并由有符号频带同步提供间接训练约束，但不再声称存在独立的可微节律代理项。

## 6. Metrics 设计区

状态：primary、IBI 与两个 test-only 补充指标的算法、边界、无效值和选模规则已冻结

设计来源：

- `docs/temp/metrics overall.md`
- `docs/temp/metrics  ibi.md`

两份草稿只提供 metric 候选与参数依据，不自动视为最终口径。本节已用领域先验和 train-only 信号审计冻结统一频带、RR 估计器、事件检测器、无效窗口、聚合与 checkpoint 规则；实现后仍须完成第 6.11 节的一次性确定性验收，但不再用 validation/test 调参。

### 6.1 评价问题定义

新评价体系分成三个任务轴，不再把大量相关指标放入同一排序：

1. **节律**：整窗主节律、60 秒局部平均节律，以及逐呼吸 IBI 稳定性。
2. **相对努力**：180 秒相对努力轨迹与整窗调制范围；按 train-only target 调制量分层补充排序相关性。
3. **联合波形与极性**：允许小延迟后的有符号呼吸频带相关性。

当前保留五个 `primary` 报告指标：

1. Whole-window RR MAE。
2. Local RR MAE。
3. Envelope trajectory MAE。
4. Global envelope modulation error。
5. Lag-aware signed band-limited PCC。

这里的 `primary` 表示 validation 与 research-test 的正式任务轴，不表示五项可以直接平均成一个总分，也不表示五项都参与 checkpoint 排序；第 7.2 节只指定 Local RR 为唯一 selector。补充指标保留两组：IBI-MedAE + coverage 可用于 validation/research-test，但不参与 checkpoint 排序；coherence 与 nDTW 只在 research-test 计算，不进入训练期、validation、early stopping 或 checkpoint 选择。CCC 删除：在 canonical 输出已经去除整窗均值/尺度、再复用 signed PCC 最佳 lag 的条件下，它与 signed PCC 高度重叠，不能形成新的科学任务轴。

### 6.2 公共预处理与冻结频带

Loss 与 metrics 共用第 5.2 节定义的整窗任务输出算子 $\Pi=S\circ B$，其中 FFT 频带投影冻结为：

$$
\mathcal B=0.05\text{--}0.70\ \mathrm{Hz}
=3\text{--}42\ \mathrm{bpm}
$$

频带依据只来自领域先验和第 5.2.1 节的 train-only 审计。Val 用于 checkpoint 与当前阶段模型选择但不修改 metric 定义；research-test 按同一频带报告，以保持可比性。若其结果未来触发频带修改，该修改必须进入新的、明确标注 test-informed 的研究阶段，而不能覆盖当前阶段定义。

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

### 6.4 相对努力包络指标

直接在第 5.2 节完整 180 秒 canonical $\hat x/x$ 上计算，不再次调用 $B$、$S$，不使用
$\tau^*_{\mathrm{eval}}$，也不为包络单独搜索 lag。窗口采用零基、左闭右开、valid、无 padding：

$$
q_x[j]
=\frac12\log\left(
\frac1{1000}\sum_{r=0}^{999}x[500j+r]^2+10^{-8}
\right),\qquad j=0,1,\ldots,34.
$$

$\log$ 为自然对数；完整 180 秒固定产生 35 点，窗口起点为 0、5、…、170 秒。统计量使用
float64。预测与 target 分别做一次整窗中位数中心化：

$$
\tilde q_{\mathrm{pred}}=q_{\mathrm{pred}}-\operatorname{median}(q_{\mathrm{pred}}),
\qquad
\tilde q_{\mathrm{target}}=q_{\mathrm{target}}-\operatorname{median}(q_{\mathrm{target}}).
$$

不得在更短窗口内重复归一化。所有分位数固定使用 NumPy `method="linear"`。

#### 6.4.1 Envelope trajectory MAE

$$
E_{\mathrm{traj},i}
=\frac1{35}\sum_{j=0}^{34}
\left|\tilde q_{\mathrm{pred},i}[j]-\tilde q_{\mathrm{target},i}[j]\right|.
$$

代码名称为 `envelope_trajectory_mae`，越低越好。它评价相对努力变化是否在正确时间发生以及变化程度
是否接近。35 个包络点各计一次，不再通过重叠局部窗重复加权。

#### 6.4.2 Global envelope modulation error

定义稳健调制范围：

$$
R(q)=Q_{0.90}(q)-Q_{0.10}(q),
$$

则：

$$
E_{\mathrm{mod},i}
=\left|R(q_{\mathrm{pred},i})-R(q_{\mathrm{target},i})\right|.
$$

代码名称为 `global_envelope_modulation_error`，越低越好。它评价整窗相对努力波动范围，不评价变化
发生的时间位置，因此必须与 trajectory MAE 配套报告；二者不得组合成总 envelope score。

两个主包络指标对所有 admitted sample 都有定义，不设置 target/prediction dynamic eligibility。Target
平坦和 prediction 平坦都保留实际有限误差；任一 raw/canonical waveform、log-RMS 或主指标出现
NaN/Inf 时整个评价失败，不能静默过滤。

#### 6.4.3 Target-stratified envelope Spearman（补充分析）

对完整 admitted training targets 的 $R(q_{\mathrm{target}})$ 使用 linear quantile 冻结三分层：

$$
c_{\mathrm{low}}=Q_{1/3}^{\mathrm{train}}=0.30875308839006915,
\qquad
c_{\mathrm{high}}=Q_{2/3}^{\mathrm{train}}=0.7031542121234101.
$$

本次阈值来自 10141 个 training targets，sample strategy 为 `stratified_random`、seed 为 `20260610`，
有序 `dataset_row_id` 的 SHA-256 为
`f290e569140a2ff7745cf1a5cfa6a4da943644d76498c9b85517d3ae0702c45e`。冻结产物由
`scripts/freeze_envelope_strata.py` 生成；validation/test 不得重新估计。

分层固定为 Low：$R<c_{\mathrm{low}}$；Medium：$c_{\mathrm{low}}\le R<c_{\mathrm{high}}$；
High：$R\ge c_{\mathrm{high}}$。对 target log-envelope 总体方差大于 $10^{-8}$ 的样本，分别使用
average ranks 处理 ties，再计算 prediction/target rank 的标准 Pearson 相关。Target 不满足该条件时
Spearman target-ineligible；target eligible 但 prediction log-envelope 总体方差不大于 $10^{-8}$ 时固定
记为 `-1`，不得排除。

Low/Medium/High 只分别报告 mean Spearman、`n_total`、`n_eligible`、target-ineligible 数/比例和
prediction-degenerate 数/比例；不再汇总为一个总体 Spearman。某层没有 eligible sample 时报告 `NA`
（机器可读结果为缺失），不填 0 或 −1。该项只作补充解释，不进入 checkpoint 选择。

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

### 6.6 Research-test-only 补充指标

以下两项只在 research-test 计算。它们不写入训练期 epoch metrics，不进入 validation summary、early stopping、checkpoint 选择，也不组成新的总分。二者可以用于解释阶段性模型差异，并形成后续独立研究假设；二者直接使用第 5.2 节得到的 canonical $\hat x/x$，不再次调用 $B$、$S$ 或按各自区间重新标准化。

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
| Envelope trajectory MAE | 相对努力变化是否在正确时间发生且程度接近 | 180 秒、35 个 log-RMS 点 | envelope point → sample → direct mean | `primary` | validation + test | 已冻结 |
| Global envelope modulation error | 整窗相对努力波动范围是否正确 | 180 秒、Q90−Q10 | sample → direct mean | `primary` | validation + test | 已冻结 |
| Target-stratified envelope Spearman | 不同 target 调制水平下的包络排序是否一致 | train-frozen Low/Medium/High | envelope point → sample → stratum direct mean | `supplementary` | validation + test | 已冻结；不汇总总体值 |
| Lag-aware signed PCC | 方向正确的呼吸频带波形是否同步 | 180 秒，$\pm0.3$ 秒 lag | sample → direct mean | `primary` | validation + test | 已冻结 |
| Respiratory-band coherence | 跨区段频域耦合是否一致 | Welch 60 秒/50% overlap，40 个频带 bin | frequency → sample → direct mean | `supplementary` | research-test only | 已冻结 |
| Constrained nDTW | 有限局部变形下的波形差异 | 10 Hz、180 秒，warping ≤0.3 秒 | path → sample → direct mean | `supplementary` | research-test only | 已冻结 |

### 6.8 Eligibility 与无效 prediction 处理

Eligibility 只能由 target、固定质量元数据和预先冻结的规则决定，prediction 不得参与样本是否被评价的判定。全局/局部 RR 与 signed PCC 使用第 6.2 节在 $b$ 上的 dynamic 条件；两个主包络误差覆盖所有 admitted sample，不设动态资格；分层 envelope Spearman 只要求 target log-envelope 总体方差大于 $10^{-8}$；IBI 使用共同区间的 $b$ dynamic 加至少两个 target 正峰。各类资格分别保存，不能用一个总 valid mask 代替。

- Target、$B(y)$ 或 $x$ 出现 NaN/Inf：视为数据错误，训练/validation/test 立即失败，不算作普通 target-ineligible。
- 有限 target 低于对应 target 动态/变化门槛：该项不具备评价资格，排除时必须保留 eligibility 计数。
- Raw head、$B(\hat y)$ 或 $\hat x$ 出现 NaN/Inf：视为模型错误，训练立即失败；validation 时整个 checkpoint 判为不合格，不按窗口静默丢弃。
- Target 有效但 prediction 为数值常量、无有效峰或其他退化输出：不得排除，按预定义最差值计入。

冻结的最差值语义：

| 情况 | 处理 |
|---|---|
| signed PCC 的 prediction 退化 | `-1` |
| Target-stratified envelope Spearman 的 prediction 退化 | target eligible 时记 `-1` |
| Whole/Local RR 无法从 prediction 估计 | RR 绝对误差记为 `42-3=39 bpm` |
| IBI 无有效 prediction 事件对 | `coverage=0`，`IBI-MedAE=NaN`，且不能因 MedAE 缺失提高汇总 |

Local RR prediction-valid fraction 定义为：target-eligible 局部窗中 prediction dynamic 且能返回有限 RR 的比例。Prediction 无效窗已经按 39 bpm 计错；该比例只在选中 checkpoint 的完整 validation/test 报告中帮助解释误差来源，不再作为 checkpoint eligibility 门槛。

两个 research-test-only 指标复用 signed PCC 的整窗 target eligibility，不增加各自的 sample mask 或 valid fraction。Coherence 的 prediction 退化时所有零分母 bin 按 0 处理，样本结果自然为 0。nDTW 对任何有限 prediction（包括常量）均按实际路径代价计算，不排除样本，也不人为指定上界或失败常数；nDTW 没有自然有限上界，且不参与选模，增加任意 cap 只会污染指标定义。任一 raw/canonical prediction 非有限仍使整个 checkpoint 评价失败。

Loss 中若某个分量在整个 batch 没有任何 target-valid 样本，该分量返回与计算图相连的 0，并记录有效数为 0；不得返回 NaN，也不得改变其他分量的固定权重。低幅但经 $\Pi$ 后仍 dynamic 的 prediction 不因原始 head 幅值小而判无效，因为本任务不恢复绝对尺度。

### 6.9 聚合与报告层级

本轮沿用既有评价入口的**逐 sample 直接平均**，暂不实现 subject-balanced 聚合、比例分子分母跨 sample pooling、paired-seed delta 或 bootstrap。局部窗和事件先收敛成 180 秒 sample 指标仍属于 metric 自身定义，不属于额外统计均衡：

1. **sample 内**：Local RR 对 target-eligible 局部误差取算术均值；envelope trajectory MAE 对 35 点等权平均；IBI 对有效 IBI 误差取中位数。
2. **标量指标汇总**：两个主包络指标对全部 admitted sample 直接平均；其余标量对各自 target-eligible 180 秒 sample 直接平均。Prediction 导致的失败已经按第 6.8 节写成有限最差值，不能在求均值前删除。
3. **比例项汇总**：先在每个 sample 内计算 IBI coverage 和 Local RR prediction-valid fraction，再对 sample ratio 直接取算术平均；不跨 sample 合并分子分母。
4. **IBI 条件结果**：IBI-MedAE 只对 sample coverage $\geq0.80$ 的样本级 MedAE 直接取均值，同时单独报告 sample-level coverage 的直接均值和可解释 sample fraction；IBI-MedAE 不得脱离二者单独引用。
5. **headline**：沿用 `mean` 作为 checkpoint 和主表数值；median、p95 等只在既有汇总入口已经提供时作为描述，不新增 subject 层统计。

多 seed 的跨 seed 配对、mean ± SD 或统计推断属于实验结果最终统计层，等 seed 数量和实验矩阵确定后再单独决定，不进入当前 metrics 实现。正式产物仍保留逐样本数值、sample-level eligibility/coverage 及数据集中的 subject 标识 `samp_id`，使未来可以独立增加 subject-balanced 分析，而无需改变当前默认汇总。

本轮除第 6.4.3 节已冻结的 target-only envelope modulation 三分层外，不实现 easy/hard、质量、RR 区间或其他探索性分层；若后续产生明确科研问题，再作为独立分析任务设计。

### 6.10 与 Loss 时间尺度的关系

消融后不再存在 30 秒 rhythm loss。Local RR 继续独立使用 60 秒窗口、15 秒 step，以获得较稳定、约 1 bpm 分辨率的局部平均 RR；更细的逐呼吸变化由 IBI-MedAE 检查。该评价口径没有因删除训练代理项而改变，也不新增 loss–metric 桥接验证。

effort loss 与 metrics 统一使用 10 秒 log-RMS 包络、5 秒采样。Loss 在 lag 对齐后的共同区间使用可微 Pearson；metrics 不使用 lag，直接在完整 180 秒 35 点上报告中心化轨迹 MAE、Q90−Q10 调制范围误差，并以 train-only target 三分层 Spearman 作补充解释。两者共用观察尺度，但不要求 loss 与 metric 采用相同聚合或 eligibility。

### 6.11 Metric 一次性实现验收

每个 metric 首次实现或语义变更时至少验证一次：

- 构造完全正确、完全错误和部分错误的信号对，验证排序符合直觉。
- 分别注入幅度缩放、单调非线性幅值映射、时延、极性翻转、倍频、漏周期和额外周期。
- 验证固定延迟不影响 IBI，但局部时变延迟会增加 IBI 误差。
- 验证 IBI-MedAE 不能通过只保留容易事件或生成多余事件获得虚假优势。
- 验证两个主包络指标对整体正幅值缩放不敏感；target 平坦、prediction 伪调制、相同范围但时间顺序错误时返回语义符合定义。
- 验证 35 点边界、linear quantile、train-frozen 三分层端点、target-ineligible 与 prediction-degenerate 计数。
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
| 全局与局部呼吸节律 | 无独立 loss；由 $\mathcal L_{\mathrm{sync}}$ 间接约束 | Whole-window RR MAE；Local RR MAE（60 秒/15 秒） | Local RR 是唯一 checkpoint selector；无额外阈值 | IBI-MedAE + coverage；Local RR prediction-valid fraction | 不要求绝对能量或相位单独匹配 | 无效 RR 计 39 bpm；prediction-valid fraction 只解释结果，不作门槛 |
| 相对呼吸努力趋势 | $\mathcal L_{\mathrm{effort}}$：10 秒 log-RMS、5 秒采样 | Envelope trajectory MAE；Global envelope modulation error | 两个主指标覆盖全部 admitted sample | Train-frozen target-stratified envelope Spearman | 容忍整体正幅值缩放；不为包络搜索 lag | 主指标非有限使评价失败；分层 Spearman target 常量时不可评价、prediction 常量记 `-1` |

### 7.2 Validation、checkpoint 与 early stopping

每个 epoch 在**固定且完整的 validation 窗口**上只计算两个 epoch 级数值。正式实验不沿用当前配置的 `max_val_windows=256` 上限；实现时只取消窗口上限，不改变既有 validation subject/session split。

第一项是只用于观察优化过程、不参与 checkpoint 排序的稳定核心 validation loss：

$$
\mathcal L_{\mathrm{val,core}}
=
\mathcal L_{\mathrm{sync}}
+0.25\mathcal L_{\mathrm{effort}}
$$

两个分量分别在完整 validation 上按各自 target-eligible sample 汇总后再组合，不能先求 batch total loss 再对 batch 等权平均。该式与训练最终 loss 完全一致，仅用于观察优化过程，不参与 checkpoint 排序。

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

### 7.3 独立测试集角色与评价规则

独立测试集用于在阶段性模型整理后评价固定的validation-selected checkpoint。每次报告五项primary、IBI-MedAE + coverage、三层envelope Spearman，以及coherence、nDTW。它可以被重复评价，并可用于提出或决定下一项独立科研任务；但不得据此重选同一训练run的epoch/checkpoint，也不得在一组test结果中事后只保留表现最好的seed或模型。

结果统一表述为`独立测试集证据（independent test-set evidence）`。每次评价必须显式传入`--confirm-research-test`，并记录日期、模型与checkpoint、seed、代码commit、metric版本、完整逐sample结果路径，以及该结果是否影响后续任务。当前阶段同时报告B0/T2/T4的全部三个预注册seed与无训练F0/IEWT；T1/T3已由validation退出，不因test补做。

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
| 正式 baseline | `B0_formal` | 完整 train + 完整 validation，全部从头训练 | `20260811 / 20260812 / 20260813` | `50`（由 pilot 规则冻结） | 三个 seed 分别报告及作描述性 mean ± SD；不做 paired delta、bootstrap 或显著性检验 | 已通过；最佳 epoch=`7 / 7 / 8` |

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
- 新增 `resp_train/losses/task.py`；消融后只实现冻结的 sync、effort 与 eligible 聚合，rhythm、polarity 及 optimizer-step 退火已物理删除。
- 新增 `resp_train/metrics/task.py`，只实现五项 primary、IBI + coverage、test-only coherence/nDTW、逐 sample 结果和 direct-mean summary；删除旧 metrics 模块与旧指标列。
- 重写 `resp_train/experiments/tho.py` 为独立的当前生命周期：每 epoch 最小日志、完整 Local RR 选模、两个 checkpoint 和选中 checkpoint 完整 validation 评价；删除旧 `BaseExperiment`、gate、best-task、topK 和 early-stopping 路径。
- 直接更新 `configs/tho_research_v2.yaml` 为纯时域 PatchMixer baseline 冻结配置；旧 run 的 resolved config 继续留在各自 run 目录作溯源。
- 当前入口固定为 `scripts/train_tho.py` 和 `scripts/eval_tho.py`；删除旧 small/test 入口，不提供转发壳。Research-test 必须显式传入 `--confirm-research-test` 并遵守第 7.3 节。
- 修改 `resp_train/engine/train.py`：只增加 loss 模块训练/评价状态、optimizer-step 与 eligible sum/count 聚合的通用钩子。
- 更新包导出、`AGENTS.md` 和 `scripts/README.md`，只描述当前新协议；旧 probe 代码、配置和长期说明退出当前工作树，由 Git 历史追溯。
- 新增定向测试：输出投影、两项最终 loss/梯度、RR/IBI/effort/PCC/coherence/nDTW、无效值、Local RR checkpoint 和配置契约。
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
| 2026-08-01 | 统一频带 | waveform、当时的四项候选 loss、五项 primary、IBI 与 test-only 指标统一使用 `0.05–0.70 Hz`（3–42 bpm） | 避免不可逆删除潜在连续 OSA 慢变化，也避免多套频带带来的解释分叉 | 0.05–0.10 Hz 更易受漂移影响，30/60 秒短窗在低端仅有 1.5/3 个周期 | 当前阶段冻结；最终两项 loss 沿用 |
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
- 记录四项候选 loss，并由正式消融冻结最终 sync + effort 两项、lag、envelope 与无效值语义。
- 冻结五项 primary、IBI + coverage、RR/IBI 算法、逐 sample direct-mean 汇总与 checkpoint 规则。
- 冻结 test-only coherence 与 nDTW 的精确实现和失败语义；CCC 因与 signed PCC 冗余删除。
- 冻结纯时域 PatchMixer baseline、单 seed pilot、`50/80` epoch 预算分支和三 seed 正式 baseline 矩阵。

### 阶段 B：实现前设计收口

已完成，无剩余设计 blocker。

Subject-macro、跨 sample 比例 pooling、paired-seed delta、bootstrap 与显著性检验均明确搁置，不是开始代码实现前的 blocker；若未来需要，作为独立的最终统计任务增加，不改变本轮默认指标输出。

### 阶段 C：当前实施

Loss、metrics、validation/checkpoint、配置、现行入口与定向测试已经统一替换。合成信号、轻量实验生命周期、真实数据 CPU `B0_smoke`、GPU batch 128 验收、完整 `B0_pilot`、三 seed `B0_formal` 和三项 loss 消融均已通过；此后普通训练不重复整套资格验证。Loss–metric 桥接验证不属于本阶段，若未来确有必要另立任务。当前先用候选最终 loss 重建 PatchMixer baseline，再进入多尺度模型；designated test 继续封存。

## 13. 留档与兼容策略

- 不创建 `archive/`，避免旧代码继续占据当前目录和被误当成可运行入口。
- 已提交过的旧代码、旧文档与旧配置由 Git 历史负责恢复；未修改历史 `runs/`、checkpoint、CSV、图表或原始数据。
- 当前代码只承诺冻结协议、现行配置和两个现行入口；模型实现与数据基础设施继续保留，旧实验 runner、旧 loss/metrics 及其测试不再维护。

## 14. 实现验收记录（2026-08-02）

- `./.venv/bin/python -m pytest -q tests`：`240 passed`。覆盖保留的数据/模型能力，以及输出投影、四项 loss、metrics、无效输出、lag、配置冻结、Local RR checkpoint 平局和 designated-test 封存。
- CPU `B0_smoke`：32 train + 32 validation、batch 8、2 epochs、seed `20260802`，运行成功；`val_local_rr_mae` 从 `2.022188` 降至 `1.059387 bpm`。
- Smoke 生成 resolved config、数据审计、最小复现 manifest、两个 checkpoint、训练历史和选中 checkpoint 的完整 validation metrics/summary。验收目录为 `/tmp/tho_restart_b0_smoke_final/20260802_012915_672737`。
- Validation checkpoint 复评成功；test 入口在缺少显式确认时拒绝执行。
- 以上仅属实现验收，不构成科研结果；pilot 与三 seed 正式 baseline 分别记录于第 15、16 节。当前仍未运行 designated test。

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

## 16. B0 Formal 三 seed Validation Baseline（2026-08-03）

三个正式 run 分别位于：

- `runs/tho_restart_b0_formal/seed_20260811/20260802_222224_529372`
- `runs/tho_restart_b0_formal/seed_20260812/20260802_224223_339456`
- `runs/tho_restart_b0_formal/seed_20260813/20260802_230108_464045`

三次运行均基于 Git commit `ecbe073` 且启动时工作树干净，固定完整 `10141` 个 train 窗口、`2675` 个 validation 窗口、`batch_size=128` 和 50 epochs，只改变训练 seed。三次均完整结束，训练历史全部有限；选中 checkpoint 的 epoch 依次为 `7 / 7 / 8`。逐样本结果只包含 validation，没有执行 designated test，因而未计算 test-only coherence 或 nDTW。

按第 8.2 节冻结口径，每个 seed 先对 sample 直接算术平均，再对三个 seed 报告描述性算术 mean ± sample SD：

| 指标 | seed 20260811 | seed 20260812 | seed 20260813 | 三 seed mean ± SD |
|---|---:|---:|---:|---:|
| Whole-window RR MAE（bpm） | `0.534223` | `0.524641` | `0.535556` | `0.531473 ± 0.005954` |
| Local RR MAE（bpm） | `0.630805` | `0.631485` | `0.636688` | `0.632993 ± 0.003218` |
| Local RR prediction-valid fraction | `1.000000` | `1.000000` | `1.000000` | `1.000000 ± 0.000000` |
| Global effort Spearman | `0.485754` | `0.493683` | `0.487456` | `0.488964 ± 0.004174` |
| Local effort Spearman | `0.497981` | `0.500445` | `0.498872` | `0.499099 ± 0.001248` |
| Lag-aware signed PCC | `0.839783` | `0.838648` | `0.839176` | `0.839202 ± 0.000568` |
| IBI-MedAE（s） | `0.082414` | `0.083606` | `0.084327` | `0.083449 ± 0.000966` |
| IBI coverage | `0.847246` | `0.843796` | `0.845377` | `0.845473 ± 0.001727` |
| IBI interpretable fraction | `0.730841` | `0.726729` | `0.727850` | `0.728474 ± 0.002126` |

三个 seed 的 joint target eligible fraction 均为 `1.0`，prediction degenerate fraction 均为 `0.0`，且 mean signed PCC 均为正；每个 seed 只有 `1/2675` 个样本 signed PCC 为负，不构成 checkpoint 级极性失败。核心指标中未发现 NaN/Inf，IBI-MedAE 的缺失只发生在按冻结定义不可解释的样本，并由 interpretable fraction 与 coverage 同时披露。

Lag 边界风险在三个 seed 上稳定存在：最佳 lag 命中 `±0.30 s` 的比例分别为 `18.06% / 18.21% / 17.91%`，三 seed mean ± SD 为 `18.06% ± 0.15%`，且各 seed 的 p95 均为 `0.30 s`。正式 baseline 不据此改变已冻结 lag；若未来研究范围截断或放宽容忍度，必须作为独立 lag-sensitivity 任务，不与本结果混合。

本节只建立新协议下的 validation baseline，不声称优于历史模型。Designated test 继续按第 7.3 节封存；只有候选模型、训练预算与 validation 选择全部冻结后，才对最终模型集合统一评价一次，且 test 结果不得反向修改当前协议。

## 17. 下一阶段：Loss 消融与多尺度模型（2026-08-03）

下一阶段按五个科学实验规划，但不把五个实验不加判断地连续执行。前三项 leave-one-term-out 消融已经证明最终 loss 需要改变，因此第五个科学实验按预注册分支改为重建 PatchMixer baseline；多尺度纯时域模型顺延为第六个实验。实现 smoke、配置验收和 batch 128 显存验收只属于工程检查，不计为新的科学实验。

| 顺序 | 实验 ID | 唯一变化 | seed | 最大 epoch | 当前状态 |
|---:|---|---|---|---:|---|
| 1 | `B0_full_loss_patchmixer` | 无；第 16 节正式 baseline | `20260811 / 20260812 / 20260813` | 50 | 已完成 |
| 2 | `A1_no_rhythm` | `rhythm_weight: 0.5 → 0` | `20260811 / 20260812 / 20260813` | 50 | 已完成；支持删除 |
| 3 | `A2_no_effort` | `effort_weight: 0.25 → 0` | `20260811 / 20260812 / 20260813` | 50 | 已完成；支持保留 |
| 4 | `A3_no_pol` | `pol_start_weight: 0.05 → 0` | `20260811 / 20260812 / 20260813` | 50 | 已完成；支持删除 |
| 5 | `B0_final_loss_patchmixer` | 同时使用 `rhythm_weight=0`、`pol_start_weight=0`，模型不变 | 同上 | 50 | 已完成；接受 provenance 例外 |
| 6 | `M1_multiscale_time` | 参数匹配的 `2.56 / 5.12 / 10.24 / 20.48 s` 四分支纯时域模型 | 同上 | 50 | 已完成；effort 改善但其余任务轴退化，未入选 |

不增加 `no_sync`：`L_sync` 定义了方向正确的残余时延容忍同步任务，是核心目标而不是可选辅助约束；删除它会改变任务，而不是普通的 loss 精简消融。

### 17.1 第一步：B0 参照

第 16 节的三 seed 结果是唯一固定参照，不重跑、不改 summary，也不把 pilot 纳入正式数值。后续消融沿用相同数据、split、模型、优化器、batch、50-epoch 预算、Local RR checkpoint selector、五项 primary 与 IBI 口径。

### 17.2 第二步：三个 Loss 消融

`A1/A2/A3` 已在同一代码版本下全部完成，每个实验直接运行三个正式 seed，共 9 个 run；没有先用单 seed 筛掉“不理想”实验，也没有根据前一个消融的 validation 结果决定是否运行后一个。三个消融只把目标项权重精确置零，其余权重未补偿、归一化或搜索。

配置层只允许五组精确权重组合：`full`、`no_rhythm`、`no_effort`、`no_pol` 与消融后冻结的 `final_sync_effort`。不增加通用 loss-variant runner，也不开放任意权重覆盖；resolved config 和 run manifest 继续记录实际权重、seed、命令与代码版本。

消融的预注册解释轴为：

- `A1_no_rhythm`：主要观察 Whole/Local RR；其余指标只报告副作用。
- `A2_no_effort`：主要观察 Global/Local effort Spearman；其余指标只报告副作用。
- `A3_no_pol`：主要观察 signed PCC、负 PCC 样本及跨 seed 稳定性；其余指标只报告副作用。

每个实验仍先做 sample direct mean，再对三个 seed 报告描述性 mean ± sample SD；不构造综合分数，不做 paired delta、bootstrap、显著性检验或 subject-macro。是否保留某项根据对应解释轴的方向一致性、效应大小和跨 seed 稳定性共同判断，不因单 seed 或末位小数变化删除 loss。

### 17.3 第三步：冻结最终 Loss；第四步：条件模型实验

三个消融的冻结决策为：删除 `L_rhythm`、保留 `L_effort`、删除 `L_pol`，候选最终目标为 $L_{\mathrm{sync}}+0.25L_{\mathrm{effort}}$。因此第五个实验确定为 `B0_final_loss_patchmixer`，用相同 PatchMixer、三个正式 seed 和 50 epochs 检查同时删除两项后的交互，并建立最终 loss 下的新参照。`M1_multiscale_time` 顺延为第六个实验，避免同时改变 loss 与模型。

多尺度候选只允许围绕当前任务重新定义一个干净的纯时域模型，不直接恢复历史 STFT 双分支、旧辅助头、旧输出约束或旧实验 runner。其尺度应覆盖统一频带对应的快慢周期，且正式输出仍只由公共 $\Pi=S\circ B$ 定义，避免模型内部再叠加一套与协议竞争的输出语义。

### 17.4 Test 与独立任务边界

本阶段全部只使用 train/validation。Designated test 继续封存，直到最终 loss、候选模型集合、训练预算和 validation 选择全部冻结。Lag 范围敏感性继续作为独立任务，不与 loss 消融或多尺度模型实验混合，也不因第 16 节约 18% 的边界命中率临时修改 `±0.30 s`。

三个消融均只生成 validation 结果，没有运行 designated test。详细结果和冻结决定见第 18 节。

## 18. 三项 Loss 消融结果与候选最终 Loss（2026-08-03）

三个实验根目录为 `runs/tho_restart_a1_no_rhythm`、`runs/tho_restart_a2_no_effort`、`runs/tho_restart_a3_no_pol`。每项均使用 seed `20260811 / 20260812 / 20260813`、完整 train/validation、50 epochs 与 Local RR checkpoint selector，共 9 个 run。运行均基于干净 commit `aea78a3`，权重组合正确，训练历史有限，checkpoint 与历史最小 Local RR epoch 一致，逐样本结果只包含 validation。

下表沿用 sample direct mean，再对三个 seed 报告算术 mean ± sample SD；箭头只表示该 metric 自身的优劣方向，不构成综合分数：

| 实验 | Whole RR MAE ↓ | Local RR MAE ↓ | Global effort ↑ | Local effort ↑ | signed PCC ↑ |
|---|---:|---:|---:|---:|---:|
| `B0_full_loss_patchmixer` | `0.5315 ± 0.0060` | `0.6330 ± 0.0032` | `0.4890 ± 0.0042` | `0.4991 ± 0.0012` | `0.8392 ± 0.0006` |
| `A1_no_rhythm` | `0.5156 ± 0.0026` | `0.6272 ± 0.0011` | `0.4874 ± 0.0025` | `0.4977 ± 0.0010` | `0.8397 ± 0.0006` |
| `A2_no_effort` | `0.5188 ± 0.0100` | `0.6328 ± 0.0019` | `0.4786 ± 0.0058` | `0.4937 ± 0.0060` | `0.8435 ± 0.0012` |
| `A3_no_pol` | `0.5320 ± 0.0062` | `0.6336 ± 0.0040` | `0.4893 ± 0.0042` | `0.4993 ± 0.0011` | `0.8392 ± 0.0005` |

冻结解释严格沿用第 17.2 节的预注册任务轴：

- `A1_no_rhythm` 的 Whole/Local RR 在三个 seed 上均低于 B0。幅度有限，但方向一致，且该项没有在自身目标轴上显示收益；结合精简方向，删除 `L_rhythm`。
- `A2_no_effort` 的 Global/Local effort 在三个 seed 上均不高于 B0，说明 `L_effort` 提供了独立 effort 监督。PCC 与 IBI 的改善属于跨任务交换，不覆盖预注册解释轴，因此保留 `L_effort` 及权重 `0.25`。
- `A3_no_pol` 与 B0 的五项核心指标、最佳 epoch、负 PCC 样本和跨 seed 波动近似不变；`L_sync` 已足以固定方向，因此删除 `L_pol`。

九个消融 run 的 Local RR prediction-valid fraction 均为 `1.0`、prediction degenerate fraction 均为 `0.0`；每个 seed 仍只有 `1/2675` 个负 PCC 样本。未发现 NaN/Inf、数据泄漏、split 变化或 checkpoint 失配。未做 paired delta、显著性检验或 test 推理。

候选最终 loss 冻结为：

$$
L_{\mathrm{final}}=L_{\mathrm{sync}}+0.25L_{\mathrm{effort}}.
$$

由于两个 leave-one-term-out 结果不能单独排除联合删除的交互，第五个实验运行 `B0_final_loss_patchmixer`，只同时设置 `rhythm_weight=0` 和 `pol_start_weight=0`，其他科学配置完全不变。第 20 节结果未见不良交互，因此零权重分量已从当前实现物理删除，第六个实验进入多尺度模型。

第五个实验与 provenance 例外见第 20 节。旧 run 的 resolved config 和结果原地保留；当前默认配置不再携带已删除分量的参数。

## 19. 固定呼吸带传统基线（2026-08-03）

为建立深度学习与传统固定滤波方法的同口径参照，新增确定性基线
`F0_fixed_band_bcg`。它不重新设计滤波器，而是直接读取当前 research v2 数据集已经导出的
`bcg_resp_band_state_aligned_segment_soft_z`，将其作为 prediction 与同一窗口的
`target_waveform_segment_soft_z_key` 比较。该源信号来自 `0.05–0.70 Hz` 四阶零相位
Butterworth 呼吸带 BCG，并使用与当前深度学习输入相同的 segment soft-z 表示层级。

比较口径完全复用现行协议：相同 dataset admission、完整 validation split、180 秒窗口、
$\Pi=S\circ B$、五项 primary、IBI + coverage、target-only eligibility、prediction 失败值和
逐 sample direct mean。该方法没有训练、随机初始化或 checkpoint selector，因此只产生一个
确定性结果，不构造 seed mean ± SD。Validation 不计算 test-only coherence/nDTW，designated
test 继续封存。

完整 validation 共 `2675` 个 sample，结果保存于
`runs/tho_fixed_band_baseline/20260803_152032_531386`：

| 指标 | `F0_fixed_band_bcg` | `B0_full_loss_patchmixer` 三 seed mean | 方向 |
|---|---:|---:|---|
| Whole-window RR MAE（bpm） | `1.059457` | `0.531473` | 越低越好 |
| Local RR MAE（bpm） | `1.662157` | `0.632993` | 越低越好 |
| Local RR prediction-valid fraction | `1.000000` | `1.000000` | 越高越好 |
| Global effort Spearman | `0.431058` | `0.488964` | 越高越好 |
| Local effort Spearman | `0.434005` | `0.499099` | 越高越好 |
| Lag-aware signed PCC | `0.731464` | `0.839202` | 越高越好 |
| IBI-MedAE（s） | `0.071889` | `0.083449` | 越低越好，必须结合 coverage |
| IBI coverage | `0.581893` | `0.845473` | 越高越好 |
| IBI interpretable fraction | `0.428411` | `0.728474` | 越高越好 |

固定呼吸带基线全部 Local RR prediction 有效，joint target eligibility 为 `1.0`，prediction
degenerate fraction 为 `0.0`。其较低的条件 IBI-MedAE 只来自 coverage 达标的 `1146` 个
sample；由于 IBI coverage 和 interpretable fraction 明显更低，不能据此单独声称逐呼吸节律
优于 PatchMixer。整体结果表明固定呼吸带信号已经提供较强的波形与节律信息，但在 Whole/Local
RR、努力趋势、signed PCC 和 IBI 覆盖上均留下明确的深度学习改善空间。

实现入口为 `scripts/eval_tho_fixed_band_baseline.py`，逐 sample 指标、summary、resolved config
与执行 manifest 均已保存。未修改数据、split、target、metrics、checkpoint 规则或 designated
test 状态。新增定向测试与现有协议测试通过，当前全量测试为 `257 passed`。

## 20. 最终 Loss 的 PatchMixer 联合删除实验（2026-08-03）

`B0_final_loss_patchmixer` 位于 `runs/tho_restart_b0_final_loss_patchmixer`，使用相同 PatchMixer、完整 train/validation、50 epochs、seed `20260811 / 20260812 / 20260813` 和 Local RR checkpoint selector，只同时将 `rhythm_weight` 与 `pol_start_weight` 置零。三个 run 均完成 50 epochs，最佳 epoch 为 `7 / 7 / 8`，checkpoint epoch 与训练历史一致，逐样本结果只包含 validation。

按 sample direct mean，再对三个 seed 报告算术 mean ± sample SD：

| 指标 | `B0_full_loss_patchmixer` | `A1_no_rhythm` | `B0_final_loss_patchmixer` |
|---|---:|---:|---:|
| Whole-window RR MAE（bpm） | `0.5315 ± 0.0060` | `0.5156 ± 0.0026` | `0.5127 ± 0.0030` |
| Local RR MAE（bpm） | `0.6330 ± 0.0032` | `0.6272 ± 0.0011` | `0.6270 ± 0.0008` |
| Global effort Spearman | `0.4890 ± 0.0042` | `0.4874 ± 0.0025` | `0.4880 ± 0.0025` |
| Local effort Spearman | `0.4991 ± 0.0012` | `0.4977 ± 0.0010` | `0.4982 ± 0.0011` |
| Lag-aware signed PCC | `0.8392 ± 0.0006` | `0.8397 ± 0.0006` | `0.8397 ± 0.0007` |
| IBI-MedAE（s） | `0.08345 ± 0.00097` | `0.08429 ± 0.00145` | `0.08430 ± 0.00117` |
| IBI coverage | `0.84547 ± 0.00173` | `0.84561 ± 0.00100` | `0.84553 ± 0.00110` |
| IBI interpretable fraction | `0.72847 ± 0.00213` | `0.72735 ± 0.00374` | `0.72710 ± 0.00411` |

联合删除相对 `A1_no_rhythm` 没有出现任务轴退化：Local RR、effort 与 signed PCC 保持相当，Whole RR 的三 seed mean 进一步降低。三个 run 的 Local RR prediction-valid fraction 均为 `1.0`、prediction degenerate fraction 均为 `0.0`，每个 seed 仍只有 `1/2675` 个负 PCC 样本；未发现 NaN/Inf、checkpoint 失配、split 变化或 test 推理。最佳 lag 边界命中率为 `18.13% ± 0.04%`，继续按既有独立 lag-sensitivity 边界处理。

### 20.1 Provenance 例外

三个 run 的 manifest 均记录 commit `7d86a0d`，但 `git_dirty=true`。运行现场审计发现 dirty 内容来自并行存在的固定呼吸带 baseline、EWT 资料以及文档修改；当前可见变更没有修改或导入 `train_tho.py` 所使用的配置、模型、loss、训练引擎或数据路径。固定带 baseline 只由其独立入口和测试导入，协议文档在首个 run 启动后发生修改，也不参与运行时计算。

因此数值可用于确认联合删除与冻结最终 loss，但不包装成完全干净工作树上的最高等级复现证据。2026-08-03 用户明确接受该 `dirty-but-runtime-audited` provenance 例外并选择继续，不重复三个 seed；若未来将该结果作为严格论文复现证据，应从干净隔离 worktree 重新运行。该例外不改变数据、split、target、metrics、checkpoint selector 或 designated-test 封存规则。

### 20.2 当前实现结论

最终训练目标确认并物理精简为 $L_{\mathrm{sync}}+0.25L_{\mathrm{effort}}$。当前 loss 实现、默认配置和训练日志已删除 rhythm 频谱计算、polarity SmoothL1、退火状态及对应参数/列；旧 run 与旧 resolved config 原地保留用于追溯，当前训练配置若出现旧 rhythm/polarity 权重字段会明确失败。定向实验生命周期与当前固定带 baseline 测试均通过。下一科学实验为第六个 `M1_multiscale_time`，仍只使用 train/validation。

## 21. M1 参数匹配多尺度纯时域模型（2026-08-03）

第六个实验 `M1_multiscale_time` 只检验“在近似相同参数预算下，把单一短 patch 表示重新分配到多个呼吸周期尺度是否有益”。当前 PatchMixer 使用 `patch_len=256`、`stride=128`、`base_channels=16`、2 个 mixer block，共 `11408` 个可训练参数。M1 冻结为：

- 模型注册名：`multiscale_patch_mixer1d`。
- 四个纯时域分支，patch 长度 `256 / 512 / 1024 / 2048` 点，即 `2.56 / 5.12 / 10.24 / 20.48 s`。
- 每个分支 stride 固定为 patch 长度的 `0.5`，使用 Hann overlap-add 和 2 个 mixer block。
- 每个分支 `base_channels=1`，四个 waveform 分支用 4 个可学习 softmax 标量融合；总参数 `11664`，比 B0 多 `256`（`2.24%`）。
- 模型只输出未投影 raw head，不内置低通、高通、FFT mask、尺度规范化或旧 bandlimited output；正式输出仍统一由公共 $\Pi=S\circ B$ 定义。

选择 `base_channels=1` 不是声称单通道最优，而是防止四分支模型把“多尺度”与约 12 倍参数增长混在一起。该实验回答的是参数匹配下的结构重分配；若结果不佳，只能否定这一冻结候选，不能外推为所有多尺度模型无效。

除模型外，数据、split、最终两项 loss、optimizer、batch 128、50 epochs、三个正式 seed、Local RR checkpoint selector、metrics、聚合与无效值规则全部不变。不增加预算 pilot或权重搜索。工程阶段先运行一次 batch 128 单 batch GPU 验收；通过后直接运行 seed `20260811 / 20260812 / 20260813`，逐 seed 报告并给出描述性 mean ± sample SD。Designated test、lag sensitivity 和历史 STFT 双分支继续不进入本实验。

模型注册、raw-head 直通融合、参数匹配、架构冻结、最终两项 loss、实验生命周期和固定带 baseline 的全量回归测试为 `255 passed`。Batch 128 GPU 验收与正式三 seed 已完成，结果和模型冻结决定见第 22 节。

## 22. M1 Validation 结果与纯时域阶段收口（2026-08-04）

M1 正式结果位于 `runs/tho_restart_m1_multiscale_time`。三个 run 均使用 commit `a47475d`、完整 train/validation、50 epochs、batch 128、seed `20260811 / 20260812 / 20260813` 和冻结的最终两项 loss。三次训练均完整结束，历史与核心指标 finite，checkpoint epoch 与最低 Local RR epoch 一致，逐样本结果只包含 validation。

按 sample direct mean，再对三个 seed 报告算术 mean ± sample SD：

| Validation 指标 | `B0_final_loss_patchmixer` | `M1_multiscale_time` | M1 相对结果 |
|---|---:|---:|---|
| Whole-window RR MAE（bpm） | `0.5127 ± 0.0030` | `0.5286 ± 0.0075` | 退化 |
| Local RR MAE（bpm） | `0.6270 ± 0.0008` | `0.6514 ± 0.0057` | 退化 |
| Global effort Spearman | `0.4880 ± 0.0025` | `0.5116 ± 0.0054` | 改善 |
| Local effort Spearman | `0.4982 ± 0.0011` | `0.5261 ± 0.0048` | 改善 |
| Lag-aware signed PCC | `0.8397 ± 0.0007` | `0.8284 ± 0.0085` | 退化 |
| IBI-MedAE（s） | `0.08430 ± 0.00117` | `0.09168 ± 0.00182` | 退化 |
| IBI coverage | `0.84553 ± 0.00110` | `0.83388 ± 0.00491` | 退化 |
| IBI interpretable fraction | `0.72710 ± 0.00411` | `0.70604 ± 0.00617` | 退化 |

M1 的最佳 epoch 为 `50 / 48 / 4`，而最终 loss 下的 PatchMixer baseline 为 `7 / 7 / 8`。M1 三个 seed 的 effort 指标均改善，但 Whole/Local RR、signed PCC、IBI 与 coverage 均下降，且跨 seed 波动更大。Local RR 是预注册的唯一 checkpoint selector，因此不构造综合分数用 effort 改善覆盖主任务退化。两个 seed 的最佳 epoch 接近预算末端不触发事后 80-epoch 扩展：M1 固定 50 epochs 是参数匹配结构比较的一部分，结果已在多数任务轴落后，追加预算会成为结果驱动搜索。

完整性检查均通过：Local RR prediction-valid fraction 全部为 `1.0`，prediction degenerate fraction 全部为 `0.0`，每个 seed 只有 `1/2675` 个负 PCC 样本；无 NaN/Inf、split 变化、checkpoint 失配或 test 推理。最佳 lag 边界命中率为 `19.17% ± 0.14%`，略高于 PatchMixer baseline 的 `18.13% ± 0.04%`，不改变已冻结 `±0.30 s` 规则。

三个 M1 manifest 均记录 `git_dirty=true`。运行时主工作树的可见 dirty 内容为 `.gitignore` 与 `docs/methods/`，不在训练导入或数据路径中；用户已明确拒绝额外 worktree，并按既定口径接受 runtime-audited dirty provenance。M1 结果可用于当前 validation 模型选择，但不声称是完全干净工作树的最高等级复现证据。

### 22.1 纯时域阶段结论

M1 的 validation 决策至此冻结，不再继续其预算、base channel 或纯时域多尺度权重搜索：

- 协议学习 baseline：`patch_mixer1d + L_sync + 0.25 L_effort`，保留 `B0_final_loss_patchmixer` 的三个 `checkpoint_best_local_rr.pt` 作为后续模型的 validation 参照；它不是尚未完成的最终研究模型。
- 确定性传统参照：`F0_fixed_band_bcg`。
- `M1_multiscale_time`：作为参数匹配的纯时域任务交换/负结果留档，不继续调参；是否进入最终 designated test 不在此处提前决定。

原“下一步直接执行 designated test”的判断撤回。下一阶段先在冻结的数据、loss、metrics、checkpoint selector 与训练预算下完成时域 + STFT/时频融合候选；只有该阶段的候选集合和 validation 决策全部冻结后，才能重新确定最终 test 模型集合。Test 不参与模型、epoch、频带、阈值或 detector 的选择。

## 23. 下一阶段：时域 + STFT/时频融合（2026-08-04）

### 23.1 科研角色与范围

后续主研究路线确定为**时域 + STFT/时频融合**。`B0_final_loss_patchmixer` 只作为统一协议下的纯时域参照，M1 只回答已经完成的参数匹配多尺度问题；不再围绕纯时域模型继续宽度、尺度或预算搜索。当前也不恢复历史模型 zoo、旧双分支 runner、辅助 STFT target head、复数输出头、门控/cross-attention 组合或旧实验结论。

第一项融合实验只回答一个问题：**在保持 PatchMixer 时间分支、训练目标和评价协议不变时，从同一 BCG 输入提取的局部呼吸带 STFT 表示能否提供有用的互补信息。** STFT 只读取输入 BCG，不读取 target，不产生新的监督项，也不改变最终输出仍为 raw waveform、正式评价统一经过公共 $\Pi=S\circ B$ 的语义。

### 23.2 第一候选 `T1_time_stft_fusion`（设计已冻结）

为避免再次把多个结构变量捆绑在一次实验中，第一候选采用以下最小设计：

| 部分 | 冻结值 | 理由 |
|---|---|---|
| 时间分支 | 与 B0 完全相同的 `PatchMixer1D` | 只增加时频表示，保留直接可比的时间域参照 |
| STFT 输入 | 同一个 `bcg_rawish_segment_soft_z_key` | 不引入第二数据源或 target 派生特征 |
| STFT 窗/步长 | 30 秒 / 10 秒，`center=False`、不 padding、symmetric Hann | 与已解释的局部呼吸尺度一致；180 秒产生 16 帧，避免额外引入另一套局部时间口径 |
| STFT 频带 | `0.05–0.70 Hz`，端点包含 | 遵守当前统一呼吸频带，不利用 target 选带，也不在第一版增加宽频带解释 |
| STFT 特征 | `log1p` 功率谱；每帧频率形状规范化，并增加一条跨 16 帧规范化的相对带内能量通道 | 同时表达局部主频/谱形与相对呼吸努力，且不需要 train/test 统计量 |
| 时频编码器 | `Conv1d(21,16,3)` → `GroupNorm(1,16)` → SiLU → `Conv1d(16,16,3)` → SiLU | 先验证表示本身，不同时比较 Conv2D、Transformer、分频带或可学习频带 |
| 对齐 | 将 16 帧特征线性插值到 PatchMixer token 数 | 只做确定性时间轴对齐，不引入可学习重采样 |
| 融合 | 16 帧线性插值至 token 数，`align_corners=False`；经 `Conv1d(16,16,1)` 后在 mixer 后残差相加，projection 零初始化 | 单一融合位置；初始输出严格等于 B0，输出头与 B0 相同 |
| 输出与 loss | raw waveform；$L_{\mathrm{sync}}+0.25L_{\mathrm{effort}}$ | 不改变已冻结输出空间和训练目标 |

精确特征定义如下。对第 $t$ 个 3000 点 frame 先减去该 frame 均值，再乘 symmetric Hann $w[n]$：

$$
P_{t,k}=\left|\operatorname{rFFT}_{3000}\left((x_t[n]-\bar x_t)w[n]\right)_k\right|^2.
$$

固定 `rFFT norm="backward"`。100 Hz、3000 点下，`0.05–0.70 Hz` 实际包含 $k=2,\ldots,21$ 共 20 个离散 bin（$0.066\overline6,\ldots,0.70$ Hz）；“端点包含”不虚构不存在的 0.05 Hz bin。令 $\epsilon=10^{-8}$、$u_{t,k}=\log(1+P_{t,k})$，每帧谱形为：

$$
z_{t,k}=\frac{u_{t,k}-\operatorname{mean}_j u_{t,j}}
{\sqrt{\operatorname{mean}_j(u_{t,j}-\operatorname{mean}_\ell u_{t,\ell})^2+\epsilon}}.
$$

带内能量先取 $r_t=\log(1+\operatorname{mean}_k P_{t,k})$，再只在同一个 180 秒 sample 的 16 帧之间规范化：

$$
a_t=\frac{r_t-\operatorname{mean}_s r_s}
{\sqrt{\operatorname{mean}_s(r_s-\operatorname{mean}_q r_q)^2+\epsilon}}.
$$

时频分支输入为每帧 $[z_{t,2},\ldots,z_{t,21},a_t]$ 共 21 维。该定义保留窗内相对 effort，主动丢弃与公共 $S$ 不一致的整段绝对尺度；零信号严格得到全零有限特征。谱计算固定 float32，不受 AMP 影响。

T1 共 `13520` 个可训练参数，相比 B0 的 `11408` 增加 `2112`（`18.51%`）。第一版不强制参数完全相同：T1 回答的是“增加一个受限时频分支”是否有益。如果 T1 在三 seed validation 上形成值得继续的改善，再把“STFT 信息收益”与“新增容量收益”的区分作为后续单独对照；不提前把纯时域容量搜索扩展为主线。

### 23.3 实验顺序与停止边界

1. 第 23.2 节的精确数学和模块契约已经冻结，并以独立 `time_stft_fusion1d` 实现；模型注册表保留旧模型，但 T1 不调用旧 `time_stft_dual1d` 的历史多模式配置。
2. 运行定向单测、CPU 生命周期 smoke 和一次 batch 128 GPU 前向/反向验收；这些只证明实现成立，不形成科研结果。
3. 保持完整 train/validation、50 epochs、batch 128、seed `20260811 / 20260812 / 20260813`、Local RR checkpoint selector 和现有 metrics 不变，运行 T1 三 seed 正式实验。
4. T1 完成前不增加 STFT-only、频带、窗长、融合位置、门控、编码器或 loss 消融。T1 结果完成后，再依据预先声明的多指标解释边界决定是否值得展开一个归因对照；不因单个 test 结果返工。

Designated test 在本阶段继续封存。最终 test 集合至少要等 T1 validation 完成后重新明确，不能沿用第 22 节曾经误写的“PatchMixer 已是最终模型”结论。

### 23.4 实现验收（2026-08-04）

- 新增独立 `resp_train/models/time_stft_fusion.py`，只包含冻结的呼吸带 STFT 特征和 T1 融合模型；没有调用或扩展旧 `time_stft_dual1d`。
- 当前配置已切换为 `time_stft_fusion1d`、seed `20260811` 和独立 T1 run root；配置校验拒绝窗长、步长、通道、epsilon 或 PatchMixer 骨干漂移。
- 合成验收覆盖：20 个冻结频点、16 帧、零输入有限全零语义、0.2 Hz 频点定位、幅度调制相对 effort、零初始化时与 B0 输出严格相同，以及 projection 暖启后 STFT encoder 获得有限非零梯度。
- `./.venv/bin/python -m pytest -q tests`：`272 passed`。
- 一次性 CPU 生命周期 smoke 使用 8 train + 8 validation、batch 4、1 epoch，成功生成两个 checkpoint、完整最小训练历史、逐 sample metrics、summary、config、audit 与 manifest；输出目录为 `/tmp/tho_restart_t1_cpu_smoke/20260804_021007_216139`。该 smoke 的数值不构成科研结果。
- Batch 128 GPU 验收使用 128 train + 32 validation、1 epoch、seed `20260811`，运行目录为 `/tmp/tho_restart_t1_batch128_acceptance/20260804_104201_521013`。运行生成完整生命周期产物，无 OOM、Traceback 或非有限训练值；两个 checkpoint 的全部模型 tensor 均 finite，零初始化 STFT projection 在一次 optimizer step 后达到非零最大绝对值约 `9.9997e-4`，证明融合路径已开始学习。该验收的 validation 数值不构成科研结果。
- GPU 验收 manifest 记录 commit `ce8b092` 且 `git_dirty=true`，符合实现尚未提交时的预期；只用于工程验收，不作为正式 provenance。验收完成后执行的正式三 seed T1 见第 24 节；designated test 仍未运行。

## 24. T1 三 seed Validation 结果（2026-08-04）

T1 正式结果位于 `runs/tho_restart_t1_time_stft_fusion`。三个 run 均使用 commit `4f7f60c`、完整 train/validation、50 epochs、batch 128、seed `20260811 / 20260812 / 20260813` 和冻结的两项 loss；最佳 epoch 均为第 7 epoch。三个 checkpoint 都与各自训练历史的最低 Local RR epoch 一致，checkpoint tensor 与训练历史全部 finite，每个 run 均输出 2675 个 validation sample，prediction-valid fraction 为 `1.0`、prediction-degenerate fraction 为 `0.0`，每个 seed 只有 `1/2675` 个负 signed PCC 样本。

按 sample direct mean，再对三个 seed 报告算术 mean ± sample SD：

| Validation 指标 | B0 final-loss PatchMixer | T1 time + STFT | T1 相对 B0 |
|---|---:|---:|---|
| Whole-window RR MAE（bpm） | `0.512706 ± 0.002988` | `0.523501 ± 0.003348` | 退化 `+0.010794`；0/3 seed 改善 |
| Local RR MAE（bpm） | `0.626951 ± 0.000812` | `0.626562 ± 0.002510` | 均值改善 `-0.000389`，但仅 1/3 seed 改善，视为基本持平 |
| Global effort Spearman | `0.488025 ± 0.002532` | `0.483581 ± 0.003852` | 退化 `-0.004444`；0/3 seed 改善 |
| Local effort Spearman | `0.498227 ± 0.001125` | `0.493902 ± 0.002162` | 退化 `-0.004325`；0/3 seed 改善 |
| Lag-aware signed PCC | `0.839730 ± 0.000659` | `0.838840 ± 0.002091` | 退化 `-0.000889`；0/3 seed 改善 |
| IBI-MedAE（s） | `0.084295 ± 0.001170` | `0.085213 ± 0.001297` | 退化 `+0.000918`；1/3 seed 改善 |
| IBI coverage | `0.845532 ± 0.001099` | `0.847339 ± 0.001961` | 改善 `+0.001807`；2/3 seed 改善 |
| IBI interpretable fraction | `0.727103 ± 0.004112` | `0.727477 ± 0.003605` | 改善 `+0.000374`；2/3 seed 改善但幅度很小 |
| Lag 边界命中比例 | `0.181308 ± 0.000374` | `0.182928 ± 0.002704` | 略增 `+0.001620` |

T1 增加 `18.51%` 参数后，没有在最高优先级 Local RR 上形成跨 seed 一致改善，并在 Whole RR、两项 effort 与 signed PCC 上三个 seed 一致退化。IBI coverage 的小幅改善不足以覆盖主要任务轴的退化，因此 T1 不作为当前有效候选，也不因第 7 epoch 即最佳而追加预算。

该结果不能外推为“时频融合无效”。T1 只检验了一个受限候选：输入 STFT 与输出算子共用 `0.05–0.70 Hz`、只使用 magnitude/power 表示、30 秒/10 秒单尺度，并在 PatchMixer mixer 后残差注入。三个最佳 checkpoint 的 STFT projection 权重 L2 norm 为 `0.881–0.956`，说明分支已经学习到非零注入，失败不能简单归因于零初始化导致分支未启动；更直接的解释是当前窄带表示没有提供足够互补信息，或注入位置无法有效利用它。

三个正式 manifest 均记录 commit `4f7f60c` 且 `git_dirty=true`。运行时未提交内容为独立 IEWT baseline、`.gitignore` 和说明文档；训练入口及其导入路径不导入 `resp_train.baselines`，因此按用户此前接受的 `dirty-but-runtime-audited` provenance 例外使用这些 validation 结果。该例外不改变数据、split、target、loss、metrics、checkpoint selector 或训练预算。三个 run 均未执行 designated test。

下一候选不能同时搜索频带、窗长、编码器与融合位置。当前最值得先重新审视的是 **输入 STFT 频带是否必须等于输出/评价频带**：`0.05–0.70 Hz` 对公共输出、loss 与 metrics 仍应冻结，但把同一窄带强加给 BCG 输入分支，可能使 STFT 与时间分支看到的信息高度冗余。若继续 T2，优先只扩大输入侧频率支持并保持输出侧 `0.05–0.70 Hz` 不变；具体输入上限与固定维度压缩方式需在实现前单独冻结。

## 25. 既有有效结构复核与 T2–T4 模型矩阵（2026-08-04）

### 25.1 对 T1 后续方向的修正

回看重启前仍被列为 active evidence 的模型与普通 validation 结果后，撤回“下一步自行设计 `0.05–3 Hz` T2”的建议。旧 G2 已比较 `0.05–1.2 / 3 / 8 Hz`、bandgroup、bandenergy 与 high-only；普通 checkpoint 口径下，`0.05–8 Hz` 的 fullband Conv2D 最稳，`0.05–3 Hz` 没有超过它。直接重复 `0.05–3 Hz` 会忽略已有负/混合证据。

T1 也不是旧有效 G3 结构的复现。它同时使用 30 秒/10 秒、`0.05–0.70 Hz`、20 个频点、16 帧、逐帧规范化 power、Conv1D 和 post-mixer 注入；旧 G3 anchor 使用 20 秒/2.5 秒、`0.05–8 Hz`、160 个频点、73 帧、N0 `log1p` magnitude、Conv2D 和 pre-mixer 注入。因此 T1 只作为独立负结果，不再围绕它的小改动继续搜索。

旧结果的 loss、checkpoint 和 metrics 与当前协议不同，且部分旧证据包含历史 test 观察，不能直接与 B0/T1 数值比较，也不能作为当前 designated test 调参依据。本节只使用旧普通 validation 结果选择**待重训结构来源**；所有新结论必须来自当前 train/validation、当前 loss/metrics 与当前 Local RR checkpoint。

### 25.2 冻结候选

三个候选均直接复用现有 `TimeStftDual1D`、`STFTEncoder`、`FusionHead` 和 PatchMixer，不新增模型类，不恢复旧 runner、旧 loss、旧 checkpoint gate、top-k、early stopping、辅助 target-STFT head 或兼容配置。

| 实验 | 结构来源与科学问题 | STFT 输入 | 编码/融合 | 参数量 | 状态 |
|---|---|---|---|---:|---|
| `T2_g3c_wide_native` | 复核旧 active anchor `G3_C_wide_8p0` 在当前协议下是否仍有净收益 | `win=2000`、`hop=250`、`center=True`、`0.05–8 Hz`、N0 `log1p magnitude`，73 帧/160 bins | Conv2D 16 ch；零初始化 `1×1`；`pre_mixer native_inject`；原生 PatchMixer decoder | `14192` | 第一顺位 |
| `T3_e3a0_wide_concat` | 复核旧 E1-D/E3-A0.0 强简单融合；检验 richer fusion decoder 是否比窄 token 注入更会利用 STFT | `win=3000`、`hop=500`、`center=True`、`0.05–8 Hz`、N0，37 帧/239 bins | Conv2D 16 ch；time/STFT 对齐到 600；concat + deep FusionHead | `16305` 总参数，其中 PatchMixer 原生 decoder 在该路径不参与 forward | 第二顺位 |
| `T4_g3c_bandenergy_native` | 复核旧 G3 中唯一仍有讨论价值的低维/条件修正候选 | 与 T2 相同；按既有 5 个重叠频带生成能量序列 | bandenergy Conv1D 16 ch；`pre_mixer native_inject` | `12752` | 第三顺位/条件候选 |

T4 的五个既有重叠频带固定为 `0.05–0.30 / 0.10–0.70 / 0.30–1.20 / 0.70–3.00 / 3.00–8.00 Hz`，不重新选带。T3 的 `fuse_len=600` 和 deep FusionHead 是该完整历史结构的一部分；它不是 T2 的单变量消融，因此只比较完整模型结果，不把差异归因给单独的融合位置。

输入频带与输出频带明确分工：`0.05–8 Hz` 只定义模型从 BCG 可观察的时频上下文，可能包含呼吸位移、心动及呼吸调制信息；公共输出、loss 与所有正式 metrics 继续固定 `0.05–0.70 Hz`。STFT 只读取 BCG，不读取 target，不存在 target 选带或 test-target 泄漏。

### 25.3 执行顺序

1. 为三个冻结候选准备独立 current-protocol config 和严格字段校验；复用现有模型实现，只补当前生命周期测试。
2. T2 与 T3 各做一次 batch 128 GPU 验收；T4 与 T2 共用更低维的 native 路径，T2 通过后无需单独做显存验收。
3. 先运行 T2 三 seed，再根据完整五项 primary、IBI + coverage 和有效性结果决定是否依次运行 T3、T4；候选配置本身不因 T2 数值修改。
4. 每个正式候选均固定完整 train/validation、50 epochs、batch 128、seed `20260811 / 20260812 / 20260813` 和 Local RR checkpoint selector。不同候选不追加 80 epoch，不做 top-k、容差级联或模型内诊断搜索。
5. 三个候选都不自动进入 designated test。只有当前模型研究阶段收口后，才重新冻结最终 test 集合。

明确停止：不再准备 `0.05–3 Hz` 单独频带臂、STFT-only、gated/cross-attention、token-context、SST/CWT dense map、target-STFT loss、complex STFT 输出或 auxiliary/residual head；这些路线已有旧负/混合证据，且不符合当前精简方向。

### 25.4 当前实现验收

- T2 作为当前默认配置；T3、T4 分别使用独立 config。三个 config 只启用冻结字段，额外 auxiliary、gate、attention、输出头或未预注册模型字段会被配置校验拒绝。
- 未新增模型类或复制既有融合逻辑；三个候选均由现有 `time_stft_dual1d` 注册入口构建。
- 定向验收覆盖三个 config 的参数量、完整 18000 点 forward、finite 输出、频点数、bandenergy 数量，以及 T2 零初始化时与同一 PatchMixer 时间分支逐元素相同。
- 三个候选分别用 8 train + 8 validation、batch 4、1 epoch 完成一次性 CPU 生命周期 smoke，均生成完整 checkpoint、history、metrics 和 manifest；数值不构成科研结果。
- `./.venv/bin/python -m pytest -q tests`：`282 passed`。
- T2 batch 128 GPU 验收目录为 `/tmp/tho_restart_t2_batch128_acceptance/20260805_155009_160323`；T3 为 `/tmp/tho_restart_t3_batch128_acceptance/20260805_155030_379024`。两者均使用 commit `b508a8a`、128 train + 32 validation、1 epoch、seed `20260811`，无 OOM、Traceback、非有限历史或非有限 checkpoint tensor，prediction-valid fraction 均为 `1.0`、prediction-degenerate fraction 均为 `0.0`。T2 零初始化 projection 在一次 optimizer step 后为非零，T3 完整 concat-deep 生命周期成功。
- 两个验收 manifest 均因独立 IEWT 工作区内容记录 `git_dirty=true`；这些内容不在训练入口导入路径中。验收仅属工程证据。T4 与 T2 共用更低维 native 路径，不重复 GPU 验收。
- GPU 验收完成时尚未运行正式 T2–T4 validation；随后完成的 T2 结果见第 26 节。Designated test 仍未执行。

## 26. T2 `G3_C_wide_8p0` 当前协议复核结果（2026-08-05）

T2 正式结果位于 `runs/tho_restart_t2_g3c_wide_native`。三个 run 均使用 commit `5632710`、完整 train/validation、50 epochs、batch 128 和 seed `20260811 / 20260812 / 20260813`；最佳 epoch 为 `8 / 6 / 10`，全部与训练历史的最低 Local RR epoch 一致。三个 checkpoint tensor 与训练历史全部 finite，每个 run 均输出 2675 个 validation sample，prediction-valid fraction 为 `1.0`、prediction-degenerate fraction 为 `0.0`，每个 seed 只有 `1/2675` 个负 signed PCC 样本。STFT projection L2 norm 为 `1.218–1.745`，证明宽频分支已被实际使用。

按 sample direct mean，再对三个 seed 报告算术 mean ± sample SD：

| Validation 指标 | B0 final-loss PatchMixer | T2 wide native | T2 相对 B0 |
|---|---:|---:|---|
| Whole-window RR MAE（bpm） | `0.512706 ± 0.002988` | `0.512559 ± 0.002194` | 基本持平 `-0.000148`；2/3 seed 改善 |
| Local RR MAE（bpm） | `0.626951 ± 0.000812` | `0.624996 ± 0.001908` | 改善 `-0.001955`；2/3 seed 改善 |
| Global effort Spearman | `0.488025 ± 0.002532` | `0.493966 ± 0.003763` | 改善 `+0.005940`；2/3 seed 改善，另 1 seed 近似持平 |
| Local effort Spearman | `0.498227 ± 0.001125` | `0.502769 ± 0.002875` | 改善 `+0.004541`；3/3 seed 改善 |
| Lag-aware signed PCC | `0.839730 ± 0.000659` | `0.840499 ± 0.000245` | 改善 `+0.000770`；3/3 seed 改善 |
| IBI-MedAE（s） | `0.084295 ± 0.001170` | `0.087541 ± 0.002440` | 退化 `+0.003245`；仅 1/3 seed 改善 |
| IBI coverage | `0.845532 ± 0.001099` | `0.845130 ± 0.001002` | 略退 `-0.000403`；1/3 seed 改善 |
| IBI interpretable fraction | `0.727103 ± 0.004112` | `0.728224 ± 0.002617` | 略增 `+0.001121`；1/3 seed 改善 |
| Lag 边界命中比例 | `0.181308 ± 0.000374` | `0.180312 ± 0.009758` | 均值略降，但 seed 波动明显 |

T2 与 T1 不同：它没有在主要任务轴上形成系统性退化，而是在 Local RR、两项 effort 和 signed PCC 上给出小幅正信号，Whole RR 持平。因此旧 G3 结构在当前协议下仍有一定有效性，T2 保留为 active candidate。与此同时，Local RR 改善幅度很小且只有 2/3 seed，IBI-MedAE 退化约 `3.25 ms`，不能把 T2 表述为全面胜出或直接冻结为最终模型。

该结果触发已预注册的后续比较：先运行 T3，检验旧 concat-deep 强融合是否能放大时频收益；之后是否运行 T4，结合 T3 结果和 T2 的 IBI 退化再决定，但不修改 T4 冻结配置。三个 T2 manifest 均记录 `git_dirty=true`；未提交内容仍为独立 IEWT baseline、`.gitignore` 和说明文档，不在训练入口导入路径中，按既定 runtime-audited provenance 例外使用。未执行 designated test。

## 27. 包络 metrics 第二版冻结与实现（2026-08-05）

原 Global/Local effort Spearman 同时受到低 target 变化量、短局部序列和重叠局部聚合影响，不能继续
作为当前两个包络主轴。它们从当前默认输出和 summary 退出，替换为第 6.4 节的 envelope trajectory
MAE 与 global envelope modulation error；train-frozen target-stratified Spearman 只作补充解释。

实现位于 `resp_train/metrics/task.py`。公共 evaluator 仍只执行一次 $\Pi=S\circ B$，随后在完整
180 秒 canonical waveform 上计算 35 点 log-RMS，不复用 waveform lag。两个主指标覆盖全部 admitted
sample；主指标缺失或非有限使评价失败。逐样本输出同时保存 target modulation、stratum、Spearman
eligibility 和 prediction-degenerate 状态，summary 分层报告计数与均值，不生成总体 envelope Spearman。

Train-only 阈值由 `scripts/freeze_envelope_strata.py` 在 10141 个 admitted training targets 上复现并写入
三个 current config；生成产物为 `runs/envelope_strata_train_20260805.json`。旧 checkpoint sidecar 只允许在
复评入口补入这三个 evaluation-only 冻结字段；data、window、model、loss 及其他 evaluation 字段仍必须与
checkpoint 一致。该兼容只支持固定 checkpoint 重评，不改变训练 forward、loss 或 checkpoint selector。

本次 metric 变更不要求重训，但 2026-08-05 之前所有 Global/Local effort 数值不再属于当前主结果口径。
固定传统 baseline、IEWT 与已选模型 checkpoint 必须依次按新指标重评后才能形成新的 validation 比较。
Designated test 尚未运行，也不得因本次 validation 重评结果调整已冻结指标定义。

实现验收：定向协议测试 `47 passed`；完整当前测试 `289 passed`。Train-only 阈值复现成功，未读取
validation/test target，未启动训练或 designated test。

## 28. 新包络 metrics 的 F0 validation 重评（2026-08-05）

第一项重评为确定性传统基线 `F0_fixed_band_bcg`。运行目录：
`runs/tho_fixed_band_baseline_envelope_v2/20260805_165716_296054`。使用完整 validation 2675 个
sample、`max_windows=null`、固定 val seed `20260611`，未运行 designated test。Manifest 记录 commit
`da0c479`、`git_dirty=true`；按用户已接受的 dirty provenance 处理，不据此改动指标。

| 指标 | F0 validation |
|---|---:|
| Envelope trajectory MAE ↓ | `0.168553` |
| Global envelope modulation error ↓ | `0.212177` |
| Low target modulation Spearman ↑ | `0.221103`（1136/1136 eligible） |
| Medium target modulation Spearman ↑ | `0.458006`（652/652 eligible） |
| High target modulation Spearman ↑ | `0.680142`（887/887 eligible） |

两个主包络指标、target modulation 均为 2675/2675 finite；Spearman target-ineligible 和
prediction-degenerate 均为 0。Low/Medium/High 分别占 validation 的 `42.47% / 24.37% / 33.16%`；
不强求各占三分之一，因为边界严格由 training targets 冻结。按各层样本数加权后的 Spearman 恰好恢复
旧 Global effort Spearman `0.431058`，说明分层实现没有改变 sample-level 排序相关语义，只把低调制
sample 的影响显式拆开。

Whole RR、Local RR、signed PCC、IBI 与旧 F0 重评逐项完全一致，证明本次只替换包络指标路径，未改变
公共 canonical 输出或其他任务轴。不同 stratum 的 trajectory/modulation 绝对误差不可直接横向解释为
“高调制更差”，因为 target 动态范围本身不同；后续方法比较应在同一整体 split 及同一 target stratum
内进行。F0 重评通过，可进入 IEWT validation 重评。

## 29. 新包络 metrics 的 IEWT validation 重评（2026-08-05）

第二项重评为零相位 Python `IEWT`。运行目录：
`runs/tho_iewt_baseline_envelope_v2/20260805_170706_038393`。使用与 F0 完全相同的完整 validation
2675 个 sample、train-frozen envelope strata 和公共 evaluator；`max_windows=null`，未运行 designated
test。Manifest 记录 commit `da0c479`、`git_dirty=true`、`filter_phase=zero_phase` 和
`matlab_pointwise_parity=false`。

| 包络指标 | F0 | IEWT | IEWT − F0 | 方向 |
|---|---:|---:|---:|---:|
| Envelope trajectory MAE | `0.168553` | `0.193583` | `+0.025030` | 越低越好，IEWT 较差 |
| Global envelope modulation error | `0.212177` | `0.222981` | `+0.010804` | 越低越好，IEWT 均值较差 |
| Low Spearman | `0.221103` | `0.245810` | `+0.024707` | IEWT 较高 |
| Medium Spearman | `0.458006` | `0.403953` | `−0.054053` | IEWT 较低 |
| High Spearman | `0.680142` | `0.590512` | `−0.089630` | IEWT 较低 |

2675 个 `dataset_row_id` 一对一匹配，target modulation 和 stratum 逐样本完全相同。IEWT 的两个主
包络指标及 target modulation 均为 2675/2675 finite，Spearman target-ineligible 和 prediction-degenerate
均为 0。

逐样本配对显示：IEWT 的 trajectory MAE 只有 `23.96%` sample 优于 F0，三个 target stratum 的均值
都退化，其中 High 层增量最大（`+0.044621`）。因此 IEWT 在相对努力变化发生时间和轨迹形状上稳定不如
直接固定带基线。Global modulation error 的结论更有异质性：IEWT 有 `52.26%` sample 优于 F0，配对
delta 中位数为 `−0.002649`，但均值为 `+0.010804`，说明少量较大退化拉高均值。分层后，IEWT 只在
High 层改善调制范围误差（`0.335966` vs `0.346514`），Low/Medium 层均退化。

High 层同时出现“调制范围略好、trajectory MAE 和 Spearman 更差”，说明 IEWT 有时能恢复整体强弱范围，
但没有把增强/减弱放在正确时段。Low 层 Spearman 略高但两个主误差都更差，也再次说明低调制 target
上的排序相关只能作补充，不能覆盖主指标。

非包络任务轴上，IEWT 相对 F0 明显改善 Whole RR（`0.584319` vs `1.059457`）、Local RR
（`0.798060` vs `1.662157`）、signed PCC（`0.794842` vs `0.731464`）和 IBI coverage
（`0.798221` vs `0.581893`）。IBI-MedAE 条件样本数同时从 1146 增至 1726，因此其数值不能脱离
coverage 直接解释。综合结论是 IEWT 更擅长节律与波形同步，但不优于 F0 的相对努力轨迹恢复；两个
传统方法形成互补而非单一全面胜负。IEWT 重评通过，可进入已选模型 checkpoint 重评。

## 30. 新包络 metrics 的 B0 final-loss PatchMixer 重评（2026-08-06）

第三项重评为 `B0_final_loss_patchmixer` 三个既有 Local RR 最优 checkpoint，seed 为
`20260811 / 20260812 / 20260813`。逐 seed 新产物使用 `metrics_envelope_v2.csv` 与对应 summary/manifest，
不覆盖旧 metrics。每个 seed 都评价完整 validation 2675 个 sample；三个 manifest 记录复评 commit
`da0c479`、`git_dirty=true`，未运行 designated test。

三个 seed 的 sample ID、target modulation 和 stratum 完全一致；每个 seed 的两个主包络指标均为
2675/2675 finite，Spearman target-eligible 为 2675/2675，prediction-degenerate 为 0。旧 metrics 中
Whole RR、Local RR、prediction-valid fraction、signed PCC、IBI-MedAE 和 IBI coverage 与本次输出
逐样本最大绝对差均为 0，证明旧 checkpoint evaluation-only 配置迁移没有改变其他任务轴。

| Validation 指标 | B0 mean ± sample SD | F0 | IEWT |
|---|---:|---:|---:|
| Envelope trajectory MAE ↓ | `0.155620 ± 0.000653` | `0.168553` | `0.193583` |
| Global envelope modulation error ↓ | `0.244096 ± 0.005189` | `0.212177` | `0.222981` |
| Low Spearman ↑ | `0.291283 ± 0.005623` | `0.221103` | `0.245810` |
| Medium Spearman ↑ | `0.571272 ± 0.002540` | `0.458006` | `0.403953` |
| High Spearman ↑ | `0.678806 ± 0.001620` | `0.680142` | `0.590512` |

B0 的 trajectory MAE 三个 seed 均优于 F0 和 IEWT；相对 F0 平均降低 `0.012934`，逐 seed
`65.61%–67.33%` 的 sample 更好；相对 IEWT 平均降低 `0.037963`，逐 seed `80.56%–82.21%` 的
sample 更好。Low/Medium 层 B0 trajectory 明显最好；High 层 B0（`0.280477`）略差于 F0
（`0.273663`），但优于 IEWT（`0.318284`）。

Global modulation error 给出不同结论。B0 在 Low/Medium 层最好（分别为 `0.074821 / 0.185484`），
但 High 层升至 `0.503975`，明显差于 F0 的 `0.346514` 和 IEWT 的 `0.335966`，从而使整体均值反而
最差。B0 相对 F0/IEWT 的逐样本改善比例也只有 `42.21% / 41.76%`。三个 seed 都出现相同分层模式，
因此不是单 seed 偶然波动。

这一结果解释了新指标的必要性：B0 的 Low/Medium Spearman 最高、High Spearman 与 F0 基本持平，说明
努力变化的时间排序和趋势已学到；但训练 `L_effort` 使用相关性，本身不直接约束 log-envelope 动态范围，
所以旧 Spearman 无法暴露 High-modulation sample 的范围失配。当前证据不能仅由绝对误差判断是过度调制
还是调制压缩，因为逐样本产物只保存绝对 modulation error；在不新增诊断并重评前不对方向作猜测。

综合结论：B0 是三者中最好的相对努力轨迹恢复方法，也在节律和 waveform 指标上明显优于两个传统方法；
但它不是包络维度上的全面胜者，High target modulation 的全局变化幅度是明确短板。该短板不触发重训、
改 loss 或重选 checkpoint；先按同一口径重评 active T2 checkpoint，判断宽频时频分支是否改变这一模式。

## 31. 新包络 metrics 的 T2 wide-native 重评（2026-08-06）

第四项重评为 active candidate `T2_g3c_wide_native` 的三个既有 Local RR 最优 checkpoint，seed 为
`20260811 / 20260812 / 20260813`。每个 seed 生成独立 `metrics_envelope_v2.csv`、summary 和 manifest，
不覆盖旧结果；每个文件包含完整 validation 2675 个 sample，未运行 designated test。

三个 seed 的主包络指标均为 2675/2675 finite，Spearman target-eligible 为 2675/2675，
prediction-degenerate 为 0；sample ID、target modulation 和 stratum 完全一致。Whole RR、Local RR、
prediction-valid fraction、signed PCC、IBI-MedAE 和 IBI coverage 与旧 T2 metrics 的逐样本最大绝对差
均为 0。

| Validation 指标 | T2 mean ± sample SD | B0 | F0 | IEWT |
|---|---:|---:|---:|---:|
| Envelope trajectory MAE ↓ | `0.153601 ± 0.001251` | `0.155620` | `0.168553` | `0.193583` |
| Global envelope modulation error ↓ | `0.231247 ± 0.005794` | `0.244096` | `0.212177` | `0.222981` |
| Low Spearman ↑ | `0.274319 ± 0.004658` | `0.291283` | `0.221103` | `0.245810` |
| Medium Spearman ↑ | `0.583196 ± 0.002222` | `0.571272` | `0.458006` | `0.403953` |
| High Spearman ↑ | `0.709681 ± 0.004435` | `0.678806` | `0.680142` | `0.590512` |

与同 seed B0 逐样本配对时，T2 的 trajectory MAE 平均降低 `0.002019`。Seed `20260811/13` 有
`55.07% / 59.81%` sample 改善，seed `20260812` 仅 `47.33%`，且该 seed 均值只改善
`0.000098`；因此总体正信号很小，不应表述为每个 seed 均稳定胜出。Global modulation error 三个 seed
均降低，平均改善 `0.012849`，逐 seed 改善样本比例为 `62.92% / 68.71% / 50.36%`。

分层结果显示宽频分支主要帮助 Medium/High target modulation，而不是 Low：

| 分层指标 | T2 | B0 | T2 − B0 |
|---|---:|---:|---:|
| Low trajectory MAE ↓ | `0.075023` | `0.072589` | `+0.002434` |
| Medium trajectory MAE ↓ | `0.130643` | `0.130428` | `+0.000215` |
| High trajectory MAE ↓ | `0.271113` | `0.280477` | `−0.009364` |
| Low modulation error ↓ | `0.070583` | `0.074821` | `−0.004238` |
| Medium modulation error ↓ | `0.176802` | `0.185484` | `−0.008681` |
| High modulation error ↓ | `0.477033` | `0.503975` | `−0.026942` |
| Low Spearman ↑ | `0.274319` | `0.291283` | `−0.016964` |
| Medium Spearman ↑ | `0.583196` | `0.571272` | `+0.011924` |
| High Spearman ↑ | `0.709681` | `0.678806` | `+0.030875` |

T2 把 High trajectory MAE 改善到 `0.271113`，略好于 F0 的 `0.273663`，并取得四种方法中最高的
Medium/High Spearman；说明宽频时频上下文确实改善了较明显努力变化的时间结构。与此同时，High
modulation error 虽由 B0 的 `0.503975` 降至 `0.477033`，仍明显差于 F0/IEWT 的
`0.346514 / 0.335966`。因此宽频分支缓解但没有解决动态范围失配，并以 Low 层 trajectory/Spearman
小幅退化为代价。

当前新指标重评阶段已经完成：F0、IEWT、B0 与 active T2 均使用相同 validation targets 和冻结口径。
T2 保持 active candidate，但证据仍是多任务权衡而非全面优势；本轮重评本身不授权修改 loss、重训、
重选 checkpoint 或开启 designated test。

## 32. T3 `E3-A0 concat-deep` 当前协议结果（2026-08-06）

T3 正式结果位于 `runs/tho_restart_t3_e3a0_wide_concat`。三个 run 均使用 metrics 冻结提交 `aebe81e`、完整 train/validation、50 epochs、batch 128 和 seed `20260811 / 20260812 / 20260813`，启动时工作树干净。最佳 epoch 为 `13 / 46 / 7`，全部与训练历史的最低 Local RR epoch 一致。三个 checkpoint tensor 与训练历史全部 finite，每个 run 均输出 2675 个 validation sample；prediction-valid fraction 为 `1.0`、prediction-degenerate fraction 为 `0.0`，每个 seed 只有 `1/2675` 个负 signed PCC 样本。第 2 个 seed 的最佳 epoch 为 46 不触发追加预算：T3 已在节律、PCC 和 IBI coverage 上明显落后，事后延长只会形成结果驱动搜索。

按 sample direct mean，再对三个 seed 报告算术 mean ± sample SD：

| Validation 指标 | B0 PatchMixer | T2 wide native | T3 concat-deep | T3 相对 T2 |
|---|---:|---:|---:|---|
| Whole-window RR MAE（bpm）↓ | `0.512706 ± 0.002988` | `0.512559 ± 0.002194` | `0.597246 ± 0.006841` | 退化 `+0.084688`；0/3 seed 改善 |
| Local RR MAE（bpm）↓ | `0.626951 ± 0.000812` | `0.624996 ± 0.001908` | `0.681710 ± 0.013891` | 退化 `+0.056715`；0/3 seed 改善 |
| Envelope trajectory MAE ↓ | `0.155620 ± 0.000653` | `0.153601 ± 0.001251` | `0.151420 ± 0.002287` | 改善 `-0.002181`；2/3 seed 改善 |
| Global envelope modulation error ↓ | `0.244096 ± 0.005189` | `0.231247 ± 0.005794` | `0.217393 ± 0.008465` | 改善 `-0.013854`；3/3 seed 改善 |
| Lag-aware signed PCC ↑ | `0.839730 ± 0.000659` | `0.840499 ± 0.000245` | `0.804695 ± 0.006370` | 退化 `-0.035804`；0/3 seed 改善 |
| IBI-MedAE（s）↓ | `0.084295 ± 0.001170` | `0.087541 ± 0.002440` | `0.086780 ± 0.000661` | 相对 T2 改善 `-0.000761`，但仍比 B0 差 `+0.002485` |
| IBI coverage ↑ | `0.845532 ± 0.001099` | `0.845130 ± 0.001002` | `0.809956 ± 0.004072` | 退化 `-0.035173`；0/3 seed 改善 |
| Low target-modulation Spearman ↑ | `0.291283 ± 0.005623` | `0.274319 ± 0.004658` | `0.321271 ± 0.022939` | 改善 `+0.046951`；3/3 seed 改善 |
| Medium target-modulation Spearman ↑ | `0.571272 ± 0.002540` | `0.583196 ± 0.002222` | `0.581626 ± 0.005124` | 基本持平 `-0.001570` |
| High target-modulation Spearman ↑ | `0.678806 ± 0.001620` | `0.709681 ± 0.004435` | `0.723746 ± 0.009212` | 改善 `+0.014065`；3/3 seed 改善 |
| IBI interpretable fraction ↑ | `0.727103 ± 0.004112` | `0.728224 ± 0.002617` | `0.655078 ± 0.014133` | 退化 `-0.073146`；0/3 seed 改善 |

T3 验证了旧 concat-deep 结构的真实归纳偏置：它在三 seed 上一致降低 global envelope modulation error，并提高 Low/High 分层的包络排序；但这种收益来自一个显著改变输出解码路径的完整模型，同时伴随 Whole/Local RR、signed PCC、IBI coverage 与 interpretable fraction 的一致大幅退化。当前不构造加权总分让包络收益覆盖其他任务轴，因此 T3 不进入 active candidate，也不进入 designated test。

T3 的结果支持继续运行冻结的 T4：T4 保留 T2 的原生 PatchMixer decoder 和 pre-mixer 注入，只把 fullband Conv2D 表示替换为既有五带 bandenergy，能够检验低维频带条件信息是否在不复现 T3 解码退化的前提下改善 Local RR/IBI 或包络轴。T4 配置不因 T3 结果修改。尚未执行 designated test。

## 33. T4 `G3_C bandenergy` 当前协议结果与模型阶段收口（2026-08-07）

T4 正式结果位于 `runs/tho_restart_t4_g3c_bandenergy_native`。三个 run 均使用 commit `be66fa0`、完整 train/validation、50 epochs、batch 128 和 seed `20260811 / 20260812 / 20260813`，启动时工作树干净。最佳 epoch 为 `9 / 35 / 50`，全部与训练历史的最低 Local RR epoch 一致。三个 checkpoint tensor 与训练历史全部 finite，每个 run 均输出 2675 个 validation sample；prediction-valid fraction 为 `1.0`、prediction-degenerate fraction 为 `0.0`，每个 seed 只有 `1/2675` 个负 signed PCC 样本。STFT projection L2 norm 为 `1.545 / 2.694 / 3.437`，低维频带分支已经实际参与训练。

按 sample direct mean，再对三个 seed 报告算术 mean ± sample SD：

| Validation 指标 | B0 PatchMixer | T2 wide native | T4 bandenergy native | T4 相对 T2 |
|---|---:|---:|---:|---|
| Whole-window RR MAE（bpm）↓ | `0.512706 ± 0.002988` | `0.512559 ± 0.002194` | `0.536918 ± 0.020185` | 退化 `+0.024359`；0/3 seed 改善 |
| Local RR MAE（bpm）↓ | `0.626951 ± 0.000812` | `0.624996 ± 0.001908` | `0.614648 ± 0.016329` | 改善 `-0.010348`；2/3 seed 改善；相对 B0 为 3/3 改善 |
| Envelope trajectory MAE ↓ | `0.155620 ± 0.000653` | `0.153601 ± 0.001251` | `0.149889 ± 0.002279` | 改善 `-0.003712`；3/3 seed 改善 |
| Global envelope modulation error ↓ | `0.244096 ± 0.005189` | `0.231247 ± 0.005794` | `0.222447 ± 0.013806` | 改善 `-0.008800`；2/3 seed 改善 |
| Lag-aware signed PCC ↑ | `0.839730 ± 0.000659` | `0.840499 ± 0.000245` | `0.845950 ± 0.004923` | 改善 `+0.005451`；2/3 seed 改善；相对 B0 为 3/3 改善 |
| IBI-MedAE（s）↓ | `0.084295 ± 0.001170` | `0.087541 ± 0.002440` | `0.082872 ± 0.004286` | 改善 `-0.004669`；3/3 seed 改善；均值也优于 B0 `-0.001423` |
| IBI coverage ↑ | `0.845532 ± 0.001099` | `0.845130 ± 0.001002` | `0.842107 ± 0.003661` | 退化 `-0.003023`；1/3 seed 改善 |
| Low target-modulation Spearman ↑ | `0.291283 ± 0.005623` | `0.274319 ± 0.004658` | `0.333122 ± 0.016895` | 改善 `+0.058802`；3/3 seed 改善 |
| Medium target-modulation Spearman ↑ | `0.571272 ± 0.002540` | `0.583196 ± 0.002222` | `0.586800 ± 0.002208` | 改善 `+0.003604`；2/3 seed 改善 |
| High target-modulation Spearman ↑ | `0.678806 ± 0.001620` | `0.709681 ± 0.004435` | `0.726429 ± 0.018738` | 改善 `+0.016748`；2/3 seed 改善 |
| IBI interpretable fraction ↑ | `0.727103 ± 0.004112` | `0.728224 ± 0.002617` | `0.724112 ± 0.006388` | 略退 `-0.004112` |
| Lag 边界命中比例 ↓ | `0.181308 ± 0.000374` | `0.180312 ± 0.009758` | `0.171340 ± 0.009052` | 改善 `-0.008972`；3/3 seed 改善 |

T4 没有复现 T3 的强解码器退化。它相对 T2 在 Local RR、两项包络主指标、signed PCC、IBI-MedAE 和三层包络 Spearman 的多数轴上改善，且 `12752` 参数少于 T2 的 `14192`；但 Whole RR 三 seed 一致退化，IBI coverage 与 interpretable fraction 略降，跨 seed SD 和最佳 epoch 波动明显大于 T2。seed `20260813` 的最佳 epoch 位于 50 不触发 80 epoch 扩展：预算已冻结，当前证据足以确认其任务交换，追加预算会成为结果驱动搜索。

模型层面不构造加权总分。T2 和 T4 均不被另一者支配：

- **T2 wide native**：Whole/Local RR 和 coverage 更均衡，跨 seed 更稳定；包络、PCC 和 IBI 不如 T4。
- **T4 bandenergy native**：Local RR、包络轨迹/范围、PCC 和 IBI 更强，结构更小且频带解释更直接；Whole RR、coverage 和跨 seed 稳定性较弱。
- **T1 / T3**：分别因缺乏净收益和显著节律/coverage 退化退出 active candidate；M1 继续只作纯时域任务交换留档。

本轮不继续窗长、频带、融合位置、bandenergy 边界、通道数、训练预算或新结构搜索。下一步只需冻结 designated test 模型集合。建议将 T2 作为均衡主候选、T4 作为预注册的 Pareto 辅候选；若二者都进入 designated test，必须同时报告且不得根据 test 结果再选“赢家”或返工协议。Designated test 尚未执行。

## 34. 第一阶段 research-test 评价计划（2026-08-07）

本节依据第 7.3 节的新角色定义，取代第 33 节“只运行一次 designated test”的后续安排；第 33 节及更早文字保留为当时的历史决策记录，不再作为当前 test 使用限制。

本阶段固定评价集合为：B0、T2、T4 各三个 validation-selected `checkpoint_best_local_rr.pt`，以及不训练、无 seed 的 F0 和 IEWT。选择发生在读取本阶段 research-test 之前：B0 是协议基线，T2/T4 是 validation 上互不支配的两个 active candidate，F0/IEWT 是输入和算法不同的传统参照；T1/T3 已由 validation 退出，不补做 test。九个学习模型 checkpoint 不因 research-test 重选 epoch，也不事后删 seed。

输出约定：

- 每个学习模型 run 写入 `research_test_metrics.csv`、`research_test_metrics_summary.csv` 和 `research_test_metrics_manifest.json`，不覆盖 validation 产物。
- F0/IEWT 分别写入新的 `runs/tho_fixed_band_research_test/<timestamp>` 与 `runs/tho_iewt_research_test/<timestamp>`。
- 完整评价使用固定 2310 个 admitted test 窗口，包含五项 primary、IBI-MedAE + coverage、三层 envelope Spearman、coherence 和 nDTW。
- 阶段汇总仍沿用逐 sample direct mean；B0/T2/T4 再对三个 seed 报告 mean ± sample SD，F0/IEWT 报告单次确定性结果，不混合成加权总分。
- 完成后在本节追加代码 commit、运行时间、所有结果路径与阶段结论；若结果影响下一项研究，在新任务中明确标记为 `research-test informed`。

评价已于 2026-08-07 完成。全部入口运行在 commit `c9f4cb9` 且 `git_dirty=false`；11 份逐 sample 结果均为完整 2310 行，`split=test`、`joint_prediction_degenerate=0`，数值列无 Inf。学习模型结果写回各自 run 的 `research_test_metrics*.csv/json`；F0 位于 `runs/tho_fixed_band_research_test/20260807_134222_131480`，IEWT 位于 `runs/tho_iewt_research_test/20260807_134559_593049`。

| Research-test 指标 | B0 PatchMixer | T2 wide native | T4 bandenergy native | F0 fixed band | IEWT |
|---|---:|---:|---:|---:|---:|
| Whole-window RR MAE（bpm）↓ | `0.709086 ± 0.011954` | `0.705539 ± 0.021057` | **`0.692235 ± 0.010912`** | `1.884850` | `0.837039` |
| Local RR MAE（bpm）↓ | `0.677298 ± 0.009476` | `0.679782 ± 0.037255` | **`0.661218 ± 0.023321`** | `2.345699` | `0.930880` |
| Envelope trajectory MAE ↓ | `0.142866 ± 0.000333` | `0.141325 ± 0.000983` | **`0.141216 ± 0.000461`** | `0.164745` | `0.189647` |
| Global envelope modulation error ↓ | `0.178228 ± 0.003203` | **`0.170461 ± 0.003441`** | `0.170746 ± 0.002559` | `0.178759` | `0.216155` |
| Lag-aware signed PCC ↑ | `0.857810 ± 0.001413` | `0.860520 ± 0.000189` | **`0.868514 ± 0.005008`** | `0.703697` | `0.778774` |
| IBI-MedAE（s）↓ | **`0.095513 ± 0.001601`** | `0.098471 ± 0.003650` | `0.098189 ± 0.002619` | `0.139625` | `0.131820` |
| IBI coverage ↑ | `0.792968 ± 0.000761` | `0.789149 ± 0.007811` | **`0.803697 ± 0.008849`** | `0.350175` | `0.661448` |
| Respiratory-band coherence ↑ | `0.585303 ± 0.007407` | `0.576825 ± 0.002434` | `0.569568 ± 0.009701` | **`0.659590`** | `0.401598` |
| Constrained nDTW ↓ | `0.238343 ± 0.001788` | `0.235043 ± 0.000943` | **`0.223197 ± 0.008429`** | `0.378765` | `0.331405` |
| Low envelope Spearman ↑ | `0.406388 ± 0.006026` | **`0.408161 ± 0.006844`** | `0.400642 ± 0.012006` | `0.317435` | `0.296122` |
| Medium envelope Spearman ↑ | `0.504139 ± 0.006653` | **`0.520065 ± 0.003004`** | `0.504806 ± 0.011091` | `0.433666` | `0.408830` |
| High envelope Spearman ↑ | `0.728067 ± 0.001210` | **`0.744641 ± 0.002706`** | `0.737162 ± 0.004465` | `0.700866` | `0.668992` |
| IBI interpretable fraction ↑ | `0.604473 ± 0.004788` | `0.598124 ± 0.018490` | **`0.629293 ± 0.020362`** | `0.069264` | `0.356710` |
| Lag 边界命中比例 ↓ | **`0.164502 ± 0.004393`** | `0.174603 ± 0.004629` | `0.178788 ± 0.012487` | `0.134199` | `0.374892` |

本表粗体只标记该行数值最优，不构造总分，也不把 F0 的最高 coherence 解释为总体最好。阶段观察如下：

1. T4 在学习模型中取得最低 Whole/Local RR、最低 envelope trajectory MAE、最高 signed PCC、最高 IBI coverage 和最低 nDTW；因此 research-test 支持把 T4 作为下一项时频研究的主要结构锚点。
2. T2 的 global envelope modulation error 仅比 T4 低 `0.000285`，并在 Low/Medium/High 三个 target 调制层的 Spearman 全部最高；它仍是必要的 fullband 对照，不能因 T4 在多数轴占优而删除。
3. B0 保持最低 IBI-MedAE，并在三个学习模型中 coherence 最高；时频融合没有在所有任务轴上支配纯时域基线。B0 继续只作协议基线，不改变“纯时域不是主要研究方向”的定位。
4. F0 的 coherence 为全表最高，但其 RR、PCC、IBI coverage 和 nDTW 显著较差。这正说明 coherence 主要度量稳定频域耦合，不能单独代表重建质量；其 supplementary 角色合理，不升级为 selector。
5. IEWT 在五项 primary、IBI/coverage、coherence 与 nDTW 上均未超过三个学习模型，本阶段不投入 IEWT 参数搜索。

这些结论属于 research-test informed 阶段证据。它们不回头修改本批 checkpoint、频带、metric、seed 或表中模型集合；可以用于定义下一项新的模型研究任务。

## 35. CRD-Net v1.1 S0/S1 新模型阶段（2026-08-08）

第一阶段 research-test 已完成，因此允许以其为背景证据定义新的模型研究任务，但不得回头改写 B0/T2/T4 checkpoint 或旧阶段结论。新任务命名为 CRD-Net v1.1，当前仅激活 S0/S1：先在统一的 80-epoch AdamW/bf16/eligible-aware accumulation 协议下重训旧 B0/T4，再按 decoder bridge、local Mamba2、Direct analytic frontend、可选 1-Hz global Mamba2 的单因素顺序推进。

本阶段不改变数据、split、target、正式 `Pi`、`L_sync + 0.25 L_effort`、评价指标或 validation Local-RR checkpoint selector。S0/S1 禁止读取 research-test；其结果只作为 development/validation evidence。AM、Morphology、gate、auxiliary、capacity/TCN control 和 S2 以后阶段尚未激活，不得提前实现进正式候选或混入本阶段 run。

CRD 的完整模型、tensor、初始化、依赖、逐 optimizer-step 训练语义、配置角色、S1 停止/保留规则和验收契约冻结在 `docs/experiments/crd_v1_protocol_20260808.md`。该附件由本节及后续修订章节纳入当前唯一实验协议；若其与本文冲突，以本文为准。正式运行只能使用 `configs/crd_v1/` 下由最新协议章节明确激活的冻结配置与 `scripts/train_crd.py`，且在目标 GPU 的依赖、finite forward/backward 和当前冻结 physical batch 128 acceptance 通过之前不得启动正式三 seed 队列。

实现级验收已于 2026-08-08 完成：固定 `mamba-ssm/causal-conv1d` 版本检查通过；实际 BiMamba2 `B=2,L=1800,D=96` fast-path forward/backward finite；CRD_103 与 CRD_104 各一个 synthetic batch-1 的完整 model/core-loss/backward 均 finite；CRD_101 完成一次 4-train/2-validation CPU 生命周期 smoke，并成功独立复评。随后按预注册工程规则比较 `32×4 / 64×2 / 128×1`：`128×1` 稳态吞吐相对 `32×4` 提升 30.36%，peak reserved 为 10.868 GiB（RTX 4070 Ti SUPER 总显存的 68.17%），因此原六个配置在任何正式 run 前统一修订为 physical batch 128、accumulation 1，effective batch 与 LR/update 序列不变；修订后的 CRD_103 完整 acceptance 已通过。全量仓库测试为 `312 passed`。以上均是工程证据，不进入模型效果比较；代码提交且工作树干净后，正式 S0 队列可按顺序启动。

## 36. CRD_101 结果后 S1D 诊断修订（2026-08-08）

CRD_001/002 与 CRD_101 已在 commit `6f58f36f4839904014031970e5f69262aa6e96f8` 下完成三 formal seeds。CRD_101 相对 CRD_001 的 Local RR seed mean 改善 1.2777%，但 lag-aware signed PCC seed mean 下降 0.052458，严格触发第 35 节所纳入附件的 0.01 停止线；两个服务器故障中断 run 不进入该比较。因此原 `101→102→103→104` 队列关闭，结果不得通过事后重选 checkpoint 改写。

为区分 Local-RR selector、patch-token bridge 与共享 coarse head，现仅激活 post-result S1D：先对 CRD_001/101 各三个 final checkpoint 做配对 validation 归因复评，再运行新增的 `CRD_105 Direct-Coarse` 三 seed 诊断。CRD_105 使用 Direct analytic frontend 和与 CRD_101 相同的 refinement/head，不含任何 Mamba。其结构、192,781 参数契约、产物隔离、gate 与条件分支均冻结在附件第 15–16 节。

本修订不改变数据、split、target、loss、metrics 或 checkpoint selector，不授权 CRD test，也不自动重新开放 CRD_102。CRD_105 通过原 CRD_001 coarse gate 后，路线改为 `105→103→可选104`；若失败则 coarse 路线停止。CRD_102 只有在未来另行冻结 selector/训练修订且修订版 CRD_101 重新通过三 seed gate 后才可开放。自本节起，`configs/crd_v1/` 下七个配置中仅 001/002/101 的既有结果、105 的诊断队列以及由附件条件开放的 103/104 有效；正式运行仍只使用 `scripts/train_crd.py`，且禁止读取 research-test。

D0 paired final-checkpoint 复评随后在 commit `baeb7c5` 下完成。CRD_001/101 的 fixed-final signed PCC seed mean 分别为 0.841586/0.788269，CRD_101 仍下降 0.053317，且三个配对 seed 全部下降；selector 不能解释原 PCC 失败。D0 不改变正式 checkpoint 或原 gate，现允许按附件第 16.2 节进入 CRD_105 synthetic/acceptance 与 formal diagnostic。

CRD_105 的 CUDA synthetic finite 检查与独立 physical-batch-128 acceptance 已在 commit `be21ba0` 下通过；acceptance 严格使用 128/32 个 train/validation windows、一次 optimizer update，完整生命周期与所有 checkpoint tensors finite。该工程证据不参与模型比较，现解除 CRD_105 三 formal seeds 的工程阻塞。

CRD_105 三 formal seeds 随后在 commit `2bee3e5` 下完成并通过完整性审计。相对 CRD_001，Local RR seed mean 改善 8.1006%，signed PCC 增加 0.006043，三个配对 seed 两项均同方向改善，故通过 Direct-Coarse gate。相对 CRD_101，CRD_105 的 signed PCC 增加 0.058501，支持把原退化定位到 frontend package 而非共享 coarse head。按附件冻结分支，现跳过仍关闭的 CRD_102，开放 `CRD_103 vs CRD_105`；CRD_104 继续等待 103 gate。

CRD_103 三 formal seeds 随后在 commit `ed9d68e` 下完成并通过完整性审计。相对 CRD_105，它的 Local RR seed mean 恶化 0.6905%、0/3 配对 seed 改善、trajectory MAE 恶化 5.2364%，虽然 signed PCC 增加 0.006461，但仍同时违反三项必要保留条件。因此 CRD_103 不保留、CRD_104 不运行，S1D 当前保留 CRD_105；Mamba 带来的 PCC/global-envelope/IBI 收益仅记录为后续独立研究背景，不改变本轮停止决策。

## 37. CRD S1E 结果后探索性补全（2026-08-09）

在上述 S1D 决策与 CRD_105 保留状态均冻结后，为获得完整结构响应信息，允许额外运行原已停止的 CRD_102 与 CRD_104。该批命名为 S1E，protocol 固定为 `crd-v1.1-s1e-20260809`，证据属性为 post-result exploratory completion；`run_role=formal` 只表示使用完整数据、80 epochs 与三个固定 seed，不把它升级为预注册模型选择证据。

S1E 只描述 `102 vs 101` 的 Local Mamba 补偿效应与 `104 vs 103` 的 Global Mamba 边际效应，并将二者与 CRD_001/105 做全指标背景比较。原 0.5%/2-of-3/PCC/trajectory 条件只作描述性参照，不重新选择模型；CRD_105 的当前保留状态不因 S1E 自动改变。S1E 不修改数据、split、target、loss、metrics、selector 或 seed，禁止读取 research-test。完整比较口径、工程门槛、产物身份与未来证据边界由附件第 17 节冻结。

CRD_102/104 的 CUDA synthetic 与独立 physical-batch-128 acceptance 已在 commit `d60b050` 的干净工作树下通过。两者均以 128 个 train windows 形成一次 update，并完成 32 条 validation metrics 与完整 checkpoint 生命周期；所有 tensors/metrics finite、无 prediction degeneracy。该结果仅解除六个 S1E 完整 run 的工程阻塞，不形成效果证据。

六个 S1E full-budget run 随后在 commit `f8fa658` 下完成并通过完整性审计。102 相对 101 强烈改善 RR、PCC、global envelope 与 IBI，但 trajectory 恶化 3.4048%；104 相对 103 描述性满足原四项条件，显示 Global Mamba + FiLM 对失败的 103 存在补偿。相对已冻结的 105，102 的 Local RR 改善 4.3138%、3/3 paired 改善、PCC 增加 0.018342且 trajectory 仅恶化 0.2882%，形成强探索性候选；104 同样改善 RR/PCC/global envelope，但 trajectory 恶化 3.0375%。这证明原顺序 gate 会漏掉非单调模块交互，但不允许在观察后回改 S1D 结论：105 仍是本轮正式保留结果，102/104 只进入下一版确认协议的候选背景，不授权 CRD research-test。

## 38. CRD candidate lock 与确认前规则冻结（2026-08-09）

S1E 结束后，CRD_102/104/105 各三个 Local-RR-selected checkpoints 已作为候选集合锁定，CRD_001 三 checkpoint 作为只读 reference。精确路径、seed/epoch、训练 commit/protocol、文件大小及 checkpoint/config/manifest/validation-summary SHA-256 固定在 `docs/experiments/crd_v1_candidate_lock_20260809.json`，不得替换为其他 timestamp、final checkpoint 或重训结果。

未来选择采用“相对 CRD_105 的四项资格门槛 + 五项 primary Pareto”规则：Local RR mean 改善至少 0.5%、至少 2/3 paired seed 改善、PCC 下降不超过 0.005、trajectory 恶化不超过 1.5%，全部通过后才进入 Whole/Local/trajectory/global-envelope/PCC 的非支配比较。无唯一非支配候选时保留 Pareto set，不构造加权总分；IBI、三层 Spearman 与 lag-boundary 仅作 secondary。当前只冻结候选和规则，尚未建立独立确认阶段，也不授权读取 CRD research-test；是否建立确认阶段及其数据口径必须在未来另行修订。

## 39. CRD S1C 独立测试集确认阶段（2026-08-09）

现决定使用独立测试集建立S1C确认阶段，协议标识为`crd-v1.1-s1c-research-20260809`。第38节“尚未激活”的状态至此结束，但其candidate lock和选择规则不变；独立测试集不参与本阶段checkpoint、candidate或超参数选择。

S1C 只评价 candidate lock `9a14db8be8af22e1ce1c5a332b4912ab5c13c7fb03cdf1894fc5c6ed6ff7f8cc` 中的 CRD_102/104/105 九个候选 checkpoint 与 CRD_001 三个 reference checkpoint，按 lock 顺序、使用完整 2310-window/8-subject research-test 各评价一次。索引级 split 审计确认 train/validation/test 为 `10141/2675/2310` windows、`32/7/8` 个 `samp_id`，三个 split pair 的 subject 与 segment overlap 均为 0。专用入口为 `scripts/eval_crd_s1c.py --confirm-research-test`；普通 `eval_crd.py` 仍不开放 test，固定隔离输出不得覆盖。

必须等 12 项全部完成并审计后，才由预先实现的 `scripts/summarize_crd_s1c.py` 执行第 38 节冻结的资格门槛与五项 primary Pareto；不得依据部分 test 结果停止、替换 checkpoint、重训、删 seed、事后加权或让 secondary 指标推翻 primary 规则。完整数据、指标、失败、access receipt 与产物契约见附件第 19 节。

S1C 随后在干净 commit `3b280013d898287613709c4dd5648f8b94e14c9d` 下完成。12 个 manifest 与 12 份逐 sample metrics 全部齐备，每份恰含 2310 行，总计 27720 行；protocol、candidate-lock hash、checkpoint 身份与 split 均一致，所有 manifest 均为 `git_dirty=false`，数值列无 Inf。冻结汇总产物位于 `runs/crd_v1/crd_s1c_research_confirmation/`。

相对 CRD_105，CRD_102 的 test Local RR 改善 `6.8065%`、三个配对 seed 全部改善，signed PCC 增加 `0.012860`，trajectory 不仅未恶化反而改善 `1.1863%`，故四项资格门槛全部通过；其 Whole RR 与 global-envelope error 还分别改善 `6.0737% / 11.0571%`，因此在五项 primary 上严格 Pareto-dominate CRD_105。CRD_104 的 Local RR 仅改善 `0.2940%`，trajectory 恶化 `6.3707%`，同时未过 `0.5% / 1.5%` 两个门槛，不能进入 Pareto。冻结结果为 `eligible={102,105}`、唯一 Pareto 候选 `CRD_102`。

该结果不回写S1D当时“保留105”的历史结论，但在新S1C协议下将CRD_102更新为后续阶段的当前结构锚点。它也不表示CRD_102在所有模型和指标上全面最优：只读reference CRD_001的Whole RR、IBI-MedAE与coherence仍更好；102相对001的五项primary中Whole RR恶化`1.8155%`，其余Local RR、trajectory、global-envelope、PCC分别改善`0.4404% / 2.1736% / 2.1565% / +0.020974`。这些结果属于独立测试集证据。S1C队列至此关闭，不重复评价；S2、AM/Morphology/gate/auxiliary/control仍未激活。

## 40. CRD S1F global-stage 缺失格（2026-08-09）

S1C选中的CRD_102使用B0/PatchMixer frontend + local Mamba，而现有global-stage单因素比较`104 vs 103`只覆盖Direct frontend。由于本阶段已出现明显的frontend/Mamba非单调交互，在进入机制启发表征S2前，仅允许新增一个由独立测试集结果启发的development variant：`CRD_106_B0_HIER_MAMBA = CRD_102 + CRD_104 的同构 1-Hz global Mamba/FiLM`，补齐frontend × global-stage的缺失格。

S1F 不读取 research-test，不修改数据、loss、metrics、selector、训练预算或 seed；正式比较只用 validation。106 相对 102 必须同时达到 Local RR mean 改善 `≥0.5%`、`≥2/3` paired seeds 改善、PCC 下降 `≤0.005`、trajectory 恶化 `≤1.5%`。通过则未来 `S2 BASE=106`，否则 `S2 BASE=102`；无论结果如何均不再增加 S1F variant。完整结构、初始化、参数契约、工程门槛和输出边界见附件第 20 节。S2、AM/Morphology/gate/auxiliary/control 继续关闭。

CRD_106 的 CUDA synthetic 与独立 physical-batch-128 acceptance 随后在干净 commit `8fa56f8` 下通过。Acceptance 严格完成 128/32 个 train/validation windows、1 optimizer update、两个 finite checkpoint 与 32 条 finite validation metrics，无 prediction degeneracy；该单 epoch 数值只作工程证据。现解除 106 三 formal seeds 的运行阻塞，仍不得读取 research-test。

CRD_106 三 formal seeds 随后在干净 commit `80e6350` 下完成并通过完整性审计；Local-RR-selected epochs 为 `40/5/26`，每个 run 均完成 80 epochs、6400 updates、2675 条 validation metrics，checkpoint/history/metrics 全 finite。相对冻结 CRD_102，106 的 Local RR 改善 `2.2123%` 且 3/3 paired seeds 改善，PCC 下降 `0.003178` 仍在护栏内，但 trajectory 恶化 `5.0351%`，明显超过 `1.5%`。因此四项条件未全部通过，106 不保留，S1F 关闭并固定未来 `S2 BASE=CRD_102`；Whole RR 改善 `8.5536%` 等信号只记录为任务交换，不能推翻停止规则。S2 仍需另立协议后才可激活。

## 41. CRD S2 表征分支阶段（2026-08-09）

现以 candidate lock 中冻结的 CRD_102 三 checkpoint 作为只读 `CRD_201_BASE`，不重训 BASE。S2 明确放弃把 Direct frontend 当作基础分支：新增 legacy energy、analytic AM、amplitude-normalized morphology 都在 CRD_102 PatchTokenFrontend 输出后、local Mamba 前以 zero-init static residual 注入。当前只激活 S2A 的 `CRD_202_BASE_LEGACY_ENERGY / CRD_203_BASE_ANALYTIC_AM / CRD_204_BASE_MORPHOLOGY` 实现与独立工程验收；在代码、参数数量、shared-state/zero-init、CUDA finite 与 physical-batch-128 acceptance 通过前不得启动 formal runs。

Energy 以 trajectory MAE 为模块 primary，并用 Local RR/PCC/global-envelope 护栏选择 `X∈{E,A,none}`；Morphology 以 signed PCC 为 primary，并用 Local RR/coverage/trajectory 护栏。只有 X 与 M 都 eligible 才开放一个对应静态组合和一个 deterministic parameter-matched control；组合通过 Local RR/容量/护栏后才允许未来 gate stage。S2 全程只读 train/validation，不计算确认性 p-value，不访问 research-test。唯一结构、token chunking、loss、决策表和条件分支见附件第 21 节；S2B、gate、auxiliary 和最终消融仍未激活。

S2A 三分支实现与 CPU 协议回归现已完成：配置为 `crd_202_base_legacy_energy / crd_203_base_analytic_am / crd_204_base_morphology`，trainable parameter 数为 `1,071,449 / 1,197,785 / 1,106,857`。实现未改变数据、split、core loss、metrics 或 Local-RR checkpoint selector；204 仅加入本节预注册的 prototype regularizer、逐 epoch 记录和 checkpoint provenance。CUDA synthetic 与三个独立 physical-batch-128 acceptance 尚未完成，故 formal runs 继续阻塞。

首轮 CUDA 验收中，202/203 已通过 physical batch 128，peak reserved fraction 为 `62.18%/74.25%`；204 synthetic 通过但 batch-128 训练真实 OOM。为避免改变预注册的 physical/effective batch 与优化口径，现于任何 formal run 前登记 204 morphology 每 128-token chunk 的非重入 activation checkpoint：该 encoder 没有 dropout 或 batch-dependent normalization，只在 backward 重算相同前向，不改变模型参数、loss 或输出定义。只重跑 204 工程验收；若仍 OOM 或 peak reserved fraction 超过 `80%`，不得静默减小 batch，须停止并另行决定是否关闭 204 或修订整个 S2A 执行协议。

204 重验已在干净 commit `a149913` 下完成，peak allocated/reserved 为 `8,636.81/10,406 MiB`，reserved fraction `65.30%`；完整 checkpoint lifecycle、32 条 primary-finite validation metrics和 prediction-degeneracy 检查通过。单 update acceptance 的 IBI-MedAE 因全部样本不满足 interpretable eligibility 而为空，此状态由 flag/count 显式保留，不改变工程通过结论。结合 202/203 的 `62.18%/74.25%` 峰值与完整性审计，S2A 三分支现均允许在统一的新干净 commit 上运行三个固定 seed；S2B 与 research-test 继续关闭。

九个 S2A formal runs 已在统一干净 commit `41ed41d` 下完成并通过 80 epochs/6400 updates、checkpoint、2675-window validation、finite 与显存审计。冻结规则给出 E trajectory 改善 `-1.6232%`/paired `0/3`，A trajectory 改善 `-6.9786%`/paired `0/3` 且 PCC drop `0.007549`，所以 `X=none`；M 的 PCC increase `-0.006737`/paired `0/3`、Local RR 恶化 `2.6374%`、coverage drop `0.010307`，所以 M 不 eligible。S2B/S3 关闭并保留 CRD_102。当前只补齐不参与选择的 204 validation prototype usage/entropy 与 samp 分布，再固化 S2A summary；不得据此重开 eligibility 或读取 research-test。

204 三 seed validation prototype 描述与冻结 S2A summary 已在干净 commit `3c3598c` 下完成。Hard-usage entropy 为 `0.5131/0.5731/0.6254`，soft-usage entropy 为 `0.9021/0.8809/0.9540`，全局 dominant hard fraction 最高 `47.50%`、逐 samp 最高 `68.97%`，没有全局单 prototype 坍缩；但该结构没有带来预注册的 PCC/Local-RR/coverage 收益。最终不可覆盖 decision 为 `X=none / M ineligible / S2B=false / S3=false / retain CRD_102`，产物固定在 `runs/crd_v1/crd_s2a_validation_summary/`。S2A 至此关闭，prototype 与 summary 入口不再重复运行。

## 42. CRD S2B-R 结果知情交互补救（2026-08-10）

研究者在获知 S2A 三个单因素均失败后，明确要求继续检验多因素非线性补偿。现新增 result-informed exploratory S2B-R，而不伪装成第 41 节条件自然触发：同时实现 `CRD_205 BASE+E+M / CRD_206 BASE+A+M` 与各自确定性参数匹配 control `CRD_207/208`。四项初始化均退化为 CRD_102，沿用数据、core loss、metrics、Local-RR selector、80 epochs、physical batch 128 和三个 seed；只读 train/validation，不访问 research-test。

每个组合必须同时优于 BASE、自己的 capacity control 与冻结的最佳 constituent，并守住 PCC/trajectory/coverage；另报告 `combo−energy−morphology+BASE` factorial interaction descriptives。完整结构、参数匹配、显存策略、门槛和停止规则见附件第 22 节。当前只激活实现、测试与四项独立工程验收；S3 gate 仍未定义或实现。

205/206/207/208 的 CUDA synthetic 与独立 physical-batch-128 acceptance 已在统一干净 commit `0e541f8` 下通过；peak reserved fraction 分别为 `67.37%/67.15%/65.41%/71.46%`，完整 checkpoint lifecycle、eligible primary finite 与 prediction-degeneracy 审计通过。现允许四项各三个 formal seeds；仍不得读取 research-test、改动 batch/预算或实现 S3。

12 个 S2B-R formal runs 已在统一干净 commit `c2bcfb0` 下完成并通过 80 epochs/6400 updates、checkpoint、2675-window validation、finite、degeneracy 与显存审计。205 相对 BASE/control/best constituent 的 Local RR 改善为 `-1.0387%/-0.9260%/-2.5769%`，paired seeds 为 `1/3、1/3、0/3`，且 PCC/coverage 护栏失败；206 的三项 Local RR 改善为 `-1.0295%/-1.4515%/-1.0295%`，paired seeds 均为 `1/3`，且 PCC/trajectory 护栏失败。因此两个组合均不 eligible。Factorial contrast 在 seed mean 上显示 205/206 的 Local RR interaction 为 `-0.000552/-0.010117`、trajectory 为 `-0.003398/-0.003762`、PCC 为 `+0.001168/+0.002339`，说明非线性补偿方向存在但不足以转化为优于 BASE/容量对照的绝对收益。冻结产物位于 `runs/crd_v1/crd_s2br_validation_summary/`，最终保留 CRD_102、S3 不激活，S2B-R 关闭且未读取 research-test。

## 43. CRD_102 validation 失败诊断（2026-08-11）

S2B-R 关闭后不立即增加结构，而先对 candidate-lock 中 CRD_102 三个冻结 validation checkpoint 做一次结果知情但不参与选择的探索性诊断。逐 seed 使用 eligible-window worst decile，并以至少 `2/3` seeds 命中定义 persistent failure；固定分层为 samp、coupling state、target modulation stratum、IBI interpretable 一致性和 lag-boundary 一致性，同时报告跨 seed agreement、error-aligned metric associations 与失败签名。完整边界与输出见附件第 23 节。该入口不重推理、不重选 checkpoint、不读取 research-test，签名不作因果解释；当前只允许在干净 commit 上生成一次性冻结产物。

若第一层观察到 target-modulation 关联和相邻 row 聚集，再按附件第 23 节冻结字段启动一次结果知情 metadata follow-up：只连接既有 validation consensus 与冻结 dataset index，检查质量/confidence 元数据和重叠窗口 failure episodes，不新增模型或选择门槛。该层与第一层分目录保存且同样禁止覆盖。

两层诊断已分别从干净 commit `81fab55/7a1b29b` 生成并通过 checkpoint/metrics/index/hash 与 validation-only 审计。High target-modulation 占 Local-RR persistent failures 的 `185/262=70.61%`、multimetric-core failures 的 `323/352=91.76%`；Local-RR 三 seed 排名 Spearman/Jaccard 为 `0.9668/0.7710`，说明难例高度可重复。Waveform/rate confidence 与 Local-RR 的 Spearman 为 `-0.5487/-0.5465`，motion ratio 为 `+0.4811`，但这些关联受 modulation/samp composition 混杂，不作因果解释。Local-RR 与 multimetric failure windows 中 `84.73%/80.40%` 位于多窗口连续 episodes，最长 `420/540 s`。诊断因此把下一研究问题收敛为 high-modulation 连续片段的可观测性/局部跟踪，以及独立的 low-modulation rank-metric 适用性；不自动启动 S3 或新结构。

## 44. CRD_102 high-modulation matched observability（2026-08-11）

现先执行 high-modulation 连续 failure episode 与同 samp/状态/相近 target-modulation、waveform-confidence、motion 的成功窗口配对。使用 rawish direct/fixed-band 两种输入 proxy 复用冻结任务指标，并补充 band coherence/dominant-frequency error；不运行模型、不训练、不读 research-test。完整 case/control、Hungarian/caliper、primary/sensitivity 和停止规则见附件第 24 节。当前只允许实现、测试并从干净 commit 生成一次性产物，不据结果自动启动新结构。

唯一产物从干净 commit `c951325` 生成：21 个 exact-state primary pairs 覆盖 3 个 samp，28 个 same-samp sensitivity pairs 覆盖 5 个 samp。Primary 中 rawish proxy 的 Local-RR/PCC case-worse 为 `16/21、13/21`，fixed-band 为 `17/21、16/21`，只有 fixed-band 同时通过 `2/3`，冻结 outcome 为 `mixed_observability_and_model_tracking`。两个 proxy 的 Local-RR delta 与 CRD delta 中度相关，但 CRD PCC delta 与 proxy 仅约 `0.21`；且 sensitivity 中两个 proxy 都未同时通过门槛。结果支持输入可观测性与模型特异跟踪两个亚型共存，不支持单一全局结构或数据过滤方案。Matched observability 至此关闭；若继续，仅可另立 inference-only waveform decomposition 协议。

## 45. CRD_102 锚点控制线 C0/C1/C2（2026-08-11）

S2B-R 与后续 CRD_102 failure/matched-observability diagnostics 关闭后，不续接已关闭且未定义的 S3/S4/S5。现以 candidate lock 中冻结的 CRD_102 三个 Local-RR-selected checkpoint 为只读锚点，另立 `crd-v1.1-controls-research-informed-20260811` 控制线，依次回答：当前 10-Hz/Fourier decoder 是否在正式呼吸带内近似无损、Local BiMamba2 是否能被参数匹配 full-context TCN 替代、以及 learned 100-Hz nonlinear decoder 是否优于同参数的 10-Hz decoder-capacity control。完整数学、结构、门槛、产物和停止规则冻结在 `docs/experiments/crd_v1_controls_protocol_20260811.md`；该附件由本节纳入当前唯一实验协议，冲突时以本文为准。

控制线属于既有 research-test 与多轮 validation 结果知情后的 development/validation controls，不形成确认性统计推断。数据、split、target、正式 `Pi`、core loss、metrics、Local-RR selector、三个 seed 和 80×128×1 训练语义保持不变；S1C research-test 队列继续关闭。C1/C2 即使分别通过，也不得在本协议内自动组合。

当前只激活 C0：对完整 2675-window/7-`samp_id` validation target 执行 `Pi(target) → ::10 → Fourier 1800→18000 → Pi`，生成不可覆盖的逐 sample metrics、数值审计、decision 和 manifest；不运行模型、不读取 checkpoint tensor、不训练。C0 正式结果必须在干净 commit 下完成并由本节登记后，才可激活 C1 的实现。C1/C2 目前只冻结未来定义，禁止提前实现、验收或运行。

C0 随后在干净 commit `5de0c0f459c46e6021034d63bf3d4efbd8a39ac0` 下完成。2675 条逐 sample primary 全部 finite、无 prediction degeneracy；全局最大绝对误差/RMSE 为 `5.538454e-7 / 8.217932e-8`，首末 15 秒 RMSE 均小于 `9e-8`，Whole/Local RR、trajectory、global-envelope 误差均约 `1e-8`，signed PCC 为 `0.9999999999994397`，全部预注册门槛通过并冻结为 `roundtrip_negligible=true`。该结果不支持把 learned 100-Hz decoder 表述为恢复正式呼吸带的采样损失；未来 C2 只能检验 nonlinear capacity/placement。C0 入口关闭且不得重复执行，现开放 C1 parameter-matched full-context TCN 的实现、测试、CUDA synthetic 与独立 physical-batch-128 acceptance；在工程结果由本节登记前，C1 formal 三 seed 与全部 C2 代码继续关闭。

C1 唯一实现随后完成：`crd_c101_b0_local_tcn` 只将 CRD_102 的六个 Local BiMamba2 替换为十个 `C=96/H=488`、dilation `1…512` 的 residual TCN blocks；感受野为 4093 tokens，总参数 `1,062,001`，相对 CRD_102 少 `0.6310%`。同 seed frontend/refinement/head state、TCN zero-init identity、参数/感受野与配置身份测试通过，CPU synthetic batch-1 的 model/core-loss/backward finite。该结果仍不解除 formal 队列；下一步只允许在当前实现提交后的同一干净 commit 执行 CUDA bf16 synthetic 与独立 physical-batch-128 acceptance，C2 继续关闭。

C1 CUDA synthetic 与独立 physical-batch-128 acceptance 随后在干净 commit `529de747cfee17675432fad1a969570703df1791` 下通过。Synthetic batch-1 的 output/input/全部 parameter gradients finite，peak allocated `92.88 MiB`。Acceptance 严格完成 128/32 个 train/validation windows、1 optimizer update；best/final checkpoint 的 108 个 model tensors 与 324 个 optimizer tensors全部 finite，32 条 validation primary 全部 finite、无 prediction degeneracy，peak allocated/reserved 为 `10018.08/10664.00 MiB`，reserved fraction `66.9161%`。单 epoch数值不形成效果证据。现解除 C1 三 formal seeds 工程阻塞；正式 run 必须来自包含本登记的统一新干净 commit。C2 仍关闭，等待 C1 三 seed 冻结 decision。

C1 三 formal runs 随后在统一干净 commit `930212ac66cb95697b659fc7504259cfa3cf71c2` 下完成。每个 run 均为 80 epochs/6400 updates、2675 条 validation metrics，selected epochs 为 `25/26/12`；checkpoint/optimizer、primary finite、prediction-degeneracy 与显存初审通过。现只激活 `scripts/summarize_crd_c1.py` 的一次性冻结汇总，在完整 lifecycle 重审后应用附件第 4.2 节 quality-superior/near 规则并生成 paired/failure-strata 描述。固定输出不得覆盖；summary decision 登记前 C2 继续关闭。

C1 冻结 summary 随后从干净 commit `f0ac01b` 生成。TCN 相对 CRD_102 的 Local RR 改善 `3.8241%` 且 `3/3` paired seeds 改善，PCC drop `0.003797` 在护栏内，但 trajectory worsening `2.5468%` 超过 `1.5%`，因此 quality-superior 与 quality-near 均失败，decision 固定为 `mamba_retained_control_failure / retain CRD_102`。TCN 的 Whole RR 改善 `9.7621%`，failure-strata 中 Local RR 收益集中于既有 persistent/high-modulation/matched-case 难例，但非 persistent windows 略恶化且 high-modulation trajectory 明显退化；这些只解释任务交换，不能覆盖总体 gate。冻结产物位于 `runs/crd_v1/crd_c1_validation_summary/`，C1 关闭且 summary 不得重复运行。现按附件第 5 节开放 C2 的两个 decoder candidates 实现、测试、CUDA synthetic 与独立 batch-128 acceptance；C2 formal、TCN+decoder 与 research-test 继续关闭。

C2 唯一实现随后完成：`crd_c201_decoder_10hz_cap / crd_c202_decoder_100hz` 均保留 CRD_102 的完整 Mamba trunk 与 coarse head，只增加同一个 1,057-parameter zero-init pointwise nonlinear residual，分别在 10 Hz feature 上作用后 Fourier 上采样，或先将 32-channel feature Fourier 上采样至 100 Hz 后作用。两个候选总参数均为 `1,069,802`，共享模块和 residual 初始 state 逐 tensor相同，初始化 waveform 与 CRD_102 逐点相同；旧 variant forward/state 未改变。CPU 结构与全仓回归通过；官方 Mamba fast path 按依赖契约不支持 CPU synthetic，故下一步只允许在当前实现提交后的干净 commit 分别执行 CUDA synthetic 与独立 physical-batch-128 acceptance。C2 formal、TCN+decoder 与 research-test 继续关闭。

C2 两项 CUDA synthetic 与独立 physical-batch-128 acceptance 随后在统一干净 commit `0e2d05824a50426e3ff0443a6875829422a6160d` 下通过。C201/C202 synthetic output/input/全部 parameter gradients finite，peak allocated 为 `349.17/349.21 MiB`。两项 acceptance 均完成 128/32 个 train/validation windows、1 optimizer update，checkpoint/optimizer 与 32 条 primary metrics 全 finite、无 prediction degeneracy；peak reserved fraction 为 `62.3479%/57.6794%`。单 epoch数值不形成效果证据。现解除 C201/C202 各三个 formal seeds 的工程阻塞；正式 run 必须来自包含本登记的统一新干净 commit。TCN+decoder、其他 decoder 变体和 research-test 继续关闭。

C2 六个 formal runs 随后在统一干净 commit `4ec737164e20461ab8b3f3595cb1813a38ff1ddd` 下完成。每项均为 80 epochs/6400 updates、2675 条 validation metrics；C201/C202 selected epochs 均为 `10/11/13`，checkpoint/optimizer、primary finite、prediction-degeneracy 与显存初审通过。冻结 summary 前只激活 `scripts/eval_crd_c202_residual_spectrum.py`：对三个 C202 selected checkpoints 各完整读取一次 validation，按附件第 5.3 节冻结口径报告 `Pi` 前 residual 的带内/带外能量比例。该描述不参与 gate、不读取 research-test且固定输出不可覆盖；三项齐备前不生成 C2 selection 或新增 decoder。

C202 三项 residual diagnostics 随后从统一干净 commit `3ef5cf0c18d000159aad18a5c751be680f275d8d` 生成。每项覆盖 2675 windows/7 `samp_id`，checkpoint hash/epoch、finite 与 research-test=false 审计通过；三个 seed 的带外能量比例 mean 为 `3.1199%/18.7155%/10.6439%`，只作 `Pi` 前数值解释。现只激活 `scripts/summarize_crd_c2.py` 的一次性冻结汇总，重审六个 runs/三项 diagnostics 并应用两项 basic gate 与 C202-vs-C201 placement gate；固定输出不得覆盖。Summary decision 登记前不新增 decoder 或访问 research-test。

C2 冻结 summary 随后从干净 commit `6e893a300cf683e6e0de8be7998799cabafaf32a` 生成。C201 相对 CRD_102 的 Local RR 改善 `0.6978%`、paired `2/3`、PCC drop `0.001155`、trajectory worsening `0.7648%`，基本资格通过；C202 的对应结果为 `0.5845% / 3/3 / 0.001090 / 0.7657%`，也通过基本资格。但 C202 相对同参数 C201 的 Local RR 改善为 `-0.1141%`，未达到 `+0.25%` placement 门槛，因此 decision 固定为 `decoder_capacity_supported_100hz_placement_not_supported`，选择 `crd_c201_decoder_10hz_cap`，并保留 CRD_102 Mamba backbone。该结论只支持 decoder capacity 的小幅 validation-development 收益，不支持 100-Hz placement 或带宽恢复。C0/C1/C2 控制线全部关闭；summary/formal/diagnostic 不得重跑，research-test、TCN+decoder、其他 decoder 与自动后续实验继续关闭。若继续，须以 C201 另建 candidate lock 和新协议。

## 46. CRD-TF v1 固定时频表示与交互阶段（2026-08-12）

现以 C2 选择的 `crd_c201_decoder_10hz_cap` 三个既有 validation-selected checkpoint 建立 `TF000_C201_ANCHOR`，形成 `docs/experiments/crd_tf_v1_candidate_lock_20260812.json`，lock SHA-256 为 `c8d4823500e6096fcacb8d2e8787f7b3422160813eabe31adf01b1f1f75cc139`；checkpoint/config/manifest/metrics-summary 的路径、大小和 SHA-256 已逐项复核。由此建立 research-test-informed 的 CRD-TF v1 新阶段。设计来源为 `docs/temp/时频融合_2026_08_12__0022.md`；规范性协议固定在 `docs/experiments/crd_tf_v1_protocol_20260812.md`，该附件由本节纳入当前唯一实验协议，冲突时以本文为准。

新阶段保持数据、admission、train/validation split、target、正式 `Pi`、`L_sync + 0.25 L_effort`、五项 primary、Local-RR selector 和三个 seed 不变，只研究输入 BCG 的四类表示：multi-resolution STFT、analytic Morlet CWT、learnable analytic carrier-modulation filterbank 与固定 WSST ridge。M/W/S 允许只从 train/validation 输入预计算不可覆盖 cache；L 最终特征保持可学习，只允许复用固定 input spectrum。任何 cache 均不得读取 target 或 research-test。

Stage-1 预注册为 12 个 representation arms 加 `CTRL1/2/3`，共 15 个新 variant、45 个 formal runs；single 结果不得关闭 pair/triple。若统一 physical batch 从 `128×1` 回退到 `64×2` 或 `32×4`，必须新增同 batch 的 C201 三 seed control，总规模变为 48 runs。P0–P3 已完成；当前等待用户确认 P4 的 45-run 成本，不授权提前执行正式训练。

对 54 个既有完整 CRD formal histories 的回顾显示 selected epoch median 为 13、`43/54` 不晚于 25，但最大为 72；patience 20/30 的回放分别会错过 4/2 个历史全局 Local-RR 最佳。因此本 Stage-1 继续固定 80 epochs、关闭 early stopping、保留 Local-RR best checkpoint，不同时引入新的停止变量。Optimizer/LR 默认沿用 80×128×1、AdamW、`3e-4→3e-5`、5% warmup + exact cosine；梯度累计只作为全矩阵统一显存 fallback，不允许按 variant 临时改变。

P3 CUDA/physical-batch acceptance 已完成；P4 三 seed formal、P5 冻结汇总与 P6 gated/cross-attention Fusion 继续关闭。只有附件第 12 节工程项全部完成、精确参数/缓存/batch/命令写回、工作树在统一干净 commit，且用户明确确认长时间 GPU 队列后，才可由本节后续修订开放 P4。P5 即使形成候选也不自动开放 research-test；强泛化证据需新的锁定 cohort、外部数据或 prospective holdout。

用户随后接受默认 `runs/crd_tf_v1/cache/` 与新增 `CTRL3` 后的 45-run 设计。P1 已实现 M/W/S 固定表示、L learnable modulation/固定 input-spectrum cache、synthetic calibration、只允许 train/validation 的不可覆盖 cache builder 和定向测试；`10 passed`，M/L 轻量 synthetic 检查通过。完整 W/S calibration 与约 4 GiB 全量 cache 尚未运行；入口要求干净工作树，运行完成且结果写回前 P2–P6 状态不变。

P1 随后由用户在干净 commit `6d16976010324c73b4c3b7aa9e313fd7f0d358c4` 完成。Synthetic calibration 四项全部通过，固定 S 参数为 `smoothness_penalty=2.0 / suppression_radius_bins=2`，产物 SHA-256 为 `044494e6c6966a98eb5dc00fb8bee1dcb68539910aeb9d750eb647781da7e3c0`。完整 fixed cache 覆盖 `10141 train + 2675 validation`，14 个文件共 `3.6398 GiB`；逐文件 hash/shape/dtype/finite、row identity 与 train–validation row-id 零交集审计通过，manifest SHA-256 为 `6fb44aad2689d9426ad78dc1f054db5aaac698792af5818bc01a54563cb9f0b8`，固定目录为 `runs/crd_tf_v1/cache/bd6cea7348f6b51ed768b89cf9b3425530b6358a82ba78277844517a1c27fea0/`。Manifest 固定 `target_read=false / research_test_used=false / test_cache_created=false / model_inference_used=false`。P1 至此关闭并只开放 P2 代码、配置与 CPU 定向测试；P3 GPU、P4 formal、P5 汇总、P6 Fusion 和 research-test 继续关闭。

P2 随后在 commit `4d53444` 完成：实现冻结 cache reader、M/W/L/S encoder、统一 zero-init FiLM、CTRL1/2/3、15-arm 严格配置和预注册汇总纯函数；共同 branch 预算为 `150,000±2%`，定向回归 `60 passed`。P3 入口随后实现为全 15-arm CUDA synthetic 和固定最大 single/pair/triple `TF102-W / TF204-WL / TF302-WLS` 的独立 batch-128 acceptance，并预注册同 commit 完整性、finite checkpoint/metrics、prediction degeneracy、吞吐与 peak reserved `≤80%` 审计。P3 尚未产生 GPU 结果，P4–P6 继续关闭。

首次 P3 CUDA synthetic 的 15 项虽均报告 finite，但准入审计发现新增 TF temporal mixer 错误继承通用 CRD block 的 `Dropout(0.10)`，违反附件冻结的新增 encoder dropout=0。该 receipt 作废且未开放 acceptance；修订仅将新增 TF mixer dropout 固定为 0，C201 主干和所有科学口径不变，须从修订后的新干净 commit 重跑全矩阵 synthetic。

Dropout 修订后的 15-arm CUDA synthetic 全部通过，但最大 single TF102-W 的首轮 `128×1` acceptance 虽完成完整生命周期，peak reserved 为 `14,488/15,936 MiB=90.91%`，超过 80% 工程线，故判定失败并停止 pair/triple。按附件已预注册的 activation-checkpoint 优先路径，新增 TF representation/control branch 训练态固定按 sample 轴 `chunk=8` 做 non-reentrant checkpoint，eval 不分块；该修订不改变数学输出、参数、C201、effective batch 或 LR，并须从新干净 commit 重跑 synthetic 与 TF102。只有该路径仍不满足 80% 时才由用户决定统一 `64×2` 或 `32×4` fallback。

上述修订后，commit `56cabf1d37fa01104902b6aeaef3d256bee6b2a1` 的 15-arm CUDA synthetic 与 TF102-W / TF204-WL / TF302-WLS 三项 `128×1` acceptance 全部通过；最大 peak reserved fraction 为 TF302 的 `64.76%`，最低单-update lifecycle throughput 为 `7.361 samples/s`。统一 P3 receipt SHA-256 为 `d68790e5db45ce65f60a17badab535196a913466176b7ac52a5cac4910f49f14`，固定 batch 决策为 `128×1`，不触发 fallback 或 TF000 batch control。P3 至此关闭；45-run 顺序、三个 seed 和输出根目录已在附件第 19 节冻结，P4 仍等待用户明确确认成本，config gate 继续拒绝 formal。

用户随后明确选择执行“完整”45-run 矩阵，但拒绝把所有实验整理成统一入口。P4 改为 15 个独立 formal arm 配置 × 3 seeds，各自通过通用 `train_crd.py` 单独启动；不实现队列状态机、自动跳过或自动重试。每个 run 仍强制干净 commit/P3 receipt/关键实现 identity，完整 45 项计划、batch/LR/early-stop 与证据约束不变；P5/P6 继续关闭。

P4 随后完成全部 45 个正式 run，总审计确认均来自干净 commit `68b3b85`，每项 80 epochs/6400 updates、2675 条 validation、best/final checkpoint 与 primary/degeneracy 契约合格。TF302-WLS seed 20260811 有一个不含 history/checkpoint/metrics 的早期 incomplete 目录，随后完整重跑有效；partial 保留并须在 P5 exclusion 表显式登记。当前只开放附件第 21 节的一次性 P5 validation 汇总入口，P4 不重跑，research-test/P6 继续关闭。

P5 随后从干净 commit `c7b65b1` 一次性完成，summary SHA-256 为 `afb6feba1600c9c5e07d713db9cac7c886aa87a033037649d3e9ac32cb753b4e`。M/W/S/MS 通过 base guardrail 与匹配 capacity qualification；absolute Pareto 为 M/W/MS，single Pareto 为 M/W，MS 五项 interaction 均 descriptive positive，最终未来 P6 候选并集冻结为 M/W/MS。未构造总分或唯一赢家，证据仍是 validation-development；P4/P5/research-test 均关闭，P6 需用户另立协议确认。

## 47. CRD-TF v1 P6a 三路加法与有界门控（2026-08-15）

用户已确认按 9-run P6a 继续。新附件 `docs/experiments/crd_tf_v1_p6a_protocol_20260815.md` 由本节纳入唯一协议：只新增 `MWS-ADD / MWS-GATE / CTRL-GATE × seeds 20260811/12/13`，保留既有 MS、CTRL3 和 C201 作冻结对照。Gate 只读共享 temporal latent，以 zero-init 三路 logits 产生 `[0.5,1.5]` factor；MWS-GATE 与 CTRL-GATE 总增量分别为 `463,427 / 463,203`，差 224 parameters。P4/P5 的 15-arm 集合与历史结果不变。

P6a 保持 cache、数据/split/target、loss/metrics、physical batch `128×1`、AdamW 与 6400-update LR schedule 不变。对 45 条 P4 history 回放时，patience `10/15/20/25/30` 分别错过 `5/4/3/1/0` 个全程最优，因此冻结 `max_epochs=80 / early-stop patience=30 / min_delta=0`；该回放不保证新结构没有更晚最优，仍是残余风险。三个新 variant 的 CUDA synthetic 与 MWS-GATE/CTRL-GATE batch-128 acceptance 已在 engineering commit `3802423` 通过，最大 reserved fraction `67.48%`；统一 receipt SHA-256 为 `a23c1dd9aca724ecae3d867429911043a0da1843ede61a3784578abbf99040a9`，batch 决策保持 `128×1`。9-run formal 现通过 frozen-receipt/critical-identity preflight 开放。Research-test、local cross-attention、batch/LR 搜索与统一矩阵 runner 继续关闭。

P6a 9/9 formal 随后从干净 commit `94033ce` 完成，全部由 patience=30 在 epoch `38–51` 正常停止，selected epoch 为 `8–21`，无 incomplete；最大长期 reserved fraction 为 `85.48%`，所有生命周期均完整且 finite。冻结 summary SHA-256 为 `b970a6ea6d77e6d6858d8b7dbd77ed4c2ed8ff633c7f48eca15aeba227ef6e64`。MWS-ADD 没有任何 primary 实质优于 MS 且 PCC 未过 base 护栏；MWS-GATE 没有任何 primary 实质优于 MWS-ADD。最终无 qualified P6a candidate，保留 P5 的 M/W/MS 候选池，不选唯一赢家。P6a/formal/summary 关闭，P6b 与 research-test 仍须另立协议。

## 48. CRD-TF v1 冻结候选池独立测试集评价（2026-08-16）

用户随后授权开始独立测试集评价。附件`docs/experiments/crd_tf_v1_research_test_protocol_20260816.md`由本节纳入唯一执行协议；candidate与checkpoint均在本阶段test评价前冻结。

评价矩阵固定为 C201 anchor 与 P5 保留的 M/W/MS，各三个 validation-selected checkpoints，共 12 次；不加入 P6a 失败模型或其他 P4 arms，不重选 epoch、不重训。第一步只开放独立的完整 2310-window test input-only M/W/S cache builder，保持原 train/validation cache 不变且不读取 test target array。Cache manifest 审计写回前，checkpoint evaluation 与 test target 继续关闭。

完整 input-only cache 随后从干净 commit `dfd9313` 生成，覆盖 `2310 windows / 8 samp_id`，目录 514 MiB；manifest SHA-256=`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`。7 个受管文件的实际 SHA/size/shape/dtype/finite、row identity 与 manifest 重新计算完全一致，且 `test_target_array_read=false / model_inference_used=false`。现开放只接受冻结 allowlist 的 12-checkpoint 专用入口；普通 CRD eval 仍保持 validation-only，R4 汇总在 12 项齐备前关闭。

12/12独立测试集evaluations随后从干净commit`9f429da`完成；全部checkpoint/cache/row/finite/eligibility/degeneracy identity通过，无checkpoint reselection。一次性summary从干净commit`a1ce90c`生成，SHA-256=`e9430d3449e1e75cbab1804f1c887803ba8c12dcc4b11582f94090a6a1d7c6c0`。M/W/MS均通过C201 guardrail与至少一项paired material improvement；tolerance-aware Pareto为W/MS，M被支配。W是Local-RR lead（mean `0.609566`），但W的global envelope与若干secondary不占优，因此不构造总分或唯一赢家。独立测试集评价阶段关闭。

## 49. CRD-TF-W v2 机制、频带与效率计划（2026-08-17 修订）

用户已确认以 `docs/experiments/crd_tf_w_v2_protocol_20260817.md` 作为规范性附件；原 `crd_tf_w_v2_protocol_20260816.md` 与 `docs/temp/实验计划20260817.md` 只保留为设计演化记录。新附件冲突时仍以本文为准。

本阶段由独立测试集结果启发形成新的development问题；测试集允许在阶段之间重复评价，但不参与当前阶段的训练或validation选择。当前明确暂不做`samp_id` delta、`5/7`方向门槛或leave-one-`samp_id`-out；`samp_id`仅保留作逐sample身份追溯，正式汇总继续使用逐sample direct mean与三个paired training seeds。

修订后的固定顺序为：P0 candidate lock；P−1 对冻结 W 三 checkpoint 做 gamma/beta、频带遮挡和时间负对照的 validation-only 功能审计；P1 运行 RESP-only、CARRIER-only 与 full-band 6-voice 各三个 seed；P2 仅在 P−1 预注册 near 规则触发时运行 ADD 或 SCALE 中一个 arm；P3 只在当前 `W0_FULL_12V_FILM` 上训练 D4 三 seed，并复用既有 D6。各问题保持独立，不构造未经训练的频带/voices/融合/D4 复合候选。

质量候选采用 `0.5% / 0.002` 实质改善、`2/3` paired seeds、Local RR 最多恶化 `0.5%`、其他 error primary 最多恶化 `1.5%`、PCC 最多下降 `0.003`；`3% / 0.005` 只作 catastrophic failure。效率候选要求四个 error primary 在 1% 内、PCC 下降不超过 0.003、预注册结构缩减，并达到 throughput `+10%` 或 peak allocated `−15%`。不构造加权总分。

剩余预算固定为 12 个新 training runs；P−1 未触发融合训练，原条件硬上限 15 已关闭。Morlet Q、concat、D8、gate、attention、local cross-attention、TCN+decoder、其他 decoder 与全因子矩阵均关闭。每个新 variant 在 formal 前须通过至少 400 optimizer updates、两次完整 validation 的隔离 stress；peak reserved `<85%` 通过、`85%–90%` warning 通过、`>90%` 或 OOM 关闭，不为单臂改变 batch。

P0 静态审计随后完成，candidate lock 固定为 `docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json`，SHA-256=`6ae35076bbd89bec688bfd4918cfecd20c7d5ea7f845f460034a88045432c7b6`。C201/W0/CTRL1 的 9 个 checkpoint 与 36 项配套文件 identity 全部复核通过；三类模型各一个代表 checkpoint 的 current-code strict-load 与同 seed base initialization identity 通过。W0/C201/W-branch/CTRL1 参数数为 `1,219,850 / 1,069,802 / 150,048 / 1,219,754`，D4 静态准确参数为 `902,722`。

P0 同时纠正名义频带与 cache 实际离散映射的表述：W 的名义目标网格为 `0.03125–8.00 Hz`，实际 97 个 mapped centers 为 `0.03662109375–7.99560546875 Hz`；RESP 固定为 indices `0..55`（56 scales，`≤0.80 Hz`），CARRIER 为 `56..96`（41 scales，`>0.80 Hz`），6V 为偶数 indices `0,2,…,96`（49 scales）。P0 未推理、未改 cache、未训练、未访问 research-test；审计时 runtime code 无 dirty。

P0 随后关闭并只保留 provenance。用户明确允许进入 P−1 实现阶段；独立 `tf_w_v2_audit.py` 包装器和专用 `eval_crd_tf_w_v2_functional_audit.py` 已实现，未修改锁定的 v1 model/cache reader。入口只接受固定 lock/hash、干净 Git、CUDA、完整 validation、三个 W0 checkpoints 与 10 项干预；FULL 必须在绝对 `1e-6` 内复现锁定 primary/degeneracy summary。定向 `py_compile`、candidate-lock input verification 与新审计/model/data/experiment 回归为 `40 passed`；测试覆盖去除原生 Mamba 执行后的真实 W0 active-FiLM wrapper identity，没有加载冻结 W0 checkpoint、运行完整 validation、训练或 research-test。

第一次完整 P−1 audit 从干净 commit `82926b2` 在 seed `20260811` FULL 锚点停止，最大 summary 绝对差为 `2.6775125796740795e-05`；失败记录按约定保留。随后同 checkpoint 的原生 validation 复评与冻结六项 summary 差均为 `0.0`，定位为包装器在 decoder 前执行统计算子改变 CUDA 执行路径，而非 checkpoint、cache、指标或环境漂移。实现已修订为 FULL 直接调用原生 W0 forward、用只读 hook 捕获 FiLM 输出并在原生输出完成后统计；`1e-6` 门槛保持不变。第一次失败不形成科研结果，须从新的干净 commit 重跑。

第二次完整 audit 随后从干净 commit `d91db6e2177fa8e4fea3ac27237cff2cdb8b9c35` 完成 30/30 validation evaluations，source manifest SHA-256=`249c761799b1f8020a77ed51875985776718d9cf1e9701900ce5dae783492f3f`。FULL 三 seed 最大锚点差均低于 `9e-17`；80,250 行逐 sample metrics、8,025 行 FiLM statistics 与 40 行 summary 的身份、行数、finite 和 hash 通过。BETA_ONLY/GAMMA_ONLY 均非 quality-near，P2 固定为 `retain_film_no_p2_training`，不训练 ADD/SCALE。

独立复核发现 source `film_statistics.csv` 的 gamma/beta 相邻帧统计错用了 `mean(diff(abs(x)))`，分别产生 `1,974 / 1,887` 个负值；协议要求 `mean(abs(diff(x)))`。该后处理错误不影响 prediction、primary、干预 summary 或 P2 decision，source audit 目录保持不可改写。修订实现新增只读 source hash 校验和独立 correction 入口，定向回归为 `42 passed`。

FiLM statistics correction 随后从干净 commit `4eb9b3ce937792d151393a40c0f95b5cab0e7c9b` 完成，manifest SHA-256=`1c6f7218c280b2a1579b2169a2b4752d7c10416d89de67ed5e9a1cc58d104463`，corrected CSV SHA-256=`2b7a4edef8e9828356a336c3c1ec7880235914207b00d164bf016ce1cd7a5203`。三个 FULL 锚点通过，8,025 行 identity/finite/nonnegative 通过，gamma/beta 负值数归零，六项非目标统计最大漂移 `<1e-16`；source audit 三项 hash 未变。修正后的 `mean|Δ_t g| / mean|Δ_t b|` 为 `0.02261490 / 0.02396674`。P−1 至此关闭，完整 audit 和 correction 均不得重复运行。

P1 三臂实现随后完成并锁定为 `docs/experiments/crd_tf_w_v2_p1_implementation_lock_20260817.json`，SHA-256=`fb822ca7f8607e45443e07a94d91150bcc108217fb25c47f0b4504fbdf644f58`。W1/W2 为 encoder 前 `[97,360]` input-only mask；W3 为固定 even indices 的 `[49,360]` strided view，source cache 和 host→device tensor 仍为完整 12V，不声称磁盘/传输减半。三臂参数均为 `1,219,850`，W0/C201 同 seed state identity、source/frequency/index hash、六份 config 与定向 receipt/formal gate 均通过；定向回归为 `112 passed`。

三个 isolation stress 与 9/9 formal 随后均从干净 commit `1d1b22edc6f1b9b96459c1b05eddf39208041162` 完成。stress receipt 全部 passed；formal 每 run 完整 80 epochs / 6,400 updates，seed/config/cache/view/checkpoint identity、finite 与 degeneracy 复核通过，未修改 source cache、未读取 research-test。W1/W2/W3 selected epoch 分别为 `9/18/29`、`10/9/15`、`9/12/12`。

相对 W0，W1 的 Local RR/PCC 分别变化 `+1.2168% / −0.003938`，不通过质量门槛；W2 虽使 Whole/Local RR 改善 `6.7179% / 2.4838%`，但 global envelope 恶化 `1.7038%`、PCC 下降 `0.003158`，仍不通过严格门槛。W3 的 Whole/Local/trajectory/global-envelope/PCC 变化为 `−3.7254% / −0.7968% / −2.4228% / −0.2133% / −0.002064`，进入质量候选池；其 active elements 减少 `49.4845%`，但 throughput 仅 `+4.2809%` 且 peak allocated 未下降，不进入效率候选池。P4 最终候选池仍等待 P3。

P1 executable config、实现测试与 W0 anchor 均固定 `early_stopping_enabled=false`、完整 80 epochs；附件第 3 节原 `patience 30` 为文档错误，现已透明纠正。所有 P1 selected epoch 均不晚于 29，尾部训练不改变所选 checkpoint。P1 至此关闭且不得重跑。

P3 D4 实现随后完成并锁定为 `docs/experiments/crd_tf_w_v2_p3_implementation_lock_20260818.json`，SHA-256=`0aa2f520a52a667640ea6550a6d26c48b0f65dd088de6b4616938705a7e40da7`。唯一 variant `crd_tfw_v2_d4_full_12v_film` 保持 full-12V W branch/FiLM 和同 seed 共享 state，只移除 local blocks 4/5；三个固定 seed identity 通过，准确参数为 `902,722`，减少 `317,128 = 25.9973%`。P2/D8/复合 variant 未实现。P3 stress/formal config、专用 receipt 与 same-commit formal preflight 已完成，定向回归为 `123 passed`，未运行 GPU。

D4 isolation stress 随后从干净 commit `6c4f6229eda6eb72c82e4cd17571bdc73bd97d54` 完成，receipt SHA-256=`431604d505f10733082ad5a380a75c2544ff45649f86fc3452301d7d0bb61a8c`；400 updates、5 次完整 validation、finite/degeneracy/梯度与显存线全部通过。3/3 formal 继续从同一 commit 完成，selected epoch 为 `13/26/14`，所有 lifecycle、identity、finite 与 degeneracy 复核通过。

D4 相对 W0 的 Whole/Local/trajectory/global-envelope/PCC 变化为 `+0.2519% / +1.8308% / +1.2847% / −0.3187% / −0.002938`。其参数减少 `25.9973%`、throughput 提升 `24.8925%`、peak allocated 降低 `22.5204%`，但 Local RR/trajectory 均超过效率候选的 `1%` 保护线；D4 因此不进入质量或效率池，也不触发 catastrophic failure。P3 已关闭，不开放 D8 或组合补跑。

12 个新 training runs 已全部完成，剩余训练预算为 0。P4 专用 summarizer 与 schema 测试随后完成，定向回归为 `127 passed`；从干净 commit `cd81f68b65125f3bf604b25cb2fd010056018077` 一次性审计真实 15-run matrix 并冻结六文件 summary。严格质量池/Pareto=W3，严格效率池/Pareto=空，描述性质量—效率 Pareto=W3/D4；summary SHA-256=`0f5a62448aa8db6b3bc9b07633bedf2e857b6369427792d37872851841cbd70b`，manifest SHA-256=`d9f32d26fa9bf8b98ec739762a30717a881d62a32ccb44099e603fafe94b0e97`。该双层口径保留 D4“显著计算收益换取轻度、非灾难性质量损失”的研究信息，但不绕过预注册硬门槛。

P5 随后获得用户明确授权并从干净 commit `4ef901c17ef6167f2531232e43e0941361563c81` 完成专用 allowlist/evaluator/summarizer 实现，定向回归为 `141 passed`。allowlist SHA-256=`c3fe1320a8342a9580fff2864218c948c451b21c05a863db63efe269f1358b07`，只允许 W3 三个 validation-selected checkpoints（epoch `9/12/12`）；D4 仍保留为描述性 efficiency trade-off，但不越过 strict gate。

W3 三项 P5 evaluation 与一次性 summary 随后从干净 commit `508a936b5c3b997c19c5861b6a6e0874e41dcd43` 完成。summary SHA-256=`d2d2f24ba9628a5f099c0698137c88918b51e802c0aa2ee0faf6f8ff1212f862`，manifest SHA-256=`3169d062b3b93131d1bb20f38f7870ad23cd5c9536706efc50a1854d23ec415e`。W3 相对 C201 五项 mean 均改善；相对 W0 则 Whole/Local RR 恶化 `9.8265% / 7.1294%`，trajectory/global/PCC 改善 `1.7417% / 3.9962% / +0.002117`。结论固定为 W0 保持 RR rate 优势、W3 提供 morphology/correlation 优势，不构造总分或唯一赢家。D4 的显著计算收益与轻度 validation 性能损失继续保留为描述性结果，但未做 research-test。P5 未重选 checkpoint/候选，未做 `samp_id` 分析；本阶段全部关闭。

## 50. RTM-v1 合理容量时序模型家族比较（2026-08-20）

现另立 `Respiration Temporal Modeling v1`，协议 ID 为 `resp-temporal-v1-validation-20260820`，规范性附件为 `docs/experiments/resp_temporal_v1_protocol_20260820.md`。该阶段不续接或重开 CRD/CRD-TF/CRD-TF-W，也不把旧 M1、C1 或 D4 伪装成新实验：M1 继续只解释为 `base_channels=1` 的参数匹配多尺度控制；C1 是合理但单点、H=488 参数匹配的 full-context TCN control；D4 是 W0 条件下的 Mamba 深度效率控制。它们都不能证明多尺度家族无效、TCN 普遍弱于 Mamba或 Mamba 普遍优越。

RTM-v1 的科学选择顺序改为 signal-first：先按 `docs/experiments/resp_temporal_v1_signal_audit_20260820.md` 只读 train，审计 0.05–0.8-Hz displacement、0.8–3/3–8-Hz carrier modulation、carrier 解调与10-Hz降采样顺序、抗混叠以及 cycle/rate/effort/global 时间尺度；再由用户锁定公共 signal substrate、家族集合和每家族唯一核心代表。C0 只证明 target 的10-Hz输出表示近似无损，不证明100-Hz BCG可在未充分解调前直接压到10 Hz。

核心家族假设为 dilated TCN、BiMamba2、BiLSTM 和 feature-level multiscale，T0 仍是 trunk-attribution control；当前所谓 PatchMixer 实为固定1800位置的 global token mixer，只有“弱局部先验的全窗位置混合”构成独立反方假设时才条件进入。现有 Full/Compact 共11项仅为 implementation probes，不是 formal candidates；同家族第二容量点只能另立 capacity-sensitivity control。参数、吞吐、显存与 wall time只作工程 receipt 和描述性质量—效率 Pareto，不能选择 Full/Compact、替换不可运行代表或改变科学家族集合。

完整 train-only signal audit 随后从干净 commit `2ce00509184954ec764e9cf80041f6c5f6aadfda` 完成：`10141` windows/`32` samp_ids与全部 row/content hashes、finite/null closure、artifact hashes及 access flags通过；receipt/manifest/summary SHA-256分别为`1dbdc18836c13d8c2cd888c186477c40b044e190afed3e8817c082bd9b235428 / 19ab1ece34e40b0b970f1f620f20015742c7ae1de57df3f20a2f904403763eb2 / 61b5c5d17b11b71731bf399ac2d12c684c724ac1aebe4979b577797a96b49483`。审计只读train，未访问validation/research-test/checkpoint，未训练/推理模型或使用GPU，现已关闭且不得重跑。

用户确认 lock 草案后，signal-substrate lock 固定显式anti-aliased `100→20 Hz`、20-Hz shared carrier-sensitive filtering + nonlinearity、再显式anti-aliased `20→10 Hz`；不增加手工carrier输入。Multiscale固定`10/2/1 Hz`、显式low-pass+decimation，1 Hz只作context/effort并用dilation至32获得253秒感受野；关闭0.5 Hz和average pooling。核心矩阵固定T0 + `TCN d9/H384` + `BiMamba2 D96×6` + `BiLSTM H96×2` + `multiscale H384`，global token mixer与全部同家族第二容量点关闭。Signal/candidate lock SHA-256分别为`11bfcad00f4532d4bdfe1413a375b5f06f46eb8ac67dfcd475701872322fee69 / b4a2c83310fa2ce9519e3ca25814aea0b179458ab52d6380a932545c99c25f9b`。下一阶段只允许exact CPU implementation与轻量确定性测试；GPU engineering和formal training仍未授权。

Exact CPU implementation随后在commit `61988b28fe41a429761cc531f1e035141673580d`完成：公共stem严格执行固定Kaiser FIR `100→20`、20-Hz learned filtering/nonlinearity、固定Kaiser FIR `20→10`；multiscale严格为reflect-FIR/block-center `10/2/1 Hz`，不使用average pooling或分支waveform head。五项strict configs与候选参数数固定为`59042 / 729506 / 1010426 / 449474 / 1129730`。Float64 audit等价 tolerance固定`atol=rtol=5e-12`，四项max-abs error均不超过`1.12e-15`；signal-audit回归与implementation定向测试共`37 passed`。Implementation receipt SHA-256=`6ef3ca0d48e1ac819ff55bab4247d542c8bae0adfddb6951c8a026e81fea7872`。该receipt只证明synthetic CPU实现、数值等价和结构identity，不含任何数据访问、模型质量或GPU证据；GPU engineering、formal training、validation与research-test继续关闭。

用户随后明确授权GPU engineering实现。统一synthetic-only harness在commit `146fb678522a71cc27571727d720685b42d3d5c6`完成：目标设备固定RTX 4070 Ti SUPER/bf16，五项候选固定batch-1 inference、batch-8 forward/真实loss backward，并按`128×1→64×2→32×4`只为executability选择第一个finite scheme；吞吐/显存/wall time不选择或替换科学候选。输出固定fail-closed、不可覆盖、原子发布并登记严格hash/access/finite receipt。相关50项CPU定向测试通过，harness implementation receipt SHA-256=`2f27a6582d71036e8480dc839dc563673a93551c32ba38e5d1ce302e6957fdfe`。尚未运行GPU smoke/benchmark，未访问任何dataset/split/checkpoint；用户返回execution receipt前formal training继续关闭。

GPU engineering v1随后从干净commit `5d7cea0ea914ab833bb06201d4bf13a1b9c15d5f`手动执行并在BiLSTM迁移CUDA时fail closed：PyTorch编译要求cuDNN 9.20，进程因外部`LD_LIBRARY_PATH`加载9.8。Receipt/manifest SHA-256=`3ae8e96ebfbcb2ef1eeb75134088d364bea085ed37fbe9e72041d569f50d3ad0 / 96b8c5ddddf62df2d890e3293ceb69804c97ca96dee37702981222a8daeae28c`；3/5项完成、nonfinite=0、全部禁用access flags通过。该环境失败不构成BiLSTM或family负证据，前三项partial measurements也不形成统一比较；v1产物冻结且不得重跑。V2 correction在commit `73e7681d35ea64c3323d9178238e7ddea9736f0a`只增加unset `LD_LIBRARY_PATH/LD_PRELOAD`、runtime cuDNN=92000与输出创建前LSTM canary，并使用新v2目录；候选与benchmark合同完全不变。52项CPU定向测试通过，v2 implementation receipt SHA-256=`3c2124d6f4b637b8020540fd1e97256bc291972ac70b03c58af323cbf22d6ff9`，等待用户手动执行。

V2随后从干净commit `d51659d18c8905a5dc62e9a00ed399df198c89fc`完整执行：receipt/manifest SHA-256=`c8d34d5f1a9f944af58945f74110b0c9ff74e15696c7a7f240465ce4c83de38a / e90df905b2708853a761328995c4fd0c6db1e2564b1bb16eb9328ed709f25812`，cuDNN=92000且LSTM canary通过；5/5 candidates均在首个128×1 scheme finite通过，OOM=0，numeric finite/null/nonfinite=`315/10/0`，全部artifact/access/hash闭合。用户确认GPU engineering lock，decision=`all_five_hardware_feasible_at_128x1`，lock SHA-256=`5632b0e404f943e61c646a8ddcf1761c2391884412915ef1d79aa342f8bc0183`。若以后另行授权formal，五项统一physical/effective batch=`128/128`、accumulation=1；工程吞吐、显存与规划wall time不得选择或替换候选。Formal training、validation与research-test继续关闭。

用户随后授权formal training实现与用户手动执行。冻结附件/plan为`docs/experiments/resp_temporal_v1_formal_protocol_20260820.md / configs/resp_temporal_v1/formal_v1.yaml`，SHA-256=`208b8e0f80c4a2bd426e215567702e41ee78f768e513fdb65266be3d8757c05d / fb1d4652c3e65bbcb1eb4b4b770e96315dbef04011cdae8e66391a33265d0697`。单run runner在commit `f2b59458d95ba0adac342b38aa0b9994f2b6c549`完成：5项candidate×3 seeds固定15项，每项80 epochs/6400 updates、physical128×accum1，只允许train/validation与full-validation Local-RR strict-`<` selector；preflight闭合clean Git、signal/candidate/CPU/GPU source hash、v2 execution、cuDNN 92000/LSTM canary及固定data identity，输出不可覆盖/resume并登记lifecycle/count/finite/hash/access receipt。66项RTM-v1 CPU定向测试通过，implementation receipt SHA-256=`0d2dae3fea51f18821a47a11e69d8a8b2ca765898c3d84c228d2fedffcaff0fa`；Codex未读取dataset/index/split或既有checkpoint，未运行GPU/formal/validation。当前0/15，长任务只由用户逐项执行；15/15前不得汇总partial quality或停止正常arm，validation summarizer与research-test继续关闭。

用户在0/15 formal runs时要求分成两组分别绑定两颗GPU。只读`nvidia-smi`确认physical 0/1均为RTX 4070 Ti SUPER、16376 MiB；原plan因receipt不能区分物理卡而冻结为`superseded_before_execution`，未删除、覆盖或产生run。修订附件/唯一plan SHA-256=`1c2cad1c5cc401776d894a0bb039e0ad75a8cd3030f66ea70b48817f2de1b84a / c6594d160a3f7ecb1f39a4e036996491dd6f1919aedeea28949ff8533a1dd8c3`，预注册`gpu_0/CUDA_VISIBLE_DEVICES=0` 8项与`gpu_1/CUDA_VISIBLE_DEVICES=1` 7项，candidate与seed均不完全绑定单卡；组内串行、组间可并行。Runner在commit `ff1fa163b7f9488f226b138c24c3f35ea9d0b92c`强制group/环境变量匹配，并把group、物理可见索引和逻辑`cuda:0`写入lifecycle/manifest/receipt；其余科学与训练合同不变。68项RTM-v1 CPU定向测试通过，dual-GPU implementation receipt SHA-256=`fa07aae4f83b74c86bf5113840058af99a768d33b16aad883dc6d6368998d888`；没有创建CUDA context、读取data/checkpoint或启动formal。当前仍为0/15。

用户随后完成全部15项formal。只读验收确认15/15 receipt与lifecycle均complete、每项80 epochs/6400 updates、总计96000 updates、8/7 GPU分组正确、无failed lifecycle；全部来自同一干净commit `24c54a88ea8b17ad183b48976bd1f690e3867706`，formal receipt sidecar、manifest、artifact hash/size/CSV rows、count/finite/access全部闭合。汇总附件随后冻结为`docs/experiments/resp_temporal_v1_validation_summary_protocol_20260821.md`：只读15×2675 validation metrics，按candidate报告三seed arithmetic mean±sample SD、50项paired-seed方向，分别计算五primary quality Pareto与加参数/standardized throughput/peak allocated的quality-efficiency Pareto，并按固定tolerance登记T0停止线；不构造总分或p-value，不读取checkpoint内容/dataset/research-test。Summarizer在commit `5e8aaae63989087b0803d8850e4be030104679bd`完成，config/implementation receipt SHA-256=`21f723580c9cbbe39a3b85c68290e5ace26c6dadf0a9b37abdee9dd8a046816f / ffe4e658ddf9ddb9fb0f0456cf0425c7b2f58994ef4af4d53c693d4911f6149c`，81项RTM-v1 CPU定向测试通过。正式summary尚待用户手动执行；当前不得解释候选结果，research-test继续关闭。

用户随后从干净commit `670d3bd36182fb4c4b982427616e8d1bc4fef42d`一次性完成validation summary。Receipt/manifest SHA-256=`6f9f1e873b8910b22241bc0e9f2c510909835b0bbc2a9edc12f0e1788fa59aad / aab22094d6efd11927c952e9f12bcbab24e30cedc9a2a5f282f61056f1f193dc`；15项formal、40125行validation、5项candidate、50项paired-seed与40项dominance audit闭合，numeric finite/null/nonfinite=`3425/120/0`，null仅为不适用资源列，且summary未读取checkpoint内容、dataset/index、signal/target或research-test。冻结candidate-mean tolerance下，质量Pareto仅含`rtm_v1_multiscale_10_2_1_h384`；其相对T0的Whole/Local/trajectory/global改善为`4.5777% / 5.9896% / 6.9115% / 32.9483%`，PCC绝对增加`0.037931`，五项paired material improvement均为`3/3`。它在mean/tolerance下亦支配TCN、BiMamba2和BiLSTM representative，但相对BiLSTM Whole/trajectory仅`1/3 / 2/3`原始seed更优、相对BiMamba2 Whole仅`2/3`、相对TCN trajectory仅`2/3`，不得称为逐seed一致、唯一最佳模型或multiscale family普遍优越。加入资源维度后全部五项均在质量—效率Pareto且无overall winner；BiLSTM仅作描述性质量—效率trade-off，T0仍是trunk-attribution control。用户确认validation lock，文件SHA-256=`989ef0a3a5941ead3e80aba25606878f88d315bd23a1cf5ca4260317f3cffce6`；停止线=`at_least_one_trunk_material_primary_improvement`，协议关闭且不追加搜索，research-test继续关闭。

用户随后要求将论文交付收窄为五候选validation `mean ± SD`主指标表，并明确授权开始独立测试集评价。主指标表SHA-256=`417fe73b491d459fe649a365fdad590d00c1d0aa992df0e150b144dde4ba8f21`；它只包含五项primary及三seed sample SD。独立测试集不改validation lock，固定评价全部五候选×三seed的15个validation-selected best checkpoint，只报告逐sample metrics、15行seed direct mean与5行candidate `mean ± sample SD`；关闭secondary、Pareto、排名、paired方向、总分与p-value。协议/config/implementation receipt SHA-256=`1da00281ad435a14cc0b1e9a26b84554cae35ec02784ea7f2f369ec539be8184 / 3d8551989fbfa07e8ef9454fbb348f2908151f35c681e15a6191b61a0c60a406 / e9e31c01b8f2da7a441d26b115f446c9fd71a7fc213ea3b07b5528b63a1615e6`；93项RTM-v1 CPU定向测试通过，未访问test signal/target、未生成prediction、未使用GPU。

用户从干净commit `972f158cf459f4d63bffdd5e7850455f70417e78`完成独立测试集一次性评价。Execution receipt/manifest/主指标表 SHA-256=`f9b1b216f80c4bde5a7be27aa9df76c66b8e3765ebe491ebe45fa880dd380a1e / a18d1459a3e4e9f11728c2bcebffd64f1f016f65129345c359794eee2a39ce4b / 91be5de5607de03a678b34f597100991ce81051f150b7c5cfdd7dfb508613454`；15/15 checkpoint、34650逐sample rows、5行candidate、nonfinite=0，且train/validation access=false、training/reselection=false。后续论文口径统一称“独立测试集”：其独立性由split隔离与不参与本阶段选择保证，不要求整个研究过程只能访问一次。当前仅待用户确认测试结论锁，不得重评或根据测试结果重选。

## 51. 论文实验补充与证据闭环执行协议（2026-09-01）

用户要求把论文实验补充转化为可实施的新任务，同时保持设计来源文件
`/mnt/disk_code/marques/paper/resp_rec/paper_rewriting_output/实验补充与证据闭环计划.md`
只读。规范性附件现建立为 `docs/experiments/paper_evidence_closure_protocol_20260901.md`，协议 ID 为
`paper-evidence-closure-v1-20260901`；该附件由本节纳入当前唯一实验协议，若有冲突仍以本文为准。

论文主模型固定为 W0：该选择以 Whole/Local RR 为最高任务优先级，认为 W3 的努力/PCC 收益不足以覆盖其
RR 退化；这是一项论文任务优先级选择，不改写既有 W0/W3 Pareto 权衡，也不声称 W0 在五项 primary 上逐项支配 W3。

新增窗口任务固定为 C201/`W-reduced-center60` × `60/90/180 s` 输入、统一中心 `60 s` 输出。它是游离于论文
主模型实验之外的独立上下文敏感性任务，结论限定为合理精简 W 表征下的上下文效应。W-reduced 固定从原
97 点全频程名义网格取偶数 indices `0,2,…,96`，形成 direct 49-scale、约 6 voices/octave 的 length-specific CWT；
三种输入来自同一 180-s 父 row 的嵌套裁剪，中心 target 完全相同；新任务独立定义 6000-point `Pi_60`、
`L_sync_60 + 0.25 L_effort_60`、严格 center-RR selector，以及 center RR、IBI-MedAE、trajectory、
global-envelope 和 signed PCC 五项中心指标，IBI 必须配套 coverage。单 seed 诊断矩阵为 6 runs；只有预注册模式
触发且用户确认成本后，才追加两个 seed 的 12 runs，形成完整 18-run 矩阵。现有固定 180-s 模型、cache、配置、
runner 和冻结产物不得为此放宽或覆盖。

IoT 阶段只冻结 W0/W3/D4 的 batch-1 cached/online-W、CPU/GPU 分阶段延迟、内存、参数、权重与 profiler coverage
测量合同。必须报告实际 median/p95/rounds 和 30-s step 数值余量，但不得在结果产生前写入实时、流式、部署通过或
失败结论；完整 180-s 上下文等待必须与处理开销分开说明。

用户随后授权 P0/P1 实现和定向 CPU 测试，并把窗口任务明确定位为游离于主实验之外的上下文敏感性比较。
P0 从干净 commit `936d70813b5619c4eedb8be048806cc7ce3637e3` 完成正式只读审计，固定 manifest SHA-256 为
`1e05c3de9a75c08ee016af379e075adc1b71b4d1162f7bbf81d716b4942d2542`，12 组来源均 compatible，10 个预注册方法
进入 primary table。P1 数据/cache/model/`Pi_60`/loss/metrics/selector/lifecycle 已实现。C201/W-reduced 参数数为
`1,069,802 / 1,219,850`，同 seed 的 165 个共享 state tensors
逐 tensor 相同。未运行训练、GPU、CPU lifecycle、全量 cache、benchmark 或新的独立测试集访问。
P2 固定为 synthetic-only GPU 工程验收：单一入口先执行六臂 batch-1 真实 Mamba forward/loss/backward，再只对
W-reduced-180 执行 physical batch 128 的 forward/loss/backward/AdamW step；不运行 CPU/真实数据 lifecycle，不访问
dataset/cache/checkpoint。P2 随后从干净 commit `7249fca27e5a960a3e74e43a5acf6ac0abc3a4b5` 完成：六臂 batch-1
全部 finite，W-reduced-180 physical batch 128 完成 AdamW step，peak allocated/reserved 为
`10,834.94 / 10,948.00 MiB`，decision=`batch128_accepted`；receipt SHA-256 为
`cb43ee480018afd483ce2bef15342d5bdaf29b6bc999d58ce209abac58df04ce`，manifest SHA-256 为
`09b9e3347967cd7a4eac45b39c4054c4db689ccbbfe1bdaf2a2e23469957d582`。六臂统一冻结为 `128×1`，P2 关闭。
P3 所需 60/90/180 s 完整 W cache 随后全部完成，manifest SHA-256 分别为
`e9d270c930d6862f9b5a9cbdb3c26765fe3b57d2573946640cba99e597389793`、
`9442a33ea2c633b15642d033efa3d3e09278f30c4692a707abc9de460c784757`、
`72402d8543adf3cda504967b87fc4f9528cacca58b9aa080f0dcfe06707037fe`；三份均为 `10141/2675` train/validation rows、
49-scale float32 finite input-only cache，dataset/row identity 一致，target/test access=false。六个 P3 配置固定 seed `20260811`、
80 epochs、`128×1`、统一输出根和对应 cache path/manifest hash，并显式固定/记录 matmul TF32、cuDNN TF32 与
cuDNN benchmark 均为 false；P3 gate 拒绝其他 seed。当前 35 项 paper-evidence CPU 定向测试与 64 项相关冻结回归
共 99 项通过。P3 六项等待用户按附件固定顺序手动执行；P4、P5 新增功能推理、
P6 test target-only 属性访问和 P7 效率 benchmark 仍未开放。
