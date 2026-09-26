# E8：FiLM 条件末端 × 波形解码端重设计析因实验

日期：2026-09-25。协议 ID：`e8-film-decoder-redesign-factorial-v1-20260925`。

状态：**P0–P5 已完成，36/36 formal 与完整 validation 汇总已冻结；P6 research-test 专项代码与协议已实现，当前只开放 36-checkpoint allowlist 生成，不授权 test 数组访问。**

独立开发位置：

```text
branch:   codex/e8-film-decoder-redesign
worktree: /mnt/disk_code/marques/resp_reconstruction/.worktrees/e8_film_decoder_redesign_v1
output:   /mnt/disk_code/marques/resp_reconstruction/runs/e8_film_decoder_redesign_v1
```

源码与协议在独立 worktree 中演化；运行产物未来只写入新的不可覆盖 output identity。E7、W0 三因素结构对照及全部历史 `runs/`、checkpoint、cache、锁和汇总保持只读，不补写、不覆盖、不以原 identity 重跑。

## 1. 研究问题

W0 的 CWT 条件分支在二维尺度—时间编码、尺度平均、360→1800 插值和三层时间块后，使用参数预算匹配形成的 `96→65→96` 逐点残差，再用零初始化 `96→192` 投影产生 FiLM 参数。W0 解码器在共享 32-channel 特征上使用线性基础读出和纯逐点非线性残差。

本轮回答：

1. 条件表示已经经过充分时频与时间建模后，是否可以删除任意的 65-channel 参数填充，直接投影 FiLM 参数？
2. 如果直接投影容量不足，同宽 `96→96→96` 或固定两倍扩展 `96→192→96` 是否有稳定增益？
3. 解码端应保留当前逐点残差、退回单一读出，还是把相近容量用于具有局部时间意义的残差？
4. 条件末端和解码端的作用是否相互依赖？
5. 推荐的干净结构 `direct + temporal` 是否对五项主要属性形成 tolerance-aware Pareto 改善，而不只是参数或单一指标变化？

核心证据必须来自本轮完整 12-arm、三 seed 同合同从头训练。历史 checkpoint 只提供来源与设计背景，不填充本轮单元格。

## 2. 本轮一次实现的改进清单

### 2.1 条件分支末端

前置 CWT 编码和时间细化共同输出 `c∈R^(B×96×1800)`。四个水平为：

| ID | 定义 | 末端可学习参数 | 解释 |
|---|---|---:|---|
| `fill65` | `c + PW96→65 → SiLU → PW65→96` | 12,576 | 同轮 W0 参照；65 来自历史参数预算匹配 |
| `direct` | 恒等传递 | 0 | 条件末端不再增加通道重组 |
| `res96` | `c + PW96→96 → SiLU → PW96→96` | 18,528 | 与表示/主干同宽的通道残差 |
| `res192` | `c + PW96→192 → SiLU → PW192→96` | 36,960 | 固定 2× 扩展率的容量探针 |

四组随后统一执行：

```text
PWConv1D 96→192，weight/bias 全零初始化
→ split gamma_raw, beta_raw
→ gamma=0.5*tanh(gamma_raw)
→ beta =0.5*tanh(beta_raw)
```

所有逐点 refiner 都保持时间索引，不引入额外时间混合。`direct` 是本轮优先解释的简洁候选；`res96/res192` 用于判断是否确实存在末端通道容量需求，不能因更宽而预设更好。

### 2.2 波形解码端

FiLM 后两个 refinement block 与共享 coarse head 固定：

```text
96 × 1800
→ GN(12,96)
→ Conv1D 96→64,k5 → SiLU
→ DWConv1D 64,k5 → SiLU
→ PWConv1D 64→32 → SiLU
→ h ∈ R^(B×32×1800)
```

三个读出水平为：

| ID | 定义 | residual 参数 | 解释 |
|---|---|---:|---|
| `pointwise` | `PW32→1 + [PW32→32→SiLU→zero-init PW32→1]` | 1,057 | 当前 W0/C201 解码器 |
| `single` | 只保留 `PW32→1` | 0 | 最简单一解码头 |
| `temporal` | `PW32→1 + [DW k5,d2→SiLU→PW32→32→SiLU→zero-init PW32→1]` | 1,217 | 局部时间残差修正 |

`temporal` 的残差自身理论感受野为 9 个 10-Hz token，即 0.9 s 窗口。连同共享 head 的两个 k=5 卷积，总读出路径理论感受野为 17 token（首末中心跨度 1.6 s），与 0.70 Hz 上界对应约 1.43 s 周期处于同一量级。该结构把新增容量用于局部峰谷、相位和形态修正，而不重复长程 Mamba。

### 2.3 所有单元格共同保留

1. FiLM 最终 `96→192` 投影零初始化，使所有结构从恒等调制开始。
2. 有 residual decoder 的末层零初始化；`single` 没有残差分支。三种 decoder 初始函数一致。
3. 10 Hz 生成波形，再用确定性 Fourier interpolation 到 100 Hz；固定任务投影仍在模型外执行。
4. W0 的 Patch frontend、六层 BiMamba2、CWT 数学表示、二维编码、尺度 mean、三层条件时间块、FiLM 位置、FiLM 界限和两个 post-FiLM refinement block 固定。

## 3. 为什么本轮不再加入尺度聚合因素

E7 已独立实现并预注册 `3 encoder × 2 aggregation × 3 seeds`，其中 aggregation 正是 mean 与 frequency-aware content attention。E7 当前 formal train/validation 尚未完成。把同一聚合因素再次并入 E8 会：

- 把本轮扩大为 `4×3×2×3=72` 次训练；
- 重复尚未收口的 E7 问题；
- 使条件末端/解码端结论依赖另一项仍未形成稳定整体收益的因素；
- 增加与正在执行实验的代码、资源和解释耦合。

因此 E8 固定 mean aggregation。频率感知聚合已经“实现”，但由 E7 单独给出证据；E8 不把它直接替换为新主模型。若 E7 完整 validation 后形成预注册的稳定 Pareto 改善，是否做跨实验组合确认应另立增量协议，不能回填或改变本矩阵。

## 4. 完整 `4×3` 矩阵

| Arm | 条件末端 | Decoder | Trainable params | 变动模块 covered MACs |
|---|---|---|---:|---:|
| `e8_fill65_pointwise` | fill65 | pointwise | 1,219,850 | 24,364,800 |
| `e8_fill65_single` | fill65 | single | 1,218,793 | 22,464,000 |
| `e8_fill65_temporal` | fill65 | temporal | 1,220,010 | 24,652,800 |
| `e8_direct_pointwise` | direct | pointwise | 1,207,274 | 1,900,800 |
| `e8_direct_single` | direct | single | 1,206,217 | 0 |
| `e8_direct_temporal` | direct | temporal | 1,207,434 | 2,188,800 |
| `e8_res96_pointwise` | res96 | pointwise | 1,225,802 | 35,078,400 |
| `e8_res96_single` | res96 | single | 1,224,745 | 33,177,600 |
| `e8_res96_temporal` | res96 | temporal | 1,225,962 | 35,366,400 |
| `e8_res192_pointwise` | res192 | pointwise | 1,244,234 | 68,256,000 |
| `e8_res192_single` | res192 | single | 1,243,177 | 66,355,200 |
| `e8_res192_temporal` | res192 | temporal | 1,244,394 | 68,544,000 |

Covered MACs 只覆盖变化的条件 refiner 与 residual readout，不是完整模型 FLOPs，不含公共最终 FiLM projection、共享 coarse head、Mamba、CWT encoder 或 Fourier interpolation。

每个 arm 使用 seeds `20260811 / 20260812 / 20260813`，共 36 个 train/validation run。`e8_fill65_pointwise` 是同轮 W0 参照；`e8_direct_temporal` 是设计时优先候选，但不享有 selector 或解释特权。

## 5. 初始化与配对

每个 arm 先按同一 seed 完整构造原生 W0，再做后置替换：

1. `direct` 把 active fill 替换为 parameter-free Identity；`res96/res192` 分别在命名子 seed `e8_condition_res96/e8_condition_res192` 中构造。
2. `single` 把 decoder residual 替换为 parameter-free zero correction；`temporal` 在命名子 seed `e8_decoder_temporal` 中构造。
3. 共同 W0 tensor 保持逐 tensor 同初始化；相同因素跨另一因素水平也保持逐 tensor 配对。
4. 命名子 seed 使用隔离 CPU RNG context，不改变公共 CPU/CUDA training RNG。
5. 因 FiLM projection 和 residual readout 末层为零，十二组在同 seed、同输入、eval 模式下初始 waveform 必须逐元素相同。

零初始化会造成分阶段梯度开启：第一步先更新零初始化末层，前置 refiner/temporal decoder 内层在后续 update 才收到非零梯度。工程验收必须覆盖至少三次真实 optimizer update，不能以单次 backward 判定模块失活。

## 6. 冻结训练与评价合同

本轮沿用 W0 三因素结构对照已统一的 train/validation 合同：

- `research_v2` input、target、admission、train/validation split、subject/session 隔离和 row identity；
- train/validation W cache、97 scales、预处理、sample seeds；
- `L_sync + 0.25 L_effort`；AdamW、weight decay、gradient clipping、warm-up cosine；
- physical/effective batch 128、accumulation 1、BF16、完整尾 batch、branch checkpoint chunk 8；
- 最多 80 epochs，每完整 epoch 80 updates，planned 6,400 updates；
- early stopping `min_epoch=30 / patience=15 / min_delta=0`，wait 从 epoch 1 累计；
- checkpoint selector 为完整 validation Local RR 严格最小，并列取最早；
- 五主指标、逐窗口直接平均、subject-macro、eligibility 与 finite 语义不变。

36 个单元格必须在同一冻结实现身份下全部完成后一次汇总。不得根据 partial seed、中间结果、速度或显存修改剩余结构、停止合同、容差或 selector。

## 7. 计划对比

每个 seed、每项指标先形成方向统一的效用值：四项 error 取负，PCC 保持原值。

### 7.1 条件末端简单效应

在每个 decoder 水平分别比较：

```text
direct - fill65     # 删除预算填充
res96  - direct     # 同宽通道残差的增量容量
res192 - res96      # 2× 扩展相对同宽的增量容量
```

并报告四级因素跨三个 decoder 的等权边际均值。由于四级不是等距连续剂量，不对宽度拟合线性趋势并替代这些计划对比。

### 7.2 Decoder 简单效应

在每个条件末端水平分别比较：

```text
single   - pointwise  # 删除当前逐点非线性修正
temporal - pointwise  # 相近容量改为局部时间修正
temporal - single     # 是否需要 residual decoder
```

并报告三组 decoder 对比跨四个条件水平的等权边际均值。

### 7.3 交互

以 `fill65` 为条件参照、`pointwise` 为 decoder 参照，对每个非参照水平报告 difference-in-differences：

```text
[Y(condition, decoder)-Y(condition, pointwise)]
-[Y(fill65, decoder)-Y(fill65, pointwise)]
```

共 `3×2=6` 个独立参照编码交互。交互只有结合对应简单效应解释，不能把正交互单独写成整体改善。

### 7.4 判断规则

材料性容差沿用：四项 error 相对 0.5%，PCC 绝对 0.002。一个结构只有至少一项材料性改善且其余主属性均无材料性退化时，才构成 tolerance-aware Pareto 改善。三 seed 描述优化随机性，不作为独立人群；窗口与 seed 不相乘构造样本量，不构造跨指标总分。

重点解释：

- `direct` 与 `fill65` 等效或更好：支持删除任意 65-width 参数填充。
- `res96` 改善而 `res192-res96` 容差内：支持同宽通道重组，不支持额外扩展。
- `res192` 稳定改善：说明额外末端容量值得保留，但结论是本结构/预算下的经验结果。
- `single` 与 `pointwise` 等效或更好：当前双读出没有必要。
- `temporal` 相对两者形成 Pareto 改善：支持把 residual 容量用于局部时间建模。
- 显著交互或多个属性反号：报告条件依赖/属性取舍，不选择单一普遍赢家。

## 8. 工程验收

P1 synthetic CPU 必须覆盖：

1. 12-arm spec、derived config、参数数和 optimizer 完整覆盖；
2. 条件 refiner/decoder 的模块类型、shape、finite、state round-trip；
3. direct 不含 refiner 参数，single 不含 residual 参数；
4. temporal decoder 固定 `k=5/dilation=2/padding=4/groups=32`；
5. FiLM 和 decoder 零初始化；同 seed 十二组初始函数相同；
6. 命名子 seed 的公共 state/RNG 配对；
7. 多步梯度通路真实开启；
8. 36-cell 计划完整且 formal gate 关闭。

P2 由用户执行：

1. 12 arms × 3 seeds 的 36 个 batch-1 cell，每项三次原生 loss/optimizer update，验证零初始化后的多步梯度开启与参数真实变化；
2. 最大资源 arm `e8_res192_temporal` 的 synthetic batch-128、三次原生 update，固定 branch checkpoint chunk=8 并要求 peak reserved 不超过可见设备总显存的 80%；
3. 最大资源 arm 的一轮 synthetic 原生 trainer lifecycle，保存 config、history、best/final checkpoint、validation metrics 和 runtime summary；
4. 12 arms × eval/train 的 24 个独立进程 benchmark；每项 warm-up 5 次、测量 20 次，eval batch=1、train batch=128。

P2 只使用确定性 synthetic tensor，不读取 dataset index、真实 waveform、W cache、历史 checkpoint 或 research-test。它以干净 Git commit 和关键源码逐文件 SHA 构成 engineering identity；成功与失败 attempt 均不可覆盖保留。只有 acceptance 与 benchmark 完成并收口后，才能新增 formal 入口。

当前允许命令：

```bash
./.venv/bin/python -m pytest tests/test_e8_film_decoder_redesign_v1.py -q
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py check-p1
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py describe
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py formal-plan
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py gpu-acceptance --device cuda:0
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py benchmark --device cuda:0
```

`formal-plan` 只列计划，所有 cell 状态必须为 `blocked_until_engineering_acceptance`，不执行数据访问或训练。

## 9. 阶段门控

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 问题、矩阵、冻结变量、计划对比 | 已完成 |
| P1 | 十二个模型、配置、CLI、synthetic CPU 测试 | 已完成 |
| P2 | Synthetic GPU acceptance 与 benchmark | 已完成并冻结 |
| P3 | 实现身份冻结与 36-cell formal 入口 | 已完成 |
| P4 | 36 次 train/validation | 已完成；36/36 |
| P5 | 完整 validation 汇总与冻结 | 已完成并冻结 |
| P6 | Research-test 专项协议、allowlist 与固定 checkpoint 评价 | 专项代码已实现；allowlist 待生成，test 访问未授权 |

本协议当前不授权独立测试集访问。E8 的 test 若未来开放，只能评价 P5 前已冻结的全部 validation-selected checkpoints；test 不参与 early stopping、checkpoint 选择、结构筛选或阈值修改。
