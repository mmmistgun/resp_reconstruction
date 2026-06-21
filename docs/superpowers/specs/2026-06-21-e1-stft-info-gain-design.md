# E1 STFT 输入信息增益验证 · 设计规格

## 背景与范围

本规格是 [时频输入与多分支融合实验规划](../../experiments/time_frequency_input_fusion_plan.md)
的第一批落地。规划覆盖 E0–E5；本规格只锁定 **E0 收尾 + E1（STFT 信息增益验证）**
这一个可独立交付的子项目：把现有纯时序训练栈扩展到能跑“时序 + STFT 双分支”，
产出 E1a/a'/b/c/d 对照，回答“STFT 输入是否有稳定信息增益”。

E2 及以后（分频带、可学习前端、CWT/SST、融合升级）各自有独立进入条件，
等 E1 出结论后再单独头脑风暴，不在本规格范围内。

### 目标

- 新增双分支能力：时序主干 + STFT 时频分支，特征级 concat 融合，输出呼吸波形。
- 复用现有时序主干（patch/mixer 类与多尺度低频类各一），不破坏任何纯时序行为。
- 编排并跑完 E0 收尾 + E1 首波实验，产出同口径汇总与分层诊断，给出研究判定。

### 非目标

- 不引入 gated fusion / cross-attention（规划 E5）。
- 不引入分频带 / 可学习前端 / CWT / SST（规划 E2–E4）。
- 不改动 dataset / engine / collate / loss / checkpoint 校验 / 指标实现。
- 不重新定义评价指标或 loss 权重。

## 现状约束（探索结论）

- 所有模型统一 forward 签名 `model(x)`，`x` 为 `(B,1,18000)` 单通道时序张量
  （`resp_train/engine/train.py:40,80,135`）。
- `RespWindowDataset` / `ResearchV2WindowDataset` 只产出 `{"x","target","meta"}`，
  `x` 为纯时序波形（`resp_train/data/research_v2.py:99`）。
- 模型经 `build_model(cfg)` 注册表构造，参数全从 `cfg.model.*` 取
  （`resp_train/models/registry.py`）。
- 当前栈无任何 STFT / 双分支痕迹，本能力从零新增。
- 输入采样率 `target_fs=100Hz`，窗长 180s（18000 样本）。
- 输入 BCG 已在数据准备阶段做过“段间 normalize”（按体动规则识别稳定段后段内归一化），
  因此绝对幅值不具跨段可比性，窗口间相对能量已被段间 normalize 编码。
- 两个时序主干内部均可在“出波形前”自然取到 `(B,C,T')` 特征：
  - `PatchMixer1D`：`patch_head` 之前的 `tokens`，形状 `(B,base_channels,patch_count)`
    （`resp_train/models/timeseries.py:182` 前）。
  - `MultiScaleDecompMixer1D`：`self.fuse` 之前的 `cat`，形状
    `(B,base_channels*n_scale,18000)`（`resp_train/models/lowfreq.py:154` 前）。

## 架构总览

```text
新增 resp_train/models/stft_branch.py
  ├─ STFTEncoder        固定算子 torch.stft → log1p magnitude → 频带裁剪(low/high_hz)
  │                     → 频带尺度对齐(N0/N1) → 轻编码 → (B,Cs,Ts')
  └─ TimeStftDual1D     通用双分支包装器：time_backbone(return_features=True)
                        + STFTEncoder + FusionHead；branch_mode ∈ {time_only,stft_only,dual}

小改 resp_train/models/timeseries.py   PatchMixer1D 加 return_features 开关（默认 False）
小改 resp_train/models/lowfreq.py      MultiScaleDecompMixer1D 加 return_features 开关（默认 False）
小改 resp_train/models/registry.py     注册 time_stft_dual1d

新增 scripts/precompute_stft_band_scale.py   N1 归一化的 per-freq-bin 鲁棒尺度预统计
新增 scripts/run_e1_stft_info_gain.py         E1 批次编排（笛卡尔积 + override）

完全不动：dataset / engine / collate / loss / checkpoint 校验 / 指标
```

### 边界保证

- `model(x)` 签名不变：`TimeStftDual1D.forward(x)` 仍只吃 `(B,1,18000)`，STFT 在内部算，
  engine 与数据管道零改动。
- `return_features=False` 是默认值，所有现有纯时序 run 字节级不受影响。
- E1a（纯时序 baseline）**不经过包装器**，直接用现有模型，保证对照基线严格等同 E0 代码路径。

### 数据流（dual 模式）

```text
x(B,1,18000)
 ├─ time_backbone(x, return_features=True) → f_t(B,Ct,Tt'), length
 └─ STFTEncoder(x)                         → f_s(B,Cs,Ts')
 → align_to_time 两路插值到公共网格 T_fuse=600
 → concat → FusionHead(conv 解码 + 上采样回 18000)
 → pred(B,1,18000) → 现有 loss / 指标不变
```

## 组件设计

### 1. 中间特征开关契约

现有主干 `forward` 加 `return_features: bool = False`：

- `return_features=False` → 返回 `(B,1,18000)` 波形（现状，默认，逐元素与改动前相等）。
- `return_features=True`  → 在解码头之前返回 `(features, length)`，
  `features` 形状契约统一为 `(B,C,T')` 时间序列：
  - `PatchMixer1D`：`tokens`，`(B,base_channels,patch_count)`，T'=patch_count。
  - `MultiScaleDecompMixer1D`：`cat`，`(B,base_channels*n_scale,18000)`，T'=18000。

返回 `length` 供融合头知道目标输出长度（patch 类 T' ≠ 18000）。改动各约 5 行。

### 2. STFTEncoder

```text
输入 x:(B,1,18000)
1. torch.stft(n_fft/win=3000=30s, hop=500=5s, window=hann, center)  → 复数谱
2. magnitude → log1p                                                 → 压缩动态范围（确定）
3. 频带裁剪 [0.05Hz, high_hz]，high_hz ∈ {3,8,12} 由 cfg 传入
4. 频带尺度对齐（N0/N1 对照变量，见下）
5. 编码器（1D / 2D 两种，见“编码器选型”）                            → (B,Cs,Ts')，Ts'≈31
```

**编码器选型（encoder_type，零号消融变量）：**

STFT 产出时频图 `(freq, time)`，怎么“读图”是 E1 的一个前置变量。E1 阶段不引入 E3 的
频域专用结构（frequency-token mixer / 可学习频带前端），只在两种通用读图器中二选一：

- **1D-conv（`encoder_type=conv1d`）**：频率轴塞进通道维，只沿时间做 1D 卷积。最朴素、
  自由度最低。代价：谐波/相邻频带等频率局部结构在首层即被压平，可能低估 STFT 信息量。
- **2D-CNN（`encoder_type=conv2d`）**：把 `(freq,time)` 当单通道图像做小 2D 卷积，
  给频率方向局部感受野，能捕谐波间隔与心冲击带位置；仍是通用读图器，非 E3 频域专用结构。

**零号消融（先于主线，6 run）**：在代表档 `E1b / patch_mixer1d / high_hz=8 / 3 seed`
上对跑 `conv1d` vs `conv2d`，按主护栏 + 低 SNR 分层定胜者，作为 E1 主线 48 run 的默认
编码器；另一种不再扩展。判断假设：编码器优劣在不同主干/频带上趋势一致，故一格定胜负。
若两者打平，按 YAGNI 取更简的 `conv1d`。

**归一化（log1p 确定，频带尺度对齐做成对照变量 N0/N1）：**

- 绝对幅值因段间 normalize 不可比，故**不做全局 z-score、不减均值中心化**（避免重新引入
  绝对幅值基准）；窗口间相对能量已被段间 normalize 编码，故**不做 per-window 标准化**。
- **N0（默认主跑）**：仅 `log1p`，频带尺度交给编码器首层 GroupNorm 自适应。
- **N1（对照）**：`log1p` + per-freq-bin 鲁棒尺度对齐——用 train 上的 per-freq-bin
  IQR（中位数/IQR，对翻身离群帧不敏感）**只除尺度、不减中心**，仅解决频带间数值量纲差异。
  IQR 向量由 `precompute_stft_band_scale.py` 对每个 high_hz 档算一次，存小文件注入 buffer。
- N0/N1 其余口径完全一致（同模型、同 seed、同频带），只差“频带尺度对齐”一个开关。
  N1 仅在选定代表档（E1b/patch_mixer/high_hz=3）加挂一组 3 seed 对照，不扩大主线规模。

### 3. TimeStftDual1D 包装器与三模式

```python
class TimeStftDual1D(nn.Module):
    branch_mode ∈ {"time_only","stft_only","dual"}   # cfg.model.branch_mode
    time_backbone = build_backbone(cfg.model.time_backbone, cfg)  # dual/time_only 用
    stft_encoder  = STFTEncoder(cfg)                              # dual/stft_only 用
    fusion_head   = FusionHead(...)   # concat → conv 解码 → 上采样 → (B,1,18000)

    forward(x):                          # 签名不变
        feats = []
        if mode in {time_only,dual}: feats.append(align_to_time(time_backbone(x, return_features=True)))
        if mode in {stft_only,dual}: feats.append(align_to_time(stft_encoder(x)))
        return fusion_head(cat(feats, dim=1))
```

**time_backbone 仅跑到编码器出特征**，不跑原解码头（原解码头在双分支下不参与）。

**时间轴对齐 `align_to_time`**：两路特征插值到公共网格 **T_fuse=600（0.3s/格）**，
保留节律时间结构（分层诊断需要），又避免融合头吃 18000 长序列爆显存；融合头再上采样回 18000。

### 4. 三模式 → E1 对照映射

| 标签 | 路径 | 隔离的变量 |
|---|---|---|
| E1a | 现有模型直跑（不经包装器） | 纯时序基线（= E0 代码） |
| E1a' | 包装器 `time_only` | 融合头容量基线（排除容量混淆） |
| E1b | 包装器 `dual` | + STFT 信息（主问题） |
| E1c | 包装器 `stft_only` | STFT 自身任务信息 |
| E1d | 包装器 `dual` × high_hz∈{3,8,12} | 频带上限消融 |

E1d 不是独立批次：它就是 E1b（`dual`）在 high_hz∈{3,8,12} 三档上的展开视角，
与 E1b 共享同一批 run（见“实验编排 / E1 首波”的频带维度），只是从“频带消融”角度命名。

归因逻辑：`E1a' ≈ E1a` → 融合头不贡献容量增益；`E1b > E1a'` → 增益干净归因到 STFT。

## 实验编排

### E0 收尾（本批先跑）

- 跑 `scripts/audit_split_independence.py --config configs/tho_research_v2.yaml`，
  确认 `has_samp_id_leakage=false`、`overlap_samp_id_count=0`，产出审计 CSV 存档。
- 补跑 `downsampled_ssm1d` 在 20260620 soft-z 口径 3 seed（沿用
  `run_20260620_softz_candidates.py` 的 COMMON_OVERRIDES + 方向门控），入候选汇总。

### E1 零号消融（先于主线，6 run）

在代表档对跑 STFT 编码器 `conv1d` vs `conv2d`，定主线默认编码器：

```text
固定 E1b / patch_mixer1d / high_hz=8 / N0
encoder_type ∈ {conv1d, conv2d}
seed ∈ 3 个
→ 2 × 3 = 6 run
```

按主护栏 + 低 SNR 分层选胜者；打平则取更简的 `conv1d`。胜者作为下面主线的默认编码器。

### E1 主线（铺满 3 频带，48 run）

新增 `scripts/run_e1_stft_info_gain.py`，沿用现有候选脚本“笛卡尔积循环 + override”模式，
支持 `--skip` 续跑。维度：

```text
time_backbone ∈ {patch_mixer1d, multiscale_decomp_mixer1d}
标签 ∈ {E1a, E1a', E1b, E1c}
high_hz ∈ {3,8,12}     # E1b/E1c 展开；E1a/E1a' 与频带无关，各算一次
encoder_type = 零号消融胜者（固定）
seed ∈ 3 个
→ 2 × 3 ×（1+1+3+3）= 48 run
```

固定口径：复用“固定实验口径”——数据 seed、方向门控 `auto_direction/max=0.5`、
batch128、loss 权重全程冻结。跑完用 `summarize_tho_runs.py` 产出同口径汇总。

本批 E1 总计 6（零号消融）+ 48（主线）= 54 run，N1 对照另在代表档加挂 3 run。

## 测试策略

按测试规范，聚焦真正有回归风险的契约，避免无价值测试。

**单元测试：**

1. 中间特征开关向后兼容：`PatchMixer1D(x)` 与 `PatchMixer1D(x, return_features=False)`
   输出逐元素相等（默认路径零改动护栏）；`return_features=True` 返回 `(B,C,T')` 形状正确。
   `MultiScaleDecompMixer1D` 同理。
2. STFTEncoder 契约：输入 `(B,1,18000)` → 输出 `(B,Cs,Ts')`；频带裁剪后频点数随
   `high_hz∈{3,8,12}` 单调变化；N0/N1 两路前向均通；含 NaN/全零输入不崩。
3. TimeStftDual1D 三模式：`time_only`/`stft_only`/`dual` 均输出 `(B,1,18000)`；
   各模式只激活应有分支（用参数是否收到梯度间接验证）。

**集成测试：**

4. 用 `tho_small.yaml` 跑 `time_stft_dual1d` 极短训练（几 step），确认能进 engine、
   出 metrics.csv、checkpoint 存取往返一致——不 mock engine。

**不写：** 具体 conv 权重数值、STFT 某频点精确幅值、N1 的 IQR 脆弱数值断言。

## 验收标准

- **E0**：split 审计 CSV 产出且 `has_samp_id_leakage=false`；ssm soft-z 3 seed 跑完并入汇总。
- **E1**：零号消融 6 run 选定编码器，主线 48 run 跑完（共 54），`summarize_tho_runs.py`
  产出同口径汇总；按规划“记录模板”填好 E1a/a'/b/c/d 各项指标 + 分层诊断表。
- **代码**：上述测试全绿；所有现有纯时序模型行为零变化（测试 1 保证）。
- **研究判定**：明确回答“在 high_hz∈{3,8,12} 下，E1b 是否相对 E1a' 改善主护栏
  （`rr_peak_band_abs_error` / `rr_spec_abs_error`）”；若仅分层改善，记录为条件增益方向
  （指向 gated/E5），不算失败。

## 风险与缓解

- **容量混淆**：E1b 比 E1a 多融合头容量 → 用 E1a' (`time_only`) 对照桩隔离。
- **归因混淆**：concat 无收益时分不清“信息没用”还是“融合太笨” → 用 E1c (`stft_only`)
  直接测时频分支自身信息量，绕过融合。
- **归一化引入绝对基准**：与段间 normalize 冲突 → 不减中心、只除鲁棒尺度（N1）或纯 log1p（N0）。
- **跨 samp_id 捷径**：高频/宽频带可能学个体心率捷径 → E0 已确认 samp_id 隔离，
  E1d 以“训练 loss 改善但跨 samp_id 不改善”为捷径判据。
