# E8：FiLM 条件末端 × 解码端 validation 结果

日期：2026-09-26。状态：**36/36 formal train/validation 与一次性 P5 汇总均已完成并冻结；本文件是当前 validation 结果入口。Research-test 尚未开放。**

## 1. 结论

本轮不支持把设计时优先候选 `e8_direct_temporal` 直接替换为新主模型。更精确的结论是：

1. **删除 `96→65→96` 参数填充、直接投影 FiLM 参数得到边际 Pareto 改善。**跨三个 decoder 等权平均，`direct-fill65` 的 Whole RR 改善 0.6225%、global modulation 改善 0.8642%，Local RR、trajectory 与 PCC 均在预设容差内。因此 65-channel 宽度没有获得保留证据。
2. **`96→96→96` 和 `96→192→96` 没有形成普遍容量收益。**`res96-direct` 同时改善 global modulation、恶化 Whole RR；`res192-res96` 改善 PCC，但恶化 trajectory 与 global modulation。两者都是属性交换。
3. **局部时间残差 decoder 不是默认赢家。**相对 pointwise residual，temporal residual 边际上改善 global modulation 1.7271%，但 Whole RR 与 Local RR 分别恶化 2.5194% 和 0.7990%。相对 single readout 仍是同方向交换。
4. **设计时推荐的 `direct+temporal` 未形成 Pareto 改善。**相对同轮 W0，Whole RR 恶化 1.3245%、trajectory 恶化 1.1005%、global modulation 改善 1.8284%，Local RR/PCC 在容差内。
5. **最值得继续确认的简化候选是 `direct+single`，但尚不足以直接替换 W0。**窗口直接平均中，它相对 W0 的 Local RR 改善 1.2816%、global modulation 改善 0.6813%，Whole RR、trajectory 与 PCC 在容差内；Local RR 尾部也小幅改善，并减少 13,633 个参数。但 subject-macro PCC 从 `0.853561` 降至 `0.850887`，差 `-0.002674`，超过 PCC 材料性阈值，而且 Whole RR 跨 seed SD 明显增大。因此证据存在聚合层级与优化稳定性风险。
6. **不存在唯一跨属性赢家。**Tolerance-aware Pareto 集包含 7 个 arm；W0 同轮参照不在集合中，但不同非支配 arm 分别占优于 RR、形态、全局调制或 PCC，不能用单项最好值宣称全面优越。

因此，当前结构判断为：**条件分支末端优先采用 direct；不保留 temporal residual decoder 作为默认改进；若继续确认更干净模型，优先考察 `direct+single`，同时保留 subject-macro PCC 与 seed 稳定性风险。**在新的跨 split 证据前，论文主模型不自动改写。

## 2. 冻结身份与范围

- Formal execution lock：`40b30fc6ecdcc9b40750c393be8fd2faf4e223da0566432df739aacb513713c5`。
- Runtime amendment：`517bdf281a0db0902e0c10481b0ea210a7c514d627eb6397e9e982d95ba270e3`。
- P5 summary 源码 commit：`7bde32d7d987cc861523adc132306f78f85e860c`。
- Formal：12 arms × 3 seeds = 36 个唯一成功 attempt；1 个训练前 GPU 容量门控失败 lifecycle 保留。
- Validation：每个 attempt 2,675 行、7 个 `samp_id`；总计 96,300 行。
- 训练合计：1,152 epochs、92,160 optimizer updates；每 run 30–41 epochs，selected epoch 为 5–26。
- 逐 run wall-time 求和约 21.40 h；该值不表示并行调度后的端到端时间。
- P5 只读取 validation 产物并回放 72 个 best/final checkpoint；没有训练、推理、GPU 或 research-test 访问。

冻结 P5 路径：

```text
runs/e8_film_decoder_redesign_v1/summary/
summary_40b30fc6ecdc_20260926T081641Z_d1cb238ad1b5
```

- Manifest SHA-256：`3d9ad92fa42df7d2f0e9e66a20a134b4328e62d3e3940c04c15bdafa7805871f`。
- Decision SHA-256：`1743548f3894cae2238827016cce0fef1fe9fc50ef46b48b3a1ca65794388b51`。
- Summary receipt SHA-256：`44f2279ddc10398f89bec91e2f1d477d488b501921b8600b7c5771202a9a94af`。

## 3. 五主指标

表中为三个 seed 的 window-direct arithmetic mean ± sample SD。前四项越低越好，PCC 越高越好。

| Arm | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| `fill65/pointwise` | 0.500708 ± 0.018403 | 0.553073 ± 0.012575 | 0.152950 ± 0.001474 | 0.191168 ± 0.002073 | 0.865227 ± 0.001423 |
| `fill65/single` | 0.499903 ± 0.042469 | 0.546504 ± 0.020599 | 0.154230 ± 0.005679 | 0.197416 ± 0.008948 | 0.861115 ± 0.007843 |
| `fill65/temporal` | 0.511270 ± 0.011961 | 0.549440 ± 0.009504 | 0.154074 ± 0.003002 | 0.184812 ± 0.005409 | 0.864675 ± 0.003607 |
| `direct/pointwise` | 0.492749 ± 0.031169 | 0.545727 ± 0.021832 | 0.152145 ± 0.005044 | 0.190626 ± 0.006415 | 0.861968 ± 0.004021 |
| `direct/single` | 0.501230 ± 0.046776 | 0.546087 ± 0.019175 | 0.152739 ± 0.002380 | 0.189877 ± 0.005728 | 0.864296 ± 0.002161 |
| `direct/temporal` | 0.507564 ± 0.035248 | 0.553135 ± 0.012057 | 0.154648 ± 0.003857 | 0.187705 ± 0.007121 | 0.864342 ± 0.001452 |
| `res96/pointwise` | 0.498948 ± 0.025619 | 0.545092 ± 0.009639 | 0.154157 ± 0.000559 | 0.185980 ± 0.002262 | 0.863997 ± 0.003827 |
| `res96/single` | 0.503063 ± 0.023026 | 0.550601 ± 0.015394 | 0.152319 ± 0.005915 | 0.186588 ± 0.002870 | 0.862536 ± 0.002135 |
| `res96/temporal` | 0.502455 ± 0.012038 | 0.549822 ± 0.015422 | 0.152380 ± 0.005369 | 0.188797 ± 0.001482 | 0.863411 ± 0.002638 |
| `res192/pointwise` | 0.490762 ± 0.014062 | 0.544298 ± 0.011659 | 0.154646 ± 0.004454 | 0.193695 ± 0.007865 | 0.865859 ± 0.001329 |
| `res192/single` | 0.505200 ± 0.019035 | 0.547316 ± 0.015722 | 0.156440 ± 0.004368 | 0.197212 ± 0.007285 | 0.864912 ± 0.003189 |
| `res192/temporal` | 0.511462 ± 0.007782 | 0.553018 ± 0.006517 | 0.153017 ± 0.005156 | 0.186996 ± 0.001026 | 0.865418 ± 0.000887 |

单项 seed mean 最优分别为：Whole RR、Local RR、PCC=`res192/pointwise`；trajectory=`direct/pointwise`；global modulation=`fill65/temporal`。这些最优值来自不同 arm，正是属性交换而非单一结构全面优越。

## 4. 计划边际对比

表中为方向统一后的跨 seed mean；error 使用相对改善比例，PCC 使用绝对差。正值表示候选改善。

| Contrast | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| `direct-fill65` | +0.6225% | +0.2550% | +0.3651% | +0.8642% | -0.000137 |
| `res96-direct` | -0.5049% | -0.0535% | +0.1476% | +1.1378% | -0.000221 |
| `res192-res96` | -0.1902% | +0.0399% | -1.2124% | -2.9513% | +0.002081 |
| `single-pointwise` | -1.3163% | -0.1006% | -0.3020% | -1.2615% | -0.001048 |
| `temporal-pointwise` | -2.5194% | -0.7990% | -0.0344% | +1.7271% | +0.000199 |
| `temporal-single` | -1.2515% | -0.7194% | +0.2638% | +2.9299% | +0.001247 |

`direct-fill65` 是唯一边际 tolerance-aware Pareto 改善。其余对比要么存在材料性恶化，要么是明确属性交换。多个 interaction 在五项指标上达到材料性阈值，说明 decoder 作用依赖条件末端水平；边际结论不能替代具体 arm 的比较。

## 5. 简洁候选与尾部/subject 复核

`direct/single` 相对 `fill65/pointwise`：

- 参数：1,219,850 → 1,206,217，减少 13,633（1.118%）。
- 本协议覆盖的变化模块 MAC：24,364,800 → 0；这不是完整模型 FLOPs。
- Window-direct：Whole RR -0.1195%（恶化、容差内），Local RR +1.2816%，trajectory +0.1414%，global modulation +0.6813%，PCC -0.000931（容差内）。
- Local RR tail：median `0.113292→0.112687`，P90 `1.600846→1.552491`，P95 `2.384233→2.378046`，`>2 bpm` `6.9283%→6.5794%`，`>5 bpm` `0.9470%→0.9097%`。
- Subject-macro：Whole RR `0.624832→0.618022`，Local RR `0.649613→0.638573`，global modulation `0.223289→0.221199`；但 PCC `0.853561→0.850887`，差 `-0.002674`。
- Benchmark train throughput `226.217→223.209 samples/s`，没有形成可重复组支持的速度收益；不能把参数减少写成实测加速。
- Whole RR seed SD `0.018403→0.046776`，优化随机性明显增加。

因此它是最清楚的简化候选，而不是已确认的新主模型。

## 6. Pareto 与单项最优边界

Tolerance-aware Pareto 集为：

```text
fill65/temporal
direct/pointwise
direct/single
res96/pointwise
res96/single
res192/pointwise
res192/temporal
```

同轮 W0 `fill65/pointwise` 被 `direct/single` 与 `res96/temporal` 支配；但 Pareto 集仍有 7 个 arm，且 `direct/single` 的 subject-macro PCC 与 seed 稳定性提出反向证据。Validation 不支持选择唯一全面赢家。

## 7. 工程风险

P2 最大 synthetic training reserved fraction 为 `0.636628`。Formal 的 `runtime_summary.json` 显示所有 36 个成功 run 的峰值 reserved fraction 为约 `0.8136–0.8165`，最高 `0.816497`；峰值 allocated 为 8,745–8,802 MiB（约 8.54–8.60 GiB），全部运行成功且没有 OOM/非有限量。

该差异很可能来自 P2 只用 batch-1 eval、batch-128 train 和 batch-32 native validation，没有覆盖正式 batch-128 validation 的 allocator reserved 峰值。它不改变已完成结果的数值有效性，但说明 80% P2 guard 没有覆盖完整 formal 生命周期。未来复用该结构时应在新协议中增加 batch-128 validation 验收或显式定义 eval chunk；不得回改本轮 identity 或重跑 36-cell。

## 8. 后续边界

P5 已关闭，以上 formal 与 summary 不重复执行。若用户希望查看跨 split 行为，应另立 research-test 专项协议，固定全部 36 个 validation-selected checkpoints 并完整评价，避免根据 validation 结果只挑有利 arm。该 test 只能作为 reused research/development evidence，不构成新的独立 held-out 确认，也不得用于回改 selector、阈值或本轮 validation 结论。
