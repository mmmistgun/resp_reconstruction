# E7 P5：完整 validation 汇总与 selected-checkpoint 诊断

日期：2026-09-26。协议 ID：`e7-scale-encoding-aggregation-p5-validation-v1-20260926`。

状态：**P5 汇总、诊断和最终冻结入口已实现；CPU summary 由 Codex 执行，18-cell GPU diagnostics 由用户执行。当前阶段仍不读取 research-test。**

实现修订：前 17 个 diagnostic cell 完成后，`s2_axis_spanning__frequency_attention/20260813` 的有限 FP32 attention 权重出现最大 `|Σw−1|=1.0728836e-6`，略高于旧诊断固定阈值 `1e-6`。诊断入口改用 `8·eps_fp32·ceil(log2(97))` 的归约误差界，并保存实际最大误差与阈值；softmax、模型、checkpoint、数据、指标和诊断公式均未改变。旧 17 项与修订后单元格保持可比，失败 lifecycle 保留。

最终冻结修订：`s0_shallow__mean` 没有适用的 checkpoint 内干预，其三个 `intervention_delta.csv` 合法为零行；早期实现只写出换行符。Finalize 现以固定 schema 读取这些历史零行文件，后续诊断也始终写出列头。该修订不改变任何已计算数值，首次失败 final lifecycle 保留。

## 1. 来源与阶段身份

P5 复用 P4 execution lock `068ba8ec6c5866ebf1b17d560b4fb5f5448e7e921e0dc8b15dda66c2f907c193`，只接受其 18 个唯一成功 formal attempt。P5 不建立新的实验锁；运行 manifest 直接保存 P5 代码提交、源码 SHA、P4 lock identity、正式来源和输出身份。

## 2. CPU summary

Summary 逐项重放 18 份 formal history、early stopping、checkpoint、optimizer、validation metrics、分母和 W0 row/eligibility identity，然后生成：

- 18 行 `per_seed.csv` 与 30 行 `across_seed.csv`；
- 七组条件简单效应及 `C_local/C_span/I_local/I_span/I_endpoint`；
- error 相对 0.5%、PCC 绝对 0.002 的逐 seed 材料性分类；
- 逐 `samp_id` 与等权 subject-macro，以及 subject-macro 简单效应。

所有效用差正值表示改善；不构造跨指标总分。

## 3. 表征诊断

每个 selected checkpoint 使用完整 2,675-window validation。跨尺度方差、encoder residual RMS、attention correction RMS、逐层 residual RMS 和 attention 权重统计覆盖完整激活；`97×97` 尺度相关矩阵与 entropy effective rank 使用每个窗口固定的 12 个通道（索引 `0:96:8`）和 45 个时间点（索引 `0:360:8`），保留全部 97 个尺度。该固定网格约为每窗口 540 个观测，兼顾全窗口覆盖和可执行计算量；不保存完整四维激活。

每个 cell 保存 `x0/xe` 相关矩阵、表征汇总、模块幅度和来源 receipt。诊断 microbatch 固定为 4，不改变模型或正式指标计算。

## 4. Checkpoint 内干预

对适用 arm 在 selected checkpoint 上评价完整 validation：

- `residual_off`；
- `uniform_attention`；
- `both_off`。

`full` 直接引用 formal 已冻结逐窗口指标，其他条件重新完整评价。逐窗口 identity、eligibility 和五主指标沿用正式口径。干预差以正值表示干预改善，只用于判断训练后模块利用，不替代六组从头训练比较。

## 5. 执行命令

CPU summary：

```bash
git status --short  # 必须无输出
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p5.py summarize
```

18-cell diagnostics：

```bash
for E7_SEED in 20260811 20260812 20260813; do
  for E7_ARM in \
    s0_shallow__mean \
    s0_shallow__frequency_attention \
    s1_deep_local__mean \
    s1_deep_local__frequency_attention \
    s2_axis_spanning__mean \
    s2_axis_spanning__frequency_attention; do
    env -u LD_LIBRARY_PATH -u LD_PRELOAD \
      ./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p5.py diagnose \
      --arm "$E7_ARM" --seed "$E7_SEED" --device cuda:0 || break 2
  done
done
```

全部完成后：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p5.py check-diagnostics

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p5.py finalize
```
