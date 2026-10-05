# CWT-APOR v2 FULL重放失败诊断

日期：2026-10-02。会话：`session_20261001T042244Z_e5e53ad16731`。本次只诊断，不修改冻结指标、重放阈值或已完成结果。

## 结论

seed 20260812的失败由BF16下batch128与batch8的数值差异触发。一个近乎等高的双谱峰窗口发生最大峰切换，放大了整段RR指标差异。对失败窗口的受控对照中，同batch8的普通前向与机制捕获前向输出逐点完全相同；batch128重跑与已保存test波形也完全相同。

因此，跨batch的五项指标均值统一要求相对差≤0.1%，不适合作为机制实现正确性的唯一验收。当前证据支持把同batch、同精度、同输入的普通前向与FULL波形比较作为直接检查，并独立记录跨batch的数值敏感性；本次尚未实施这一验收口径修订。

## 已保存结果的检查

首seed 20260811已完成。第二seed 20260812完成全部18条件的逐窗计算，在最后FULL重放验收失败；第三seed未启动。60-cell常规test及其完整汇总已完成。

第二seed的全部2310窗行身份和目标资格一致。FULL/NAT与FULL/FIXED的五项主指标逐窗差值均为0。

| 指标 | 常规test均值 | FULL重放均值 | 相对变化 |
|---|---:|---:|---:|
| Whole RR绝对误差，bpm | 0.7037539493 | 0.7015035265 | −0.319774% |
| Local RR MAE，bpm | 0.6635259695 | 0.6634191472 | −0.016099% |
| Envelope trajectory MAE | 0.1432941779 | 0.1432937385 | −0.000307% |
| Global envelope modulation error | 0.1633030594 | 0.1633433383 | +0.024665% |
| Lag-aware signed PCC | 0.8815809206 | 0.8815860240 | +0.000579% |

只有Whole RR超出`rtol=1e-3, atol=0`。其均值差−0.0022504228 bpm主要来自row 5807（受试者726，test顺序index2225）：单窗误差由5.289089010变为0.056717835 bpm，差−5.232371175 bpm。该单窗除以2310的贡献约−0.00226510 bpm；其余窗口的净贡献约+0.00001467 bpm。不能用任意放宽总体均值阈值来代替逐窗原因检查。

## 固定失败窗口的GPU对照

固定原checkpoint、相同输入及cuda:0；按原数据顺序构造包含失败窗的完整batch128及其对应batch8。

- batch128 BF16普通前向与保存的常规test波形：最大绝对差0。
- batch8 BF16，no_grad与inference_mode：最大绝对差0。
- batch8 BF16普通前向与机制捕获前向：最大绝对差0。
- batch128/8 BF16波形相关系数0.9999936753，相对RMS差0.354173%。
- batch128/8 FP32相对RMS差0.002479%，约为BF16跨batch差异的1/143；两者选中同一谱峰。FP32仅作定位，不替换已完成评价。

用公共指标入口及相同canonicalization复核诊断波形，精确重现已保存的两种Whole RR误差。batch128 BF16中15和21 bpm两个FFT峰功率仅差0.091696%；batch8 BF16中排序翻转，差0.080445%。插值后预测RR分别为15.468148和20.700519 bpm。RR实现采用频带内最大谱峰及邻点插值，因此近似连续的波形变化可导致离散的选峰切换。

## 证据入口

以下路径均相对于正式会话：

- 普通test：`research_test/evaluation/A0/seed_20260812/attempt_20261002T050836Z_6a966a6345e4`。
- 失败机制结果：`research_test/mechanisms/seed_20260812/attempt_20261002T062624Z_a2c4728c941a`。
- 有限GPU对照：`research_test/full_replay_diagnostic/attempt_20261002T073411Z_941a7d6563b5`，保存脚本、固定窗口波形、对照报告及manifest。
- 公共指标口径复核：`research_test/full_replay_canonical_diagnostic/attempt_20261002T073612Z_1b0c8a155796/report.json`。前一诊断报告的RR直接作用于原波形，用于定位；精确对应正式指标的数值使用此处canonicalization后的报告。

后续若修订验收，应保持样本身份、target、资格和同batch的FULL/FIXED检查，补充同batch普通前向/FULL逐窗波形检查；跨batch差异保留为诊断。修订需要明确记录，并独立验证与恢复，不能把原失败attempt直接改记为成功。
