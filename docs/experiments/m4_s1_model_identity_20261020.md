# M4-S1 实验代号与模型身份

2026-10-20 记录：用户倾向采用单投影结构，当前实验代号为 **M4-S1**，其中 **S1 = Single Projection（单投影）**。后续讨论和实验说明使用该代号指代下述配置。

## 对应配置

- 对应已完成的 M4 linear fusion 消融 **L1（M4-SingleProjection）**。
- 交叉注意力保留频率 bias，移除 Δt 时间 bias。
- 条件路径为 `attention.output → residual add`，保留一个输出投影并直接残差相加；训练初始化时该输出投影为零。
- 保持片段时频物理对齐、时域注意力池化及其位置表示、W-full CWT、BiMamba2 与波形解码结构。
- 参数量：**1,046,945**。
- 模型实现：`scripts/m4_linear_fusion_model.py` 中的 `LinearFusionModel`，`arm="L1"`。

## 实验来源

- 训练 session：`runs/m4_linear_fusion_v1/formal_v1_20261019/`。
- seeds：20260811、20260812、20260813；validation 选出的最佳 epoch：4、5、8。
- research-test：`runs/m4_linear_fusion_v1/research_test_v1_20261019/`。
- [Train/validation 结果](m4_linear_fusion_v1_validation_results_20261019.md)。
- [完整三 seed research-test 与双投影对照](m4_linear_fusion_research_test_results_20261019.md)。

本记录确认实验代号及用户的单投影研发偏好。性能判断引用完整三 seed 结果，包含 seed 20260811；历史产物沿用原有 L1 标签、路径和冻结身份。后续新增实验以 M4-S1 标识模型配置，并单独定义实验版本与输出 identity。
