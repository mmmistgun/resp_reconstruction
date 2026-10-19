# M4 简化基线与后续融合消融

2026-10-19，采用交叉注意力频率 bias 保留、Δt 时间 bias 移除的 M4，作为后续结构消融的基线。模型入口为 `scripts/m4_cross_time_bias_model.py` 的 A1，三个冻结训练 cell 位于 `runs/m4_cross_time_bias_v1/formal_r2_20261018/`。

## 选择依据

当前结构选择以窗口加权的五项主指标为主要依据：绝对均值差小于 0.01 视为无明显变化，同时检查逐 seed 差值。仅移除时间 bias 后，validation/test 的五项窗口加权均值及所有逐 seed 差值均低于该阈值。该结果支持采用较简的时间 bias 配置。

受试者宏平均继续作为诊断保留：test Whole RR 在移除后增加 0.030485 bpm，主要来自受试者 670；完整结果见 [test 与 validation 对照](m4_cross_time_bias_research_test_results_20261019.md)。本项模型选择明确采用窗口加权口径，历史实验按其原定联合判据形成的结果与判断记录保持可追溯。

## 基线结构与身份

- 连续波形 CNN、1 秒 patch/0.5 秒 hop 的时域注意力池化及其相对位置表示。
- W-full 97-scale CWT、每 patch 97×2 个物理对齐时频格、原条件编码器与 GroupNorm。
- 交叉注意力 Q/K/V、频率 bias、attention output 投影及独立 fusion 投影。
- 六层 BiMamba2、局部波形解码及 Hann overlap-add。
- 参数量 1,056,257；seeds=20260811/20260812/20260813，最佳 epoch=25/5/8。
- 训练 session SHA-256：`96975f061d9e1e0e51cbd75d8f0154f383a07389535d6423c0b9853d68c5ee43`。
- validation summary receipt SHA-256：`e3f6206ef63157aaf9ebde106d6c9ffb10f5e5d2c350107e047866c44e1f0458`。
- test summary receipt SHA-256：`662e194c73329c5c32c40846c388e293e7a8b5b294270b36deec5d36d34987e5`。

数据、split、loss、指标、checkpoint selector 沿用原冻结口径。既有 test 已用于研究开发，本次选择属于开发迭代。

## Linear fusion 消融目标

当前条件路径为 `context → attention.output → fusion.projection → residual add`。两个连续 affine 之间没有非线性，函数上可合成为一个 affine；它们的参数化和零初始化位置会影响优化过程。

后续直接比较单输出投影与当前双投影基线：保留交叉注意力输出投影，将其零初始化，随后直接残差相加；其余共有权重与初始化顺序保持配对。新增三个 seed 的 train/validation，与本页冻结基线比较，seed 分配至不同空闲 GPU 并发执行。具体配置、运行身份和验收由专项协议管理。
