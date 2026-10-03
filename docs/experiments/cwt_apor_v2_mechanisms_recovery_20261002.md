# CWT-APOR v2机制导出接口修复与恢复

日期：2026-10-02。适用会话：`session_20261001T042244Z_e5e53ad16731`。用户此前授权“开启test”，本次要求检查机制分析错误；本恢复仅完成该范围内的三个seed机制评价和汇总。

60-cell常规test与汇总已完成，来源为`research_test/summary/attempt_20261002T054624Z_e22f113cfb30`。首个机制seed 20260811的`attempt_20261002T054646Z_fba49c6b285a`在首批FULL预测转指标字典时失败：`_prediction_dict_from_arrays`要求显式的`pred_key`和`target_key`，调用未传入。

恢复入口`scripts/recover_cwt_apor_v2_mechanisms.py`仅在机制导出作用域内将上述两个参数绑定为公共指标入口既有字段`r_tho_hat`、`tho_ref`。模型前向、干预、GN、目标、指标公式、checkpoint、样本集合、batch8、FULL重放阈值均沿用原实现。

原科学源码及session快照保持其冻结身份。恢复脚本以显式增量适配方式执行；每个新机制attempt在计算前保存修复脚本、定向测试、本附件的副本和SHA256，并在`recovery_source.json`绑定原session。原失败attempt和原pipeline失败状态保留，首seed使用新的retry attempt，其余seed首次执行；已完成常规test与汇总直接引用。

定向验证使用CPU合成fixture，覆盖原错误复现、字段与公共构造器逐项一致、异常退出后函数引用恢复、18条件指标、末尾不足batch、案例NPZ/JSON和GN统计导出。修复通过后沿用已授权GPU阶段运行完整机制矩阵，并保留原有FULL重放核对。恢复结果以`research_test/mechanisms_recovery`及各阶段回执为准。

验证结果：`tests/test_cwt_apor_v2_mechanisms_recovery.py`两项测试通过，耗时26.52秒；仅使用合成数据。
