# CWT 时频系列：95%显存策略修订与工程验收结果

日期：2026-10-01。协议：`cwt-time-frequency-v1-20260930`。

状态：**合成校准及双GPU完整工程验收通过。** 仅执行合成数据的forward、loss、backward、三步参数更新及条件重放；未读取真实数据，未进行正式训练、validation或research-test。

## 1. 原失败原因和修订

原会话`session_20260930T181903Z_41fd3a1bc6eb`两张卡均在`Q_mu6_v24`（193尺度、batch128）触发`显存超过85%验收门槛`。触发点在三步更新及eval之后，原日志未出现CUDA OOM；旧实现未将超限时的具体峰值写入独立结果，因此不反推原运行的精确占比。

用户明确允许放宽到90%乃至95%，本次将硬上限设为`peak_reserved/total_memory ≤ 0.95`，超过90%单独标记。每个用例先保存allocated/reserved峰值字节、占比、设备总显存及门槛，再判断是否通过；OOM继续显式失败。没有改变模型、batch128、BF16、loss、科学指标或正式训练合同。

显存边界与校准复用范围的6项CPU定向测试通过（5.34 s）。原失败session、日志和产物均保留。

## 2. 校准复用与新来源身份

复用已完成的合成校准：

`runs/cwt_time_frequency_v1/calibration/6747b9d9e0ae2fa2/attempt_20260930T181831Z_29affe4ce73a`。

已核验原manifest、receipt、全部快照与产物；变换规格、依赖环境、源码集合保持一致。仅工程验收模块、后台入口、定向测试、本协议说明四个允许文件发生差异，旧/新字节身份逐项写入新session的`calibration_reuse.source_changes`。未重跑校准，未修改旧来源快照。

新会话：`runs/cwt_time_frequency_v1/session_20260930T182730Z_84618e17624b`。其`execution_scope=synthetic_only`，真实数据/训练/test入口仍被阻止。后台任务`j-vbeb90`退出码为0。

## 3. 验收结果

| 项目 | cuda:0 | cuda:1 |
|---|---:|---:|
| 更新/eval用例 | 80/80通过 | 80/80通过 |
| GN用例（3seed×batch1/128，每例18条件） | 6/6通过 | 6/6通过 |
| 最大reserved占比 | 89.4181% | 89.3788% |
| 最大allocated占比 | 55.1936% | 55.1693% |
| reserved峰值所在臂/batch | P_025/128 | P_025/128 |
| 更新/eval中超过90%的用例 | 0 | 0 |

每卡80个用例包括20臂×3seed的batch1、20臂各一个batch128；每例三次原生训练态更新及eval，检查shape、finite、逐模块梯度和参数更新。另完成B三个seed在FP32/BF16下的非零FiLM等价，以及GN同源重放、固定统计、受保护输入与FULL_FIXED核验。

原失败项`Q_mu6_v24/20260811/batch128`在新会话的reserved峰值分别为89.2424%和89.2032%，两卡均通过。新运行的实际峰值表明90%足够覆盖本轮合成验收；95%作为用户授权的硬上限保留，不据此承诺真实数据训练的显存与耗时。

## 4. 回执

- [后台完成回执](../../runs/cwt_time_frequency_v1/engineering_background/20260930T182729Z_513d27474efe/attempt_20260930T182729Z_ed2e8ca776fa/completion.json)
- [GPU0验收](../../runs/cwt_time_frequency_v1/session_20260930T182730Z_84618e17624b/engineering/cuda_0/attempt_20260930T182749Z_56ffd7f59418/acceptance.json)
- [GPU1验收](../../runs/cwt_time_frequency_v1/session_20260930T182730Z_84618e17624b/engineering/cuda_1/attempt_20260930T182749Z_d8fcfb91bb7f/acceptance.json)
- [新会话与校准复用审计](../../runs/cwt_time_frequency_v1/session_20260930T182730Z_84618e17624b/session.json)

后台及两卡阶段的manifest/receipt已只读核验。后续真实数据阶段仍需落实预处理有效带宽与参数核查，并建立相应范围的新会话；本报告不包含任何重建性能结论。
