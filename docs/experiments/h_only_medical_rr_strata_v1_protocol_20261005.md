# CWT-APOR v2 H-only 医学参考呼吸率三区间分析 v1

日期：2026-10-05（Asia/Shanghai）。协议 ID：`h-only-medical-rr-strata-v1-20261005`。

状态：**分析已完成并关闭**，见[结果记录](h_only_medical_rr_strata_v1_results_20261005.md)。用户本次授权 research-test 分析；本协议仅开放已保存 THO 参考波形与 H-only 三 seed 指标的读取和新增分层汇总。执行目录保存计算前的协议副本；本文件仅更新完成状态。

## 分层与统计口径

- 慢区间：参考 Whole RR `<12 bpm`；参考范围区间：`12≤RR≤20 bpm`；快区间：`>20 bpm`。不舍入后分组，不按分位数调阈值。
- 依据：成人静息呼吸率 12–20 bpm、bradypnea <12、tachypnea >20，见 [Nursing Skills 呼吸评估](https://www.ncbi.nlm.nih.gov/books/NBK593192/)；[RCP NEWS2](https://www.rcp.ac.uk/media/a4ibkkbf/news2-final-report_0_0.pdf) 的呼吸率零分区间同为12–20。用于睡眠 THO 谱估计的参考分层，不据此诊断临床异常。
- 分组变量来自同源 `test_reference.npy`，即 `tho_waveform_segment_soft_z`；调用冻结 Whole RR target-only 实现。180秒窗口、100 Hz，经既有 Π 标准化后，汇总60秒谱窗（30秒步长）的中位功率谱，再作呼吸频带谱峰插值估计。不是 RR 相邻变化，也不使用模型预测分组。
- 暂停不单独标记或筛除；所有既有窗口保留。参考 RR 非有限或不可定义时显式失败，不静默删除。
- 全量 research-test 固定2310窗口、8受试者。模型仅 CWT-APOR v2 H-only（arm H），条件载频 `(0.8,8] Hz`、H65；seed 20260811/20260812/20260813，沿用 allowlist 的 validation-selected checkpoint，epoch 8/13/22。
- 各组先逐 seed 计算窗口算术平均，再报告三 seed 均值与 sample SD（ddof=1）。五指标为 Whole RR绝对误差、Local RR MAE、包络轨迹MAE、全局包络调制误差、lag-aware signed PCC；每项保留原 target-only 资格和有效分母。
- 每组报告窗口数、占比、受试者数、逐受试者贡献、参考RR分布及既有质量标签构成；不改变窗口权重。空组显式记录 n=0、结果未定义。
- 结果为既有 research/development test 上的新探索性描述。三 seed 不作为独立人群重复，窗口有时间依赖；不做窗口独立性显著性推断，不从组间误差直接归因生理机制。

## 来源与产物

来源根目录：`/home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction`。

固定 session：`runs/cwt_apor_v2/session_20261001T042244Z_e5e53ad16731`；固定 allowlist：`research_test/allowlist/attempt_20261002T045020Z_d38da46561ce`；参考：`research_test/data/attempt_20261002T045020Z_0be812da89ce`；指标使用三份已完成 H 评价。既有主结果入口为来源仓库的 `docs/experiments/cwt_apor_v2_results_summary_20261002.md`。

入口：`scripts/analyze_h_only_medical_rr_strata.py`。新输出置于本仓库 `runs/h_only_medical_rr_strata_v1/attempt_<UTC时间>_<随机标识>`，保存脚本、协议、输入哈希、checkpoint身份、配置、逐窗分组、逐seed/三seed分层指标、构成表、复算核验与成功/失败回执。历史产物保持只读。

验收：输入 manifest/文件哈希匹配，2310行顺序与subject身份一致，THO参考与三seed同源，参考 RR资格与既有 Whole RR资格一致，五主指标有限，三组分母之和恢复全量分母，按分母加权恢复原每seed全量均值，核对既有 test summary。

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/h-only-medical-rr-strata-v1
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/analyze_h_only_medical_rr_strata.py --confirm-research-test
```

本轮由 Codex 在当次授权范围执行短时 CPU 产物分析，预期数分钟内完成。
