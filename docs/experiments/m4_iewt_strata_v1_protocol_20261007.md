# M4与IEWT：医学参考RR分层与窗口SA分析 v1

日期：2026-10-07（Asia/Shanghai）。协议ID：`m4-iewt-strata-v1-20261007`。状态：**完成并关闭**，见[结果记录](m4_iewt_strata_v1_results_20261007.md)。用户授权补齐M4呼吸率分层，并计算IEWT同格式结果。执行目录保存计算前协议副本，本文件仅更新阶段状态。

- M4为P1-Full-Add，复用已完成M4 SA分析绑定的三seed逐窗口指标，seed 20260811/20260812/20260813、checkpoint epoch 25/5/8。
- IEWT为`protocolized_python_iewt`零相位确定性单次基线，来源为主仓库`runs/tho_iewt_research_test/20260807_134559_593049`。只报告单次窗口均值；seed SD未定义。该实现没有声称MATLAB逐点等价。
- 固定research-test2310窗/8人，输入与THO目标沿用既有身份。RR组别复用THO Whole RR `<12`、`12–20`（含端点）、`>20 bpm`，分母1122/1125/63。SA组别复用暂停三类型及低通气事件与窗口正时长相交的规则，分母430/1880。
- RR及SA两个轴分别汇总，不交叉细分。M4和IEWT均保存两轴结果；M4 SA作为本次来源一致性复核，既有结果不覆盖。
- 保持五主指标和各自target-only资格；每实例组内窗口平均。M4再汇总三seed均值/sample SD（ddof=1）；IEWT单次均值与n。暂停不额外筛除，RR分组采用已有参考谱估计。
- 复用冻结窗口组别；核对row ID/主体/split/顺序和资格，并核对IEWT原指标中的input_set、coupling_state_id、target modulation与原M4 test指标一致；resolved config核对数据/输入/THO来源口径。关键指标非有限显式失败。
- 原IEWT run manifest不包含逐指标文件哈希，本次记录当前四份来源文件哈希并与原summary逐项对账，不声称重新恢复缺失的历史哈希链。
- 两轴分组有效分母均须恢复全量，按分母加权恢复每实例的五项全量均值。执行结束复核输入文件哈希。
- 组间窗口平均差异为探索性描述，可能受受试者构成和信号质量影响；三seed不扩大独立人群分母。

入口：`scripts/analyze_m4_iewt_strata.py --confirm-research-test`。新输出使用当前独立worktree的`runs/m4_iewt_strata_v1/attempt_<UTC时间>_<随机标识>`，保存脚本/协议副本、环境、来源身份、窗口组别、逐实例与汇总指标、主体构成、核验、manifest与receipt。

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/h-only-medical-rr-strata-v1
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/analyze_m4_iewt_strata.py --confirm-research-test
```
