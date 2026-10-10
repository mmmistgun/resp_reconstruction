# Patch-aligned TF-Mamba M4：窗口 SA 事件有无分析 v1

日期：2026-10-07（Asia/Shanghai）。协议ID：`p1-m4-sa-presence-v1-20261007`。状态：**完成并关闭**，见[结果记录](p1_m4_sa_presence_v1_results_20261007.md)。用户当次授权按既有格式开展M4的research-test产物分析。执行目录保存计算前的协议副本，本文件仅更新阶段状态。

## 对象和口径

- 对象为P1组件实验的M4（P1-Full-Add）：1秒patch、W-full、对齐cross-attention、零初始化线性residual-add、BiMamba2。固定seed 20260811/20260812/20260813，validation-selected epoch 25/5/8。来源：`/home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction/runs/p1_components_v1/research_test_v1_20261007`。Allowlist SHA256为`8ac8b2305f66cd4f54f6a317f7c1dad9336bd88d393206fa84b5f300ad68fa47`。
- 复用已冻结H-only SA分析的`window_groups.csv`及分组协议：阻塞性、中枢性、混合性暂停和低通气任一事件与180秒窗口正时长相交即为含SA事件；其余为无SA标注事件。标签按THO秒级轴的Start/End判定，仅端点接触不算。
- 固定2310窗/8人，含SA为430窗/8人，无SA标注为1880窗/7人。必须逐行核对M4的row ID、主体、split、THO来源及采样起止与原分组来源一致，跨seed样本与target资格一致。
- 沿用五主指标与原target-only资格：Whole RR误差、Local RR MAE、包络轨迹MAE、全局调制误差、lag-aware signed PCC。每seed组内窗口平均后，报告三seed均值与sample SD（ddof=1）。空组显式未定义，关键指标非有限显式失败。
- 只描述M4本身的含事件/无标注窗口性能；组间差异可能包含主体构成和信号质量差异。三seed不作为独立人群，重叠窗口不作为独立显著性样本。

## 来源与验收

SA来源：当前worktree下`runs/h_only_sa_presence_v1/attempt_20261005T175626Z_3c7db0d6459b`，复用manifest/receipt绑定的分组、逐窗口指标资格与主体构成，不重新读取标签或重跑已完成SA审计。

M4来源是完整P1固定checkpoint research-test，三份成功evaluation receipt绑定metrics、test_rows、evaluation及checkpoint身份。旧汇总位于`summary/attempt_b4769c4f0e724872b110fe20c6db99eb`。本轮只读指标、行元数据及provenance文件，不运行模型或读取波形。

入口：`scripts/analyze_p1_m4_sa_presence.py --confirm-research-test`。新目录`runs/p1_m4_sa_presence_v1/attempt_<UTC时间>_<随机标识>`保存执行前协议/脚本副本、环境、输入身份、checkpoint、逐窗分组指标、逐seed及三seed汇总、主体构成与核验回执。

验收：冻结来源哈希一致，2310行与THO定位一致，430/1880分母一致，三seedtarget资格与原SA指标来源一致；两组加权恢复15项逐seed全量均值，并核对P1原冻结summary。执行结束复核所有读取输入哈希。历史产物保持原身份。

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/h-only-medical-rr-strata-v1
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/analyze_p1_m4_sa_presence.py --confirm-research-test
```
