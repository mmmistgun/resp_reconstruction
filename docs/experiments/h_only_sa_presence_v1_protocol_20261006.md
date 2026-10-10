# CWT-APOR v2 H-only：窗口 SA 事件有无分析 v1

日期：2026-10-06（Asia/Shanghai）。协议 ID：`h-only-sa-presence-v1-20261006`。

状态：**完成并关闭**，结果见[完成记录](h_only_sa_presence_v1_results_20261006.md)。用户明确授权按窗口是否包含 SA 事件进行 research-test 分析。本协议开放既存 PSG 标签、时间轴身份与 H-only 指标的短时 CPU 分析。执行目录保存计算前的协议副本，本文件仅更新阶段状态。

## 科学口径

- 模型：CWT-APOR v2 H-only（H65，条件频带 `(0.8,8] Hz`），seed 20260811/20260812/20260813，原 validation-selected checkpoint epoch 8/13/22。
- 固定完整 research-test 集2310个180秒窗口、8名受试者，沿用已有窗口集合和五主指标，不改变资格、指标算法或 checkpoint。
- 标签源：`/mnt/disk_wd/marques_dataset/Resp_Pair_Dataset/HYS/Raw/<samp_id>/<samp_id>_SA Label.csv`，核对其与 `DataCombine2023/HYS/PSG_Aligned/<samp_id>/SA Label_Sync.csv` 哈希相同，并核对配套原 THO 与 Sync/RoughCut THO 字节相同。THO 预处理保持起点，事件和窗口使用现有 THO 秒级轴，不额外平移标签。
- SA事件包括 `Obstructive apnea`、`Central apnea`、`Mixed apnea`、`Hypopnea`。使用标签 `Start/End`；`Duration` 保留为来源信息。非法或未知事件类型、非有限边界、End≤Start显式失败。
- 区间采用 `[start,end)`；事件与窗口正时长相交即为 `sa_present`，仅端点接触不算。其余为 `no_sa_annotation`，表示当前表中未标注相交事件。
- THO覆盖范围以冻结导出 signal-bank JSON 的 `duration_s` 为准。完全在覆盖范围外的事件登记并不参与窗口分类；部分跨界时只取覆盖范围内的交集，并保留原边界及覆盖状态。不静默删改原标签。预检发现286/1006分别有30/28个完全超出末尾的事件。
- 先逐seed计算每组窗口算术平均，再报告三seed均值及sample SD（ddof=1）；五项为 Whole RR绝对误差、Local RR MAE、包络轨迹MAE、全局调制误差和lag-aware signed PCC，各自保留原资格及分母。预检分组为430/1880窗，执行时核对此分母。
- 报告组内窗口数、占比、受试者数与逐受试者贡献。受试者构成不同，组间差异只作描述；窗口有时间依赖，三seed不是独立人群，不做窗口独立性显著性推断。

## 来源、输出与验收

固定来源 session 为 `/home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction/runs/cwt_apor_v2/session_20261001T042244Z_e5e53ad16731`，复用医学RR分层脚本中指定的 allowlist、test rows、三个 H 评价及全量冻结 summary 路径。仅消费保存的指标，不加载预测或执行模型。

入口 `scripts/analyze_h_only_sa_presence.py`；新输出为当前独立 worktree 的 `runs/h_only_sa_presence_v1/attempt_<UTC时间>_<随机标识>`。保存执行脚本及协议副本、环境/代码版本、checkpoint身份、标签与信号来源哈希、完整事件覆盖审计、逐窗口组别、逐seed与三seed指标、主体构成、加权复算与成功/失败回执。历史产物只读。

验收包括：2310窗口身份和顺序固定；标签/THO复制关系逐文件核对；430/1880分组与覆盖审计吻合；三seed样本及资格一致、关键指标有限；两组分母恢复全量，15项逐seed全量均值复算与冻结 summary 一致。

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/h-only-medical-rr-strata-v1
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/analyze_h_only_sa_presence.py --confirm-research-test
```
