# M4与IEWT：医学参考RR分层与窗口SA research-test 结果

日期：2026-10-07（Asia/Shanghai）。协议：`m4-iewt-strata-v1-20261007`。状态：**完成并关闭**。

**M4在慢和12–20 bpm参考范围区间的五项窗口指标均值均优于IEWT；快区间中，IEWT的两项RR误差较低，M4的包络轨迹、全局调制和PCC较好。IEWT在含SA事件窗口的五项指标均较无SA标注窗口差。**

## 呼吸率三区间

RR由同源THO Whole RR谱估计决定，使用未舍入值按`<12`、`12≤RR≤20`、`>20 bpm`分组。窗口平均，四项误差越低越好、PCC越高越好，RR误差单位bpm。M4（P1-Full-Add）为三个固定训练seed均值±sample SD（ddof=1），checkpoint epoch 25/5/8；IEWT为零相位protocolized Python确定性单次结果，seed SD未定义。

| 模型 | RR区间 | 窗口／人数 | Whole RR误差 | Local RR MAE | 包络轨迹MAE | 全局调制误差 | PCC |
|---|---|---:|---:|---:|---:|---:|---:|
| M4 | 慢：<12 | 1122／7 | 0.7928±0.1519 | 0.7793±0.1374 | 0.1418±0.0090 | 0.1586±0.0069 | 0.8766±0.0068 |
| M4 | 参考范围：12–20 | 1125／8 | 0.3413±0.0092 | 0.4152±0.0175 | 0.1459±0.0091 | 0.1759±0.0099 | 0.8703±0.0061 |
| M4 | 快：>20 | 63／5 | 1.9744±0.5785 | 2.1864±0.0329 | 0.2062±0.0172 | 0.1784±0.0242 | 0.8117±0.0163 |
| IEWT | 慢：<12 | 1122／7 | 1.2902 | 1.2909 | 0.1924 | 0.2192 | 0.7450 |
| IEWT | 参考范围：12–20 | 1125／8 | 0.3692 | 0.5084 | 0.1842 | 0.2118 | 0.8134 |
| IEWT | 快：>20 | 63／5 | 1.1207 | 2.0623 | 0.2391 | 0.2388 | 0.7615 |

快区间的RR排序没有延续慢及参考范围的排序，不能用全量均值概括全部RR区间。快区间59/63窗来自726和670两名受试者；分层比较是当前窗口分布上的描述。完整受试者构成复用[THO医学RR分析](h_only_medical_rr_strata_v1_results_20261005.md)，同时保存于本次`rr_stratum_subject_composition.csv`。

## IEWT窗口SA事件有无

组别使用已冻结PSG事件与180秒窗口的正时长交集，包含三类型暂停与低通气。

| 组别 | 窗口／人数 | Whole RR误差 | Local RR MAE | 包络轨迹MAE | 全局调制误差 | PCC |
|---|---:|---:|---:|---:|---:|---:|
| 含SA事件 | 430／8 | 3.2500 | 3.0285 | 0.3053 | 0.3324 | 0.6821 |
| 无SA标注事件 | 1880／7 | 0.2851 | 0.4511 | 0.1632 | 0.1896 | 0.8009 |
| 全量 | 2310／8 | 0.8370 | 0.9309 | 0.1896 | 0.2162 | 0.7788 |

M4在两个SA组别中的五项窗口均值均优于IEWT，M4具体数值见[原M4 SA报告](p1_m4_sa_presence_v1_results_20261007.md)，本次汇总CSV同时保留两模型的两个轴结果。670全部79窗属于含事件组，占该组18.37%。两组主体构成不同，不能将组间均值差单独归因于SA；“无SA标注事件”按当前事件表定义，不等同于正常人群。

全量M4 Whole/Local RR为0.605136/0.640376，包络/调制为0.145531/0.167532，PCC为0.871763；IEWT全量值恢复原历史summary。每实例、每组五主指标资格分母均等于窗口数。本次没有新增交叉分层或独立人群显著性推断。

## 来源、产物与验证

- RR组别：`/mnt/disk_code/marques/resp_reconstruction/runs/h_only_medical_rr_strata_v1/attempt_20261005T154759Z_17fabd96d171/window_strata.csv`。
- SA组别和M4三seed指标：当前worktree的`runs/p1_m4_sa_presence_v1/attempt_20261007T150849Z_4d5ba53300e8`，通过receipt/manifest核对来源。
- IEWT：`/mnt/disk_code/marques/resp_reconstruction/runs/tho_iewt_research_test/20260807_134559_593049`。原commit为`c9f4cb9466f83afe2b813507a6e9f92435d38b30`，原输入key为`bcg_rawish_wideband_state_aligned_segment_soft_z`。原manifest缺少逐指标文件SHA256，本次绑定当前run_manifest、resolved_config、sample_metrics与summary的哈希，并与原summary数值对账。未声称恢复缺失的历史哈希链或MATLAB逐点等价。

新产物绝对目录：

`/mnt/disk_code/marques/resp_reconstruction/.worktrees/h-only-medical-rr-strata-v1/runs/m4_iewt_strata_v1/attempt_20261007T151448Z_a44c1cb5d98d`

保存逐窗组别和指标、`metrics_per_instance.csv`、`metrics_summary.csv`、两个分层轴的主体构成表、sources/verification、执行前协议和脚本副本、manifest及receipt。基础commit为`e77685f`，新增脚本以SHA256绑定。

验证：1项synthetic CPU测试通过，确认确定性单次实例不产生seed SD，三seed正确计算sample SD并保持窗口权重。2310行及target资格、IEWT input_set/coupling与THO target modulation均和原M4来源一致，IEWT resolved config的数据/输入/THO配置一致；两个轴合计40项全量均值复算与冻结原summary一致，最大绝对差`2.22e-16`；读取输入执行前后哈希一致。分析执行约6秒，未重新推理或读取波形。
