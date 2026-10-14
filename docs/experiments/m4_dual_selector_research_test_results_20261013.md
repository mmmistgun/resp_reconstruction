# M4/SP双选点research-test结果

协议：`m4-dual-selector-research-test-v1-20261013`。12个固定checkpoint在GPU1完成评价，完整汇总已完成并关闭。两臂×三seed×rr/loss两个selector均来自本轮重训，未混入上一轮test结果。执行合同见[专项协议](m4_dual_selector_research_test_v1_20261013.md)。

## 核验与来源

- 每组2310窗口/8受试者，跨selector/arm/seed的样本身份和目标资格集合一致。
- 12份evaluation、checkpoint/config/completion和汇总文件哈希复核通过，正式评价失败记录为0；joint及envelope Spearman prediction-degenerate计数均为0。
- 训练session：`runs/m4_dual_selector_v1/formal_v1_20261013`。
- test根目录：`runs/m4_dual_selector_v1/research_test_v1_20261013`。
- allowlist SHA-256：`5605eb5a73e2b87fbafe4da069cfda0cf976c245447d295a43e469318a91540f`。
- 汇总：`summary/attempt_9e66f7a2cf2b47eb9862d2d3713cf5d2`。
- summary receipt SHA-256：`5701811999b50770f7886cc2e6c3d99a92781683b711ced3da870dced0907e8b`。

## 20260811：最低loss相对最低RR

| 模型/选点 | epoch | Local RR MAE↓ | Whole RR AE↓ | 包络轨迹MAE↓ | 全局包络调制误差↓ | PCC↑ |
|---|---:|---:|---:|---:|---:|---:|
| M4 / RR | 25 | 0.589329 | 0.544377 | 0.154793 | 0.176028 | 0.864467 |
| M4 / loss | 8 | 0.660885 | 0.629790 | 0.142920 | 0.166155 | 0.871504 |
| SP / RR | 4 | 0.802164 | 0.784452 | 0.134704 | 0.167827 | 0.875209 |
| SP / loss | 8 | 0.656525 | 0.672502 | 0.141966 | 0.166882 | 0.872004 |

SP的Local RR下降0.145640 bpm（18.16%），Whole RR下降0.111950 bpm（14.27%）。包络轨迹MAE升高0.007262，PCC下降0.003205，全局包络调制误差略降。原始M4改用最低loss后，两个RR误差反而分别增加0.071556和0.085413 bpm。

SP的受试者671 Local RR为1.435324→0.782516 bpm，Whole RR为1.105709→0.543552 bpm；670 Local RR为7.232697→7.253085，Whole RR为8.912897→9.376378。因而主要缓解的是671上的RR问题，670仍未改善。相同epoch 8下，两臂Local RR接近（M4 0.660885、SP 0.656525），SP Whole RR仍较高（0.672502对0.629790）。

## 三seed完整比较

以下为三seed均值±样本标准差（ddof=1）。

| 指标 | M4 / RR | SP / RR | M4 / loss | SP / loss |
|---|---:|---:|---:|---:|
| Local RR MAE↓ | 0.64372±0.06083 | 0.71002±0.08690 | 0.63440±0.02514 | 0.62221±0.03134 |
| Whole RR AE↓ | 0.60731±0.05460 | 0.68242±0.09048 | 0.63200±0.01996 | 0.63393±0.03580 |
| 包络轨迹MAE↓ | 0.14584±0.00947 | 0.13926±0.00694 | 0.14296±0.00326 | 0.14265±0.00433 |
| 全局包络调制误差↓ | 0.16803±0.00693 | 0.16665±0.00113 | 0.16587±0.00861 | 0.16573±0.00989 |
| PCC↑ | 0.87189±0.00678 | 0.87498±0.00107 | 0.87489±0.00531 | 0.87434±0.00431 |

SP按seed20260811/12/13的Local RR为0.802164→0.656525、0.698346→0.595099、0.629535→0.615013，三组均改善；配对差均值−0.087803、样本SD0.066909 bpm。Whole RR为0.784452→0.672502、0.650872→0.601766、0.611946→0.627510，2/3改善，配对差均值−0.048497、SD0.063759 bpm。

两臂均按最低loss比较时，SP−M4的Local RR差值−0.012185±0.006785 bpm，3/3seed改善；Whole RR为+0.001930±0.036004，2/3改善，均值接近。包络轨迹差−0.000316、全局调制差−0.000143、PCC差−0.000553，差距较小且方向混合。

## 辅助指标与解释

三seed均值：

| 模型/选点 | IBI MedAE↓(s) | IBI coverage↑ | Coherence↑ | constrained nDTW↓ |
|---|---:|---:|---:|---:|
| M4 / RR | 0.094997 | 0.802339 | 0.573563 | 0.213739 |
| M4 / loss | 0.092712 | 0.797864 | 0.578981 | 0.210553 |
| SP / RR | 0.093168 | 0.803130 | 0.578663 | 0.209108 |
| SP / loss | 0.092545 | 0.794860 | 0.578709 | 0.210639 |

IBI MedAE的资格窗口集合依赖匹配和coverage，按原生分母解释；完整资格、分层Spearman、lag和tail见汇总CSV。参数量未变：M4 1,056,357，SP 1,047,045。

本轮支持“SP的RR对选点有实质敏感性，最低loss在这三个seed上降低Local RR”；同时存在包络/PCC取舍，原始M4也不因换loss选点而全面改善。20260811的选点差异确实影响了此前对比，但不足以将全部差异归因为epoch，或认定原结果是训练错误。本实验已受既有research-test观察影响，属于重复使用的研究开发证据；不据此按每seed的test结果择优混合selector，也不作统计显著性或全新独立测试的泛化断言。

逐seed与配对：`rr_per_seed.csv`、`loss_per_seed.csv`、`selector_paired_delta.csv`、`selector_paired_across_seed.csv`、`seed20260811_selector_delta.csv`。各selector的`paired_across_seed.csv`保存SP−M4差值。`all_per_subject.csv`、原生summary、tail、denominators及`sources.json`保留完整分层与来源。
