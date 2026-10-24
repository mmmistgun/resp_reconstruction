# M4-S1 validation总loss选点：完整research-test及逐项对照

状态：2026-10-24，完整84项（82项新增评价＋2项同checkpoint复用）及自动汇总已完成，逐项来源与数值核验通过；对照表浮点往返造成的持平计数已在独立核验产物中修订。本阶段关闭，默认模型保持不变。

## 主要发现

- **总loss选点在本轮test上普遍降低Local RR误差，但并非所有配置获益。** 28配置中24个三seed均值改善达到0.01 bpm，3个不足0.01，NoMamba退步。84个同cell配对中58个改善、24个退步、2个相同checkpoint持平；54个改善、18个退步达到0.01。
- **基准B的Local RR从0.648923降至0.613764 bpm，Whole RR从0.651176降至0.630394。** 两项均为2个seed改善、1个退步；Local RR差的SD为0.057376，不能表述成三个seed一致改善。
- **选点规则改变了部分结构判断。** 新规则下NoTF相对B的Local RR优势缩小至0.009921；NoFrequencyBias的Whole RR由原先改善转为退步；NoPoolPosition与ScorePositionOnly的Local RR均比新B低约0.034，三个seed一致。
- **BiMamba的作用保持稳健。** loss选点后NoMamba对新B的Whole/Local RR分别差+0.121597/+0.103290，PCC差−0.023128，三个seed一致；但NoMamba自身换为loss选点后，global modulation error改善而RR退步，说明选点目标存在指标取舍。

## 核验与修订

82项新增评价及2项复用均成功，两个worker退出码为0，自动汇总于北京时间2026-10-24 17:10:36完成。复用项为 `A_UniformPatchPool_seed20260812` 和 `A_NoTF_seed20260813`，checkpoint、预测和逐窗指标来源完全相同。
全部84项均为2310窗/8人，五项主指标有效n均为2310。已核验冻结runtime及新入口身份、完整epoch记录中的最小总loss/最早并列、选中权重哈希、配置、原始数据来源、20种缓存与前次字节一致性、预测形状/有限性、两种选点的逐窗顺序与target资格。
17张原生test统计表通过逐窗复算。自动选点对照中，CSV浮点往返使两个复用cell的部分delta落在约10⁻¹⁶量级，造成9个“配置×指标×聚合”方向计数组需要修订。独立核验对照从两侧逐窗CSV按相同路径计算，复用项五指标精确持平；所有mean/SD在浮点容差内一致，0.01分类及达阈值方向数完全不变。原自动汇总和首次审计失败日志保留，详情见[精度修订](m4_s1_loss_selected_test_precision_note_20261024.md)。
逐项对照入口：[84行完整cell对照CSV](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/cell_comparison_84.csv)，包含两种epoch、validation总loss/Local RR、全部五项test原值与差；[逐seed五指标配对](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/selector_paired_verified.csv)和[三seed汇总](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/selector_summary_verified.csv)。

## 选点含义与validation中的取舍

每项在原训练已经完成的epoch中选择 `validation.loss` 最低的checkpoint，并列取最早。原轨迹按validation Local RR早停；本轮未重训，因此结论限定于既有轨迹，不能外推成“按总loss早停重新训练”的结果。82项更换epoch、2项相同；82项的validation总loss均下降，validation Local RR均上升。这是不同选点目标的样本内取舍，本身不能证明过拟合。
loss选点epoch分布为6×28、7×50，8/9/12/14/15/21各1项。基准B的具体变化如下：

| seed | 原epoch→loss epoch | 原val loss→新val loss | 原val Local RR→新val Local RR |
|---|---|---|---|
| 20260812 | 5→7 | 0.241706→0.231071 | 0.520067→0.563910 |
| 20260813 | 8→6 | 0.230746→0.225977 | 0.501677→0.547879 |
| 20260814 | 10→7 | 0.243874→0.235166 | 0.541416→0.565209 |

validation总loss在84项中的平均变化为−0.007518，validation Local RR平均变化为+0.033285 bpm；这里的跨配置平均仅描述本矩阵。test上的Local RR却在多数配置下降，说明validation Local RR的最优排序未稳定外推到本次研究test。
以下delta均为“loss选点test−原Local RR选点test”。误差负值/PCC正值表示改善；SD为逐seed配对差的样本SD；0.01使用各指标原生单位。28配置共享人群，三个训练seed也共享test人群，计数不是独立样本的显著性检验。test仍为既有research/development evidence。

## 基准B五指标对照

| 指标 | 原Local RR选点 | loss选点 | 配对Δ ± SD | 改善/退步/持平seed |
|---|---|---|---|---|
| Whole RR AE | 0.651176 ± 0.035037 | 0.630394 ± 0.041325 | -0.020782 ± 0.031759 | 2/1/0 |
| Local RR MAE | 0.648923 ± 0.038627 | 0.613764 ± 0.022071 | -0.035159 ± 0.057376 | 2/1/0 |
| Envelope MAE | 0.142644 ± 0.006096 | 0.143248 ± 0.003835 | +0.000604 ± 0.009794 | 2/1/0 |
| Global modulation error | 0.169939 ± 0.006211 | 0.169760 ± 0.011767 | -0.000179 ± 0.010608 | 1/2/0 |
| Signed PCC | 0.873832 ± 0.001464 | 0.873765 ± 0.005221 | -0.000067 ± 0.003831 | 1/2/0 |

B的test Local RR在seed20260812/13/14上的变化分别为−0.098478、−0.020383、+0.013384；Whole RR变化为−0.056407、+0.004565、−0.010505。宏平均Whole RR改善−0.079405，三个seed一致；宏平均Local RR改善−0.067599，2好1差。

## 28配置的逐项选点对照

完整五指标的140项配置均值比较，以及420项逐seed比较，见上述核验表；下面列出全部28配置的两项RR。

| 配置 | Local RR原→loss | Local Δ ± SD | Local好/差/平 | Whole RR原→loss | Whole Δ ± SD |
|---|---|---|---|---|---|
| Q_mu6_v4 | 0.667041→0.638279 | -0.028762 ± 0.046280 | 2/1/0 | 0.664864→0.655356 | -0.009507 ± 0.034879 |
| Q_mu6_v8 | 0.642637→0.632943 | -0.009694 ± 0.054568 | 1/2/0 | 0.650490→0.642928 | -0.007562 ± 0.038339 |
| Q_mu6_v12 | 0.640412→0.634122 | -0.006289 ± 0.051858 | 1/2/0 | 0.654169→0.641209 | -0.012959 ± 0.051890 |
| Q_mu6_v24 | 0.659806→0.633461 | -0.026346 ± 0.037504 | 2/1/0 | 0.650514→0.636744 | -0.013770 ± 0.025915 |
| Q_mu13p4_v4 | 0.655572→0.617613 | -0.037959 ± 0.057227 | 2/1/0 | 0.665051→0.654607 | -0.010444 ± 0.051748 |
| Q_mu13p4_v8 | 0.648830→0.622214 | -0.026616 ± 0.054818 | 2/1/0 | 0.657780→0.641732 | -0.016048 ± 0.046071 |
| B | 0.648923→0.613764 | -0.035159 ± 0.057376 | 2/1/0 | 0.651176→0.630394 | -0.020782 ± 0.031759 |
| Q_mu13p4_v24 | 0.654127→0.613269 | -0.040858 ± 0.038774 | 3/0/0 | 0.648620→0.633814 | -0.014806 ± 0.017555 |
| Q_mu20_v4 | 0.630197→0.606730 | -0.023467 ± 0.038477 | 2/1/0 | 0.639680→0.631583 | -0.008097 ± 0.055148 |
| Q_mu20_v8 | 0.652079→0.613896 | -0.038183 ± 0.025183 | 3/0/0 | 0.650548→0.627552 | -0.022995 ± 0.047289 |
| Q_mu20_v12 | 0.655297→0.611013 | -0.044284 ± 0.019844 | 3/0/0 | 0.647454→0.626861 | -0.020593 ± 0.045593 |
| Q_mu20_v24 | 0.654366→0.609571 | -0.044795 ± 0.017203 | 3/0/0 | 0.645465→0.630227 | -0.015238 ± 0.048447 |
| P_025 | 0.663740→0.623097 | -0.040644 ± 0.041344 | 3/0/0 | 0.654442→0.637058 | -0.017384 ± 0.060076 |
| P_100 | 0.636404→0.613125 | -0.023278 ± 0.053423 | 2/1/0 | 0.632290→0.642568 | +0.010278 ± 0.024410 |
| C_0p8 | 0.631206→0.611307 | -0.019899 ± 0.032968 | 3/0/0 | 0.618231→0.615109 | -0.003122 ± 0.037901 |
| C_2 | 0.643630→0.628123 | -0.015507 ± 0.037591 | 2/1/0 | 0.640357→0.661478 | +0.021121 ± 0.049015 |
| C_4 | 0.660462→0.633883 | -0.026578 ± 0.047994 | 2/1/0 | 0.643295→0.660332 | +0.017037 ± 0.062928 |
| C_12 | 0.650718→0.612783 | -0.037935 ± 0.044488 | 2/1/0 | 0.635494→0.637616 | +0.002122 ± 0.042668 |
| C_20 | 0.641575→0.613304 | -0.028270 ± 0.051704 | 2/1/0 | 0.627769→0.644077 | +0.016308 ± 0.069462 |
| H | 0.634307→0.609145 | -0.025162 ± 0.043684 | 2/1/0 | 0.621401→0.623206 | +0.001805 ± 0.046462 |
| A_NoTF | 0.603656→0.603843 | +0.000187 ± 0.000320 | 0/2/1 | 0.632316→0.593649 | -0.038667 ± 0.045991 |
| A_UniformTFReadout | 0.646211→0.613606 | -0.032605 ± 0.047646 | 2/1/0 | 0.645717→0.635626 | -0.010091 ± 0.047790 |
| A_NoFrequencyBias | 0.654557→0.617941 | -0.036615 ± 0.038882 | 3/0/0 | 0.623379→0.656683 | +0.033303 ± 0.061949 |
| A_NoMamba | 0.683827→0.717053 | +0.033226 ± 0.006239 | 0/3/0 | 0.713877→0.751990 | +0.038113 ± 0.034803 |
| A_UniformPatchPool | 0.728363→0.644958 | -0.083405 ± 0.096621 | 2/0/1 | 0.697449→0.643918 | -0.053532 ± 0.069738 |
| A_NoPoolPosition | 0.652440→0.579242 | -0.073198 ± 0.046783 | 3/0/0 | 0.666485→0.595326 | -0.071159 ± 0.066506 |
| A_ScorePositionOnly | 0.635199→0.579926 | -0.055273 ± 0.082251 | 2/1/0 | 0.639664→0.603677 | -0.035987 ± 0.101277 |
| A_ValuePositionOnly | 0.685263→0.638524 | -0.046739 ± 0.082627 | 2/1/0 | 0.672779→0.665743 | -0.007037 ± 0.058976 |

| 指标 | 均值改善≥0.01配置数 | 均值差<0.01配置数 | 均值退步≥0.01配置数 |
|---|---|---|---|
| Whole RR AE | 15 | 7 | 6 |
| Local RR MAE | 24 | 3 | 1 |
| Envelope MAE | 0 | 28 | 0 |
| Global modulation error | 2 | 26 | 0 |
| Signed PCC | 0 | 28 | 0 |

Local RR均值下降达到0.01的24配置中，8配置三个seed原始方向均改善，其余仍含seed分歧或复用持平。按seed分组，20260812为24好/3差/1平，20260813为18好/9差/1平，20260814为16好/12差/0平，收益并不均匀。25/28配置的test Local RR跨seed SD降低；C_0p8、C_20、P_100例外，这也是三seed上的描述性结果。
NoMamba是RR的明显例外：自身改为loss选点后，Whole/Local RR差+0.038113/+0.033226，三个seed均退步且均达到0.01；global modulation error则改善−0.010005，三个seed同向。NoTF的Local RR几乎不变（+0.000187），Whole RR改善−0.038667（2好1平）。

## 受试者贡献

B的Local RR净改善−0.035159主要来自670和671：分别贡献−0.014112和−0.022813 bpm，二者合计−0.036925，其余6人净贡献为+0.001766。B在8人中5人的Local RR均值下降；220、704、726上升。
NoPoolPosition相对自身原选点的Local RR改善−0.073198，671贡献−0.051902、670贡献−0.013272。UniformPatchPool的改善−0.083405主要来自670/671，两者合计贡献约95%。NoMamba自身Local RR退步+0.033226，671贡献+0.027464，而670反而改善并抵消部分退步。

| 受试者 | B ΔLocal RR | A_NoTF ΔLocal RR | A_NoMamba ΔLocal RR | A_NoPoolPosition ΔLocal RR | A_UniformPatchPool ΔLocal RR |
|---|---|---|---|---|---|
| 220 | +0.007400 | +0.037558 | +0.048148 | -0.021185 | -0.006819 |
| 229 | -0.009375 | +0.015003 | +0.030436 | +0.004460 | +0.002702 |
| 286 | -0.015967 | -0.026820 | +0.005721 | -0.024508 | -0.016497 |
| 670 | -0.412652 | +0.017113 | -0.217530 | -0.388070 | -1.177695 |
| 671 | -0.104562 | -0.058052 | +0.125877 | -0.237882 | -0.178433 |
| 704 | +0.015224 | +0.034915 | +0.004298 | -0.003648 | -0.005362 |
| 726 | +0.012555 | +0.014689 | +0.009306 | -0.026641 | -0.005400 |
| 1006 | -0.033419 | -0.034198 | +0.029246 | +0.024178 | +0.011809 |

以上贡献始终保留原8人和2310窗分母；未根据个别结果排除人群。

## 同一loss选点规则下，结构判断如何变化

下表换用另一比较轴：各候选减同一选点规则下的B。它与前面的“同模型换选点”差值不同。

| 配置 | 原规则Local Δ对B | loss规则Local Δ对B | 原规则Whole Δ对B | loss规则Whole Δ对B |
|---|---|---|---|---|
| A_NoTF | -0.045267 | -0.009921 | -0.018860 | -0.036745 |
| A_UniformTFReadout | -0.002712 | -0.000158 | -0.005458 | +0.005232 |
| A_NoFrequencyBias | +0.005634 | +0.004177 | -0.027796 | +0.026289 |
| A_NoMamba | +0.034905 | +0.103290 | +0.062701 | +0.121597 |
| A_UniformPatchPool | +0.079440 | +0.031194 | +0.046274 | +0.013524 |
| A_NoPoolPosition | +0.003517 | -0.034522 | +0.015310 | -0.035067 |
| A_ScorePositionOnly | -0.013724 | -0.033838 | -0.011512 | -0.026717 |
| A_ValuePositionOnly | +0.036341 | +0.024760 | +0.021604 | +0.035349 |

### 保持稳健的部分

NoMamba在loss规则下相对B的Whole/Local RR差+0.121597/+0.103290，Signed PCC差−0.023128，三个seed一致；主干贡献仍是稳定结论。UniformPatchPool相对新B的Local RR差+0.031194，三个seed均退步，支持可学习池化在窗口加权RR上的作用；但受试者宏平均Local RR反而好−0.026540，仍有人群权重取舍。
UniformTFReadout相对新B的五项窗口加权均值差仍全部不足0.01；其宏平均Local RR差+0.011042，因此不表述为所有视图均等价。

### 对选点敏感的部分

NoTF相对B的Local RR优势从0.045267缩至0.009921，低于既定0.01阈值；Whole RR仍改善0.036745（2好1差）。新规则下NoTF的Local RR只在670和1006两人的均值较低，其余6人较高；宏平均改善−0.082026且三seed同向，主要由670较大的改善支撑。CWT分支的收益仍需按指标和人群解释，不能从这一比较推出所有人群都应移除分支。
NoFrequencyBias相对B的Whole RR从原规则的改善−0.027796，变为loss规则下退步+0.026289±0.006198，三个seed均退步；宏平均Whole RR亦退步+0.081559，三个seed同向。频率bias的必要性判断明显依赖选点规则。
NoPoolPosition、ScorePositionOnly在新规则下Local RR分别为0.579242、0.579926，是原始均值最低的两臂，相对新B分别改善−0.034522/−0.033838，三个seed一致。但两者仅差0.000684；NoPoolPosition的PCC配对差SD为0.017243，不能只按RR均值宣称各指标稳胜。本轮不据此修改默认结构。

### 位置2×2与CWT参数

新规则下值路径平均主效应使Whole/Local RR增加+0.048567/+0.046560，三个seed均为正；评分路径Whole RR主效应为−0.013499，三个seed为负。交互项Whole/Local RR为−0.043699/−0.025444，三个seed均负。这不等于两路径共同开启优于关闭：值路径本身的误差增加仍占主导，评分路径缓解了其部分影响。
具体地，评分位置关闭时开启值位置，Whole/Local RR差+0.070416/+0.059282，三个seed均退步；评分位置开启时再开值位置，Local RR差+0.033838，三个seed均退步。评分位置仅在值路径已开时显示较清晰的RR改善，原始2×2表提供完整条件比较。
19个CWT参数候选相对新B的Local RR没有一个均值改善达到0.01：13个不足0.01，6个退步（μ=6的四个voices，以及C_2、C_4），这6个配置三个seed均变差。Whole RR只有C_0p8均值改善达到0.01（−0.015284，2好1差），9个退步、9个不足0.01；其余三个非RR指标全部19候选的均值差均不足0.01。
原Local RR选点下P_100相对P_025有三个seed一致的RR改善；loss规则下P_025−P_100的Local RR差+0.009971、Whole RR差−0.005510，均不足0.01且seed方向混合。此前的时间池化偏好没有跨选点规则稳定保留。

## 证据边界与产物入口

本轮支持：在固定训练轨迹和既有研究test上，validation总loss选点对多数配置的Local RR更有利，也改变了部分消融效应。它尚不足以证明总loss是所有配置、全部指标或独立人群上的最优selector。原Local RR选点结果继续有效地描述其自身选点合同；后续引用消融结论时应明确selector，避免混合两种规则的配置进行排名。默认模型与训练早停规则保持不变。

- [原执行协议](m4_s1_loss_selected_research_test_v1_20261024.md)、[方向计数精度修订](m4_s1_loss_selected_test_precision_note_20261024.md)。
- [完整84-checkpoint清单](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_research_test_v1_20261024/checkpoint_list.csv)；allowlist SHA256=`4521211dd5f65116c222897209edda696ae4356733091b138714f0e366eda471`。
- [原自动报告](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_research_test_v1_20261024/summary/attempt_20261024T091035Z_8046093ea8ab/report.md)；summary manifest SHA256=`cb8c8bf3c22e0fa32d5f72b452f304905174292ba6769722f987252fd767466c`。
- [84项逐cell完整对照](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/cell_comparison_84.csv)、[核验后的窗口加权配对](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/selector_summary_verified.csv)、[宏平均配对](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/subject_macro_selector_summary_verified.csv)。
- [受试者配对与贡献](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/subject_selector_summary.csv)、[精度修订明细](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/precision_direction_corrections.csv)、[核验回执](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/audit.json)。
- [窗口加权五指标选点对照图](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/selector_weighted_deltas.png)、[受试者宏平均对照图](/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/selector_macro_deltas.png)。
