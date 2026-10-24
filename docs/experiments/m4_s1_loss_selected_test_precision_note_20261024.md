# M4-S1选点规则对照的浮点往返计数修订

2026-10-24完整结果核验发现：自动汇总将本轮逐窗指标计算出的均值与前次summary CSV重新读取的均值相减。十进制往返会产生约10⁻¹⁶量级误差，使两个完全相同checkpoint的部分指标被计为微小改善或退步。

涉及只读复用的 `A_UniformPatchPool_seed20260812` 和 `A_NoTF_seed20260813`。两者引用的预测和逐窗指标文件完全相同；其选点对照在全部五指标上应为持平。

修订按同一计算路径分别从两侧逐窗CSV计算均值，再按原规则逐seed配对。原指标公式、mask、分母、0.01阈值和checkpoint保持不变；不以额外epsilon阈值替换原判断。独立核验要求：复用cell五指标delta精确为0；与原自动表的mean/SD差仅为浮点精度；0.01分类及达到阈值的方向数完全相同；原始方向计数变化仅发生在上述复用cell对应的指标。

原自动汇总及首次审计失败日志保留。修订表存放在 `runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/` 的 `selector_paired_verified.csv`、`selector_summary_verified.csv` 及对应 `subject_macro_` 文件。最终结果页以这些已核验对照表的seed方向计数为准，均值和阈值结论与原自动汇总一致。来源、具体变化和核验回执由同目录 `audit.json` 和 `precision_direction_corrections.csv` 记录。
