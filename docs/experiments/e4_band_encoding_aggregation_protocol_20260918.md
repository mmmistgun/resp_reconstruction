# E4：频带编码与聚合权重的 validation 诊断

协议 ID：`e4-band-encoding-aggregation-v1-20260918`。日期：2026-09-18。

状态：专项方案、评价入口及 synthetic CPU 验证已完成，**36 passed，15.44 s**。训练均值参考、GPU smoke、真实 validation 及最终汇总尚未执行，留由用户运行。新的干预评价使用独立锁与输出身份；既有 E4、W0 的模型、checkpoint、selector、指标与冻结结果继续保持各自身份。

## 1. 问题与解释边界

同时观察三层量：聚合前表示及权重分布；固定表示时重加权的预测效果；CWT 入口干预后整条路径的预测效果。特征幅值、权重和加权贡献均称为描述量。较大权重不能单独等同于对最终预测较高的任务重要性。

代码定位：`CwtBranch` 先使用 5×3 conv、GroupNorm、SiLU，再使用 3×3 depthwise conv、1×1 conv、SiLU，得到 `X[B,96,97,360]`。除卷积在邻近槽位混合信息外，GroupNorm 的组内统计量跨尺度和时间计算；入口区域替换可能通过归一化影响其他尺度。因此，X 的尺度坐标是经过混合的表示槽位，不能直接解释为纯原始频带成分。聚合后仍通过插值、temporal、parameter fill、FiLM 和主干。

第二步与第三步改变的对象不同，二者指标差值不相减作为“卷积贡献”和“注意力贡献”。本专项检验现有 checkpoint 内的干预依赖，不执行再训练，也不将干预有效性解释为另一个训练方案的泛化效果。

## 2. 冻结矩阵与来源

- 模型：`W0_FULL`、`static_scale`、`scale_attention`、`frequency_attention`、`channel_region`。
- seeds：20260811、20260812、20260813；共 15 个原 selected checkpoint。
- W0 selected epochs 为 13/15/14；四候选分别为 13/18/12、13/32/14、13/30/14、13/30/14。
- 四候选源汇总：`runs/e4_scale_aggregation_v2/summary/summary_71e6b8f753cc_20260918T055104Z_da596d615deb`。
- 训练来源锁 SHA：`71e6b8f753cc88fdf150ba7e6a7373419415e3c1f19a5d78cdcaeb68f56df90f`。
- validation：原 2675 窗口、7 个 `samp_id`、sample seed=20260611、有序 dataset_row_id 完全一致。沿用数据集的 `samp_id` 受试者/记录标识分层；重叠窗口不视为独立受试者。
- `samp_id` 窗口数：952=579、956=213、961=468、971=820、972=182、1308=57、1378=356。
- physical batch=128，完整尾 batch=115，eval/no_grad，BF16 autocast；原 180 s/100 Hz 输入、THO target、admission、五项指标、target-only eligibility 和分母继续适用。

四区域沿用原始索引和真实频率网格：

| 区域 | 槽位切片 | 个数 | 边界约 Hz |
|---|---|---:|---|
| R0 | `[0:25]` | 25 | 0.0366–0.1408 |
| R1 | `[25:49]` | 24 | 0.1408–0.5411 |
| R2 | `[49:73]` | 24 | 0.5411–2.0800 |
| R3 | `[73:97]` | 24 | 2.0800–7.9956 |

内部边界归右区，最后端点包含；重复频率槽位保留。全部窗口参与统计；展示窗口在 prepare-lock 阶段按每个 samp_id 的 `(segment_id, window_start_s, dataset_row_id)` 排序，取首、中、末三个，共 21 个固定窗口，全部模型使用相同行，按结果不追加或替换示例。

## 3. 第一类：完整 FULL 表示记录

每个模型完整运行原模型条件 FULL，并与自身冻结 validation 五项指标均值核对，容差 `rtol=1e-3, atol=0`。失败停止该模型后续干预，保留现场。

保存完整 X、实际槽位权重 alpha、区域权重质量及逐窗口描述统计。W0 的 alpha 为常数 1/97；静态和区域权重按共享轴紧凑保存，动态 attention 保留 `[window,1,97,360]`；区域保留 `[1,96,97,1]`，可按区域求和恢复 96×4 区域质量，不能先平均通道后推断所有通道。

逐窗口计算：

1. `z0=mean_s(X)`、`Delta_ideal=sum_s((alpha-1/97)*X)`、实际 BF16 聚合输出差 `Delta_executed=z_native-z0`。区域使用模型原生区域 mean 路径作为实际输出；理想槽位公式与实际输出的舍入差单独保留。
2. mean/ideal correction/executed correction 的 RMS、后二者相对 mean RMS 的比。mean RMS=0 时比例记 NA，并记录 `mean_is_zero`，不加任意 epsilon。
3. 权重相对均匀先验的 TV，在通道及时间维度分别给均值和最大值；R0–R3 的质量、特征 RMS、加权贡献 RMS、修正贡献 RMS。
4. `cancellation_ratio=||sum_s v_s||2 / sum_s ||v_s||2`，其中 `v_s=(alpha_s-1/97)X_s`，范数覆盖通道×时间。分母为零时记 NA 与显式标志。低值表明修正项抵消，不代表频带无信息；另保存跨尺度特征标准差的 RMS。

21 个预固定窗口绘制相对先验权重图、区域质量时间曲线、mean/两类 correction RMS 时间曲线。频率轴采用槽位索引并标注真实 Hz，避免把非等距频率当作等距；区域图的通道平均只用于总览，另绘完整 96×4 通道质量图。全体窗口另保存区域质量随时间的 mean/SD，属于描述性分布。

## 4. 第二类：固定 X 的聚合干预

FULL 保存的 X 以原 BF16 位模式恢复，后续五个条件直接读取同一窗口的 X，跳过前卷积及 GroupNorm；BCG、主干、target 和其余模块仍调用原模型。所有条件的输入行顺序必须一致。

| 条件 | 操作 |
|---|---|
| `UNIFORM` | 聚合结果直接替换为同一 X 的原始尺度 mean |
| `RESET_R0`…`RESET_R3` | 区内每槽位恢复 1/97，区外权重按比例重新分配 |

对一个区域 B，槽位数 n_B：区内 `alpha'_s=1/97`；区外 `alpha'_s=alpha_s*(1-n_B/97)/sum_{j not in B} alpha_j`。按样本、时间、通道独立归一化，非区域方案权重在通道间共享。区域方案等价于目标区域质量恢复 n_B/97、其余三块质量按原比例重分配，保持区内原始槽位等权。

区外总质量必须正且有限；若出现数值零，明确失败，不用 epsilon 或另一套权重替代。所有条件检查质量守恒。区域干预包含维持总质量所需的区外再分配，其结果解释为该干预的整体效应，不是孤立区域因果贡献。W0 的五种聚合干预均必须逐点复现 FULL，作为精确 no-op 对照。

## 5. 第三类：入口频带替换

在原 `[B,97,360]` W cache 张量、进入 conv_in 之前替换区域，随后正常重新计算卷积、GroupNorm、attention 和全部后续模块。

- 主参考 `INPUT_MEAN_R0`…`R3`：从冻结的完整 train W 数组计算每尺度所有训练窗口与时间点的均值，共 97 个固定值；FP64 累加，保存 FP32。该条件同时移除区域内部的时间及窗口间变化，保持训练集逐尺度平均水平。
- 复核参考 `INPUT_ZERO_R0`…`R3`：对应已预处理 W 表示中的固定零值。它不等同于物理信号不存在；与 train mean 的差异用于检验替换方式敏感性。
- 第二种参考预先覆盖全部四区域和全部 15 个模型，不依据主参考结果选择“关键频带”后再决定复核范围。
- 参考回执记录 train mean 与零参考的最大差及是否完全相同；若二者相同，不能把重复结果当作两种不同替换方式的稳健性证据。
- 不修改原 cache、其他区域、BCG 或 target；不在遮挡后重新做 cache 标准化。模型中的 GroupNorm 正常重算。

共 `1 FULL + 1 UNIFORM + 4 RESET + 8 INPUT = 14` 条件/模型，即 **210 个条件、561750 条窗口级评价记录**。其中 75 个聚合条件复用 FULL 的 X，120 个入口条件正常重算前端。

## 6. 配对报告及判断

五项指标固定为 Whole RR、Local RR、包络轨迹误差、全局包络调制误差、lag-aware signed PCC。每一条件与**同模型、同 seed、同窗口 FULL** 配对；error 差为 intervention−FULL，PCC 差为 FULL−intervention，统一正值表示变差。误差相对变化仅在 FULL 分母非零时报告。

保存窗口级完整 metrics、每 seed pooled 差、7 个 samp_id 各自差、samp_id 等权宏平均、三 seed mean/sample SD 及改善/恶化/相等方向数。不得将 7×3 个方向或 2675 个重叠窗口当作独立重复进行显著性推断。无新增主指标或候选选择阈值；“明显”通过原始效应量、seed SD、个体方向和两种参考一致性描述，不以热图颜色深浅判断。

- 权重质量较大、干预变化较小：记录模型偏好与该干预下任务依赖不一致。
- 入口替换有较大效应、UNIFORM 效应较小：该 checkpoint 依赖相应 CWT 信息，但没有显示整套特殊加权的必要性。
- UNIFORM 后变差：重加权对该 checkpoint 内的固定特征有帮助；不等同于候选整体胜过 W0。
- UNIFORM 后改善：是权重分配可能造成性能损失的干预证据，仍需结合 seed 和受试者方向。
- 两种入口参考方向不一致：报告替换敏感，不给出统一的频带依赖排序。

## 7. 存储、冻结与失败

完整 X 原 BF16 数据约 **250.55 GiB**（每模型约 16.70 GiB），完整动态 alpha 另约 2.09 GiB，另有 metrics、图和元数据。BF16 使用 uint16 保存原始位模式，不做有损量化。每模型开跑前检查完整 X 预算与额外 24 GiB 余量；失败不自动缩减窗口或改变保存粒度。当前检查时磁盘约 421 GiB 可用，实际运行按当时空间预检。

输出根 `runs/e4_band_encoding_aggregation_v1/`，阶段为 reference/gpu_smoke/evaluation/summary。实现锁 `docs/experiments/e4_band_encoding_aggregation_lock_20260918.json` 记录代码、协议、15 个 checkpoint/config/冻结 validation CSV、频率、有序 rows、固定展示行、cache/index 字节身份。每阶段排他创建独立 attempt，成功写 manifest 与 freeze receipt，失败写 lifecycle_failed；成功 cell 同身份拒绝重跑，失败目录保留。15 个模型必须使用同一 reference manifest。

历史训练/test 锁与运行产物保持原样。新的 FULL 是专项复现对照，不覆盖历史评价。当前专项只开放 train W 参考准备和 validation 推理，test 不进入此矩阵。

## 8. 验收与执行

CPU synthetic 验收覆盖：区内先验恢复、区外比例及质量守恒、逐通道行为、入口局部替换与源 tensor 不变、FULL 与原分支等价、固定 X 跳过前卷积、W0 聚合 no-op、BF16 位模式 round-trip、零分母标志、配对/PCC 方向、失败保留、SHA 防漂移和完整 15-model 汇总。

实际 CPU 命令：`env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./.venv/bin/python -m pytest tests/test_e4_band_audit.py -q`。包含 native metrics 的 2-window、14-condition 临时流程和 BF16 wrapper 重放；未访问真实波形或运行 GPU。

用户 GPU smoke 使用原 selected checkpoint 和 synthetic 输入：全部 15 个模型 batch-1 检查原 forward、FULL wrapper 和缓存 X 的 FULL replay 等价、全部干预 finite、W0 no-op；每 arm seed 20260811 另做 batch-128 全条件检查。保留 `rtol=1e-5, atol=1e-6`、peak reserved ≤80% 门槛。真实 FULL 必须另通过冻结 validation 五项均值的复现门槛。

代码与锁准备后先提交，保持工作树干净。以下实际数据/GPU 阶段由用户运行：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/eval_e4_band_encoding_aggregation.py prepare-reference
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/eval_e4_band_encoding_aggregation.py gpu-smoke --device cuda:0
```

填入上述两次成功输出目录后：

```bash
E4_BAND_REFERENCE='/实际成功的reference目录'
E4_BAND_GPU='/实际成功的gpu_smoke目录'
(
  for E4_BAND_SEED in 20260811 20260812 20260813; do
    for E4_BAND_ARM in W0_FULL static_scale scale_attention frequency_attention channel_region; do
      env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
        scripts/eval_e4_band_encoding_aggregation.py evaluate \
        --arm "$E4_BAND_ARM" --seed "$E4_BAND_SEED" --device cuda:0 \
        --reference "$E4_BAND_REFERENCE" --gpu-receipt "$E4_BAND_GPU" || exit $?
    done
  done
  env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
    scripts/eval_e4_band_encoding_aggregation.py summarize
)
```

每模型交付完整 X/alpha、描述统计、预固定窗口图、14 个条件的 metrics/summary 和 560 行配对记录。最终交付 8400 行 seed×条件×指标×pooled/subject 明细、2800 行三 seed 方向表、1050 行按 seed 的受试者等权宏平均，以及全部来源回执。

最终汇总另提供 40125 行 `feature_statistics_by_window.csv`，并在 `feature_statistics_by_seed_subject.csv` 中分别保存每模型/seed 的 pooled 及各 samp_id 描述量 mean/median/p10/p90、总数与有定义数。零分母的未定义项保留为 NA，不借分布汇总隐藏缺失定义。
