# 原始 M4 高频时间结构机制补充实验

协议 ID：`m4-high-frequency-v1-20261009`。日期：2026-10-09。

当前状态：实现与 synthetic CPU 验证已完成，22 项定向测试通过；未执行真实数据推理、GPU 验收或 research-test。用户本轮要求补充此前讨论的实验，实现范围为本协议、独立运行/汇总/绘图入口与定向合成测试。真实数据阶段默认由用户运行；Codex 代跑须取得当次明确授权。已有训练、组件评价与科研结论保持冻结。

## 1. 研究问题及既有证据

主问题：原始 P1-M4 是否利用 (0.8,8] Hz CWT 对数幅度的时间结构，该结构的时间对应对呼吸包络和波形相关有什么作用？

证据链为“高频幅度具有呼吸相关变化 → M4 条件读取与加性残差响应 → 输出配对误差变化”。主要机制观察指标预定为 envelope trajectory MAE 与 lag-aware signed PCC；Whole RR、Local RR、global envelope modulation 与主指标共同完整报告。全部沿用原公式、资格、target、split 和 validation Local RR checkpoint selector，不据新结果调整指标或选点。

已完成的结构对照只读复用：

- [M4 组件 validation](m4_components_v1_validation_results_20261009.md)。
- [M4 组件 research-test](m4_components_research_test_results_20261009.md)。B0/B1/B4/B5 为 Full/NoTF/H/L，三 seed 均完整。
- [CWT-APOR 信号及机制证据](cwt_apor_v2_mechanism_evidence_results_20261003.md)。旧信号定义核对后可作背景；旧模型效应不能作为 M4 结果。

现有 Full/H/L/NoTF 有属性与 seed 取舍，不能预设 M4 整体高频增益已成立。H/L 训练对照改变 GN 支持域和卷积边界，时域路径仍接收宽带 BCG。当前新增干预回答固定模型功能依赖；不重新训练，不扩大输入路径频带限制。更高载频 H2 或低频错位属于后续可另行定义的问题，不根据本轮中间结果追加到当前矩阵。

## 2. 固定模型与来源

对象为原始 P1 `M4`（Full CWT、局部交叉注意力、双投影 residual-add），不使用 M0 FiLM 或 M4-v2 单投影。seeds 为 20260811/20260812/20260813，对应 validation-selected epoch 为 25/5/8；每模型 1,056,357 参数。

复用 `runs/p1_components_v1/research_test_v1_20261007/allowlist.json` 的 M4 三项来源；allowlist SHA256 固定为 `8ac8b2305f66cd4f54f6a317f7c1dad9336bd88d393206fa84b5f300ad68fa47`。该清单记录配置、checkpoint、completion、源码和原训练环境，prepare 核验来源，不读取 test index/cache/波形/指标值。

每个新 session 保存同一来源的 manifest、代码快照、SHA 和准备回执；不增加独立实验锁。运行核验 checkpoint 配置、epoch、参数量、频率坐标及训练环境（torch/CUDA/packages/GPU 型号）。共享模型和正式 metrics 的源码不修改。

## 3. 十条件矩阵

H 按 checkpoint/cache 的实际中心频率选择 `(0.8,8] Hz`，应为 41 scales；低频为 56 scales。保持完整 97×360 网格与原物理坐标，W 时间采样率 2 Hz。CWT 载频与幅度调制频率分别标注。

| 变换 | 定义 | GN 模式 |
|---|---|---|
| FULL | 原 W | NAT / FIXED |
| H_SHIFT_1/2/3 | H 全尺度使用同一窗口偏移共同循环移动；低频逐点不变 | NAT / FIXED |
| H_MEAN | 各 H 尺度用该窗口自身时间均值替换；低频逐点不变 | NAT / FIXED |

每个 dataset_row_id 用 `SHA256("m4-hf-v1:20261009:<row_id>")` 前 16 字节构造 PCG64 seed，从整数 `[60,300]` 无放回取三个偏移，分别对应 30–150 秒。偏移与训练 seed、batch 大小、运行顺序无关，完整 `shifts.csv` 保存实际取值。不逐窗口寻找最不利偏移；共同移动保留高频尺度间时间对应。

NAT 使用原生 GN；FIXED 使用同 batch、同窗口 FULL 捕获的条件编码器 `encoder.2` GN 均值和逆标准差。固定统计复用已验收的原生输出校正公式，另以独立 affine 公式逐批检查（FP32 rtol=1e-5/atol=1e-6；低精度按 dtype epsilon）。GN 拓扑不同则失败。

每批 FULL hook 前向与普通前向逐点相等，FULL/FIXED 与 FULL/NAT 预测及 z/context/u/residual 逐点相等；固定统计等于当批 FULL，所有条件 z 不变。基准来自本次相同 batch 的 FULL，不与旧 batch32 常规评价混用。

## 4. 数据、执行与访问边界

两个 session 使用相同预定矩阵，分别显式准备：

| split | 窗口/受试者 | 条件窗口指标行数（3 seed×10 条件） |
|---|---|---:|
| val | 2675/7 | 80250 |
| test | 2310/8 | 69300 |

真实推理固定 batch=8、BF16、无 shuffle、完整尾批、CUDA；每批额外普通 FULL 用于数值重放验收，共 11 次模型前向。没有训练和 benchmark。validation 行文件须与原选点来源逐字节一致；test 沿用现有 test 行身份、cache manifest 与 development 受试者隔离检查。

test 入口在加载 checkpoint 和数据之前要求 `--confirm-research-test`。该参数供用户执行已确认的本协议矩阵；不是 Codex 自动代跑授权。已有 test 参与开发，新的 test 效应仍为 reused research/development evidence，不称独立确认结果。validation 与 test 分开汇总，查看 validation 后不改变 test 候选、偏移、指标、案例或矩阵。

任何 input、target、checkpoint、prediction 或关键指标的非有限异常显式失败。沿用原指标定义的 target-ineligible NaN 必须与资格逐项一致，保存完整行和分母，不删除样本。跨条件、跨 seed 的样本及 target 资格必须一致。

## 5. 配对统计与机制路径

五项既有主指标完整报告。误差 deterioration=干预−FULL，PCC deterioration=FULL−干预；正值统一表示变差。保留逐窗口差值、逐受试者×seed×条件、窗口加权与受试者等权的逐 seed 结果，以及三 seed mean/sample SD 和变差 seed 数。配对先在同 seed 同窗口内计算，拒绝缺 seed/条件的正式汇总。

主图每点为一名受试者的三 seed 平均差；各 SHIFT 独立显示。重复 seed 与重叠窗口不作为独立受试者，也不把 seed SD 画成人群置信区间。小效应同时用原始单位解释；不把所有指标的变化概括为同向收益。

每批捕获时域 z、条件读取 context、融合 u 与 `residual=u−z`；保存每窗 context/residual 的 FULL 配对 RMS 差和 GN 公式残差。RMS 只描述路径响应，输出性能由配对指标判断，不作中介因果分解。

## 6. 信号诊断与固定案例

信号诊断只在 seed20260811 的 FULL 数据流计算一次，不额外运行模型，也不把重复 seed 当作信号重复。

1. 每窗逐 CWT 尺度计算 fs=2 Hz、Hann 窗、去均值的周期图；将 DC 置零后按该尺度非 DC 总功率归一化。常量尺度显式标记无效并保存 valid_count。先在受试者内平均窗口，再受试者等权；保存载频×调制频率谱及逐尺度支持数量。
2. H 尺度平均 log 幅度做 10 秒矩形均值，与既有指标使用的 THO 中心化 log-RMS 包络按物理时间对齐，计算零时滞有符号 Pearson r。比较 FULL 与三个预设 SHIFT，不搜索最佳时延。平滑仅使用完整支持区；常量信号记录原因及配对有效分母。

第 2 项与旧 APOR “先投影呼吸调制带、再取包络”的代理定义不同，独立标注，不拼接数值。相关性、调制功率比例都不命名为模型贡献率或生理来源。

各受试者按冻结数据行顺序选首/中/末窗口，选择不访问预测或干预效果；所有三个 seed 导出全部预选案例，validation 为 21 例/seed，test 为 24 例/seed。全部十条件保留 NPZ；图册统一比较 FULL 与 H_SHIFT_1/FIXED，显示原始/错位 CWT、加性残差差、呼吸波形、包络及绝对误差差。完整 180 秒，不按有利效应重选或裁剪。输出 PNG/SVG/HTML。

## 7. 命令与验收

在 M4 工作树执行：

```bash
cd /home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction
env PYTHONDONTWRITEBYTECODE=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_high_frequency.py plan
```

validation（以下为交付命令，本轮未代跑）：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONDONTWRITEBYTECODE=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_high_frequency.py prepare --split val --output runs/m4_high_frequency_v1/validation_20261009
for seed in 20260811 20260812 20260813; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_high_frequency.py run --session runs/m4_high_frequency_v1/validation_20261009 --seed "$seed" --device cuda:0 || exit "$?"
done
env PYTHONDONTWRITEBYTECODE=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_high_frequency.py summarize --session runs/m4_high_frequency_v1/validation_20261009
env PYTHONDONTWRITEBYTECODE=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/plot_m4_high_frequency.py --session runs/m4_high_frequency_v1/validation_20261009 --output runs/m4_high_frequency_v1/validation_figures_20261009
```

research-test 需执行者明确确认；使用独立 identity：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONDONTWRITEBYTECODE=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_high_frequency.py prepare --split test --output runs/m4_high_frequency_v1/research_test_20261009
for seed in 20260811 20260812 20260813; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_high_frequency.py run --session runs/m4_high_frequency_v1/research_test_20261009 --seed "$seed" --device cuda:0 --confirm-research-test || exit "$?"
done
env PYTHONDONTWRITEBYTECODE=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_high_frequency.py summarize --session runs/m4_high_frequency_v1/research_test_20261009
env PYTHONDONTWRITEBYTECODE=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/plot_m4_high_frequency.py --session runs/m4_high_frequency_v1/research_test_20261009 --output runs/m4_high_frequency_v1/test_figures_20261009
```

prepare 拒绝已有目录；成功 seed 和 summary 经 SHA 校验后直接复用，不重跑。失败与中断目录保留，查明原因后 `run/summarize --retry` 新建 attempt；同一 session 的代码身份变化会拒绝继续。图件输出同样拒绝覆盖。

验收：三 seed×十条件全部完成；样本、target 资格、频率和 checkpoint 身份一致；FULL 两项重放差为零；固定 GN 与原统计完全一致；五指标分母合法；source/manifest/environment/config/receipt 齐全；完整受试者配对图、调制谱和案例图册生成。任何一项不符则不声明实验完成。

CPU 定向验证命令：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_m4_high_frequency.py
```

CPU 测试使用可微 Mamba 替身和合成 180 秒数据，覆盖 FP32/BF16、FULL 与固定 GN 数值重放、路径改变、hook 异常清理、低频保护、row 偏移稳定、资格/错序/partial 拒绝、失败保留、真实指标调用及绘图接口。正式 Mamba CUDA 内核、真实 cache 与实际 checkpoint 的数值验收仍由获准运行阶段执行。

## 8. 实现验证记录

2026-10-09：22 项定向 CPU 测试通过（22.65 秒）；使用合成 180 秒窗口、Mamba 替身和正式任务指标，生成并检查 PNG/SVG 案例图。低频保护、FP32/BF16 FULL/FIXED 重放、跨 seed 样本及资格拒绝、信号常量的明确无效标记与失败产物保留均通过。

在临时独立目录执行一次来源准备检查，原 P1 allowlist、当前依赖源码及三份 M4 checkpoint/config/completion 的 SHA 核验通过；该检查仅访问既有 provenance、配置与 checkpoint 文件字节，未执行模型、未打开真实 index/cache/波形。正式 session 应使用第 7 节命令另行准备。Git diff 空白检查通过；未产生科研指标或改写历史结果。
