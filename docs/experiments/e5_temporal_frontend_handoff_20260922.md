# E5 时域前端对照：规划交接

日期：2026-09-22。工作目录：`/mnt/disk_code/marques/resp_reconstruction`。

## 1. 当前任务与交付

用户已确认进入 E5 的独立规划，并要求在新会话开始。请直接核对代码、原 W0 来源和现有抗混叠实现，制定可审查的专项方案。首轮推荐一个前端候选、三个训练 seed；本阶段交付方案、控制变量、来源及验收设计，再向用户汇报。

科学问题：**替换时域前端，能否改善固定 W0 其余结构下的多属性呼吸重建？**

规划交付至少包括：

1. 原前端和候选的真实张量流、时间网格、参数量、归一化和边界语义。
2. 一个首轮候选的精确定义、科学配置和独立命名；说明选择理由与归因范围。
3. 公共模块、初始化、数据、损失、预算与 selector 的控制清单。
4. 三 seed train/validation 矩阵、五指标配对与受试者汇总、参数/计算/效率代价。
5. synthetic CPU、用户执行 GPU 验收及正式训练的分阶段门槛，生命周期和来源记录要求。
6. 明确会影响正确性的未决问题；可由代码和冻结来源确定的细节先自行核实。

## 2. 已确认的研究边界

E5 属于新候选开发。原 W0 继续作为冻结对照和既定论文主模型。首轮先规划 train/validation，候选从头训练；test 阶段需要匹配专项协议与届时用户授权。

固定 97-scale W 条件分支及均匀尺度聚合、六层 BiMamba2、FiLM、refinement/decoder、数据 split、target、损失、训练预算和 validation selector。前端输出接口保持 `B×96×1800`。不把前端比较与尺度聚合、主干深度或 FiLM 位置变化组合成全因子搜索。

关键归因限制：替换完整前端会同时改变局部编码、采样方式、归一化及参数容量。首轮可回答前端整体替换的收益，不能独立归因于线性插值、抗混叠或某一种归一化。140 个多通道 token 也不能直接等同于同采样率原始波形，不能仅凭 token 数量认定呼吸信息已丢失。若要进一步分解机制，应明确新增对照的必要性与规模。

## 3. 原 W0 前端：已核对代码

文件：`resp_train/crd/frontends.py`。

- `PatchTokenFrontend`（约第 22–38 行）使用 `LegacyPatchTokenEncoder`。
- 输入 `B×1×18000`，100 Hz、180 s。
- Patch 长度 256、步长 128、16 个 embedding 通道、两个 mixer blocks，右侧补齐后输出 140 个 token。
- `F.interpolate(tokens, size=1800, mode="linear", align_corners=False)`。
- 后接 `Conv1d(16,96,kernel_size=1,bias=False)`、`GroupNorm(12,96,eps=1e-5)` 和 SiLU，输出 `B×96×1800`。
- `LegacyPatchTokenEncoder` 先构造原 `PatchMixer1D`，再接管共享 encoder modules，以保持原初始化语义。

文件：`resp_train/crd/model.py`，前端约在第 232–233 行以独立命名子 seed `patch_frontend` 构造。还需沿 `resp_train/crd/tf_v1_model.py` 核对 W0 实际包装、前端替换位置及全部公共参数路径。

规划需核对 token 对应的物理时间、补齐与插值端点、最终 10-Hz latent 网格；不要将数组长度匹配直接等同于严格物理时间对齐。

## 4. 首轮候选参考：已有显式抗混叠前端

已有代码：`resp_train/temporal/blocks.py` 中 `TemporalStem`（约第 163–202 行）及 `FixedPolyphaseDecimator1D`。

当前已核对的结构：

```text
B×1×18000，100 Hz
→ 固定抗混叠降采样 100→20 Hz
→ Conv1d 1→48，k21、padding10、bias=False
→ channel-only LayerNorm → SiLU
→ depthwise Conv1d 48→48，k5、padding2
→ channel-only LayerNorm → SiLU
→ Conv1d 48→96，k5、padding2、bias=False
→ channel-only LayerNorm → SiLU
→ 固定抗混叠降采样 20→10 Hz
→ B×96×1800
```

已有 FIR 合同：100→20 使用 255 taps、cutoff 9.0 Hz；20→10 使用 127 taps、cutoff 4.5 Hz；Kaiser beta=8.6，参考 `padtype=line`。20-Hz 学习编码与非线性位于两次固定降采样之间。

来源文档：

- `docs/experiments/resp_temporal_v1_signal_audit_20260820.md`，尤其第 6 节。
- `docs/experiments/resp_temporal_v1_protocol_20260820.md`，尤其第 5 节和约第 138 行的冻结 substrate 定义。
- `resp_train/temporal/config.py` 的科学配置校验。

这里参考的是已实现并审计的最终显式 FIR substrate；协议中还保留有早期 strided-convolution probe 的历史描述，核对时要区分版本。E5 只借鉴/复用已审计前端算子，主干和 decoder 按原 W0 固定。

候选与 W0 的归一化不同（channel-only LayerNorm 与 GroupNorm），是必须公开的结构变化。需要核对候选参数量、感受野、延迟/边界与数值精度，以及公共模块同 seed 初始化是否保持逐 tensor 一致。E5 应使用独立入口和来源记录，避免改变既有实验的模型行为。

## 5. 冻结对照与训练合同

W0 配置：`configs/crd_tf_v1/crd_tf102_w_formal.yaml`。下面三个 run 位于仓库根目录下：

| Seed | Selected epoch | 原 W0 run |
|---|---:|---|
| 20260811 | 13 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260811/20260812_210725_400861` |
| 20260812 | 15 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260812/20260812_223203_048130` |
| 20260813 | 14 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260813/20260812_235740_452006` |

可核对的既有合同：全模型 1,219,850 参数，W 条件分支 150,048 参数，active fill 为 96→65→96；FiLM gamma/beta 系数为 0.5/0.5。

原训练每 seed 80 epochs / 6400 updates，batch=128、accumulation=1、BF16，完整目标 `L_sync + 0.25 L_effort`，AdamW 与原 step-exact warm-up cosine 日程；完整 validation Local RR 严格最小选择 checkpoint，平局取更早 epoch。制定新协议前从原 resolved config 与相关锁复核字段，不仅依赖本交接摘要。

train/validation/test 窗口数为 10141/2675/2310，samp_id 数为 32/7/8；样本 seed 分别为 20260610/20260611/20260612。

五主指标：Whole RR absolute error、Local RR MAE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC。同时报告绝对量级、同 seed 配对差、三 seed sample SD 与方向、samp_id 分组和等权宏平均。重叠窗口与模型 seed 不作为额外独立受试者。

## 6. E4 结项状态及可用材料

- 当前状态：`docs/experiments/e4_closeout_20260922.md`。
- 完整结构比较：`docs/experiments/e4_closeout_comparison_20260922.csv`。
- 性能、代价及机制数据：`docs/experiments/e4_closeout_data_20260922/README.md`。
- 完成产物索引：`docs/experiments/e4_closeout_artifact_index_20260922.json`。
- R3/GN 结果：`docs/experiments/e4_r3_temporal_normalization_results_20260921.md`。

E4 没有产生足以替换 W0 的稳定整体收益；机制结果支持部分重加权有用，并支持当前模型使用 R3 对数幅度时间结构。它们提供研究背景，不预先证明 E5 的抗混叠前端会改善表现。

用户已授权并完成删除 15 份聚合前 `FULL/x.npy`，释放约 250.55 GiB；CSV、checkpoint 和原 manifest 保留。精确范围见 `docs/experiments/e4_feature_x_cleanup_20260922.json`。原完整产物校验将报告这些 X 缺失，这是已登记的清理状态。E5 规划使用代码、配置与冻结汇总即可，不依赖这些已清理的数组。

最近相关提交：`0cb82fc`（E4 结项结果与证据数据）、`f83f2c0`（完整特征清理及数据复核）。开始工作前再次检查 Git 状态，保留其他任务改动。

## 7. 协作与执行约束

首先读取仓库 `AGENTS.md` 和 `/home/marques/.codex/AGENTS.dl.md`。默认中文，回答以“我在！”开始。文件操作遵循仓库 FastCtx 偏好。

本次目标为 E5 规划文档。允许必要的只读来源核对和短时 synthetic CPU 检查；GPU、正式训练、真实数据 smoke、benchmark 与 test 默认由用户执行。沿用 E4 已关闭状态，E5 使用新的协议/identity。

交付优先采用说明文档、数据表、字段与来源记录。论文工作区 `/mnt/disk_code/marques/paper/resp_rec` 有其他未提交修改；如需读取 E5 背景，入口为 `paper_rewriting_output/实验项目待补实验清单_20260915.md` 第 6.2 节和内部事实索引，先读取该目录 AGENTS.md。本阶段主要在实验仓库形成专项方案。
