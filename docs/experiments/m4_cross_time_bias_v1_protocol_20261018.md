# M4 交叉注意力 Δt 时间 bias 消融 v1

协议 ID：`m4-cross-time-bias-v1-20261018`。

2026-10-18 用户授权仅移除原始 M4 交叉注意力中的 Δt 时间 bias，保留频率 bias 和其余结构，三 seed train/validation 在空闲 GPU 后台执行，完成后自动与原始 M4 比较。本阶段包含必要的 synthetic CPU 验证和同配置 GPU 合成 loss 验收。运行状态由 session、cell completion、summary receipt 记录；本协议在 prepare 后保持冻结身份。

## 问题与最小改动

原始注意力 logits 为 `qkᵀ/√d + bf(log f) + bt(Δt)`。当前 1 秒 patch 对应 CWT 的两个时间格，Δt 为 `[-0.25, +0.25]` 秒。A1 仅将 `bt(Δt)` 替换为无参数零项；删除 time_bias 两层 MLP 的全部 100 个参数。

| Arm | 交叉频率 bias | 交叉时间 bias | 本轮角色 | 参数量 |
|---|---|---|---|---:|
| A0 原始 M4 | 保留 | 保留 | 复用三个冻结 P1-M4 cell | 1,056,357 |
| A1 NoCrossTimeBias | 保留 | 移除 | 新增三个训练 cell | 1,056,257 |

维持 97×2 局部时频读取、真实频率和相对时间 buffer、Q/K/V/output、独立零初始化残差投影、GroupNorm、连续波形 CNN、时域注意力池化及其评分/值位置编码、6 层 BiMamba2、局部解码和 Hann overlap-add。该实验检验显式交叉时间 bias；读取窗口的物理对齐保持一致。

构造时先执行原始 `P1ComponentsModel("M4", seed, ...)` 的完整初始化，再用无参数模块替换 time_bias。所有剩余状态逐张量匹配原始 M4；不增加随机初始化调用，保留原构造的 CPU/CUDA RNG 序列。prepare 保存三个 seed 的官方 Mamba2 初始化摘要；正式运行逐 cell 再核对摘要与参数量。

## 原始参照与来源

参照沿用历史组件实验 B0 对应的三个 P1-M4 cell，科学配置、指标和选择规则保持该轮身份，见 [原始 M4 validation 结果](m4_components_v1_validation_results_20261009.md)。A0 在本协议中是新的参照标签，保存结果只在内存映射标签。

- 迁存根：`runs/home_storage_migration_20261010/preserved_worktrees/worktrees_20261010/model-architecture-review/`。
- P1 session：上述根下 `runs/p1_components_v1/formal_v1_20261006/`；session SHA-256=`34d0c62cb64adae16fca4db512687d28bb96e74e89857c049a506d66c4e71fb6`。
- 原始组件 session：上述根下 `runs/m4_components_v1/formal_v1_20261008/session.json`；SHA-256=`9f5f43120f2664f4b2eb214e73890e431c18320944c0f88aa70ad4c10b64f833`，其中保存三个参照 completion 与环境摘要。
- seeds=20260811/20260812/20260813；参照最佳 epoch=25/5/8。

prepare 核验参照 session/源码快照、配置、全部完成产物哈希、环境历史摘要和共有科学合同。迁存来源必须与历史摘要一致。正式训练环境的软件版本、关键包及 GPU 型号须与参照一致；编号可不同。参照失败或身份不符时显式停止，不重训参照或替换来源。

原配置的 `outputs.run_root` 保留历史字符串；必须经兼容链接解析到相同迁存 cell，再核对其余完整配置。输出路径的等价核验不扩展到数据或其他科学配置字段。

历史 P1 快照之后，仓库新增了独立的 `resp_train/paper_evidence/h_only_ablation_v1/` 文件。参照核验要求历史清单的全部源码仍逐项匹配，同时记录这一目录追加文件的 SHA；其他新增路径或任一旧文件变化均失败。本轮源码快照纳入当前完整依赖清单。

## 数据、训练与选点

复用 `configs/crd_tf_v1/crd_tf102_w_formal.yaml` 的数据、窗口、优化器、loss、指标和种子口径。train=10141 窗口/32 人，validation=2675 窗口/7 人；180 秒、100 Hz、1 秒 patch、0.5 秒 hop。使用既有冻结 train/validation W-full cache，核验 index 身份、行 ID 摘要和受试者隔离。

batch=32，梯度累积4，BF16；最多80 epoch，最早30 epoch停止，patience=15，min_delta=0；每 epoch 80 次 optimizer update。沿用 AdamW、原 lr/scheduler/loss，不改变预算。按完整 validation Local RR 严格最小、并列最早选择 best checkpoint。

GPU 合成验收使用正式 batch=32、累积4的两个 optimizer update、官方 Mamba2 和正式 RespirationTaskLoss，验证分支打开、全部梯度/状态有限及短 validation。正式运行前沿用 seed 重置，合成验收不改变正式初始化。

非有限 input、target、prediction、checkpoint、loss 或关键指标显式失败。每个 cell 保存 resolved config、原始命令、软件/GPU 环境、数据访问与 row identity、模型初始化摘要、optimizer groups、逐 epoch history/checkpoint、best/final checkpoint、逐窗口与汇总指标、失败或成功 receipt。

## 比较规则

完整 A1 三 seed 成功后，与三个冻结 A0 组成六 cell 比较。主指标为 Local RR MAE、Whole RR AE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC。保留原生辅助指标、分母、预测退化、逐人结果和 Local RR tail。

delta=A1−A0；误差下降/PCC 上升为正 improvement。同 seed 配对，输出 mean/sample SD、原始方向计数，以及窗口加权和受试者等权两种汇总。跨 cell 样本身份、顺序与 target 资格必须一致。

沿用本次会话的描述性判据：五项主指标各按其原生数值单位，绝对均值差 **<0.01** 记为“均值无明显差异”，达到或超过阈值按方向记为改善/变差；同时保存逐 seed 达阈值方向数。该阈值不改变 checkpoint 选择，也不是统计等效检验。不根据部分 seed 调整后续训练、模型、候选、预算或判据。

自动产物包括逐 seed/跨 seed、同 seed 配对、受试者宏平均配对、0.01 判据、尾部、原生辅助指标、来源 CSV、两个 PNG 配对图与中文 `report.md`。本阶段仅使用 train/validation；research-test 属于后续另行授权的阶段。

## 生命周期与运行

所有持久输出置于仓库 `runs/m4_cross_time_bias_v1/`；正式 identity 为 `formal_r2_20261018`。已有成功 cell 与 summary 验证哈希后复用；失败记录原地保留。训练中断后的显式 `--resume` 使用冻结引擎的已完成 epoch 恢复，剩余配置保持一致。

```bash
cd /mnt/disk_code/marques/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_m4_cross_time_bias.py
bash -n scripts/run_m4_cross_time_bias_shard.sh
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/run_m4_cross_time_bias.py prepare --session runs/m4_cross_time_bias_v1/formal_r2_20261018
bash scripts/run_m4_cross_time_bias_shard.sh runs/m4_cross_time_bias_v1/formal_r2_20261018 1
```

状态入口：`scripts/run_m4_cross_time_bias.py status --session ...`。shell 在单卡串行三个 seed，全部成功后自动 summarize；任一步失败立即退出，保留日志与失败 attempt。正式启动前检查空闲 GPU，后台日志保存于 session 内。完成验收为三个新 completion、三个持续通过身份校验的参照、每项固定 validation 分母、完整六 cell 配对汇总及所有产物哈希一致。
