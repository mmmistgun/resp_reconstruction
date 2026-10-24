# RespDiff-BCG P6 独立 research-test v1

协议 ID：`respdiff-bcg-research-test-v1-20261019`。

当前状态（2026-10-24）：**固定三模型 × 三配置 research-test、完整汇总、产物回收及复核均已完成并关闭。** 五指标、逐 seed/受试者差异、validation/test 对照、采样成本与冻结来源统一见 [P6 结果](respdiff_bcg_research_test_results_20261024.md)。执行经过见 [双卡迁移记录](respdiff_bcg_research_test_gpu_migration_20261023.md)、[恢复记录](respdiff_bcg_research_test_recovery_20261021.md)，初始验收与来源见 [T630 部署记录](respdiff_bcg_research_test_t630_deployment_20261019.md)。工作树中的原执行版协议作为 manifest 绑定来源继续冻结；本主工作区副本维护当前状态路由。

## 固定科学合同

使用 P5 的 `long_stretched` final16000 三个模型：training seeds20260811/20260812/20260813。模型 SHA256 分别为 `c1e16cfdfba3498a60eafad11e1a72007b5e9969ba921d2eb7b679690d558731`、`5a0c9f1b06879ea62aa6173571ad29a54de28b07ca919699db9bccaea12aab0e`、`b07c5efc3ed135b43578db38b197c0a0b4922ab55b595525f09414872f0c36cb`。

每模型完整评价 DDIM6/N64、DDPM50/N16 和固定参照 DDIM6/N16。sampling seed20261003、FP32、B64/G1、尾批保留、DDIM eta=0、原反向更新、FP64 在线 ensemble 均值、180 s parent 双 1 Hz LPF、20 Hz/30 s chunk/15 s hop/Hann OLA、100 Hz 插值与输出 parent LPF 均保持 P5 合同。DDIM N16/N64 共享同一 N64 sweep 的前缀，各配置另行独立计时（首批、warmup=1、repeats=2）。

完整使用原 research-v2 索引中按相同准入过滤得到的 `split=test`，不抽样、不挑案例、不改变 target eligibility。测试集 parent/chunk 数、顺序、主体、来源文件与 index hash 在正式预测前写入 manifest；train/val/test 的 subject、row ID、来源记录必须隔离。数据路径跨主机仅允许同一数据集根目录前缀映射；规范化后的有序身份和原始来源文件内容 SHA256 必须一致。

噪声身份中的 split 使用真实 `test`，初始 trajectory0 与后续 trajectory/reverse-step 延用原哈希规则；默认开发入口仍拒绝 test。新工作树显式开放专用 spec 和数据参数，不将 test 伪装成 val。输出文件使用 `research_test_` 前缀，逐行 split 保留 test。P5 历史源码、配置、结果与冻结执行计划保持原状。

指标、窗口加权聚合、raw/postfiltered 双视图及有限性约束均沿用 P5，包括 Whole/Local RR、包络轨迹、全局包络调制、lag-aware signed PCC、幅值尾部和能量诊断；公共指标调用保持 `include_test_only=False`。本阶段不新增评价指标、阈值、checkpoint 选择或训练。三 seed × 三配置全部完成后汇总均值/样本标准差/范围，并与 P5 作描述性比较。不能依据部分 test 结果改变剩余矩阵；P4 的 N 未满足稳定判据结论继续保留。

## 实现与持久产物

本地工作树：`/mnt/disk_code/marques/resp_reconstruction/.worktrees/respdiff-research-test-v1`，分支 `codex/respdiff-research-test-v1`。原始 P5 manifest 的 242 项文件作为移植起点；P6 对 test 数据/噪声准入和输出命名的增量单独记录，数值等价性通过 synthetic 测试核验。

T630：`marques@192.168.199.188`，项目实体根目录 `/data/disk1/cxh/code/resp_reconstruction`。按用户本次远程部署要求，P6 使用该项目下独立 `.worktrees/respdiff-research-test-v1` 快照；本地保留完整相同部署包、源码、输入身份和部署回执。密钥只由 SSH 客户端读取，不进入部署包。

入口为 `scripts/run_respdiff_bcg_research_test_v1.py`，分阶段 `freeze`、`preflight`、`gpu-check`、`run`、`summarize`。真实数据相关阶段要求 `--confirm-research-test`；GPU synthetic 验收要求 `--confirm-gpu-check`。CPU 输入冻结和远程内容核验仅记录身份，不运行测试集模型前向。

输出根为 `runs/respdiff_bcg_research_test_v1/`：`inputs_v1/` 保存只读模型和 P5 来源文件副本，`manifest_v1/` 保存本轮唯一正式身份清单，`remote_preflight_v1/` 保存远程文件完整性与本机路径/stat，`gpu_check_0_v1/` 至 `gpu_check_2_v1/` 保存三个设备的合成验收。正式运行分别为 `test_s20260811_v1/`、`test_s20260812_v1/`、`test_s20260813_v1/`，完整九项汇总为 `summary_v1/`。

## 三卡执行与验收

物理 GPU0/1/2 分别固定 seed20260811/20260812/20260813，每个子进程仅暴露对应显卡、逻辑设备 `cuda:0`。T630 当前为 3×RTX2080Ti（各 11 GiB）；使用现有项目 `.venv`，清除 `LD_LIBRARY_PATH`，记录实际软件、硬件和运行命令，不安装或升级依赖。

先完成 CPU synthetic 的默认 test 拒绝、显式开放、真实 split 噪声身份、train/val 数值兼容、数据/subject 隔离、路径迁移和三配置完整性检查，再在每张卡执行原尺寸 B64/G1 的 synthetic DDIM/DDPM 参考实现与 condition-cache 实现对比。相同输入/噪声容差预设 `atol=1e-5, rtol=1e-4`；非有限、参数变化或对比失败均阻止正式启动。不会通过修改 B、G 或容差来规避失败。

三卡验收全部成功后同时提交三个正式评价。各 seed 独立记录完整失败/完成回执；一项失败不取消另两项已授权的固定预算，完整汇总仅在三项 complete 后执行。所有新输出 exclusive 创建；中断/失败结果保留，恢复须使用新 identity，并仅复用有完整回执的结果。

执行预算按远程 synthetic 首批耗时与完整 test 批数给出估计，并记录到部署回执；P5 的 RTX4070TiSUPER 耗时不直接当作 T630 运行时间。最终验收要求三个完整回执、全部九项配置、完整逐 parent 指标与双视图波形、模型/数据/源码身份前后不变、独立计时、环境记录及产物 hashes。运行期间冻结实际执行源码和本协议，后续进度与结果另用 receipt/closeout 维护。
