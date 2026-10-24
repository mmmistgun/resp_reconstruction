# RespDiff-BCG P6：T630 三卡部署记录

当前结果见 [P6 完整结果](respdiff_bcg_research_test_results_20261024.md)，评价、汇总与回收核验均已完成关闭。

部署状态（2026-10-19）：**三个正式 research-test 进程并行进入采样。** 18:13:15（Asia/Shanghai）提交三个模型，18:13:49 均开始 DDIM N16/N64 的完整 test sweep。固定合同见 [P6 专项协议](respdiff_bcg_research_test_v1_protocol_20261019.md)，P5 完整结果见 [validation 结果](respdiff_bcg_p5_validation_results_20261019.md)。以下保留初始部署及验收记录。

## 部署身份

远程主机：`marques@192.168.199.188`，实际 hostname `dell-PowerEdge-T630`。

远程独立工作树：`/data/disk1/cxh/code/resp_reconstruction/.worktrees/respdiff-research-test-v1`。

本地完整副本：`/mnt/disk_code/marques/resp_reconstruction/.worktrees/respdiff-research-test-v1`，分支 `codex/respdiff-research-test-v1`。远程工作树基于其现有仓库的 `dd47e41`，本地起点为 `3766c37`；部署复制本地实际源码及输入，以逐文件内容清单和源码快照确定运行身份，不以两个基底 commit 相同作为前提。

唯一正式清单：`runs/respdiff_bcg_research_test_v1/manifest_v1/research_test_manifest.json`，SHA256 为 `0113fd0910e9466acef90ab2404ab64df41b47f8e30f626cd7e96c729f210aa4`。

该清单固定三个原 final16000 模型、三个配置、245 项执行源码/协议文件、38 项模型/P5 输入文件，以及 94 个数据来源 NPZ 的内容 hashes。P5 到 P6 的七个已有源码文件增量明确记录在清单中：仅增加显式 test 准入、真实 split 噪声域及输出前缀；原 P5 工作树与清单保持冻结。

## 数据与验证

完整 test 为 **2310 个 parent、30030 个 chunk、8 名受试者**，B64 共 470 批，末批 B14。test 主体为 `[220,229,286,670,671,704,726,1006]`。train/val/test 的主体、row ID、来源记录隔离核验通过，三个 split 的有序规范化身份在远端一致。

远程 `remote_preflight_v1` 已完整通过：模型输入文件内容、index hash、94 个来源 NPZ 内容均与本地冻结值相同。路径迁移仅将数据集根映射到远程 `/data/disk1/cxh/code/resp_prepare/dataset/...`；保留原配置中的兼容入口与原始来源副本。

CPU 定向检查 **22 项通过**，覆盖 test 默认拒绝/显式开放、原噪声哈希算法、真实 test 域、val 兼容、共享 N16/N64 前缀、参考/缓存采样、身份隔离、跨根路径映射、test 输出标识和原 P5 编排验证。控制器的模拟检查覆盖三卡映射、全部 GPU 验收通过才开放正式任务、失败回执保留、独立任务完成及完整汇总条件。

三张 RTX2080Ti 分别执行原尺寸 B64/G1 synthetic DDIM/DDPM 验收，**六项比较全部通过，最大绝对差均为 0**，模型参数/buffer 未变；峰值 allocated 约 2.13 GiB。验收使用合成数据，不根据 test 表现修改数值容差。

环境沿用远程现有 `.venv`：PyTorch 2.12.0+cu130、NumPy 1.26.4、SciPy 1.17.1、pandas 3.0.3、OmegaConf 2.3.0。清除 `LD_LIBRARY_PATH`，各进程 OMP/MKL 线程为 4，逻辑设备均为 `cuda:0`；未安装或升级依赖。

## 三卡正式运行

| 物理 GPU | training seed | PID | 输出（相对运行根目录） | 估算纯采样时长 |
|---|---:|---:|---|---:|
| 0 | 20260811 | 37475 | `test_s20260811_v1` | 48.70 小时 |
| 1 | 20260812 | 37476 | `test_s20260812_v1` | 49.08 小时 |
| 2 | 20260813 | 37477 | `test_s20260813_v1` | 47.00 小时 |

三个进程实际 `CUDA_VISIBLE_DEVICES` 分别为 0/1/2，均已记录 `parents=2310 chunks=30030 batch=64` 并进入 DDIM sweep；检查时每卡进程显存约 2506 MiB。

运行根目录为远程工作树下的 `runs/respdiff_bcg_research_test_v1/`。控制器 PID **37285**，入口 `launch_t630_v1/launch.py`，通过 SSH 启动并脱离交互连接持续运行。控制器自动等待三项完成，再生成 `summary_v1/` 的九项 test 表、跨 seed 汇总和 validation/test 对照。各任务失败显式记录，完整汇总要求三个任务全部成功。

预算来自各 GPU 的 synthetic B64/N1 计时按固定 N 与 470 批外推；三个模型并行，总墙钟主要取决于最慢任务。加上数据准备、指标、保存和独立计时，预留 **2–2.5 天**。实际吞吐可能随负载变化。

## 日志与本地证据

远程日志：

```bash
ssh -i ~/.ssh/ubuntu805 marques@192.168.199.188 \
  'tail -n 30 /data/disk1/cxh/code/resp_reconstruction/.worktrees/respdiff-research-test-v1/runs/respdiff_bcg_research_test_v1/launch_t630_v1/console.log'
```

同目录 `run_gpu0_console.log`、`run_gpu1_console.log`、`run_gpu2_console.log` 分别为三个正式进程的进度；`budget_estimate.json` 保存预算。正式运行实体产物保存在 T630 项目目录，任务完成后再按完整回执核验并同步需要的结果。

本地 `runs/respdiff_bcg_research_test_v1/deployment_evidence_v1/` 已保存远程 preflight、三卡验收和启动控制记录；下载的 12 项已完成阶段产物 hashes 校验通过。密钥内容未复制到代码、输入包或产物。

运行期间冻结 P6 工作树的执行源码和专项协议。主工作区的文档路由与本记录用于查看当前状态；后续结果与恢复使用独立 receipt/closeout，保留全部已有产物。
