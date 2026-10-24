# 第二台工作站环境

记录日期：2026-10-06。

## 主机与目录

| 项目 | 配置 |
| --- | --- |
| 主机名 | `dell-PowerEdge-T630` |
| 系统 | Ubuntu 24.04.5 LTS（Noble Numbat） |
| SSH 地址 | `marques@192.168.199.188` |
| GPU | 3 × NVIDIA GeForce RTX 2080 Ti，每张 11264 MiB |
| NVIDIA 驱动 | `595.91.07` |
| 项目目录 | `/data/disk1/cxh/code/resp_reconstruction` |
| Python 解释器 | `/data/disk1/cxh/code/resp_reconstruction/.venv/bin/python` |

数据访问入口：

```text
/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf
```

索引为该目录下的 `training/dataset_index.csv`。

## 软件版本

以下版本依据该机终端输出记录。

| 组件 | 版本 |
| --- | --- |
| Python | `3.12.13` |
| uv | `0.12.23`（x86_64-unknown-linux-gnu） |
| PyTorch | `2.12.0+cu130` |
| PyTorch CUDA runtime | `13.0` |
| cuDNN | `9.20.0`（`torch.backends.cudnn.version()` 返回 `92000`） |
| causal-conv1d | `1.6.2.post1` |
| mamba-ssm | `2.3.2.post1` |
| NumPy | `1.26.4` |
| SciPy | `1.17.1` |
| pandas | `3.0.3` |
| OmegaConf | `2.3.0` |

完整已安装版本见[依赖快照](workstation2_poweredge_t630_requirements.txt)，来自该机 `uv pip freeze --python .venv/bin/python` 的输出，用于环境版本核对。

## 运行方式

使用项目虚拟环境的解释器，并在启动 Python 进程时清除 `LD_LIBRARY_PATH`，使 PyTorch 使用当前环境的 CUDA/cuDNN 库。

```bash
cd /data/disk1/cxh/code/resp_reconstruction

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 .venv/bin/python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
PY
```

各独立任务通过 `CUDA_VISIBLE_DEVICES=0`、`1` 或 `2` 选择物理 GPU；每个进程只暴露一张卡时，程序内的设备编号为 `cuda:0`。
