# 分支归并与 worktree 产物保存记录（2026-10-10）

## 代码归并

代码归并点为 `9643628c2cae76cf235356ede5026e136807b68f`，包含以下分支的完整提交历史。主工作目录 `/mnt/disk_code/marques/resp_reconstruction` 使用 `main`。

| 来源分支 | 来源提交 | 内容 |
| --- | --- | --- |
| `main`（归并前） | `b40c2d1fb98ccbdd12c899c7aa33575cf8280dc1` | CWT 与 APOR 实现和实验记录 |
| `codex/patch-module-suite` | `6736657741e641533503b9dddfda048e43e8c319` | APOR 系列与受试者敏感性分析 |
| `codex/h-only-ablation-v1` | `7e552a3ef6c466e96cfdf3538d08dc7859f54d0a` | H-only/H64 消融实现 |
| `codex/h-only-medical-rr-strata-v1` | `e2de1b136564137a75ef5967fbeb08515e131902` | 呼吸率与 SA 分层分析 |
| `codex/patch-aligned-tf-mamba` | `d45d16c553313b3f3a5673e497c183ede402e2d6` | 时频 Mamba、P1/M4 消融和高频机制分析 |

M4 待提交的 21 个文件以 `d45d16c` 保存。合并冲突仅涉及 `AGENTS.md` 与 `scripts/README.md` 的入口增补，保留各实验入口并去除重复条目。实验源码、指标口径及冻结结果保持原身份。H-only/H64 消融协议中的待执行阶段仍为待执行，分支归并不改变实验阶段状态。

上述四个 `codex/` 本地分支在归并后清理；对应远端分支保留，作为历史来源入口。Respdiff 分支、worktree 和主目录中的未跟踪计划文件维持原状。

## 静态快照与历史路径

四个 worktree 的源码、产物及被 Git 忽略的文件整体迁存，原路径以符号链接连接到保存位置。迁存使用同一文件系统内的目录重命名；源数据文件未跨盘复制。Git 指针和 worktree 登记元数据分别保存后，清理这四个 worktree 的登记。

| 原 worktree | 快照保存位置 | 快照代码提交 | runs 文件数 |
| --- | --- | --- | ---: |
| `/home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction` | `/home/marques/resp_reconstruction_artifacts/worktrees_20261010/cwt-time-frequency-v1` | `9643628` | 6,578 |
| `/home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction` | `/home/marques/resp_reconstruction_artifacts/worktrees_20261010/model-architecture-review` | `d45d16c` | 8,929 |
| `/mnt/disk_code/marques/resp_reconstruction/.worktrees/h-only-ablation-v1` | `/mnt/disk_code/marques/resp_reconstruction/runs/worktree_preservation_20261010/snapshots/h-only-ablation-v1` | `7e552a3` | 0 |
| `/mnt/disk_code/marques/resp_reconstruction/.worktrees/h-only-medical-rr-strata-v1` | `/mnt/disk_code/marques/resp_reconstruction/runs/worktree_preservation_20261010/snapshots/h-only-medical-rr-strata-v1` | `e2de1b1` | 46 |

快照共 19,723 个普通文件，其中 `runs/` 为 15,553 个文件、97,160,373,059 字节（90.49 GiB）。CWT 快照包含本次归并后的代码；归并前代码可由首表提交恢复。主工作目录原有的 `runs/` 产物保持原位。

这些目录作为静态快照保留；日常开发使用主工作目录。旧绝对路径及通过旧 worktree 根目录解析的相对产物路径仍可访问。快照已解除 Git worktree 身份，需要历史开发环境时，在新的空路径恢复对应提交：

```bash
git -C /mnt/disk_code/marques/resp_reconstruction worktree add --detach /新的空目录 <首表或快照表中的完整提交>
```

产物保存于本机，未上传 Git；此次同盘迁存属于整理和追溯保全，不提供独立磁盘故障备份。

## 校验与证据入口

本地产物记录根目录：`/mnt/disk_code/marques/resp_reconstruction/runs/worktree_preservation_20261010`。

- `manifest.json`：四个快照的来源分支、完整提交、原路径、保存路径、文件数和各清单 SHA-256。其 SHA-256 为 `f99bb820fb774e31ecb656c70e12b0f0ec2023de2dbecb178cf9cf48f0ff5ebb`。
- `<worktree>.inventory.json`：迁存前逐条登记相对路径、设备、inode、权限类型、大小、mtime 和符号链接目标。迁存后通过原路径重新读取并逐项比较，全部一致。
- 所有不大于 4 MiB 的普通文件另计算 SHA-256，迁存前后一致；更大文件依据同盘重命名及上述文件身份校验确认保存，未声明重新全量计算大文件内容哈希。
- `<worktree>.prepared.json`、`<worktree>.verified.json`：迁存前身份和完成校验记录；Git 指针记录路径见 `verified.json`。
- `git_worktree_metadata/`：清理前的四组 Git worktree 登记、索引和日志。
- `preserve_worktrees.py`、`preservation_verified.log`：迁存脚本和成功日志。
- `completion_receipt.json`：最终主线提交、远端核对、剩余 worktree 与历史路径校验结果。

首次尝试在移动 Git 指针文件时触发跨设备错误，立即恢复工作树后修订为同盘保存，再执行成功。首次清单保留于同级 `worktree_preservation_20261010_attempt01_failed/`，错误日志为本记录根目录中的 `preservation_failed.log`。

## 验证

验证均使用合成或临时 fixture，限制 CPU 线程并关闭 CUDA 可见性：

- M4 组件、research-test 入口合同及高频机制定向测试：117 passed，69.25 秒，见 `m4_cpu_tests.log`。首次工具调用在 60 秒时超时终止，随后完整重跑通过。
- 归并后的 H-only/H64、医学呼吸率、SA 与 M4/IEWT 分层定向测试：63 passed，22.68 秒，见 `integration_cpu_tests.log`。
- 合并来源祖先关系、差异格式、迁存前后文件身份及旧路径链接均通过核对。Mamba 分支相对归并点的既有实验源码未改动；差异为 H-only/分层新增文件及文档入口。

本次整理未启动实验或读取真实数据、独立测试集。
