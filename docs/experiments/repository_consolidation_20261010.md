# 仓库归并与产物索引（2026-10-10）

## 当前入口

项目根目录为 `/mnt/disk_code/marques/resp_reconstruction`，下文路径均相对此目录。主工作目录使用 `main`，代码归并点为 `9643628c2cae76cf235356ede5026e136807b68f`。实验阶段状态、执行权限及结论以对应实验协议为准。

## 代码来源

| 来源分支 | 归并来源提交 | 内容 |
| --- | --- | --- |
| 归并前 `main` | `b40c2d1fb98ccbdd12c899c7aa33575cf8280dc1` | CWT/APOR |
| `codex/patch-module-suite` | `6736657741e641533503b9dddfda048e43e8c319` | APOR 受试者敏感性 |
| `codex/h-only-ablation-v1` | `7e552a3ef6c466e96cfdf3538d08dc7859f54d0a` | H-only/H64 |
| `codex/h-only-medical-rr-strata-v1` | `e2de1b136564137a75ef5967fbeb08515e131902` | RR/SA 分层 |
| `codex/patch-aligned-tf-mamba` | `d45d16c553313b3f3a5673e497c183ede402e2d6` | Mamba/P1/M4 及高频机制 |

上述四个 `codex/` 本地分支已完成归并并删除，远端历史分支保留。

## 实体存储与历史路径

项目持久产物统一存放在项目根目录内，新 worktree 使用 `.worktrees/` 下的独立目录。历史绝对路径通过符号链接保留，链接对应的实体均解析到项目根目录内。四个旧 worktree 已转为静态快照并解除 worktree 登记：

| 静态快照路径 | 代码提交 |
| --- | --- |
| `runs/home_storage_migration_20261010/preserved_worktrees/worktrees_20261010/cwt-time-frequency-v1` | `9643628` |
| `runs/home_storage_migration_20261010/preserved_worktrees/worktrees_20261010/model-architecture-review` | `d45d16c` |
| `runs/worktree_preservation_20261010/snapshots/h-only-ablation-v1` | `7e552a3` |
| `runs/worktree_preservation_20261010/snapshots/h-only-medical-rr-strata-v1` | `e2de1b1` |

RespDiff 的迁移登记路径为 `.worktrees/respdiff-paper`；迁移时分支为 `codex/respdiff-paper-settings`，提交为 `df37b3d4437489694f08a6b7817aa3fe37d1b4e8`。历史可视化位于 `runs/home_storage_migration_20261010/visualizations/`。

## 迁存范围与完整性

本次跨盘迁存共 **19,398 个普通文件、106,780,956,486 字节（约 99.45 GiB）**，全量 SHA-256 校验通过。以下统计覆盖本次迁存范围：

| 分项 | 普通文件数 | 容量 |
| --- | ---: | ---: |
| CWT/Mamba 和 Git 指针 | 17,764 | 90.5121 GiB |
| RespDiff | 1,450 | 8.8843 GiB |
| 可视化 | 184 | 52.35 MiB |

源文件系统为 ext4，目标为 fuseblk；目标权限由挂载方式表示，mtime 精度为 100 ns。源端精确元数据与目标端元数据分别留档，仓库使用 `core.filemode=false`。

## 证据入口

`runs/worktree_preservation_20261010/` 保存快照身份及归并验证证据：

- `manifest.json`：四份快照的来源完整提交与历史路径；`<worktree>.inventory.json`：初始身份清单。
- `git_worktree_metadata/`：原 Git worktree 登记信息。
- `m4_cpu_tests.log`：117 passed；`integration_cpu_tests.log`：63 passed，均为合成数据 CPU 验证。

`runs/home_storage_migration_20261010/` 保存迁存与发布证据：

- `plan.json`：20 个入口的路径映射。
- `<name>.sha256.json`：逐文件哈希；`<name>.verified.json`：对应清单哈希。
- `<name>.before.json`、`<name>.destination_metadata.json`：源端与目标端元数据。
- `completion_receipt.json`：最终路径核对；`git_publication_receipt.json`：推送记录。

原始执行日志与过程证据保留在上述本地产物目录。

## 历史代码恢复

按代码来源表中的完整提交恢复代码，在 `.worktrees/` 下选择全新目录；快照对应的完整提交也可从 `manifest.json` 查得：

```bash
git -C /mnt/disk_code/marques/resp_reconstruction worktree add --detach /mnt/disk_code/marques/resp_reconstruction/.worktrees/<全新目录名> <完整提交>
```

静态快照、路径映射和校验清单用于追踪历史产物；继续实验前读取相应协议的当前状态与执行边界。
