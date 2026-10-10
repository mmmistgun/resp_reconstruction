# 项目产物存储迁移记录（2026-10-10）

本项目的持久产物实体统一保存在 `/mnt/disk_code/marques/resp_reconstruction`。本记录接续 [分支归并记录](repository_consolidation_20261010.md)，更新其中的当前存储位置；原有实验结果、来源清单及历史哈希继续保留。

## 清点与位置

本次确认位于 `/home/marques` 的项目源码与产物共 **19,398 个普通文件、106,780,956,486 字节（99.45 GiB）**。分类如下：

| 类别 | 普通文件数 | 逻辑大小 | 当前实体位置（相对于项目根目录） |
| --- | ---: | ---: | --- |
| CWT/Mamba 静态快照与 Git 指针记录 | 17,764 | 90.5121 GiB | `runs/home_storage_migration_20261010/preserved_worktrees/` |
| Respdiff 开发 worktree 与产物 | 1,450 | 8.8843 GiB | `.worktrees/respdiff-paper/` |
| 历史可视化、标注文件及配套源码 | 184 | 52.35 MiB | `runs/home_storage_migration_20261010/visualizations/` |

两个静态快照的完整位置为：

- `/mnt/disk_code/marques/resp_reconstruction/runs/home_storage_migration_20261010/preserved_worktrees/worktrees_20261010/cwt-time-frequency-v1`
- `/mnt/disk_code/marques/resp_reconstruction/runs/home_storage_migration_20261010/preserved_worktrees/worktrees_20261010/model-architecture-review`

原 `/home/marques/resp_reconstruction_artifacts`、Respdiff worktree 路径，以及 18 个可视化目录入口改为指向项目内实体的兼容链接。原 CWT/Mamba worktree 链接经上述入口仍能解析到保存快照。后续创建产物直接使用项目内路径。

Respdiff 的 Git worktree 登记更新为 `.worktrees/respdiff-paper`，分支仍为 `codex/respdiff-paper-settings`，代码提交保持 `df37b3d4437489694f08a6b7817aa33575cf8280dc1`。

## 迁移与验证

迁移按“复制 → 全量校验 → 切换路径 → 清理源端实体副本”执行，使用 `rsync -aHAX --numeric-ids` 复制。逐文件比较两端 SHA-256，所有普通文件均参与，包括大 checkpoint/cache 文件。切换前再核对文件集合和元数据，检测并发修改；源端清理只在副本已校验后进行。

源盘为 ext4，目标盘为 fuseblk。目标的权限由挂载方式表示，mtime 精度为 100ns；源权限和精确时间保存在 `before.json`，目标元数据另存为 `destination_metadata.json`。跨盘核对要求文件类型、大小和符号链接目标相同，时间仅允许不足 100ns 的截断差异，普通文件内容由全量 SHA-256 确认。仓库原有 `core.filemode=false` 保持不变。

每个原目录切换为兼容链接，历史绝对路径仍可访问。Respdiff 的 Git 登记使用 `git worktree repair` 修复。实验源码和产物内容保持原值，迁存不改变实验口径或阶段状态。

证据根目录：`runs/home_storage_migration_20261010/`。

- `plan.json`：20 个迁移入口的源、目标、文件数、逻辑字节及占用字节。
- `<name>.before.json`：源目录的逐路径类型、大小、mtime、权限和符号链接目标。
- `<name>.destination_metadata.json`：内容校验完成时的目标元数据，用于切换前后核对。
- `<name>.sha256.json`：每个普通文件的 SHA-256，以及目录和链接元数据。
- `<name>.verified.json`：各入口的校验数量和清单 SHA-256。
- `copy_verified.json`、`<name>.cutover.json`、`cutover_complete.json`：全量校验及源端切换记录。
- `completion_receipt.json`：最终 Git 身份、路径解析、实体存储边界和源端清理核对。
- `migrate_home.py`、`copy_and_hash.log`、`*.rsync.log`：迁移实现及执行日志。

首次复制后的检查因 ext4/fuseblk 元数据差异停止，源文件全部保留。确认目标权限表示和 100ns 时间精度后，继续对已复制副本执行全量内容校验。该次停止日志保留为 `copy_and_hash_attempt01_failed.log`，对应脚本为 `migrate_home_attempt01.py`。

首次 worktree 整理的清单、Git 元数据和失败记录仍保存在 `runs/worktree_preservation_20261010/` 及同级失败记录目录，内容未重写。其中历史 inode/device 描述首次同盘迁存时的身份，本次跨盘迁移由新 SHA-256 清单承接。

## 清点边界

扫描了 `/home` 可访问目录的项目名称线索，核对项目文档中的 `/home` 路径，并检查历史可视化及常用用户文件夹。名称相似的第三方库文件未当作项目产物。通用开发环境、全局应用状态和会话索引由其应用维护。

额外发现 9 个 Claude 会话记录文件、42,791,388 字节（40.81 MiB），其中有私有权限文件；目标 fuseblk 不能保持其访问权限，因此作为应用状态保留原位，未复制。最初清点计划保留于 `inventory_plan_initial.json`。识别此权限边界后，第二次校验进程主动停止以调整迁移范围，日志为 `hash_attempt02_stopped.log`；随后按最终计划完整校验。

目录扫描中有八个权限拒绝，涉及其他用户目录及其他服务的私有数据目录，详情保存在 `home_name_inventory.json`；不能据此声称已经检查 `/home` 的所有文件。用户已确认下载目录中的 `骏丰测试数据.zip` 属于其他项目，该文件保留原位。

此迁移只读写已有文件以校验和改变存储位置，不执行训练、评价、真实数据解析或指标重算。
