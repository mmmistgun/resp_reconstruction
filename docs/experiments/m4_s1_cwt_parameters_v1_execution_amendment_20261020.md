# M4-S1 正式执行增量修订：JSON回执绑定兼容

日期：2026-10-20。适用范围：`m4-s1-cwt-parameters-v1-20261020` 正式train/validation启动和恢复。[原执行记录](m4_s1_cwt_parameters_v1_formal_execution_20261020.md)的矩阵、数据、训练和评价合同继续有效；本记录更新启动入口与生命周期，不修改冻结session或原源码快照。

## 原因与修复

首次启动GPU0在复用已完成B验收时显式失败：`ValueError("阶段绑定不符")`。诊断显示唯一差异为 `architecture.stem_dilations`：运行时是元组 `(1,2,4,8,16)`，JSON回执是列表 `[1,2,4,8,16]`；两者规范JSON哈希相同。GPU1被主动中断，以便统一恢复。两个原cell失败attempt及中断的GPU1 B合成验收现场均保留；此时正式训练完整epoch数为0。

新增启动兼容层 `scripts/run_m4_s1_receipt_compatibility.py`，在调用原冻结入口期间，将期待的阶段绑定进行严格JSON往返后交给原 `verify_stage`。原检查仍逐项比较完整绑定、manifest/receipt身份及全部产物字节；真实字段差异仍失败。上下文退出时恢复原函数。该改动仅处理JSON元组/列表表示，不影响模型、训练、loss、数据、指标、checkpoint selector或数值验收阈值。

兼容层作为明确的执行增量记录和保存源码快照，不加入或改写已冻结session的源码文件。后续本session操作使用兼容入口；它接受原入口全部参数。原始源码及配置继续由原 `load_session` 完整核验，兼容层另外由启动manifest和快照约束。既有验收只读复用。

定向synthetic CPU验证：**3 passed，0.12秒**。覆盖元组/列表兼容、原回执字节不变、调用后恢复原函数，以及batch/架构字段漂移和产物篡改继续拒绝。真实冻结B回执复用检查通过，原session源码身份仍通过。

## 恢复执行

新启动目录：

`/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/formal_launch_v2_20261020/`

保存兼容层、定向测试快照、完整84-cell固定分片、manifest/receipt、worker及两卡生命周期日志。原 `formal_launch_v1_20261020` 全部保留。

先在GPU1用兼容入口执行 `smoke --arms B --device cuda:1 --confirm-synthetic --retry-failed`，创建独立attempt并保留中断现场；GPU0成功验收不重跑。通过后双worker使用原固定分片，并在 `run` 命令追加 `--resume`，恢复已建立的两个cell生命周期，其余82个cell照常从头创建。所有正式cell保持最多80epoch、每epoch80update及既定早停规则。

两worker实际入口为：

```bash
scripts/run_m4_s1_receipt_compatibility.py run \
  --session /mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/acceptance_v1_20261020 \
  --device cuda:0 --shard-index 0 --shard-count 2 --confirm-training --resume
```

GPU1使用 `cuda:1`、`--shard-index 1`；环境、解释器和工作目录沿用原执行记录。research-test仍未开放。
