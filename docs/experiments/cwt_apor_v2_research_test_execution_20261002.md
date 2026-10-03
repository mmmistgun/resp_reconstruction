# CWT-APOR v2固定checkpoint research-test执行附件

日期：2026-10-02（Asia/Shanghai）。主协议：`cwt-apor-v2-20261001`。用户当次明确要求“开启test”，授权本附件的固定checkpoint评价及预定机制分析。实际完成状态以各阶段manifest/receipt为准。

## 来源与固定范围

- 正式会话：`runs/cwt_apor_v2/session_20261001T042244Z_e5e53ad16731`。
- validation汇总：`summary/attempt_20261001T154248Z_f22f23ce3ef0`；60/60 cell均完成。
- 使用全部20配置×3seed的validation-selected best Local RR checkpoint；在首次test数据访问前生成60项allowlist，绑定checkpoint/config/history及正式manifest。
- A0三个seed的selected epoch分别为8、13、13；所有其他cell的epoch及完整SHA256以allowlist为准。
- 保持主协议模型、表示、指标、分母、配对方向、训练集分层阈值、shift seed及案例规则。test结果不用于重新选择checkpoint、调整矩阵或更改阈值。

## 评价与机制分析

1. 核验test行身份及与train/validation的受试者隔离，保存公共参考、固定shift及每受试者首/中/末案例。
2. 为20种表示分别构建test cache，固定60个checkpoint执行推理和指标导出。
3. 生成三个预定视图：full=2310窗/8受试者，exclude670=2231窗/7受试者，subject670=79窗/1受试者；完整视图与敏感性视图同时保留。
4. 使用A0自己的三个固定checkpoint执行18条件TF-S6/S7：FULL/NAT、FULL/FIXED，以及H=(0.8,8]、H2=(2,8]各自WINDOW_MEAN与三个共同SHIFT×NAT/FIXED。采用原有batch8，验证FULL与常规test评价重放一致，保存指标、案例及FiLM/潜在表征。
5. 完整三seed机制结果按相同三个视图汇总。

## 调度与运行边界

沿用已完成的两卡APOR工程验收。先冻结allowlist并准备test数据；两卡各负责10种表示的缓存和三seed评价，随后完整汇总。机制分析三个seed顺序执行，以控制内存和案例导出的磁盘峰值。科学计算使用既有已冻结实现，新增调度脚本和本附件随pipeline保存副本及身份；不改写原session、训练源码快照或历史产物。

任一数值、身份、样本分母或重放检查失败时停止后续步骤并保留失败现场。已完成的阶段不重跑；失败重试须在查明原因后使用独立attempt。入口不包含训练动作。

```bash
cd /home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -u scripts/run_cwt_apor_v2_test_pipeline.py \
  --session runs/cwt_apor_v2/session_20261001T042244Z_e5e53ad16731 \
  --devices cuda:0 cuda:1 --confirm-research-test
```
