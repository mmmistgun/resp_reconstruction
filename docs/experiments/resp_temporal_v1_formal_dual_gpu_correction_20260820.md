# RTM-v1 formal 双GPU执行修订

日期：2026-08-20

状态：**用户在0/15 formal runs时要求分成两组，分别以环境变量绑定一颗GPU。本修订只改变执行调度与provenance，不改变任何科学候选、seed、数据、split、target、loss、metrics、selector、batch、updates或输出identity。**

## 1. 修订原因与旧plan状态

原`configs/resp_temporal_v1/formal_v1.yaml`及其implementation receipt已建立，但尚未产生任何formal run。原runner可接受`CUDA_VISIBLE_DEVICES`，却未把物理可见卡选择写入严格receipt；若直接并行执行，两颗同型号GPU在run内都会显示为逻辑`cuda:0`，无法仅凭receipt还原物理分组。

因此原formal v1 plan SHA-256=`fb1d4652c3e65bbcb1eb4b4b770e96315dbef04011cdae8e66391a33265d0697`冻结为`superseded_before_execution`，不得用于执行。其implementation receipt SHA-256=`0d2dae3fea51f18821a47a11e69d8a8b2ca765898c3d84c228d2fedffcaff0fa`只保留单GPU入口实现provenance，不删除、不覆盖。

## 2. 双GPU硬件与逻辑设备

执行前只读硬件枚举确认：

```text
physical 0: NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB
physical 1: NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB
```

每个进程必须恰好设置`CUDA_VISIBLE_DEVICES=0`或`1`。进程内冻结device仍为逻辑`cuda:0`；既有cuDNN=92000、bf16、显存、设备名与LSTM canary preflight会在实际绑定卡上逐run执行。Runner必须拒绝缺失、多个值或与冻结group不一致的`CUDA_VISIBLE_DEVICES`，并在run manifest/formal receipt中记录group ID与原始环境值。

## 3. 预注册8/7分组

分组在任何formal validation结果产生前固定，并同时平衡candidate与seed，避免让任一candidate或任一seed完全绑定到单卡。

### Group `gpu_0`：`CUDA_VISIBLE_DEVICES=0`，8 runs

1. T0：20260811、20260813；
2. TCN：20260812、20260813；
3. BiMamba2：20260811、20260813；
4. BiLSTM：20260812；
5. Multiscale：20260811。

### Group `gpu_1`：`CUDA_VISIBLE_DEVICES=1`，7 runs

1. T0：20260812；
2. TCN：20260811；
3. BiMamba2：20260812；
4. BiLSTM：20260811、20260813；
5. Multiscale：20260812、20260813。

两组可在两个终端中并行，但每组内部必须按v2 plan列出的顺序串行。任一run失败时只停止该组并保留失败目录；另一组若正在运行可完成当前run，但在原因审计与协议决定前不得启动下一run。不得依据partial validation结果调整分组、停止正常arm或形成family结论。

## 4. 唯一执行入口

本修订后的唯一plan为`configs/resp_temporal_v1/formal_dual_gpu_v2.yaml`。命令形态固定为：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=<LOCKED_GPU> ./.venv/bin/python scripts/train_resp_temporal_v1.py --plan configs/resp_temporal_v1/formal_dual_gpu_v2.yaml --candidate-id <LOCKED_ID> --seed <LOCKED_SEED>
```

除新增的单卡可见性与分组receipt外，`docs/experiments/resp_temporal_v1_formal_protocol_20260820.md`全部科学、数据、训练、selector、failure与输出约束继续有效。Formal v2仍由用户手动执行；Codex不启动GPU或训练。15/15完成前validation summarizer与research-test继续关闭。
