# RTM-v1 GPU engineering v1 failure 与 v2 correction

日期：2026-08-20

状态：**v1 已按 fail-closed 合同终止并冻结；只允许修正统一运行环境 preflight 后在新 v2 输出目录执行一次。Formal training、validation evaluation 与 research-test 仍未授权。**

## 1. v1 冻结事实

用户从干净commit `5d7cea0ea914ab833bb06201d4bf13a1b9c15d5f`手动执行v1。固定产物位于：

```text
runs/resp_temporal_v1/gpu_engineering/rtm_v1_gpu_engineering_v1/
```

- access receipt SHA-256=`3ae8e96ebfbcb2ef1eeb75134088d364bea085ed37fbe9e72041d569f50d3ad0`；
- manifest SHA-256=`96b8c5ddddf62df2d890e3293ceb69804c97ca96dee37702981222a8daeae28c`；
- status=`failed`，actual candidates=`3/5`；
- T0/TCN/BiMamba2完成且finite，BiLSTM在model迁移到CUDA时触发非OOM RuntimeError，multiscale未执行；
- dataset/index/train/validation/research-test/checkpoint访问与formal training均为false；
- numeric finite/null/nonfinite=`189/6/0`，全部artifact hashes与receipt hash闭合。

失败信息固定为：PyTorch `2.12.0+cu130`编译要求cuDNN `9.20.0`，该进程实际加载cuDNN `9.8.0`。错误明确指出外部`LD_LIBRARY_PATH`覆盖了PyTorch wheel自带cuDNN。该失败是统一运行环境不兼容，不是BiLSTM候选的能力、显存或数值失败。

v1目录不得删除、覆盖、补写或重跑。前三项partial measurements不构成完整统一工程比较，不进入candidate选择或formal gate。

## 2. v2 唯一修正

v2不得改变候选、模型、synthetic tensors、precision、loss、optimizer、batch/repeat、physical-batch ladder、统计或wall-time口径。唯一变化为：

1. 手动命令必须在进程启动前unset `LD_LIBRARY_PATH`与`LD_PRELOAD`；
2. harness在创建v2临时/最终输出目录前，严格确认两个变量均未设置；
3. harness在创建输出目录前确认runtime cuDNN恰为`92000`，并执行一个最小CUDA cuDNN LSTM canary及同步；
4. 任一环境/canary失败都在输出目录创建前退出，因此不消耗v2唯一输出路径；
5. v2 provenance必须验证v1 frozen receipt/manifest hashes与失败分类；
6. v2固定发布到新目录`rtm_v1_gpu_engineering_v2`，绝不触碰v1。

这不是候选、容量或benchmark问题的事后修改，只补上原本应在五项候选前完成的统一依赖preflight。

## 3. 唯一手动命令

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python scripts/benchmark_resp_temporal_v1_gpu.py --config configs/resp_temporal_v1/gpu_engineering_v2.yaml
```

固定输出目录：

```text
runs/resp_temporal_v1/gpu_engineering/rtm_v1_gpu_engineering_v2/
```

成功/失败schema、不可覆盖、原子发布、access/hash/finite验收与v1相同。用户返回v2完整receipt/manifest/hash后才能形成GPU engineering lock；formal training仍需另行明确授权。
