# CWT 时频系列：正式train/validation执行交接

日期：2026-10-01。用户在完整工程验收通过后明确要求执行正式GPU训练。

## 授权与矩阵

执行已批准的20配置×3seed=60-cell矩阵，固定80epochs、batch128、BF16、原loss/metrics/optimizer、完整validation Local RR严格最小选点。包括正式训练必需的train/validation数据身份准备、CWT缓存、TF-S1信号检查、S3分层阈值固定及最终validation汇总。本次不运行research-test或新增干预评价。

正式训练使用独立`execution_scope=train_validation`会话，保留原仅合成会话。验收来源：

`runs/cwt_time_frequency_v1/session_20260930T182730Z_84618e17624b`。

该来源两卡各80个更新/eval用例及6组GN检查通过，最高reserved约89.42%，显存硬上限95%。新会话只读复用原GPU验收和合成校准，不重跑成功阶段，不修改原回执。

## 来源核验与参数核查

正式交接仅允许阶段范围/回执引用适配器、相应测试和新启动入口变化；模型、CWT、校准计算、训练器、数据加载、loss和metrics等实际科学依赖必须保持逐字节一致。全部差异及新增文件身份随session记录，任一未声明实际依赖变化即失败。

以冻结索引核对train/validation全部39名受试者的导出signal-bank元数据：100 Hz，rawish低/高截止0.03/20 Hz，filter_order=4。保存每份元数据的文件身份、实际CWT尺度/帧数/中心频率/重复中心、有限窗冲激边界半径，以及标称四阶Butterworth零相位滤波响应。

频率解释保持原计划：宽带滤波之后有对齐和soft-z非线性处理，标称滤波响应不等于最终输入的线性传递函数。保留原20配置和频带端点，不依据新信号关联或validation中间值裁剪矩阵。边界影响和低方差的未定义关联保留分母。该核查由Codex在用户已批准矩阵与执行授权内完成并记录，不伪称用户亲自进行了数值核查。

## 后台顺序与失败规则

1. 验证原验收、源码差异、环境、磁盘预算，生成正式会话。
2. `prepare-data`：只使用train/val，保存样本身份、源文件哈希、公共validation reference、预选案例及固定shift。
3. 构建C_20缓存；完成既定TF-S1信号检查和train-derived分层阈值。
4. 构建其余19配置的原生CWT缓存。
5. 复用两卡验收，按既定交错分片执行60-cell正式train/validation；全部完成后汇总。

CPU准备先于GPU训练，准备阶段GPU可能空闲。实际阶段以主日志`PHASE_START/PHASE_COMPLETE`及对应回执为准，排入后台流程不等于GPU已开始更新。训练/准备任一失败会保留traceback并停止后续步骤，不自动重训或改变矩阵。

## 入口

```bash
cd /home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -u \
  scripts/run_cwt_time_frequency_v1_formal.py \
  --engineering-session /home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction/runs/cwt_time_frequency_v1/session_20260930T182730Z_84618e17624b \
  --devices cuda:0 cuda:1 --confirm-training
```

日志打印新`SESSION`。缓存、信号分析、worker日志、60-cell产物和汇总均位于该独立会话。执行状态以产物回执为准；本说明不预先声明正式训练成功。
