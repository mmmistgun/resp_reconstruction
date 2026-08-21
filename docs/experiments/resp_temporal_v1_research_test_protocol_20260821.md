# RTM-v1 reused research-test 协议

日期：2026-08-21

状态：**用户已明确授权实现并在实现验收后手动执行一次；尚未访问RTM-v1 research-test signal/target、生成prediction或形成test结果。**

协议 ID：

```text
resp-temporal-v1-reused-research-test-20260821
```

## 1. 目的与证据边界

本阶段只回答：五个已冻结RTM-v1 candidate及其三个validation-selected checkpoint在既有research-test split上的五项主指标是多少。证据标签固定为：

```text
reused research-test evidence; candidate and checkpoint selection frozen before RTM-v1 test access; not untouched held-out evidence
```

该split在仓库既有研究中已被使用，因此不得表述为首次触碰、无偏held-out、外部泛化或确认性检验。Research-test不参与checkpoint、candidate、epoch、超参数或模型家族重选。

## 2. 冻结输入

- candidate固定为validation lock中的五项，顺序为T0、TCN、BiMamba2、BiLSTM、Multiscale；
- 每项固定seed `20260811/20260812/20260813`，共15个checkpoint；
- checkpoint固定为各formal目录中的`checkpoint_best_local_rr.pt`，其选择器仍是完整validation Local-RR strict `<`；
- checkpoint身份必须由冻结validation summary receipt中的15项formal receipt SHA-256继续追溯到formal artifact manifest和checkpoint SHA-256，运行时重新计算文件hash后才可反序列化；
- 不读取validation metrics/prediction/target，不重新评价validation，不读取`checkpoint_final.pt`；
- 数据、split、input、target、Pi、loss、metrics及聚合口径均不改变。

Research-test数据身份固定为：dataset index SHA-256=`f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f`，`2310` windows、`8` samp_ids、dataset-row-id SHA-256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`，sample seed=`20260612`。运行时必须先按split过滤metadata，只允许test行进入adapter和signal reader。

## 3. 唯一评价与汇总口径

用户只执行一次冻结入口；入口在单个进程中依次评价15个checkpoint，不训练、不更新参数、不保存prediction array。每个seed保存逐sample metrics并对五项primary做sample direct mean；每个candidate再对三个seed报告arithmetic mean和sample SD（ddof=1）。

只输出五项主指标：

1. `whole_rr_abs_error_bpm`；
2. `local_rr_mae_bpm`；
3. `envelope_trajectory_mae`；
4. `global_envelope_modulation_error`；
5. `lag_aware_signed_pcc`。

不计算test-only supplementary metrics、Pareto、paired方向、materiality、总分、排名或p-value。Research-test只报告冻结五候选的`mean ± SD`表，不以结果追加训练、调参、容量点或复评。

## 4. 访问、失败与复现合同

- 执行前要求干净Git、冻结config/protocol/validation lock/summary receipt哈希闭合；
- 要求unset `LD_LIBRARY_PATH/LD_PRELOAD`、cuDNN runtime=`92000`及LSTM canary通过；
- 只允许单一可见GPU上的逻辑`cuda:0`，AMP固定bf16，batch固定128；
- 输出固定为`runs/resp_temporal_v1/research_test/rtm_v1_research_test_v1/`，目录必须预先不存在；
- 输出创建后若失败，保留failed lifecycle，不删除、不覆盖、不resume且不得以同identity重跑；
- complete receipt必须登记15/15 checkpoint、34650逐sample rows、5行candidate表、finite/null/nonfinite、每个输入与输出SHA-256及access flags；
- complete后入口只作provenance，不得重复运行。

长时间GPU评价只由用户手动执行。Codex只实现、运行synthetic/CPU定向测试并提供命令；收到complete receipt前不读取test结果或形成结论。
