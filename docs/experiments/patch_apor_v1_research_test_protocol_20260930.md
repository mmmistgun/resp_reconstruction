# PATCH/APOR：固定39个checkpoint的research-test协议

日期：2026-09-30。协议ID：`patch-apor-v1-research-test-20260930`。

状态：用户当次明确授权“全部进行test测试”。本附件开放13臂、39个既定validation-selected checkpoint的完整research-test评价与汇总；当次可由Codex执行，沿用两张GPU并行。完成状态与身份以allowlist、evaluation及summary回执为准。

## 1. 固定来源与范围

- 训练session：`runs/patch_apor_v1/session_20260929T181607Z_77e89165fef0`。
- 训练session SHA256：`cdf6337db77d73724f93508a3568669863a17011066e7d332cf69b18b4e98fa2`。
- Validation summary：`summary/attempt_20260930T033450Z_da517e1ea2f1`，manifest SHA256=`e99a9aa7a4b3d933de215bd896e2d659473b6d23ca404f11bbd776ae21bf9638`。
- 模型：M0–M7、A0–A4；seeds20260811/20260812/20260813，39个cell完整覆盖。
- 每个cell只使用`checkpoint_best_local_rr.pt`，按完整validation Local RR最小值、并列最早选点。Allowlist在读取test数组前固定全部checkpoint哈希、epoch、训练config、formal manifest与development subject集合。
- 模型结构、Pi、loss、核心指标、sample聚合与selector均沿用训练合同，test不参与结构、epoch或阈值修改。

本test集合曾用于既有开发，证据属性为 **reused research/development evidence**。当前结果用于固定矩阵的跨split复核，不作为未参与开发的新独立确认集。

## 2. 样本与表示

- Test共2310窗口、8个samp_id，180 s/100 Hz，sample seed20260612。
- Dataset index SHA256沿用`f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f`。
- 有序row ID字节SHA256：`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。
- W条件复用`runs/crd_tf_v1/research_test_cache/40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839`的冻结input-only缓存；manifest SHA256=`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`。原reader核验test W与row ID文件哈希、shape、dtype和覆盖。
- M1只加载时域输入，不读取W cache。其余12臂输入形状为`x[B,1,18000]`与`w[B,97,360]`。
- 读取索引后核验row顺序、数量、test标记、samp_id集合及其与development subject零交集；逐batch再次验证元数据与input/target/W finite。

## 3. 推理和指标

- 原生模型strict加载冻结state dict，eval模式，batch128、BF16；末尾batch完整保留。
- Config只调整操作性device/show_progress；test cache路径添加到独立data config，训练config原文保留并与checkpoint核对。
- 主指标为Whole RR absolute error、Local RR MAE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC，与本轮validation相同。
- 启用原生test辅助指标：IBI及coverage、coherence、constrained nDTW和分层envelope Spearman；保存原生逐seed汇总与全部资格/分母。IBI只在可解释窗口内按原合同计算，必须连同coverage解释。
- 非有限输入、参考、预测、checkpoint或关键指标显式失败。Prediction degeneracy单独记录为质量标记，不通过删除窗口改善结果。

预设比较：M1–M7相对M0；A0相对M0；A1–A4相对A0。保存全部13臂三seedmean/SD、配对差、方向数、subject-macro、尾部、target资格和辅助指标分母。不得依据partial test结果改变剩余矩阵。

## 4. 身份与生命周期

入口为独立脚本`scripts/eval_patch_apor_v1_research_test.py`。训练源码、协议、成功/失败产物及validation summary保持原位冻结。

Allowlist准备先复核训练session源码与全部完成来源，读取既有checkpoint/config/history和test cache manifest；此时`test_array_read=false`，不读取test波形、参考或cache数组，不执行推理。新allowlist与源码快照构成唯一test证据边界，输出到`runs/patch_apor_v1/research_test/allowlist_<UTC>_<uuid>`。

后续每次评价核验allowlist、源码、环境、checkpoint和dataset index身份，登记access_started后才读取test。已成功cell验证manifest后复用，失败/中断保留现场，显式`--retry-failed`才创建新attempt。相同训练session和协议的allowlist复用现有身份，不重复创建已完成科学矩阵。

两张GPU按固定39-cell计划交错分为20/19，各自独立进程、CPU线程4。共享会话锁与cell/分片锁防止重复评价。任何worker失败时父进程停止另一路并保留现场；两路成功后才生成完整39-cell汇总。全部评价新增90090行窗口指标。

## 5. 执行命令

CPU定向验证只使用synthetic/disposable fixtures：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_patch_apor_v1_research_test.py -q
```

固定allowlist（命令返回其实际绝对路径）：

```bash
./.venv/bin/python scripts/eval_patch_apor_v1_research_test.py prepare-allowlist
```

使用返回路径执行全部test：

```bash
PATCH_APOR_TEST_ALLOWLIST='/实际allowlist绝对路径'
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/eval_patch_apor_v1_research_test.py parallel \
  --allowlist "$PATCH_APOR_TEST_ALLOWLIST" --devices cuda:0 cuda:1 --confirm-research-test
```

状态及完整汇总：

```bash
./.venv/bin/python scripts/eval_patch_apor_v1_research_test.py status --allowlist "$PATCH_APOR_TEST_ALLOWLIST"
./.venv/bin/python scripts/eval_patch_apor_v1_research_test.py summary --allowlist "$PATCH_APOR_TEST_ALLOWLIST" --confirm-research-test
```

全部成功后不重复推理或重新选择checkpoint。结果记录另建文件，承接当前状态与完整矩阵结论。

## 6. 实现验证

2026-09-30：15项synthetic CPU定向检查通过（35.90 s），双GPU分片及失败汇总门禁另2项通过（5.33 s）。覆盖最早并列checkpoint选择、39-cell完整性、M1与W条件输入路径、原生test指标、shape/finite/样本顺序、成功产物复用及失败现场保留。三种端到端fixture使用M0/M1/A0，Mamba临时替换为Identity；原生GPU模型前向已由训练session的两卡工程验收覆盖。
