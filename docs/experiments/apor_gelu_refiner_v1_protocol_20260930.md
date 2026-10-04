# APOR：统一GELU与条件残差瓶颈的独立对照

日期：2026-09-30。协议ID：`apor-gelu-refiner-v1-20260930`。

状态：用户已授权实现。模型、配置、来源与运行入口已完成，15项synthetic CPU定向测试通过；GPU验收、真实数据训练及validation尚未执行。当前入口仅覆盖train/validation。

## 1. 研究问题与范围

以当前APOR原生140-token结构为共同基准，分别检验：普通隐藏层统一GELU、条件残差H65规整为H64、删除条件残差映射。三个变体各自只改变一个因素，完整矩阵为4 arms × 3 seeds；B0复用三个既有A0 checkpoint/history/validation，新训练9个cell。

当前APOR条件路径是W `[97,360]` → 浅层二维编码与尺度平均 `[96,360]` → 按每个patch的五个真实位置采样 `[96,140,5]` → 展开为`[480,140]` → 480→96线性映射 → 条件残差 → FiLM。该路径不经过1800点插值。本轮保持五点条件映射与时间坐标，条件残差的必要性单独评价。

| Arm | 相对B0的唯一变化 | 参数量 |
|---|---|---:|
| B0 | 原A0/N0，混合GELU/SiLU、H65 | 1,077,640 |
| GELU | 五处普通隐藏层SiLU→GELU | 1,077,640 |
| H64 | 条件残差96→65→96改为96→64→96 | 1,077,448 |
| DIRECT | 条件残差映射改为Identity | 1,065,064 |

GELU的五处变化为frontend adapter后1处、CWT编码2处、条件残差隐藏层1处、片段decoder隐藏层1处。PatchMixer原四处GELU保留。GELU统一限定为这些普通隐藏层；Mamba内部算子、FiLM的tanh和输出线性映射沿用原定义。归一化、分组与epsilon保持不变。

H64采用嵌套初始化：同seed原H65的expand前64行、project前64列及完整project bias复制到H64，不缩放保留权重。其余参数和buffer逐tensor复用同seed原构造初始化。DIRECT只删除条件残差的expand/project三组参数；480→96五点适配及96→192零初始化FiLM投影均保留。所有候选从初始化训练，不从已训练基准继续优化。

H65最初来自分支参数预算匹配，本轮研究其功能作用。H64仅减少192个参数，不能据此宣称显著效率提升。统一GELU是整体激活配置对照；结果不能自动归因于某一个被替换位置。

## 2. 参照来源

- 原session：`runs/patch_apor_v1/session_20260929T181607Z_77e89165fef0`。
- Session SHA256：`cdf6337db77d73724f93508a3568669863a17011066e7d332cf69b18b4e98fa2`。
- B0来源：原A0三seed，selected epoch=8/13/13，seeds=20260811/20260812/20260813。
- 结构决定沿用v2完整validation的N0：`runs/apor_grid_decoder_v2/session_20260930T093614Z_511f2ff06f0f/summary/attempt_20260930T111411Z_d03d6929f568/validation_decision.json`，SHA256=`4e69b062349e4139746cc83c7b7aff4f2e305b27afbdf10b2cdc9fd3a4810a8c`。
- 上一轮SiLU完成记录：[activation v1结果](apor_activation_v1_results_20260930.md)。本轮是新的独立问题和输出identity，不修改前序来源或结论。

`prepare`校验历史来源、训练合同、history、既定checkpoint身份及依赖，保存独立源码快照与resolved配置模板，不打开真实波形/cache数组。实际GPU验收绑定历史环境，B0严格等价核验覆盖三个seed。

## 3. 数据、训练和评价合同

沿用原APOR合同：train/validation为10141/2675窗口、32/7人；180 s、100 Hz；sample seeds=20260610/20260611；原Pi、`L_sync+0.25L_effort`、五项主指标与target eligibility。优化器仍为原AdamW分组，batch128、BF16、最大80 epochs，每epoch 80次更新，上限6400 updates；early stopping min_epoch30/patience15/min_delta0。Checkpoint按完整validation Local RR严格最小、并列最早选择。

新增9次训练上限57,600 updates。禁止根据partial seed改变矩阵、损失、样本或checkpoint规则。非有限input、target、prediction、checkpoint或关键指标显式失败。保持完整样本集合。

三项预设配对对比为GELU−B0、H64−B0、DIRECT−B0。保存逐窗口指标与预测、reference身份、逐seed五指标、三seedmean/sample SD、配对差和改善seed数、逐受试者、subject-macro、RR尾部、资格/退化分母及片段重叠诊断。

各候选独立检查前轮相同的工程保护线：窗口均值与subject-macro两种口径中，四项error各自相对B0增加不超过0.5%，PCC下降不超过0.002，才列入`eligible_arms`。该规则不是统计非劣效检验。汇总保存全部12个既定checkpoint身份；不合并通过候选的因素，也不根据跨因素总分强选单一赢家。新的组合属于后续独立问题。

当前不提供test执行入口。本阶段完整validation之后，固定checkpoint research-test需另有匹配专项附件及当次访问授权；既有test授权不外推到本轮。

## 4. 工程验收与并行执行

入口：`scripts/run_apor_gelu_refiner_v1.py`；模型与运行器为同名前缀的`_model.py`和`_runtime.py`，配置为`configs/apor_gelu_refiner_v1/experiment.yaml`。

CPU定向验证使用合成输入，覆盖公共初始化、H64嵌套初始化、GELU全部五处替换（包含函数调用）、B0非零条件下严格前后向等价、候选三步梯度/optimizer覆盖、checkpoint回载、原生训练器生命周期、独立保护线及完整矩阵门禁。CPU验证以Identity代替Mamba算子；正式Mamba路径由下述GPU验收覆盖。

每张GPU均须完成四臂×三seed的batch1三步更新，以及四臂各一次batch128三步训练和eval；所有活跃参数组第三步有有限梯度及更新，峰值reserved不超过设备容量85%。B0另在隔离确定性进程核验三个seed FP32/BF16非零FiLM前后向严格等价。正式运行沿用历史默认算法设置。两卡验收全部通过后才派发训练。

按seed优先、arm顺序GELU/H64/DIRECT建立九cell矩阵，交错拆成5/4两个固定worker，每进程CPU线程4。完整九cell成功后，父进程汇总含B0的12-cell矩阵。失败或中断attempt保留，显式`--retry-failed`创建新attempt；成功cell核验后复用。

共享validation参考采用有界等待锁（120秒）及同目录临时文件原子发布，后续writer逐元素核对dtype和参考内容。执行控制锁仍立即拒绝冲突。任何写入失败、超时或参考不一致均显式失败；不会发布半写入的参考数组。

## 5. 命令与预期产物

快速CPU验证和静态矩阵查看：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_apor_gelu_refiner_v1.py -q
./.venv/bin/python scripts/run_apor_gelu_refiner_v1.py plan
./.venv/bin/python scripts/run_apor_gelu_refiner_v1.py describe
```

用户执行两卡验收→九次训练→完整validation汇总：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONUNBUFFERED=1 \
  ./.venv/bin/python scripts/run_apor_gelu_refiner_v1.py parallel \
  --devices cuda:0 cuda:1 --confirm-training
```

命令会创建独立session并打印`SESSION`及worker日志路径。也可先`prepare`后传`--session /返回路径`。同一session的恢复需使用原设备分片，检查失败现场后添加`--retry-failed`。

预期输出位于`runs/apor_gelu_refiner_v1/session_<UTC>_<uuid>/`，包含源码/配置、两个GPU验收receipt、九cell的checkpoint/history/metrics/prediction、共同reference、运行环境、访问及来源记录、完整summary与`validation_decision.json`。成功attempt保存manifest/freeze，失败保存traceback；历史产物保持原位。

验收标准：九个新增cell各一个成功attempt，两个worker退出码0，完整12-cell汇总；每cell validation恰为2675窗口/7人，样本顺序和target资格一致，五项主指标有限，全部固定checkpoint身份可追溯。效率使用正式运行记录；本阶段不单独启动benchmark。
