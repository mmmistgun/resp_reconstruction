# M4-S1 CWT参数与模块消融：完整84-checkpoint research-test

协议ID：`m4-s1-research-test-v1-20261022`。

2026-10-22用户授权：先冻结完整84个checkpoint清单和专项协议，在空闲GPU并行执行，完成后完整汇总并与validation对照，暂不改变默认模型。本协议仅开放此次固定矩阵的test数据准备、20种原生CWT表示缓存、84项评价与完整汇总；train/validation阶段保持关闭。

## 来源与冻结顺序

训练session为项目根下 `runs/m4_s1_cwt_parameters_v1/acceptance_v1_20261020`，SHA256=`87b201abb30dea105ca8c9ada5c2f48a33e1f5ebd47b78af11276f67e93740cc`。完整validation summary为其 `summary/attempt_20261022T114304Z_0a0fef94d95b`，manifest SHA256=`ea6b3a8767f8e1ca8f680ae255fc2aec5369489d57a9c6e5d7f49b7b411628a7`。

冻结清单覆盖原28配置×seeds 20260812/20260813/20260814，全部使用完整validation Local RR严格最小、并列最早规则已选出的最佳checkpoint。prepare核验session源码、依赖、全部cell产物哈希、完整history选点和best/final配置、有限性，再保存每项checkpoint、config、completion、epoch、参数量和validation指标来源。清单同时冻结本协议、评价实现与依赖源码快照、20种CWT表示、development样本身份和完整比较规则。prepare不访问test样本、缓存或指标；清单成功后才允许prepare-data。采用独立输出 `runs/m4_s1_cwt_parameters_v1/research_test_v1_20261022`。

## 数据与推理合同

固定既有index、test sample seed=20260612、180秒/100Hz、2310窗口/8受试者，有序row ID SHA256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。核验与train/validation受试者隔离、原始源文件身份和全部样本元数据，保存公共target供逐cell精确对比。输入预处理、target、loss定义和指标不变。

每个模型strict加载对应训练结构及权重；20种表示分别按训练session中固定尺度、频率、时间池化、ssqueezepy变换建立test缓存，NoTF不接收CWT。使用训练环境的同型号GPU、batch=32、BF16、eval模式、完整尾批、固定样本顺序；仅调整评价配置副本的device和输出目录。逐batch核验输入、target、CWT形状与有限性，核验预测与公共target。任何非有限checkpoint/input/target/prediction/关键指标均失败，不丢弃样本。保存完整逐窗指标、预测、环境、运行命令、身份和回执。

## 完整矩阵分析

所有84项成功后才汇总读取指标。五项主指标、窗口加权主视图、受试者宏平均诊断、每seed/mean/样本SD、全部预设配对、各seed方向及0.01阈值方向数、位置2×2主效应/条件效应/交互、μ×voices交互、逐受试者、Local RR tail与分母均继承validation分析。delta=候选−参照；误差负值/PCC正值表示改善；绝对均值差<0.01为“均值无明显差异”，达到阈值按方向分类。保留原生评价器辅助指标及预测退化诊断。

跨split对照包含全部28配置五指标原值及27候选对B的配对差、分类和seed方向，并同时保存全部预设配对及交互的validation/test表。两split的绝对指标反映不同人群，不解释为相同样本上的泛化差；结构判断以各split内部配对比较为据。不按中途结果更改剩余矩阵、指标、阈值、checkpoint或模型默认值。

该test已用于研究开发，证据角色是reused research/development evidence；0.01是描述性阈值，不是统计等效检验。三个训练seed共享test人群，不构成三份独立人群证据。此次只报告结果，不自动选新模型。

## 执行和验收

入口 `scripts/eval_m4_s1_research_test.py` 提供prepare、prepare-data、worker、summarize、status；真实test命令要求 `--confirm-research-test`。两张本机空闲RTX4070 Ti SUPER各运行一个worker，表示缓存按独立阶段互斥，同一GPU串行评价，每cell独立追加式attempt，成功产物禁止覆盖，失败保留并须显式retry。Ubuntu186已有其他GPU任务，本轮不占用。

验收：清单84身份完整、20缓存对应原生表示、84成功receipt、各cell2310窗/8人且target资格一致、全矩阵CSV与PNG及validation对照报告、全部manifest哈希有效。科研实体产物只存项目根内，原训练结果与源文件保持原身份。默认模型不变。
