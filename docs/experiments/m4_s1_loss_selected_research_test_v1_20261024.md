# M4-S1 validation总loss选点的固定84-checkpoint research-test

协议ID：`m4-s1-loss-selected-research-test-v1-20261024`。

2026-10-24用户授权完整28配置×3 seed，按validation总loss最低选择checkpoint并计算test指标；并列取最早epoch。新阶段使用独立输出 `runs/m4_s1_cwt_parameters_v1/loss_selected_research_test_v1_20261024`。原训练、Local RR选点research-test及其结论保持冻结；默认模型不变。

## 选择与证据边界

来源为 `acceptance_v1_20261020` 的完整84项训练，每项在**既有已完成epoch**中按 `record.json` 的 `validation.loss` 严格最小选点，并列按最早epoch。总loss沿用训练时定义，禁止改用train loss、单个loss分量或test指标。核验每个epoch记录/回执、完成清单、候选checkpoint哈希及checkpoint内部history/config/epoch/有限性，先固定84项allowlist才访问test数据与指标。

原训练按validation Local RR执行早停；本轮是固定轨迹内的checkpoint选择规则比较，不代表按总loss早停重新训练的结果。旧结果继续保留；新选点的模型需要新增test评价，checkpoint身份与旧选点完全相同者经旧回执核验只读复用。本轮不重新训练。

## 固定运行合同

训练session SHA256=`87b201abb30dea105ca8c9ada5c2f48a33e1f5ebd47b78af11276f67e93740cc`；前次test allowlist SHA256=`e02211e4c330644a30324e0ada597059bb14f9289d1cce595c732a4b48c90338`。前次已完成结果为 `research_test_v1_20261022`。

从前次test的已校验源码快照恢复独立runtime，逐文件核验冻结来源和当前依赖版本；不将主仓库后续数据适配改动引入此次评价。新选点入口、专项协议、定向测试和运行命令另存快照并绑定allowlist。

完全沿用前次2310窗/8人、180秒/100Hz、test sample seed=20260612及有序row hash `184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。完整来源核验后复制前次公共target、样本表及20种原生CWT缓存至新阶段；NoTF不接收CWT。输入预处理、target、五项主指标及有效性口径均不变。每模型strict加载，batch32、BF16、同型号RTX4070 Ti SUPER、完整尾批；输入/target/条件/预测/关键指标非有限时失败。

## 完整矩阵与比较

全部84项完成（新增评价或同checkpoint复用）后才汇总。保存原生逐窗指标、预测或其冻结来源、配置、环境、回执。输出28配置五指标的每seed、mean/样本SD、窗口加权及受试者宏平均、逐受试者、尾部误差、分母、原预设模块配对和交互。

新增同cell的选点规则比较：delta＝validation总loss选点test指标−原Local RR选点test指标，误差负值/PCC正值表示改善；按三个seed配对报告mean/样本SD、方向及0.01描述性阈值。两个选点的validation总loss和Local RR均从已保存history读取，不冒充新checkpoint的完整validation五指标。test属于既有research/development evidence，不根据结果继续选择新checkpoint或修改剩余矩阵。

## 执行与收尾

入口 `scripts/eval_m4_s1_loss_selected_test.py`，提供prepare、prepare-inputs、worker、summarize和status。真实test数据操作要求 `--confirm-research-test`，失败attempt完整保留，成功阶段只读复用。按冻结84项顺序固定分片；只使用启动时空闲的GPU，每卡串行评价。完整矩阵成功后自动汇总。用户已表示长任务采用后台执行，完成后由用户通知核验。

验收为84项选点身份完整、全部test结果成功、每项2310窗/8人、五指标有限性/资格/样本顺序一致、完整选点对照与来源manifest。源码快照、缓存、checkpoint、日志和生成产物存项目内，Git仅保存源码/协议/测试/结果说明。
