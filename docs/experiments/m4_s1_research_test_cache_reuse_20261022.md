# M4-S1 research-test缓存复用执行记录

2026-10-22，本记录补充 `m4-s1-research-test-v1-20261022` 的缓存物理来源。原84-checkpoint清单、协议和评价实现保持冻结。发现CWT-APOR v2已完成相同20种原生表示的research-test缓存；此次对尚未开始生成的表示，经完整核验后复制其缓存数组建立本轮独立阶段，以减少重复计算。

只读来源：项目根下 `runs/home_storage_migration_20261010/preserved_worktrees/worktrees_20261010/cwt-time-frequency-v1/runs/cwt_apor_v2/session_20261001T042244Z_e5e53ad16731`。原始缓存及其manifest不作任何修改。已经开始生成的表示继续完成。

复用必须满足：历史session/allowlist/data/cache成功回执及所有相关文件哈希有效；历史变换源码快照与本轮CWT实现、底层变换及源文件读取实现逐字节一致；依赖版本完全相同；20表示元数据与当前allowlist逐字段一致；index、原始源文件哈希、完整窗口元数据、row顺序及公共target完全一致；复制后的数组字节、float32形状、int64 row ID和全数组有限性通过。每项导入记录来源cache manifest、来源session、导入脚本、此执行记录和复制数组哈希，使用本轮既有cache阶段绑定及不可覆盖回执。

历史基准表示标签为A0，本轮对应B。核验时仅将表示的name字段从A0映射到B；所有尺度、频率、时刻、变换参数和数组字节必须相同，其余表示名称原样匹配。另需区分配置里的BCG键列名与样本表中解析出的数组键：前者按训练config比较，后者按两份完整样本表逐行比较。前两次导入探测分别在目录标签和键名层级检查处停止，未生成任何缓存阶段；脚本和失败日志均保留，v3正确区分上述标签及字段层级。

复用只影响缓存准备方式。所有84项模型推理仍在本轮逐项执行，样本、checkpoint、输入张量、指标与汇总规则保持不变。不读取历史预测或模型评价指标。执行脚本及逐项回执位于本轮research-test输出根；最终实际新建/复用数量以 `cache_import_completed.json` 与缓存阶段回执为准。
