# RespDiff-BCG P6 执行与复核源码

本目录保存已完成 P6 的三卡控制器、逐批恢复、双卡分片和结果复核源码。每份 Python 文件与原运行目录中的版本逐字节相同；下表记录原内容 SHA256。正式入口位于 `scripts/run_respdiff_bcg_research_test_v1.py`，完整结果见 [结果报告](../../docs/experiments/respdiff_bcg_research_test_results_20261024.md)。

## 原执行布局

表中的相对路径原位于工作树的 `runs/respdiff_bcg_research_test_v1/` 下。源码通过 `__file__` 推导原执行目录，恢复/迁移程序还引用同级来源文件。审计或复现代码时，应在独立 checkout 中按该原布局放置副本，再核对协议、模型、数据与来源回执；本目录是 Git 中的源码保存位置。

P6 已完成关闭。历史输出 identity、缓存、完成/失败回执与原运行目录保持冻结。后续执行须使用独立版本、匹配协议和授权。

## 文件身份

| 原运行根下的相对路径 | SHA256 |
|---|---|
| `launch_t630_v1/launch.py` | `4106abe1733a3fe2c5cb7af414c12251e9f9b8cd810e6ecb33605095f20a11f3` |
| `recovery_20261021_v1/recover.py` | `10d73c814d192109bacd1b87309851b222cb623b98a5b34d6c92ce93aa28f6e7` |
| `recovery_20261021_v1/launch.py` | `ee167f234604faa7b6e43ba97b2a84f013e82898e0c6167eb392a083b10b3f52` |
| `recovery_20261021_v1/test_recover.py` | `681a3030369370b0dcc81935c90bc8f29fcd4f60a3cef7b36384ecd57caccbfc` |
| `migration_20261023_v1/migrate.py` | `20504044018f9a739bda332c9f609221617b7b8d7ea42646d8fffde1c93dd316` |
| `migration_20261023_v1/launch.py` | `9fb9851b587c00f5e47d23391273b4d425c479951df04d8e9f138c35a1c1950a` |
| `migration_20261023_v1/test_migrate.py` | `9c7c044b0baa3c3f0b4400192b8a3d02294f41d5b7030b40d5b205951858e800` |
| `review_20261024_v1/review.py` | `153229fc79507310e232623839d491bb0a25c4101aeb2d9d78ed9c5e66623ad3` |

原 P6 manifest SHA256：`0113fd0910e9466acef90ab2404ab64df41b47f8e30f626cd7e96c729f210aa4`。八份源文件的功能与证据边界由部署、恢复、迁移及最终结果文档说明。完成回执、波形、checkpoint、缓存、日志、逐窗口指标和生成分析表保存在项目产物目录，以最终结果报告的入口定位。
