# E8 research-test：36-checkpoint allowlist 回执

日期：2026-09-26。状态：**36 个 validation-selected checkpoint 已全部固定并通过回载核验；test 数组尚未读取，评价尚未授权。**

## 冻结身份

```text
runs/e8_film_decoder_redesign_v1/research_test/allowlist/
allowlist_8b67c189b10c_20260926T083406Z_b300e95e53bc
```

- Allowlist SHA-256：`76ef0c8ac05cdddcf89caab46abb0a03b78d0d3e72d09b6ed5773521dc2576db`。
- Manifest SHA-256：`2ff666e796111b1952ad18f36bf91ea87b649d5091f42395c0e38f55082147ae`。
- 条目：12 arms × 3 seeds = 36；每项固定 selected epoch、checkpoint、config、history、formal manifest/receipt、development subjects 与逐文件 SHA。
- Validation P5 manifest：`3d9ad92fa42df7d2f0e9e66a20a134b4328e62d3e3940c04c15bdafa7805871f`。
- Validation decision：`1743548f3894cae2238827016cce0fef1fe9fc50ef46b48b3a1ca65794388b51`。
- Formal lock：`40b30fc6ecdcc9b40750c393be8fd2faf4e223da0566432df739aacb513713c5`。
- Runtime amendment：`517bdf281a0db0902e0c10481b0ea210a7c514d627eb6397e9e982d95ba270e3`。

## 访问边界

Allowlist 生成阶段的冻结回执为：

```text
validation_artifacts_read = true
checkpoint_bytes_read = true
test_cache_manifest_read = true
test_array_read = false
dataset_index_read = false
model_inference_used = false
```

因此当前只完成候选冻结，没有产生任何 research-test 指标。后续每个评价 cell 必须引用上述 allowlist，并显式传入 `--confirm-research-test`；36 项全部完成前不得汇总或根据 partial test 改变队列。
