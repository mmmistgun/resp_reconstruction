# B2-0 patch 融合改走原生解码（native_inject）实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法跟踪进度。

**目标：** 修掉 patch 双分支「(B,C,T') 通用融合头」这个被文档量化坐实的最大质量泄漏——wrapper 解码路径把长尾误拣率几乎翻倍（原生 E1a 11.3% → wrapper deep 22.4%，中位数不变）。改法：在 token 栅格（140）把 STFT 以**零初始化加性注入**到主干特征，再走主干**原生 `patch_head + overlap-add` 解码**，使 `time_only` 臂在架构上**等价于原生 E1a**。单独测一遍（不带 gating），确认 (a) time_only 回到原生质量、(b) dual−time_only 配对增益仍 ≤ 基准 −2.34pp，再进 B2-1 gating。

**非目标（本计划明确不做）：**
- 不做 gating / FiLM / attention（那是 B2-1，必须在本计划通过后、在干净 substrate 上单独上）。
- 不动 multiscale（已弃，`concat_generic` 旧路径保持可用、保持默认，零回归）。
- 不改 `configs/tho_research_v2.yaml` 默认 `fusion_mode`（默认仍 `concat_generic`，避免改动既有管线行为；B2-0 用 `--set` 显式切 `native_inject`）。
- 不动 dataset / engine / loss / checkpoint 校验 / 指标。

**架构：** `TimeStftDual1D` 新增 `fusion_mode ∈ {concat_generic, native_inject}`。
- `concat_generic`（现状默认）：`align_to_time → concat → FusionHead`，逐元素不变。
- `native_inject`（新，仅 patch）：主干出 `(B,16,140)` token 特征；dual 时把 STFT `(B,16,T_s)` 对齐到 token 栅格并经 `stft_proj`（1×1，**末层零初始化**）加到 token 特征上，再调主干 `decode_from_features(fused, length)` 走原生 `patch_head + overlap-add`。`time_only` 时不建 STFT 分支、不建 `stft_proj`、不建 `FusionHead`，前向**逐算子等价**于原生主干。
  - 加性而非 concat：concat 进原生头会改变 `patch_head` 的输入通道数（16→32），破坏「time_only ≡ 原生」这一可验证不变量；加性注入是保「原生解码」前提下唯一干净的最小融合，且 gating（B2-1）正是它的自然推广（`feats + g⊙proj(stft)`）。
  - 零初始化 `stft_proj`：dual 在 init 时输出 == time_only == 原生，warm-start 不偏离原生盆地（对极性稳定性也友好）；1×1 conv 零初始化只让输出起点为 0，权重梯度非零，STFT 分支照常学习（ReZero 式残差暖启）。

**技术栈：** PyTorch、OmegaConf、pytest、现有 `resp_train` 训练栈（`build_model` 注册表 + `ThoExperiment`）。

**判定口径承接 E1-D：** 8Hz N0、patch（base16 / patch_len256 / stride128 / hann / conv2d STFT）、默认极性 warmup 配方（`signed_corr` 0.6→0.2 + `signed_cosine` 0.1→0）、复用 E1-D 那 6 个代表性 seed、`peak_band_misclass_rate.py` + 配对 delta。

---

## 文件结构

**修改：**
- `resp_train/models/timeseries.py` — `PatchMixer1D` 抽出 `decode_from_features(tokens, length)`，`forward` 非特征路径改为调用它（纯提取、逐元素等价）。
- `resp_train/models/stft_branch.py` — `TimeStftDual1D` 加 `fusion_mode`、`native_inject` 前向、`stft_proj`（零初始化）；`native_inject` 仅在主干暴露 `decode_from_features` 时可用，否则报错。
- `resp_train/models/registry.py` — `_build_time_stft_dual1d` 读 `cfg.model.fusion_mode`（默认 `concat_generic`）透传；`native_inject` 校验 `time_backbone=patch_mixer1d`。
- `tests/test_model_registry.py` — 补 `decode_from_features` 等价/契约、registry `fusion_mode` 透传与校验测试。
- `tests/test_stft_branch.py` — 补 `native_inject` time_only 恒等、dual init 等价、dual 双臂梯度、multiscale 报错测试。
- `tests/test_engine_smoke.py` — 补 `native_inject` dual 前向/反向/checkpoint 往返。

**创建：**
- `scripts/run_b2_native_decode.py` — B2-0 批次编排（time_only / dual × 6 seed，`fusion_mode=native_inject`，复用 train_tho_small），含 manifest。
- `tests/test_run_b2_overrides.py` — 编排 override 生成单测（不训练）。

**关键契约（命名固定）：**
- `PatchMixer1D.decode_from_features(tokens: Tensor(B,base_channels,T'), length: int) -> Tensor(B,out_channels,L)`：等价于现状 `forward` 在 `return_features` 之后那段（`patch_head → view → overlap_add → output_smoother`）。
- `PatchMixer1D.forward(x)`（非特征路径）逐元素等价于 `decode_from_features(*self.forward(x, return_features=True)[::-1] 形参顺序对应)`，即 `tokens, length = forward(x, return_features=True); decode_from_features(tokens, length)` == `forward(x)`。
- `TimeStftDual1D(..., fusion_mode="native_inject")` 且 `branch_mode="time_only"`：`forward(x)` == `self.time_backbone(x)`（同实例同权重，逐元素相等）。
- `fusion_mode="native_inject"` 且 `branch_mode="dual"` 在 init（`stft_proj` 零初始化）：`forward(x)` == 同权重 time_only 前向。
- cfg 新字段：`model.fusion_mode ∈ {concat_generic, native_inject}`，缺省 `concat_generic`。

---

## 任务 1：PatchMixer1D 抽出 decode_from_features（逐元素等价）

**文件：**
- 修改：`resp_train/models/timeseries.py`（`PatchMixer1D`，约 172-205 行）
- 测试：`tests/test_model_registry.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_model_registry.py` 末尾追加：

```python
def test_patch_mixer_decode_from_features_matches_native_forward():
    torch.manual_seed(0)
    model = PatchMixer1D(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64)
    model.eval()
    x = torch.randn(2, 1, 4096)

    with torch.no_grad():
        native = model(x)
        tokens, length = model(x, return_features=True)
        recomposed = model.decode_from_features(tokens, length)

    assert torch.equal(native, recomposed)


def test_patch_mixer_decode_from_features_shape():
    model = PatchMixer1D(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64)
    x = torch.randn(2, 1, 4096)
    tokens, length = model(x, return_features=True)

    out = model.decode_from_features(tokens, length)

    assert out.shape == (2, 1, 4096)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_model_registry.py -k "decode_from_features" -v`
预期：FAIL，`PatchMixer1D` 无 `decode_from_features`（AttributeError）。

- [ ] **步骤 3：编写最少实现代码**

把 `resp_train/models/timeseries.py` 中 `PatchMixer1D.forward` 的「非特征路径」抽成方法，`forward` 改为调用它（逻辑不变，只是提取）：

```python
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, int]:
        batch, _, length = x.shape
        padded_length = self._padded_length(length)
        x_padded = _match_length(x, padded_length)
        patches = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        patch_count = patches.size(2)
        tokens = patches.permute(0, 2, 1, 3).reshape(batch, patch_count, -1)
        tokens = self.patch_embed(tokens).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        if return_features:
            return tokens, length
        return self.decode_from_features(tokens, length)

    def decode_from_features(self, tokens: torch.Tensor, length: int) -> torch.Tensor:
        """把 (B, base_channels, T') token 特征经原生 patch_head + overlap-add 解码回 (B, out_channels, length)。

        与 forward 非特征路径逐元素等价；供双分支在 token 栅格注入后复用原生解码契约。
        """
        batch = tokens.size(0)
        patch_count = tokens.size(-1)
        padded_length = self._padded_length(int(length))
        patch_values = self.patch_head(tokens.transpose(1, 2))
        patch_values = patch_values.view(batch, patch_count, self.out_channels, self.patch_len)
        return self.output_smoother(self._overlap_add(patch_values, int(length), padded_length))
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_model_registry.py -k "decode_from_features" -v`
预期：两条均 PASS。

- [ ] **步骤 5：跑 PatchMixer 全部既有测试确认零回归**

运行：`pytest tests/test_model_registry.py -k "patch_mixer" -v`
预期：全部 PASS（含 `return_features` 向后兼容、overlap_add / smoothing）。

- [ ] **步骤 6：Commit**

```bash
git add resp_train/models/timeseries.py tests/test_model_registry.py
git commit -m "refactor: PatchMixer1D 抽出 decode_from_features 原生解码契约"
```

---

## 任务 2：TimeStftDual1D 加 native_inject 模式（零初始化加性注入）

**文件：**
- 修改：`resp_train/models/stft_branch.py`（`TimeStftDual1D`）
- 测试：`tests/test_stft_branch.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_stft_branch.py` 追加：

```python
def _native_dual(branch_mode: str, backbone: str = "patch_mixer1d") -> TimeStftDual1D:
    kwargs = (
        dict(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64, overlap_window="hann")
        if backbone == "patch_mixer1d"
        else dict(in_channels=1, out_channels=1, base_channels=8, downsample_factors=[1, 4, 16])
    )
    feat_ch = 8 if backbone == "patch_mixer1d" else 8 * 3
    return TimeStftDual1D(
        time_backbone_name=backbone,
        time_backbone_kwargs=kwargs,
        time_feat_channels=feat_ch,
        branch_mode=branch_mode,
        out_length=18000,
        fuse_len=600,
        stft_kwargs=dict(sample_rate=100.0, stft_win=3000, stft_hop=500, low_hz=0.05, high_hz=8.0, out_channels=16, norm="n0", encoder_type="conv2d"),
        fusion_mode="native_inject",
    )


def test_native_inject_time_only_equals_native_backbone_forward():
    model = _native_dual("time_only")
    model.eval()
    x = torch.randn(2, 1, 18000)

    with torch.no_grad():
        wrapped = model(x)
        native = model.time_backbone(x)  # 原生非特征前向

    assert torch.equal(wrapped, native)


def test_native_inject_dual_equals_time_only_at_init_due_to_zero_proj():
    model = _native_dual("dual")
    model.eval()
    x = torch.randn(2, 1, 18000)

    with torch.no_grad():
        dual_out = model(x)
        # 零初始化 stft_proj：注入项为 0，dual 应等于纯原生解码
        tokens, length = model.time_backbone(x, return_features=True)
        native = model.time_backbone.decode_from_features(tokens, length)

    assert torch.allclose(dual_out, native, atol=1e-6)


def test_native_inject_dual_has_both_branch_gradients():
    model = _native_dual("dual")
    x = torch.randn(2, 1, 18000)

    model(x).square().mean().backward()

    time_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.time_backbone.parameters())
    stft_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.stft_encoder.parameters())
    proj_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.stft_proj.parameters())
    assert time_grad and stft_grad and proj_grad


def test_native_inject_rejects_backbone_without_decode_from_features():
    with pytest.raises((ValueError, TypeError, AttributeError)):
        _native_dual("dual", backbone="multiscale_decomp_mixer1d")
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_stft_branch.py -k "native_inject" -v`
预期：FAIL，`TimeStftDual1D` 不接受 `fusion_mode`（TypeError）。

- [ ] **步骤 3：编写最少实现代码**

修改 `resp_train/models/stft_branch.py` 的 `TimeStftDual1D`：
1. `__init__` 末尾加形参 `fusion_mode: str = "concat_generic"`。
2. 校验 `fusion_mode ∈ {concat_generic, native_inject}`。
3. `native_inject` 分支：
   - 要求 `use_time` 且主干有 `decode_from_features`；否则 `raise ValueError("native_inject 仅支持暴露 decode_from_features 的主干（当前仅 patch_mixer1d）")`。
   - 不建 `FusionHead`（设 `self.fusion_head = None`）。
   - dual 时建 `self.stft_proj = nn.Conv1d(stft_out_channels, time_feat_channels, kernel_size=1)`，并对其 `weight`/`bias` 做**零初始化**；time_only 时 `self.stft_proj = None`。
4. `forward` 按 `fusion_mode` 分流。`concat_generic` 路径保持现状不动。

参考实现骨架（仅示意分流，`concat_generic` 原逻辑保留）：

```python
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in {"concat_generic", "native_inject"}:
            raise ValueError("fusion_mode 必须是 concat_generic 或 native_inject")

        if self.fusion_mode == "native_inject":
            if not (use_time and hasattr(self.time_backbone, "decode_from_features")):
                raise ValueError("native_inject 仅支持暴露 decode_from_features 的主干（当前仅 patch_mixer1d）")
            self.fusion_head = None
            if use_stft:
                self.stft_proj = nn.Conv1d(int(self.stft_encoder.out_channels), int(time_feat_channels), kernel_size=1)
                nn.init.zeros_(self.stft_proj.weight)
                nn.init.zeros_(self.stft_proj.bias)
            else:
                self.stft_proj = None
        else:
            # concat_generic：保持现状
            self.fusion_head = FusionHead(fused_channels, out_length=out_length, hidden=fusion_hidden, decoder_style=fusion_decoder)
            self.stft_proj = None
```

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "native_inject":
            time_feats, length = self.time_backbone(x, return_features=True)
            if self.stft_encoder is not None:
                stft_feats = align_to_time(self.stft_encoder(x), time_feats.size(-1))
                time_feats = time_feats + self.stft_proj(stft_feats)
            return self.time_backbone.decode_from_features(time_feats, length)

        # concat_generic：现状逻辑不变
        features: list[torch.Tensor] = []
        if self.time_backbone is not None:
            time_feats, _ = self.time_backbone(x, return_features=True)
            features.append(align_to_time(time_feats, self.fuse_len))
        if self.stft_encoder is not None:
            features.append(align_to_time(self.stft_encoder(x), self.fuse_len))
        return self.fusion_head(torch.cat(features, dim=1))
```

注意：`native_inject` 下 `stft_only` 不在支持范围（无时间主干即无 `decode_from_features`）；若 `branch_mode=stft_only` 且 `fusion_mode=native_inject`，由上面 `use_time` 校验自然报错，符合预期。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_stft_branch.py -k "native_inject" -v`
预期：4 条均 PASS。

- [ ] **步骤 5：跑 stft_branch 全部测试确认 concat_generic 零回归**

运行：`pytest tests/test_stft_branch.py -v`
预期：全部 PASS（旧 concat_generic 三模式、FusionHead、align_to_time 不受影响）。

- [ ] **步骤 6：Commit**

```bash
git add resp_train/models/stft_branch.py tests/test_stft_branch.py
git commit -m "feat: TimeStftDual1D 增加 native_inject 原生解码融合模式"
```

---

## 任务 3：registry 透传 fusion_mode

**文件：**
- 修改：`resp_train/models/registry.py`（`_build_time_stft_dual1d`）
- 测试：`tests/test_model_registry.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_model_registry.py` 末尾追加：

```python
def test_build_time_stft_dual1d_native_inject_patch_preserves_shape():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "conv2d",
                "fusion_mode": "native_inject",
            },
        }
    )
    model = build_model(cfg)
    y = model(torch.randn(2, 1, 18000))
    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()
    assert model.fusion_head is None  # native_inject 不建通用融合头


def test_build_time_stft_dual1d_native_inject_multiscale_raises():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 4096},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "multiscale_decomp_mixer1d",
                "downsample_factors": [1, 4, 16], "mixer_layers": 1,
                "stft_win": 512, "stft_hop": 128, "stft_low_hz": 0.05, "stft_high_hz": 3.0,
                "stft_out_channels": 16, "stft_norm": "n0",
                "fusion_mode": "native_inject",
            },
        }
    )
    with pytest.raises(ValueError, match="native_inject"):
        build_model(cfg)


def test_build_time_stft_dual1d_defaults_to_concat_generic():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1,
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0",
            },
        }
    )
    model = build_model(cfg)
    assert model.fusion_mode == "concat_generic"
    assert model.fusion_head is not None
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_model_registry.py -k "native_inject or concat_generic" -v`
预期：FAIL（`fusion_mode` 未透传，native_inject 不报错或建了融合头）。

- [ ] **步骤 3：编写最少实现代码**

在 `resp_train/models/registry.py` 的 `_build_time_stft_dual1d` 返回里加一行透传：

```python
    return TimeStftDual1D(
        ...,
        fusion_decoder=str(cfg.model.get("fusion_decoder", "deep")),
        fusion_mode=str(cfg.model.get("fusion_mode", "concat_generic")),
    )
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_model_registry.py -k "native_inject or concat_generic" -v`
预期：3 条均 PASS。

- [ ] **步骤 5：跑全部模型注册测试确认零回归**

运行：`pytest tests/test_model_registry.py -v`
预期：全部 PASS。

- [ ] **步骤 6：Commit**

```bash
git add resp_train/models/registry.py tests/test_model_registry.py
git commit -m "feat: registry 透传 model.fusion_mode（默认 concat_generic）"
```

---

## 任务 4：native_inject 集成训练冒烟

**文件：**
- 测试：`tests/test_engine_smoke.py`（追加）

- [ ] **步骤 1：先看现有冒烟测试搭建方式**

运行：`pytest tests/test_engine_smoke.py -v` 并阅读该文件，复用其 cfg 构造/调用方式。

- [ ] **步骤 2：编写测试**

追加（自包含，仅依赖 `build_model`）：

```python
def test_time_stft_dual1d_native_inject_forward_backward_and_state_roundtrip():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d", "in_channels": 1, "out_channels": 1, "base_channels": 8,
                "branch_mode": "dual", "time_backbone": "patch_mixer1d",
                "patch_len": 256, "patch_stride": 128, "mixer_layers": 1, "overlap_window": "hann",
                "stft_win": 3000, "stft_hop": 500, "stft_low_hz": 0.05, "stft_high_hz": 8.0,
                "stft_out_channels": 16, "stft_norm": "n0", "stft_encoder_type": "conv2d",
                "fusion_mode": "native_inject",
            },
        }
    )
    model = build_model(cfg)
    x = torch.randn(2, 1, 18000)
    target = torch.randn(2, 1, 18000)

    pred = model(x)
    (pred - target).square().mean().backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert pred.shape == (2, 1, 18000)
    assert grad_norm > 0.0

    state = model.state_dict()
    fresh = build_model(cfg)
    fresh.load_state_dict(state)
    model.eval(); fresh.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), fresh(x), atol=1e-5)
```

- [ ] **步骤 3：运行测试**

运行：`pytest tests/test_engine_smoke.py::test_time_stft_dual1d_native_inject_forward_backward_and_state_roundtrip -v`
预期：PASS（任务 1-3 正确则直接绿；FAIL 则回对应任务修）。

- [ ] **步骤 4：Commit**

```bash
git add tests/test_engine_smoke.py
git commit -m "test: native_inject 前向/反向/checkpoint 往返集成冒烟"
```

---

## 任务 5：B2-0 批次编排脚本

**文件：**
- 创建：`scripts/run_b2_native_decode.py`
- 测试：`tests/test_run_b2_overrides.py`

沿用 `scripts/run_e1_stft_info_gain.py` 的「笛卡尔积 + subprocess + manifest + --dry-run/--skip/--max-parallel」模式，调用 `scripts/train_tho_small.py --config configs/tho_research_v2.yaml`。固定 8Hz N0、patch、conv2d、默认 warmup 配方，扫 `branch_mode ∈ {time_only, dual}` × 6 seed，全部 `model.fusion_mode=native_inject`。

> **SEEDS 占位**：步骤 3 实现前，从 E1-D 复用的 6 个 seed 在执行期确认（见任务 6 步骤 1）。计划内以 `SEEDS = [...]  # 执行期填 E1-D 同款 6 seed` 占位，单测只校验「2 臂 × len(SEEDS)」结构与关键 override，不绑定具体 seed 值。

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_run_b2_overrides.py`：

```python
import scripts.run_b2_native_decode as b2


def test_specs_cover_two_arms_per_seed():
    specs = b2.build_run_specs()
    arms = {s["branch_mode"] for s in specs}
    assert arms == {"time_only", "dual"}
    assert len(specs) == 2 * len(b2.SEEDS)


def test_all_specs_are_native_inject_patch_8hz_n0():
    for s in b2.build_run_specs():
        joined = " ".join(s["overrides"])
        assert "model.name=time_stft_dual1d" in joined
        assert "model.fusion_mode=native_inject" in joined
        assert "model.time_backbone=patch_mixer1d" in joined
        assert "model.stft_high_hz=8.0" in joined
        assert "model.stft_norm=n0" in joined
        assert "model.stft_encoder_type=conv2d" in joined


def test_manifest_row_preserves_factors():
    row = b2.manifest_row(b2.build_run_specs()[0])
    assert {"tag", "branch_mode", "seed", "overrides"} <= set(row)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_run_b2_overrides.py -v`
预期：FAIL，模块不存在。

- [ ] **步骤 3：编写最少实现代码**

创建 `scripts/run_b2_native_decode.py`（结构照搬 `run_e1_stft_info_gain.py`：`COMMON_OVERRIDES` 不含 loss warmup 项——warmup 已是 `configs/tho_research_v2.yaml` 默认；`STFT_BASE` 固定 patch+8Hz+N0+conv2d+`fusion_mode=native_inject`；`build_run_specs` 产 2 臂 × SEEDS；`main` 支持 `--dry-run/--skip/--max-parallel/--device/--manifest`）。`SEEDS` 用 E1-D 同款 6 seed。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_run_b2_overrides.py -v`
预期：全部 PASS。

- [ ] **步骤 5：dry-run 校验 run 数**

运行：`python scripts/run_b2_native_decode.py --dry-run | wc -l`
预期：`2 臂 × 6 seed = 12`，并生成 manifest。

- [ ] **步骤 6：Commit**

```bash
git add scripts/run_b2_native_decode.py tests/test_run_b2_overrides.py
git commit -m "feat: B2-0 native_inject 批次编排脚本"
```

---

## 任务 6：全量回归 + 实验执行 + 判定

代码与测试全绿后执行实验（CUDA 命令按项目规则提权）。

- [ ] **步骤 1：确认 E1-D 的 6 个 seed**

从 E1-D 产物（`runs/e1_d_warmup_status.csv` / `runs/e1_clean_patch600_deep_dual_8hz` 与 `_time_only` 目录的 config.yaml `training.seed`）确认 6 个代表性 seed，填入 `scripts/run_b2_native_decode.py` 的 `SEEDS`。

- [ ] **步骤 2：全量回归**

运行：`pytest tests/test_model_registry.py tests/test_stft_branch.py tests/test_engine_smoke.py tests/test_run_b2_overrides.py -v`
预期：全部 PASS，concat_generic 零回归。

- [ ] **步骤 3：跑 B2-0（12 run）**

运行：`python scripts/run_b2_native_decode.py --max-parallel <N> --device cuda:<i>`（提权）。

- [ ] **步骤 4：可选 sanity——原生 patch 对照（1-2 seed）**

用 `model.name=patch_mixer1d`（不经包装器）在同 seed 跑 1-2 个，交叉确认 time_only native_inject 的误拣率与原生同档（任务 2 的恒等测试已在架构上保证前向一致，本步只兜底训练 RNG 差异）。

- [ ] **步骤 5：误拣率 + 配对 delta 汇总**

运行：`python scripts/peak_band_misclass_rate.py --runs-root runs/<b2_time_only> --runs-root runs/<b2_dual> --threshold 1.0 2.0 --output runs/b2_native_detail.csv --grouped-output runs/b2_native_grouped.csv`，再按 E1-D 同款脚本算 `(dual − time_only)` 同 seed 配对 delta。

- [ ] **步骤 6：判定门（强制）**

- (1) **time_only native_inject 误拣率@1 ≈ 原生（~11%）**，明显低于旧 wrapper（~22%）→ 头修正确实救回原生质量。
- (2) **dual − time_only 配对 Δfrac_gt_1 ≤ −2.34pp**（理想更负）→ STFT 净增益在干净 substrate 上保住。
- (3) **gate-fail 数不劣于 E1-D**（warmup 在原生头下仍有效）。
- **全过** → B2-0 成功，进 B2-1 单 gating 探针（另开计划）。
- **(2) 不过**（增益缩水/消失）→ 这是结论本身：旧通用解码栅格在「意外帮 STFT」；停下报告，不在破/疑 substrate 上盲目投 gating。

- [ ] **步骤 7：文档**

在 `docs/experiments/e1_stft_info_gain_20260622.md` 增「B2-0 原生解码融合」小节，记三判定门结果 + 数据产物路径；结论稳定后视情况同步主候选配方。

```bash
git add docs/experiments/e1_stft_info_gain_20260622.md
git commit -m "docs(b2): 记录 B2-0 native_inject 头修正与增益复测"
```

---

## 自检结果

**目标覆盖度：**
- 抽出原生解码契约（逐元素等价）→ 任务 1 ✓
- native_inject 加性注入 + 零初始化 + time_only 恒等 + dual init 等价 + 双臂梯度 + multiscale 报错 → 任务 2 ✓
- registry 透传 + 默认 concat_generic 零回归 + native_inject 校验 → 任务 3 ✓
- 集成（engine + checkpoint 往返）→ 任务 4 ✓
- 实验编排（2 臂 × 6 seed，8Hz N0，复用默认 warmup）→ 任务 5 ✓
- 三判定门 + 失败即报告（不盲目进 gating）→ 任务 6 ✓

**关键不变量一致性：** `decode_from_features` 在任务 1 定义、任务 2 native_inject 前向消费一致；`fusion_mode` 字段在任务 2（构造）/3（registry）/5（override）命名一致；`time_only ≡ 原生`、`dual@init ≡ time_only` 两条不变量由任务 2 测试强制；concat_generic 旧路径在任务 2/3 全程零改动并被既有测试守住。

**风险与边界：**
- 加性注入≠concat（已在架构段论证：保「原生解码 + time_only≡原生」前提下的唯一最小干净融合；gating 是其自然推广）。
- native_inject 仅 patch；multiscale 显式报错（已弃，不补）。
- 默认配置不切 native_inject，避免改既有管线行为；B2-0 用 `--set` 显式启用。
```