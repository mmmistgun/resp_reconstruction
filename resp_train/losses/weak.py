from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from resp_train.losses.stft import TargetStftLogMagLoss, TargetStftLoss


class WeakSyncLoss(torch.nn.Module):
    """用于波形弱同步训练的组合损失。"""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.fs = float(cfg.window.target_fs)
        self.envelope_window = max(1, int(float(cfg.loss.envelope_window_sec) * self.fs))
        self.env_weight = float(cfg.loss.envelope_weight)
        self.spec_weight = float(cfg.loss.spectrum_weight)
        self.smooth_weight = float(cfg.loss.smooth_weight)
        self.high_freq_weight = float(cfg.loss.get("high_freq_weight", 0.0))
        self.relative_env_weight = float(cfg.loss.get("relative_envelope_weight", 0.0))
        self.phase_alignment_weight = float(cfg.loss.get("phase_alignment_weight", 0.0))
        self.band_waveform_weight = float(cfg.loss.get("band_waveform_weight", 0.0))
        self.curvature_weight = float(cfg.loss.get("curvature_weight", 0.0))
        self.rhythm_weight = float(cfg.loss.get("rhythm_weight", 0.0))
        self.signed_cosine_weight = float(cfg.loss.get("signed_cosine_weight", 0.0))
        self.signed_corr_weight = float(cfg.loss.get("signed_corr_weight", 0.0))
        self.signed_rms_envelope_weight = float(cfg.loss.get("signed_rms_envelope_weight", 0.0))
        self.signed_mean_weight = float(cfg.loss.get("signed_mean_weight", 0.0))
        self.si_sdr_weight = float(cfg.loss.get("si_sdr_weight", 0.0))
        self.stft_dist_weight = float(cfg.loss.get("stft_dist_weight", 0.0))
        self.stft_band_energy_weight = float(cfg.loss.get("stft_band_energy_weight", 0.0))
        self.stft_peak_anchor_weight = float(cfg.loss.get("stft_peak_anchor_weight", 0.0))
        self.fb_aux_weight = float(cfg.loss.get("fb_aux_weight", 0.0))
        self.fb_consistency_weight = float(cfg.loss.get("fb_consistency_weight", 0.0))
        self.fb_consistency_start_epoch = int(cfg.loss.get("fb_consistency_start_epoch", 1))
        self.fb_consistency_detach_aux = bool(cfg.loss.get("fb_consistency_detach_aux", True))
        self.stft_sample_weight_mode = str(cfg.loss.get("stft_sample_weight_mode", "none")).lower()
        self.stft_sample_weight_min = float(cfg.loss.get("stft_sample_weight_min", 0.0))
        self.log_component_grad_norms = bool(cfg.loss.get("log_component_grad_norms", False))
        self.signed_cosine_schedule = cfg.loss.get("signed_cosine_schedule", None)
        self.signed_corr_schedule = cfg.loss.get("signed_corr_schedule", None)
        self.epoch = 1
        self.phase_alignment_zero_weight = float(cfg.loss.get("phase_alignment_zero_weight", 0.5))
        self.phase_alignment_lag_weight = float(cfg.loss.get("phase_alignment_lag_weight", 0.5))
        self.phase_alignment_lag_penalty_weight = float(cfg.loss.get("phase_alignment_lag_penalty_weight", 0.1))
        self.phase_alignment_max_sec = float(cfg.loss.get("phase_alignment_max_sec", 0.5))
        self.phase_alignment_step_sec = float(cfg.loss.get("phase_alignment_step_sec", 0.1))
        self.phase_alignment_temperature = float(cfg.loss.get("phase_alignment_temperature", 0.05))
        self.signed_env_temperature = float(cfg.loss.get("signed_env_temperature", 0.25))
        self.signed_mean_window = max(1, int(float(cfg.loss.get("signed_mean_window_sec", 2.0)) * self.fs))
        self.rhythm_min_period_sec = float(cfg.loss.get("rhythm_min_period_sec", 2.0))
        self.rhythm_max_period_sec = float(cfg.loss.get("rhythm_max_period_sec", 20.0))
        self.rhythm_step_sec = float(cfg.loss.get("rhythm_step_sec", 0.25))
        self.rhythm_temperature = float(cfg.loss.get("rhythm_temperature", 0.08))
        for name, value in (
            ("phase_alignment_weight", self.phase_alignment_weight),
            ("phase_alignment_zero_weight", self.phase_alignment_zero_weight),
            ("phase_alignment_lag_weight", self.phase_alignment_lag_weight),
            ("phase_alignment_lag_penalty_weight", self.phase_alignment_lag_penalty_weight),
            ("band_waveform_weight", self.band_waveform_weight),
            ("curvature_weight", self.curvature_weight),
            ("rhythm_weight", self.rhythm_weight),
            ("signed_cosine_weight", self.signed_cosine_weight),
            ("signed_corr_weight", self.signed_corr_weight),
            ("signed_rms_envelope_weight", self.signed_rms_envelope_weight),
            ("signed_mean_weight", self.signed_mean_weight),
            ("si_sdr_weight", self.si_sdr_weight),
            ("stft_dist_weight", self.stft_dist_weight),
            ("stft_band_energy_weight", self.stft_band_energy_weight),
            ("stft_peak_anchor_weight", self.stft_peak_anchor_weight),
            ("fb_aux_weight", self.fb_aux_weight),
            ("fb_consistency_weight", self.fb_consistency_weight),
            ("stft_sample_weight_min", self.stft_sample_weight_min),
        ):
            if value < 0:
                raise ValueError(f"{name} 必须非负，当前={value}")
        if self.fb_consistency_start_epoch < 1:
            raise ValueError(f"fb_consistency_start_epoch 必须为正数，当前={self.fb_consistency_start_epoch}")
        if self.stft_sample_weight_min > 1:
            raise ValueError(f"stft_sample_weight_min 必须 <= 1，当前={self.stft_sample_weight_min}")
        if self.stft_sample_weight_mode not in {
            "none",
            "waveform_confidence_score_inverse",
            "waveform_confidence_level_medlow",
        }:
            raise ValueError(f"未知 stft_sample_weight_mode: {self.stft_sample_weight_mode}")
        if self.rhythm_min_period_sec >= self.rhythm_max_period_sec:
            raise ValueError(
                "rhythm_min_period_sec 必须小于 rhythm_max_period_sec，"
                f"当前 min={self.rhythm_min_period_sec} max={self.rhythm_max_period_sec}"
            )
        if self.rhythm_step_sec <= 0:
            raise ValueError(f"rhythm_step_sec 必须为正数，当前={self.rhythm_step_sec}")
        if self.rhythm_temperature <= 0:
            raise ValueError(f"rhythm_temperature 必须为正数，当前={self.rhythm_temperature}")
        if self.phase_alignment_max_sec < 0:
            raise ValueError(f"phase_alignment_max_sec 必须非负，当前={self.phase_alignment_max_sec}")
        if self.phase_alignment_step_sec <= 0:
            raise ValueError(f"phase_alignment_step_sec 必须为正数，当前={self.phase_alignment_step_sec}")
        if self.phase_alignment_temperature <= 0:
            raise ValueError(f"phase_alignment_temperature 必须为正数，当前={self.phase_alignment_temperature}")
        if self.signed_env_temperature <= 0:
            raise ValueError(f"signed_env_temperature 必须为正数，当前={self.signed_env_temperature}")
        if self.signed_mean_window <= 0:
            raise ValueError(f"signed_mean_window_sec 必须为正数，当前={cfg.loss.get('signed_mean_window_sec', 2.0)}")
        self._validate_signed_schedule("signed_cosine_schedule", self.signed_cosine_schedule, self.signed_cosine_weight)
        self._validate_signed_schedule("signed_corr_schedule", self.signed_corr_schedule, self.signed_corr_weight)
        self.relative_env_trend_window = max(self.envelope_window, int(round(self.fs * 20.0)))
        self.low_hz = float(cfg.loss.spectrum_low_hz)
        self.high_hz = float(cfg.loss.spectrum_high_hz)
        self.target_stft_loss = (
            TargetStftLoss.from_config(cfg)
            if (
                self.stft_dist_weight > 0
                or self.stft_band_energy_weight > 0
                or self.stft_peak_anchor_weight > 0
                or self.log_component_grad_norms
            )
            else None
        )
        self.fb_stft_loss = (
            TargetStftLogMagLoss.from_config(cfg)
            if (self.fb_aux_weight > 0 or self.fb_consistency_weight > 0)
            else None
        )

    def set_epoch(self, epoch: int) -> None:
        """更新当前 epoch，供课程式 loss 权重调度使用。"""
        self.epoch = max(1, int(epoch))

    def current_weights(self) -> dict[str, float]:
        """返回当前 epoch 的有效 loss 权重，便于测试和实验诊断。"""
        return {
            "signed_cosine": self._scheduled_signed_cosine_weight(),
            "signed_corr": self._scheduled_signed_corr_weight(),
        }

    def forward(
        self,
        pred: torch.Tensor | Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        waveform, aux_outputs = self._split_model_output(pred)
        self._validate_inputs(waveform, target)
        # AMP 下模型输出可能是 float16；cuFFT 对非 2 次幂长度的 half FFT
        # 有限制，因此损失内部统一用 float32 做频域和包络计算。
        pred_loss = waveform.float()
        target_loss = target.float()
        components = self._loss_components(
            pred_loss,
            target_loss,
            sample_weight=sample_weight,
            aux_logmag=aux_outputs.get("aux_target_stft_logmag"),
        )
        total = self._weighted_total(components)
        return total, {name: value.detach() for name, value in components.items()}

    def sample_weights_from_meta(self, meta: Mapping[str, object], *, device: torch.device) -> torch.Tensor | None:
        """从 batch meta 生成 target STFT loss 的样本权重；默认关闭。"""
        if self.stft_sample_weight_mode == "none":
            return None
        if not isinstance(meta, Mapping):
            raise TypeError("meta 必须是 Mapping，才能生成 stft sample weights")
        if self.stft_sample_weight_mode == "waveform_confidence_score_inverse":
            if "waveform_confidence_score" not in meta:
                raise KeyError("meta 缺少 waveform_confidence_score，无法生成 STFT 样本权重")
            score = _meta_float_tensor(meta["waveform_confidence_score"], device=device)
            return torch.clamp(1.0 - score, min=self.stft_sample_weight_min, max=1.0)
        if "waveform_confidence_level" not in meta:
            raise KeyError("meta 缺少 waveform_confidence_level，无法生成 STFT 样本权重")
        levels = _meta_string_list(meta["waveform_confidence_level"])
        weights = [
            1.0 if level.strip().lower() in {"low", "medium"} else self.stft_sample_weight_min
            for level in levels
        ]
        return torch.tensor(weights, device=device, dtype=torch.float32)

    def _loss_components(
        self,
        pred_loss: torch.Tensor,
        target_loss: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        aux_logmag: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        env = self._envelope_loss(pred_loss, target_loss)
        spec = self._spectrum_loss(pred_loss, target_loss)
        smooth = torch.mean(torch.abs(pred_loss[..., 1:] - pred_loss[..., :-1]))
        high_freq = self._high_frequency_energy(pred_loss)
        relative_env = self._relative_envelope_loss(pred_loss, target_loss)
        band_waveform = self._band_waveform_loss(pred_loss, target_loss)
        curvature = self._curvature_loss(pred_loss)
        if self.rhythm_weight > 0:
            rhythm = self._rhythm_loss(pred_loss, target_loss)
        else:
            rhythm = pred_loss.new_tensor(0.0)
        if self.phase_alignment_weight > 0:
            phase_alignment = self._phase_alignment_loss(pred_loss, target_loss)
        else:
            phase_alignment = pred_loss.new_tensor(0.0)
        signed_cosine_weight = self._scheduled_signed_cosine_weight()
        signed_corr_weight = self._scheduled_signed_corr_weight()
        signed_cosine = (
            self._signed_cosine_loss(pred_loss, target_loss)
            if signed_cosine_weight > 0
            else pred_loss.new_tensor(0.0)
        )
        signed_corr = (
            self._signed_corr_loss(pred_loss, target_loss)
            if signed_corr_weight > 0
            else pred_loss.new_tensor(0.0)
        )
        signed_rms_envelope = (
            self._signed_rms_envelope_loss(pred_loss, target_loss)
            if self.signed_rms_envelope_weight > 0
            else pred_loss.new_tensor(0.0)
        )
        signed_mean = (
            self._signed_mean_loss(pred_loss, target_loss) if self.signed_mean_weight > 0 else pred_loss.new_tensor(0.0)
        )
        si_sdr = self._si_sdr_loss(pred_loss, target_loss) if self.si_sdr_weight > 0 else pred_loss.new_tensor(0.0)
        if self.target_stft_loss is not None:
            stft_parts = self.target_stft_loss(pred_loss, target_loss, sample_weight=sample_weight)
            stft_dist = stft_parts["stft_dist"] if self.stft_dist_weight > 0 else pred_loss.new_tensor(0.0)
            stft_band_energy = (
                stft_parts["stft_band_energy"] if self.stft_band_energy_weight > 0 else pred_loss.new_tensor(0.0)
            )
            stft_peak_anchor = (
                stft_parts["stft_peak_anchor"] if self.stft_peak_anchor_weight > 0 else pred_loss.new_tensor(0.0)
            )
        else:
            stft_dist = pred_loss.new_tensor(0.0)
            stft_band_energy = pred_loss.new_tensor(0.0)
            stft_peak_anchor = pred_loss.new_tensor(0.0)
        fb_aux = pred_loss.new_tensor(0.0)
        fb_consistency = pred_loss.new_tensor(0.0)
        if self.fb_stft_loss is not None:
            if aux_logmag is None:
                raise KeyError("F-B loss 已启用，但模型输出缺少 aux_target_stft_logmag")
            if self.fb_aux_weight > 0:
                fb_aux = self.fb_stft_loss.aux_loss(aux_logmag, target_loss)
            if self.fb_consistency_weight > 0 and self.epoch >= self.fb_consistency_start_epoch:
                consistency_target = aux_logmag.detach() if self.fb_consistency_detach_aux else aux_logmag
                fb_consistency = self.fb_stft_loss.consistency_loss(pred_loss, consistency_target)
        return {
            "envelope": env,
            "spectrum": spec,
            "smooth": smooth,
            "high_freq": high_freq,
            "relative_envelope": relative_env,
            "phase_alignment": phase_alignment,
            "band_waveform": band_waveform,
            "curvature": curvature,
            "rhythm": rhythm,
            "signed_cosine": signed_cosine,
            "signed_corr": signed_corr,
            "signed_rms_envelope": signed_rms_envelope,
            "signed_mean": signed_mean,
            "si_sdr": si_sdr,
            "stft_dist": stft_dist,
            "stft_band_energy": stft_band_energy,
            "stft_peak_anchor": stft_peak_anchor,
            "fb_aux": fb_aux,
            "fb_consistency": fb_consistency,
        }

    def _weighted_total(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self._weighted_base_total(components)
            + self._weighted_stft_total(components)
            + self._weighted_fb_total(components)
        )

    def _weighted_base_total(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.env_weight * components["envelope"]
            + self.spec_weight * components["spectrum"]
            + self.smooth_weight * components["smooth"]
            + self.high_freq_weight * components["high_freq"]
            + self.relative_env_weight * components["relative_envelope"]
            + self.phase_alignment_weight * components["phase_alignment"]
            + self.band_waveform_weight * components["band_waveform"]
            + self.curvature_weight * components["curvature"]
            + self.rhythm_weight * components["rhythm"]
            + self._scheduled_signed_cosine_weight() * components["signed_cosine"]
            + self._scheduled_signed_corr_weight() * components["signed_corr"]
            + self.signed_rms_envelope_weight * components["signed_rms_envelope"]
            + self.signed_mean_weight * components["signed_mean"]
            + self.si_sdr_weight * components["si_sdr"]
        )

    def _weighted_stft_total(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.stft_dist_weight * components["stft_dist"]
            + self.stft_band_energy_weight * components["stft_band_energy"]
            + self.stft_peak_anchor_weight * components["stft_peak_anchor"]
        )

    def _weighted_fb_total(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.fb_aux_weight * components["fb_aux"]
            + self.fb_consistency_weight * components["fb_consistency"]
        )

    def component_gradient_norms(
        self,
        pred: torch.Tensor | Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """返回关键加权 loss 分项对模型输出的梯度范数，用于 F-A 权重诊断。"""

        waveform, aux_outputs = self._split_model_output(pred)
        self._validate_inputs(waveform, target)
        pred_loss = waveform.float()
        target_loss = target.float()
        components = self._loss_components(
            pred_loss,
            target_loss,
            sample_weight=sample_weight,
            aux_logmag=aux_outputs.get("aux_target_stft_logmag"),
        )
        weighted = {
            "base_total": self._weighted_base_total(components),
            "stft_total": self._weighted_stft_total(components),
            "stft_dist": self.stft_dist_weight * components["stft_dist"],
            "stft_band_energy": self.stft_band_energy_weight * components["stft_band_energy"],
            "stft_peak_anchor": self.stft_peak_anchor_weight * components["stft_peak_anchor"],
            "fb_total": self._weighted_fb_total(components),
            "fb_aux": self.fb_aux_weight * components["fb_aux"],
            "fb_consistency": self.fb_consistency_weight * components["fb_consistency"],
        }
        return {f"grad_norm_{name}": self._grad_norm(value, waveform) for name, value in weighted.items()}

    @staticmethod
    def _grad_norm(value: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if not value.requires_grad:
            return pred.new_tensor(0.0)
        grad = torch.autograd.grad(value, pred, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            return pred.new_tensor(0.0)
        return torch.linalg.vector_norm(grad.detach())

    @staticmethod
    def _validate_signed_schedule(name: str, schedule: object, fallback_weight: float) -> None:
        if not schedule:
            return
        mode = str(schedule.get("mode", "none")).lower()
        if mode in {"none", "constant"}:
            return
        if mode != "linear":
            raise ValueError(f"{name}.mode 未知: {mode}")
        start_epoch = int(schedule.get("start_epoch", 1))
        end_epoch = int(schedule.get("end_epoch", start_epoch))
        start_weight = float(schedule.get("start_weight", fallback_weight))
        end_weight = float(schedule.get("end_weight", fallback_weight))
        if start_epoch <= 0 or end_epoch <= 0:
            raise ValueError(f"{name} 的 start_epoch/end_epoch 必须为正数")
        if end_epoch < start_epoch:
            raise ValueError(f"{name}.end_epoch 必须大于等于 start_epoch")
        if start_weight < 0 or end_weight < 0:
            raise ValueError(f"{name} 的 start_weight/end_weight 必须非负")

    def _scheduled_signed_cosine_weight(self) -> float:
        return self._scheduled_weight(self.signed_cosine_schedule, self.signed_cosine_weight)

    def _scheduled_signed_corr_weight(self) -> float:
        return self._scheduled_weight(self.signed_corr_schedule, self.signed_corr_weight)

    def _scheduled_weight(self, schedule: object, fallback_weight: float) -> float:
        if not schedule:
            return fallback_weight
        mode = str(schedule.get("mode", "none")).lower()
        if mode in {"none", "constant"}:
            return fallback_weight
        start_epoch = int(schedule.get("start_epoch", 1))
        end_epoch = int(schedule.get("end_epoch", start_epoch))
        start_weight = float(schedule.get("start_weight", fallback_weight))
        end_weight = float(schedule.get("end_weight", fallback_weight))
        if self.epoch <= start_epoch:
            return start_weight
        if self.epoch >= end_epoch:
            return end_weight
        progress = (self.epoch - start_epoch) / max(1, end_epoch - start_epoch)
        return start_weight + progress * (end_weight - start_weight)

    @staticmethod
    def _validate_inputs(pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.shape != target.shape:
            raise ValueError(f"pred 和 target shape 必须一致: pred={tuple(pred.shape)} target={tuple(target.shape)}")
        if pred.ndim != 3 or pred.shape[1] != 1:
            raise ValueError(f"pred 和 target 必须为 [B, 1, T]，当前 shape={tuple(pred.shape)}")

    @staticmethod
    def _split_model_output(pred: torch.Tensor | Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        if isinstance(pred, Mapping):
            if "waveform" not in pred:
                raise KeyError("模型输出 dict 必须包含 waveform")
            waveform = pred["waveform"]
            if not torch.is_tensor(waveform):
                raise TypeError("模型输出 waveform 必须是 Tensor")
            return waveform, pred
        return pred, {}

    def _envelope_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_env = self._zscore(self._rms_envelope(pred))
        target_env = self._zscore(self._rms_envelope(target))
        corr = torch.mean(pred_env * target_env, dim=-1)
        return torch.mean(1.0 - corr)

    def _rms_envelope(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.envelope_window // 2
        smoothed = F.avg_pool1d(x.square(), kernel_size=self.envelope_window, stride=1, padding=pad)
        if smoothed.shape[-1] > x.shape[-1]:
            smoothed = smoothed[..., : x.shape[-1]]
        return torch.sqrt(torch.clamp(smoothed, min=1e-8))

    def _spectrum_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dist = self._band_distribution(pred)
        target_dist = self._band_distribution(target)
        return torch.mean(torch.sum(torch.abs(pred_dist - target_dist), dim=-1))

    def _band_distribution(self, x: torch.Tensor) -> torch.Tensor:
        centered = self._center(x)
        power = torch.fft.rfft(centered, dim=-1).abs().square().squeeze(1)
        mask = self._band_mask(x.shape[-1], x.device)
        band = power[:, mask]
        return band / torch.clamp(band.sum(dim=-1, keepdim=True), min=1e-8)

    def _band_limited_waveform(self, x: torch.Tensor) -> torch.Tensor:
        centered = self._center(x)
        spectrum = torch.fft.rfft(centered, dim=-1)
        mask = self._band_mask(x.shape[-1], x.device).view(1, 1, -1)
        filtered = torch.where(mask, spectrum, torch.zeros_like(spectrum))
        return torch.fft.irfft(filtered, n=x.shape[-1], dim=-1)

    def _lag_samples(self, n_samples: int, max_sec: float, step_sec: float) -> list[int]:
        max_lag = int(round(max_sec * self.fs))
        step = max(1, int(round(step_sec * self.fs)))
        max_lag = min(max_lag, max(0, n_samples - 2))
        non_negative_lags = list(range(0, max_lag + 1, step))
        if max_lag not in non_negative_lags:
            non_negative_lags.append(max_lag)
        lags = non_negative_lags + [-lag for lag in non_negative_lags]
        return sorted(set(lags))

    def _lagged_corr(self, pred_band: torch.Tensor, target_band: torch.Tensor, lag_samples: int) -> torch.Tensor:
        n = pred_band.shape[-1]
        if lag_samples > 0:
            pred_slice = pred_band[..., lag_samples:]
            target_slice = target_band[..., : n - lag_samples]
        elif lag_samples < 0:
            lead = -lag_samples
            pred_slice = pred_band[..., : n - lead]
            target_slice = target_band[..., lead:]
        else:
            pred_slice = pred_band
            target_slice = target_band
        pred_norm = self._zscore(pred_slice)
        target_norm = self._zscore(target_slice)
        return torch.mean(pred_norm * target_norm, dim=-1).squeeze(1)

    def _phase_alignment_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_band = self._band_limited_waveform(pred)
        target_band = self._band_limited_waveform(target)
        zero_corr = self._lagged_corr(pred_band, target_band, 0)
        lags = self._lag_samples(pred.shape[-1], self.phase_alignment_max_sec, self.phase_alignment_step_sec)
        lag_corrs = [self._lagged_corr(pred_band, target_band, lag) for lag in lags]
        corr = torch.stack(lag_corrs, dim=-1)
        weights = torch.softmax(corr / self.phase_alignment_temperature, dim=-1)
        soft_best_corr = torch.sum(weights * corr, dim=-1)
        lag_sec = pred.new_tensor(lags, dtype=pred.dtype) / self.fs
        soft_abs_lag_sec = torch.sum(weights * lag_sec.abs(), dim=-1)
        max_lag_sec = max(self.phase_alignment_max_sec, 1.0 / self.fs)
        lag_penalty = soft_abs_lag_sec / max_lag_sec
        loss = (
            self.phase_alignment_zero_weight * (1.0 - zero_corr)
            + self.phase_alignment_lag_weight * (1.0 - soft_best_corr)
            + self.phase_alignment_lag_penalty_weight * lag_penalty
        )
        return torch.mean(loss)

    def _band_waveform_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_band = self._zscore(self._band_limited_waveform(pred))
        target_band = self._zscore(self._band_limited_waveform(target))
        return F.smooth_l1_loss(pred_band, target_band)

    def _signed_cosine_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_band = self._center(self._band_limited_waveform(pred))
        target_band = self._center(self._band_limited_waveform(target))
        numerator = torch.sum(pred_band * target_band, dim=-1)
        denominator = torch.sqrt(torch.clamp(torch.sum(pred_band.square(), dim=-1), min=1e-8)) * torch.sqrt(
            torch.clamp(torch.sum(target_band.square(), dim=-1), min=1e-8)
        )
        cosine = numerator / torch.clamp(denominator, min=1e-8)
        return torch.mean(1.0 - cosine)

    def _signed_corr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_band = self._zscore(self._band_limited_waveform(pred))
        target_band = self._zscore(self._band_limited_waveform(target))
        return self._corr_loss(pred_band, target_band)

    def _signed_rms_envelope_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_trace = self._signed_rms_envelope_trace(pred)
        target_trace = self._signed_rms_envelope_trace(target)
        return self._corr_loss(self._zscore(pred_trace), self._zscore(target_trace))

    def _signed_rms_envelope_trace(self, x: torch.Tensor) -> torch.Tensor:
        band = self._band_limited_waveform(x)
        envelope = self._rms_envelope(band)
        # 近似 sign(LPF(x))，保留梯度，避免 hard sign 阻断方向修正。
        soft_sign = torch.tanh(self._zscore(band) / self.signed_env_temperature)
        return envelope * soft_sign

    def _signed_mean_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_trace = self._moving_average(self._band_limited_waveform(pred), self.signed_mean_window)
        target_trace = self._moving_average(self._band_limited_waveform(target), self.signed_mean_window)
        return self._corr_loss(self._zscore(pred_trace), self._zscore(target_trace))

    def _si_sdr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_band = self._center(self._band_limited_waveform(pred))
        target_band = self._center(self._band_limited_waveform(target))
        target_energy = torch.sum(target_band.square(), dim=-1, keepdim=True)
        scale = torch.sum(pred_band * target_band, dim=-1, keepdim=True) / torch.clamp(target_energy, min=1e-8)
        projected = scale * target_band
        noise = pred_band - projected
        noise_ratio = torch.sum(noise.square(), dim=-1) / torch.clamp(torch.sum(projected.square(), dim=-1), min=1e-8)
        return torch.mean(torch.log1p(noise_ratio))

    @staticmethod
    def _corr_loss(pred_trace: torch.Tensor, target_trace: torch.Tensor) -> torch.Tensor:
        corr = torch.mean(pred_trace * target_trace, dim=-1)
        return torch.mean(1.0 - corr)

    @staticmethod
    def _moving_average(x: torch.Tensor, window: int) -> torch.Tensor:
        window = min(max(int(window), 1), x.shape[-1])
        if window > 1 and window % 2 == 0:
            window -= 1
        pad = window // 2
        smoothed = F.avg_pool1d(x, kernel_size=window, stride=1, padding=pad)
        if smoothed.shape[-1] > x.shape[-1]:
            smoothed = smoothed[..., : x.shape[-1]]
        return smoothed

    def _curvature_loss(self, pred: torch.Tensor) -> torch.Tensor:
        pred_norm = self._zscore(pred)
        if pred_norm.shape[-1] < 3:
            return pred_norm.new_tensor(0.0)
        second_diff = pred_norm[..., 2:] - 2.0 * pred_norm[..., 1:-1] + pred_norm[..., :-2]
        return torch.mean(torch.abs(second_diff))

    def _rhythm_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dist = self._rhythm_distribution(pred)
        target_dist = self._rhythm_distribution(target)
        if pred_dist.shape[-1] == 0:
            return pred.new_tensor(0.0)
        return torch.mean(torch.sum(torch.abs(pred_dist - target_dist), dim=-1))

    def _rhythm_distribution(self, x: torch.Tensor) -> torch.Tensor:
        x_band = self._zscore(self._band_limited_waveform(x))
        lags = self._rhythm_lags(x_band.shape[-1], x_band.device)
        if lags.numel() == 0:
            return x_band.new_zeros((*x_band.shape[:-1], 0))

        n_samples = x_band.shape[-1]
        n_fft = self._next_power_of_two(2 * n_samples - 1)
        spectrum = torch.fft.rfft(x_band, n=n_fft, dim=-1)
        autocorr = torch.fft.irfft(spectrum.abs().square(), n=n_fft, dim=-1)[..., :n_samples]
        zero_lag = torch.clamp(autocorr[..., :1], min=1e-8)
        lag_corr = autocorr.index_select(dim=-1, index=lags) / zero_lag
        overlap = (n_samples - lags).to(device=x_band.device, dtype=x_band.dtype).view(1, 1, -1)
        lag_corr = lag_corr * (float(n_samples) / torch.clamp(overlap, min=1.0))
        lag_corr = torch.clamp(lag_corr, min=-1.0, max=1.0)
        return torch.softmax(lag_corr / self.rhythm_temperature, dim=-1)

    def _rhythm_lags(self, n_samples: int, device: torch.device) -> torch.Tensor:
        min_lag = max(1, int(round(self.rhythm_min_period_sec * self.fs)))
        max_lag = min(int(round(self.rhythm_max_period_sec * self.fs)), max(0, n_samples - 2))
        if max_lag < min_lag:
            return torch.empty(0, dtype=torch.long, device=device)
        step = max(1, int(round(self.rhythm_step_sec * self.fs)))
        lags = torch.arange(min_lag, max_lag + 1, step=step, dtype=torch.long, device=device)
        if int(lags[-1].item()) != max_lag:
            lags = torch.cat([lags, torch.tensor([max_lag], dtype=torch.long, device=device)])
        return lags

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 << (int(value) - 1).bit_length()

    def _band_mask(self, n_samples: int, device: torch.device) -> torch.Tensor:
        freqs = torch.fft.rfftfreq(n_samples, d=1.0 / self.fs).to(device)
        mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        if not bool(mask.any()):
            raise ValueError(
                f"频谱损失频带为空: low_hz={self.low_hz} high_hz={self.high_hz} "
                f"fs={self.fs} n={n_samples}"
            )
        return mask

    def _high_frequency_energy(self, pred: torch.Tensor) -> torch.Tensor:
        """惩罚预测中高于呼吸频带上限的相对频谱能量。"""
        centered = pred - pred.mean(dim=-1, keepdim=True)
        power = torch.fft.rfft(centered, dim=-1).abs().square().squeeze(1)
        freqs = torch.fft.rfftfreq(pred.shape[-1], d=1.0 / self.fs).to(pred.device)
        mask = freqs > self.high_hz
        if not bool(mask.any()):
            return power.new_tensor(0.0)
        total = torch.clamp(power.sum(dim=-1), min=1e-8)
        return torch.mean(power[:, mask].sum(dim=-1) / total)

    def _relative_envelope_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_rel = self._relative_envelope_trace(pred)
        target_rel = self._relative_envelope_trace(target)
        return torch.mean(torch.abs(pred_rel - target_rel))

    def _relative_envelope_trace(self, x: torch.Tensor) -> torch.Tensor:
        env = self._rms_envelope(x)
        trend_window = min(self.relative_env_trend_window, env.shape[-1])
        # avg_pool1d 要求 padding 不超过半个 kernel；偶数窗口短序列时使用奇数窗口更稳。
        if trend_window > 1 and trend_window % 2 == 0:
            trend_window -= 1
        pad = trend_window // 2
        trend = F.avg_pool1d(env, kernel_size=trend_window, stride=1, padding=pad)
        if trend.shape[-1] > env.shape[-1]:
            trend = trend[..., : env.shape[-1]]
        rel = torch.log(torch.clamp(env, min=1e-8)) - torch.log(torch.clamp(trend, min=1e-8))
        return rel - rel.mean(dim=-1, keepdim=True)

    @staticmethod
    def _center(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=-1, keepdim=True)

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=-1, keepdim=True)
        return (x - x.mean(dim=-1, keepdim=True)) / torch.clamp(std, min=1e-6)


def _meta_float_tensor(value: object, *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 0:
        raise ValueError("waveform_confidence_score 为空")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("waveform_confidence_score 包含非有限值")
    if bool(((tensor < 0) | (tensor > 1)).any()):
        raise ValueError("waveform_confidence_score 必须位于 [0, 1]")
    return tensor


def _meta_string_list(value: object) -> list[str]:
    if torch.is_tensor(value):
        flattened = value.detach().cpu().reshape(-1).tolist()
        return [str(item) for item in flattened]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted]
        return [str(converted)]
    return [str(value)]
