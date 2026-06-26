from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


DEFAULT_BAND_ENERGY_BANDS = (
    (0.05, 0.3),
    (0.1, 0.7),
    (0.3, 1.2),
    (1.2, 3.0),
)
DEFAULT_BAND_ENERGY_WEIGHTS = (0.5, 1.0, 0.5, 0.05)


class TargetStftLoss(torch.nn.Module):
    """目标 waveform 的 STFT 派生监督项。"""

    def __init__(
        self,
        *,
        sample_rate: float,
        win_length: int = 3000,
        hop_length: int = 500,
        n_fft: int | None = None,
        center: bool = False,
        dist_low_hz: float = 0.067,
        dist_high_hz: float = 1.2,
        dist_beta: float = 3.0,
        frame_weight_mode: str = "target_band_energy",
        band_energy_bands: Sequence[Sequence[float]] | None = None,
        band_energy_weights: Sequence[float] | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft or win_length)
        self.center = bool(center)
        self.dist_low_hz = float(dist_low_hz)
        self.dist_high_hz = float(dist_high_hz)
        self.dist_beta = float(dist_beta)
        self.frame_weight_mode = str(frame_weight_mode).lower()
        self.band_energy_bands = _as_band_pairs(band_energy_bands or DEFAULT_BAND_ENERGY_BANDS)
        self.band_energy_weights = _as_weights(
            band_energy_weights or DEFAULT_BAND_ENERGY_WEIGHTS,
            expected=len(self.band_energy_bands),
        )
        self.eps = float(eps)
        self._validate_config()
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    @classmethod
    def from_config(cls, cfg) -> "TargetStftLoss":
        loss_cfg = cfg.loss
        bands = loss_cfg.get("stft_band_energy_bands", None)
        weights = loss_cfg.get("stft_band_energy_weights", None)
        if OmegaConf.is_config(bands):
            bands = OmegaConf.to_container(bands, resolve=True)
        if OmegaConf.is_config(weights):
            weights = OmegaConf.to_container(weights, resolve=True)
        return cls(
            sample_rate=float(cfg.window.target_fs),
            win_length=int(loss_cfg.get("stft_win_length", 3000)),
            hop_length=int(loss_cfg.get("stft_hop_length", 500)),
            n_fft=int(loss_cfg.get("stft_n_fft", loss_cfg.get("stft_win_length", 3000))),
            center=_as_bool(loss_cfg.get("stft_center", False)),
            dist_low_hz=float(loss_cfg.get("stft_dist_low_hz", 0.067)),
            dist_high_hz=float(loss_cfg.get("stft_dist_high_hz", 1.2)),
            dist_beta=float(loss_cfg.get("stft_dist_beta", 3.0)),
            frame_weight_mode=str(loss_cfg.get("stft_frame_weight_mode", "target_band_energy")),
            band_energy_bands=bands,
            band_energy_weights=weights,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(pred, target)
        sample_weight = _prepare_sample_weight(
            sample_weight,
            batch_size=pred.shape[0],
            device=pred.device,
            dtype=pred.dtype,
            eps=self.eps,
        )
        pred_logmag, pred_power = self._stft_features(pred)
        target_logmag, target_power = self._stft_features(target)
        dist_mask = self._frequency_mask(self.dist_low_hz, self.dist_high_hz, pred.device)
        frame_weights = self._frame_weights(target_power, dist_mask)
        return {
            "stft_dist": self._distribution_loss(
                pred_logmag,
                target_logmag,
                dist_mask,
                frame_weights,
                sample_weight,
            ),
            "stft_band_energy": self._band_energy_loss(pred_power, target_power, frame_weights, sample_weight),
        }

    def _validate_config(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate 必须为正数，当前={self.sample_rate}")
        if self.win_length <= 0 or self.hop_length <= 0 or self.n_fft <= 0:
            raise ValueError("win_length、hop_length 和 n_fft 必须为正数")
        if self.n_fft < self.win_length:
            raise ValueError(f"n_fft 必须大于等于 win_length，当前 n_fft={self.n_fft} win={self.win_length}")
        if self.dist_low_hz >= self.dist_high_hz:
            raise ValueError("stft_dist_low_hz 必须小于 stft_dist_high_hz")
        if self.dist_beta <= 0:
            raise ValueError(f"stft_dist_beta 必须为正数，当前={self.dist_beta}")
        if self.frame_weight_mode not in {"none", "target_band_energy"}:
            raise ValueError(f"未知 stft_frame_weight_mode: {self.frame_weight_mode}")
        for (low, high), weight in zip(self.band_energy_bands, self.band_energy_weights, strict=True):
            if low >= high:
                raise ValueError(f"band energy 频带 low 必须小于 high，当前=({low}, {high})")
            if weight < 0:
                raise ValueError(f"band energy 权重必须非负，当前={weight}")

    @staticmethod
    def _validate_inputs(pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.shape != target.shape:
            raise ValueError(f"pred 和 target shape 必须一致: pred={tuple(pred.shape)} target={tuple(target.shape)}")
        if pred.ndim != 3 or pred.shape[1] != 1:
            raise ValueError(f"pred 和 target 必须为 [B, 1, T]，当前 shape={tuple(pred.shape)}")

    def _stft_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.center and x.shape[-1] < self.win_length:
            raise ValueError(
                f"center=False 时 STFT win_length 不能大于输入长度: win={self.win_length} length={x.shape[-1]}"
            )
        window = self.window.to(device=x.device, dtype=x.dtype)
        spectrum = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            return_complex=True,
        )
        magnitude = spectrum.abs()
        return torch.log1p(magnitude), magnitude.square()

    def _frequency_mask(self, low_hz: float, high_hz: float, device: torch.device) -> torch.Tensor:
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate).to(device)
        mask = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
        if not bool(mask.any()):
            raise ValueError(
                f"STFT 频带为空: low_hz={low_hz} high_hz={high_hz} "
                f"fs={self.sample_rate} n_fft={self.n_fft}"
            )
        return mask

    def _frame_weights(self, target_power: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.frame_weight_mode == "none":
            return torch.ones(target_power.shape[0], target_power.shape[-1], device=target_power.device)
        energy = target_power[:, mask, :].sum(dim=1)
        mean_energy = energy.mean(dim=-1, keepdim=True)
        return energy / torch.clamp(mean_energy, min=self.eps)

    def _distribution_loss(
        self,
        pred_logmag: torch.Tensor,
        target_logmag: torch.Tensor,
        mask: torch.Tensor,
        frame_weights: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        pred_prob = torch.softmax(self.dist_beta * pred_logmag[:, mask, :], dim=1)
        target_prob = torch.softmax(self.dist_beta * target_logmag[:, mask, :], dim=1)
        middle = 0.5 * (pred_prob + target_prob)
        pred_kl = torch.sum(pred_prob * (torch.log(torch.clamp(pred_prob, min=self.eps)) - torch.log(middle)), dim=1)
        target_kl = torch.sum(
            target_prob * (torch.log(torch.clamp(target_prob, min=self.eps)) - torch.log(middle)),
            dim=1,
        )
        jsd = 0.5 * (pred_kl + target_kl)
        return _weighted_frame_mean(jsd, frame_weights, self.eps, sample_weight=sample_weight)

    def _band_energy_loss(
        self,
        pred_power: torch.Tensor,
        target_power: torch.Tensor,
        frame_weights: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        total = pred_power.new_tensor(0.0)
        total_weight = 0.0
        for band, weight in zip(self.band_energy_bands, self.band_energy_weights, strict=True):
            if weight == 0:
                continue
            mask = self._frequency_mask(band[0], band[1], pred_power.device)
            pred_energy = torch.log1p(pred_power[:, mask, :].sum(dim=1))
            target_energy = torch.log1p(target_power[:, mask, :].sum(dim=1))
            loss = F.smooth_l1_loss(pred_energy, target_energy, reduction="none")
            total = total + float(weight) * _weighted_frame_mean(
                loss,
                frame_weights,
                self.eps,
                sample_weight=sample_weight,
            )
            total_weight += float(weight)
        if total_weight <= 0:
            return pred_power.new_tensor(0.0)
        return total / total_weight


def _weighted_frame_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    denom = weights.sum(dim=-1)
    numerator = (values * weights).sum(dim=-1)
    per_sample = numerator / torch.clamp(denom, min=eps)
    per_sample = torch.where(denom > eps, per_sample, torch.zeros_like(per_sample))
    if sample_weight is not None:
        sample_denom = sample_weight.sum()
        if float(sample_denom.detach().cpu()) <= eps:
            return per_sample.new_tensor(0.0)
        return (per_sample * sample_weight).sum() / torch.clamp(sample_denom, min=eps)
    return per_sample.mean()


def _prepare_sample_weight(
    sample_weight: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor | None:
    if sample_weight is None:
        return None
    weight = torch.as_tensor(sample_weight, device=device, dtype=dtype).reshape(-1)
    if weight.numel() != int(batch_size):
        raise ValueError(f"sample_weight 长度必须等于 batch size: weight={weight.numel()} batch={batch_size}")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("sample_weight 包含非有限值")
    if bool((weight < 0).any()):
        raise ValueError("sample_weight 必须非负")
    return torch.where(weight > eps, weight, torch.zeros_like(weight))


def _as_band_pairs(value: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    pairs = []
    for item in value:
        if len(item) != 2:
            raise ValueError(f"band energy 频带必须是 [low, high]，当前={item}")
        pairs.append((float(item[0]), float(item[1])))
    return tuple(pairs)


def _as_weights(value: Sequence[float], *, expected: int) -> tuple[float, ...]:
    weights = tuple(float(item) for item in value)
    if len(weights) != int(expected):
        raise ValueError(f"band energy 权重数量必须等于频带数量: weights={len(weights)} bands={expected}")
    return weights


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
