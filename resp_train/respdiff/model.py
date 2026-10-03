"""按 RespDiff 3ff0554/model_fft.py 移植；保留来源参数名与初始化顺序。"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} 含非有限值")


def waveform(name: str, value: torch.Tensor) -> None:
    if value.ndim != 3 or value.shape[0] < 1 or value.shape[1] != 1 or value.shape[2] < 1:
        raise ValueError(f"{name} 要求非空 [B,1,L]，实际 {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} 当前仅验收FP32，实际 {value.dtype}")
    finite(name, value)


@dataclass(frozen=True)
class RespDiffSpec:
    hidden_dim: int = 1024
    num_layers: int = 6
    output_dim: int = 128
    profile: str = "source_fft"

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.hidden_dim, self.num_layers, self.output_dim)):
            raise ValueError("网络维数与层数必须为正整数")
        if self.profile not in {"source_fft", "source_plain"}:
            raise ValueError(f"未知来源profile: {self.profile}")


class ConditionalTimeGrad(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.rnn = nn.RNN(384, spec.hidden_dim, spec.num_layers,
                          nonlinearity="tanh", batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * spec.hidden_dim, spec.hidden_dim)
        self.fc2 = nn.Linear(spec.hidden_dim, spec.output_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.fc2(F.relu(self.fc1(x)))


class Decoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class SignalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, 32, ks, padding=ks // 2) for ks in (1, 3, 5, 7, 9, 11)
        ])
        for layer in self.conv_layers:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        return torch.cat([layer(x) for layer in self.conv_layers], dim=1)


class DilatedConv(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        for index, dilation in enumerate((1, 2, 4)):
            setattr(self, f"layer{index}", nn.Conv1d(
                32, 32, kernel_size, dilation=dilation, padding=kernel_size // 2 * dilation))
        for index in range(3):
            setattr(self, f"bn{index}", nn.BatchNorm1d(32, track_running_stats=False))
        self.relu = nn.ReLU()
        for layer in (self.layer0, self.layer1, self.layer2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        self.bottle = nn.Conv1d(1, 32, 1)
        nn.init.kaiming_normal_(self.bottle.weight)

    def forward(self, x):
        x = self.bottle(x)
        for index in range(3):
            x = getattr(self, f"bn{index}")(self.relu(getattr(self, f"layer{index}")(x)) + x)
        return x


class SignalEncoderDilated(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList([DilatedConv(ks) for ks in (3, 5, 7, 9, 11, 13)])

    def forward(self, x):
        return torch.cat([layer(x) for layer in self.conv_layers], dim=1)


class DiffusionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        steps = torch.arange(50).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(64.0) / 63.0 * 4.0).unsqueeze(0)
        table = steps * frequencies
        self.register_buffer("embedding", torch.cat([table.sin(), table.cos()], dim=1), persistent=False)
        self.projection1 = nn.Linear(128, 192)
        self.projection2 = nn.Linear(192, 192)

    def forward(self, step):
        return F.silu(self.projection2(F.silu(self.projection1(self.embedding[step]))))


class RespDiffDenoiser(nn.Module):
    def __init__(self, spec: RespDiffSpec = RespDiffSpec()):
        super().__init__()
        self.spec = spec
        self.diff_model = ConditionalTimeGrad(spec)
        self.diffusion_embedding = DiffusionEmbedding()
        self.ppg_encoder1 = SignalEncoderDilated()
        self.ppg_encoder2 = SignalEncoder()
        self.noise_encoder = SignalEncoder()
        self.de = Decoder(spec.output_dim)
        self.weight = nn.Parameter(torch.ones(1, 192, 1))

    def forward(self, condition, noisy_target, step):
        waveform("condition", condition)
        waveform("noisy_target", noisy_target)
        if condition.shape != noisy_target.shape or condition.device != noisy_target.device:
            raise ValueError("condition/noisy_target的shape与device必须相同")
        if (step.dtype != torch.long or step.shape != (condition.shape[0],)
                or step.device != condition.device or torch.any((step < 0) | (step >= 50))):
            raise ValueError("step须为同device的[B] int64，取值0…49")
        embedding = self.diffusion_embedding(step).unsqueeze(2)
        f1 = self.ppg_encoder2(condition) + embedding + self.weight * self.ppg_encoder1(condition)
        f2 = self.noise_encoder(noisy_target)
        if self.spec.profile == "source_plain":
            f2 = f2 + embedding
        features = torch.cat([f1, f2], dim=1).permute(0, 2, 1)
        output = self.de(F.relu(self.diff_model(features))).permute(0, 2, 1)
        finite("predicted_noise", output)
        return output
