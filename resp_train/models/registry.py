from __future__ import annotations

from typing import Any, Callable

from resp_train.models.unet1d import UNet1DTiny


ModelFactory = Callable[[Any], UNet1DTiny]


_REGISTRY: dict[str, ModelFactory] = {
    "unet1d_tiny": lambda cfg: UNet1DTiny(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
    ),
}


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(cfg: Any) -> UNet1DTiny:
    name = cfg.model.name
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(list_models())
        raise KeyError(f"未知模型: {name}。可用模型: {available}") from exc
    return factory(cfg)
