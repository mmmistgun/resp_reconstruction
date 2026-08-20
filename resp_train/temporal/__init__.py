"""RTM-v1 锁定 CPU 实现；当前不开放训练或 evaluation 入口。"""

from resp_train.temporal.config import RTM_PROTOCOL_VERSION, load_resp_temporal_config
from resp_train.temporal.model import (
    RTM_VARIANT_SPECS,
    RespTemporalModel,
    build_resp_temporal_model,
    trainable_parameter_count,
)

__all__ = [
    "RTM_PROTOCOL_VERSION",
    "RTM_VARIANT_SPECS",
    "RespTemporalModel",
    "build_resp_temporal_model",
    "load_resp_temporal_config",
    "trainable_parameter_count",
]
