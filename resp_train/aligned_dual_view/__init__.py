"""ADV-v1 的固定输入表示与时间对齐双视图网络。"""

from .features import CWTBatch, extract_cwt_10hz, prepare_cwt_batch, representation_spec
from .model import AlignedDualViewV1, ModelConfig, load_model_config

__all__ = [
    "AlignedDualViewV1", "ModelConfig", "load_model_config", "CWTBatch",
    "extract_cwt_10hz", "prepare_cwt_batch", "representation_spec",
]
