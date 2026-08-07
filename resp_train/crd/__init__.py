"""CRD-Net v1.1 的独立模型、配置与训练入口。"""

from resp_train.crd.config import check_crd_dependencies, load_crd_config
from resp_train.crd.model import build_crd_model

__all__ = ["build_crd_model", "check_crd_dependencies", "load_crd_config"]
