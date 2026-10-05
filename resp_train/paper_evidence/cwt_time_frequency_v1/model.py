"""只扩展 W 输入几何；主干、条件末端、FiLM、读出与初始化保持 W0。"""
import torch
from torch.nn import functional as F
from resp_train.crd.initialization import module_seed
from resp_train.crd.tf_v1_model import CRDTfV1Model, CwtBranch, _require_feature


class NativeCwtBranch(CwtBranch):
    def __init__(self, scale_count, frames):
        super().__init__(scale_count=97)
        if not isinstance(scale_count, int) or scale_count < 1 or frames not in (180, 360, 720):
            raise ValueError("原生尺度数/时间帧非法")
        self.scale_count, self.frames = scale_count, frames

    def forward(self, tf):
        value = _require_feature(tf, "w", (self.scale_count, self.frames)).float()[:, None]
        value = F.silu(self.norm(self.conv_in(value)))
        value = F.silu(self.conv_out(self.depthwise(value))).mean(dim=2)
        value = F.interpolate(value, size=1800, mode="linear", align_corners=False)
        return self.project_condition(self.temporal(value))


def build_model(seed, rep):
    model = CRDTfV1Model("crd_tf102_w", seed)
    with module_seed(seed, "tf_branch_w"):
        branch = NativeCwtBranch(*rep["shape"])
    for key, value in branch.state_dict().items():
        if not torch.equal(value, model.branches["w"].state_dict()[key]):
            raise RuntimeError(f"公共初始化漂移: {key}")
    model.branches["w"] = branch
    if sum(p.numel() for p in model.parameters()) != 1219850:
        raise RuntimeError("W0 参数合同漂移")
    return model
