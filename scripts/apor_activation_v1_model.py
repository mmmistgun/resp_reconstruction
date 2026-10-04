"""已冻结N0结构上的单因素激活规范化。"""
from dataclasses import dataclass

from torch import nn
from omegaconf import OmegaConf
from resp_train.paper_evidence.patch_apor_v1_model import PatchAporModel, model_contract as original_contract

PROTOCOL = "apor-activation-v1-20260930"
SEEDS = (20260811, 20260812, 20260813)


@dataclass(frozen=True)
class ArmSpec:
    source_arm: str = "A0"
    unified: bool = False
    condition: bool = True


ARM_SPECS = {"U0": ArmSpec(), "U1": ArmSpec(unified=True)}
ARMS = tuple(ARM_SPECS)
TRAIN_ARMS = ("U1",)
REFERENCE_ARMS = ("U0",)


def model_contract(arm):
    spec = ARM_SPECS[arm]
    return {"protocol": PROTOCOL, "arm": arm, "source": original_contract("A0"),
            "parameters": 1077640, "patch_mixer_activation": "silu" if spec.unified else "gelu",
            "changed_activation_count": 4 if spec.unified else 0,
            "normalization_unchanged": True, "selected_structure": "N0"}


class ActivationModel(PatchAporModel):
    def __init__(self, arm, seed):
        spec = ARM_SPECS[arm]
        super().__init__("A0", seed)
        self.study_arm = arm
        if spec.unified:
            changed = 0
            for block in self.base.frontend.encoder.blocks:
                if not isinstance(block.patch_mixer[2], nn.GELU) or not isinstance(block.channel_mixer[1], nn.GELU):
                    raise ValueError("冻结PatchMixer激活位置漂移")
                block.patch_mixer[2] = nn.SiLU()
                block.channel_mixer[1] = nn.SiLU()
                changed += 2
            if changed != 4:
                raise ValueError("激活比较要求恰好替换4处GELU")
        if sum(p.numel() for p in self.parameters()) != 1077640:
            raise RuntimeError("激活规范化改变了参数数量")


def build_model(cfg):
    contract = OmegaConf.to_container(cfg.model.apor_activation_v1, resolve=True)
    arm = contract["arm"]
    if cfg.protocol.name != PROTOCOL or contract != model_contract(arm):
        raise ValueError("激活规范化模型合同不符")
    return ActivationModel(arm, int(cfg.model.initialization_seed))
