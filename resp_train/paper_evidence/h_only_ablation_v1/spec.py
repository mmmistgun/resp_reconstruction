"""冻结矩阵与训练合同；导入时不访问数据。"""
from pathlib import Path
from dataclasses import dataclass, asdict
from resp_train.paper_evidence.cwt_apor_v2.spec import config as source_config, SEEDS

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "h-only-ablation-v1-20261004"
GAIN_ARMS = ("HA0", "HA13", "HA14", "HA15")
GAIN_THRESHOLDS = {"near_zero_abs_lt": 0.1, "extreme_abs_gt": 3.0}
QUANTILES = (0., .01, .05, .25, .5, .75, .95, .99, 1.)


@dataclass(frozen=True)
class Arm:
    condition: bool = True
    patch_mixing: bool = True
    channel_mixing: bool = True
    mamba: bool = True
    scale_neighborhood: bool = True
    center_only: bool = False
    hidden: int = 64
    film: str = "both"
    decoder: str = "mlp"
    overlap: str = "hann"
    effort_weight: float = .25
    gain: str = "bounded"


ARMS = {
    "HA0": Arm(), "HA1": Arm(condition=False), "HA2": Arm(patch_mixing=False),
    "HA3": Arm(channel_mixing=False), "HA4": Arm(mamba=False),
    "HA5": Arm(scale_neighborhood=False), "HA6": Arm(center_only=True),
    "HA7": Arm(hidden=0), "HA8": Arm(film="add"), "HA9": Arm(film="scale"),
    "HA10": Arm(decoder="linear"), "HA11": Arm(overlap="uniform"),
    "HA12": Arm(effort_weight=0.), "HA13": Arm(gain="positive"),
    "HA14": Arm(gain="signed"), "HA15": Arm(gain="unbounded"), "HA16": Arm(hidden=65),
}


def plan():
    return [{"arm": arm, "seed": seed} for arm in ARMS for seed in SEEDS]


def contract():
    return {"protocol": PROTOCOL, "seeds": list(SEEDS),
            "arms": {k: asdict(v) for k, v in ARMS.items()},
            "gain_thresholds": GAIN_THRESHOLDS, "gain_quantiles": list(QUANTILES),
            "epochs": 80, "min_epoch": 30, "patience": 15, "min_delta": 0.,
            "planned_updates": 6400, "updates_per_epoch": 80, "batch_size": 128,
            "reference": "HA0", "historical_reference": "CWT-APOR-v2/H"}


def config(arm, seed, output, device="cpu"):
    from omegaconf import OmegaConf
    if OmegaConf.to_container(OmegaConf.load(ROOT / "configs/h_only_ablation_v1/experiment.yaml"), resolve=True) != contract():
        raise ValueError("配置与实现合同不一致；修改前须修订协议")
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("未知消融 cell")
    cfg = source_config("H", seed, output, device)
    cfg.protocol.name = PROTOCOL
    cfg.protocol.stage = "h_only_ablation_v1"
    cfg.protocol.execution_gate = "h_only_ablation_v1_session"
    del cfg.model.cwt_apor_v2
    cfg.model.variant = "h_only_ablation_v1"
    cfg.model.h_only_ablation_v1 = {"arm": arm, **asdict(ARMS[arm])}
    cfg.loss.effort_weight = ARMS[arm].effort_weight
    return cfg
