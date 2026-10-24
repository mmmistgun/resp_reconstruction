"""拒绝配置中的隐式合同漂移；批量大小只能显式改变并重新做工程检查。"""

from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/respdiff_bcg_v1/experiment.yaml"


def load_config(path=DEFAULT_CONFIG):
    cfg = OmegaConf.load(path)
    reference = OmegaConf.to_container(OmegaConf.load(DEFAULT_CONFIG), resolve=True)
    actual = OmegaConf.to_container(cfg, resolve=True)
    # 只允许运行身份、设备与显式 batch 调整；其他科学合同修改需新版本。
    mutable = {("training", "seed"), ("training", "device"),
               ("training", "batch_size"), ("inference", "batch_size"),
               ("inference", "noise_seed")}
    for section, key in mutable:
        value = actual.get(section, {}).get(key)
        if key == "device":
            if not isinstance(value, str) or not value.startswith("cuda:"):
                raise ValueError("正式配置要求显式 cuda:N")
        elif type(value) is not int or value < (1 if key == "batch_size" else 0):
            raise ValueError(f"非法 {section}.{key}")
        reference[section][key] = value
    if actual != reference:
        raise ValueError("配置偏离 RespDiff-BCG v1 合同")
    return cfg
