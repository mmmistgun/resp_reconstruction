"""矩阵与配置合同；导入本模块不访问数据或构造模型。"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path

PROTOCOL = "cwt-time-frequency-v1-20260930"
ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "configs/cwt_time_frequency_v1/experiment.yaml"
OUTPUT_ROOT = ROOT / "runs/cwt_time_frequency_v1"
SOURCE_LOCK = ROOT / "docs/experiments/w0_structural_factorial_v1_implementation_lock_r2_20260924.json"
SOURCE_LOCK_SHA = "32141eab672ea41435055c45cbd7ec96325f8ddef2481a210db2941222c9e9f3"
SEEDS = (20260811, 20260812, 20260813)
COUNTS = {"train": 10141, "val": 2675}
SUBJECTS = {"train": 32, "val": 7}
BASELINE = ROOT / "configs/crd_tf_v1/crd_tf102_w_formal.yaml"


@dataclass(frozen=True)
class Arm:
    name: str
    mu: float = 13.4
    voices: int = 12
    pool_samples: int = 50
    high_hz: float = 8.0
    high_only: bool = False

    def __post_init__(self):
        if self.mu not in (6, 13.4, 20) or self.voices not in (4, 8, 12, 24):
            raise ValueError("μ/voices 不在本轮矩阵")
        if self.pool_samples not in (25, 50, 100) or self.high_hz not in (.8, 2, 4, 8, 12, 20):
            raise ValueError("时间压缩/频带不在本轮矩阵")

    @property
    def frames(self):
        return 18000 // self.pool_samples


ARMS = {}
for _mu in (6, 13.4, 20):
    for _v in (4, 8, 12, 24):
        _name = "B" if (_mu, _v) == (13.4, 12) else f"Q_mu{str(_mu).replace('.', 'p')}_v{_v}"
        ARMS[_name] = Arm(_name, _mu, _v)
ARMS.update(P_025=Arm("P_025", pool_samples=25), P_100=Arm("P_100", pool_samples=100))
for _high in (.8, 2, 4, 12, 20):
    _name = "C_" + str(_high).replace(".", "p")
    ARMS[_name] = Arm(_name, high_hz=_high)
ARMS["H"] = Arm("H", high_only=True)


def plan():
    return [{"arm": arm, "seed": seed} for seed in SEEDS for arm in ARMS]


def comparisons():
    pairs = [(a, "B") for a in ARMS if a != "B"]
    pairs += list(zip(("C_2", "C_4", "B", "C_12", "C_20"), ("C_0p8", "C_2", "C_4", "B", "C_12")))
    pairs += [("B", "C_0p8"), ("B", "H"), ("H", "C_0p8")]
    return list(dict.fromkeys(pairs))


def load_spec():
    from omegaconf import OmegaConf
    value = OmegaConf.to_container(OmegaConf.load(SPEC_PATH), resolve=True)
    expected = {"protocol": PROTOCOL, "seeds": list(SEEDS), "arms": [asdict(a) for a in ARMS.values()],
                "epochs": 80, "early_stopping": False, "batch_size": 128,
                "updates_per_epoch": 80, "planned_updates": 6400,
                "shift_seed": 20260930, "shift_seconds": [30, 150], "shifts_per_row": 3,
                "case_rule": "first_middle_last_per_subject_in_row_order",
                "frequency_grid": "preserve_8hz_base_extend_scales_geometrically_select_actual_centers"}
    if value != expected:
        raise ValueError("配置与实现合同不一致；修改矩阵须同时修订协议和身份")
    return value


def config(arm, seed, output, device="cpu"):
    from resp_train.crd.config import load_crd_config
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("未知 cell")
    load_spec()
    cfg = load_crd_config(BASELINE, overrides=[f"training.seed={seed}", f"model.initialization_seed={seed}"])
    cfg.protocol.name = PROTOCOL
    cfg.protocol.stage = "cwt_time_frequency_v1"
    cfg.protocol.execution_gate = "session_data_calibration_gpu"
    cfg.model.cwt_time_frequency_v1 = asdict(ARMS[arm])
    cfg.training.device = str(device)
    cfg.training.early_stopping_enabled = False
    cfg.outputs.run_root = str(Path(output).resolve())
    return cfg
