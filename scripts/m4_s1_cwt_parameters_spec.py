"""M4-S1 的固定矩阵；plan/config 不读取数据或构造模型。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/m4_s1_cwt_parameters_v1/design.yaml"
PROTOCOL = "m4-s1-cwt-parameters-v1-20261020"
PROTOCOL_PATH = ROOT / "docs/experiments/m4_s1_cwt_parameters_v1_design_20261020.md"
OUTPUT_ROOT = Path("/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1")
SEEDS = (20260812, 20260813, 20260814)
COUNTS = {"train": 10141, "val": 2675}
SUBJECTS = {"train": 32, "val": 7}


@dataclass(frozen=True)
class Arm:
    name: str
    representation: str | None
    readout: str = "attention"
    frequency_bias: bool = True
    mamba: bool = True
    pooling: str = "attention"
    score_position: bool = True
    value_position: bool = True


def cwt_settings():
    rows = []
    for mu in (6, 13.4, 20):
        for voices in (4, 8, 12, 24):
            name = "B" if (mu, voices) == (13.4, 12) else f"Q_mu{str(mu).replace('.', 'p')}_v{voices}"
            rows.append(dict(id=name, mu=mu, voices=voices, pool_samples=50, high_hz=8, high_only=False))
    rows.extend(dict(id=name, mu=13.4, voices=12, pool_samples=p, high_hz=8, high_only=False)
                for name, p in (("P_025", 25), ("P_100", 100)))
    rows.extend(dict(id=f"C_{str(h).replace('.', 'p')}", mu=13.4, voices=12,
                     pool_samples=50, high_hz=h, high_only=False) for h in (.8, 2, 4, 12, 20))
    rows.append(dict(id="H", mu=13.4, voices=12, pool_samples=50, high_hz=8, high_only=True))
    return rows


CWT_SETTINGS = {r["id"]: r for r in cwt_settings()}
ARMS = {a: Arm("M4-S1" if a == "B" else a, a) for a in CWT_SETTINGS}
ARMS.update({
    "A_NoTF": Arm("NoTF", None, readout="none", frequency_bias=False),
    "A_UniformTFReadout": Arm("UniformTFReadout", "B", readout="uniform", frequency_bias=False),
    "A_NoFrequencyBias": Arm("NoFrequencyBias", "B", frequency_bias=False),
    "A_NoMamba": Arm("NoMamba", "B", mamba=False),
    "A_UniformPatchPool": Arm("UniformPatchPool", "B", pooling="uniform", score_position=False),
    "A_NoPoolPosition": Arm("NoPoolPosition", "B", score_position=False, value_position=False),
    "A_ScorePositionOnly": Arm("ScorePositionOnly", "B", value_position=False),
    "A_ValuePositionOnly": Arm("ValuePositionOnly", "B", score_position=False),
})
CORE = ("B", "A_NoTF", "A_UniformTFReadout", "A_NoFrequencyBias", "A_NoMamba",
        "A_UniformPatchPool", "A_NoPoolPosition")
POSITION = {"B": (True, True), "A_NoPoolPosition": (False, False),
            "A_ScorePositionOnly": (True, False), "A_ValuePositionOnly": (False, True)}
POOL_PAIRS = {
    "score_given_value_off": ("A_ScorePositionOnly", "A_NoPoolPosition"),
    "score_given_value_on": ("B", "A_ValuePositionOnly"),
    "value_given_score_off": ("A_ValuePositionOnly", "A_NoPoolPosition"),
    "value_given_score_on": ("B", "A_ScorePositionOnly"),
}
POOL_EFFECTS = {
    "score_main": {"B": .5, "A_ScorePositionOnly": .5, "A_ValuePositionOnly": -.5, "A_NoPoolPosition": -.5},
    "value_main": {"B": .5, "A_ScorePositionOnly": -.5, "A_ValuePositionOnly": .5, "A_NoPoolPosition": -.5},
    "interaction": {"B": 1, "A_ScorePositionOnly": -1, "A_ValuePositionOnly": -1, "A_NoPoolPosition": 1},
}
MODULE_CHANGES = (
    ("A_NoTF", "none", "remove_condition_branch"),
    ("A_UniformTFReadout", "B", "uniform_Fx2_weights_keep_encoder_value_output"),
    ("A_NoFrequencyBias", "B", "remove_frequency_bias"),
    ("A_NoMamba", "B", "replace_bimamba2_with_identity"),
    ("A_UniformPatchPool", "B", "uniform_pool_keep_position_values"),
    ("A_NoPoolPosition", "B", "remove_pool_score_and_value_position_keep_content_attention"),
    ("A_ScorePositionOnly", "B", "keep_score_position_remove_value_position"),
    ("A_ValuePositionOnly", "B", "keep_value_position_remove_score_position"),
)


def spec():
    value = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    expected_budget = {"cwt_configurations": 20, "core_module_configurations": 7, "core_module_cells": 21,
        "supplementary_position_configurations": 2, "supplementary_position_cells": 6,
        "module_comparison_configurations": 11, "shared_configurations": 3,
        "unique_configurations": 28, "scientific_cells": 84, "historical_reference_cells": 0, "new_training_cells": 84}
    expected_training = {"source": "configs/m4_linear_fusion_v1/experiment.yaml",
        "batch_size": 32, "gradient_accumulation_steps": 4, "amp_dtype": "bfloat16", "epochs": 80,
        "updates_per_epoch": 80, "planned_updates": 6400, "early_stopping_min_epoch": 30,
        "early_stopping_patience": 15, "early_stopping_min_delta": 0.,
        "selector": "full_validation_local_rr_strict_minimum_earliest_tie"}
    if (value["design_id"] != PROTOCOL or value["seeds"] != list(SEEDS)
            or value["baseline_arm"] != "B" or value["budget"] != expected_budget
            or value["arms"] != cwt_settings() or value["training"] != expected_training
            or value["module_arms"] != [dict(id=a, representation=r, change=c) for a, r, c in MODULE_CHANGES]
            or value["core_module_arms"] != list(CORE)
            or value["supplementary_position_arms"] != ["A_ScorePositionOnly", "A_ValuePositionOnly"]
            or value["module_comparison_arms"] != [*CORE, "A_ScorePositionOnly", "A_ValuePositionOnly", "H", "C_0p8"]):
        raise ValueError("M4-S1 矩阵、seed或训练预算与实现合同不符")
    if value["readout"] != {"patch_samples": 100, "hop_samples": 50, "patch_count": 359,
            "anchor_offsets_samples": [24.5, 74.5], "stage": "encoded_features_before_key_value",
            "interpolation": "linear_float32_weights_accumulation", "boundary": "nearest_endpoint",
            "pool50_path": "exact_integer_gather"}:
        raise ValueError("物理读取合同不符")
    if value["pool_position_ablation"] != {"content_attention": True, "retained": ["stem", "score", "project"],
            "factorial": [dict(arm=a, score_position=s, value_position=v) for a, (s, v) in POSITION.items()]}:
        raise ValueError("池化位置2×2合同不符")
    if value["model"] != {"name": "M4-S1", "source_class": "scripts.m4_linear_fusion_model.LinearFusionModel",
            "source_arm": "L1", "parameters": 1046945, "frequency_bias": True, "time_bias": False,
            "output_projections": 1, "output_initialization": "zero"}:
        raise ValueError("M4-S1基准结构合同不符")
    if value["representation"] != {"wavelet": "morlet", "library": "ssqueezepy==0.6.6", "sample_rate": 100,
            "window_samples": 18000, "feature": "mean_pool_log1p_abs", "padtype": "reflect",
            "frequency_grid": "preserve_8hz_base_extend_scales_geometrically_select_actual_centers",
            "frequency_sort": "stable_ascending_keep_duplicates"}:
        raise ValueError("CWT变换合同不符")
    if value["uniform_tf_readout"] != {"weights": "one_over_F_times_2_per_head",
            "retained": ["encoder", "value", "output"],
            "inactive_modules": "remove_query_key_and_coordinate_bias_after_baseline_initialization",
            "output_initialization": "baseline_zero", "accumulation_dtype": "float32"}:
        raise ValueError("均匀读取的值路径合同不符")
    analysis = value["analysis"]
    if (analysis["main_pair_count"] != 27 or analysis["paired_against"] != "B"
            or analysis["primary_aggregation"] != "window_weighted"
            or analysis["diagnostic_aggregation"] != "subject_macro" or analysis["mean_delta_threshold"] != .01
            or analysis["automatic_default_change"] is not False
            or analysis["additional_comparisons"] != ["mu_voices_interaction", "adjacent_frequency_caps", "full_L_H", "P_025_vs_P_100"]
            or analysis["module_additional_pairs"] != [
                ["A_UniformTFReadout", "A_NoTF"], ["A_NoFrequencyBias", "A_UniformTFReadout"],
                ["B", "A_UniformTFReadout"], ["B", "A_NoPoolPosition"], ["B", "A_NoFrequencyBias"],
                ["B", "A_NoTF"], ["H", "A_NoTF"], ["C_0p8", "A_NoTF"]]
            or analysis["pool_position_conditional_pairs"] != {k: list(v) for k, v in POOL_PAIRS.items()}
            or analysis["pool_position_effect_coefficients"] != POOL_EFFECTS):
        raise ValueError("汇总或交互效应合同不符")
    if (value["training_identity"] != {"policy": "all_cells_from_scratch_in_new_session", "historical_reference_cells": 0}
            or value["execution"]["intended_training_splits"] != ["train", "val"]
            or value["execution"]["research_test_authorized"] is not False):
        raise ValueError("训练身份或数据访问范围不符")
    return value


def contract(arm):
    if arm not in ARMS:
        raise ValueError("未知实验臂")
    return {"arm": arm, **asdict(ARMS[arm])}


def plan(shard_index=0, shard_count=1):
    spec()
    if type(shard_count) is not int or type(shard_index) is not int or not 1 <= shard_count <= 84 or not 0 <= shard_index < shard_count:
        raise ValueError("非法分片编号/数量")
    return [{"arm": a, "name": ARMS[a].name, "seed": s, "cell": f"{a}_seed{s}", "role": "train"}
            for s in SEEDS for a in ARMS][shard_index::shard_count]


def representations():
    """只生成物理网格，不读取真实信号。保持前序变换的尺度/频率定义。"""
    from resp_train.paper_evidence.cwt_time_frequency_v1.features import representation
    from resp_train.paper_evidence.cwt_time_frequency_v1.spec import Arm as CWTArm
    return {a: representation(CWTArm(name=a, **{k: v for k, v in row.items() if k != "id"}))
            for a, row in CWT_SETTINGS.items()}


def config(arm, seed, output, device="cpu", *, representation=None):
    from omegaconf import OmegaConf
    from resp_train.crd.config import load_crd_config
    from scripts.p1_components_model import architecture
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("cell不属于28配置×三个seed")
    settings = spec()
    # 先核验旧协议的完整科学模板；本轮seed白名单由上面的独立合同控制。
    # 历史CRD formal校验只接受11/12/13，不能为新实验14号seed扩大旧入口。
    cfg = load_crd_config(ROOT / "configs/crd_tf_v1/crd_tf102_w_formal.yaml")
    cfg.training.seed = seed
    cfg.model.initialization_seed = seed
    cfg.protocol.name = PROTOCOL
    cfg.protocol.stage = "m4_s1_cwt_parameters_v1"
    cfg.protocol.run_role = "formal"
    cfg.protocol.execution_gate = "session_data_gpu_acceptance"
    cfg.model.name = "m4_s1_cwt_parameters_v1"
    cfg.model.variant = ARMS[arm].name
    cfg.model.m4_s1_cwt_parameters = contract(arm)
    cfg.model.architecture = asdict(architecture())
    cfg.model.cwt_representation = representation
    # 数据层先构造原始波形dataset，再显式注入当前表示的独立cache。
    cfg.model.tf_representations = []
    cfg.data.tf_cache_path = None
    cfg.data.preload_windows = False
    for key in ("batch_size", "gradient_accumulation_steps", "amp_dtype", "epochs",
                "early_stopping_min_epoch", "early_stopping_patience", "early_stopping_min_delta"):
        cfg.training[key] = settings["training"][key]
    cfg.training.device = str(device)
    cfg.training.early_stopping_enabled = True
    cfg.training.show_progress = False
    cfg.outputs.run_root = str(Path(output).resolve())
    # 提前解析初始化seed，避免快照中的插值随外部配置改变。
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
