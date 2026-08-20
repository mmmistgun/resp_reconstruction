from pathlib import Path

import pytest

from resp_train.temporal.config import (
    CANDIDATE_LOCK_SHA256,
    RTM_PROTOCOL_VERSION,
    SIGNAL_LOCK_SHA256,
    load_resp_temporal_config,
)
from resp_train.temporal.model import RTM_VARIANT_SPECS


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs/resp_temporal_v1"


def _candidate_paths() -> list[Path]:
    return sorted(CONFIG_ROOT.glob("rtm_v1_*.yaml"))


def test_exactly_five_locked_candidate_configs_are_implementation_only() -> None:
    paths = _candidate_paths()
    assert len(paths) == len(RTM_VARIANT_SPECS) == 5
    seen: set[str] = set()
    for path in paths:
        cfg = load_resp_temporal_config(path)
        variant = str(cfg.model.variant)
        seen.add(variant)
        spec = RTM_VARIANT_SPECS[variant]
        assert str(cfg.protocol.name) == RTM_PROTOCOL_VERSION
        assert str(cfg.protocol.stage) == "exact_cpu_implementation"
        assert cfg.protocol.exact_implementation_enabled is True
        assert cfg.protocol.gpu_engineering_enabled is False
        assert cfg.protocol.formal_training_enabled is False
        assert cfg.protocol.validation_evaluation_enabled is False
        assert cfg.protocol.research_test_enabled is False
        assert str(cfg.locks.signal_substrate.sha256) == SIGNAL_LOCK_SHA256
        assert str(cfg.locks.candidate.sha256) == CANDIDATE_LOCK_SHA256
        assert list(cfg.substrate.order) == [
            "fixed_anti_alias_100_to_20",
            "shared_learned_filtering_and_nonlinearity_at_20",
            "fixed_anti_alias_20_to_10",
        ]
        assert list(cfg.multiscale.grids_hz) == [10.0, 2.0, 1.0]
        assert cfg.multiscale.average_pooling_allowed is False
        assert int(cfg.model.expected_trainable_parameters) == spec.expected_trainable_parameters
        assert str(cfg.model.family) == spec.family
        assert str(cfg.model.role) == spec.role
        assert int(cfg.model.initialization_seed) == int(cfg.training.seed) == 20260811
    assert seen == set(RTM_VARIANT_SPECS)


@pytest.mark.parametrize(
    "override",
    [
        "protocol.gpu_engineering_enabled=true",
        "protocol.formal_training_enabled=true",
        "protocol.validation_evaluation_enabled=true",
        "protocol.research_test_enabled=true",
        "substrate.anti_alias.stage_100_to_20.cutoff_hz=8.5",
        "multiscale.grids_hz=[10.0,2.0,0.5]",
        "multiscale.average_pooling_allowed=true",
        "model.family=bilstm",
        "training.epochs=50",
    ],
)
def test_config_rejects_scientific_or_authorization_drift(override: str) -> None:
    path = CONFIG_ROOT / "rtm_v1_tcn_d9_h384.yaml"
    with pytest.raises(ValueError):
        load_resp_temporal_config(path, overrides=[override])


def test_config_rejects_unknown_fields_and_unpaired_seed() -> None:
    path = CONFIG_ROOT / "rtm_v1_bilstm_h96_l2.yaml"
    with pytest.raises(ValueError, match="字段必须严格"):
        load_resp_temporal_config(path, overrides=["model.unregistered=true"])
    with pytest.raises(ValueError, match="initialization_seed"):
        load_resp_temporal_config(path, overrides=["model.initialization_seed=20260812"])
