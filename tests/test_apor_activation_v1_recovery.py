"""故障恢复的CPU合成测试：checkpoint迁移与样本故障边界。"""
import copy

import pytest
import torch
from omegaconf import OmegaConf

from scripts.recover_apor_activation_v1 import assert_same_tree, migrate_checkpoint


def test_migration_preserves_original_and_all_training_state(tmp_path):
    old = OmegaConf.create({"outputs": {"run_root": "old"}, "training": {"seed": 17}})
    new = copy.deepcopy(old)
    new.outputs.run_root = "new"
    payload = {"config": OmegaConf.to_container(old), "model_state_dict": {"weight": torch.arange(4.)},
               "optimizer_state_dict": {"state": {0: {"step": torch.tensor(80.), "exp_avg": torch.ones(4)}}},
               "epoch": 1, "metrics": {"rr": 2.5}}
    source, target = tmp_path / "old.pt", tmp_path / "new.pt"
    torch.save(payload, source)
    before = source.read_bytes()
    receipt = migrate_checkpoint(source, target, old, new)
    saved = torch.load(target, weights_only=False)
    assert saved["config"]["outputs"]["run_root"] == "new"
    saved["config"] = payload["config"]
    assert_same_tree(payload, saved)
    assert source.read_bytes() == before
    assert receipt["other_payload_exact"] is True
    with pytest.raises(FileExistsError):
        migrate_checkpoint(source, target, old, new)


def test_migration_rejects_scientific_config_change(tmp_path):
    old = OmegaConf.create({"outputs": {"run_root": "old"}, "training": {"seed": 17}})
    new = copy.deepcopy(old)
    new.training.seed = 18
    source = tmp_path / "old.pt"
    torch.save({"config": OmegaConf.to_container(old)}, source)
    with pytest.raises(ValueError, match="只允许改变"):
        migrate_checkpoint(source, tmp_path / "new.pt", old, new)
    assert not (tmp_path / "new.pt").exists()


def test_exact_check_rejects_weight_and_optimizer_mutation():
    for key in ("model", "optimizer"):
        left = {"model": torch.ones(2), "optimizer": [torch.zeros(2)]}
        right = copy.deepcopy(left)
        tensor = right[key][0] if key == "optimizer" else right[key]
        tensor[0] += 1
        with pytest.raises(ValueError, match="tensor改变"):
            assert_same_tree(left, right)
