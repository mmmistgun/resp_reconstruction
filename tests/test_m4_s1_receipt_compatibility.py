"""只用临时回执验证JSON绑定兼容层，保留字段漂移和文件篡改拒绝。"""
from pathlib import Path

import pytest

from scripts import m4_s1_cwt_parameters_artifacts as io
from scripts import m4_s1_cwt_parameters_spec as spec
from scripts.run_m4_s1_receipt_compatibility import canonical_stage_bindings


@pytest.fixture
def receipt(tmp_path):
    binding = {"architecture": {"stem_dilations": (1, 2, 4, 8, 16)}, "batch_size": 32}
    payload = tmp_path / "payload.txt"
    payload.write_text("synthetic fixture", encoding="utf-8")
    io.write_json(tmp_path / "manifest.json", {
        "protocol": spec.PROTOCOL, "status": "completed", "binding": binding,
        "files": {"payload.txt": io.identity(payload)},
    })
    io.write_json(tmp_path / "receipt.json", io.identity(tmp_path / "manifest.json"))
    return tmp_path, binding


def test_json_roundtrip_and_restore(receipt):
    path, binding = receipt
    original = io.verify_stage
    before = {p.name: io.identity(p) for p in path.iterdir()}
    with pytest.raises(ValueError, match="阶段绑定不符"):
        original(path, binding)
    with canonical_stage_bindings():
        assert io.verify_stage(path, binding)["binding"]["batch_size"] == 32
        assert isinstance(binding["architecture"]["stem_dilations"], tuple)
    assert io.verify_stage is original
    assert {p.name: io.identity(p) for p in path.iterdir()} == before


def test_semantic_mismatch_rejected(receipt):
    path, binding = receipt
    with canonical_stage_bindings():
        with pytest.raises(ValueError, match="阶段绑定不符"):
            io.verify_stage(path, {**binding, "batch_size": 16})
        with pytest.raises(ValueError, match="阶段绑定不符"):
            io.verify_stage(path, {**binding, "architecture": {"stem_dilations": (1, 2, 4)}})


def test_artifact_corruption_rejected(receipt):
    path, binding = receipt
    (path / "payload.txt").write_text("changed disposable fixture", encoding="utf-8")
    original = io.verify_stage
    with pytest.raises(ValueError, match="文件身份变化"):
        with canonical_stage_bindings():
            io.verify_stage(path, binding)
    assert io.verify_stage is original
