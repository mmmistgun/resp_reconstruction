from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.crd.config import CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION, FORMAL_SEEDS, load_crd_config
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.tf_v1_research_test import RESEARCH_TEST_VARIANTS, expected_research_test_checkpoints
from resp_train.crd.tf_v1_research_test_data import TfV1ResearchTestCacheReader


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_array(root: Path, name: str, values: np.ndarray) -> dict[str, object]:
    path = root / name
    np.save(path, values, allow_pickle=False)
    return {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "finite": True,
    }


def test_research_test_reader_verifies_manifest_files_and_row_lookup(monkeypatch, tmp_path) -> None:
    row_ids = np.asarray([7, 9], dtype=np.int64)
    files = {
        "test_row_ids.npy": _save_array(tmp_path, "test_row_ids.npy", row_ids),
        "test_m_slow.npy": _save_array(tmp_path, "test_m_slow.npy", np.zeros((2, 36, 101), np.float32)),
        "test_m_fast.npy": _save_array(tmp_path, "test_m_fast.npy", np.zeros((2, 44, 349), np.float32)),
    }
    manifest = {
        "protocol": CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION,
        "cache_identity_sha256": "identity",
        "complete": True,
        "representations": ["m", "w", "s"],
        "research_test_used": True,
        "research_test_input_used": True,
        "test_target_array_read": False,
        "target_read": False,
        "test_cache_created": True,
        "model_inference_used": False,
        "git": {"commit": "cache-commit", "dirty": False},
        "splits": {
            "test": {
                "split": "test",
                "count": 2,
                "row_ids_sha256": hashlib.sha256(row_ids.tobytes(order="C")).hexdigest(),
            }
        },
        "files": files,
    }
    manifest_path = tmp_path / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("resp_train.crd.tf_v1_research_test_data.FROZEN_RESEARCH_TEST_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(
        "resp_train.crd.tf_v1_research_test_data.FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256",
        _sha256(manifest_path),
    )
    monkeypatch.setattr("resp_train.crd.tf_v1_research_test_data.FROZEN_RESEARCH_TEST_CACHE_IDENTITY", "identity")
    monkeypatch.setattr("resp_train.crd.tf_v1_research_test_data.FROZEN_RESEARCH_TEST_CACHE_COMMIT", "cache-commit")
    monkeypatch.setattr("resp_train.crd.tf_v1_research_test_data.EXPECTED_TEST_COUNT", 2)

    reader = TfV1ResearchTestCacheReader(tmp_path, split="test", representations=("m",))

    assert reader.position(9) == 1
    assert set(reader.get(7)) == {"m_slow", "m_fast"}
    reader.verify_rows([7, 9])


def test_expected_research_test_matrix_is_exactly_c201_m_w_ms_three_seeds(
    monkeypatch, tmp_path
) -> None:
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "seeds": [
                    {
                        "seed": seed,
                        "run_dir": f"c201/{seed}",
                        "selected_epoch": 10,
                        "files": {"checkpoint_best_local_rr.pt": {"sha256": f"c201-{seed}"}},
                    }
                    for seed in FORMAL_SEEDS
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "variant": variant,
            "seed": seed,
            "run_dir": str(tmp_path / variant / str(seed)),
            "selected_epoch": 11,
            "checkpoint_sha256": f"{variant}-{seed}",
        }
        for variant in RESEARCH_TEST_VARIANTS[1:]
        for seed in FORMAL_SEEDS
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "formal_run_audit.csv", index=False)
    monkeypatch.setattr("resp_train.crd.tf_v1_research_test._audit_selection_inputs", lambda: None)
    monkeypatch.setattr("resp_train.crd.tf_v1_research_test.CANDIDATE_LOCK", lock_path)
    monkeypatch.setattr("resp_train.crd.tf_v1_research_test.P5_ROOT", tmp_path)
    monkeypatch.setattr("resp_train.crd.tf_v1_research_test.REPO_ROOT", tmp_path)

    matrix = expected_research_test_checkpoints()

    assert len(matrix) == 12
    assert {(row["variant"], row["seed"]) for row in matrix.values()} == {
        (variant, seed) for variant in RESEARCH_TEST_VARIANTS for seed in FORMAL_SEEDS
    }


class _ScaledIdentity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))


def test_tf_checkpoint_test_evaluation_injects_cache_only_after_checkpoint_identity(
    monkeypatch, tmp_path
) -> None:
    cfg = load_crd_config("configs/crd_tf_v1/crd_tf203_ms_formal.yaml")
    # strict formal config 已先通过校验；测试只隔离 lifecycle，不运行真实 CUDA 模型。
    cfg.training.device = "cpu"
    cfg.training.show_progress = False
    model = _ScaledIdentity()
    checkpoint_path = tmp_path / "checkpoint_best_local_rr.pt"
    torch.save(
        {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    observed = {}

    def fake_build_window_data(data_cfg, **kwargs):
        observed["cache"] = str(data_cfg.data.tf_research_test_cache_path)
        observed["kwargs"] = kwargs
        return SimpleNamespace(loader="test-loader")

    monkeypatch.setattr("resp_train.crd.experiment.build_crd_model", lambda _cfg: _ScaledIdentity())
    monkeypatch.setattr("resp_train.crd.experiment.build_window_data", fake_build_window_data)
    experiment = CRDExperiment(cfg)
    monkeypatch.setattr(
        experiment,
        "_evaluate_model",
        lambda *_args, **_kwargs: pd.DataFrame([{"evaluation_split": "test"}]),
    )
    cache_path = tmp_path / "frozen-cache"

    result = experiment.evaluate_checkpoint(
        checkpoint_path,
        split="test",
        tf_research_test_cache_path=cache_path,
    )

    assert result.loc[0, "evaluation_split"] == "test"
    assert observed["cache"] == str(cache_path.resolve())
    assert observed["kwargs"]["split"] == "test"
    assert "tf_research_test_cache_path" not in cfg.data
