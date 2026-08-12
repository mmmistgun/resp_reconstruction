from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from resp_train.crd.config import load_crd_config
from resp_train.crd.tf_v1_model import TF_VARIANTS
from resp_train.crd.tf_v1_p3 import (
    P3_ACCEPTANCE_VARIANTS,
    acceptance_overrides,
    audit_p3_acceptance,
)


def test_p3_acceptance_arms_are_frozen_maximum_single_pair_triple() -> None:
    assert P3_ACCEPTANCE_VARIANTS == ("crd_tf102_w", "crd_tf204_wl", "crd_tf302_wls")


@pytest.mark.parametrize(
    ("variant", "representations"),
    (
        ("crd_tf102_w", ["w"]),
        ("crd_tf204_wl", ["w", "l"]),
        ("crd_tf302_wls", ["w", "l", "s"]),
    ),
)
def test_p3_acceptance_overrides_resolve_strict_config(
    tmp_path: Path, variant: str, representations: list[str]
) -> None:
    cfg = load_crd_config(
        "configs/crd_tf_v1/crd_tf101_m_smoke.yaml",
        overrides=acceptance_overrides(variant, device="cuda:0", output_root=tmp_path),
    )

    assert str(cfg.protocol.execution_gate) == "p3_cuda_acceptance"
    assert str(cfg.protocol.run_role) == "acceptance"
    assert str(cfg.model.variant) == variant
    assert list(cfg.model.tf_representations) == representations
    assert (cfg.training.epochs, cfg.training.batch_size, cfg.training.gradient_accumulation_steps) == (1, 128, 1)
    assert (cfg.data.max_train_windows, cfg.data.max_val_windows, cfg.data.max_test_windows) == (128, 32, None)


def test_p3_acceptance_overrides_reject_non_maximum_arm(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="最大 single/pair/triple"):
        acceptance_overrides("crd_tf101_m", device="cuda:0", output_root=tmp_path)


def test_p3_audit_rejects_incomplete_synthetic_before_reading_runs(tmp_path: Path) -> None:
    receipt = tmp_path / "synthetic.json"
    receipt.write_text(
        json.dumps(
            {
                "protocol": "crd-tf-v1-research-informed-20260812",
                "status": "failed",
                "complete": False,
                "git_dirty": False,
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="synthetic receipt"):
        audit_p3_acceptance(synthetic_receipt=receipt, run_dirs={}, output_root=tmp_path / "out")


def test_p3_audit_accepts_complete_same_commit_engineering_receipts(tmp_path: Path) -> None:
    commit = "a" * 40
    receipt = tmp_path / "synthetic.json"
    receipt.write_text(
        json.dumps(
            {
                "protocol": "crd-tf-v1-research-informed-20260812",
                "status": "passed",
                "complete": True,
                "git_commit": commit,
                "git_dirty": False,
                "results": [{"variant": variant, "status": "passed"} for variant in TF_VARIANTS],
            }
        ),
        encoding="utf-8",
    )
    run_dirs = {}
    for variant in P3_ACCEPTANCE_VARIANTS:
        run_dir = tmp_path / variant
        run_dir.mkdir()
        cfg = load_crd_config(
            "configs/crd_tf_v1/crd_tf101_m_smoke.yaml",
            overrides=acceptance_overrides(variant, device="cuda:0", output_root=tmp_path / "unused"),
        )
        OmegaConf.save(cfg, run_dir / "config.yaml")
        (run_dir / "audit.csv").write_text("split,n_windows\nval,32\n", encoding="utf-8")
        (run_dir / "optimizer_parameter_groups.json").write_text("{}\n", encoding="utf-8")
        (run_dir / "metrics.csv").write_text("value\n1\n", encoding="utf-8")
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "git_commit": commit,
                    "git_dirty": False,
                    "protocol": "crd-tf-v1-research-informed-20260812",
                    "stage": "tf",
                    "run_role": "acceptance",
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "train_history.csv").write_text(
            "epoch,optimizer_update,train_loss_total,train_elapsed_seconds,train_samples_per_second\n"
            "1,1,0.5,2.0,64.0\n",
            encoding="utf-8",
        )
        (run_dir / "metrics_summary.csv").write_text(
            "n_samples,whole_rr_abs_error_bpm_mean,local_rr_mae_bpm_mean,"
            "envelope_trajectory_mae_mean,global_envelope_modulation_error_mean,"
            "lag_aware_signed_pcc_mean,joint_prediction_degenerate_fraction\n"
            "32,1.0,1.0,1.0,1.0,0.5,0.0\n",
            encoding="utf-8",
        )
        (run_dir / "runtime_summary.json").write_text(
            json.dumps(
                {
                    "peak_allocated_mib": 1000.0,
                    "peak_reserved_mib": 1200.0,
                    "peak_reserved_fraction": 0.5,
                }
            ),
            encoding="utf-8",
        )
        checkpoint = {
            "epoch": 1,
            "model_state_dict": {"weight": torch.ones(1)},
            "optimizer_state_dict": {"state": {0: {"step": torch.ones(1)}}},
            "extra_state": {"update_index": 1, "total_updates": 1},
        }
        torch.save(checkpoint, run_dir / "checkpoint_best_local_rr.pt")
        torch.save(checkpoint, run_dir / "checkpoint_final.pt")
        run_dirs[variant] = run_dir

    output = audit_p3_acceptance(
        synthetic_receipt=receipt,
        run_dirs=run_dirs,
        output_root=tmp_path / "audit_output",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["batch_decision"] == "128x1"
    assert payload["maximum_peak_reserved_fraction"] == 0.5
    assert payload["minimum_train_samples_per_second"] == 64.0
