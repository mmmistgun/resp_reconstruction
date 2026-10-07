#!/usr/bin/env python3
"""固定 source-equivalent checkpoint：nested sampler 预算曲线与选定 N 完整 validation。"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.respdiff_bcg.baseband import config_contract, make_baseband_model
from resp_train.respdiff_bcg.inference_v2 import (
    NAME, ParentSelection, baseband_dataset, choose_ensemble, evaluate_setting, load_frozen_source,
    load_sampler_config, predict_prefixes, sampler_spec, select_diagnostic_indices, state_digest, validation_data,
)
from resp_train.respdiff_bcg.inference_plots import plot_tradeoffs
from resp_train.respdiff_bcg.runtime import (
    LOGGER, artifact_manifest, environment, run_logging, save_source_snapshot, seed_all, sha256,
    source_file_inventory, source_identity, write_json,
)
from scripts.run_respdiff_bcg_baseband_v1 import fixture

CONFIG_DIR = ROOT / "configs/respdiff_bcg_epsilon_inference_v2"
DEFAULT_SOURCE = ROOT / "runs/respdiff_bcg_baseband_v1_from_t630_20261007/source_equivalent_seed20260811_v1"


def implementation_identity(cfg):
    identity = source_identity(cfg)
    for path in (Path(__file__).resolve(), ROOT / "scripts/run_respdiff_bcg_baseband_v1.py",
        ROOT / "configs/respdiff_bcg_v1/gpu_b64_b64.yaml",
        ROOT / "docs/experiments/respdiff_bcg_epsilon_inference_v2_protocol_20261007.md",
        CONFIG_DIR / "ddim.yaml", CONFIG_DIR / "ddpm.yaml"):
        identity["files"][str(path.relative_to(ROOT))] = sha256(path)
    return identity


def read_subset_selection(path, checkpoint_identity, configurations):
    receipt = json.loads((path / "receipt.json").read_text())
    if receipt.get("status") != "complete" or receipt.get("mode") != "subset":
        raise ValueError("full-validation 要求完成的真实 diagnostic subset")
    for name in ("selection.json", "ensemble_curve.csv", "inference_identity.json"):
        if sha256(path / name) != receipt["artifacts"][name]:
            raise ValueError(f"subset 选择产物身份变化：{name}")
    identity = json.loads((path / "inference_identity.json").read_text())
    if (identity["checkpoint"]["checkpoint_sha256"] != checkpoint_identity["checkpoint_sha256"]
            or identity["configurations"] != configurations):
        raise ValueError("full-validation 与 subset 的 checkpoint/sampler 合同不同")
    selection = json.loads((path / "selection.json").read_text())
    recomputed = choose_ensemble(pd.read_csv(path / "ensemble_curve.csv"))
    # CSV round-trip 可产生最后一位差异；比较选择身份和每对 plateau 决定。
    if (selection["selected_N"] != recomputed["selected_N"] or selection["status"] != recomputed["status"]
            or [v["plateau"] for v in selection["comparisons"]] != [v["plateau"] for v in recomputed["comparisons"]]):
        raise ValueError("subset selection 与固定规则不符")
    return selection


def run(mode, output, *, source_run=DEFAULT_SOURCE, subset_run=None, device="cuda:0"):
    if mode not in {"synthetic-smoke", "subset", "full-validation"}:
        raise ValueError("未知 mode")
    if mode == "full-validation" and subset_run is None:
        raise ValueError("full-validation 必须指定 --subset-run")
    tiny = mode == "synthetic-smoke"
    if not tiny and (not str(device).startswith("cuda:") or not str(device)[5:].isdigit()):
        raise ValueError("正式推理须显式 cuda:N")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with run_logging(output):
        stage = "initialization"
        started = time.perf_counter()
        try:
            configs = {kind: load_sampler_config(CONFIG_DIR / f"{kind}.yaml") for kind in ("ddim", "ddpm")}
            seed_all(20260811)
            if tiny:
                torch.set_num_threads(1)
                cfg = config_contract("source_equivalent")
                model = make_baseband_model(cfg, tiny=True).eval()
                dataset = fixture(output, cfg, "val")
                checkpoint_identity = {"synthetic": True, "checkpoint_sha256": None}
                index_path = output / "val_fixture_index.csv"
                indices = list(range(len(dataset.rows)))
                grids = {"ddim": (1, 2, 4), "ddpm": (1, 2)}
                batch_size = 8
            else:
                stage = "source_checkpoint"
                model, cfg, checkpoint_identity = load_frozen_source(source_run, torch.device(device))
                stage = "validation_identity"
                parents, rows, index_path = validation_data(cfg, source_run)
                indices = (select_diagnostic_indices(rows) if mode == "subset" else list(range(len(rows))))
                selected = ParentSelection(parents, rows, indices)
                dataset = baseband_dataset(selected, selected.rows)
                grids = {kind: tuple(value["ensemble"]["counts"]) for kind, value in configs.items()}
                batch_size = 64
            parameter_identity = state_digest(model)
            selection = None
            if mode == "full-validation":
                selection = read_subset_selection(Path(subset_run).resolve(), checkpoint_identity, configs)
                grids = {"ddim": (selection["selected_N"],)}
                write_json(output / "selection.json", {**selection, "subset_run": str(Path(subset_run).resolve())})
            inventory = source_file_inventory(dataset.rows)
            index_hash = sha256(index_path)
            implementation = implementation_identity(cfg)
            write_json(output / "source_identity.json", implementation)
            save_source_snapshot(output, implementation)
            write_json(output / "environment.json", environment())
            OmegaConf.save(cfg, output / "source_resolved_config.yaml")
            write_json(output / "inference_identity.json", {"protocol": NAME, "mode": mode,
                "checkpoint": checkpoint_identity, "configurations": configs, "execution_counts": grids,
                "model_spec": vars(model.spec), "device": str(next(model.parameters()).device),
                "batch_size": batch_size, "parent_indices": indices,
                "source_file_stats": inventory, "index_path": str(index_path), "index_sha256": index_hash,
                "parameter_sha256_before": parameter_identity})
            dataset.rows.to_csv(output / "validation_parents.csv", index=False)
            dataset.manifest().to_csv(output / "validation_chunks.csv", index=False)
            LOGGER.info("阶段=%s；固定 parents=%d chunks=%d；checkpoint=%s", mode,
                        len(dataset.rows), len(dataset), checkpoint_identity["checkpoint_sha256"])
            curves, profiles = [], []
            for kind, counts in grids.items():
                stage = f"{kind}_sampling"
                spec = sampler_spec(configs[kind], count=max(counts))
                sampling_output = output / f"{kind}_sampling"
                sampling_output.mkdir()
                predictions, profile = predict_prefixes(model, dataset, spec, counts, sampling_output,
                                                         batch_size=batch_size)
                profiles.extend(profile)
                for count in counts:
                    stage = f"{kind}_N{count}_metrics"
                    curve = evaluate_setting(predictions.pop(count), dataset, cfg,
                                             output / f"{kind}_N{count}", spec, count)
                    curve["sampling_seconds"] = next(p["sampling_seconds"] for p in profile if p["N"] == count)
                    curves.append(curve)
                    pd.DataFrame(curves).to_csv(output / "ensemble_curve.csv", index=False)
                    write_json(output / "ensemble_curve.json", curves)
                    pd.DataFrame(profiles).to_csv(output / "runtime_profile.csv", index=False)
            if mode != "full-validation":
                selection = choose_ensemble(pd.DataFrame(curves), counts=grids["ddim"])
                write_json(output / "selection.json", selection)
                LOGGER.info("固定选择：N=%d，%s", selection["selected_N"], selection["status"])
            stage = "plots"
            plot_tradeoffs(output)
            if mode == "full-validation":
                # root 提供正式主结果的固定文件入口；原文件保留在 sampler_N 目录。
                setting = output / f"ddim_N{selection['selected_N']}"
                import shutil
                for name in ("validation_raw_waveforms.npz", "validation_postfiltered_waveforms.npz",
                    "validation_raw_diagnostics.csv", "validation_postfiltered_diagnostics.csv",
                    "validation_summary.csv", "row_10382_waveforms.npz", "row_12226_waveforms.npz"):
                    if not (output / name).exists():
                        shutil.copyfile(setting / name, output / name)
            stage = "source_immutability"
            if state_digest(model) != parameter_identity:
                raise ValueError("推理改变了 checkpoint 参数或 buffer")
            if not tiny and sha256(Path(source_run) / "final.pt") != checkpoint_identity["checkpoint_sha256"]:
                raise ValueError("推理期间 checkpoint 文件字节变化")
            if inventory != source_file_inventory(dataset.rows) or index_hash != sha256(index_path):
                raise ValueError("推理期间数据身份变化")
            if implementation_identity(cfg) != implementation:
                raise ValueError("推理期间代码或配置变化")
            # nested 设置目录单独列哈希；root artifact_manifest 不遍历目录。
            nested_artifacts = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                                if p.is_file() and p.parent != output}
            LOGGER.info("推理完成：mode=%s N=%d wall=%.1fs", mode, selection["selected_N"], time.perf_counter() - started)
            write_json(output / "receipt.json", {"status": "complete", "mode": mode,
                "formal_evidence": not tiny, "selected_N": selection["selected_N"],
                "parameter_sha256_after": parameter_identity, "checkpoint_bytes_unchanged": not tiny,
                "total_wall_seconds": time.perf_counter() - started,
                "artifacts": {**artifact_manifest(output), **nested_artifacts}})
        except BaseException as error:
            LOGGER.error("推理终止：stage=%s %s: %s", stage, type(error).__name__, error)
            LOGGER.debug("异常堆栈", exc_info=True)
            write_json(output / "failure.json", {"stage": stage, "status": "failed",
                       "error_type": type(error).__name__, "error": str(error)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("synthetic-smoke", "subset", "full-validation"))
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--subset-run", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.output, source_run=args.source_run, subset_run=args.subset_run, device=args.device)
