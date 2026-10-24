"""计划阶段的运行组织；真实运行需要 CLI 显式开放。"""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from resp_train.respdiff.data import assert_split_independence
from .baseband import config_contract, make_baseband_model
from .inference_v2 import (CHECKPOINT_SHA256, evaluate_setting, predict_prefixes, select_diagnostic_indices, state_digest)
from .runtime import LOGGER, seed_all, sha256, write_json
from .sampling_accelerated import AccelerationSpec
from .signal import restore_parent
from .study_contract import (NAME, PROBE_STEPS, PROBE_SEEDS, PAIRING_SEEDS, PRESSURE_ROWS,
                             GRID, StudySamplerSpec, TrainingSpec, choose_configurations, choose_training, grid_specs)
from .study_diagnostics import evaluate_anchors, run_forward_probes
from .study_evaluation import independent_timing, plot_grid, rule_sample_summary
from .study_io import (ROOT, data_identity, implementation_identity, load_source, save_provenance,
                       selected_dataset, source_data, stratified_indices, study_run, training_config,
                       validate_training_config, verify_artifact, verify_data_unchanged)
from .study_state import preserved_evaluation
from .study_training import train_with_snapshots

DEFAULT_SOURCE = ROOT / "runs/respdiff_bcg_baseband_v1_from_t630_20261007/source_equivalent_seed20260811_v1"
SETTINGS = ROOT / "configs/respdiff_bcg_study_v1/plan.yaml"


def expected_settings():
    return {"protocol": NAME,
            "diagnostics": {"parents_per_split": 32, "selection_seed": 20261010,
                "timesteps": list(PROBE_STEPS), "forward_noise_seeds": list(PROBE_SEEDS),
                "pairing_seeds": list(PAIRING_SEEDS), "pressure_rows": list(PRESSURE_ROWS)},
            "sampling_grid": [{"sampler": kind, "S": steps, "N": list(counts)}
                              for (kind, steps), counts in GRID.items()],
            "batch_size": 64, "sampling_noise_seed": 20261003, "trajectory_group_size": 1}


def load_settings():
    actual = OmegaConf.to_container(OmegaConf.load(SETTINGS), resolve=True)
    if actual != expected_settings():
        raise ValueError("plan.yaml 偏离已制定合同")
    return actual


def _fixture(output, cfg, split):
    from scripts.run_respdiff_bcg_baseband_v1 import fixture
    return fixture(output, cfg, split)


def _initialize(output, phase, source_run, device, *, tiny, training=False, expected_sha256=None):
    load_settings()
    if tiny:
        cfg = config_contract("source_equivalent")
        seed_all(20260811, cpu_only=True)
        model = make_baseband_model(cfg, tiny=True).eval()
        datasets = {split: _fixture(output, cfg, split) for split in ("train", "val")}
        index = output / "train_fixture_index.csv"
        checkpoint = {"synthetic": True, "checkpoint_sha256": None, "training_seed": 20260811,
                      "regime": "baseline", "update": 0}
        bundles = {split: (dataset.parents.raw, dataset.rows, index) for split, dataset in datasets.items()}
    else:
        seed_all(20260811)
        destination = torch.device("cpu" if training else device)
        model, cfg, checkpoint = load_source(source_run, destination, expected_sha256=expected_sha256)
        bundles = source_data(cfg, source_run, mapping_output=output / "source_path_mapping.json")
        index = bundles["train"][2]
        datasets = {split: selected_dataset(parents, rows, list(range(len(rows))))
                    for split, (parents, rows, _) in bundles.items()}
    assert_split_independence(datasets["train"].rows, datasets["val"].rows)
    return model, cfg, checkpoint, bundles, datasets, index


def diagnostic_sets(bundles, *, tiny=False):
    sets, selections = {}, {}
    for split in ("train", "val"):
        parents, rows, _ = bundles[split]
        count = min(2, len(rows)) if tiny else 32
        indices = stratified_indices(rows, count)
        key = f"{split}_representative"
        sets[key] = selected_dataset(parents, rows, indices)
        selections[key] = {"source_parent_indices": indices,
                           "dataset_row_ids": sets[key].rows.dataset_row_id.astype(int).tolist(),
                           "n_subjects": int(sets[key].rows.samp_id.nunique()),
                           "n_source_records": int(sets[key].rows.target_source_npz.nunique())}
    parents, rows, _ = bundles["val"]
    present = {row: rows.index[rows.dataset_row_id == row].tolist() for row in PRESSURE_ROWS}
    represented = set(selections["val_representative"]["dataset_row_ids"])
    extra = sorted(index for row, indices in present.items() if row not in represented for index in indices)
    if extra:
        sets["val_pressure"] = selected_dataset(parents, rows, extra)
        selections["val_pressure"] = {"source_parent_indices": extra,
                                      "dataset_row_ids": sets["val_pressure"].rows.dataset_row_id.astype(int).tolist()}
    selections["pressure_cases"] = {str(row): {"present": bool(indices), "in_representative": row in represented}
                                    for row, indices in present.items()}
    return sets, selections


def _check_source(checkpoint, source_run, model=None):
    if not checkpoint.get("synthetic") and sha256(Path(source_run) / "final.pt") != checkpoint["checkpoint_sha256"]:
        raise ValueError("运行期间来源 checkpoint 字节变化")
    if model is not None and "parameter_sha256" in checkpoint and state_digest(model) != checkpoint["parameter_sha256"]:
        raise ValueError("运行改变了来源模型参数或 buffer")


def _plot_cases(path, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = sorted(set([0, *[i for i, row in enumerate(dataset.rows.dataset_row_id) if row in PRESSURE_ROWS]]))
    settings = sorted(path.glob("*_correct/validation_postfiltered_waveforms.npz"))
    for index in cases:
        figure, axes = plt.subplots(1+len(settings), 1, figsize=(12, 3*(1+len(settings))), squeeze=False)
        # 用实际送入模型的 chunks 重建输入，明确展示 LPF/AA 后的 20 Hz 条件。
        input_chunks = np.stack([dataset[index*13+chunk]["x"].numpy() for chunk in range(13)])
        bcg, _ = restore_parent(input_chunks,chunk_indices=range(13))
        axes[0,0].plot(np.arange(len(bcg))/20,bcg,linewidth=.8,color="tab:blue")
        axes[0,0].set_title("BCG input: parent 1 Hz LPF + anti-alias 100 to 20 Hz; 30 s chunks / 15 s hop")
        axes[0,0].set_xlabel("Time (s)")
        axes[0,0].set_ylabel("BCG soft-z")
        for ax, file in zip(axes[1:,0], settings):
            with np.load(file) as data:
                ax.plot(np.arange(18000)/100, data["tho_ref"][index], label="THO", linewidth=.9)
                ax.plot(np.arange(18000)/100, data["prediction_postfiltered"][index], label="prediction", linewidth=.7)
            ax.set_title(file.parent.name)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("THO soft-z")
            ax.legend()
        figure.tight_layout()
        figure.savefig(path / f"row_{int(dataset.rows.iloc[index].dataset_row_id)}.png", dpi=120)
        plt.close(figure)


def run_diagnosis(output, *, source_run=DEFAULT_SOURCE, device="cuda:0", tiny=False, expected_sha256=None):
    with study_run(output, "diagnose", synthetic=tiny, source=None if tiny else source_run) as (out, receipt):
        model, cfg, checkpoint, bundles, _, index = _initialize(
            out, "diagnose", source_run, device, tiny=tiny, expected_sha256=expected_sha256)
        sets, selections = diagnostic_sets(bundles, tiny=tiny)
        batch_size = 8 if tiny else 64
        pairing = PAIRING_SEEDS[:1] if tiny else PAIRING_SEEDS
        steps, seeds = ((0, 49), PROBE_SEEDS[:1]) if tiny else (PROBE_STEPS, PROBE_SEEDS)
        identity = save_provenance(out, cfg, cpu_only=tiny,
            execution={"phase": "diagnose", "synthetic": tiny, "checkpoint": checkpoint,
                       "device": str(next(model.parameters()).device), "batch_size": batch_size})
        data = data_identity(out, sets, index)
        rows = pd.concat([s.rows for s in sets.values()], ignore_index=True)
        batches = sum((len(s)+batch_size-1)//batch_size for s in sets.values())
        n_ddim, n_ddpm = (1, 1) if tiny else (16, 4)
        budget = {"n_parents": len(rows), "n_chunks": sum(len(s) for s in sets.values()),
                  "n_batches": batches, "teacher_denoiser_calls": batches*len(steps)*len(seeds)*(1+len(pairing)),
                  "sampling_denoiser_calls": batches*(6*n_ddim+50*n_ddpm)*(1+len(pairing)),
                  "trajectory_group_size": 1}
        write_json(out / "budget.json", budget)
        write_json(out / "diagnostic_identity.json", {"protocol": NAME, "checkpoint": checkpoint,
            "selections": selections, "batch_size": batch_size, "timesteps": list(steps),
            "forward_noise_seeds": list(seeds), "pairing_seeds": list(pairing),
            "strata": {"subject": "samp_id", "source_record": "target_source_npz"},
            "noise_identity": "sha256(protocol,forward,seed,split,row_id,chunk_index)",
            "initial_sampling_noise_seed": 20261003})
        LOGGER.info("P1 固定预算：%s", budget)
        anchors = []
        for cohort, dataset in sets.items():
            run_forward_probes(model, dataset, out / cohort / "forward", batch_size=batch_size,
                               steps=steps, noise_seeds=seeds, pairing_seeds=pairing)
            path = out / cohort / "anchors"
            curves = evaluate_anchors(model, dataset, cfg, path, batch_size=batch_size,
                                      pairing_seeds=pairing, tiny=tiny, trace=True)
            anchors.extend({"cohort": cohort, **row} for row in curves)
            _plot_cases(path, dataset)
        pd.DataFrame(anchors).to_csv(out / "anchor_metrics.csv", index=False)
        verify_data_unchanged(data, rows)
        _check_source(checkpoint, source_run, model)
        if implementation_identity(cfg) != identity:
            raise ValueError("诊断期间源码或配置变化")
        receipt.update({"checkpoint": checkpoint, "parameter_sha256_after": state_digest(model),
                        "diagnostic_complete": True, "scientific_interpretation": "pending_review"})


def read_stage(path, phase):
    path = Path(path).resolve()
    receipt = json.loads((path / "receipt.json").read_text())
    if (receipt.get("status") != "complete" or receipt.get("phase") != phase
            or receipt.get("protocol") != NAME or not receipt.get("formal_evidence")):
        raise ValueError(f"要求已完成的真实 {phase} 运行")
    return path, receipt


def training_datasets(bundles, diagnostic_run, *, overfit, tiny=False):
    if tiny:
        sets, _ = diagnostic_sets(bundles, tiny=True)
        train = sets["train_representative"]
        return train, sets
    path, receipt = read_stage(diagnostic_run, "diagnose")
    verify_artifact(path, receipt, "diagnostic_identity.json")
    selections = json.loads((path / "diagnostic_identity.json").read_text())["selections"]
    probes = {}
    for split in ("train", "val"):
        key = f"{split}_representative"
        parents, rows, _ = bundles[split]
        indices = selections[key]["source_parent_indices"]
        if rows.iloc[indices].dataset_row_id.astype(int).tolist() != selections[key]["dataset_row_ids"]:
            raise ValueError("P1 parent 选择与当前数据不一致")
        probes[key] = selected_dataset(parents, rows, indices)
    if overfit:
        pool = probes["train_representative"]
        indices = stratified_indices(pool.rows, 8)
        # 这里的 parents 已经低通；取原始父对象以避免二次 LPF。
        original = pool.parents.raw
        train = selected_dataset(original, pool.rows, indices)
        probes["train_representative"] = train
    else:
        parents, rows, _ = bundles["train"]
        train = selected_dataset(parents, rows, list(range(len(rows))))
    return train, probes


def run_training(output, *, regime, seed=20260811, source_run=DEFAULT_SOURCE, diagnostic_run=None,
                 overfit_run=None, device="cuda:0", resume=None, tiny=False):
    spec = TrainingSpec(regime, seed)
    phase = "overfit" if regime == "overfit" else "train"
    if not tiny and diagnostic_run is None:
        raise ValueError("训练必须指定已完成的 --diagnostic-run")
    if not tiny and phase == "train":
        if overfit_run is None:
            raise ValueError("长训练前须提供已完成的 --overfit-run，并人工审核 P1/P2 结论")
        _, overfit_receipt = read_stage(overfit_run, "overfit")
    with study_run(output, phase, synthetic=tiny, source=None if tiny else source_run) as (out, receipt):
        old_model, base, checkpoint, bundles, full, index = _initialize(out, phase, source_run, device,
                                                                      tiny=tiny, training=True)
        del old_model
        diagnostic_digest = None
        if not tiny:
            diagnostic_path, diagnostic_receipt = read_stage(diagnostic_run, "diagnose")
            verify_artifact(diagnostic_path, diagnostic_receipt, "diagnostic_identity.json")
            diagnostic_digest = sha256(diagnostic_path / "diagnostic_identity.json")
            diagnostic_identity = json.loads((diagnostic_path / "diagnostic_identity.json").read_text())
            if diagnostic_identity["checkpoint"]["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]:
                raise ValueError("训练数据来源与 P1 checkpoint 来源不一致")
            if phase == "train" and overfit_receipt.get("diagnostic_identity_sha256") != diagnostic_digest:
                raise ValueError("P2 与当前 P1 的诊断身份不一致")
        train, probes = training_datasets(bundles, diagnostic_run, overfit=phase == "overfit", tiny=tiny)
        cfg = training_config(base, spec)
        if not tiny:
            cfg.training.device = device
        seed_all(seed, cpu_only=tiny)
        model = make_baseband_model(cfg, tiny=tiny).to(torch.device("cpu" if tiny else device))
        identity = save_provenance(out, cfg, cpu_only=tiny,
            execution={"phase": phase, "training": spec.identity(), "synthetic": tiny,
                       "source": checkpoint, "device": str(next(model.parameters()).device),
                       "diagnostic_run": None if diagnostic_run is None else str(Path(diagnostic_run).resolve()),
                       "overfit_run": None if overfit_run is None else str(Path(overfit_run).resolve())})
        # 来源的完整父身份与实际训练子集分开保存，便于新模型后续做相同 validation。
        data = data_identity(out, full, index)
        train.rows.to_csv(out / "actual_training_parents.csv", index=False)
        train.manifest().to_csv(out / "actual_training_chunks.csv", index=False)
        for key, dataset in probes.items():
            dataset.rows.to_csv(out / f"{key}_parents.csv", index=False)
        callback_updates = {0, 2} if tiny else set(spec.snapshots)
        validation_updates = {2} if tiny else set(spec.validation_updates)

        def monitor(update):
            if update not in callback_updates:
                return
            with preserved_evaluation(model):
                for key, dataset in probes.items():
                    path = out / "monitor" / f"update_{update:06d}" / key
                    run_forward_probes(model, dataset, path / "forward", batch_size=8 if tiny else 64,
                        steps=(49,) if tiny else PROBE_STEPS,
                        noise_seeds=PROBE_SEEDS[:1] if tiny else PROBE_SEEDS, pairing_seeds=())
                    if update in validation_updates and (key == "val_representative" or phase == "overfit"):
                        # 预算对照的过程曲线固定 DDIM6；P2 同时检查两个 anchor。
                        if phase == "overfit":
                            evaluate_anchors(model, dataset, cfg, path / "anchors", batch_size=8 if tiny else 64,
                                             tiny=tiny)
                        else:
                            sampler = StudySamplerSpec("ddim", 6, 1 if tiny else 16)
                            sample_path = path / "ddim_sampling"
                            sample_path.mkdir()
                            predictions, _ = predict_prefixes(model, dataset, sampler, (sampler.n_trajectories,),
                                                               sample_path, batch_size=8 if tiny else 64)
                            evaluate_setting(predictions[sampler.n_trajectories], dataset, cfg,
                                path / "ddim_metrics", sampler, sampler.n_trajectories,
                                method_prefix="RespDiff-study-v1/training-monitor")

        outcome = train_with_snapshots(model, train, cfg, spec, out, callback=monitor, resume=resume,
                                       fixture_updates=2 if tiny else None, fixture_batch_size=2 if tiny else None)
        if not tiny and phase == "train":
            parents, rows, _ = bundles["val"]
            selected = selected_dataset(parents, rows, select_diagnostic_indices(rows))
            selected.rows.to_csv(out / "final_anchor_parents.csv", index=False)
            with preserved_evaluation(model):
                evaluate_anchors(model, selected, cfg, out / "final_anchors")
        all_rows = pd.concat([d.rows for d in full.values()], ignore_index=True)
        verify_data_unchanged(data, all_rows)
        _check_source(checkpoint, source_run)
        if implementation_identity(cfg) != identity:
            raise ValueError("训练期间源码或配置变化")
        receipt.update({"training": spec.identity(), "outcome": outcome, "objective": "source_equivalent",
                        "diagnostic_identity_sha256": diagnostic_digest,
                        "scientific_interpretation": "pending_review"})


def select_training_runs(output, *, baseline_inference_run, absolute_run, stretched_run):
    """从保存的三个方案 anchor 选择训练合同；不加载 waveform 或运行模型。"""
    baseline = Path(baseline_inference_run).resolve()
    base_receipt = json.loads((baseline / "receipt.json").read_text())
    if (base_receipt.get("status") != "complete" or base_receipt.get("mode") != "subset"
            or not base_receipt.get("formal_evidence")):
        raise ValueError("B0 必须是历史已完成的真实 ε-v2 subset")
    for name in ("ensemble_curve.csv", "inference_identity.json", "validation_parents.csv"):
        verify_artifact(baseline, base_receipt, name)
    base_identity = json.loads((baseline / "inference_identity.json").read_text())
    if base_identity["checkpoint"]["checkpoint_sha256"] != CHECKPOINT_SHA256 or base_identity["batch_size"] != 64:
        raise ValueError("B0 anchor 来源不匹配")
    parents = pd.read_csv(baseline / "validation_parents.csv")
    base_curve = pd.read_csv(baseline / "ensemble_curve.csv")
    mask = ((base_curve.sampler == "ddim") & (base_curve.N == 16)) | ((base_curve.sampler == "ddpm") & (base_curve.N == 4))
    frames = [base_curve[mask].assign(regime="baseline")]
    candidates = {"baseline": {"checkpoint_sha256": CHECKPOINT_SHA256,
                                "source_run": base_identity["checkpoint"]["source_run"]}}
    parent_columns = ["dataset_row_id", "samp_id", "split", "window_start_sample", "window_end_sample"]
    for regime, run in (("long_absolute", absolute_run), ("long_stretched", stretched_run)):
        path, receipt = read_stage(run, "train")
        for name in ("resolved_config.yaml", "final_anchors/anchor_metrics.csv", "final_anchor_parents.csv", "final.pt"):
            verify_artifact(path, receipt, name)
        spec = validate_training_config(OmegaConf.load(path / "resolved_config.yaml"))
        if spec.regime != regime or spec.seed != 20260811:
            raise ValueError("训练方案或首轮 seed 与 anchor 标签不一致")
        actual = pd.read_csv(path / "final_anchor_parents.csv")
        if not parents[parent_columns].equals(actual[parent_columns]):
            raise ValueError("训练比较 anchor 的 parent 身份或顺序不同")
        curve = pd.read_csv(path / "final_anchors/anchor_metrics.csv")
        frames.append(curve[curve.view == "correct"].assign(regime=regime))
        candidates[regime] = {"source_run": str(path), "checkpoint_sha256": receipt["artifacts"]["final.pt"]}
    curve = pd.concat(frames, ignore_index=True)
    selection = choose_training(curve)
    selection["selected_source"] = candidates[selection["selected_regime"]]
    selection["candidates"] = candidates
    with study_run(output, "select-training", synthetic=False) as (out, receipt):
        curve.to_csv(out / "training_comparison.csv", index=False)
        write_json(out / "training_selection.json", selection)
        receipt["selected_regime"] = selection["selected_regime"]


def _read_selection(grid_run):
    path, receipt = read_stage(grid_run, "grid")
    for name in ("selection.json", "sampler_grid.csv", "grid_identity.json"):
        verify_artifact(path, receipt, name)
    selection = json.loads((path / "selection.json").read_text())
    recomputed = choose_configurations(pd.read_csv(path / "sampler_grid.csv"))
    if selection["selected"] != recomputed["selected"]:
        raise ValueError("selection 不符合预先固定规则")
    return selection, json.loads((path / "grid_identity.json").read_text())


def evaluation_code_identity(cfg):
    return {key: value for key,value in implementation_identity(cfg)["files"].items()
            if key.startswith(("resp_train/respdiff/", "resp_train/respdiff_bcg/", "resp_train/metrics/"))}


def freeze_final_evaluation(output, *, grid_run, source_runs):
    """只读核对三个已完成模型，生成实际全量评价要使用的清单。"""
    selection, grid_identity = _read_selection(grid_run)
    checkpoints = []
    for source in source_runs:
        model, cfg, checkpoint = load_source(source, torch.device("cpu"))
        if (checkpoint["regime"] != grid_identity["checkpoint"]["regime"]
                or checkpoint["update"] != grid_identity["checkpoint"]["update"]
                or vars(model.spec) != grid_identity["model_spec"]
                or evaluation_code_identity(cfg) != grid_identity["evaluation_code"]
                or sha256(Path(source)/"val_parents.csv") != grid_identity["source_val_parents_sha256"]):
            raise ValueError("三 seed 模型的训练／数据／评价合同与选择阶段不一致")
        if (checkpoint["training_seed"] == 20260811
                and checkpoint["checkpoint_sha256"] != grid_identity["checkpoint"]["checkpoint_sha256"]):
            raise ValueError("首轮模型与 grid checkpoint 不一致")
        checkpoints.append(checkpoint)
        del model
    if sorted(item["training_seed"] for item in checkpoints) != [20260811,20260812,20260813]:
        raise ValueError("最终清单须包含恰好三个不同的计划训练 seed")
    manifest = {"protocol": NAME, "checkpoint_sources": checkpoints,
        "grid_identity_sha256": sha256(Path(grid_run)/"grid_identity.json"),
        "selection_sha256": sha256(Path(grid_run)/"selection.json"),
        "configurations": [{"role": row["role"], **StudySamplerSpec(row["sampler"],row["S"],row["N"]).identity()}
                           for row in selection["selected"]],
        "source_val_parents_sha256": grid_identity["source_val_parents_sha256"],
        "batch_size": 64, "trajectory_group_size": 1, "evaluation_code": grid_identity["evaluation_code"]}
    with study_run(output, "freeze-final", synthetic=False) as (out, receipt):
        write_json(out/"final_evaluation_manifest.json",manifest)
        receipt["training_seeds"] = [20260811,20260812,20260813]


def verify_final_manifest(path, grid_run, checkpoint):
    root, receipt = read_stage(path,"freeze-final")
    verify_artifact(root,receipt,"final_evaluation_manifest.json")
    manifest = json.loads((root/"final_evaluation_manifest.json").read_text())
    if (manifest["grid_identity_sha256"] != sha256(Path(grid_run)/"grid_identity.json")
            or manifest["selection_sha256"] != sha256(Path(grid_run)/"selection.json")):
        raise ValueError("最终清单与 grid 选择不一致")
    found = [item for item in manifest["checkpoint_sources"] if item["training_seed"] == checkpoint["training_seed"]]
    if len(found) != 1 or found[0]["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]:
        raise ValueError("实际评价 checkpoint 未列入冻结清单")
    return manifest


def run_grid(output, *, source_run=DEFAULT_SOURCE, device="cuda:0", tiny=False,
             grid_run=None, expected_sha256=None, training_selection_run=None, noise_check=False,
             final_manifest_run=None):
    phase = "noise-check" if noise_check else "full-validation" if grid_run is not None else "grid"
    if noise_check and grid_run is None:
        raise ValueError("噪声敏感性检查必须提供已冻结的 --grid-run")
    if phase == "full-validation" and final_manifest_run is None:
        raise ValueError("完整最终评价必须指定 --final-manifest-run")
    with study_run(output, phase, synthetic=tiny, source=None if tiny else source_run) as (out, receipt):
        model, cfg, checkpoint, bundles, _, index = _initialize(out, phase, source_run, device, tiny=tiny,
                                                               expected_sha256=expected_sha256)
        parents, rows, _ = bundles["val"]
        indices = list(range(len(rows))) if tiny or phase == "full-validation" else select_diagnostic_indices(rows)
        dataset = selected_dataset(parents, rows, indices)
        batch_size = 8 if tiny else 64
        evaluation_code = evaluation_code_identity(cfg)
        if phase == "full-validation":
            manifest = verify_final_manifest(final_manifest_run,grid_run,checkpoint)
            write_json(out/"final_evaluation_manifest.json",manifest)
        if not tiny and phase == "grid":
            if checkpoint["training_seed"] != 20260811:
                raise ValueError("参数搜索固定在 train seed20260811，其他 seed 只做冻结评价")
            if checkpoint["regime"] != "baseline" and training_selection_run is None:
                raise ValueError("新训练候选进入 grid 前须提供 --training-selection-run")
            if training_selection_run is not None:
                selection_path, selection_receipt = read_stage(training_selection_run, "select-training")
                verify_artifact(selection_path, selection_receipt, "training_selection.json")
                training_selection = json.loads((selection_path / "training_selection.json").read_text())
                if training_selection["selected_source"]["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]:
                    raise ValueError("grid checkpoint 不匹配冻结的训练方案选择")
        if grid_run is not None:
            selection, grid_identity = _read_selection(grid_run)
            reference = grid_identity["checkpoint"]
            if grid_identity.get("evaluation_code") != evaluation_code:
                raise ValueError("选择与最终评价使用不同的模型／采样／指标实现")
            if (checkpoint["regime"] != reference["regime"] or checkpoint["update"] != reference["update"]
                    or vars(model.spec) != grid_identity["model_spec"]):
                raise ValueError("最终评价模型的训练方案与选择阶段不一致")
            if checkpoint["training_seed"] == reference["training_seed"] and checkpoint["checkpoint_sha256"] != reference["checkpoint_sha256"]:
                raise ValueError("相同训练 seed 的 checkpoint 发生变化")
            if sha256(Path(source_run)/"val_parents.csv") != grid_identity["source_val_parents_sha256"]:
                raise ValueError("最终 validation 父身份与选择阶段不一致")
            noise_seeds = (20261013,20261014) if noise_check else (20261003,)
            grids = [(StudySamplerSpec(row["sampler"], row["S"], row["N"], seed), (row["N"],))
                     for seed in noise_seeds for row in selection["selected"]]
            write_json(out / "selection.json", selection)
        else:
            grids = grid_specs()
            if tiny:
                grids = [(StudySamplerSpec(spec.sampler, spec.nfe, 2), (1, 2)) for spec, _ in grids]
        identity = save_provenance(out, cfg, cpu_only=tiny,
            execution={"phase": phase, "synthetic": tiny, "checkpoint": checkpoint,
                       "device": str(next(model.parameters()).device), "batch_size": batch_size})
        data = data_identity(out, {"val": dataset}, index)
        grid_identity = {"protocol": NAME, "checkpoint": checkpoint, "model_spec": vars(model.spec),
            "evaluation_code": evaluation_code,
            "source_val_parents_sha256": None if tiny else sha256(Path(source_run)/"val_parents.csv"),
            "source_parent_indices": indices, "batch_size": batch_size, "trajectory_group_size": 1,
            "grids": [{"spec": spec.identity(), "counts": list(counts)} for spec, counts in grids]}
        write_json(out / "grid_identity.json", grid_identity)
        curves, profiles = [], []
        rule_ids = rows.iloc[np.linspace(0,len(rows)-1,min(64,len(rows)),dtype=int)].dataset_row_id.astype(int).tolist()
        with preserved_evaluation(model):
            for spec, counts in grids:
                name = f"{spec.sampler}_S{spec.nfe}_seed{spec.noise_seed}"
                sampling = out / f"{name}_sampling_N{max(counts)}"
                sampling.mkdir()
                predictions, profile = predict_prefixes(model, dataset, spec, counts, sampling,
                                                         batch_size=batch_size, acceleration=AccelerationSpec(1))
                profiles.extend(profile)
                for count in counts:
                    setting = out / f"{name}_N{count}"
                    curve = evaluate_setting(predictions.pop(count), dataset, cfg,
                        setting, spec, count, method_prefix=f"RespDiff-study-v1/{phase}/S{spec.nfe}")
                    if phase != "full-validation":
                        curve.update(rule_sample_summary(setting,rule_ids))
                    curve.update({"training_seed": checkpoint["training_seed"],
                                  "checkpoint_sha256": checkpoint["checkpoint_sha256"], "sampling_seed": spec.noise_seed})
                    curves.append(curve)
                    pd.DataFrame(curves).to_csv(out / "sampler_grid.csv", index=False)
                    pd.DataFrame(profiles).to_csv(out / "runtime_profile.csv", index=False)
        if grid_run is None:
            selection = choose_configurations(pd.DataFrame(curves), require_full_grid=not tiny)
            write_json(out / "selection.json", selection)
        plot_grid(pd.DataFrame(curves),out)
        if phase == "full-validation":
            timings = independent_timing(model,dataset,[spec for spec,_ in grids],batch_size=batch_size)
            write_json(out/"independent_runtime.json",timings)
        verify_data_unchanged(data, dataset.rows)
        _check_source(checkpoint, source_run, model)
        if implementation_identity(cfg) != identity:
            raise ValueError("采样期间源码或配置变化")
        receipt.update({"checkpoint": checkpoint, "selected": selection["selected"],
                        "parameter_sha256_after": state_digest(model)})


def gpu_check(output, *, device="cuda:0"):
    """原尺寸网络的训练与新增 sampler 工程检查；只使用合成输入。"""
    with study_run(output, "gpu-check", synthetic=True) as (out, receipt):
        cfg = config_contract("source_equivalent")
        seed_all(20260811)
        model = make_baseband_model(cfg).to(torch.device(device))
        identity = save_provenance(out, cfg, cpu_only=False,
            execution={"phase": "gpu-check", "synthetic": True, "device": device, "batch_size": 64})
        from .sampling import ensemble_prefixes
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        for _ in range(3):
            model.train()
            condition, target = (torch.randn(64, 1, 600, device=device) for _ in range(2))
            optimizer.zero_grad(set_to_none=True)
            losses = model.training_loss(condition, target)
            losses["loss"].backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("GPU 合成梯度非有限")
            optimizer.step()
        model.eval()
        keys = [("synthetic", i//13, i%13) for i in range(64)]
        for spec, _ in grid_specs():
            spec = StudySamplerSpec(spec.sampler, spec.nfe, 2)
            for _, value in ensemble_prefixes(model, condition, keys, spec, checkpoints=(1, 2)):
                if not torch.isfinite(value).all():
                    raise FloatingPointError("GPU 合成采样非有限")
        if implementation_identity(cfg) != identity:
            raise ValueError("工程检查期间来源变化")
        receipt.update({"model_spec": vars(model.spec), "n_training_updates": 3,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(device)),
                        "coverage": "B64 training; DDIM6/10/20/50 and DDPM50 N1/N2; synthetic only"})
