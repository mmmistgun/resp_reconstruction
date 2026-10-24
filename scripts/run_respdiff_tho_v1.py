"""RespDiff开发入口：来源核验与短时CPU合成流程。"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import scipy
import torch
import yaml

from resp_train.respdiff import RespDiff, RespDiffSpec
from resp_train.respdiff.data import chunk_manifest, prepare_condition, prepare_target
from resp_train.respdiff.experiment import (
    evaluate_chunk_predictions, load_checkpoint, save_checkpoint, train_step,
)
from resp_train.respdiff.provenance import SOURCE_COMMIT, verify_source
from resp_train.respdiff.settings import build_source_optimizer, validate_source_settings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/respdiff_tho_v1/experiment.yaml"


def load_development_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if (config["protocol"]["stage"] != "synthetic_development"
            or config["protocol"]["formal_enabled"] is not False
            or config["source"]["commit"] != SOURCE_COMMIT
            or config["source"]["parameter_policy"] != "paper_then_source"):
        raise ValueError("当前CLI仅接受来源固定的synthetic开发合同")
    diffusion = {"steps": 50, "beta_start": .0001, "beta_end": .5, "spectral_weight": .01,
                 "noise_reduction": "sum_div_length", "fft_reduction": "mean", "sampler": "ddpm"}
    signal = config["signal"]
    if config["diffusion"] != diffusion or any(signal[key] != value for key, value in {
        "parent_fs": 100, "parent_samples": 18000, "model_fs": 30, "chunk_samples": 150,
        "chunks_per_parent": 36, "resample": "scipy.signal.resample",
        "lowpass": {"order": 8, "cutoff_hz": 1, "padtype": "odd", "padlen": 27},
    }.items()):
        raise ValueError("声明的信号/扩散设置与已实现合同不符")
    validate_source_settings(config)
    return config


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def synthetic_smoke(output, source_root, config_path=DEFAULT_CONFIG):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        source = verify_source(source_root)
        config = load_development_config(config_path)
        smoke = config["synthetic_smoke"]
        torch.set_num_threads(1)
        torch.manual_seed(smoke["seed"])
        spec = RespDiffSpec(hidden_dim=smoke["hidden_dim"], num_layers=smoke["num_layers"],
                            output_dim=smoke["output_dim"], profile=config["model"]["profile"])
        batch = smoke["sampling_batch"]
        if type(batch) is not int or not 1 <= batch <= 36 or not 1 <= smoke["train_batch"] <= 36:
            raise ValueError("synthetic batch必须在1…36之间")
        execution = {"stage": "synthetic_cpu_only", "spec": asdict(spec), "seed": smoke["seed"],
                     "normalization": "segment_soft_z", "n_samples": smoke["n_samples"], "sampling_batch": batch,
                     "resolved_config": config,
                     "metric_operator": "resp_train.metrics.final_evaluation.evaluate_window",
                     "command": sys.argv, "torch": torch.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__, "source": source, "implementation": {}}
        paths = sorted((Path(__file__).resolve().parents[1] / "resp_train/respdiff").glob("*.py"))
        paths.append(Path(__file__).resolve())
        paths.extend([Path(config_path).resolve(), ROOT / "resp_train/metrics/final_evaluation.py",
                      ROOT / "resp_train/protocols/respiration.py"])
        for path in paths:
            execution["implementation"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(output / "execution.json", execution)
        model = RespDiff(spec)
        optimizer = build_source_optimizer(model, config)
        t = np.arange(18000) / 100
        reference = (1 + .25 * np.sin(2 * np.pi * .02 * t)) * np.sin(2 * np.pi * .2 * t)
        condition = reference + .15 * np.sin(2 * np.pi * 3 * t)
        x, y = prepare_condition(condition), prepare_target(reference)
        train_batch = smoke["train_batch"]
        history = train_step(model, optimizer, torch.from_numpy(x[:train_batch]), torch.from_numpy(y[:train_batch]),
                             generator=torch.Generator().manual_seed(41))
        save_checkpoint(output / "synthetic.pt", model, update=1, metadata={"stage": "synthetic"})
        model, _ = load_checkpoint(output / "synthetic.pt")
        model.eval()
        rows = pd.DataFrame([{"dataset_row_id": 1, "split": "val", "samp_id": 1,
                              "source_npz": "/synthetic/bcg.npz", "target_source_npz": "/synthetic/tho.npz",
                              "window_start_sample": 0, "window_end_sample": 18000}])
        manifest = chunk_manifest(rows)
        results = []
        for start in range(0, 36, batch):
            results.append(model.sample_mean(torch.from_numpy(x[start:start + batch]),
                           keys=manifest.sampling_key.iloc[start:start + batch].tolist(),
                           seed=17, n_samples=smoke["n_samples"]).numpy())
        predictions = np.concatenate(results)
        evaluated = evaluate_chunk_predictions(predictions, manifest, rows, reference[None])
        np.save(output / "prediction.npy", evaluated["waveform"])
        manifest.to_csv(output / "chunks.csv", index=False)
        write_json(output / "metrics.json", {"training": history, "validation": evaluated["summary"]})
        artifacts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(output.iterdir()) if path.is_file()}
        write_json(output / "receipt.json", {"status": "complete", "stage": "synthetic_cpu_only",
                   "parent_rows": 1, "chunks": 36, "model_updates": 1, "artifacts": artifacts,
                   "limitation": "小RNN合成流程，不代表原规模验收或真实质量"})
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error),
                                            "traceback": traceback.format_exc()})
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-source", "synthetic-smoke"))
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/disk_code/marques/reference_repos/RespDiff"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.command == "check-source":
        print(json.dumps(verify_source(args.source_root), ensure_ascii=False, indent=2))
    else:
        if args.output is None:
            parser.error("synthetic-smoke要求新的--output目录")
        print(synthetic_smoke(args.output, args.source_root, args.config))


if __name__ == "__main__":
    main()
