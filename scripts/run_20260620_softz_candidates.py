from __future__ import annotations

import argparse
import subprocess
import sys


COMMON_OVERRIDES = [
    "data.max_train_windows=null",
    "data.max_val_windows=null",
    "data.train_sample_seed=20260610",
    "data.val_sample_seed=20260611",
    "loss.phase_alignment_weight=0.0",
    "loss.signed_corr_weight=0.2",
    "training.epochs=50",
    "training.batch_size=128",
    "training.patience=8",
    "training.min_delta=0.001",
    "training.device=cuda:0",
    "training.show_progress=false",
    "training.checkpoint_gate.metric=auto_direction",
    "training.checkpoint_gate.max=0.5",
    "outputs.run_root=runs/tho_research_v2_20260620_softz_model_candidates",
]


MODEL_OVERRIDES = {
    "patch_mixer1d": [
        "model.name=patch_mixer1d",
        "model.patch_len=256",
        "model.patch_stride=128",
        "model.mixer_layers=2",
        "model.overlap_window=hann",
        "model.output_smoothing_kernel=1",
    ],
    "patch_hann_bandlimited_output1d": [
        "model.name=patch_hann_bandlimited_output1d",
        "model.patch_len=256",
        "model.patch_stride=128",
        "model.mixer_layers=2",
        "model.max_freq_hz=0.7",
    ],
    "polyphase_patch_hann_bandlimited1d": [
        "model.name=polyphase_patch_hann_bandlimited1d",
        "model.offsets=[1,2,4]",
        "model.patch_len=256",
        "model.patch_stride=128",
        "model.mixer_layers=2",
        "model.max_freq_hz=0.7",
    ],
    "multiscale_patch_hann_bandlimited1d": [
        "model.name=multiscale_patch_hann_bandlimited1d",
        "model.patch_lengths=[256,512,1024,2048]",
        "model.patch_stride_ratio=0.5",
        "model.mixer_layers=2",
        "model.max_freq_hz=0.7",
    ],
    "period_aware_patch_hann_bandlimited1d": [
        "model.name=period_aware_patch_hann_bandlimited1d",
        "model.period_secs=[2.5,4.0,6.0,10.0]",
        "model.patch_stride_ratio=0.5",
        "model.mixer_layers=2",
        "model.max_freq_hz=0.7",
    ],
    "multiscale_decomp_mixer1d": [
        "model.name=multiscale_decomp_mixer1d",
        "model.downsample_factors=[1,4,16]",
        "model.mixer_layers=2",
    ],
    "downsampled_ssm1d": [
        "model.name=downsampled_ssm1d",
        "model.latent_stride=20",
        "model.state_layers=2",
    ],
}


DEFAULT_MODELS = [
    "patch_mixer1d",
    "patch_hann_bandlimited_output1d",
    "polyphase_patch_hann_bandlimited1d",
    "multiscale_patch_hann_bandlimited1d",
    "period_aware_patch_hann_bandlimited1d",
    "multiscale_decomp_mixer1d",
    "downsampled_ssm1d",
]


DEFAULT_SEEDS = [20260700, 20260710, 20260837]


def main() -> None:
    parser = argparse.ArgumentParser(description="顺序重跑 20260620 soft-z 候选模型")
    parser.add_argument("--skip", action="append", default=[], help="跳过 model:seed，例如 patch_mixer1d:20260700")
    parser.add_argument("--only", action="append", default=[], help="仅运行指定模型，可重复传入")
    args = parser.parse_args()
    skipped = set(args.skip)
    model_names = args.only or DEFAULT_MODELS
    unknown = sorted(set(model_names) - set(MODEL_OVERRIDES))
    if unknown:
        raise SystemExit(f"未知模型: {unknown}，可用: {sorted(MODEL_OVERRIDES)}")

    for model_name in model_names:
        for seed in DEFAULT_SEEDS:
            tag = f"{model_name}:{seed}"
            if tag in skipped:
                print(f"skip {tag}", flush=True)
                continue
            overrides = [*COMMON_OVERRIDES, *MODEL_OVERRIDES[model_name], f"training.seed={seed}"]
            cmd = [
                sys.executable,
                "scripts/train_tho_small.py",
                "--config",
                "configs/tho_research_v2.yaml",
            ]
            for override in overrides:
                cmd.extend(["--set", override])
            print(f"start {tag}", flush=True)
            subprocess.run(cmd, check=True)
            print(f"done {tag}", flush=True)


if __name__ == "__main__":
    main()
