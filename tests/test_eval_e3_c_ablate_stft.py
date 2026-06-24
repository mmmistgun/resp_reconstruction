import torch
from torch.utils.data import DataLoader, Dataset

import scripts.eval_e3_c_ablate_stft as e3c


class _ToyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        value = torch.full((1, 8), float(idx + 1))
        return {
            "x": value,
            "target": value * 10,
            "meta": {
                "dataset_row_id": idx,
                "split": "val",
                "input_set": "toy",
                "residual_quality_class": "ok",
            },
        }


class _CountingBranch(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)
        self.calls = 0

    def forward(self, x, return_features=False):
        self.calls += 1
        feats = x * self.scale
        if return_features:
            return feats, x.size(-1)
        return feats


class _SumHead(torch.nn.Module):
    def forward(self, fused):
        return fused.sum(dim=1, keepdim=True)


class _ConcatDual(torch.nn.Module):
    branch_mode = "dual"
    fusion_mode = "concat_generic"
    fuse_len = 8

    def __init__(self):
        super().__init__()
        self.time_backbone = _CountingBranch(scale=2.0)
        self.stft_encoder = _CountingBranch(scale=3.0)
        self.fusion_head = _SumHead()


class _NativeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def token_count_for_length(self, length):
        return int(length)

    def forward_with_token_injection(self, x, token_injection, inject_position="post_mixer"):
        self.calls += 1
        return x + token_injection


class _NativeDual(torch.nn.Module):
    branch_mode = "dual"
    fusion_mode = "native_inject"
    stft_inject_position = "pre_mixer"
    sst_cached = False

    def __init__(self):
        super().__init__()
        self.time_backbone = _NativeBackbone()
        self.stft_encoder = _CountingBranch(scale=3.0)

    def _encode_stft_features(self, x, sst, target_len):
        return self.stft_encoder(x)[..., :target_len]

    def _project_stft_features(self, stft_feats):
        return stft_feats


def test_collect_ablation_predictions_reuses_one_loader_pass_and_branch_features():
    model = _ConcatDual()
    loader = DataLoader(_ToyDataset(), batch_size=2)

    outputs = e3c.collect_concat_ablation_predictions(
        model,
        loader,
        device=torch.device("cpu"),
        max_windows=2,
        modes=["normal", "stft_zero", "time_zero"],
        shuffle_seed=123,
    )

    assert model.time_backbone.calls == 1
    assert model.stft_encoder.calls == 1
    assert set(outputs) == {"normal", "stft_zero", "time_zero"}
    assert outputs["normal"]["r_tho_hat"][0, 0, 0] == 5.0
    assert outputs["stft_zero"]["r_tho_hat"][0, 0, 0] == 2.0
    assert outputs["time_zero"]["r_tho_hat"][0, 0, 0] == 3.0
    assert outputs["normal"]["dataset_row_id"].tolist() == [0, 1]


def test_collect_ablation_predictions_supports_native_inject_token_delta_modes():
    model = _NativeDual()
    loader = DataLoader(_ToyDataset(), batch_size=2)

    outputs = e3c.collect_ablation_predictions(
        model,
        loader,
        device=torch.device("cpu"),
        max_windows=2,
        modes=["normal", "stft_zero", "time_zero"],
        shuffle_seed=123,
    )

    assert model.stft_encoder.calls == 1
    assert model.time_backbone.calls == 3
    assert outputs["normal"]["r_tho_hat"][0, 0, 0] == 4.0
    assert outputs["stft_zero"]["r_tho_hat"][0, 0, 0] == 1.0
    assert outputs["time_zero"]["r_tho_hat"][0, 0, 0] == 3.0
    assert outputs["normal"]["dataset_row_id"].tolist() == [0, 1]


def test_discover_run_specs_uses_checkpoint_name_and_skips_existing_outputs(tmp_path):
    run_dir = tmp_path / "arm" / "dual" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("model:\n  name: time_stft_dual1d\n", encoding="utf-8")
    (run_dir / "checkpoint_top1.pt").write_bytes(b"ckpt")
    (run_dir / "metrics_e3c_normal_top1.csv").write_text("x\n1\n", encoding="utf-8")

    specs = e3c.discover_run_specs(
        tmp_path,
        arm="arm",
        branch="dual",
        checkpoint_name="checkpoint_top1.pt",
        modes=["normal", "stft_zero"],
        force=False,
    )
    forced = e3c.discover_run_specs(
        tmp_path,
        arm="arm",
        branch="dual",
        checkpoint_name="checkpoint_top1.pt",
        modes=["normal", "stft_zero"],
        force=True,
    )

    assert [spec.mode for spec in specs] == ["stft_zero"]
    assert specs[0].metrics_output == run_dir / "metrics_e3c_stft_zero_top1.csv"
    assert [spec.mode for spec in forced] == ["normal", "stft_zero"]


def test_assign_run_groups_round_robin_by_run_not_mode(tmp_path):
    run_a = tmp_path / "arm" / "dual" / "run_a"
    run_b = tmp_path / "arm" / "dual" / "run_b"
    for run_dir in (run_a, run_b):
        run_dir.mkdir(parents=True)
    groups = {
        (run_a, run_a / "checkpoint_top1.pt", run_a / "config.yaml"): {
            "normal": run_a / "metrics_e3c_normal_top1.csv",
            "stft_zero": run_a / "metrics_e3c_stft_zero_top1.csv",
        },
        (run_b, run_b / "checkpoint_top1.pt", run_b / "config.yaml"): {
            "normal": run_b / "metrics_e3c_normal_top1.csv",
            "stft_zero": run_b / "metrics_e3c_stft_zero_top1.csv",
        },
    }

    assignments = e3c.assign_run_groups(groups, ["cuda:0", "cuda:1"])

    assert [assignment[2] for assignment in assignments] == ["cuda:0", "cuda:1"]
    assert [len(assignment[1]) for assignment in assignments] == [2, 2]


def test_collect_ablation_predictions_reports_batch_timing(capsys):
    model = _ConcatDual()
    loader = DataLoader(_ToyDataset(), batch_size=1)

    e3c.collect_concat_ablation_predictions(
        model,
        loader,
        device=torch.device("cpu"),
        max_windows=2,
        modes=["normal"],
        shuffle_seed=123,
        progress_every=1,
        progress_label="toy-run",
    )

    captured = capsys.readouterr().out
    assert "collect toy-run batch=1" in captured
    assert "data_wait=" in captured
    assert "compute=" in captured


def test_resolve_metrics_workers_caps_to_task_count():
    assert e3c.resolve_metrics_workers(0, task_count=10) == 1
    assert e3c.resolve_metrics_workers(1, task_count=10) == 1
    assert e3c.resolve_metrics_workers(8, task_count=2) == 2
    assert e3c.resolve_metrics_workers(8, task_count=10) == 8


def test_build_metric_chunk_tasks_splits_each_mode_by_window_range(tmp_path):
    outputs = {
        "normal": tmp_path / "normal.csv",
        "stft_zero": tmp_path / "zero.csv",
    }

    tasks = e3c.build_metric_chunk_tasks(outputs, n_windows=5, chunk_size=2)

    assert [(task.mode, task.start, task.end) for task in tasks] == [
        ("normal", 0, 2),
        ("normal", 2, 4),
        ("normal", 4, 5),
        ("stft_zero", 0, 2),
        ("stft_zero", 2, 4),
        ("stft_zero", 4, 5),
    ]
