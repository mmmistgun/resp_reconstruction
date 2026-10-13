"""计划 v1 的关键科研合同；仅使用合成／disposable fixture。"""

import copy
import json
import random

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.baseband import RespDiffBCGBaseband, config_contract
from resp_train.respdiff_bcg.inference_v2 import state_digest
from resp_train.respdiff_bcg.runtime import seed_all, sha256
from resp_train.respdiff_bcg.runtime import train_updates
from resp_train.respdiff_bcg.sampling import SamplerSpec, ensemble_prefixes
from resp_train.respdiff_bcg.study_contract import (
    GRID, METRICS, StudySamplerSpec, TrainingSpec, choose_configurations, choose_training, local_stability,
)
from resp_train.respdiff_bcg.study_diagnostics import (
    condition_permutation, error_statistics, forward_noise, run_forward_probes,
)
from resp_train.respdiff_bcg.study_io import (
    load_source, stratified_indices, study_run, training_config, validate_training_config,
)
from resp_train.respdiff_bcg.study_runtime import (
    evaluation_code_identity, freeze_final_evaluation, load_settings, run_diagnosis, run_grid,
    run_training, verify_final_manifest,
)
from resp_train.respdiff_bcg.study_evaluation import independent_timing
from resp_train.respdiff_bcg.study_contract import NAME
from resp_train.respdiff_bcg.study_state import preserved_evaluation, rng_state
from resp_train.respdiff_bcg.study_training import ResumableBatches, load_resume, train_with_snapshots
from scripts.run_respdiff_bcg_baseband_v1 import fixture
from scripts.run_respdiff_bcg_study_v1 import main


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    def forbid(name):
        def forbidden(*args, **kwargs):
            raise AssertionError(f"CPU fixture 不能调用 CUDA: {name}")
        return forbidden
    for name in ("_lazy_init", "synchronize", "manual_seed_all", "get_rng_state", "get_rng_state_all"):
        monkeypatch.setattr(torch.cuda, name, forbid(name))
    yield
    torch.set_num_threads(previous)


def model(seed=123):
    seed_all(seed, cpu_only=True)
    return RespDiffBCGBaseband(RespDiffSpec(8, 1, 4), objective="source_equivalent")


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a,b in zip(left, right):
            assert_tree_equal(a,b)
    else:
        assert left == right


def test_contract_and_explicit_timestep_grid():
    load_settings()
    assert sum(s*n[-1] for (_, s), n in GRID.items()) == 1112
    for (kind, steps), counts in GRID.items():
        spec = StudySamplerSpec(kind, steps, counts[-1])
        assert len(spec.timesteps) == steps
        assert spec.timesteps[0] == 49 and spec.timesteps[-1] == 0
        assert all(a>b for a,b in zip(spec.timesteps, spec.timesteps[1:]))
    assert StudySamplerSpec().timesteps == SamplerSpec().timesteps
    assert StudySamplerSpec(nfe=10).timesteps == (49,44,38,33,27,22,16,11,5,0)
    for spec in (TrainingSpec("long_absolute"), TrainingSpec("long_stretched"), TrainingSpec("overfit")):
        cfg = training_config(config_contract("source_equivalent"), spec)
        assert validate_training_config(cfg) == spec
        assert list(cfg.diagnostics.forward_noise_seeds) == [20261010,20261011,20261012]
        assert cfg.diagnostics.train_probe_parents == (8 if spec.regime == "overfit" else 32)
        cfg.objective.spectral_weight *= 2
        with pytest.raises(ValueError, match="合同"):
            validate_training_config(cfg)
    with pytest.raises(ValueError):
        StudySamplerSpec("ddpm", 50, 16)
    with pytest.raises(ValueError):
        StudySamplerSpec("ddim", 20, 16)
    with pytest.raises(ValueError):
        SamplerSpec("ddim", 10, 1)


@pytest.mark.parametrize("kind,steps", [("ddim",6), ("ddpm",50)])
def test_legacy_sampler_exact_and_nested_prefixes(kind, steps):
    net = model().eval()
    condition = torch.randn(2,1,600)
    keys = [("val", 10, 0), ("val", 11, 1)]
    old = dict(ensemble_prefixes(net, condition, keys, SamplerSpec(kind, steps, 2), checkpoints=(1,2)))
    new = dict(ensemble_prefixes(net, condition, keys, StudySamplerSpec(kind, steps, 2), checkpoints=(1,2)))
    for n in (1,2):
        assert torch.equal(old[n], new[n])
    single = dict(ensemble_prefixes(net, condition, keys, StudySamplerSpec(kind, steps, 1)))
    assert torch.equal(single[1], new[1])


@pytest.mark.parametrize("steps", [6,10,20,50])
def test_ddim_oracle_reaches_clean_endpoint(steps, monkeypatch):
    net = model().eval()
    target = torch.randn(2,1,600) * .2
    def oracle(condition, xt, t):
        alpha = net.alpha_torch[t]
        return (xt-alpha.sqrt()*target)/(1-alpha).sqrt()
    monkeypatch.setattr(net.diffusion_model, "forward", oracle)
    result = dict(ensemble_prefixes(net, torch.zeros_like(target), [("val",10,0),("val",11,0)],
                                   StudySamplerSpec("ddim", steps, 1)))[1]
    torch.testing.assert_close(result, target, atol=1e-6, rtol=1e-6)


def test_noise_x0_identity_zero_baseline_and_nonfinite():
    generator = torch.Generator().manual_seed(7)
    target, noise, predicted = (torch.randn(3,1,600, generator=generator, dtype=torch.float64) for _ in range(3))
    for t in (0,19,39,49):
        alpha = torch.tensor(np.cumprod(1-np.linspace(.0001,.5,50))[t], dtype=torch.float64)
        xt = alpha.sqrt()*target+(1-alpha).sqrt()*noise
        actual, _ = error_statistics(target, noise, predicted, alpha)
        x0 = (xt-(1-alpha).sqrt()*predicted)/alpha.sqrt()
        expected = (x0-target).square().mean((1,2)).sqrt()
        np.testing.assert_allclose([r["x0_rmse"] for r in actual], expected.numpy(), rtol=1e-12)
        baseline, _ = error_statistics(target, noise, xt/(1-alpha).sqrt(), alpha)
        np.testing.assert_allclose([r["epsilon_mse"] for r in baseline],
                                   [r["zero_x0_epsilon_mse"] for r in baseline], rtol=1e-10)
    predicted[0,0,0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        error_statistics(target, noise, predicted, alpha)


def test_condition_derangement_preserves_multiset_and_avoids_parent():
    keys = [("train", row, chunk) for row in (1,2,3,4) for chunk in range(4)]
    perm, status = condition_permutation(keys, 20261020)
    assert sorted(perm) == list(range(len(keys))) and all(i!=j for i,j in enumerate(perm))
    assert status["same_parent"] == 0
    assert perm == condition_permutation(keys, 20261020)[0]
    assert perm != condition_permutation(keys, 20261021)[0]
    x = torch.arange(16)[:,None].float()
    assert torch.equal(x.sort(dim=0).values, x[perm].sort(dim=0).values)
    net = model().eval()
    condition = torch.randn(16,1,600)
    for encoder in (net.diffusion_model.ppg_encoder1, net.diffusion_model.ppg_encoder2):
        torch.testing.assert_close(encoder(condition)[perm], encoder(condition[perm]), rtol=1e-5, atol=1e-6)
    first = forward_noise(keys, 1, torch.device("cpu"))
    pieces = torch.cat([forward_noise(keys[:7], 1, torch.device("cpu")),
                        forward_noise(keys[7:], 1, torch.device("cpu"))])
    assert torch.equal(first, pieces)


def test_stratification_covers_subjects_then_records(tmp_path):
    rows = []
    for subject in range(4):
        for record in range(2):
            for window in range(3):
                rows.append({"dataset_row_id": len(rows), "samp_id": subject, "split": "train",
                    "source_npz": str(tmp_path/f"x_{subject}_{record}.npz"),
                    "target_source_npz": str(tmp_path/f"y_{subject}_{record}.npz"),
                    "window_start_sample": window*1000, "window_end_sample": window*1000+18000})
    rows = pd.DataFrame(rows)
    assert rows.iloc[stratified_indices(rows,4)].samp_id.nunique() == 4
    assert rows.iloc[stratified_indices(rows,8)].target_source_npz.nunique() == 8
    assert stratified_indices(rows,24) == list(range(24))
    rows.loc[0,"split"] = "test"
    with pytest.raises(ValueError, match="test"):
        stratified_indices(rows,4)


def test_probes_preserve_rng_parameters_gradients_and_optimizer(tmp_path):
    cfg = config_contract("source_equivalent")
    dataset = fixture(tmp_path, cfg, "train")
    net = model()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    net.training_loss(torch.randn(2,1,600), torch.randn(2,1,600))["loss"].backward()
    optimizer.step()
    before = (state_digest(net), rng_state(torch.device("cpu")), copy.deepcopy(optimizer.state_dict()),
              [p.grad.clone() for p in net.parameters()], copy.deepcopy(net.last_diagnostics))
    run_forward_probes(net, dataset, tmp_path/"probes", batch_size=8, steps=(0,49), noise_seeds=(7,), pairing_seeds=(10,))
    assert net.training and state_digest(net) == before[0]
    assert_tree_equal(before[1], rng_state(torch.device("cpu")))
    assert_tree_equal(before[2], optimizer.state_dict())
    assert_tree_equal(before[3], [p.grad for p in net.parameters()])
    assert_tree_equal(before[4], net.last_diagnostics)
    frame = pd.read_csv(tmp_path/"probes/timestep_metrics.csv")
    assert len(frame) == 26*2*2
    with pytest.raises(RuntimeError, match="改变"):
        with preserved_evaluation(net):
            next(net.parameters()).add_(1)


def test_resumable_batches_match_legacy_dataloader():
    stream = ResumableBatches(17, 4, 20260811)
    loader = DataLoader(range(17), batch_size=4, shuffle=True, drop_last=False,
                        generator=torch.Generator().manual_seed(20260811), num_workers=0)
    for _ in range(3):
        for batch in loader:
            assert stream.next() == batch.tolist()
    state = stream.state_dict()
    restored = ResumableBatches(17,4,20260811)
    restored.load_state_dict(state)
    for _ in range(9):
        assert stream.next() == restored.next()
    state["cursor"] = 3
    with pytest.raises(ValueError):
        restored.load_state_dict(state)


def test_cooldown_exact_state_and_random_sequence(tmp_path):
    from torch.utils.data import default_collate
    from resp_train.respdiff_bcg.study_cooldown import CooldownFork
    from resp_train.respdiff_bcg.study_state import restore_rng
    cfg = config_contract('source_equivalent')
    dataset = fixture(tmp_path, cfg, 'train')
    spec = TrainingSpec('overfit')
    control = tmp_path/'control'
    control.mkdir()
    train_with_snapshots(model(20260811), dataset, cfg, spec, control,
                         fixture_updates=4, fixture_batch_size=8)
    parent = torch.load(control/'checkpoints/update_000002.pt', weights_only=False)
    reference = model(999)
    reference.load_state_dict(parent['model'])
    optimizer = torch.optim.Adam(reference.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=.1)
    optimizer.load_state_dict(parent['optimizer'])
    scheduler.load_state_dict(parent['scheduler'])
    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['initial_lr'] = 1e-5
    scheduler.base_lrs = scheduler._last_lr = [1e-5]
    stream = ResumableBatches(len(dataset), 8, spec.seed)
    stream.load_state_dict(parent['batch_stream'])
    restore_rng(parent['rng'], torch.device('cpu'))
    for _ in range(2):
        batch = default_collate([dataset[i] for i in stream.next()])
        optimizer.zero_grad(set_to_none=True)
        reference.training_loss(batch['x'], batch['target'])['loss'].backward()
        optimizer.step()
        scheduler.step()
    expected_rng = rng_state(torch.device('cpu'))
    out = tmp_path/'cooldown'
    out.mkdir()
    fork = CooldownFork(control, out, synthetic=True)
    net = model(777)
    outcome = train_with_snapshots(net, dataset, cfg, spec, out, cooldown=fork,
                                  fixture_updates=4, fixture_batch_size=8)
    final = torch.load(out/'final.pt', weights_only=False)
    assert outcome['optimizer_updates_this_run'] == 2
    assert_tree_equal(reference.state_dict(), final['model'])
    assert_tree_equal(optimizer.state_dict(), final['optimizer'])
    assert_tree_equal(scheduler.state_dict(), final['scheduler'])
    assert_tree_equal(expected_rng, final['rng'])
    assert final['training']['cooldown']['learning_rate_after'] == 1e-5
    assert json.loads((out/'sequence_verification.json').read_text())['verified_updates'] == 2
    with pytest.raises(ValueError, match='合同'):
        load_resume(out/'final.pt', training=parent['training'], data_digest=parent['training_data_sha256'])
    with pytest.raises(FileNotFoundError):
        CooldownFork(control, tmp_path/'invalid')
    drift = tmp_path/'drift'
    drift.mkdir()
    bad_fork = CooldownFork(control, drift, synthetic=True)
    history = [json.loads(line) for line in (control/'history.jsonl').read_text().splitlines()]
    history[2]['chunk_indices'].reverse()
    (control/'history.jsonl').write_text('\n'.join(json.dumps(row) for row in history)+'\n')
    with pytest.raises(ValueError, match='batch'):
        train_with_snapshots(model(42), dataset, cfg, spec, drift, cooldown=bad_fork,
                             fixture_updates=4, fixture_batch_size=8)


def test_cooldown_cli_requires_training_authorization(tmp_path):
    from scripts.run_respdiff_bcg_cooldown_v1 import main as cooldown_main
    args = [value for name in ('baseline-run', 'p1-run', 'p2-run', 'audit-run', 'output')
            for value in ('--'+name, str(tmp_path/name))]
    for flags in ([], ['--confirm-data-access'], ['--confirm-training'],
                  ['--confirm-data-access', '--confirm-training', '--device', 'cpu']):
        with pytest.raises(SystemExit) as exc:
            cooldown_main(args+flags)
        assert exc.value.code == 2


def test_snapshot_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    cfg = config_contract("source_equivalent")
    dataset = fixture(tmp_path, cfg, "train")
    spec = TrainingSpec("long_absolute")
    monkeypatch.setattr(TrainingSpec, "milestones", property(lambda self: (2,4)))
    final = []
    for name, stop in (("whole", None), ("part",3)):
        output = tmp_path/name
        output.mkdir()
        net = model(20260811)
        train_with_snapshots(net, dataset, cfg, spec, output, fixture_updates=6, fixture_batch_size=8, stop_after=stop)
    resumed = tmp_path/"resumed"
    resumed.mkdir()
    net = model(999)
    train_with_snapshots(net, dataset, cfg, spec, resumed, fixture_updates=6, fixture_batch_size=8,
                         resume=tmp_path/"part/checkpoints/update_000003.pt")
    left = torch.load(tmp_path/"whole/final.pt", weights_only=False)
    right = torch.load(resumed/"final.pt", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "rng", "batch_stream"):
        assert_tree_equal(left[key], right[key])
    history = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
    assert history(tmp_path/"whole/history.jsonl")[3:] == history(resumed/"history.jsonl")
    legacy = tmp_path/"legacy_reference"
    legacy.mkdir()
    cfg.training.milestones = [2,4]
    legacy_model = model(20260811)
    train_updates(legacy_model,dataset,cfg,legacy,updates=6,batch_size=8,
                  history_extra=lambda: legacy_model.last_diagnostics)
    legacy_final = torch.load(legacy/"final.pt",weights_only=False)
    for key in ("model","optimizer","scheduler"):
        assert_tree_equal(left[key],legacy_final[key])
    assert history(tmp_path/"whole/history.jsonl") == history(legacy/"history.jsonl")
    finalize = tmp_path/"finish_evaluation"
    finalize.mkdir()
    recovery_model = model(999)
    recovery = train_with_snapshots(recovery_model,dataset,cfg,spec,finalize,
        fixture_updates=6,fixture_batch_size=8,resume=tmp_path/"whole/final.pt")
    assert recovery["optimizer_updates_this_run"] == 0 and recovery["complete"]
    restored = torch.load(finalize/"final.pt",weights_only=False)
    for key in ("model","optimizer","scheduler","rng","batch_stream"):
        assert_tree_equal(left[key],restored[key])
    assert restored["parent_checkpoint_sha256"] == sha256(tmp_path/"whole/final.pt")
    assert (finalize/"history.jsonl").read_text() == ""
    assert left["selector"] == "final_update"
    assert torch.load(tmp_path/"part/checkpoints/update_000003.pt", weights_only=False)["selector"] == "diagnostic_update"
    with pytest.raises(ValueError, match="旧 checkpoint"):
        load_resume(tmp_path/"legacy.pt", training={}, data_digest="x")


def curve():
    return pd.DataFrame([{"sampler": kind, "nfe_per_trajectory": steps, "N": n,
                          "denoiser_calls_per_chunk": steps*n,
                          **dict(zip(METRICS,(.8,1.,10.,.1)))}
                         for (kind,steps),counts in GRID.items() for n in counts])


def test_full_grid_selection_stability_and_training_contrast():
    data = curve()
    selected = choose_configurations(data)
    assert selected["selected"][0]["sampler"] == "ddpm" and selected["selected"][0]["N"] == 1
    assert selected["selected"][1]["sampler"] == "ddim" and selected["selected"][1]["N"] == 1
    assert selected["selected"][2] == {"sampler":"ddim", "S":6, "N":16, "role":"fixed_reference"}
    assert all(v["status"] == "local_stability" for v in selected["stability"]["status_by_sampler"])
    with pytest.raises(ValueError, match="完整"):
        choose_configurations(data.iloc[:-1])
    bad = data.copy()
    bad.loc[0, METRICS[0]] = float("nan")
    with pytest.raises(ValueError, match="非有限"):
        choose_configurations(bad)
    anchors = data[((data.sampler=="ddim")&(data.nfe_per_trajectory==6)&(data.N==16))|
                   ((data.sampler=="ddpm")&(data.N==4))]
    joined = pd.concat([anchors.assign(regime=r) for r in ("baseline","long_absolute","long_stretched")])
    assert choose_training(joined)["selected_regime"] == "baseline"
    joined.loc[joined.regime=="long_stretched", METRICS[0]] = .85
    assert choose_training(joined)["selected_regime"] == "long_stretched"


def test_failure_receipt_immutable_output_and_authorization_gates(tmp_path):
    out = tmp_path/"failed"
    with pytest.raises(FloatingPointError):
        with study_run(out,"diagnose",synthetic=True):
            raise FloatingPointError("synthetic nonfinite")
    receipt = json.loads((out/"receipt.json").read_text())
    assert receipt["status"] == "failed" and "failure.json" in receipt["artifacts"]
    for name, digest in receipt["artifacts"].items():
        assert sha256(out/name) == digest
    with pytest.raises(FileExistsError):
        with study_run(out,"diagnose",synthetic=True):
            pass
    for phase in ("diagnose","train","grid","noise-check","freeze-final","full-validation","gpu-check"):
        target = tmp_path/phase
        with pytest.raises(SystemExit):
            main([phase,"--output",str(target)])
        assert not target.exists()
    with pytest.raises(ValueError):
        load_source(out,torch.device("cpu"))


def test_final_manifest_binds_three_models_and_exact_selection(tmp_path,monkeypatch):
    cfg = config_contract("source_equivalent")
    identities, sources = {}, []
    for seed in (20260811,20260812,20260813):
        source = tmp_path/str(seed)
        source.mkdir()
        (source/"val_parents.csv").write_text("dataset_row_id\n1\n2\n")
        identities[str(source)] = {"source_run":str(source),"checkpoint_sha256":f"{seed:064x}",
            "training_seed":seed,"regime":"baseline","update":6400}
        sources.append(source)
    def fake_source(source,device):
        return model().eval(),cfg,identities[str(source)]
    monkeypatch.setattr("resp_train.respdiff_bcg.study_runtime.load_source",fake_source)
    grid = tmp_path/"grid"
    grid.mkdir()
    data = curve()
    data.to_csv(grid/"sampler_grid.csv",index=False)
    (grid/"selection.json").write_text(json.dumps(choose_configurations(data)))
    identity = {"checkpoint":identities[str(sources[0])],"model_spec":vars(model().spec),
                "evaluation_code":evaluation_code_identity(cfg),
                "source_val_parents_sha256":sha256(sources[0]/"val_parents.csv")}
    (grid/"grid_identity.json").write_text(json.dumps(identity))
    receipt = {"status":"complete","formal_evidence":True,"protocol":NAME,"phase":"grid",
               "artifacts":{name:sha256(grid/name) for name in ("selection.json","sampler_grid.csv","grid_identity.json")}}
    (grid/"receipt.json").write_text(json.dumps(receipt))
    output = tmp_path/"frozen"
    freeze_final_evaluation(output,grid_run=grid,source_runs=sources)
    manifest = verify_final_manifest(output,grid,identities[str(sources[1])])
    assert len(manifest["checkpoint_sources"]) == 3
    assert {v["sampler"] for v in manifest["configurations"]} == {"ddim","ddpm"}
    for value in manifest["configurations"]:
        assert len(value["timesteps"]) == value["nfe"]
    bad = {**identities[str(sources[1])],"checkpoint_sha256":"wrong"}
    with pytest.raises(ValueError,match="清单"):
        verify_final_manifest(output,grid,bad)
    with pytest.raises(ValueError,match="三个"):
        freeze_final_evaluation(tmp_path/"duplicate",grid_run=grid,source_runs=[sources[0]]*3)


def test_independent_runtime_is_single_configuration_and_preserves_model(tmp_path):
    cfg = config_contract("source_equivalent")
    dataset = fixture(tmp_path,cfg,"val")
    net = model().eval()
    before = state_digest(net)
    measured = independent_timing(net,dataset,[StudySamplerSpec("ddim",6,1)],batch_size=2,warmup=0,repeats=1)
    assert measured[0]["N"] == 1 and measured[0]["batch_size"] == 2
    assert measured[0]["independent_batch_sampling_seconds_median"] > 0
    assert measured[0]["peak_allocated_bytes"] == [None]
    assert before == state_digest(net)


@pytest.mark.parametrize("phase", ["diagnose","train","grid"])
def test_synthetic_stage_artifacts(tmp_path, phase):
    out = tmp_path/phase
    if phase == "diagnose":
        run_diagnosis(out,tiny=True)
        identity = json.loads((out/"diagnostic_identity.json").read_text())
        assert len(identity["selections"]["train_representative"]["dataset_row_ids"]) == 2
        assert (out/"val_representative/anchors/ddpm_sampling/reverse_trace_summary.csv").is_file()
    elif phase == "train":
        run_training(out,regime="long_absolute",tiny=True)
        state = torch.load(out/"final.pt",weights_only=False)
        assert state["update"] == 2 and state["selector"] == "final_update"
        assert "batch_stream" in state and state["training"]["synthetic"]
    else:
        run_grid(out,tiny=True)
        data = pd.read_csv(out/"sampler_grid.csv")
        assert set(zip(data.sampler,data.nfe_per_trajectory)) == set(GRID)
        assert len(data) == 10
    receipt = json.loads((out/"receipt.json").read_text())
    assert receipt["status"] == "complete" and not receipt["formal_evidence"]
    for name,digest in receipt["artifacts"].items():
        assert sha256(out/name) == digest
