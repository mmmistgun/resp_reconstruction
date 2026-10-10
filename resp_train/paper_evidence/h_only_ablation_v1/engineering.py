"""用户显式执行的原生 GPU/BF16 合成验收。"""
import torch
from resp_train.crd.training import build_crd_optimizer
from resp_train.losses.task import RespirationTaskLoss
from resp_train.paper_evidence.cwt_apor_v2.engineering import environment
from . import artifacts as io
from .spec import ROOT, config
from .runtime import require_device
from .model import build_model


def acceptance(session, arm, seed, device, retry=False):
    io.load_session(session)
    device = require_device(device)
    with io.mutex(ROOT / "runs/h_only_ablation_v1" / f".device_{device.index}.mutex"):
        with io.attempt(session / "acceptance" / arm / f"seed_{seed}",
                        io.binding(session, "acceptance", arm=arm, seed=seed), retry) as output:
            torch.manual_seed(seed)
            cfg = config(arm, seed, output / "unused", str(device))
            model = build_model(arm, seed).to(device).train()
            optimizer, _ = build_crd_optimizer(model, cfg)
            loss_fn = RespirationTaskLoss(cfg).to(device)
            batch_size = int(cfg.training.batch_size)
            x = torch.randn(batch_size, 1, 18000, device=device)
            times = torch.arange(18000, device=device) / 100
            target = ((1 + .3*torch.sin(2*torch.pi*.02*times))*torch.sin(2*torch.pi*.23*times))[None, None].repeat(batch_size, 1, 1)
            tf = {"w": torch.rand(batch_size, 41, 360, device=device)} if model.ablation.condition else None
            records = []
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(x, tf=tf)
                objective, _ = loss_fn(prediction, target)
                objective.backward()
                missing = [n for n, p in model.named_parameters() if p.grad is None]
                if missing:
                    raise RuntimeError(f"存在未参与 forward 的注册参数: {missing}")
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip_norm), error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("更新后参数非有限")
                records.append({"step": step, "loss": float(objective), "pre_clip_norm": float(norm)})
            io.write_json(output / "environment.json", environment(str(device)))
            io.write_json(output / "result.json", {"arm": arm, "seed": seed, "parameters": model.parameter_count,
                          "synthetic_only": True, "batch_size": batch_size, "native_mamba": arm != "HA4", "bf16": True, "steps": records})
    return output
