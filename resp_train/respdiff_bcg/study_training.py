"""固定更新训练与完整 batch 游标恢复；旧 checkpoint 不冒充可精确恢复。"""

import copy
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, default_collate

from resp_train.respdiff.model import finite
from .runtime import LOGGER, Progress, sha256, write_json
from .study_contract import NAME, SCHEMA
from .study_state import finite_tree, rng_state, restore_rng


class ResumableBatches:
    """预取当期索引而非 waveform，复用 PyTorch shuffle 的实际顺序。"""
    def __init__(self, size, batch_size, seed):
        if min(size, batch_size) <= 0:
            raise ValueError("训练集合和 batch_size 必须为正")
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.epoch, self.cursor, self.order = -1, 0, []

    def next(self):
        if self.cursor == len(self.order):
            # 先完整消费仅包含索引的 loader，保存本 epoch 顺序和下一 epoch 的 RNG。
            # 原 num_workers=0 loader 的 base-seed 与 RandomSampler 额外抽样也被保留。
            loader = DataLoader(range(self.size), batch_size=self.batch_size, shuffle=True,
                                drop_last=False, num_workers=0, generator=self.generator)
            self.order = [int(i) for batch in loader for i in batch.tolist()]
            self.cursor = 0
            self.epoch += 1
        end = min(self.cursor + self.batch_size, self.size)
        result = self.order[self.cursor:end]
        self.cursor = end
        return result

    def state_dict(self):
        return {"size": self.size, "batch_size": self.batch_size, "seed": self.seed,
                "epoch": self.epoch, "cursor": self.cursor, "order": self.order.copy(),
                "generator_state": self.generator.get_state().clone(),
                "ordering": "eager_index_only_pytorch_dataloader_v1"}

    def load_state_dict(self, state):
        if any(state.get(k) != getattr(self, k) for k in ("size", "batch_size", "seed")):
            raise ValueError("恢复时训练集合大小、batch 或 seed 改变")
        order, cursor, epoch = state["order"], state["cursor"], state["epoch"]
        empty = epoch == -1 and cursor == 0 and not order
        if not empty and (type(epoch) is not int or epoch < 0 or any(type(i) is not int for i in order)
                          or sorted(order) != list(range(self.size))
                          or type(cursor) is not int or not 0 <= cursor <= self.size
                          or (cursor != self.size and cursor % self.batch_size)):
            raise ValueError("恢复 batch 游标或排列损坏")
        if state.get("ordering") != "eager_index_only_pytorch_dataloader_v1":
            raise ValueError("不兼容的 shuffle 合同")
        self.order, self.cursor, self.epoch = order.copy(), cursor, epoch
        self.generator.set_state(state["generator_state"].cpu())


def training_data_digest(dataset):
    # 数据合同由父清单完整 CSV 身份约束；运行层另核验索引和来源文件身份。
    return hashlib.sha256(dataset.rows.to_csv(index=False).encode()).hexdigest()


def _save(path, state):
    finite_tree(state)
    with path.open("xb") as stream:
        torch.save(state, stream)
    write_json(path.with_suffix(".identity.json"), {"checkpoint_sha256": sha256(path),
               "schema": SCHEMA, "update": state["update"], "selector": state["selector"],
               "training": state["training"], "training_data_sha256": state["training_data_sha256"]})


def load_resume(path, *, training, data_digest):
    path = Path(path)
    identity_path = path.with_suffix(".identity.json")
    if not identity_path.is_file():
        raise ValueError("旧 checkpoint 没有完整游标身份，不能声称精确恢复；请在新 identity 从头训练")
    identity = json.loads(identity_path.read_text())
    if sha256(path) != identity["checkpoint_sha256"]:
        raise ValueError("resume checkpoint hash 不匹配")
    state = torch.load(path, map_location="cpu", weights_only=False)
    finite_tree(state)
    if (state.get("schema") != SCHEMA or state.get("training") != training
            or state.get("training_data_sha256") != data_digest or "batch_stream" not in state):
        raise ValueError("resume 的训练合同、数据或完整游标不匹配")
    if (state["update"] != identity["update"] or state["training"] != identity["training"]
            or state["training_data_sha256"] != identity["training_data_sha256"]):
        raise ValueError("resume 侧车与 checkpoint 内容不一致")
    return state


def train_with_snapshots(model, dataset, cfg, spec, output, *, callback=None, resume=None,
                         fixture_updates=None, fixture_batch_size=None, stop_after=None, cooldown=None):
    """正式总预算由 spec 决定；fixture 参数只由合成入口与定向测试传入。"""
    output = Path(output)
    device = next(model.parameters()).device
    updates = spec.updates if fixture_updates is None else fixture_updates
    batch_size = 64 if fixture_batch_size is None else fixture_batch_size
    snapshots = spec.snapshots if fixture_updates is None else tuple(range(updates + 1))
    identity = {**spec.identity(), "actual_updates": updates, "actual_batch_size": batch_size,
                "synthetic": fixture_updates is not None,
                "model_spec": vars(model.spec), "objective": model.objective}
    data_digest = training_data_digest(dataset)
    stream = ResumableBatches(len(dataset), batch_size, spec.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(spec.milestones), gamma=.1)
    start, resume_digest = 0, None
    if cooldown is not None:
        from .study_cooldown import CooldownFork
        if not isinstance(cooldown, CooldownFork) or resume is not None or stop_after is not None:
            raise ValueError("降档分支必须使用独立恢复合同，不能混用通用 resume 或提前结束")
        resume = cooldown.parent
    if resume is not None:
        state = load_resume(resume, training=identity, data_digest=data_digest)
        if cooldown is not None:
            cooldown.prepare(state, identity, data_digest, device)
            identity = {**identity, "cooldown": cooldown.identity()}
        start = int(state["update"])
        if not 0 <= start <= updates:
            raise ValueError("resume update 超出计划范围")
        resume_digest = sha256(resume)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        stream.load_state_dict(state["batch_stream"])
        restore_rng(state["rng"], device)
        if cooldown is not None:
            # Adam 动量与步数保持；仅将后续学习率及 scheduler 的对应记录降档。
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] = cooldown.learning_rate
            scheduler.base_lrs = [cooldown.learning_rate] * len(optimizer.param_groups)
            scheduler._last_lr = list(scheduler.base_lrs)
        LOGGER.info("精确恢复：update=%d epoch=%d cursor=%d", start, stream.epoch, stream.cursor)
    end = updates if stop_after is None else stop_after
    if not (start < end <= updates or start == end == updates):
        raise ValueError("执行区间非法")
    output.joinpath("checkpoints").mkdir(exist_ok=False)
    progress = Progress("study 训练 update", updates, unit="update")
    model.train()

    def snapshot(update):
        state = {"schema": SCHEMA, "protocol": NAME, "update": update,
                 "selector": "final_update" if update == updates else "diagnostic_update",
                 "training": identity, "training_data_sha256": data_digest,
                 "model_spec": vars(model.spec), "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "batch_stream": stream.state_dict(), "rng": rng_state(device),
                 "parent_checkpoint": None if resume is None else str(Path(resume).resolve()),
                 "parent_checkpoint_sha256": resume_digest}
        path = output / ("final.pt" if update == updates else f"checkpoints/update_{update:06d}.pt")
        _save(path, state)

    if start == 0:
        if callback is not None:
            callback(0)
        snapshot(0)
    elif start == updates:
        # 训练已完成但后续 monitor/anchor 失败时，在新 identity 补齐评价，优化器不再更新。
        if callback is not None:
            callback(updates)
        snapshot(updates)
    with (output / "history.jsonl").open("x") as history:
        for update in range(start + 1, end + 1):
            indices = stream.next()
            batch = default_collate([dataset[index] for index in indices])
            optimizer.zero_grad(set_to_none=True)
            condition, target = batch["x"].to(device), batch["target"].to(device)
            if cooldown is None:
                losses = model.training_loss(condition, target)
            else:
                losses = cooldown.loss(model, condition, target, update, indices)
            losses["loss"].backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    finite(f"gradient {name}", parameter.grad)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            for name, parameter in model.named_parameters():
                finite(name, parameter)
            record = {"update": update, "epoch_zero_based": stream.epoch, "lr": lr,
                      "chunk_indices": indices, **{key: float(v.detach()) for key, v in losses.items()},
                      **copy.deepcopy(model.last_diagnostics)}
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()
            progress.update(update, f"loss={record['loss']:.6g} lr={lr:.3g}")
            if update in snapshots or update == end:
                if callback is not None:
                    callback(update)
                snapshot(update)
    if cooldown is not None:
        cooldown.finish(rng_state(device), stream.state_dict())
    return {"start_update": start, "end_update": end, "final_update": updates,
            "complete": end == updates, "training": identity, "training_data_sha256": data_digest,
            "optimizer_updates_this_run": end-start, "parent_checkpoint_sha256": resume_digest}
