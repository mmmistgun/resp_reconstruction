"""开发阶段的更新、完整父窗口评价和checkpoint工具；正式预算尚未冻结。"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from resp_train.metrics.final_evaluation import evaluate_window, select_nonoverlap_centers, summarize_windows
from resp_train.metrics.task import validation_local_rr_mean
from .data import array, chunk_manifest, restore_parent, validate_rows
from .diffusion import RespDiff
from .model import RespDiffSpec, finite


def train_step(model, optimizer, condition, target, *, generator):
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("optimizer必须精确覆盖所有活跃参数且不重复")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = model.training_loss(condition, target, generator=generator)
    losses["loss"].backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ValueError(f"活跃参数缺少gradient: {name}")
        finite(f"gradient:{name}", parameter.grad)
    optimizer.step()
    for name, value in model.state_dict().items():
        finite(f"state:{name}", value)
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                finite(f"optimizer:{name}", value)
    return {key: float(value.detach()) for key, value in losses.items()}


@dataclass
class LocalRRSelector:
    best: float = float("inf")
    best_update: int | None = None
    last_update: int = 0

    def consider(self, *, update, local_rr, observed_ids, expected_ids):
        if list(observed_ids) != list(expected_ids) or not expected_ids or len(set(expected_ids)) != len(expected_ids):
            raise ValueError("selector要求完整、唯一且同顺序的validation父row")
        if type(update) is not int or update <= self.last_update or not np.isfinite(local_rr) or local_rr < 0:
            raise ValueError("selector要求递增update与有限非负Local RR")
        self.last_update = update
        improved = local_rr < self.best
        if improved:
            self.best, self.best_update = float(local_rr), update
        return improved


def select_validation_checkpoint(evaluated, references, *, metric_config, selector, update, expected_ids):
    """复用项目原Local RR；与报告用的最终五指标保持独立命名。"""
    observed = [record["dataset_row_id"] for record in evaluated["records"]]
    if observed != list(expected_ids):
        raise ValueError("Local RR要求完整且有序的validation父row")
    score = validation_local_rr_mean({"r_tho_hat": evaluated["waveform"], "tho_ref": references}, metric_config)
    improved = selector.consider(update=update, local_rr=score, observed_ids=observed, expected_ids=expected_ids)
    return {"selector_local_rr_mae": score, "improved": improved, "best_update": selector.best_update}


def evaluate_chunk_predictions(predictions, manifest, rows, references):
    """仅消费完整生成结果，按父row执行统一五指标；无模型/target重标定。"""
    validate_rows(rows)
    if set(rows.split) != {"val"}:
        raise ValueError("开发评价仅允许完整validation")
    expected = chunk_manifest(rows)
    if not manifest.equals(expected):
        raise ValueError("预测chunk身份/顺序/覆盖与声明不符")
    predictions = array("完整chunk预测", predictions, (len(expected), 1, 150))
    references = array("原始评价参考", references, (len(rows), 18000))
    selected = select_nonoverlap_centers(rows)
    records, waveforms = [], []
    for index, row in enumerate(rows.itertuples(index=False)):
        prediction = restore_parent(predictions[index * 36:(index + 1) * 36], chunk_indices=range(36))
        metrics = evaluate_window(prediction, references[index], center_selected=bool(selected[index]))
        records.append({"dataset_row_id": int(row.dataset_row_id), **metrics})
        waveforms.append(prediction)
    return {"waveform": np.stack(waveforms), "records": records, "summary": summarize_windows(records)}


def save_checkpoint(path, model, *, update, metadata):
    path = Path(path)
    for name, value in model.state_dict().items():
        finite(f"checkpoint:{name}", value)
    payload = {"schema": "respdiff-dev-v1", "spec": asdict(model.spec), "update": update,
               "metadata": metadata, "state_dict": model.state_dict()}
    # 排他创建，部分写入也保留；失败由外层attempt记failure receipt。
    with path.open("xb") as stream:
        torch.save(payload, stream)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "respdiff-dev-v1":
        raise ValueError("未知checkpoint schema")
    for name, value in payload["state_dict"].items():
        finite(f"checkpoint:{name}", value)
    model = RespDiff(RespDiffSpec(**payload["spec"]))
    if not torch.equal(payload["state_dict"]["alpha_torch"], model.alpha_torch):
        raise ValueError("checkpoint扩散调度与来源合同不符")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload
