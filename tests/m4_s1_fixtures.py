"""M4-S1定向测试的轻量CPU主干与线程边界。"""
import pytest
import torch
from torch import nn


class FixtureMamba(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.project = nn.Linear(d_model, d_model)

    def forward(self, value):
        return self.project(value)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(before)
