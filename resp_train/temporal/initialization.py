from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager

import torch


def named_subseed(base_seed: int, name: str) -> int:
    """为 RTM-v1 模块派生稳定且彼此隔离的 CPU 子 seed。"""

    digest = hashlib.sha256(f"resp-temporal-v1:{int(base_seed)}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


@contextmanager
def module_seed(base_seed: int, name: str) -> Iterator[None]:
    """隔离 trunk 初始化，保证同 seed 的公共 stem/decoder 逐 tensor 一致。"""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(named_subseed(base_seed, name))
        yield
