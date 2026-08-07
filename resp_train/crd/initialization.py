from __future__ import annotations

import hashlib
from contextlib import contextmanager
from collections.abc import Iterator

import torch


def named_subseed(base_seed: int, name: str) -> int:
    """从模型 seed 和模块名稳定派生独立子 seed。"""

    digest = hashlib.sha256(f"crd-v1.1:{int(base_seed)}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


@contextmanager
def module_seed(base_seed: int, name: str) -> Iterator[None]:
    """隔离 CPU RNG，使可选模块不会扰动共享模块初始化。"""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(named_subseed(base_seed, name))
        yield
