"""仅检查参考源码身份，不导入有顶层训练副作用的脚本。"""

import hashlib
from pathlib import Path

SOURCE_COMMIT = "3ff05545c34f67e1ad415e36e4daea80269880b0"
SOURCE_HASHES = {
    "model.py": "48d95640b8e0c7a0ebadfc1efd0975b2bca93e85fd003465ad487e4f357ed59d",
    "model_fft.py": "e46ada04c541cf0742656fad283a0455883bd7400700a8db48977cfa678b798f",
    "breathing_bidmc.py": "2e15c37f7faf732c2b9d195fcbc77a186b65638df4c199a0d10a6c61a67c0360",
    "breathing_bidmc_fft.py": "38b6191e163a37ca6b9dc3832cba7de3b495d007459c4915f3086f77ce07fec8",
}


def verify_source(root):
    records = {}
    for name, expected in SOURCE_HASHES.items():
        path = Path(root) / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"来源哈希不符: {path}")
        records[name] = {"path": str(path.resolve()), "sha256": digest}
    return {"source_commit": SOURCE_COMMIT, "files": records}
