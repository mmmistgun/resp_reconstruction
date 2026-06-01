from __future__ import annotations

from pathlib import Path

import numpy as np


class WholeNightCache:
    def __init__(self, index_csv_path: str | Path):
        self.index_csv_path = Path(index_csv_path)
        self.base_dir = self.index_csv_path.parent
        self._cache: dict[Path, dict[str, np.ndarray]] = {}

    def resolve(self, source_npz: str) -> Path:
        path = (self.base_dir / source_npz).resolve()
        if not path.exists():
            raise FileNotFoundError(f"源 NPZ 不存在: {path}")
        return path

    def get_arrays(self, source_npz: str, keys: list[str]) -> dict[str, np.ndarray]:
        path = self.resolve(source_npz)
        if path not in self._cache:
            with np.load(path) as data:
                missing = [key for key in keys if key not in data.files]
                if missing:
                    raise KeyError(f"{path} 缺少数组: {missing}")
                self._cache[path] = {key: np.asarray(data[key]) for key in keys}

        # 同一个整晚文件可能被后续窗口请求新的数组键，按需补入缓存。
        cached = self._cache[path]
        missing_cached = [key for key in keys if key not in cached]
        if missing_cached:
            with np.load(path) as data:
                for key in missing_cached:
                    if key not in data.files:
                        raise KeyError(f"{path} 缺少数组: {key}")
                    cached[key] = np.asarray(data[key])
        return {key: cached[key] for key in keys}
