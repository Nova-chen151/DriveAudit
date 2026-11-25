from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from driveaudit_eval.video.utils import ensure_dir, read_image, stable_hash, write_json


class DiskCache:
    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        if self.enabled:
            ensure_dir(self.root)

    def path(self, namespace: str, key_parts: list[Any], suffix: str) -> Path:
        key = stable_hash(key_parts)
        return self.root / namespace / f"{key}.{suffix}"

    def load_npz(self, namespace: str, key_parts: list[Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self.path(namespace, key_parts, "npz")
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    def save_npz(self, namespace: str, key_parts: list[Any], **arrays: Any) -> Path:
        path = self.path(namespace, key_parts, "npz")
        if self.enabled:
            ensure_dir(path.parent)
            np.savez_compressed(path, **arrays)
        return path

    def save_json(self, namespace: str, key_parts: list[Any], data: Any) -> Path:
        path = self.path(namespace, key_parts, "json")
        if self.enabled:
            write_json(path, data)
        return path
