from __future__ import annotations

from driveaudit_eval.utils.io import numpy_json_default as _json_default
from driveaudit_eval.utils.io import write_ordered_csv as write_csv

from driveaudit_eval.consistency.config import load_config as read_config

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    return read_config(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def file_hash(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for value in sorted(paths, key=str):
        if isinstance(value, dict):
            digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        else:
            path = Path(value)
            digest.update(str(path).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


from driveaudit_eval.utils.aggregation import mean


def cosine(a: Iterable[float], b: Iterable[float], eps: float = 1e-6) -> float:
    a, b = np.asarray(list(a), float), np.asarray(list(b), float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps and nb < eps: return 1.0
    if na < eps or nb < eps: return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

