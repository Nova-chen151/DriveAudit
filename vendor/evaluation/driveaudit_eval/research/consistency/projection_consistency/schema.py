from __future__ import annotations

from driveaudit_eval.consistency.config import load_config as read_config

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

CAMERAS = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
)
NOT_AVAILABLE = "not_available"
INSUFFICIENT_EVENT = "not_available_insufficient_event_evidence"


def load_config(path: str | Path) -> dict[str, Any]:
    return read_config(path)


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return NOT_AVAILABLE
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value: Any) -> Any:
    value = clean(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def sha256(path: str | Path) -> str:
    path = Path(path)
    if path.is_symlink() and path.is_dir():
        path = path.resolve()
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def numeric(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def mean(values: Iterable[Any]) -> float | str:
    values = numeric(values)
    return float(np.mean(values)) if values else NOT_AVAILABLE


def rate(values: Iterable[Any]) -> float | str:
    values = list(values)
    return float(np.mean([bool(value) for value in values])) if values else NOT_AVAILABLE


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
