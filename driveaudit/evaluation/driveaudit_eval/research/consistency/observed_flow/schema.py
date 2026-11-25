from __future__ import annotations

from driveaudit_eval.utils.io import numpy_json_default as _json_default
from driveaudit_eval.utils.io import write_ordered_csv as write_csv

from driveaudit_eval.consistency.config import load_config as read_config

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from driveaudit_eval.consistency.mtpc.common import CAMERAS

from driveaudit_eval.research.consistency.observed_flow import NOT_AVAILABLE

VERSION = "gt_vtc_v1_oracle_paired"
PROTOCOL = "gt_vtc_paired_ground_truth"
DYNAMIC_CLASSES = {"car", "truck", "construction_vehicle", "bus", "trailer", "motorcycle", "bicycle", "pedestrian"}


def load_config(path: str | Path) -> dict[str, Any]:
    return read_config(path)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def finite_mean(values: Iterable[Any]) -> float | str:
    values = [float(value) for value in values if isinstance(value, (int, float)) and np.isfinite(float(value))]
    return float(np.mean(values)) if values else NOT_AVAILABLE


def rate(values: Iterable[Any]) -> float | str:
    values = list(values)
    return float(np.mean([bool(value) for value in values])) if values else NOT_AVAILABLE


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256(); path = Path(path)
    if path.is_file(): digest.update(path.read_bytes())
    else:
        for child in sorted(value for value in path.rglob("*") if value.is_file()):
            digest.update(str(child.relative_to(path)).encode()); digest.update(child.read_bytes())
    return digest.hexdigest()


def image_path(config: dict[str, Any], camera: str, frame: int) -> Path:
    return Path(config["inputs"]["gt_images"]) / camera / f"{frame:06d}.jpg"

