from __future__ import annotations

from driveaudit_eval.utils.io import numpy_json_default as json_default
from driveaudit_eval.utils.io import write_ordered_csv as write_csv
from driveaudit_eval.consistency.mtpc.common import load_geometry

from driveaudit_eval.consistency.config import load_config as read_config

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

from driveaudit_eval.consistency.shared.common import json_cell, read_csv
from driveaudit_eval.consistency.mtpc.common import CAMERAS, image_path, project_world_box

from driveaudit_eval.research.consistency.trajectory_verification import NOT_AVAILABLE


def load_config(path: str | Path) -> dict[str, Any]:
    return read_config(path)


def load_pair(config: dict[str, Any]) -> dict[str, Any]:
    manifest = load_config(config["pairing"]["manifest"])
    rows = [row for row in manifest["pairs"] if row["pair_id"] == config["pairing"]["pair_id"]]
    if len(rows) != 1:
        raise ValueError("TCVV pair_id must resolve exactly once")
    return rows[0]


def read_image(pair: dict[str, Any], camera: str, frame: int, image_size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(image_path(pair, camera, frame)))
    if image is None:
        raise FileNotFoundError(f"missing TCVV image camera={camera} frame={frame}")
    if (image.shape[1], image.shape[0]) != image_size:
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    return image


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


from driveaudit_eval.utils.aggregation import finite


from driveaudit_eval.utils.aggregation import mean_or_na as mean


from driveaudit_eval.utils.aggregation import weighted_mean


def summary(values: Iterable[Any]) -> dict[str, float | str]:
    usable = finite(values)
    if not usable:
        return {key: NOT_AVAILABLE for key in ("mean", "p10", "p50", "p90")}
    return {"mean": float(np.mean(usable)), "p10": float(np.percentile(usable, 10)),
            "p50": float(np.percentile(usable, 50)), "p90": float(np.percentile(usable, 90))}


def groups(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row.get(key) for key in keys)].append(row)
    return result


def expand_box(box: Iterable[float], margin: float, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = map(float, box)
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return [max(0, int(math.floor(x1 - dx))), max(0, int(math.floor(y1 - dy))),
            min(width - 1, int(math.ceil(x2 + dx))), min(height - 1, int(math.ceil(y2 + dy)))]


def cosine(a: Iterable[float], b: Iterable[float], epsilon: float = 1e-6) -> float:
    aa, bb = np.asarray(list(a), float), np.asarray(list(b), float)
    na, nb = np.linalg.norm(aa), np.linalg.norm(bb)
    if na < epsilon and nb < epsilon:
        return 1.0
    if na < epsilon or nb < epsilon:
        return 0.0
    return float(np.clip(np.dot(aa, bb) / (na * nb), -1.0, 1.0))


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    a, b = list(map(float, a)), list(map(float, b))
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def score_error(value: Any, scale: float) -> float | None:
    return max(0.0, 100.0 * (1.0 - float(value) / scale)) if isinstance(value, (int, float)) else None

