from __future__ import annotations

from driveaudit_eval.utils.io import numpy_json_default as json_default

import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from driveaudit_eval.agents.core import (
    WORLD_DREAMER_CAMERA_ORDER as CAMERA_ORDER,
    angle_delta,
    corners,
    quat_rotation,
    quat_to_yaw,
)

from driveaudit_eval.consistency.shared import NOT_AVAILABLE


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def json_cell(value: str | Sequence[float]) -> list[float]:
    return list(json.loads(value)) if isinstance(value, str) else list(value)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def write_pickle(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def true_cell(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def finite_or_na(values: Iterable[float]) -> float | str:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else NOT_AVAILABLE


def quaternion_transform(point: Sequence[float], translation: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    return quat_rotation(rotation) @ np.asarray(point, dtype=float) + np.asarray(translation, dtype=float)


def local_box_to_world(box: dict[str, Any], ego_pose: dict[str, Any]) -> dict[str, Any]:
    point = np.asarray([box["x"], box["y"], box.get("z", 0.0)], dtype=float)
    local_rotation = [1.0, 0.0, 0.0, 0.0]
    if "lidar2ego_translation" in ego_pose:
        local_rotation = ego_pose["lidar2ego_rotation"]
        point = quaternion_transform(point, ego_pose["lidar2ego_translation"], local_rotation)
    position = quaternion_transform(point, ego_pose["translation"], ego_pose["rotation"])
    yaw = float(box["yaw"]) + quat_to_yaw(local_rotation) + quat_to_yaw(ego_pose["rotation"])
    rotation = quat_rotation(ego_pose["rotation"]) @ quat_rotation(local_rotation)
    velocity = rotation[:2, :2] @ np.asarray([box.get("vx", 0.0), box.get("vy", 0.0)], dtype=float)
    result = dict(box)
    result.update(x=float(position[0]), y=float(position[1]), z=float(position[2]), yaw=wrap_angle(yaw),
                  vx=float(velocity[0]), vy=float(velocity[1]))
    return result


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2 * math.pi) - math.pi


def box_polygon(box: dict[str, Any]) -> np.ndarray:
    return corners(float(box["x"]), float(box["y"]), float(box["yaw"]),
                   float(box["length"]), float(box["width"]))


def rotated_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    from shapely.geometry import Polygon

    pa, pb = Polygon(box_polygon(a)), Polygon(box_polygon(b))
    union = pa.union(pb).area
    return float(pa.intersection(pb).area / union) if union > 0 else 0.0


def group_by(rows: Iterable[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


def percentile_summary(values: Iterable[float]) -> dict[str, float | str]:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {k: NOT_AVAILABLE for k in ("mean", "p50", "p90", "p95", "p99", "max")}
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def interpolate_xy(samples: list[dict[str, Any]], timestamps: np.ndarray) -> np.ndarray:
    valid = sorted((float(s["timestamp"]), float(s["x"]), float(s["y"])) for s in samples if s.get("valid", True))
    if not valid:
        return np.empty((0, 2))
    array = np.asarray(valid)
    return np.column_stack([np.interp(timestamps, array[:, 0], array[:, 1]),
                            np.interp(timestamps, array[:, 0], array[:, 2])])
