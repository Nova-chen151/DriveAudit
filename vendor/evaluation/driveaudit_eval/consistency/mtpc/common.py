from __future__ import annotations

from driveaudit_eval.utils.io import numpy_json_default as json_default
from driveaudit_eval.utils.io import write_ordered_csv as write_csv

from driveaudit_eval.consistency.config import load_config as read_config

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

from driveaudit_eval.agents.core import quat_rotation
from driveaudit_eval.consistency.shared.common import json_cell, read_csv
from driveaudit_eval.consistency.shared.visibility import _box_corners_3d

from driveaudit_eval.consistency.mtpc import NOT_AVAILABLE

CAMERAS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def load_config(path: str | Path) -> dict[str, Any]:
    return read_config(path)


def load_pair(config: dict[str, Any]) -> dict[str, Any]:
    manifest = load_config(config["pairing"]["manifest"])
    matches = [row for row in manifest["pairs"] if row["pair_id"] == config["pairing"]["pair_id"]]
    if len(matches) != 1: raise ValueError("MTPC pair_id must resolve exactly once")
    return matches[0]


def image_path(pair: dict[str, Any], camera: str, frame: int) -> Path:
    root = pair["image_sequence"]["history_root"] if frame < int(pair["timeline"]["image_history_frames"]) else pair["image_sequence"]["generated_future_root"]
    return Path(root) / camera / f"{frame:06d}.jpg"


def image_fingerprint(pair: dict[str, Any], frames: Iterable[int] = range(90)) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        for camera in CAMERAS:
            path = image_path(pair, camera, frame); digest.update(str(path).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def project_world_box(box: dict[str, Any], calibration: dict[str, Any], image_size: tuple[int, int], minimum_depth: float = .2) -> dict[str, Any]:
    corners = _box_corners_3d(box)
    if not len(corners): return {"projected_valid": False, "reason": "box_height_not_available", "area": 0.0}
    world_rotation = quat_rotation(calibration["ego2global_rotation"])
    ego = (corners - np.asarray(calibration["ego2global_translation"], dtype=float)) @ world_rotation
    sensor_rotation = quat_rotation(calibration["sensor2ego_rotation"])
    camera = (ego - np.asarray(calibration["sensor2ego_translation"], dtype=float)) @ sensor_rotation
    valid = camera[:, 2] > minimum_depth
    if valid.sum() < 2: return {"projected_valid": False, "reason": "behind_camera", "area": 0.0}
    intrinsic = np.asarray(calibration["camera_intrinsics"], dtype=float); points = camera[valid]
    uvw = points @ intrinsic.T; uv = uvw[:, :2] / uvw[:, 2:3]
    width, height = image_size; uv[:, 0] *= width / float(calibration.get("image_width", max(2.0 * intrinsic[0, 2], width))); uv[:, 1] *= height / float(calibration.get("image_height", max(2.0 * intrinsic[1, 2], height)))
    polygon = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
    raw = [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]
    clipped = [max(0.0, raw[0]), max(0.0, raw[1]), min(float(width - 1), raw[2]), min(float(height - 1), raw[3])]
    area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
    return {"projected_valid": area > 0, "reason": "visible" if area > 0 else "outside_image", "box_2d": raw,
            "clipped_box_2d": clipped, "projected_polygon": polygon.tolist(), "projected_corners": uv.tolist(),
            "center_2d": [float((clipped[0] + clipped[2]) / 2), float((clipped[1] + clipped[3]) / 2)],
            "area": area, "truncated": raw != clipped, "mean_depth": float(points[:, 2].mean())}


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    a, b = list(map(float, a)), list(map(float, b))
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa, bb = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]), max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def polygon_iou(a: Iterable[Iterable[float]], b: Iterable[Iterable[float]]) -> float:
    pa, pb = np.asarray(list(a), np.float32), np.asarray(list(b), np.float32)
    if len(pa) < 3 or len(pb) < 3: return 0.0
    pa, pb = cv2.convexHull(pa), cv2.convexHull(pb)
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    aa, bb = cv2.contourArea(pa), cv2.contourArea(pb)
    return float(inter / (aa + bb - inter)) if aa + bb - inter > 0 else 0.0


def bbox_polygon(box: Iterable[float]) -> list[list[float]]:
    x1, y1, x2, y2 = map(float, box); return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def clipped_box(box: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [max(0.0, min(width - 1.0, x1)), max(0.0, min(height - 1.0, y1)),
            max(0.0, min(width - 1.0, x2)), max(0.0, min(height - 1.0, y2))]


def appearance_histogram(image: np.ndarray, box: Iterable[float]) -> list[float]:
    h, w = image.shape[:2]; x1, y1, x2, y2 = map(int, clipped_box(box, w, h))
    if x2 <= x1 or y2 <= y1: return [0.0] * 64
    hsv = cv2.cvtColor(image[y1:y2 + 1, x1:x2 + 1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]); cv2.normalize(hist, hist)
    return hist.reshape(-1).astype(float).tolist()


def histogram_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a, b = np.asarray(list(a), np.float32), np.asarray(list(b), np.float32)
    if not np.any(a) or not np.any(b): return 0.0
    return float(max(0.0, 1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)))


from driveaudit_eval.utils.aggregation import mean_or_na as mean


def score_from_error(value: Any, scale: float) -> float | None:
    return max(0.0, 100.0 * (1.0 - float(value) / scale)) if isinstance(value, (int, float)) else None


def grouped(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    output: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: output[tuple(row[key] for key in keys)].append(row)
    return output


def load_geometry(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    frames = read_csv(root / "frames.csv")
    timestamps = {int(row["frame_index"]): float(row["timestamp"]) / 1e6 for row in frames}
    poses = []
    for row in read_csv(root / "ego_pose.csv"):
        frame = int(row["frame_index"])
        poses.append({"frame_index": frame, "timestamp": timestamps[frame],
                      "translation": json_cell(row["ego2global_translation"]),
                      "rotation": json_cell(row["ego2global_rotation"]),
                      "lidar2ego_translation": json_cell(row["lidar2ego_translation"]),
                      "lidar2ego_rotation": json_cell(row["lidar2ego_rotation"])})
    calibrations = []
    for row in read_csv(root / "camera_calibration.csv"):
        parsed = dict(row)
        parsed["frame_index"] = int(row["frame_index"])
        parsed["camera_name"] = row["camera"]
        for field in ("sensor2ego_translation", "sensor2ego_rotation", "ego2global_translation",
                      "ego2global_rotation", "sensor2lidar_rotation", "sensor2lidar_translation",
                      "camera_intrinsics"):
            parsed[field] = json_cell(row[field])
        calibrations.append(parsed)
    return {"frames": frames, "timestamps": timestamps, "poses": poses, "calibrations": calibrations}

