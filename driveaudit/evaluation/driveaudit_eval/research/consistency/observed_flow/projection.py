from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.agents.core import quat_rotation
from driveaudit_eval.consistency.shared.visibility import _box_corners_3d

from driveaudit_eval.research.consistency.observed_flow.schema import write_csv, write_json


def project_box(box: dict[str, Any], calibration: dict[str, Any], target_size: tuple[int, int],
                native_size: tuple[int, int] = (1600, 900), minimum_depth: float = .2) -> dict[str, Any]:
    corners = _box_corners_3d(box); world_rotation = quat_rotation(calibration["ego2global_rotation"])
    ego = (corners - np.asarray(calibration["ego2global_translation"], float)) @ world_rotation
    sensor_rotation = quat_rotation(calibration["sensor2ego_rotation"]); camera = (ego - np.asarray(calibration["sensor2ego_translation"], float)) @ sensor_rotation
    valid_indices = np.flatnonzero(camera[:, 2] > minimum_depth)
    if len(valid_indices) < 3: return {"projected_valid": False, "projection_reason": "behind_camera", "projected_area": 0.0}
    intrinsic = np.asarray(calibration["camera_intrinsics"], float); points = camera[valid_indices]
    uvw = points @ intrinsic.T; uv = uvw[:, :2] / uvw[:, 2:3]
    width, height = target_size; uv *= np.asarray([width / native_size[0], height / native_size[1]])
    raw_polygon = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
    image_polygon = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    clipped_area, clipped_polygon = cv2.intersectConvexConvex(raw_polygon.astype(np.float32), image_polygon)
    raw_area = float(cv2.contourArea(raw_polygon)); clipped_polygon = clipped_polygon.reshape(-1, 2) if clipped_polygon is not None else np.empty((0, 2))
    if clipped_area <= 0 or len(clipped_polygon) < 3: return {"projected_valid": False, "projection_reason": "outside_image", "projected_area": 0.0}
    raw_box = [float(raw_polygon[:, 0].min()), float(raw_polygon[:, 1].min()), float(raw_polygon[:, 0].max()), float(raw_polygon[:, 1].max())]
    clipped_box = [float(clipped_polygon[:, 0].min()), float(clipped_polygon[:, 1].min()), float(clipped_polygon[:, 0].max()), float(clipped_polygon[:, 1].max())]
    center_world = np.asarray([[box["x"], box["y"], box["z"]]], float)
    center_ego = (center_world - np.asarray(calibration["ego2global_translation"], float)) @ world_rotation
    center_camera = (center_ego - np.asarray(calibration["sensor2ego_translation"], float)) @ sensor_rotation
    center_uvw = center_camera @ intrinsic.T; center_uv = center_uvw[0, :2] / center_uvw[0, 2]
    center_uv *= np.asarray([width / native_size[0], height / native_size[1]])
    return {"projected_valid": True, "projection_reason": "visible", "in_front_of_camera": True,
            "intersects_image": True, "projected_polygon": clipped_polygon.astype(float).tolist(),
            "raw_projected_polygon": raw_polygon.astype(float).tolist(), "projected_box_2d": clipped_box,
            "raw_projected_box_2d": raw_box, "projected_center": center_uv.astype(float).tolist(),
            "projected_area": float(clipped_area), "raw_projected_area": raw_area,
            "projected_depth": float(center_camera[0, 2]), "truncated_ratio": max(0.0, 1 - float(clipped_area) / max(raw_area, 1e-6)),
            "truncated": float(clipped_area) + 1e-3 < raw_area, "projected_corners": uv.astype(float).tolist(),
            "projected_corner_indices": valid_indices.tolist(), "camera_corners": points.astype(float).tolist()}


def project(aligned: dict[str, Any], config: dict[str, Any], output: str | Path,
            image_size: tuple[int, int] | None = None) -> list[dict[str, Any]]:
    target = image_size or tuple(config["image"]["formal_size"]); native = tuple(config["image"]["native_size"])
    rows = []
    for sample in aligned["rows"]:
        value = project_box(sample["box_reference"], sample["calibration"], target, native, float(config["projection"]["minimum_depth_m"]))
        rows.append({key: value for key, value in {**sample, **value}.items() if key != "calibration"})
    from driveaudit_eval.research.consistency.observed_flow.visibility import apply_visibility
    apply_visibility(rows, config, target)
    output = Path(output); write_csv(output / "gt_projection/per_camera_projection.csv", _flat(rows))
    write_json(output / "gt_projection/projections.json", rows)
    return rows


def _flat(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = {"box_reference", "raw_projected_polygon", "projected_polygon", "projected_corners", "camera_corners"}
    return [{key: value for key, value in row.items() if key not in excluded} for row in rows]

