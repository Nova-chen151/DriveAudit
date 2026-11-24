from __future__ import annotations

from typing import Any

import numpy as np

from driveaudit_eval.agents.core import quat_rotation

from driveaudit_eval.consistency.shared import NOT_AVAILABLE
from driveaudit_eval.consistency.shared.common import CAMERA_ORDER, box_polygon


def _box_corners_3d(box: dict[str, Any]) -> np.ndarray:
    if "height" not in box:
        return np.empty((0, 3))
    footprint = box_polygon(box)
    z = float(box.get("z", 0.0))
    return np.asarray([[x, y, z + dz] for dz in (-float(box["height"]) / 2, float(box["height"]) / 2)
                       for x, y in footprint], dtype=float)


def project_box(box: dict[str, Any], calibration: dict[str, Any], image_size: tuple[int, int],
                minimum_depth: float = 0.2) -> dict[str, Any]:
    corners = _box_corners_3d(box)
    if not len(corners):
        return {"projected_valid": False, "reason": "box_height_not_available", "area": NOT_AVAILABLE}
    world_rotation = quat_rotation(calibration["ego2global_rotation"])
    ego = (corners - np.asarray(calibration["ego2global_translation"], dtype=float)) @ world_rotation
    sensor_rotation = quat_rotation(calibration["sensor2ego_rotation"])
    camera = (ego - np.asarray(calibration["sensor2ego_translation"], dtype=float)) @ sensor_rotation
    in_front = camera[:, 2] > minimum_depth
    if not np.any(in_front):
        return {"projected_valid": False, "reason": "behind_camera", "area": 0.0}
    camera = camera[in_front]
    intrinsic = np.asarray(calibration["camera_intrinsics"], dtype=float)
    uvw = camera @ intrinsic.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    width, height = image_size
    nominal_width = max(2.0 * intrinsic[0, 2], width)
    nominal_height = max(2.0 * intrinsic[1, 2], height)
    uv[:, 0] *= width / nominal_width
    uv[:, 1] *= height / nominal_height
    raw = [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]
    clipped = [max(0.0, raw[0]), max(0.0, raw[1]), min(float(width), raw[2]), min(float(height), raw[3])]
    area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
    return {"projected_valid": area > 0, "reason": "visible" if area > 0 else "outside_image",
            "box_2d": raw, "clipped_box_2d": clipped, "area": area,
            "truncated": bool(raw[0] < 0 or raw[1] < 0 or raw[2] > width or raw[3] > height),
            "mean_depth": float(camera[:, 2].mean())}


def rectangle_occlusion(target: list[float], nearer: list[list[float]]) -> float:
    area = max(0.0, target[2] - target[0]) * max(0.0, target[3] - target[1])
    if area == 0:
        return 1.0
    overlap = 0.0
    for box in nearer:
        overlap += max(0.0, min(target[2], box[2]) - max(target[0], box[0])) * max(0.0, min(target[3], box[3]) - max(target[1], box[1]))
    return min(1.0, overlap / area)


def evaluate_frame_visibility(samples: list[dict[str, Any]], calibrations: list[dict[str, Any]],
                              config: dict[str, Any]) -> list[dict[str, Any]]:
    image_size = (int(config["image_width"]), int(config["image_height"]))
    projected: dict[str, list[tuple[int, dict[str, Any]]]] = {camera: [] for camera in CAMERA_ORDER}
    for sample_index, sample in enumerate(samples):
        for calibration in calibrations:
            camera = calibration["camera_name"]
            value = project_box(sample["box_bev"], calibration, image_size, float(config["minimum_depth_m"]))
            if value["projected_valid"]:
                projected[camera].append((sample_index, value))
    by_sample: list[dict[str, Any]] = [{"camera_details": {}} for _ in samples]
    for camera, items in projected.items():
        ordered = sorted(items, key=lambda item: item[1]["mean_depth"])
        nearer: list[list[float]] = []
        for sample_index, value in ordered:
            value["occlusion_ratio"] = rectangle_occlusion(value["clipped_box_2d"], nearer)
            nearer.append(value["clipped_box_2d"])
            by_sample[sample_index]["camera_details"][camera] = value
    results = []
    for sample, detail in zip(samples, by_sample):
        camera_details = detail["camera_details"]
        visible = [value for value in camera_details.values()
                   if value["area"] >= float(config["minimum_projected_area_px"])
                   and value["occlusion_ratio"] <= float(config["maximum_occlusion_ratio"])]
        all_areas = [value["area"] for value in camera_details.values()]
        results.append({"frame_index": sample["frame_index"], "timestamp": sample["timestamp"],
                        "agent_id": sample["agent_id"], "visible_in_any_camera": bool(camera_details),
                        "visible_camera_count": len(visible), "projected_box_area": max(all_areas, default=0.0),
                        "occlusion_ratio": min((value["occlusion_ratio"] for value in camera_details.values()), default=1.0),
                        "expected_visible": bool(visible),
                        "small_object_flag": bool(camera_details and max(all_areas, default=0.0) < float(config["minimum_projected_area_px"])),
                        "truncated_flag": any(value["truncated"] for value in camera_details.values()),
                        "camera_details": camera_details})
    return results
