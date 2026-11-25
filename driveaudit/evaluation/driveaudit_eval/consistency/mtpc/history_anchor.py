from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.consistency.shared.visibility import rectangle_occlusion

from driveaudit_eval.consistency.mtpc.common import CAMERAS, bbox_polygon, project_world_box


def sample_mask_points(gray: np.ndarray, polygon: list[list[float]], maximum_points: int = 24,
                       quality: float = .01, minimum_distance: float = 4.0) -> np.ndarray:
    mask = np.zeros(gray.shape, np.uint8); points = np.asarray(polygon, np.int32)
    if len(points) >= 3: cv2.fillConvexPoly(mask, points, 255)
    found = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=maximum_points, qualityLevel=quality, minDistance=minimum_distance)
    if found is None:
        x, y, w, h = cv2.boundingRect(points)
        found = np.asarray([[[x + w * fx, y + h * fy]] for fy in (.3, .5, .7) for fx in (.3, .5, .7)], np.float32)
    return found.reshape(-1, 2).astype(np.float32)


def build_history_projections(history_tracks: list[dict[str, Any]], calibrations: list[dict[str, Any]],
                              image_size: tuple[int, int], minimum_area: float, maximum_occlusion: float) -> list[dict[str, Any]]:
    calibration_map = {(int(row["frame_index"]), row["camera_name"]): row for row in calibrations}
    rows = []
    for track in history_tracks:
        for sample in track["observations"]:
            for camera in CAMERAS:
                value = project_world_box(sample["box_bev"], calibration_map[(int(sample["frame_index"]), camera)], image_size)
                if value["projected_valid"]:
                    rows.append({"agent_id": track["agent_id"], "agent_type": track["agent_type"], "category": track["category"],
                                 "frame_index": int(sample["frame_index"]), "timestamp": float(sample["timestamp"]),
                                 "camera_name": camera, **value})
    by_frame_camera: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: by_frame_camera[(row["frame_index"], row["camera_name"])].append(row)
    for items in by_frame_camera.values():
        nearer = []
        for row in sorted(items, key=lambda value: value["mean_depth"]):
            row["occlusion_ratio"] = rectangle_occlusion(row["clipped_box_2d"], nearer); nearer.append(row["clipped_box_2d"])
            row["expected_visible"] = row["area"] >= minimum_area and row["occlusion_ratio"] <= maximum_occlusion
            row["mask_polygon"] = bbox_polygon(row["clipped_box_2d"])
    return rows
