from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np


def polygon_mask(polygon: list[list[float]], size: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), np.uint8)
    if len(polygon) >= 3: cv2.fillConvexPoly(mask, np.round(np.asarray(polygon, float)).astype(np.int32), 1)
    return mask


def apply_visibility(rows: list[dict[str, Any]], config: dict[str, Any], size: tuple[int, int]) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[(int(row["frame_index"]), row["camera_name"])].append(row)
    for items in grouped.values():
        occupied = np.zeros((size[1], size[0]), np.uint8)
        for order, row in enumerate(sorted(items, key=lambda value: float(value.get("projected_depth", float("inf"))))):
            row["depth_order"] = order
            if not row.get("projected_valid"):
                row.update(occlusion_ratio=1.0, expected_visible=False, visibility_reason=row.get("projection_reason", "invalid_projection")); continue
            mask = polygon_mask(row["projected_polygon"], size); pixels = int(mask.sum())
            overlap = int(np.logical_and(mask, occupied).sum()); row["occlusion_ratio"] = overlap / max(1, pixels)
            occupied |= mask
            failures = []
            if row["projected_area"] < float(config["projection"]["minimum_projected_area_px"]): failures.append("small_object")
            if row["truncated_ratio"] > float(config["projection"]["maximum_truncation_ratio"]): failures.append("high_truncation")
            if row["occlusion_ratio"] > float(config["projection"]["maximum_occlusion_ratio"]): failures.append("high_occlusion")
            if int(row.get("visibility_token", 0)) < int(config["projection"]["minimum_visibility_token"]): failures.append("gt_visibility_low")
            row["small_object"] = "small_object" in failures; row["expected_visible"] = not failures
            row["visibility_reason"] = "expected_visible" if not failures else "+".join(failures)

