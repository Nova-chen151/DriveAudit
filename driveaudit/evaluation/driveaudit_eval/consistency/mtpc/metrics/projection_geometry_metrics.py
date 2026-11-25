from __future__ import annotations

from typing import Any

import numpy as np

from driveaudit_eval.consistency.mtpc import NOT_AVAILABLE
from driveaudit_eval.consistency.mtpc.common import box_iou, mean, polygon_iou


def normalized_center_error(actual: list[float], expected: list[float], box: list[float]) -> float:
    diagonal = max(1e-6, float(np.hypot(box[2] - box[0], box[3] - box[1])))
    return float(np.linalg.norm(np.asarray(actual) - expected) / diagonal)


def compute_projection_geometry_metrics(projections: list[dict[str, Any]], visual: list[dict[str, Any]]) -> dict[str, Any]:
    actual_map = {(row["agent_id"], row["camera_name"], row["frame_index"]): row for row in visual if row["visible"]}
    rows = []
    for expected in projections:
        if not expected["expected_visible"] or not expected["history_anchored"]: continue
        actual = actual_map.get((expected["agent_id"], expected["camera_name"], expected["frame_index"]))
        if not actual: continue
        center_error = float(np.linalg.norm(np.asarray(actual["center_2d"]) - expected["center_2d"])); norm = normalized_center_error(actual["center_2d"], expected["center_2d"], expected["clipped_box_2d"])
        aw, ah = actual["bbox_2d"][2] - actual["bbox_2d"][0], actual["bbox_2d"][3] - actual["bbox_2d"][1]
        ew, eh = expected["clipped_box_2d"][2] - expected["clipped_box_2d"][0], expected["clipped_box_2d"][3] - expected["clipped_box_2d"][1]
        rows.append({"agent_id": expected["agent_id"], "camera_name": expected["camera_name"], "frame_index": expected["frame_index"],
                     "category": expected["category"], "depth_m": expected["mean_depth"], "projected_area_px": expected["area"],
                     "occlusion_ratio": expected["occlusion_ratio"], "projected_center_error_px": center_error,
                     "normalized_projected_center_error": norm, "box_iou_2d": box_iou(actual["bbox_2d"], expected["clipped_box_2d"]),
                     "mask_iou": polygon_iou(actual["mask_polygon"], expected["projected_polygon"]),
                     "polygon_iou": polygon_iou(actual["mask_polygon"], expected["projected_polygon"]),
                     "box_scale_error": abs(np.sqrt(max(1e-6, aw * ah) / max(1e-6, ew * eh)) - 1),
                     "aspect_ratio_error": abs(aw / max(ah, 1e-6) - ew / max(eh, 1e-6)),
                     "projected_area_ratio": aw * ah / max(1e-6, expected["area"]),
                     "projection_boundary_error_px": mean(abs(float(a) - float(b)) for a, b in zip(actual["bbox_2d"], expected["clipped_box_2d"]))})
    keys = ("projected_center_error_px", "normalized_projected_center_error", "box_iou_2d", "mask_iou", "polygon_iou",
            "box_scale_error", "aspect_ratio_error", "projected_area_ratio", "projection_boundary_error_px")
    return {**{key: mean(row[key] for row in rows) if rows else NOT_AVAILABLE for key in keys},
            "actual_expected_depth_order_consistency": NOT_AVAILABLE, "matched_projection_samples": len(rows), "per_frame": rows}

