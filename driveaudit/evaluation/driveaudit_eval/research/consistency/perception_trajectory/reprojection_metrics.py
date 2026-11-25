from __future__ import annotations

from collections import defaultdict
from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE
from driveaudit_eval.consistency.shared.common import CAMERA_ORDER


def compute_reprojection_metrics(visibility_rows: list[dict[str, Any]], image_tracks: list[dict[str, Any]]) -> dict[str, Any]:
    totals, valid = defaultdict(int), defaultdict(int)
    for row in visibility_rows:
        for camera in CAMERA_ORDER:
            totals[camera] += 1
            valid[camera] += int(camera in row.get("camera_details", {}))
    per_camera = {}
    for camera in CAMERA_ORDER:
        per_camera[camera] = {"projected_box_valid_ratio": valid[camera] / totals[camera] if totals[camera] else NOT_AVAILABLE,
                              "matched_2d_detection_ratio": NOT_AVAILABLE, "box_2d_iou": NOT_AVAILABLE,
                              "center_2d_error": NOT_AVAILABLE, "box_scale_error": NOT_AVAILABLE}
    support = [len(row.get("camera_support", [])) for track in image_tracks for row in track["observations"] if row.get("source") == "image_detection"]
    return {"metric_type": "camera_projection_diagnostic", "coordinate_domain": "image_pixels_not_bev_world",
            "per_camera": per_camera,
            "visible_camera_support_count": sum(support) / len(support) if support else NOT_AVAILABLE,
            "cross_view_support_consistency": NOT_AVAILABLE,
            "front_camera_consistency": NOT_AVAILABLE, "side_camera_consistency": NOT_AVAILABLE,
            "rear_camera_consistency": NOT_AVAILABLE,
            "low_support_frame_ratio": sum(value < 2 for value in support) / len(support) if support else NOT_AVAILABLE,
            "diagnostic_limitation": "reference metadata has no independent per-camera 2D detections"}
