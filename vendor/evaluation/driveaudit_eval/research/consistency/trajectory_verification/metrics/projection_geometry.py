from __future__ import annotations

from typing import Any

from driveaudit_eval.research.consistency.trajectory_verification import NOT_AVAILABLE
from driveaudit_eval.research.consistency.trajectory_verification.common import mean, weighted_mean


def projection_geometry_metrics(evidence: list[dict[str, Any]], primary_window: int,
                                camera_weights: dict[str, float] | None = None) -> dict[str, Any]:
    rows = [row for row in evidence if int(row["window_length"]) == int(primary_window) and "normalized_center_error" in row]
    weights = [float((camera_weights or {}).get(row["camera_name"], 1.0)) for row in rows]
    return {"metric_type": "conditional_visual_verification",
            "local_projected_center_error_px": mean(row["flow_propagated_center_error"] for row in rows),
            "normalized_center_error": mean(row["normalized_center_error"] for row in rows),
            "reliability_weighted_normalized_center_error": weighted_mean([row["normalized_center_error"] for row in rows], weights),
            "box_iou": NOT_AVAILABLE, "polygon_iou": NOT_AVAILABLE, "mask_iou": NOT_AVAILABLE,
            "instance_region_unavailable_reason": "no reliable instance segmentation or local detector region; flow ROI is not an instance mask",
            "flow_propagated_region_iou_diagnostic": mean(row["flow_propagated_region_iou_diagnostic"] for row in rows),
            "area_ratio_error": NOT_AVAILABLE, "aspect_ratio_error": NOT_AVAILABLE,
            "boundary_alignment_error": NOT_AVAILABLE,
            "scale_change_agreement": mean(max(0.0, 1.0 - row["scale_change_error"]) for row in rows),
            "per_window": rows}
