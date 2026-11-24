from __future__ import annotations

import numpy as np
from typing import Any

from driveaudit_eval.research.consistency.trajectory_verification import NOT_AVAILABLE
from driveaudit_eval.research.consistency.trajectory_verification.common import mean, weighted_mean


def local_motion_metrics(evidence: list[dict[str, Any]], primary_window: int,
                         camera_weights: dict[str, float] | None = None) -> dict[str, Any]:
    rows = [row for row in evidence if int(row["window_length"]) == int(primary_window) and "actual_flow" in row]
    weights = [float((camera_weights or {}).get(row["camera_name"], 1.0)) for row in rows]
    expected_moving = [float(row["expected_motion_magnitude"]) >= 0.5 for row in rows]
    actual_moving = [float(np.linalg.norm(row["actual_flow"])) >= 0.5 for row in rows]
    stop_move = mean(float(a == b) for a, b in zip(expected_moving, actual_moving))
    reversal = mean(float(row["flow_cosine_similarity"] < -0.25) for row in rows)
    return {"metric_type": "conditional_visual_verification",
            "local_flow_epe": mean(row["local_flow_epe"] for row in rows),
            "reliability_weighted_local_flow_epe": weighted_mean([row["local_flow_epe"] for row in rows], weights),
            "normalized_flow_epe": mean(row["normalized_flow_epe"] for row in rows),
            "flow_cosine_similarity": mean(row["flow_cosine_similarity"] for row in rows),
            "reliability_weighted_flow_cosine_similarity": weighted_mean([row["flow_cosine_similarity"] for row in rows], weights),
            "motion_direction_agreement": mean(row["motion_direction_agreement"] for row in rows),
            "motion_magnitude_relative_error": mean(row["motion_magnitude_relative_error"] for row in rows),
            "scale_change_error": mean(row["scale_change_error"] for row in rows),
            "rotation_change_error": NOT_AVAILABLE,
            "rotation_change_unavailable_reason": "classical local backend does not return a reliable instance orientation",
            "stop_move_accuracy": stop_move, "turning_direction_accuracy": NOT_AVAILABLE,
            "acceleration_trend_agreement": NOT_AVAILABLE, "reversal_error": reversal,
            "valid_flow_inlier_ratio": mean(row["ransac_inlier_ratio"] for row in rows),
            "valid_point_ratio": mean(row["valid_point_ratio"] for row in rows),
            "forward_backward_consistency_error": mean(row["forward_backward_error"] for row in rows),
            "valid_window_count": len(rows), "valid_flow_window_count": sum(bool(row.get("actual_flow")) for row in rows),
            "per_window": rows}
