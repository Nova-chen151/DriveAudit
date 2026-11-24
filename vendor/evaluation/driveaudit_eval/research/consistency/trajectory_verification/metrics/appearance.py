from __future__ import annotations

from typing import Any

from driveaudit_eval.research.consistency.trajectory_verification.common import mean


def appearance_metrics(evidence: list[dict[str, Any]], primary_window: int) -> dict[str, Any]:
    rows = [row for row in evidence if int(row["window_length"]) == int(primary_window) and "feature_support_score" in row]
    return {"metric_type": "conditional_visual_verification",
            "local_feature_support": mean(row["feature_support_score"] for row in rows),
            "foreground_contrast_support": mean(row["foreground_contrast_score"] for row in rows),
            "feature_match_count": mean(row["feature_matches"] for row in rows),
            "feature_inlier_count": mean(row["feature_inliers"] for row in rows)}
