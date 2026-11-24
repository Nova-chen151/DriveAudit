from __future__ import annotations

from collections import defaultdict
from typing import Any

from driveaudit_eval.research.consistency.trajectory_verification import NOT_AVAILABLE
from driveaudit_eval.research.consistency.trajectory_verification.common import CAMERAS, mean


def multiview_metrics(evidence: list[dict[str, Any]], primary_window: int) -> dict[str, Any]:
    rows = [row for row in evidence if int(row["window_length"]) == int(primary_window)]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["agent_id"], int(row["frame_index"]))].append(row)
    frame_rows, contradictions, simultaneous = [], [], []
    for (agent, frame), items in grouped.items():
        expected = {row["camera_name"] for row in items}
        actual = {row["camera_name"] for row in items if row["visual_supported"]}
        contradiction = bool(actual and expected - actual)
        contradictions.append(float(contradiction)); simultaneous.append(float(len(actual) >= 2 if len(expected) >= 2 else bool(actual)))
        frame_rows.append({"agent_id": agent, "frame_index": frame,
                           "expected_camera_support_count": len(expected), "actual_camera_support_count": len(actual),
                           "camera_support_recall": len(actual) / max(1, len(expected)),
                           "camera_support_precision": len(actual) / max(1, len(actual)),
                           "multi_view_missing_contradiction": contradiction,
                           "cross_view_motion_agreement": mean(row.get("flow_cosine_similarity") for row in items),
                           "cross_view_feature_similarity": mean(row.get("feature_support_score") for row in items)})
    per_camera = {}
    for camera in CAMERAS:
        items = [row for row in rows if row["camera_name"] == camera]
        per_camera[camera] = {"samples": len(items),
                              "support_rate": mean(float(row["visual_supported"]) for row in items),
                              "flow_cosine": mean(row.get("flow_cosine_similarity") for row in items),
                              "feature_support": mean(row.get("feature_support_score") for row in items)}
    return {"expected_camera_support_count": mean(row["expected_camera_support_count"] for row in frame_rows),
            "actual_camera_support_count": mean(row["actual_camera_support_count"] for row in frame_rows),
            "camera_support_recall": mean(row["camera_support_recall"] for row in frame_rows),
            "camera_support_precision": mean(row["camera_support_precision"] for row in frame_rows),
            "simultaneous_multiview_support": mean(simultaneous),
            "cross_view_motion_agreement": mean(row["cross_view_motion_agreement"] for row in frame_rows),
            "cross_view_feature_similarity": mean(row["cross_view_feature_similarity"] for row in frame_rows),
            "camera_handoff_timing_error_s": NOT_AVAILABLE,
            "camera_handoff_direction_agreement": NOT_AVAILABLE,
            "multi_view_missing_contradiction": mean(contradictions), "per_camera": per_camera,
            "per_frame": frame_rows}
