from __future__ import annotations

from collections import defaultdict
from typing import Any

from driveaudit_eval.consistency.mtpc import NOT_AVAILABLE
from driveaudit_eval.consistency.mtpc.common import CAMERAS, mean


def compute_multiview_metrics(projections: list[dict[str, Any]], visual: list[dict[str, Any]], motion_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected, actual = defaultdict(set), defaultdict(set)
    for row in projections:
        if row["history_anchored"] and row["expected_visible"]: expected[(row["agent_id"], row["frame_index"])].add(row["camera_name"])
    for row in visual:
        if row["visible"]: actual[(row["agent_id"], row["frame_index"])].add(row["camera_name"])
    recalls, precisions, contradictions, rows = [], [], 0, []
    for key, cameras in expected.items():
        observed = actual.get(key, set()); overlap = cameras & observed
        recalls.append(len(overlap) / len(cameras)); precisions.append(len(overlap) / len(observed) if observed else 0.0)
        contradictions += int(bool(observed) and bool(cameras - observed))
        rows.append({"agent_id": key[0], "frame_index": key[1], "expected_camera_support_count": len(cameras),
                     "actual_camera_support_count": len(observed), "camera_support_recall": len(overlap) / len(cameras),
                     "camera_support_precision": len(overlap) / len(observed) if observed else 0.0})
    by_camera = {}
    for camera in CAMERAS:
        e = sum(camera in cameras for cameras in expected.values()); a = sum(camera in cameras and camera in actual.get(key, set()) for key, cameras in expected.items())
        by_camera[camera] = {"expected": e, "actual_supported": a, "support_rate": a / e if e else NOT_AVAILABLE}
    cross_motion = mean((row["motion_direction_agreement"] + 1) / 2 for row in motion_rows if isinstance(row.get("motion_direction_agreement"), (int, float)))
    return {"expected_camera_support_count": mean(len(value) for value in expected.values()),
            "actual_camera_support_count": mean(len(actual.get(key, set())) for key in expected),
            "camera_support_recall": mean(recalls), "camera_support_precision": mean(precisions),
            "cross_view_center_consistency": NOT_AVAILABLE, "cross_view_motion_direction_consistency": cross_motion,
            "camera_handoff_consistency": mean(recalls), "cross_view_appearance_consistency": mean(row.get("appearance_similarity") for row in visual if row["visible"]),
            "cross_view_entry_exit_synchronization": NOT_AVAILABLE, "multi_view_missing_contradiction": contradictions / max(1, len(expected)),
            "per_camera": by_camera, "per_frame": rows}
