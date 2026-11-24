from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.observed_flow.schema import NOT_AVAILABLE, finite_mean, write_json


def evaluate(projections: list[dict[str, Any]], metrics: list[dict[str, Any]], output: str | Path) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in projections:
        if row.get("projected_valid"): grouped[(row["agent_id"], row["camera_name"])].append(row)
    depth_delta, area_delta, center_acceleration = [], [], []
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
        centers = []
        for first, last in zip(rows, rows[1:]):
            if int(last["frame_index"]) != int(first["frame_index"]) + 1: continue
            depth_delta.append(abs(float(last["projected_depth"]) - float(first["projected_depth"])))
            area_delta.append(abs(np.log(max(float(last["projected_area"]), 1) / max(float(first["projected_area"]), 1))))
            centers.append(np.asarray(last["projected_center"], float) - np.asarray(first["projected_center"], float))
        center_acceleration.extend(float(np.linalg.norm(b - a)) for a, b in zip(centers, centers[1:]))
    result = {"gt_appearance_time_consistency": NOT_AVAILABLE, "gt_disappearance_time_consistency": NOT_AVAILABLE,
        "visibility_duration_consistency": NOT_AVAILABLE, "event_unavailable_reason": "not_available_no_independent_per_camera_visibility_event_gt",
        "stop_start_timing": NOT_AVAILABLE, "stop_start_reason": "not_available_insufficient_event_evidence",
        "motion_continuity_px": finite_mean(center_acceleration), "depth_continuity_m": finite_mean(depth_delta),
        "scale_continuity_log_area": finite_mean(area_delta), "yaw_continuity": "validated_in_annotation_consistency",
        "image_motion_continuity": finite_mean(row.get("gt_flow_epe") for row in metrics if row.get("window_scorable"))}
    write_json(Path(output) / "temporal_consistency/metrics.json", result); return result

