from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.projection_consistency.schema import CAMERAS, INSUFFICIENT_EVENT, mean, rate
from driveaudit_eval.research.consistency.projection_consistency.consistency_metrics import summarize_generic


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["agent_id"], int(row["frame_index"]))].append(row)
    contradictions, motion, direction, scale = [], [], [], []
    expected_counts, actual_counts = [], []
    for values in grouped.values():
        usable = [row for row in values if row.get("guided_support_state") != "unscorable"]
        expected_counts.append(len(values)); actual_counts.append(len(usable))
        if len(usable) < 2:
            continue
        states = {row.get("guided_support_state") for row in usable}
        contradictions.append("supported" in states and "contradicted" in states)
        for target, key in ((motion, "guided_flow_cosine"), (direction, "guided_motion_direction_agreement"), (scale, "guided_scale_change_agreement")):
            values_numeric = [float(row[key]) for row in usable if isinstance(row.get(key), (int, float))]
            if len(values_numeric) > 1:
                target.append(float(np.clip(1 - np.std(values_numeric), 0, 1)))
    per_camera = {}
    for camera in CAMERAS:
        camera_rows = [row for row in rows if row["camera_name"] == camera]
        metrics = summarize_generic(camera_rows)
        per_camera[camera] = {
            "sample_count": len(camera_rows), "scorable_coverage": metrics["scorable_coverage"],
            "valid_point_ratio": metrics["valid_point_ratio"], "residual_epe": metrics["residual_epe"],
            "flow_cosine": metrics["flow_cosine"], "direction_agreement": metrics["direction_agreement"],
            "strong_support_rate": metrics["strong_support_rate"], "unscorable_rate": metrics["unscorable_rate"],
            "contradicted_rate": metrics["contradicted_rate"],
            "primary_state": Counter(row.get("guided_support_state") for row in camera_rows).most_common(1)[0][0] if camera_rows else "no_samples",
        }
    return {
        "expected_camera_support_count": mean(expected_counts),
        "actual_scorable_camera_count": mean(actual_counts),
        "camera_support_recall": sum(actual_counts) / max(1, sum(expected_counts)),
        "cross_view_motion_consistency": mean(motion),
        "cross_view_direction_consistency": mean(direction),
        "cross_view_scale_consistency": mean(scale),
        "multiview_contradiction_rate": rate(contradictions),
        "camera_handoff_timing_error": INSUFFICIENT_EVENT,
        "per_camera": per_camera,
        "agent_identity_source": "input_trajectory_agent_id",
        "uses_reid_or_hungarian": False,
    }

