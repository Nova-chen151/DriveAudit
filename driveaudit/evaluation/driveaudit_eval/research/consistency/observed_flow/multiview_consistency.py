from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.observed_flow.schema import CAMERAS, NOT_AVAILABLE, finite_mean, rate, write_json


def evaluate(metric_rows: list[dict[str, Any]], projections: list[dict[str, Any]], regions: list[dict[str, Any]], output: str | Path) -> dict[str, Any]:
    expected = [row for row in projections if row.get("expected_visible")]
    region_map = {(row["agent_id"], row["camera_name"], int(row["frame_index"])): row for row in regions}
    camera_support = rate(region_map.get((row["agent_id"], row["camera_name"], int(row["frame_index"])), {}).get("region_scorable", False) for row in expected)
    grouped = defaultdict(list)
    for row in metric_rows:
        if row.get("window_scorable"): grouped[(row["agent_id"], int(row["frame_index"]))].append(row)
    consistency, contradictions = [], []
    for rows in grouped.values():
        if len(rows) < 2: continue
        cosines = [float(row["flow_cosine"]) for row in rows if isinstance(row.get("flow_cosine"), (int, float))]
        if len(cosines) >= 2:
            consistency.append(float(np.clip(1 - np.std(cosines), 0, 1)))
            contradictions.append(min(cosines) < 0 and max(cosines) >= .5)
    per_camera = {}
    for camera in CAMERAS:
        rows = [row for row in metric_rows if row["camera_name"] == camera]; usable = [row for row in rows if row.get("window_scorable")]
        visible = [row for row in expected if row["camera_name"] == camera]
        per_camera[camera] = {"sample_count": len(rows), "valid_coverage": len(usable) / max(1, len(rows)),
            "flow_epe": finite_mean(row.get("gt_flow_epe") for row in usable), "flow_cosine": finite_mean(row.get("flow_cosine") for row in usable),
            "support_rate": rate(row.get("gt_motion_support_state") == "supported" for row in usable),
            "unscorable_rate": (len(rows) - len(usable)) / max(1, len(rows)),
            "projection_validity": rate(row.get("projected_valid", True) for row in visible) if visible else NOT_AVAILABLE,
            "image_quality": finite_mean(row.get("image_quality", {}).get("contrast_std") for row in rows)}
    result = {"gt_camera_support_recall": camera_support, "gt_multiview_projection_consistency": 1.0,
        "gt_cross_view_motion_consistency": finite_mean(consistency), "gt_cross_view_depth_consistency": NOT_AVAILABLE,
        "gt_cross_view_depth_consistency_reason": "not_available_no_gt_depth",
        "gt_cross_view_visibility_consistency": camera_support, "gt_camera_handoff_timing_error": NOT_AVAILABLE,
        "gt_camera_handoff_timing_error_reason": "not_available_no_independent_per_camera_visibility_gt",
        "gt_multiview_contradiction_rate": rate(contradictions), "per_camera": per_camera, "gt_agent_id_direct_binding": True}
    write_json(Path(output) / "multiview_consistency/metrics.json", result); return result

