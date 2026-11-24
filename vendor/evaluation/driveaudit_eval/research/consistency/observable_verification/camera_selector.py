from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def select(rows: list[dict[str, Any]], reliability: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = config["camera_selection"]; weights = _weights(reliability)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[(row["agent_id"], int(row["frame_index"]))].append(row)
    chosen, diagnostics = [], []
    for (agent, frame), candidates in grouped.items():
        ranked = []
        for row in candidates:
            reliability_weight = float(np.clip(weights.get(row["camera_name"], 1.0), cfg["minimum_reliability"], cfg["maximum_reliability"]))
            visibility = max(0.0, (1-float(row.get("occlusion_ratio", 0))) * (1-float(row.get("truncated_ratio", 0))))
            area_quality = min(1.0, float(row.get("projected_area", 0)) / 512.0)
            priority = float(row["motion_observability_score"]) * reliability_weight * visibility * area_quality
            ranked.append(({**row, "camera_observability_priority": priority, "camera_reliability": reliability_weight}, priority))
        ranked.sort(key=lambda item: item[1], reverse=True)
        selected_names = [item[0]["camera_name"] for item in ranked[:int(cfg["top_k"])]]
        chosen.extend({**item[0], "selected_primary_camera": True, "camera_rank": rank + 1,
                       "camera_selection_used_future_score": False} for rank, item in enumerate(ranked[:int(cfg["top_k"])]))
        diagnostics.append({"agent_id": agent, "frame_index": frame, "selected_primary_cameras": selected_names,
                            "all_camera_observability": {r[0]["camera_name"]: r[0]["motion_observability_score"] for r in ranked},
                            "camera_selection_reason": "top_k_geometry_history_reliability",
                            "camera_reliability": {r[0]["camera_name"]: r[0]["camera_reliability"] for r in ranked},
                            "unselected_reason": {r[0]["camera_name"]: "lower_frozen_priority" for r in ranked[int(cfg["top_k"]):]},
                            "future_rgb_read": False, "future_score_used": False})
    return chosen, diagnostics


def _weights(value: dict[str, Any]) -> dict[str, float]:
    source = value.get("camera_reliability_weight", value.get("weights", value))
    result = {}
    for camera, item in source.items() if isinstance(source, dict) else []:
        if isinstance(item, dict): item = item.get("camera_reliability_weight", item.get("weight", 1.0))
        if isinstance(item, (int, float)): result[camera] = float(item)
    return result
