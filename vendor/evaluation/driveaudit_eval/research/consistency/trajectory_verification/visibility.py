from __future__ import annotations

from collections import defaultdict
from typing import Any

from driveaudit_eval.consistency.shared.visibility import rectangle_occlusion


def apply_visibility(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["frame_index"]), row["camera_name"])].append(row)
    for items in grouped.values():
        nearer: list[list[float]] = []
        for row in sorted(items, key=lambda value: float(value["projected_depth"])
                          if isinstance(value.get("projected_depth"), (int, float)) else float("inf")):
            if not row.get("projected_valid"):
                row.update(occlusion_ratio=1.0, expected_occluded=False, depth_order=None,
                           expected_visible=False, observability_status=row.get("projection_reason", "not_observable"))
                continue
            ratio = rectangle_occlusion(row["projected_box_2d"], nearer)
            row["occlusion_ratio"] = ratio
            row["expected_occluded"] = ratio > float(config["maximum_occlusion_ratio"])
            row["depth_order"] = len(nearer)
            nearer.append(row["projected_box_2d"])
            row["small_object"] = float(row["projected_area"]) < float(config["minimum_projected_area_px"])
            row["expected_visible"] = bool(row["in_front_of_camera"] and row["intersects_image"]
                                           and not row["small_object"]
                                           and float(row["truncated_ratio"]) <= float(config["maximum_truncation_ratio"])
                                           and not row["expected_occluded"])
            row["observability_status"] = "expected_visible" if row["expected_visible"] else "not_observable"
    by_agent_frame: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        if row.get("expected_visible"):
            by_agent_frame[(row["agent_id"], int(row["frame_index"]))] += 1
    for row in rows:
        row["expected_camera_count"] = by_agent_frame[(row["agent_id"], int(row["frame_index"]))]
    return rows
