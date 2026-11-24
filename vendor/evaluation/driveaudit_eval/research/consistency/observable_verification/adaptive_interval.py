from __future__ import annotations

from collections import defaultdict
from typing import Any


def select(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["agent_id"], row["camera_name"], int(row["frame_index"]))].append(row)
    selected, diagnostics = [], []
    for key, candidates in grouped.items():
        candidates.sort(key=lambda row: int(row["delta_frames"]))
        valid = [row for row in candidates if row["geometry_observability_status"] == "geometry_observable"]
        choice = valid[0] if valid else None
        diagnostics.append({"agent_id": key[0], "camera_name": key[1], "frame_index": key[2],
                            "selected_delta_frames": choice["delta_frames"] if choice else None,
                            "selected_delta_seconds": (choice["target_timestamp"] - choice["timestamp"]) if choice else None,
                            "candidate_observability_scores": {str(r["delta_frames"]): r["motion_observability_score"] for r in candidates},
                            "selection_reason": "shortest_geometry_observable" if choice else "geometry_unobservable_no_valid_interval",
                            "rejected_interval_reasons": {str(r["delta_frames"]): r["geometry_observability_status"] for r in candidates if r is not choice},
                            "future_rgb_read": False, "future_score_used": False})
        if choice:
            selected.append({**choice, "selected_delta_frames": choice["delta_frames"],
                             "selected_delta_seconds": choice["target_timestamp"] - choice["timestamp"],
                             "interval_selection_reason": "shortest_geometry_observable",
                             "future_visual_score_used_for_selection": False})
    return selected, diagnostics
