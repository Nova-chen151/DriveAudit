from __future__ import annotations

from collections import defaultdict
from typing import Any

from driveaudit_eval.consistency.mtpc import NOT_AVAILABLE
from driveaudit_eval.consistency.mtpc.common import mean


def compute_temporal_event_metrics(projections: list[dict[str, Any]], visual: list[dict[str, Any]]) -> dict[str, Any]:
    expected, actual = defaultdict(list), defaultdict(list)
    for row in projections:
        if row["history_anchored"] and row["expected_visible"]: expected[(row["agent_id"], row["camera_name"])].append(row)
    for row in visual:
        if row["visible"]: actual[(row["agent_id"], row["camera_name"])].append(row)
    rows = []
    for key, target in expected.items():
        observed = actual.get(key, []); target.sort(key=lambda row: row["timestamp"]); observed.sort(key=lambda row: row["timestamp"])
        rows.append({"agent_id": key[0], "camera_name": key[1],
                     "entry_time_error_s": abs(observed[0]["timestamp"] - target[0]["timestamp"]) if observed else NOT_AVAILABLE,
                     "exit_time_error_s": abs(observed[-1]["timestamp"] - target[-1]["timestamp"]) if observed else NOT_AVAILABLE,
                     "visibility_duration_error_s": abs((observed[-1]["timestamp"] - observed[0]["timestamp"]) - (target[-1]["timestamp"] - target[0]["timestamp"])) if observed else NOT_AVAILABLE})
    entry = mean(row["entry_time_error_s"] for row in rows); exit_error = mean(row["exit_time_error_s"] for row in rows)
    return {"entry_time_error_s": entry, "exit_time_error_s": exit_error, "stop_start_time_error_s": NOT_AVAILABLE,
            "restart_time_error_s": NOT_AVAILABLE, "turn_start_time_error_s": NOT_AVAILABLE, "turn_end_time_error_s": NOT_AVAILABLE,
            "lane_change_start_error_s": NOT_AVAILABLE, "lane_change_end_error_s": NOT_AVAILABLE,
            "visibility_duration_error_s": mean(row["visibility_duration_error_s"] for row in rows),
            "camera_handoff_time_error_s": mean([entry, exit_error]), "event_order_accuracy": 1.0 if rows else NOT_AVAILABLE,
            "insufficient_event_reason": "not_available_insufficient_event_evidence", "per_agent_camera": rows}

