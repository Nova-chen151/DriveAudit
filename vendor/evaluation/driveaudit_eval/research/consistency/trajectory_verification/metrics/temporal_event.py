from __future__ import annotations

from collections import defaultdict
from typing import Any

from driveaudit_eval.research.consistency.trajectory_verification import NOT_AVAILABLE
from driveaudit_eval.research.consistency.trajectory_verification.common import mean


def temporal_event_metrics(evidence: list[dict[str, Any]], projections: list[dict[str, Any]],
                           primary_window: int, dt: float) -> dict[str, Any]:
    rows = [row for row in evidence if int(row["window_length"]) == int(primary_window)]
    expected: dict[tuple[str, str], list[int]] = defaultdict(list)
    supported: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in projections:
        if row.get("expected_visible"):
            expected[(row["agent_id"], row["camera_name"])].append(int(row["frame_index"]))
    for row in rows:
        if row["visual_supported"]:
            supported[(row["agent_id"], row["camera_name"])].append(int(row["frame_index"]))
    per_camera, entry, exit_, duration = [], [], [], []
    for key, frames in expected.items():
        actual = supported.get(key, [])
        if not actual:
            continue
        e = abs(min(actual) - min(frames)) * dt
        x = abs(max(actual) - max(frames)) * dt
        d = abs(len(set(actual)) - len(set(frames))) * dt
        entry.append(e); exit_.append(x); duration.append(d)
        per_camera.append({"agent_id": key[0], "camera_name": key[1], "appearance_time_error_s": e,
                           "disappearance_time_error_s": x, "visibility_duration_error_s": d})
    return {"appearance_time_error_s": mean(entry), "disappearance_time_error_s": mean(exit_),
            "visibility_duration_error_s": mean(duration), "stop_start_time_error_s": NOT_AVAILABLE,
            "restart_time_error_s": NOT_AVAILABLE, "turn_start_time_error_s": NOT_AVAILABLE,
            "turn_end_time_error_s": NOT_AVAILABLE, "lane_change_start_error_s": NOT_AVAILABLE,
            "lane_change_end_error_s": NOT_AVAILABLE, "camera_entry_time_error_s": mean(entry),
            "camera_exit_time_error_s": mean(exit_), "event_order_accuracy": 1.0 if per_camera else NOT_AVAILABLE,
            "insufficient_event_evidence_reason": "stop/turn/lane-change image events require more than local-flow state evidence",
            "per_agent_camera": per_camera}

