from __future__ import annotations

from collections import defaultdict
from typing import Any

from driveaudit_eval.consistency.mtpc import NOT_AVAILABLE
from driveaudit_eval.consistency.mtpc.common import mean


def compute_presence_metrics(projections: list[dict[str, Any]], visual: list[dict[str, Any]], confidence_threshold: float = .1,
                             persistent_only: bool = True) -> dict[str, Any]:
    expected = [row for row in projections if row["expected_visible"] and (row["history_anchored"] or not persistent_only)]
    visual_map = {(row["agent_id"], row["camera_name"], row["frame_index"]): row for row in visual}
    matched, frame_rows = 0, []
    for row in expected:
        actual = visual_map.get((row["agent_id"], row["camera_name"], row["frame_index"])); supported = bool(actual and actual["visible"] and actual["confidence"] >= confidence_threshold)
        matched += supported; frame_rows.append({"agent_id": row["agent_id"], "camera_name": row["camera_name"], "frame_index": row["frame_index"],
                                                 "expected_visible": True, "visual_supported": supported, "status": "tracked" if supported else "missing"})
    unexpected = [row for row in visual if row["visible"] and not any(value["agent_id"] == row["agent_id"] and value["camera_name"] == row["camera_name"] and value["frame_index"] == row["frame_index"] and value["expected_visible"] for value in projections)]
    by_agent_expected, by_agent_matched = defaultdict(int), defaultdict(int)
    for row in frame_rows: by_agent_expected[row["agent_id"]] += 1; by_agent_matched[row["agent_id"]] += int(row["visual_supported"])
    survival = [by_agent_matched[agent] / count for agent, count in by_agent_expected.items() if count]
    recovered_agents = {row["agent_id"] for row in visual if row.get("track_status") == "reidentified"}
    fragments = sum(1 for rows in defaultdict(list).values())
    del fragments
    expected_times, actual_times = defaultdict(list), defaultdict(list)
    for row in expected: expected_times[(row["agent_id"], row["camera_name"])].append(row["timestamp"])
    for row in visual:
        if row["visible"]: actual_times[(row["agent_id"], row["camera_name"])].append(row["timestamp"])
    appearance_errors, disappearance_errors = [], []
    for key, times in expected_times.items():
        if actual_times.get(key): appearance_errors.append(actual_times[key][0] - min(times)); disappearance_errors.append(actual_times[key][-1] - max(times))
    return {"persistent_agent_visual_support_rate": matched / len(expected) if expected else NOT_AVAILABLE,
            "persistent_agent_missing_ratio": 1 - matched / len(expected) if expected else NOT_AVAILABLE,
            "visual_survival_rate": mean(survival), "track_recovery_rate": len(recovered_agents) / max(1, len({row['agent_id'] for row in visual})),
            "premature_visual_disappearance_ratio": mean(value < -1e-6 for value in disappearance_errors),
            "over_persistence_ratio": len(unexpected) / max(1, len(visual)), "appearance_time_error_s": mean(abs(value) for value in appearance_errors),
            "disappearance_time_error_s": mean(abs(value) for value in disappearance_errors),
            "camera_support_recall": matched / len(expected) if expected else NOT_AVAILABLE,
            "extra_visual_object_diagnostic": len(unexpected), "expected_visible_agent_camera_frames": len(expected),
            "supported_agent_camera_frames": matched, "per_frame": frame_rows}
