from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.trajectory_verification import NOT_AVAILABLE
from driveaudit_eval.research.consistency.trajectory_verification.common import mean, write_csv, write_json


def _longest_run(frames: list[int]) -> int:
    best = current = 0
    previous = None
    for frame in sorted(set(frames)):
        current = current + 1 if previous is not None and frame == previous + 1 else 1
        best = max(best, current); previous = frame
    return best


def future_entrant_metrics(evidence: list[dict[str, Any]], projections: list[dict[str, Any]],
                           primary_window: int, minimum_confirmation_frames: int,
                           output: str | Path | None = None) -> dict[str, Any]:
    rows = [row for row in evidence if not row["history_anchored"] and int(row["window_length"]) == primary_window]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["agent_id"]].append(row)
    per_agent = []
    for agent, items in grouped.items():
        supported = [row for row in items if row["visual_supported"]]
        supported_frames = [int(row["frame_index"]) for row in supported]
        expected_first = min((int(row["frame_index"]) for row in items), default=None)
        actual_first = min(supported_frames, default=None)
        expected_camera = next((row["camera_name"] for row in sorted(items, key=lambda r: r["frame_index"])), None)
        actual_camera = next((row["camera_name"] for row in sorted(supported, key=lambda r: r["frame_index"])), None)
        longest = _longest_run(supported_frames)
        per_agent.append({"agent_id": agent, "expected_windows": len(items), "supported_windows": len(supported),
                          "conditional_support_rate": len(supported) / max(1, len(items)),
                          "entry_time_error_frames": abs(actual_first - expected_first) if actual_first is not None else NOT_AVAILABLE,
                          "entry_camera_correct": actual_camera == expected_camera if actual_camera is not None else False,
                          "short_window_motion_agreement": mean(row["flow_cosine_similarity"] for row in supported),
                          "longest_confirmation_run": longest,
                          "multi_frame_confirmed": longest >= minimum_confirmation_frames,
                          "conditional_survival_rate": len(set(supported_frames)) / max(1, len(set(row["frame_index"] for row in items))),
                          "metric_scope": "conditional_future_entrant_verification"})
    result = {"status": "valid_low_sample_conditional_future_entrant_verification" if per_agent else "not_available_no_observable_entrants",
              "conditional_entrant_support_rate": mean(row["conditional_support_rate"] for row in per_agent),
              "entry_camera_accuracy": mean(float(row["entry_camera_correct"]) for row in per_agent),
              "entry_time_error_frames": mean(row["entry_time_error_frames"] for row in per_agent),
              "short_window_motion_agreement": mean(row["short_window_motion_agreement"] for row in per_agent),
              "multi_frame_confirmation_rate": mean(float(row["multi_frame_confirmed"]) for row in per_agent),
              "conditional_survival_rate": mean(row["conditional_survival_rate"] for row in per_agent),
              "agent_count": len(per_agent), "not_open_world_detection_recall": True, "per_agent": per_agent}
    if output is not None:
        write_csv(Path(output) / "per_agent.csv", per_agent)
        write_json(Path(output) / "metrics.json", result)
    return result

