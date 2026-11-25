from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.consistency.mtpc import NOT_AVAILABLE
from driveaudit_eval.consistency.mtpc.common import mean, write_json
from driveaudit_eval.consistency.mtpc.history_anchor import sample_mask_points
from driveaudit_eval.video.utils import read_image_bgr


def _track_conditional_entry(rows: list[dict[str, Any]], image_root: Path, minimum_points: int, maximum_frames: int) -> dict[str, Any]:
    rows = sorted([row for row in rows if row["expected_visible"]], key=lambda row: row["frame_index"])
    if not rows: return {"supported": False, "reason": "not_observable"}
    first = rows[0]; camera = first["camera_name"]; frame = first["frame_index"]
    image = read_image_bgr(image_root / camera / f"{frame:06d}.jpg"); gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points = sample_mask_points(gray, first["projected_polygon"], 32); initial = len(points)
    if initial < minimum_points: return {"supported": False, "reason": "insufficient_visual_features", "entry_frame": frame, "entry_camera": camera}
    survived, agreements, current = 1, [], gray
    for next_frame in range(frame + 1, min(frame + maximum_frames, 90)):
        next_image = read_image_bgr(image_root / camera / f"{next_frame:06d}.jpg"); next_gray = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)
        moved, status, _ = cv2.calcOpticalFlowPyrLK(current, next_gray, points.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3)
        if moved is None: break
        keep = status.reshape(-1) > 0; old, new = points[keep], moved.reshape(-1, 2)[keep]
        if len(new) < minimum_points: break
        expected_next = next((row for row in rows if row["frame_index"] == next_frame), None)
        expected_previous = next((row for row in rows if row["frame_index"] == next_frame - 1), None)
        if expected_next and expected_previous:
            actual_flow = np.median(new - old, axis=0); expected_flow = np.asarray(expected_next["center_2d"]) - expected_previous["center_2d"]
            na, nb = np.linalg.norm(actual_flow), np.linalg.norm(expected_flow)
            agreements.append(float(np.dot(actual_flow, expected_flow) / (na * nb)) if min(na, nb) > 1e-6 else 1.0)
        points, current, survived = new, next_gray, survived + 1
    return {"supported": survived >= 2, "reason": "confirmed_point_track" if survived >= 2 else "not_multiframe_confirmed",
            "entry_frame": frame, "entry_camera": camera, "initial_points": initial, "survived_frames": survived,
            "conditional_track_survival": survived / max(1, min(maximum_frames, 90 - frame)), "conditional_motion_agreement": mean(agreements)}


def verify_future_entrants(projections: list[dict[str, Any]], image_root: str | Path, config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in projections:
        if not row["history_anchored"]: groups[(row["agent_id"], row["camera_name"])].append(row)
    candidate_by_agent = defaultdict(list)
    for (agent, _), rows in groups.items():
        visible = [row for row in rows if row["expected_visible"]]
        if visible: candidate_by_agent[agent].append(min(visible, key=lambda row: row["frame_index"]))
    results = []
    for agent, entries in candidate_by_agent.items():
        entry = max(entries, key=lambda row: row["area"])
        rows = groups[(agent, entry["camera_name"])]
        result = _track_conditional_entry(rows, Path(image_root), int(config["minimum_points"]), int(config["maximum_track_frames"]))
        results.append({"agent_id": agent, "protocol": "conditional_presence_verification", "expected_entry_camera": entry["camera_name"],
                        "expected_entry_frame": entry["frame_index"], **result})
    supported = [row for row in results if row["supported"]]
    metrics = {"status": "valid_low_sample_conditional_presence_verification" if results else "not_available_no_future_entrants",
               "conditional_not_open_world_recall": True, "future_entrant_count": len(results),
               "future_entrant_support_rate": len(supported) / len(results) if results else NOT_AVAILABLE,
               "entry_time_error_frames": mean(abs(row["entry_frame"] - row["expected_entry_frame"]) for row in supported),
               "entry_camera_accuracy": mean(row["entry_camera"] == row["expected_entry_camera"] for row in supported),
               "conditional_track_survival": mean(row.get("conditional_track_survival") for row in supported),
               "conditional_motion_agreement": mean(row.get("conditional_motion_agreement") for row in supported), "per_agent": results}
    write_json(Path(output) / "metrics.json", metrics); write_json(Path(output) / "conditional_tracks.json", results); return metrics
