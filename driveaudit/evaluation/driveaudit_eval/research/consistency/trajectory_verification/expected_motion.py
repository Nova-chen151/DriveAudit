from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.trajectory_verification.common import write_json


def _polygon_angle(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    rectangle = cv2.minAreaRect(np.asarray(points, np.float32))
    return math.radians(float(rectangle[2]))


def _angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def expected_motion(projections: list[dict[str, Any]], window_lengths: list[int],
                    output: str | Path | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in projections:
        grouped[(row["agent_id"], row["camera_name"])][int(row["frame_index"])] = row
    result: list[dict[str, Any]] = []
    for (agent, camera), frames in grouped.items():
        for frame, first in sorted(frames.items()):
            for length in window_lengths:
                target_frame = frame + int(length) - 1
                last = frames.get(target_frame)
                if last is None or not first.get("projected_valid") or not last.get("projected_valid"):
                    continue
                center_flow = np.asarray(last["projected_center"], float) - np.asarray(first["projected_center"], float)
                corners_a = np.asarray(first["projected_corners"], float)
                corners_b = np.asarray(last["projected_corners"], float)
                count = min(len(corners_a), len(corners_b))
                corner_flow = corners_b[:count] - corners_a[:count] if count else np.empty((0, 2))
                box_a, box_b = first["projected_box_2d"], last["projected_box_2d"]
                width_a, height_a = max(box_a[2] - box_a[0], 1e-6), max(box_a[3] - box_a[1], 1e-6)
                width_b, height_b = max(box_b[2] - box_b[0], 1e-6), max(box_b[3] - box_b[1], 1e-6)
                result.append({"agent_id": agent, "agent_type": first["agent_type"],
                               "history_anchored": first["history_anchored"], "camera_name": camera,
                               "frame_index": frame, "target_frame_index": target_frame,
                               "timestamp": first["timestamp"], "target_timestamp": last["timestamp"],
                               "window_length": int(length),
                               "expected_visible": bool(first.get("expected_visible") and last.get("expected_visible")),
                               "expected_center_flow": center_flow.tolist(),
                               "expected_corner_flow": corner_flow.tolist(),
                               "expected_polygon_flow": corner_flow.tolist(),
                               "expected_scale_change": float(math.sqrt(max(last["projected_area"], 1e-6) /
                                                                         max(first["projected_area"], 1e-6))),
                               "expected_aspect_ratio_change": float((width_b / height_b) / (width_a / height_a)),
                               "expected_rotation_change": _angle_delta(_polygon_angle(last["projected_polygon"]),
                                                                          _polygon_angle(first["projected_polygon"])),
                               "expected_motion_direction": float(math.atan2(center_flow[1], center_flow[0])),
                               "expected_motion_magnitude": float(np.linalg.norm(center_flow)),
                               "expected_entry_direction": float(math.atan2(center_flow[1], center_flow[0]))
                                   if not first.get("expected_visible") and last.get("expected_visible") else None,
                               "expected_exit_direction": float(math.atan2(center_flow[1], center_flow[0]))
                                   if first.get("expected_visible") and not last.get("expected_visible") else None,
                               "source_projection": first, "target_projection": last,
                               "ego_camera_motion_included_by_per_frame_projection": True})
    if output is not None:
        write_json(Path(output) / "expected_motion.json", result)
    return result

