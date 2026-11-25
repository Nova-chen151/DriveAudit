from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from driveaudit_eval.research.consistency.observed_flow.schema import write_json


def build(projections: list[dict[str, Any]], deltas: list[int], output: str | Path | None = None) -> list[dict[str, Any]]:
    grouped = defaultdict(dict)
    for row in projections: grouped[(row["agent_id"], row["camera_name"])][int(row["frame_index"])] = row
    rows = []
    for (agent, camera), frames in grouped.items():
        for frame, first in sorted(frames.items()):
            for delta in deltas:
                last = frames.get(frame + int(delta))
                if last is None or not first.get("projected_valid") or not last.get("projected_valid"): continue
                matches_a = {index: point for index, point in zip(first.get("projected_corner_indices", []), first.get("projected_corners", []))}
                matches_b = {index: point for index, point in zip(last.get("projected_corner_indices", []), last.get("projected_corners", []))}
                common = sorted(set(matches_a).intersection(matches_b)); a = np.asarray([matches_a[index] for index in common], float); b = np.asarray([matches_b[index] for index in common], float)
                center = np.asarray(last["projected_center"], float) - np.asarray(first["projected_center"], float)
                source_angle = cv2.minAreaRect(np.asarray(first["projected_polygon"], np.float32))[2]
                target_angle = cv2.minAreaRect(np.asarray(last["projected_polygon"], np.float32))[2]
                rows.append({"agent_id": agent, "agent_type": first["agent_type"], "category": first["category"], "camera_name": camera,
                    "frame_index": frame, "target_frame_index": frame + int(delta), "timestamp": first["timestamp"], "target_timestamp": last["timestamp"],
                    "delta_frames": int(delta), "delta_seconds": float(last["timestamp"] - first["timestamp"]),
                    "source_projection": first, "target_projection": last,
                    "expected_visible": bool(first.get("expected_visible") and last.get("expected_visible")),
                    "expected_center_flow": center.tolist(), "expected_corner_source": a.tolist(), "expected_corner_target": b.tolist(),
                    "expected_corner_flow": (b - a).tolist() if len(a) else [],
                    "expected_scale_change": math.sqrt(max(float(last["projected_area"]), 1e-6) / max(float(first["projected_area"]), 1e-6)),
                    "expected_rotation_change": math.radians((target_angle - source_angle + 90) % 180 - 90),
                    "expected_motion_magnitude": float(np.linalg.norm(center)), "expected_motion_validity": len(common) >= 3,
                    "expected_motion_components": ["target_3d_motion", "ego_translation", "ego_rotation", "camera_motion", "depth_scale", "box_rotation", "perspective"]})
    if output is not None: write_json(Path(output) / "expected_motion/expected_motion.json", rows)
    return rows


def flow_at_points(expectation: dict[str, Any], points: np.ndarray) -> tuple[np.ndarray, str, float]:
    a, b = np.asarray(expectation["expected_corner_source"], float), np.asarray(expectation["expected_corner_target"], float)
    if len(a) >= 3:
        design = np.column_stack([a, np.ones(len(a))])
        if np.linalg.matrix_rank(design) == 3:
            matrix = np.linalg.lstsq(design, b, rcond=None)[0].T
            predicted_corners = np.column_stack([a, np.ones(len(a))]) @ matrix.T
            uncertainty = float(np.median(np.linalg.norm(predicted_corners - b, axis=1)))
            predicted = np.column_stack([points, np.ones(len(points))]) @ matrix.T
            return predicted - points, "visible_face_affine", uncertainty
    center = np.asarray(expectation["source_projection"]["projected_center"], float); target = np.asarray(expectation["target_projection"]["projected_center"], float)
    angle = float(expectation["expected_rotation_change"]); scale = float(expectation["expected_scale_change"])
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    predicted = (points - center) @ rotation.T * scale + target
    return predicted - points, "center_scale_rotation_fallback", 3.0


def sample_surface_points(expectation: dict[str, Any], count: int = 36) -> np.ndarray:
    polygon = np.asarray(expectation["source_projection"]["projected_polygon"], np.float32)
    x, y, w, h = cv2.boundingRect(polygon); side = max(2, int(math.sqrt(count)))
    candidates = np.asarray([[u, v] for v in np.linspace(y, y + h, side) for u in np.linspace(x, x + w, side)], float)
    return np.asarray([point for point in candidates if cv2.pointPolygonTest(polygon, tuple(point), False) >= 0], float)

