from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from driveaudit_eval.consistency.shared.common import wrap_angle


def common_timestamps(tracks: list[dict[str, Any]], rate_hz: float = 10.0) -> np.ndarray:
    times = [row["timestamp"] for track in tracks for row in track["observations"]]
    if not times:
        return np.empty(0)
    step = 1.0 / rate_hz
    return np.arange(min(times), max(times) + step * 0.25, step)


def resample_tracks(tracks: list[dict[str, Any]], timestamps: np.ndarray,
                    maximum_gap_seconds: float = 0.13) -> list[dict[str, Any]]:
    result = []
    for track in tracks:
        rows = sorted(track["observations"], key=lambda row: row["timestamp"])
        source_times = np.asarray([row["timestamp"] for row in rows], dtype=float)
        observations = []
        for frame, timestamp in enumerate(timestamps):
            right = int(np.searchsorted(source_times, timestamp, side="left"))
            if right == 0 or right == len(rows):
                if right < len(rows) and abs(source_times[right] - timestamp) < 1e-6:
                    observations.append({**copy.deepcopy(rows[right]), "frame_index": frame, "timestamp": float(timestamp)})
                continue
            left = right - 1
            if source_times[right] - source_times[left] > maximum_gap_seconds:
                continue
            alpha = float((timestamp - source_times[left]) / (source_times[right] - source_times[left]))
            a, b = rows[left], rows[right]; box = {}
            for key in set(a["box_bev"]) & set(b["box_bev"]):
                if isinstance(a["box_bev"][key], (int, float)):
                    if key == "yaw":
                        delta = wrap_angle(float(b["box_bev"][key]) - float(a["box_bev"][key]))
                        box[key] = wrap_angle(float(a["box_bev"][key]) + alpha * delta)
                    else:
                        box[key] = (1 - alpha) * float(a["box_bev"][key]) + alpha * float(b["box_bev"][key])
            velocity = ((1 - alpha) * np.asarray(a.get("velocity", [0, 0]), dtype=float) +
                        alpha * np.asarray(b.get("velocity", [0, 0]), dtype=float)).tolist()
            row = {**copy.deepcopy(a), "frame_index": frame, "timestamp": float(timestamp), "box_bev": box, "velocity": velocity}
            if "expected_visible" in a:
                row["expected_visible"] = bool(a.get("expected_visible") and b.get("expected_visible"))
            if a.get("out_of_detector_range") or b.get("out_of_detector_range"):
                row["out_of_detector_range"] = True; row["expected_visible"] = False
            observations.append(row)
        if observations:
            item = {**track, "observations": observations, "track_length": len(observations)}; result.append(item)
    return result
