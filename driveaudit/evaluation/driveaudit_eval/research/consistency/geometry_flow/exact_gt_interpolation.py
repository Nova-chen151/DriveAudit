from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.agents.core import quat_rotation
from driveaudit_eval.research.consistency.observed_flow.time_alignment import align as native_align, angle_lerp, interpolate_observations
from driveaudit_eval.research.consistency.observed_flow.schema import write_json


def quaternion_slerp(a: list[float], b: list[float], ratio: float) -> list[float]:
    first, last = np.asarray(a, float), np.asarray(b, float); first /= np.linalg.norm(first); last /= np.linalg.norm(last)
    dot = float(np.dot(first, last))
    if dot < 0: last, dot = -last, -dot
    if dot > .9995:
        value = first + ratio * (last - first); return (value / np.linalg.norm(value)).tolist()
    theta = math.acos(np.clip(dot, -1, 1)); value = (math.sin((1-ratio)*theta)*first + math.sin(ratio*theta)*last) / math.sin(theta)
    return value.tolist()


def interpolate_pose(first: dict[str, Any], last: dict[str, Any], timestamp: float) -> dict[str, Any]:
    ratio = (timestamp - float(first["timestamp"])) / max(float(last["timestamp"]) - float(first["timestamp"]), 1e-9)
    return {"timestamp": timestamp, "translation": ((1-ratio)*np.asarray(first["translation"],float)+ratio*np.asarray(last["translation"],float)).tolist(),
            "rotation": quaternion_slerp(first["rotation"],last["rotation"],ratio)}


def build(config: dict[str, Any], v1_config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    native = native_align(v1_config, Path(output) / "native_camera_time")
    row_map: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in native["rows"]: row_map.setdefault((row["agent_id"], int(row["frame_index"])), []).append(row)
    common_rows = []
    for track in native["tracks"]:
        for sample in track["observations"]:
            frame = int(sample["frame_index"])
            for row in row_map.get((track["agent_id"], frame), []):
                common_rows.append({**row, "box_reference": sample["box_bev"], "timestamp": sample["timestamp"], "alignment_protocol": "common_frame_timestamp"})
    result = {"native": native, "common_rows": common_rows, "native_rows": native["rows"],
              "native_interpolation_uses_future_visual_result": False, "valid_mask_holes_interpolated": 0}
    write_json(Path(output) / "native_camera_time/interpolation_summary.json", {"native_rows": len(native["rows"]), "common_rows": len(common_rows),
        "native_protocol": "agent_and_camera_pose_at_exposure_timestamp", "common_protocol": "agent_at_common_frame_timestamp_camera_pose_at_exposure", "future_scores_read": False})
    return result
