from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.trajectory_verification.common import load_geometry
from driveaudit_eval.research.consistency.observed_flow.schema import CAMERAS, write_csv, write_json


def audit(condition_root: str | Path, output: str | Path) -> dict[str, Any]:
    geometry = load_geometry(condition_root); frame_times = geometry["timestamps"]; grouped = defaultdict(list); offsets = defaultdict(list)
    for row in geometry["calibrations"]:
        camera = row["camera_name"]; timestamp = float(row["timestamp"]) / 1e6
        grouped[camera].append(timestamp); offsets[camera].append(timestamp - float(frame_times[int(row["frame_index"])]))
    per_camera = {}
    for camera in CAMERAS:
        values = sorted(grouped[camera]); dt = np.diff(values)
        per_camera[camera] = {"frame_count": len(values), "median_dt_s": float(np.median(dt)), "minimum_dt_s": float(dt.min()),
            "maximum_dt_s": float(dt.max()), "median_common_offset_s": float(np.median(offsets[camera])),
            "maximum_absolute_common_offset_s": float(np.max(np.abs(offsets[camera]))), "native_timestamps_strictly_increasing": bool(np.all(dt > 0))}
    result = {"formal_protocol": "native_camera_timestamp", "per_camera": per_camera,
              "all_cameras_independent_timeline": True, "future_visual_scores_read": False}
    output = Path(output); write_json(output / "native_camera_time/timestamp_audit.json", result)
    write_csv(output / "native_camera_time/per_camera.csv", [{"camera": camera, **value} for camera, value in per_camera.items()]); return result

