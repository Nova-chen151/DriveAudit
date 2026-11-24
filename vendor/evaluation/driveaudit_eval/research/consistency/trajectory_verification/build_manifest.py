from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from driveaudit_eval.research.consistency.trajectory_verification.common import CAMERAS, image_path, load_geometry, write_json


def build_manifest(pair: dict[str, Any], output: str | Path) -> dict[str, Any]:
    geometry = load_geometry(pair["image_sequence"]["conditions_root"])
    missing = [str(image_path(pair, camera, frame)) for frame in range(90) for camera in CAMERAS
               if not image_path(pair, camera, frame).is_file()]
    history = cv2.imread(str(image_path(pair, "CAM_FRONT", 0)))
    future = cv2.imread(str(image_path(pair, "CAM_FRONT", 10)))
    result = {"protocol": "tcvv_submission_conditioned", "pair_id": pair["pair_id"],
              "reference_semantics": "submitted_trajectory_as_visual_expectation",
              "scene_count": 1, "history_frames": 10, "future_frames": 80,
              "camera_count": 6, "camera_order": list(CAMERAS), "missing_images": missing,
              "history_resolution": [history.shape[1], history.shape[0]] if history is not None else None,
              "future_resolution": [future.shape[1], future.shape[0]] if future is not None else None,
              "calibration_rows": len(geometry["calibrations"]), "ego_pose_rows": len(geometry["poses"]),
              "timestamp_count": len(geometry["timestamps"]), "future_gt_required": False,
              "input_valid": not missing and len(geometry["calibrations"]) == 540 and len(geometry["poses"]) == 90}
    write_json(Path(output) / "manifest.json", result)
    write_json(Path(output) / "input_audit.json", result)
    return result

