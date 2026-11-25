from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from driveaudit_eval.research.consistency.observed_flow.schema import CAMERAS, PROTOCOL, VERSION, image_path, sha256, write_json


def build(config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    conditions = Path(config["inputs"]["conditions"])
    counts = {camera: len(list((Path(config["inputs"]["gt_images"]) / camera).glob("*.jpg"))) for camera in CAMERAS}
    first = cv2.imread(str(image_path(config, CAMERAS[0], 0)))
    result = {
        "protocol": PROTOCOL, "version": VERSION, "scene_id": config["scene"]["scene_id"],
        "reference_semantics": "paired_ground_truth_visual_trajectory_self_consistency",
        "gt_images": str(Path(config["inputs"]["gt_images"])), "conditions": str(conditions),
        "camera_order": list(CAMERAS), "image_count_by_camera": counts,
        "image_resolution": [int(first.shape[1]), int(first.shape[0])] if first is not None else None,
        "history_frames": int(config["scene"]["history_frames"]), "future_frames": int(config["scene"]["future_frames"]),
        "uses_future_gt": True, "uses_submission_data": False, "uses_bevformer": False,
        "uses_tracker": False, "uses_planning_or_closed_loop": False,
        "input_hashes": {"gt_images": sha256(config["inputs"]["gt_images"]),
                         "boxes": sha256(conditions / "boxes.csv"), "frames": sha256(conditions / "frames.csv"),
                         "calibration": sha256(conditions / "camera_calibration.csv"),
                         "ego_pose": sha256(conditions / "ego_pose.csv")},
    }
    write_json(Path(output) / "manifest.json", result)
    return result

