from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from driveaudit_eval.research.consistency.observed_flow import NOT_AVAILABLE
from driveaudit_eval.research.consistency.observed_flow.schema import CAMERAS, DYNAMIC_CLASSES, image_path, read_csv, write_json


def audit(config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    root = Path(config["inputs"]["conditions"])
    frames, boxes = read_csv(root / "frames.csv"), read_csv(root / "boxes.csv")
    calibration, poses = read_csv(root / "camera_calibration.csv"), read_csv(root / "ego_pose.csv")
    required_box = {"frame_index", "obj_id", "class_name", "x", "y", "z", "width", "length", "height", "yaw", "vx", "vy", "visibility", "valid_flag"}
    valid = [row for row in boxes if row.get("valid_flag", "").lower() in {"true", "1", "yes"}]
    dynamic = [row for row in valid if row["class_name"] in DYNAMIC_CLASSES]
    image_counts, resolutions, missing = {}, {}, []
    for camera in CAMERAS:
        paths = sorted((Path(config["inputs"]["gt_images"]) / camera).glob("*.jpg")); image_counts[camera] = len(paths)
        image = cv2.imread(str(paths[0])) if paths else None; resolutions[camera] = [image.shape[1], image.shape[0]] if image is not None else NOT_AVAILABLE
        missing.extend(str(image_path(config, camera, frame)) for frame in range(len(frames)) if not image_path(config, camera, frame).is_file())
    optional_roots = [Path(config["inputs"]["gt_images"]), root]
    def optional(patterns: tuple[str, ...]):
        matches = [str(path) for base in optional_roots for path in base.rglob("*") if path.is_file() and any(term in path.name.lower() for term in patterns)]
        return {"status": "available", "paths": matches[:20]} if matches else NOT_AVAILABLE
    camera_counts = Counter(row.get("camera") for row in calibration)
    result = {
        "status": "complete" if not missing and required_box.issubset(boxes[0]) else "incomplete",
        "gt_image_frames": len(frames), "image_count_by_camera": image_counts, "camera_count": len(CAMERAS),
        "image_resolution_by_camera": resolutions, "missing_images": missing,
        "trajectory_frames": len({row["frame_index"] for row in boxes}), "box_frames": len({row["frame_index"] for row in valid}),
        "box_rows": len(boxes), "valid_box_rows": len(valid), "dynamic_box_rows": len(dynamic),
        "agent_count": len({row["obj_id"] for row in valid}), "dynamic_agent_count": len({row["obj_id"] for row in dynamic}),
        "class_counts": dict(Counter(row["class_name"] for row in dynamic)), "visibility_counts": dict(Counter(row["visibility"] for row in dynamic)),
        "timestamps": "available" if all(row.get("timestamp") for row in frames) else NOT_AVAILABLE,
        "ego_pose_rows": len(poses), "calibration_rows": len(calibration), "calibration_by_camera": dict(camera_counts),
        "box_required_fields_present": sorted(required_box.intersection(boxes[0])),
        "gt_instance_mask": optional(("instance", "mask")), "gt_depth": optional(("depth",)),
        "gt_motion_vector": optional(("motion", "flow")), "gt_visibility_or_occlusion": "boxes.csv:visibility",
        "history_future_boundary": {"history": [0, int(config["scene"]["history_frames"]) - 1],
                                    "future": [int(config["scene"]["history_frames"]), len(frames) - 1]},
        "no_submission_inputs_read": True, "no_detector_or_tracker_used": True,
    }
    write_json(Path(output) / "input_audit.json", result); return result

