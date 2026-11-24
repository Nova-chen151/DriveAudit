from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

from driveaudit_eval.research.consistency.projection_consistency.build_manifest import build as build_manifest
from driveaudit_eval.research.consistency.projection_consistency.schema import CAMERAS, read_csv, read_json, sha256, write_json
from driveaudit_eval.research.consistency.projection_consistency.trajectory_loader import conditions_root

REQUIRED_BOX_FIELDS = {
    "frame_index", "agent_id", "class_name", "x", "y", "z",
    "width", "length", "height", "yaw", "visibility", "valid_flag",
}


def coordinate_is_fixed_metric(value: str | None) -> bool:
    return value in {"global_world", "history_last_ego_fixed_metric"}


def audit(config: dict[str, Any], output: str | Path, scene_ids: list[str] | None = None,
          hash_images: bool = True) -> dict[str, Any]:
    output = Path(output)
    manifest = build_manifest(config, output, scene_ids)
    rows, pairs, hashes = [], [], {}
    trajectory_root = Path(config["inputs"]["trajectory_root"])
    reference_image_root = Path(config["inputs"].get("reference_image_root", trajectory_root))
    for pair in manifest["pairs"]:
        scene_id = pair["scene_id"]
        candidate = Path(pair["candidate_scene_root"])
        conditions = conditions_root(config, scene_id)
        frames = read_csv(conditions / "frames.csv")
        boxes = read_csv(conditions / "boxes.csv")
        calibration = read_csv(conditions / "camera_calibration.csv")
        poses = read_csv(conditions / "ego_pose.csv")
        scene_metadata_path = Path(pair["trajectory_scene_root"]) / "metadata.json"
        metadata = read_json(scene_metadata_path) if scene_metadata_path.is_file() else read_json(conditions / "metadata.json")
        camera_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        for item in calibration:
            camera_rows[item.get("camera", "")].append(item)
        missing, decode_failures, resolutions, image_hash_match = [], [], {}, True
        image_counts = {}
        for camera in CAMERAS:
            paths = [candidate / camera / f"{frame:06d}.jpg" for frame in range(10, 90)]
            image_counts[camera] = sum(path.is_file() for path in paths)
            camera_sizes = set()
            for path in paths:
                if not path.is_file():
                    missing.append(str(path)); continue
                image = cv2.imread(str(path))
                if image is None:
                    decode_failures.append(str(path)); continue
                camera_sizes.add((int(image.shape[1]), int(image.shape[0])))
                if hash_images:
                    reference = reference_image_root / scene_id / camera / path.name
                    if not reference.is_file() or sha256(reference) != sha256(path):
                        image_hash_match = False
            resolutions[camera] = [list(size) for size in sorted(camera_sizes)]
        future_frames = {int(row["frame_index"]) for row in frames if 10 <= int(row["frame_index"]) < 90}
        box_frames = {int(row["frame_index"]) for row in boxes if row.get("valid_flag", "").lower() in {"1", "true", "yes"}}
        ids = defaultdict(set)
        for row in boxes:
            ids[row.get("agent_id", row.get("obj_id", ""))].add(row.get("instance_token", row.get("agent_id", row.get("obj_id", ""))))
        native_time_ok = True
        maximum_time_error_us = 0
        for camera in CAMERAS:
            values = sorted((int(row["timestamp"]), int(row["frame_index"])) for row in camera_rows[camera] if 10 <= int(row["frame_index"]) < 90)
            native_time_ok &= len(values) == 80 and all(a[0] < b[0] for a, b in zip(values, values[1:]))
            common = {int(row["frame_index"]): int(row["timestamp"]) for row in frames}
            maximum_time_error_us = max(maximum_time_error_us, *(abs(timestamp - common[frame]) for timestamp, frame in values), 0)
        fields_ok = bool(boxes and REQUIRED_BOX_FIELDS.issubset(boxes[0]))
        checks = {
            "candidate_future_images_80x6": all(value == 80 for value in image_counts.values()),
            "all_images_decodable": not decode_failures,
            "resolution_consistent_per_camera": all(len(value) == 1 for value in resolutions.values()),
            "native_timestamps_present_and_monotonic": native_time_ok,
            "trajectory_covers_future_80": future_frames == set(range(10, 90)) and set(range(10, 90)).issubset(box_frames),
            "agent_id_stable": all(len(value) == 1 for value in ids.values()),
            "box_fields_complete": fields_ok,
            "calibration_complete": len(calibration) >= 90 * len(CAMERAS),
            "ego_pose_complete": len(poses) >= 90,
            "candidate_scene_matches_trajectory_scene": pair["candidate_scene_id"] == pair["trajectory_scene_id"] == scene_id,
            "explicit_pair_id": bool(pair["pair_id"]),
            "coordinate_declared": coordinate_is_fixed_metric(metadata.get("trajectory_coordinate")),
            "quaternion_wxyz": all(row.get("quaternion_convention") == "wxyz" for row in calibration),
            "box_center_convention_declared": metadata.get("box_center_convention") in {"geometric_center", "bottom_center"},
            "candidate_reference_rgb_hash_match": image_hash_match if config["mode"] == "gt_example" else True,
        }
        status = "completed" if all(checks.values()) and not missing else "failed"
        rows.append({
            "scene_id": scene_id, "status": status, "checks": checks,
            "image_counts": image_counts, "resolutions": resolutions,
            "missing_image_count": len(missing), "decode_failure_count": len(decode_failures),
            "trajectory_frame_count": len(future_frames), "box_frame_count": len(box_frames),
            "agent_count": len(ids), "box_row_count": len(boxes),
            "calibration_row_count": len(calibration), "ego_pose_row_count": len(poses),
            "maximum_native_common_offset_us": maximum_time_error_us,
        })
        pairs.append({**pair, "status": status, "checks": checks})
        hashes[scene_id] = {
            "conditions_sha256": sha256(conditions),
            "candidate_future_sha256": sha256(candidate),
            "reference_future_sha256": sha256(reference_image_root / scene_id) if hash_images else "not_computed",
            "candidate_reference_rgb_equal": image_hash_match if hash_images else "not_computed",
        }
    preserved = {}
    for path in config.get("integrity", {}).get("preserve_hashes", []):
        item = Path(path)
        preserved[str(item)] = sha256(item) if item.exists() else "not_available"
    result = {
        "status": "completed" if rows and all(row["status"] == "completed" for row in rows) else "failed",
        "scene_count": len(rows), "camera_order": list(CAMERAS),
        "future_frame_range": [10, 89], "future_frames_per_camera": 80,
        "mode": config["mode"], "uses_detector": False, "uses_tracker_identity": False,
        "uses_long_propagation": False, "scenes": rows,
    }
    write_json(output / "input_audit/input_audit.json", result)
    write_json(output / "input_audit/pairing_audit.json", {"status": result["status"], "pairs": pairs})
    write_json(output / "input_audit/hash_audit.json", {"scene_hashes": hashes, "preserved_paths": preserved})
    return result
