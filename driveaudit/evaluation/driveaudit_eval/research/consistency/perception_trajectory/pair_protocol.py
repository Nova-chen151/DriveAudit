from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from driveaudit_eval.consistency.config import load_config as read_config

from driveaudit_eval.agents.core import quat_to_yaw
from driveaudit_eval.trajectory.io import load_tracks

from driveaudit_eval.consistency.shared.common import CAMERA_ORDER, read_csv, wrap_angle


def load_pair(pair_manifest: str | Path, pair_id: str | None = None) -> dict[str, Any]:
    data = read_config(pair_manifest) or {}
    pairs = data.get("pairs", [])
    ids = [pair.get("pair_id") for pair in pairs]
    if not pairs or any(not value for value in ids):
        raise ValueError("pair_id_required")
    if len(ids) != len(set(ids)):
        raise ValueError("pair_id_missing_or_not_unique")
    if not pair_id:
        if len(pairs) != 1:
            raise ValueError("pair_id_required")
        pair_id = ids[0]
    if pair_id not in ids:
        raise ValueError("pair_id_missing_or_not_unique")
    return next(pair for pair in pairs if pair["pair_id"] == pair_id)


def _trajectory_scene(pair: dict[str, Any], root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors = []
    source = pair["trajectory_sequence"]
    manifest_path = root / source["manifest_path"]
    if not manifest_path.exists():
        return None, ["trajectory_manifest_missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [scene for scene in manifest.get("scenes", []) if scene.get("scene_id") == source.get("scene_id")]
    if len(selected) != 1:
        errors.append("trajectory_source_scene_not_unique")
        return None, errors
    scene = dict(selected[0]); scene["source_manifest_path"] = str(manifest_path)
    if not Path(scene["submission_file"]).exists(): errors.append("trajectory_path_missing")
    return scene, errors


def validate_pair(pair: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    root = Path(root).resolve(); errors, warnings = [], []
    pair_id = pair.get("pair_id")
    if not pair_id: errors.append("missing_pair_id")
    image = pair.get("image_sequence", {}); trajectory = pair.get("trajectory_sequence", {})
    timeline = pair.get("timeline", {}); transform_cfg = pair.get("coordinate_transform", {})
    if pair.get("camera_order") != CAMERA_ORDER: errors.append("camera_order_mismatch")
    conditions = root / image.get("conditions_root", "")
    future_root = root / image.get("generated_future_root", image.get("future_root", ""))
    required = {"frames": conditions / "frames.csv", "calibration": root / pair.get("calibration", {}).get("path", ""),
                "ego_pose": root / pair.get("ego_pose", {}).get("path", "")}
    for name, path in required.items():
        if not path.is_file(): errors.append(f"{name}_missing")
    condition_reference = trajectory.get("source_role") == "generation_condition_reference"
    if condition_reference:
        trajectory_path = root / trajectory.get("trajectory_path", "")
        if not trajectory_path.is_file(): errors.append("trajectory_path_missing")
        scene = None
    else:
        scene, trajectory_errors = _trajectory_scene(pair, root); errors.extend(trajectory_errors)
    if errors:
        return {"pair_id": pair_id, "image_scene_id": image.get("scene_id"),
                "trajectory_scene_id": trajectory.get("scene_id"), "pair_validation_status": "invalid_pair_metadata",
                "errors": errors, "warnings": warnings}
    frame_rows = read_csv(required["frames"]); pose_rows = read_csv(required["ego_pose"]); calibration_rows = read_csv(required["calibration"])
    image_history = int(timeline.get("image_history_frames", 0)); image_future = int(timeline.get("image_future_frames", 0))
    trajectory_history = int(timeline.get("trajectory_history_frames", 0)); trajectory_start = int(timeline.get("trajectory_future_start_index", -1))
    future_indices = range(image_history, image_history + image_future)
    image_counts = {camera: sum((future_root / camera / f"{frame:06d}.jpg").is_file() for frame in future_indices)
                    for camera in CAMERA_ORDER}
    if any(count != image_future for count in image_counts.values()): errors.append("frame_count_mismatch")
    if len(frame_rows) < image_history + image_future: errors.append("frame_count_mismatch")
    if len(pose_rows) < image_history + image_future: errors.append("ego_pose_frame_count_mismatch")
    if len(calibration_rows) < (image_history + image_future) * len(CAMERA_ORDER): errors.append("calibration_mismatch")
    image_timestamps = np.asarray([float(row["timestamp"]) / 1e6 for row in frame_rows], dtype=float)
    observed_dt = float(np.median(np.diff(image_timestamps)))
    if abs(observed_dt - float(timeline.get("image_dt", observed_dt))) > 2e-3: errors.append("timestamp_mismatch")
    if condition_reference:
        future_image_times = image_timestamps[image_history:image_history + image_future]
        declared_first = float(timeline.get("first_future_timestamp", 0.0)) / 1e6
        if timeline.get("mapping") != "exact_condition_timestamp" or not len(future_image_times) or abs(declared_first - future_image_times[0]) > 1e-3:
            errors.append("timestamp_mismatch")
        boxes = read_csv(root / trajectory["trajectory_path"])
        frame_tokens = {row["token"] for row in frame_rows}
        if not boxes or any(row.get("token") not in frame_tokens for row in boxes): errors.append("trajectory_timeline_mismatch")
        return {"pair_id": pair_id, "submission_id": pair.get("submission_id"),
                "image_generation_condition_id": pair.get("image_generation_condition_id"),
                "image_scene_id": image.get("scene_id"), "trajectory_scene_id": trajectory.get("scene_id"),
                "history_frame_count": image_history, "trajectory_history_frame_count": image_history,
                "future_frame_count": image_future, "image_timestamps": future_image_times.tolist(),
                "trajectory_timestamps": future_image_times.tolist(), "timestamp_offset": 0.0,
                "timeline_mapping": timeline.get("mapping"), "image_dt": observed_dt, "trajectory_dt": observed_dt,
                "coordinate_transform": {"method": "per_frame_lidar_to_ego_to_global"},
                "history_alignment_residual": {"center_mean_m": 0.0, "center_max_m": 0.0,
                                               "yaw_mean_rad": 0.0, "yaw_max_rad": 0.0,
                                               "basis": "shared timestamped lidar/ego calibration"},
                "pair_validation_status": "valid" if not errors else "invalid_pair_metadata",
                "errors": sorted(set(errors)), "warnings": warnings, "trajectory_scene": None,
                "trajectory_path": str((root / trajectory["trajectory_path"]).resolve()),
                "image_counts": image_counts, "names_compared_for_pairing": False}
    if abs(float(scene["dt"]) - float(timeline.get("trajectory_dt", scene["dt"]))) > 1e-6: errors.append("timestamp_mismatch")
    declarations = {"coordinate_system": scene["coordinate_system"], "heading_unit": scene["heading_unit"],
                    "history_frames": scene["history_frames"], "ego_ids": [transform_cfg.get("trajectory_ego_id", "Ego")]}
    tracks = load_tracks(scene["submission_file"], "generated", declarations)
    ego_id = transform_cfg.get("trajectory_ego_id", "Ego")
    ego_tracks = [track for track in tracks if track.scene_id == scene["scene_id"] and track.agent_id == ego_id]
    if len(ego_tracks) != 1: errors.append("coordinate_mismatch")
    if trajectory_start != trajectory_history or trajectory_start >= scene["total_frames"]: errors.append("history_future_boundary_mismatch")
    future_image_times = image_timestamps[image_history:image_history + image_future]
    timestamp_scale = 1e6 if timeline.get("image_timestamp_unit") == "microseconds" else 1.0
    declared_first = float(timeline.get("first_future_timestamp", 0.0)) / timestamp_scale
    if not len(future_image_times) or abs(declared_first - future_image_times[0]) > 1e-3:
        errors.append("timestamp_mismatch")
    if timeline.get("mapping") != "elapsed_time_interpolation":
        errors.append("timestamp_mismatch")
    elapsed = future_image_times - future_image_times[0]
    trajectory_timestamps = trajectory_start * float(scene["dt"]) + elapsed
    if len(trajectory_timestamps) and trajectory_timestamps[-1] > (scene["total_frames"] - 1) * float(scene["dt"]):
        errors.append("timestamp_mismatch")
    coordinate_transform: dict[str, Any] = {"method": transform_cfg.get("method")}
    history_residual: dict[str, Any] = {}
    if transform_cfg.get("method") != "history_ego_boundary_se2":
        errors.append("coordinate_mismatch")
    if ego_tracks:
        ego = ego_tracks[0]
        pose = pose_rows[image_history - 1]
        image_xy = np.asarray(json.loads(pose["ego2global_translation"])[:2], dtype=float)
        image_yaw = quat_to_yaw(json.loads(pose["ego2global_rotation"]))
        source_xy = np.asarray(ego.positions[trajectory_start - 1], dtype=float)
        source_yaw = float(ego.headings[trajectory_start - 1])
        rotation = wrap_angle(image_yaw - source_yaw); c, s = math.cos(rotation), math.sin(rotation)
        matrix = np.array([[c, -s], [s, c]], dtype=float); translation = image_xy - matrix @ source_xy
        coordinate_transform.update({"rotation_rad": rotation, "translation_xy": translation.tolist(),
                                     "matrix_3x3": [[c, -s, float(translation[0])], [s, c, float(translation[1])], [0.0, 0.0, 1.0]],
                                     "estimated_from": "history_ego_boundary_only", "future_reference_used": False})
        history_times = image_timestamps[:image_history] - image_timestamps[image_history - 1]
        source_times = np.asarray(ego.timestamps) - float(ego.timestamps[trajectory_start - 1])
        source_history = np.column_stack([np.interp(history_times, source_times, ego.positions[:, axis]) for axis in (0, 1)])
        transformed = source_history @ matrix.T + translation
        image_history_xy = np.asarray([json.loads(row["ego2global_translation"])[:2] for row in pose_rows[:image_history]], dtype=float)
        center_error = np.linalg.norm(transformed - image_history_xy, axis=1)
        source_heading = np.interp(history_times, source_times, np.unwrap(ego.headings)) + rotation
        image_heading = np.asarray([quat_to_yaw(json.loads(row["ego2global_rotation"])) for row in pose_rows[:image_history]])
        yaw_error = np.abs((source_heading - image_heading + np.pi) % (2 * np.pi) - np.pi)
        history_residual = {"center_mean_m": float(center_error.mean()), "center_max_m": float(center_error.max()),
                            "yaw_mean_rad": float(yaw_error.mean()), "yaw_max_rad": float(yaw_error.max())}
        if center_error.max() > float(transform_cfg.get("maximum_history_center_residual_m", 3.0)): errors.append("coordinate_mismatch")
        if yaw_error.max() > float(transform_cfg.get("maximum_history_yaw_residual_rad", .2)): errors.append("coordinate_mismatch")
    if image.get("scene_id") != trajectory.get("scene_id"):
        warnings.append("source_scene_ids_differ_but_pair_id_is_authoritative")
    if image_history != trajectory_history:
        warnings.append("source_history_lengths_differ_and_are_aligned_by_boundary_time")
    return {"pair_id": pair_id, "image_scene_id": image.get("scene_id"), "trajectory_scene_id": trajectory.get("scene_id"),
            "history_frame_count": image_history, "trajectory_history_frame_count": trajectory_history,
            "future_frame_count": image_future, "image_timestamps": future_image_times.tolist(),
            "trajectory_timestamps": trajectory_timestamps.tolist(),
            "timestamp_offset": float(future_image_times[0] - trajectory_timestamps[0]) if len(future_image_times) else None,
            "timeline_mapping": timeline.get("mapping"), "image_dt": observed_dt, "trajectory_dt": float(scene["dt"]),
            "coordinate_transform": coordinate_transform, "history_alignment_residual": history_residual,
            "pair_validation_status": "valid" if not errors else "invalid_pair_metadata",
            "errors": sorted(set(errors)), "warnings": warnings, "trajectory_scene": scene,
            "image_counts": image_counts, "names_compared_for_pairing": False}
