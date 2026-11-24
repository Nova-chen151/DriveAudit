from __future__ import annotations

from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE
from driveaudit_eval.research.consistency.perception_trajectory.bev_tracker import BEVTracker
from driveaudit_eval.consistency.shared.common import json_cell, local_box_to_world, read_csv, true_cell, write_json, write_pickle


def load_conditions(condition_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(condition_root)
    frames = read_csv(root / "frames.csv")
    timestamps = {int(row["frame_index"]): float(row["timestamp"]) / 1e6 for row in frames}
    poses = []
    for row in read_csv(root / "ego_pose.csv"):
        frame_index = int(row["frame_index"])
        poses.append({"frame_index": frame_index, "timestamp": timestamps[frame_index],
                      "translation": json_cell(row["ego2global_translation"]), "rotation": json_cell(row["ego2global_rotation"]),
                      "lidar2ego_translation": json_cell(row["lidar2ego_translation"]),
                      "lidar2ego_rotation": json_cell(row["lidar2ego_rotation"])})
    calibrations = []
    for row in read_csv(root / "camera_calibration.csv"):
        parsed = dict(row)
        parsed["frame_index"] = int(row["frame_index"])
        parsed["camera_name"] = row["camera"]
        for field in ("sensor2ego_translation", "sensor2ego_rotation", "ego2global_translation", "ego2global_rotation",
                      "sensor2lidar_rotation", "sensor2lidar_translation", "camera_intrinsics"):
            parsed[field] = json_cell(row[field])
        calibrations.append(parsed)
    boxes = [row for row in read_csv(root / "boxes.csv") if true_cell(row.get("valid_flag", True))]
    return {"frames": frames, "poses": poses, "calibrations": calibrations, "boxes": boxes}


def known_history_samples(conditions: dict[str, list[dict[str, Any]]], history_frames: int) -> list[dict[str, Any]]:
    poses = {int(row["frame_index"]): row for row in conditions["poses"]}
    samples = []
    for row in conditions["boxes"]:
        frame = int(row["frame_index"])
        if frame >= history_frames:
            continue
        local = {key: float(row[key]) for key in ("x", "y", "z", "width", "length", "height", "yaw", "vx", "vy")}
        world = local_box_to_world(local, poses[frame])
        samples.append({"frame_index": frame, "timestamp": poses[frame]["timestamp"], "agent_id": row["obj_id"],
                        "category": row["class_name"], "box_bev": {key: world[key] for key in ("x", "y", "z", "height", "length", "width", "yaw")},
                        "velocity": [world["vx"], world["vy"]]})
    return samples


def warmup_history(backend: Any, tracker: BEVTracker, conditions: dict[str, list[dict[str, Any]]],
                   history_frames: int, output: str | Path) -> dict[str, Any]:
    output = Path(output)
    samples = known_history_samples(conditions, history_frames)
    frames = [{"frame_index": i, "timestamp": conditions["poses"][i]["timestamp"]} for i in range(history_frames)]
    backend_status = backend.initialize(frames, samples,
                                        [row for row in conditions["calibrations"] if row["frame_index"] < history_frames],
                                        conditions["poses"][:history_frames])
    trackable = [row for row in samples if row["category"] in getattr(backend, "tracked_classes", {row["category"]})]
    initialized = tracker.initialize_history(trackable, history_frames - 1)
    reference_metrics = backend_status.get("history_metrics", {})
    metrics = {"history_frames_used": history_frames, "known_history_box_count": len(samples),
               "tracker_initialized_agents": initialized, "history_detection_recall": NOT_AVAILABLE,
               "history_precision": NOT_AVAILABLE, "history_center_ade": NOT_AVAILABLE,
               "history_bev_iou": NOT_AVAILABLE, "history_heading_error": NOT_AVAILABLE,
               "history_velocity_error": NOT_AVAILABLE, "history_identity_consistency": NOT_AVAILABLE,
               "history_baseline_idf1": NOT_AVAILABLE, "history_gate": "failed_history_perception_not_available",
               "backend_initialize": backend_status}
    if reference_metrics:
        metrics.update(reference_metrics)
        metrics["history_gate"] = "passed_metadata_self_consistency"
    calibration = {"rigid_se2_residual": NOT_AVAILABLE, "mean_x_bias": NOT_AVAILABLE,
                   "mean_y_bias": NOT_AVAILABLE, "yaw_bias": NOT_AVAILABLE,
                   "estimated_time_offset_frames": NOT_AVAILABLE, "calibration_applied": False,
                   "reason": ("metadata reference is self-consistent; no detector bias calibration applies" if reference_metrics else
                              "history perception output unavailable; known boxes were used only for ID/Kalman warmup")}
    write_json(output / "history_metrics.json", metrics)
    write_json(output / "calibration.json", calibration)
    write_pickle(output / "history_tracks.pkl", samples)
    return metrics


def history_quality_gate(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    required = ("history_detection_recall", "history_detection_precision", "history_center_distance_ap",
                "history_center_ade", "history_heading_error", "history_velocity_error", "history_baseline_idf1")
    if any(not isinstance(metrics.get(key), (float, int)) for key in required):
        return {"passed": False, "status": "unscorable_extractor_failure", "reason": "history baseline metric unavailable"}
    checks = {"recall": metrics["history_detection_recall"] >= thresholds["minimum_recall"],
              "precision": metrics["history_detection_precision"] >= thresholds["minimum_precision"],
              "center_distance_ap": metrics["history_center_distance_ap"] >= thresholds["minimum_center_distance_ap"],
              "mATE": metrics["history_center_ade"] <= thresholds["maximum_center_ade_m"],
              "mAOE": metrics["history_heading_error"] <= thresholds["maximum_heading_mae_rad"],
              "mAVE": metrics["history_velocity_error"] <= thresholds["maximum_velocity_error_mps"],
              "IDF1": metrics["history_baseline_idf1"] >= thresholds["minimum_idf1"]}
    passed = all(checks.values())
    return {"passed": passed, "status": "passed" if passed else "unscorable_extractor_failure",
            "reason": "thresholds satisfied" if passed else "history extractor quality below threshold",
            "checks": checks, "provisional_thresholds": True}
