from __future__ import annotations

import csv
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.perception_trajectory.bev_tracker import BEVTracker
from driveaudit_eval.consistency.shared.common import CAMERA_ORDER, write_json, write_pickle


FORBIDDEN_EXTRACTION_ARGUMENTS = {"submitted_future_trajectory", "future_gt_boxes", "evaluation_results", "match_mapping", "submission_tracks"}


def assert_extraction_isolation(backend: Any) -> None:
    extraction_arguments = set(inspect.signature(extract_future_bev_tracks).parameters)
    backend_arguments = set(inspect.signature(backend.infer).parameters)
    leaked = FORBIDDEN_EXTRACTION_ARGUMENTS & (extraction_arguments | backend_arguments)
    if leaked:
        raise AssertionError(f"future evaluation truth leaked into extraction API: {sorted(leaked)}")


def extract_future_bev_tracks(backend: Any, frames: list[int], image_root: str | Path,
                              calibrations: list[dict[str, Any]], ego_poses: list[dict[str, Any]],
                              tracker: BEVTracker, output: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract from images/calibration/ego only. No submitted-future argument exists by design."""
    assert_extraction_isolation(backend)
    output, image_root = Path(output), Path(image_root)
    calibration_by_frame = {frame: [row for row in calibrations if row["frame_index"] == frame] for frame in frames}
    pose_by_frame = {int(row["frame_index"]): row for row in ego_poses}
    raw_outputs, rows = [], []
    for frame in frames:
        images = {camera: str(image_root / camera / f"{frame:06d}.jpg") for camera in CAMERA_ORDER}
        missing = [camera for camera, path in images.items() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"frame {frame} missing cameras: {missing}")
        pose = pose_by_frame[frame]
        result = backend.infer(images, calibration_by_frame[frame], pose, float(pose["timestamp"]))
        raw_outputs.append(result)
        tracked = tracker.update(result["detections"], float(pose["timestamp"]), frame)
        for item in tracked:
            box = item["box_bev"]
            rows.append({"frame_index": frame, "timestamp": item["timestamp"], "image_track_id": item["image_track_id"],
                         "category": item["category"], "x": box["x"], "y": box["y"], "length": box["length"],
                         "width": box["width"], "yaw": box["yaw"], "vx": item["velocity"][0],
                         "vy": item["velocity"][1], "score": item["score"]})
    tracks = [track for track in tracker.export() if track["track_length"] > 0]
    write_pickle(output / "raw_detections.pkl", raw_outputs)
    write_json(output / "bev_tracks.json", tracks)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_frame_detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["frame_index", "timestamp", "image_track_id"])
        writer.writeheader(); writer.writerows(rows)
    model_times = [float(row.get("profile", {}).get("model_inference_s", row.get("profile", {}).get("inference_seconds", 0.0)) or 0.0) for row in raw_outputs]
    peak_memory = [float(row.get("profile", {}).get("peak_gpu_memory_mb", 0.0) or 0.0) for row in raw_outputs]
    profile = {"frames": len(frames), "detections": sum(len(row["detections"]) for row in raw_outputs),
               "tracks": len(tracks), "mean_inference_seconds": float(np.mean(model_times)) if model_times else "not_available",
               "total_inference_seconds": float(np.sum(model_times)), "peak_gpu_memory_mib": max(peak_memory, default="not_available"),
               "backend_result_status": "formal_reference_backend" if getattr(backend, "formal_backend", False) else
                                        ("formal_vad_detection_cache" if getattr(backend, "formal_detection_cache", False) else "diagnostic_only")}
    write_json(output / "inference_profile.json", profile)
    return tracks, profile
