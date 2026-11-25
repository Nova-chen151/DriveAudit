from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory.backends.base_backend import BaseBackend
from driveaudit_eval.consistency.shared.common import local_box_to_world


CLASS_NAMES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
               "motorcycle", "bicycle", "pedestrian", "traffic_cone"]


class CachedVADPerceptionBackend(BaseBackend):
    """Read image-derived VAD frame detections; motion/planning tensors are intentionally ignored."""

    def __init__(self, cache: str | Path, score_threshold: float = 0.2,
                 tracked_classes: list[str] | None = None):
        with Path(cache).open("rb") as handle:
            records = pickle.load(handle)
        self.records = {int(record["input_audit"]["frame_id"]): record for record in records}
        self.score_threshold = float(score_threshold)
        self.tracked_classes = set(tracked_classes or CLASS_NAMES)
        self.formal_detection_cache = True
        self.scene_id = ""

    def reset(self, scene_id: str) -> None:
        self.scene_id = scene_id

    def initialize(self, history_frames: list[dict[str, Any]], history_boxes: list[dict[str, Any]],
                   calibration: list[dict[str, Any]], ego_poses: list[dict[str, Any]]) -> dict[str, Any]:
        return {"status": "history_perception_not_cached", "history_frames_consumed": len(history_frames),
                "history_boxes_received_for_tracker_only": len(history_boxes)}

    def infer(self, frame_images: dict[str, str], calibration: list[dict[str, Any]],
              ego_pose: dict[str, Any], timestamp: float) -> dict[str, Any]:
        started = time.perf_counter()
        frame_index = int(ego_pose["frame_index"])
        record = self.records.get(frame_index)
        if record is None:
            raise KeyError(f"cached VAD frame {frame_index} missing")
        raw = record["raw_scene"]
        if not all(key in raw for key in ("boxes_3d", "scores_3d", "labels_3d")):
            raise ValueError("cache_missing_detection_head_outputs")
        detections = []
        for index, (values, score, label) in enumerate(zip(raw["boxes_3d"], raw["scores_3d"], raw["labels_3d"])):
            category = CLASS_NAMES[int(label)]
            if float(score) < self.score_threshold or category not in self.tracked_classes:
                continue
            local = {"x": values[0], "y": values[1], "z": values[2], "width": values[3],
                     "length": values[4], "height": values[5], "yaw": values[6],
                     "vx": values[7], "vy": values[8]}
            world = local_box_to_world(local, ego_pose)
            detections.append({
                "temporary_track_id": f"det:{frame_index}:{index}", "category": category, "score": float(score),
                "box_bev": {key: float(world[key]) for key in ("x", "y", "length", "width", "yaw")},
                "velocity": [float(world["vx"]), float(world["vy"])], "embedding": [],
                "camera_support": [], "uncertainty": {"position_std": "not_available", "heading_std": "not_available"},
            })
        profile = record.get("profile", {})
        return {"timestamp": float(timestamp), "frame_index": frame_index, "detections": detections,
                "profile": {**profile, "cache_decode_seconds": time.perf_counter() - started,
                            "source": "vad_detection_cache_boxes_only",
                            "discarded_outputs": ["raw_planning", "planning_traj", "raw_scene.trajs_3d"]}}
