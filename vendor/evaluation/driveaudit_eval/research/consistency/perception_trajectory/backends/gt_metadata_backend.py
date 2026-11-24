from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory.backends.base_backend import BaseBackend
from driveaudit_eval.consistency.shared.common import local_box_to_world, read_csv, true_cell


class GTMetadataBackend(BaseBackend):
    """Use boxes attached to the GT image sequence; this is a reference backend, not pixel inference."""

    def __init__(self, conditions_root: str | Path, tracked_classes: list[str]):
        self.boxes: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(Path(conditions_root) / "boxes.csv"):
            if true_cell(row.get("valid_flag", True)) and row["class_name"] in tracked_classes:
                self.boxes[int(row["frame_index"])].append(row)
        self.tracked_classes = set(tracked_classes)
        self.formal_backend = True
        self.scene_id = ""

    def reset(self, scene_id: str) -> None:
        self.scene_id = scene_id

    def initialize(self, history_frames: list[dict[str, Any]], history_boxes: list[dict[str, Any]],
                   calibration: list[dict[str, Any]], ego_poses: list[dict[str, Any]]) -> dict[str, Any]:
        return {"status": "gt_metadata_reference", "history_frames_consumed": len(history_frames),
                "history_boxes": len(history_boxes), "baseline_type": "metadata_self_consistency",
                "history_metrics": {"history_detection_recall": 1.0, "history_precision": 1.0,
                                    "history_center_ade": 0.0, "history_bev_iou": 1.0,
                                    "history_heading_error": 0.0, "history_velocity_error": 0.0,
                                    "history_identity_consistency": 1.0, "history_baseline_idf1": 1.0}}

    def infer(self, frame_images: dict[str, str], calibration: list[dict[str, Any]],
              ego_pose: dict[str, Any], timestamp: float) -> dict[str, Any]:
        frame = int(ego_pose["frame_index"])
        detections = []
        for row in self.boxes.get(frame, []):
            local = {key: float(row[key]) for key in ("x", "y", "z", "width", "length", "height", "yaw", "vx", "vy")}
            world = local_box_to_world(local, ego_pose)
            detections.append({"temporary_track_id": f"metadata:{row['obj_id']}", "category": row["class_name"],
                               "score": 1.0,
                               "box_bev": {key: float(world[key]) for key in ("x", "y", "length", "width", "yaw")},
                               "velocity": [float(world["vx"]), float(world["vy"])], "embedding": [],
                               "camera_support": [],
                               "uncertainty": {"position_std": 0.0, "heading_std": 0.0}})
        return {"timestamp": float(timestamp), "frame_index": frame, "detections": detections,
                "profile": {"inference_seconds": 0.0, "source": "gt_image_attached_input_metadata",
                            "pixel_inference_performed": False}}
