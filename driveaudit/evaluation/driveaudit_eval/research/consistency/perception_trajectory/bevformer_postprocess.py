from __future__ import annotations
from driveaudit_eval.consistency.shared.conditions import _condition_poses, condition_reference_tracks

import csv
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from driveaudit_eval.research.consistency.perception_trajectory.backends.bevformer_backend import BEVFORMER_CLASSES
from driveaudit_eval.research.consistency.perception_trajectory.bev_tracker import BEVTracker
from driveaudit_eval.consistency.shared.categories import DYNAMIC_CLASSES, semantic_category
from driveaudit_eval.consistency.shared.common import json_cell, local_box_to_world, read_csv, true_cell, write_json




def bevformer_box_to_local(values: np.ndarray) -> dict[str, float]:
    """Convert MMD3D bottom-center z output to the common box-center convention."""
    return {"x": values[0], "y": values[1], "z": values[2] + values[5] * 0.5, "width": values[3],
            "length": values[4], "height": values[5], "yaw": values[6],
            "vx": values[7] if len(values) > 7 else 0.0, "vy": values[8] if len(values) > 8 else 0.0}




def _anchor_ids(tracker: BEVTracker, references: list[dict[str, Any]], maximum_distance: float = 8.0) -> int:
    samples = [(track["agent_id"], track["category"], track["observations"][-1]) for track in references if track["observations"]]
    tracks = list(tracker.tracks.values())
    if not samples or not tracks:
        return 0
    costs = np.full((len(tracks), len(samples)), 1e6)
    for i, track in enumerate(tracks):
        for j, (_, category, sample) in enumerate(samples):
            if semantic_category(track.category) == semantic_category(category):
                costs[i, j] = np.linalg.norm(track.state[:2] - [sample["box_bev"]["x"], sample["box_bev"]["y"]])
    anchored = 0
    for i, j in zip(*linear_sum_assignment(costs)):
        if costs[i, j] <= maximum_distance:
            tracks[int(i)].history_agent_id = samples[int(j)][0]; anchored += 1
    return anchored


def detections_to_tracks(raw_output: str | Path, condition_root: str | Path, tracker_config: dict[str, Any],
                         score_threshold: float, history_frames: int, output_dir: str | Path) -> dict[str, Any]:
    with Path(raw_output).open("rb") as handle:
        raw = pickle.load(handle)
    poses = _condition_poses(condition_root)
    tracker = BEVTracker(tracker_config); tracker.reset(Path(output_dir).name)
    class_counts, dynamic_counts, ignored_counts = Counter(), Counter(), Counter()
    csv_rows, anchored = [], 0
    history_reference = condition_reference_tracks(condition_root, history_frames - 1, history_frames)
    for frame_result in raw:
        frame = int(frame_result["frame_index"]); pose = poses[frame]; detections = []
        for box_values, score, label in zip(frame_result["boxes"], frame_result["scores"], frame_result["labels"]):
            category = BEVFORMER_CLASSES[int(label)]; class_counts[category] += 1
            if float(score) < score_threshold:
                continue
            if category not in DYNAMIC_CLASSES:
                ignored_counts[category] += 1; continue
            values = np.asarray(box_values, dtype=float)
            if len(values) < 7 or not np.isfinite(values).all():
                continue
            local = bevformer_box_to_local(values)
            world = local_box_to_world(local, pose); dynamic_counts[category] += 1
            detection = {"temporary_track_id": "unassigned", "category": category, "score": float(score),
                         "box_bev": {key: float(world[key]) for key in ("x", "y", "z", "length", "width", "height", "yaw")},
                         "velocity": [float(world["vx"]), float(world["vy"])], "embedding": [], "camera_support": [],
                         "uncertainty": {"position_std": "not_available", "heading_std": "not_available"}}
            detections.append(detection)
            csv_rows.append({"frame_index": frame, "timestamp": pose["timestamp"], "class": category, "score": float(score),
                             "local_x": local["x"], "local_y": local["y"], "world_x": world["x"], "world_y": world["y"],
                             "length": world["length"], "width": world["width"], "yaw": world["yaw"]})
        tracker.update(detections, pose["timestamp"], frame)
        if frame == history_frames - 1:
            anchored = _anchor_ids(tracker, history_reference)
    all_tracks = tracker.export(confirmed_only=bool(tracker_config.get("export_confirmed_only", False))); future_tracks = []
    for track in all_tracks:
        observations = [row for row in track["observations"] if int(row["frame_index"]) >= history_frames]
        if observations:
            copied = dict(track); copied["observations"] = observations; copied["track_length"] = len(observations)
            copied["mean_confidence"] = float(np.mean([row["score"] for row in observations])); future_tracks.append(copied)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "tracks.json", future_tracks)
    fields = list(csv_rows[0]) if csv_rows else ["frame_index", "status"]
    with (output_dir / "detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(csv_rows)
    audit = {"input_detections_by_class": dict(class_counts), "filtered_dynamic_detections": dict(dynamic_counts),
             "ignored_static_detections": dict(ignored_counts), "history_id_anchors": anchored,
             "dynamic_category_mapping": {category: semantic_category(category) for category in sorted(DYNAMIC_CLASSES)},
             "ignored_static_classes": ["barrier", "traffic_cone"],
             "box_z_convention": "center_converted_from_bevformer_bottom_center",
             "tracker_diagnostics": tracker.diagnostics,
             "future_track_count": len(future_tracks), "future_gt_used_to_update_tracker": False,
             "motion_prediction_used": False, "planning_head_used": False}
    write_json(output_dir / "class_audit.json", audit)
    return {"tracks": future_tracks, "audit": audit, "all_tracks": all_tracks}
