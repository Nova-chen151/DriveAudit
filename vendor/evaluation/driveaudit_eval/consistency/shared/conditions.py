from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from driveaudit_eval.consistency.shared.categories import DYNAMIC_CLASSES, semantic_category
from driveaudit_eval.consistency.shared.common import json_cell, local_box_to_world, read_csv, true_cell, write_json

def _condition_poses(condition_root: str | Path) -> dict[int, dict[str, Any]]:
    root = Path(condition_root)
    timestamps = {int(row["frame_index"]): float(row["timestamp"]) / 1e6 for row in read_csv(root / "frames.csv")}
    poses = {}
    for row in read_csv(root / "ego_pose.csv"):
        frame = int(row["frame_index"])
        poses[frame] = {"frame_index": frame, "timestamp": timestamps[frame],
                        "translation": json_cell(row["ego2global_translation"]),
                        "rotation": json_cell(row["ego2global_rotation"]),
                        "lidar2ego_translation": json_cell(row["lidar2ego_translation"]),
                        "lidar2ego_rotation": json_cell(row["lidar2ego_rotation"])}
    return poses


def condition_reference_tracks(condition_root: str | Path, frame_start: int, frame_end: int,
                               detector_range_m: float = 51.2) -> list[dict[str, Any]]:
    poses = _condition_poses(condition_root)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    categories: dict[str, str] = {}
    history_ids = set()
    with (Path(condition_root) / "boxes.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame, category = int(row["frame_index"]), row["class_name"]
            if frame >= frame_end: break  # Sorted generation conditions: future instruction is not read during history anchoring.
            if frame < frame_start:
                if category in DYNAMIC_CLASSES and true_cell(row.get("valid_flag", True)): history_ids.add(row["obj_id"])
                continue
            if category not in DYNAMIC_CLASSES or not true_cell(row.get("valid_flag", True)): continue
            local = {key: float(row[key]) for key in ("x", "y", "z", "width", "length", "height", "yaw", "vx", "vy")}
            if row.get("box_coordinate_system", "").startswith("ego_") and all(row.get(key) not in (None, "") for key in ("global_x", "global_y", "global_z", "global_yaw")):
                from driveaudit_eval.consistency.shared.common import quat_rotation
                velocity = quat_rotation(poses[frame]["rotation"])[:2, :2] @ np.asarray([local["vx"], local["vy"]])
                world = {**local, "x": float(row["global_x"]), "y": float(row["global_y"]), "z": float(row["global_z"]),
                         "yaw": float(row["global_yaw"]), "vx": float(velocity[0]), "vy": float(velocity[1])}
            else:
                world = local_box_to_world(local, poses[frame])
            box = {key: world[key] for key in ("x", "y", "z", "width", "length", "height", "yaw")}
            grouped[row["obj_id"]].append({"frame_index": frame, "timestamp": poses[frame]["timestamp"],
                                           "box_bev": box, "velocity": [world["vx"], world["vy"]],
                                           "expected_visible": False, "visibility_status": "not_evaluated",
                                           "out_of_detector_range": abs(local["x"]) > detector_range_m or abs(local["y"]) > detector_range_m})
            categories[row["obj_id"]] = category
    return [{"agent_id": agent_id, "category": categories[agent_id],
             "agent_type": semantic_category(categories[agent_id]),
             "history_present": agent_id in history_ids, "observations": sorted(rows, key=lambda item: item["frame_index"])}
            for agent_id, rows in grouped.items()]

