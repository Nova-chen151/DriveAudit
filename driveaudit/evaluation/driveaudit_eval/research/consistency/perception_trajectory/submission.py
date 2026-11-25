from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from driveaudit_eval.trajectory.io import load_tracks


TYPE_TO_CATEGORY = {"vehicle": "car", "ego": "car", "pedestrian": "pedestrian", "bicycle": "bicycle"}


def load_paired_submission(scene: dict[str, Any], future_frame_indices: list[int], poses: list[dict[str, Any]],
                           history_samples: list[dict[str, Any]], pair_validation: dict[str, Any]) -> list[dict[str, Any]]:
    declarations = {"coordinate_system": scene["coordinate_system"], "heading_unit": scene["heading_unit"],
                    "history_frames": scene.get("history_frames"), "ego_ids": ["Ego", "ego"]}
    tracks = load_tracks(scene["submission_file"], "generated", declarations)
    tracks = [track for track in tracks if track.scene_id == scene["scene_id"]]
    shape_samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for sample in history_samples:
        box = sample["box_bev"]
        if "z" in box and "height" in box: shape_samples[sample["category"]].append((box["z"], box["height"]))
    shape_defaults = {category: np.median(values, axis=0) for category, values in shape_samples.items()}
    pose_by_frame = {row["frame_index"]: row for row in poses}
    query_times = np.asarray(pair_validation["trajectory_timestamps"], dtype=float)
    transform = np.asarray(pair_validation["coordinate_transform"]["matrix_3x3"], dtype=float)
    rotation = float(pair_validation["coordinate_transform"]["rotation_rad"])
    result = []
    for track in tracks:
        if track.coordinate_system != "world" or track.metadata.get("heading_unit") != "rad":
            raise ValueError(f"{track.agent_id}: evaluator requires explicit world/rad trajectory")
        if track.agent_type == "ego":
            continue
        category = TYPE_TO_CATEGORY.get(track.agent_type, track.agent_type)
        z_height = shape_defaults.get(category)
        observations = []
        for offset, (frame, query_time) in enumerate(zip(future_frame_indices, query_times)):
            right = int(np.searchsorted(track.timestamps, query_time, side="left")); left = max(0, right - 1)
            right = min(right, len(track.timestamps) - 1)
            if not (track.valid_mask[left] and track.valid_mask[right]):
                continue
            if right == left: alpha = 0.0
            else: alpha = float((query_time - track.timestamps[left]) / (track.timestamps[right] - track.timestamps[left]))
            source_position = (1 - alpha) * track.positions[left] + alpha * track.positions[right]
            world_position = transform @ np.array([source_position[0], source_position[1], 1.0])
            heading_delta = (float(track.headings[right]) - float(track.headings[left]) + np.pi) % (2 * np.pi) - np.pi
            heading = float(track.headings[left]) + alpha * heading_delta + rotation
            box = {"x": float(world_position[0]), "y": float(world_position[1]),
                   "length": float(track.shape["length"]), "width": float(track.shape["width"]),
                   "yaw": float((heading + np.pi) % (2 * np.pi) - np.pi)}
            if z_height is not None: box.update(z=float(z_height[0]), height=float(z_height[1]))
            if right != left:
                source_velocity = (track.positions[right] - track.positions[left]) / (track.timestamps[right] - track.timestamps[left])
                velocity = (transform[:2, :2] @ source_velocity).astype(float).tolist()
            else:
                velocity = [0.0, 0.0]
            observations.append({"frame_index": frame, "timestamp": pose_by_frame[frame]["timestamp"],
                                 "source_trajectory_timestamp": float(query_time),
                                 "box_bev": box, "velocity": velocity, "expected_visible": False,
                                 "visibility_status": "not_evaluated"})
        history_present = bool(np.any(track.valid_mask[np.asarray(track.timestamps) < query_times[0]]))
        result.append({"agent_id": track.agent_id, "category": category, "agent_type": track.agent_type,
                       "history_present": history_present, "observations": observations})
    return result
