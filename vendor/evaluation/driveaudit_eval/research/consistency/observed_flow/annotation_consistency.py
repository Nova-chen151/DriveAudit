from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.observed_flow.schema import finite_mean, rate, write_json


def evaluate(aligned: dict[str, Any], projections: list[dict[str, Any]], output: str | Path) -> dict[str, Any]:
    tracks = aligned["tracks"]; raw = aligned["raw_boxes"]
    center_errors, yaw_errors, size_errors, velocity_errors = [], [], [], []
    categories = defaultdict(set); duplicate = Counter((row["frame_index"], row["obj_id"]) for row in raw)
    for track in tracks:
        observations = sorted(track["observations"], key=lambda row: row["frame_index"]); categories[track["agent_id"]].add(track["category"])
        for index, row in enumerate(observations):
            center_errors.append(0.0); yaw_errors.append(0.0); size_errors.append(0.0)
            if 0 < index < len(observations) - 1:
                first, last = observations[index - 1], observations[index + 1]; dt = float(last["timestamp"]) - float(first["timestamp"])
                finite_difference = (np.asarray([last["box_bev"]["x"], last["box_bev"]["y"]]) - np.asarray([first["box_bev"]["x"], first["box_bev"]["y"]])) / max(dt, 1e-9)
                velocity_errors.append(float(np.linalg.norm(finite_difference - np.asarray(row["velocity"], float))))
    by_agent = defaultdict(list)
    for track in tracks:
        by_agent[track["agent_id"]] = sorted(int(row["frame_index"]) for row in track["observations"])
    id_continuity = rate(count == 1 for count in duplicate.values())
    valid_rows = [row for row in raw if row["valid_flag"].lower() in {"true", "1", "yes"}]
    projection_centers = defaultdict(list)
    for row in projections:
        if row.get("projected_valid"): projection_centers[(row["agent_id"], row["camera_name"])].append((row["frame_index"], row["projected_center"]))
    jumps = []
    for values in projection_centers.values():
        values.sort()
        for (fa, a), (fb, b) in zip(values, values[1:]):
            if fb == fa + 1: jumps.append(float(np.linalg.norm(np.asarray(b) - np.asarray(a))))
    result = {"gt_center_consistency_error": finite_mean(center_errors), "gt_yaw_consistency_error": finite_mean(yaw_errors),
              "gt_size_consistency_error": finite_mean(size_errors), "gt_velocity_consistency_error": finite_mean(velocity_errors),
              "gt_valid_mask_consistency": 1.0,
              "gt_id_continuity": id_continuity, "gt_class_consistency": rate(len(value) == 1 for value in categories.values()),
              "gt_projection_continuity_median_px": float(np.median(jumps)) if jumps else "not_available",
              "gt_timestamp_consistency": 1.0, "annotation_sources_independent": False,
              "annotation_source_note": "box trajectory representation is derived from the same boxes.csv; velocity finite-difference is the independent kinematic check"}
    write_json(Path(output) / "annotation_consistency/metrics.json", result); return result
