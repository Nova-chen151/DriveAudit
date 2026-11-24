from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.consistency.mtpc.common import write_json


def compute_expected_motion(projections: list[dict[str, Any]], output: str | Path | None = None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in projections: groups[(row["agent_id"], row["camera_name"])].append(row)
    results = []
    for (agent, camera), rows in groups.items():
        rows.sort(key=lambda row: row["frame_index"])
        for previous, current in zip(rows, rows[1:]):
            if current["frame_index"] != previous["frame_index"] + 1: continue
            flow = np.asarray(current["center_2d"]) - previous["center_2d"]
            previous_box, current_box = previous["clipped_box_2d"], current["clipped_box_2d"]
            previous_area, current_area = max(previous["area"], 1e-6), max(current["area"], 1e-6)
            corner_a = np.asarray([[previous_box[0], previous_box[1]], [previous_box[2], previous_box[1]], [previous_box[2], previous_box[3]], [previous_box[0], previous_box[3]]])
            corner_b = np.asarray([[current_box[0], current_box[1]], [current_box[2], current_box[1]], [current_box[2], current_box[3]], [current_box[0], current_box[3]]])
            results.append({"agent_id": agent, "camera_name": camera, "frame_index": current["frame_index"], "timestamp": current["timestamp"],
                            "history_anchored": current["history_anchored"], "expected_visible": previous["expected_visible"] and current["expected_visible"],
                            "expected_center_flow": flow.tolist(), "expected_corner_flow": (corner_b - corner_a).tolist(),
                            "expected_scale_change": float(np.sqrt(current_area / previous_area)),
                            "expected_rotation_change": 0.0, "expected_motion_direction": float(np.arctan2(flow[1], flow[0])),
                            "expected_motion_magnitude": float(np.linalg.norm(flow)),
                            "expected_entry_direction": float(np.arctan2(flow[1], flow[0])) if not previous["expected_visible"] and current["expected_visible"] else None,
                            "expected_exit_direction": float(np.arctan2(flow[1], flow[0])) if previous["expected_visible"] and not current["expected_visible"] else None,
                            "ego_camera_motion_included": True})
    if output is not None: write_json(Path(output) / "expected_motion.json", results)
    return results

