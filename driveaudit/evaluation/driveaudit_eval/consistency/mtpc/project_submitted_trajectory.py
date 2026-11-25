from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from driveaudit_eval.consistency.shared.conditions import condition_reference_tracks
from driveaudit_eval.consistency.shared.visibility import rectangle_occlusion

from driveaudit_eval.consistency.mtpc.common import CAMERAS, project_world_box, write_csv, write_json


def project_submitted_trajectory(condition_root: str | Path, calibrations: list[dict[str, Any]],
                                 history_agent_ids: set[str], config: dict[str, Any], output: str | Path,
                                 frame_start: int = 10, frame_end: int = 90) -> dict[str, Any]:
    """Stage B only: the submitted trajectory enters here after visual tracks are persisted."""
    references = condition_reference_tracks(condition_root, frame_start, frame_end)
    calibration_map = {(int(row["frame_index"]), row["camera_name"]): row for row in calibrations}
    rows = []
    for track in references:
        for sample in track["observations"]:
            for camera in CAMERAS:
                value = project_world_box(sample["box_bev"], calibration_map[(int(sample["frame_index"]), camera)],
                                          (int(config["image_width"]), int(config["image_height"])), float(config["minimum_depth_m"]))
                if value["projected_valid"]:
                    rows.append({"agent_id": track["agent_id"], "history_anchored": track["agent_id"] in history_agent_ids,
                                 "agent_type": track["agent_type"], "category": track["category"],
                                 "frame_index": int(sample["frame_index"]), "timestamp": float(sample["timestamp"]),
                                 "camera_name": camera, **value})
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[(row["frame_index"], row["camera_name"])].append(row)
    for items in groups.values():
        nearer = []
        for row in sorted(items, key=lambda value: value["mean_depth"]):
            row["occlusion_ratio"] = rectangle_occlusion(row["clipped_box_2d"], nearer); nearer.append(row["clipped_box_2d"])
            row["small_object"] = row["area"] < float(config["minimum_projected_area_px"])
            row["expected_visible"] = not row["small_object"] and row["occlusion_ratio"] <= float(config["maximum_occlusion_ratio"])
            row["visibility_weight"] = max(0.0, 1.0 - row["occlusion_ratio"])
    output = Path(output); write_csv(output / "per_camera_projection.csv", rows)
    agents = sorted({row["agent_id"] for row in rows}); write_json(output / "per_agent_projection.json", {agent: [row for row in rows if row["agent_id"] == agent] for agent in agents})
    visibility = [{key: row[key] for key in ("agent_id", "history_anchored", "frame_index", "timestamp", "camera_name", "expected_visible", "area", "occlusion_ratio", "truncated", "mean_depth")} for row in rows]
    write_json(output / "visibility.json", visibility)
    return {"rows": rows, "reference_tracks": references, "persistent_agent_ids": sorted(history_agent_ids & set(agents)),
            "future_entrant_ids": sorted(set(agents) - history_agent_ids), "stage_b_submission_reference_read": True}

