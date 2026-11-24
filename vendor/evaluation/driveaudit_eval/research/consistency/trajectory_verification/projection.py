from __future__ import annotations

from pathlib import Path
from typing import Any

from driveaudit_eval.research.consistency.perception_trajectory.bevformer_postprocess import condition_reference_tracks

from driveaudit_eval.research.consistency.trajectory_verification.common import CAMERAS, project_world_box, write_csv, write_json
from driveaudit_eval.research.consistency.trajectory_verification.visibility import apply_visibility


def _invalid_projection(reason: str) -> dict[str, Any]:
    return {"projected_valid": False, "projection_reason": reason, "in_front_of_camera": reason != "behind_camera",
            "intersects_image": False, "projected_polygon": [], "projected_corners": [],
            "projected_box_2d": [], "raw_projected_box_2d": [], "projected_center": [],
            "projected_area": 0.0, "raw_projected_area": 0.0, "projected_depth": None,
            "truncated_ratio": 1.0, "truncated": False, "out_of_image": reason == "outside_image"}


def project_tracks(condition_root: str | Path, calibrations: list[dict[str, Any]], frame_start: int,
                   frame_end: int, history_agent_ids: set[str], config: dict[str, Any],
                   output: str | Path | None = None) -> dict[str, Any]:
    tracks = condition_reference_tracks(condition_root, frame_start, frame_end)
    calibration_map = {(int(row["frame_index"]), row["camera_name"]): row for row in calibrations}
    image_size = (int(config["image_width"]), int(config["image_height"]))
    rows: list[dict[str, Any]] = []
    for track in tracks:
        for sample in track["observations"]:
            frame = int(sample["frame_index"])
            for camera in CAMERAS:
                value = project_world_box(sample["box_bev"], calibration_map[(frame, camera)], image_size,
                                          float(config["minimum_depth_m"]))
                base = {"agent_id": track["agent_id"], "agent_type": track["agent_type"],
                        "category": track["category"], "history_anchored": track["agent_id"] in history_agent_ids,
                        "frame_index": frame, "timestamp": float(sample["timestamp"]), "camera_name": camera,
                        "box_reference": sample["box_bev"]}
                if not value.get("projected_valid"):
                    rows.append({**base, **_invalid_projection(value.get("reason", "projection_invalid"))})
                    continue
                raw = value["box_2d"]
                raw_area = max(0.0, raw[2] - raw[0]) * max(0.0, raw[3] - raw[1])
                clipped_area = float(value["area"])
                rows.append({**base, "projected_valid": True, "projection_reason": value["reason"],
                             "in_front_of_camera": True, "intersects_image": clipped_area > 0,
                             "projected_polygon": value["projected_polygon"],
                             "projected_corners": value["projected_corners"],
                             "raw_projected_box_2d": raw, "projected_box_2d": value["clipped_box_2d"],
                             "projected_center": value["center_2d"], "projected_area": clipped_area,
                             "raw_projected_area": raw_area, "projected_depth": value["mean_depth"],
                             "truncated_ratio": max(0.0, 1.0 - clipped_area / max(raw_area, 1e-6)),
                             "truncated": bool(value["truncated"]), "out_of_image": False})
    apply_visibility(rows, config)
    result = {"rows": rows, "tracks": tracks,
              "persistent_agent_ids": sorted({row["agent_id"] for row in rows if row["history_anchored"]}),
              "future_entrant_ids": sorted({row["agent_id"] for row in rows if not row["history_anchored"]})}
    if output is not None:
        output = Path(output)
        write_csv(output / "per_camera_projection.csv", rows)
        write_json(output / "per_agent_projection.json", {agent: [row for row in rows if row["agent_id"] == agent]
                                                           for agent in sorted({row["agent_id"] for row in rows})})
        write_json(output / "visibility.json", [{key: row.get(key) for key in
                   ("agent_id", "agent_type", "history_anchored", "frame_index", "timestamp", "camera_name",
                    "projected_valid", "expected_visible", "observability_status", "projected_area",
                    "truncated_ratio", "occlusion_ratio", "projected_depth", "expected_camera_count")}
                   for row in rows])
    return result

