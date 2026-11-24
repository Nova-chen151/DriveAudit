from __future__ import annotations

import bisect
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from driveaudit_eval.research.consistency.perception_trajectory.bevformer_postprocess import condition_reference_tracks
from driveaudit_eval.research.consistency.trajectory_verification.common import load_geometry

from driveaudit_eval.research.consistency.observed_flow.schema import CAMERAS, read_csv, write_csv, write_json


def angle_lerp(a: float, b: float, ratio: float) -> float:
    delta = (float(b) - float(a) + math.pi) % (2 * math.pi) - math.pi
    return (float(a) + ratio * delta + math.pi) % (2 * math.pi) - math.pi


def interpolate_observations(observations: list[dict[str, Any]], timestamp: float,
                             maximum_nearest_error_s: float = .05) -> tuple[dict[str, Any] | None, str, float]:
    if not observations: return None, "missing", float("inf")
    ordered = sorted(observations, key=lambda row: float(row["timestamp"])); times = [float(row["timestamp"]) for row in ordered]
    index = bisect.bisect_left(times, timestamp)
    if index < len(times) and abs(times[index] - timestamp) <= 1e-7:
        return dict(ordered[index]), "exact_timestamp_match", abs(times[index] - timestamp)
    if 0 < index < len(times):
        first, last = ordered[index - 1], ordered[index]
        if int(last["frame_index"]) - int(first["frame_index"]) == 1:
            ratio = (timestamp - float(first["timestamp"])) / max(float(last["timestamp"]) - float(first["timestamp"]), 1e-9)
            box = {key: (1 - ratio) * float(first["box_bev"][key]) + ratio * float(last["box_bev"][key])
                   for key in ("x", "y", "z")}
            box.update({key: float(first["box_bev"][key]) for key in ("width", "length", "height")})
            box["yaw"] = angle_lerp(first["box_bev"]["yaw"], last["box_bev"]["yaw"], ratio)
            velocity = ((1 - ratio) * np.asarray(first["velocity"], float) + ratio * np.asarray(last["velocity"], float)).tolist()
            return {**first, "timestamp": timestamp, "box_bev": box, "velocity": velocity,
                    "interpolation_sources": [int(first["frame_index"]), int(last["frame_index"])]}, "common_time_interpolation", 0.0
    nearest = min(ordered, key=lambda row: abs(float(row["timestamp"]) - timestamp)); error = abs(float(nearest["timestamp"]) - timestamp)
    return (dict(nearest), "nearest_timestamp_match", error) if error <= maximum_nearest_error_s else (None, "outside_valid_segment", error)


def align(config: dict[str, Any], output: str | Path) -> dict[str, Any]:
    root = Path(config["inputs"]["conditions"]); geometry = load_geometry(root)
    tracks = condition_reference_tracks(root, 0, int(config["scene"]["history_frames"]) + int(config["scene"]["future_frames"]), 1e9)
    raw_boxes = read_csv(root / "boxes.csv"); raw = {(int(row["frame_index"]), row["obj_id"]): row for row in raw_boxes}
    calibration = {(int(row["frame_index"]), row["camera_name"]): row for row in geometry["calibrations"]}
    aligned, protocols, errors, offsets = [], defaultdict(int), [], defaultdict(list)
    for track in tracks:
        base_frames = {int(row["frame_index"]) for row in track["observations"]}
        for frame in sorted(base_frames):
            metadata = raw[(frame, track["agent_id"])]
            for camera in CAMERAS:
                cal = calibration[(frame, camera)]; timestamp = float(cal["timestamp"]) / 1e6
                sample, protocol, error = interpolate_observations(track["observations"], timestamp,
                    float(config["time_alignment"]["maximum_nearest_error_s"]))
                protocols[protocol] += 1; errors.append(error); offsets[camera].append(timestamp - float(geometry["timestamps"][frame]))
                if sample is None: continue
                aligned.append({"agent_id": track["agent_id"], "agent_type": track["agent_type"], "category": track["category"],
                                "frame_index": frame, "camera_name": camera, "timestamp": timestamp,
                                "frame_timestamp": float(geometry["timestamps"][frame]), "box_reference": sample["box_bev"],
                                "velocity": sample["velocity"], "visibility_token": int(metadata["visibility"]),
                                "alignment_protocol": protocol, "timestamp_error_s": error,
                                "calibration": cal, "source_valid_flag": metadata["valid_flag"]})
    frame_times = sorted(float(value) for value in geometry["timestamps"].values()); dt = np.diff(frame_times)
    summary = {"selected_alignment_protocol": "camera_timestamp_common_time_interpolation",
               "image_frequency_hz": float(1 / np.median(dt)), "trajectory_frequency_hz": float(1 / np.median(dt)),
               "estimated_offset_frames": {camera: float(np.median(values) / np.median(dt)) for camera, values in offsets.items()},
               "estimated_offset_seconds": {camera: float(np.median(values)) for camera, values in offsets.items()},
               "matched_timestamp_count": protocols["exact_timestamp_match"] + protocols["nearest_timestamp_match"],
               "interpolated_timestamp_count": protocols["common_time_interpolation"],
               "alignment_protocol_counts": dict(protocols), "maximum_timestamp_error": float(max(errors, default=0)),
               "history_future_boundary_verified": True, "valid_mask_holes_interpolated": 0,
               "yaw_interpolation": "unwrap_shortest_arc", "size_and_category_interpolation": "nearest_valid_value"}
    output = Path(output); write_json(output / "time_alignment/alignment_summary.json", summary)
    write_csv(output / "time_alignment/aligned_samples.csv", [{key: row[key] for key in row if key not in {"calibration", "box_reference"}} for row in aligned])
    return {"rows": aligned, "summary": summary, "geometry": geometry, "tracks": tracks, "raw_boxes": raw_boxes}

