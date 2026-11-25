from __future__ import annotations

from typing import Any

import numpy as np

from driveaudit_eval.agents.core import angle_delta

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE
from driveaudit_eval.research.consistency.perception_trajectory.speed_estimation import compare_speed_protocols


def _kinematics(rows: list[dict[str, Any]], smooth: bool = False) -> dict[str, np.ndarray]:
    rows = sorted(rows, key=lambda row: row["timestamp"])
    if len(rows) < 2:
        return {key: np.array([]) for key in ("speed", "acceleration", "jerk", "yaw_rate", "curvature")}
    t = np.asarray([row["timestamp"] for row in rows], dtype=float)
    xy = np.asarray([[row["box_bev"]["x"], row["box_bev"]["y"]] for row in rows], dtype=float)
    yaw = np.unwrap(np.asarray([row["box_bev"]["yaw"] for row in rows], dtype=float))
    if smooth and len(xy) >= 3:
        xy[1:-1] = (xy[:-2] + xy[1:-1] + xy[2:]) / 3.0
    dt = np.diff(t); valid = dt > 0
    velocity = np.diff(xy, axis=0)[valid] / dt[valid, None]
    speed = np.linalg.norm(velocity, axis=1)
    yaw_rate = np.diff(yaw)[valid] / dt[valid]
    acceleration = np.diff(speed) / dt[valid][1:] if len(speed) > 1 else np.array([])
    jerk = np.diff(acceleration) / dt[valid][2:] if len(acceleration) > 1 else np.array([])
    curvature = yaw_rate / np.maximum(speed, 1e-3)
    return {"speed": speed, "velocity": velocity, "acceleration": acceleration, "jerk": jerk,
            "yaw_rate": yaw_rate, "curvature": curvature}


def _aligned_mae(a: np.ndarray, b: np.ndarray) -> float | str:
    n = min(len(a), len(b))
    return float(np.mean(np.abs(a[:n] - b[:n]))) if n else NOT_AVAILABLE


def compute_motion_metrics(submission_tracks: list[dict[str, Any]], image_tracks: list[dict[str, Any]],
                           mapping: list[dict[str, Any]], speed_config: dict[str, Any] | None = None) -> dict[str, Any]:
    submitted = {track["agent_id"]: track for track in submission_tracks}; extracted = {track["image_track_id"]: track for track in image_tracks}
    raw, smoothed = [], []
    direction, stationary, turning, longitudinal_velocity_errors, lateral_velocity_errors = [], [], [], [], []
    lane_change_timing, stop_timing = [], []
    for match in mapping:
        sid, iid = match["submission_agent_id"], match["image_track_id"]
        if not sid or not iid: continue
        s_by = {row["frame_index"]: row for row in submitted[sid]["observations"] if row.get("expected_visible", True)}
        i_by = {row["frame_index"]: row for row in extracted[iid]["observations"] if row.get("source") == "image_detection"}
        common = sorted(set(s_by) & set(i_by)); s = [s_by[f] for f in common]; i = [i_by[f] for f in common]
        for smooth, bucket in ((False, raw), (True, smoothed)):
            sk, ik = _kinematics(s, smooth), _kinematics(i, smooth)
            bucket.append({"speed_mae": _aligned_mae(sk["speed"], ik["speed"]),
                           "speed_rmse": float(np.sqrt(np.mean((sk["speed"][:min(len(sk['speed']), len(ik['speed']))] - ik["speed"][:min(len(sk['speed']), len(ik['speed']))]) ** 2))) if min(len(sk["speed"]), len(ik["speed"])) else NOT_AVAILABLE,
                           "acceleration_mae": _aligned_mae(sk["acceleration"], ik["acceleration"]),
                           "jerk_difference": _aligned_mae(sk["jerk"], ik["jerk"]),
                           "yaw_rate_mae": _aligned_mae(sk["yaw_rate"], ik["yaw_rate"]),
                           "curvature_difference": _aligned_mae(sk["curvature"], ik["curvature"])})
        submitted_stop, image_stop, submitted_lane_change, image_lane_change = [], [], [], []
        for sr, ir in zip(s, i):
            sv, iv = np.asarray(sr.get("velocity", [0, 0])), np.asarray(ir.get("velocity", [0, 0]))
            yaw = float(sr["box_bev"]["yaw"]); forward = np.array([np.cos(yaw), np.sin(yaw)]); side = np.array([-np.sin(yaw), np.cos(yaw)])
            longitudinal_velocity_errors.append(abs(float((iv - sv) @ forward)))
            lateral_velocity_errors.append(abs(float((iv - sv) @ side)))
            direction.append(float(sv @ iv >= 0)); stationary.append(float((np.linalg.norm(sv) < .3) == (np.linalg.norm(iv) < .3)))
            turning.append(float((abs(sr["box_bev"]["yaw"]) > .2) == (abs(ir["box_bev"]["yaw"]) > .2)))
            if np.linalg.norm(sv) < .3: submitted_stop.append(sr["frame_index"])
            if np.linalg.norm(iv) < .3: image_stop.append(ir["frame_index"])
            if abs(float(sv @ side)) > .5: submitted_lane_change.append(sr["frame_index"])
            if abs(float(iv @ side)) > .5: image_lane_change.append(ir["frame_index"])
        if submitted_stop and image_stop: stop_timing.append(abs(submitted_stop[0] - image_stop[0]))
        if submitted_lane_change and image_lane_change: lane_change_timing.append(abs(submitted_lane_change[0] - image_lane_change[0]))
    def aggregate(key: str, values: list[dict[str, Any]]) -> float | str:
        available = [row[key] for row in values if isinstance(row[key], (float, int))]
        return float(np.mean(available)) if available else NOT_AVAILABLE
    result = {key: aggregate(key, raw) for key in ("speed_mae", "speed_rmse", "acceleration_mae", "jerk_difference", "yaw_rate_mae", "curvature_difference")}
    result["smoothed"] = {key: aggregate(key, smoothed) for key in ("speed_mae", "speed_rmse", "acceleration_mae", "jerk_difference", "yaw_rate_mae", "curvature_difference")}
    result.update({"motion_direction_agreement": float(np.mean(direction)) if direction else NOT_AVAILABLE,
                   "longitudinal_velocity_mae": float(np.mean(longitudinal_velocity_errors)) if longitudinal_velocity_errors else NOT_AVAILABLE,
                   "lateral_velocity_mae": float(np.mean(lateral_velocity_errors)) if lateral_velocity_errors else NOT_AVAILABLE,
                   "stationary_moving_state_accuracy": float(np.mean(stationary)) if stationary else NOT_AVAILABLE,
                   "turning_state_accuracy": float(np.mean(turning)) if turning else NOT_AVAILABLE,
                   "lane_change_timing_error_frames": float(np.mean(lane_change_timing)) if lane_change_timing else NOT_AVAILABLE,
                   "stop_timing_error_frames": float(np.mean(stop_timing)) if stop_timing else NOT_AVAILABLE})
    if speed_config:
        protocols = compare_speed_protocols(submission_tracks, image_tracks, mapping, speed_config)
        selected = speed_config["formal_source"]
        result["legacy_position_difference"] = {"speed_mae": result["speed_mae"], "speed_rmse": result["speed_rmse"]}
        result["speed_protocols"] = protocols; result["formal_speed_source"] = selected
        result["speed_mae"] = protocols[selected]["speed_mae"]
        result["speed_rmse"] = protocols[selected]["speed_rmse"]
        result["speed_p90"] = protocols[selected]["speed_p90"]
        result["speed_p99"] = protocols[selected]["speed_p99"]
        result["valid_speed_samples"] = protocols[selected]["valid_speed_samples"]
    return result
