from __future__ import annotations

import math
from typing import Any

import numpy as np

from driveaudit_eval.agents.core import angle_delta

from driveaudit_eval.research.consistency.perception_trajectory import NOT_AVAILABLE
from driveaudit_eval.consistency.shared.common import percentile_summary, rotated_iou


def discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return math.nan
    cache = np.full((len(a), len(b)), -1.0)
    def solve(i: int, j: int) -> float:
        if cache[i, j] >= 0:
            return float(cache[i, j])
        distance = float(np.linalg.norm(a[i] - b[j]))
        if i == j == 0: value = distance
        elif i == 0: value = max(solve(0, j - 1), distance)
        elif j == 0: value = max(solve(i - 1, 0), distance)
        else: value = max(min(solve(i - 1, j), solve(i - 1, j - 1), solve(i, j - 1)), distance)
        cache[i, j] = value
        return value
    return solve(len(a) - 1, len(b) - 1)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return math.nan
    cost = np.full((len(a) + 1, len(b) + 1), np.inf); cost[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost[i, j] = np.linalg.norm(a[i - 1] - b[j - 1]) + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[-1, -1] / max(len(a), len(b)))


def compute_bev_geometry(submission_tracks: list[dict[str, Any]], image_tracks: list[dict[str, Any]],
                         mapping: list[dict[str, Any]], history_baseline_center_ade: float | str = NOT_AVAILABLE) -> dict[str, Any]:
    submitted = {track["agent_id"]: track for track in submission_tracks}
    extracted = {track["image_track_id"]: track for track in image_tracks}
    center_errors, ious, heading_errors, length_errors, width_errors, longitudinal, lateral = [], [], [], [], [], [], []
    endpoint_longitudinal, endpoint_lateral = [], []
    fdes, frechet, dtw = [], [], []
    per_track = []
    for match in mapping:
        sid, iid = match["submission_agent_id"], match["image_track_id"]
        if not sid or not iid:
            continue
        s = {row["frame_index"]: row for row in submitted[sid]["observations"] if row.get("expected_visible", True)}
        i = {row["frame_index"]: row for row in extracted[iid]["observations"] if row.get("source") == "image_detection"}
        common = sorted(set(s) & set(i))
        if not common:
            continue
        local_errors, local_longitudinal, local_lateral, sa, ia = [], [], [], [], []
        for frame in common:
            sb, ib = s[frame]["box_bev"], i[frame]["box_bev"]
            delta = np.asarray([ib["x"] - sb["x"], ib["y"] - sb["y"]])
            error = float(np.linalg.norm(delta)); local_errors.append(error); center_errors.append(error)
            c, sn = math.cos(sb["yaw"]), math.sin(sb["yaw"])
            lon, lat = float(delta @ np.array([c, sn])), float(delta @ np.array([-sn, c]))
            longitudinal.append(lon); lateral.append(lat); local_longitudinal.append(lon); local_lateral.append(lat)
            ious.append(rotated_iou(sb, ib)); heading_errors.append(abs(angle_delta(ib["yaw"], sb["yaw"])))
            length_errors.append(abs(ib["length"] - sb["length"])); width_errors.append(abs(ib["width"] - sb["width"]))
            sa.append([sb["x"], sb["y"]]); ia.append([ib["x"], ib["y"]])
        fdes.append(local_errors[-1]); frechet.append(discrete_frechet(np.asarray(sa), np.asarray(ia))); dtw.append(dtw_distance(np.asarray(sa), np.asarray(ia)))
        endpoint_longitudinal.append(local_longitudinal[-1]); endpoint_lateral.append(local_lateral[-1])
        per_track.append({"agent_id": sid, "image_track_id": iid, "common_visible_frames": len(common),
                          "bev_center_ade": float(np.mean(local_errors)), "visible_fde": local_errors[-1]})
    ade = float(np.mean(center_errors)) if center_errors else NOT_AVAILABLE
    excess = max(0.0, ade - float(history_baseline_center_ade)) if isinstance(ade, float) and isinstance(history_baseline_center_ade, (float, int)) else NOT_AVAILABLE
    ratio = ade / (float(history_baseline_center_ade) + 1e-9) if isinstance(ade, float) and isinstance(history_baseline_center_ade, (float, int)) else NOT_AVAILABLE
    return {"bev_center_ade": ade, "visible_fde": float(np.mean(fdes)) if fdes else NOT_AVAILABLE,
            "center_error": percentile_summary(center_errors), "rotated_bev_iou": float(np.mean(ious)) if ious else NOT_AVAILABLE,
            "longitudinal_position_mae": float(np.mean(np.abs(longitudinal))) if longitudinal else NOT_AVAILABLE,
            "lateral_position_mae": float(np.mean(np.abs(lateral))) if lateral else NOT_AVAILABLE,
            "box_length_mae": float(np.mean(length_errors)) if length_errors else NOT_AVAILABLE,
            "box_width_mae": float(np.mean(width_errors)) if width_errors else NOT_AVAILABLE,
            "heading_mae": float(np.mean(heading_errors)) if heading_errors else NOT_AVAILABLE,
            "heading_p90": float(np.percentile(heading_errors, 90)) if heading_errors else NOT_AVAILABLE,
            "endpoint_longitudinal_error": float(np.mean(endpoint_longitudinal)) if endpoint_longitudinal else NOT_AVAILABLE,
            "endpoint_lateral_error": float(np.mean(endpoint_lateral)) if endpoint_lateral else NOT_AVAILABLE,
            "frechet_curve_distance": float(np.mean(frechet)) if frechet else NOT_AVAILABLE,
            "dtw_distance": float(np.mean(dtw)) if dtw else NOT_AVAILABLE,
            "history_baseline_center_ade": history_baseline_center_ade, "future_center_ade": ade,
            "excess_center_ade": excess, "future_to_history_error_ratio": ratio, "per_track": per_track}
